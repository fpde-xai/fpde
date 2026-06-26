"""Numpy-only prototype baseline for conditional raw-sequence generation."""

from __future__ import annotations

from typing import Any, Dict, Optional, Sequence

import numpy as np

from .._array import _as_label_array
from .legacy import _labels_unique
from .preprocessing import validate_sequence_inputs


def _interpolate_sequence(seq: np.ndarray, length: int) -> np.ndarray:
    """Linearly interpolate a ``(T, C)`` sequence to ``(length, C)``."""
    values = np.asarray(seq, dtype=float)
    if values.ndim != 2 or values.shape[0] == 0 or values.shape[1] == 0:
        raise ValueError(f"seq must be a non-empty 2D array, got shape={values.shape}")
    if not np.all(np.isfinite(values)):
        raise ValueError("seq contains NaN or inf")
    if length <= 0:
        raise ValueError("length must be positive")
    if values.shape[0] == length:
        return values.copy()
    if values.shape[0] == 1:
        return np.repeat(values, length, axis=0)
    source = np.linspace(0.0, 1.0, values.shape[0])
    target = np.linspace(0.0, 1.0, int(length))
    out = np.empty((int(length), values.shape[1]), dtype=float)
    for channel in range(values.shape[1]):
        out[:, channel] = np.interp(target, source, values[:, channel])
    return out


# Keep the Phase 1-3 private helper name available for local callers.
_interp_matrix = _interpolate_sequence


def _feature_summary(features: np.ndarray) -> np.ndarray:
    """Return mean, std, min, and max over valid feature time steps."""
    values = np.asarray(features, dtype=float)
    if values.ndim != 2 or values.shape[0] == 0 or values.shape[1] == 0:
        raise ValueError(
            "condition_features must be a non-empty 2D array when passed as a sequence, "
            f"got shape={values.shape}"
        )
    if not np.all(np.isfinite(values)):
        raise ValueError("condition_features contains NaN or inf")
    scale = np.maximum(np.max(np.abs(values), axis=0), 1.0)
    normalized = values / scale[None, :]
    summary = np.concatenate(
        [
            np.mean(normalized, axis=0) * scale,
            np.std(normalized, axis=0) * scale,
            np.min(values, axis=0),
            np.max(values, axis=0),
        ]
    )
    if not np.all(np.isfinite(summary)):
        raise ValueError("condition_features summary contains NaN or inf")
    return summary


def _finite_euclidean_distances(candidates: np.ndarray, query: np.ndarray) -> np.ndarray:
    """Compute row-wise Euclidean distances without overflowing finite inputs."""
    scale = np.maximum(
        np.maximum(np.max(np.abs(candidates), axis=1), np.max(np.abs(query))),
        1.0,
    )
    normalized = candidates / scale[:, None] - query[None, :] / scale[:, None]
    normalized_distance = np.sqrt(np.sum(normalized * normalized, axis=1))
    max_float = np.finfo(float).max
    return np.minimum(normalized_distance, max_float / scale) * scale


class PrototypeRawGenerator:
    """Lightweight label and feature-conditioned raw generator baseline.

    Label prototypes provide the base sequence. When ``condition_features`` is
    supplied, the generator adds the raw residual from the same-label training
    sample with the nearest feature summary. This is intentionally a simple
    numpy-only baseline, not a learned VAE, diffusion, or seq2seq model.
    """

    def fit(
        self,
        *,
        raw: np.ndarray | Sequence[np.ndarray | Sequence[Sequence[float]]],
        y: Sequence[Any],
        features: Optional[np.ndarray | Sequence[np.ndarray | Sequence[Sequence[float]]]] = None,
        mask: Optional[np.ndarray | Sequence[Sequence[bool]]] = None,
    ) -> "PrototypeRawGenerator":
        """Fit label prototypes, residual scales, and conditioning records."""
        batch = validate_sequence_inputs(raw, features=features, mask=mask)
        labels = _as_label_array(y)
        if labels.shape[0] != batch.raw.shape[0]:
            raise ValueError(
                f"number of raw samples and labels differ: {batch.raw.shape[0]} vs {labels.shape[0]}"
            )
        valid_lengths = np.sum(batch.mask, axis=1).astype(int)
        if np.any(valid_lengths <= 0):
            index = int(np.flatnonzero(valid_lengths <= 0)[0])
            raise ValueError(f"mask leaves training sample {index} with no valid time steps")

        classes, inverse = _labels_unique(labels)
        prototypes: Dict[Any, np.ndarray] = {}
        prototype_masks: Dict[Any, np.ndarray] = {}
        residual_scales: Dict[Any, np.ndarray] = {}
        lengths: Dict[Any, int] = {}
        feature_prototypes: Optional[Dict[Any, np.ndarray]] = {} if batch.features is not None else None
        feature_summary_prototypes: Optional[Dict[Any, np.ndarray]] = (
            {} if batch.features is not None else None
        )

        training_raw = [sample[sample_mask].copy() for sample, sample_mask in zip(batch.raw, batch.mask)]
        training_features = (
            [sample[sample_mask].copy() for sample, sample_mask in zip(batch.features, batch.mask)]
            if batch.features is not None
            else None
        )
        training_summaries = (
            np.stack([_feature_summary(sample) for sample in training_features], axis=0)
            if training_features is not None
            else None
        )

        for class_idx, label in enumerate(classes.tolist()):
            selected = inverse == class_idx
            weights = batch.mask[selected].astype(float)
            denom = np.sum(weights, axis=0)
            valid = denom > 0.0

            raw_numer = np.sum(batch.raw[selected] * weights[:, :, None], axis=0)
            raw_proto = np.zeros((batch.raw.shape[1], batch.raw.shape[2]), dtype=float)
            raw_proto[valid] = raw_numer[valid] / denom[valid, None]
            prototypes[label] = raw_proto
            prototype_masks[label] = valid
            lengths[label] = int(np.max(valid_lengths[selected]))

            if batch.features is not None:
                assert feature_prototypes is not None
                assert feature_summary_prototypes is not None
                feature_numer = np.sum(batch.features[selected] * weights[:, :, None], axis=0)
                feature_proto = np.zeros((batch.features.shape[1], batch.features.shape[2]), dtype=float)
                feature_proto[valid] = feature_numer[valid] / denom[valid, None]
                feature_prototypes[label] = feature_proto
                assert training_summaries is not None
                feature_summary_prototypes[label] = np.mean(training_summaries[selected], axis=0)

            prototype_sequence = raw_proto[valid]
            class_residuals = []
            for sample_index in np.flatnonzero(selected):
                sample_raw = training_raw[int(sample_index)]
                sample_base = _interpolate_sequence(prototype_sequence, sample_raw.shape[0])
                class_residuals.append(sample_raw - sample_base)
            residual_scales[label] = np.std(np.concatenate(class_residuals, axis=0), axis=0)

        raw_residuals = []
        for sample_index, label in enumerate(labels.tolist()):
            sample_raw = training_raw[sample_index]
            base = _interpolate_sequence(prototypes[label][prototype_masks[label]], sample_raw.shape[0])
            raw_residuals.append(sample_raw - base)

        self.prototype_labels_ = classes.copy()
        self.raw_prototypes_ = prototypes
        self.raw_prototype_masks_ = prototype_masks
        self.residual_scales_ = residual_scales
        self.default_lengths_ = lengths
        self.feature_prototypes_ = feature_prototypes
        self.feature_summary_prototypes_ = feature_summary_prototypes
        self.training_labels_ = labels.copy()
        self.training_raw_ = training_raw
        self.training_features_ = training_features
        self.training_feature_summaries_ = training_summaries
        self.training_raw_residuals_ = raw_residuals
        self.training_lengths_ = valid_lengths.copy()
        self.training_masks_ = batch.mask.copy()
        self.raw_channels_ = int(batch.raw.shape[2])
        self.feature_channels_ = 0 if batch.features is None else int(batch.features.shape[2])
        self.feature_summary_dimension_ = 0 if batch.features is None else 4 * self.feature_channels_
        return self

    def _condition_summary(
        self,
        condition_features: np.ndarray | Sequence[float] | Sequence[Sequence[float]],
    ) -> np.ndarray:
        if self.training_features_ is None:
            raise ValueError("condition_features cannot be used because fit was called without features")
        condition = np.asarray(condition_features, dtype=float)
        if not np.all(np.isfinite(condition)):
            raise ValueError("condition_features contains NaN or inf")
        if condition.ndim == 1:
            if condition.shape[0] != self.feature_summary_dimension_:
                raise ValueError(
                    "condition_features summary dimension mismatch: "
                    f"expected {self.feature_summary_dimension_}, got {condition.shape[0]}"
                )
            return condition.copy()
        if condition.ndim == 2:
            if condition.shape[1] != self.feature_channels_:
                raise ValueError(
                    "condition_features channel dimension mismatch: "
                    f"expected {self.feature_channels_}, got {condition.shape[1]}"
                )
            return _feature_summary(condition)
        raise ValueError(
            "condition_features must have shape (C_summary,) or (T_cond, C_feat), "
            f"got shape={condition.shape}"
        )

    def generate(
        self,
        *,
        label: Any,
        length: Optional[int] = None,
        condition_features: Optional[
            np.ndarray | Sequence[float] | Sequence[Sequence[float]]
        ] = None,
        noise_scale: float = 0.0,
        random_state: Optional[int] = None,
    ) -> np.ndarray:
        """Generate one ``(length, C_raw)`` raw sequence.

        A one-dimensional condition must already be a feature summary with
        ``4 * C_feat`` values. A two-dimensional ``(T_cond, C_feat)`` condition
        is summarized over all of its time steps.
        """
        return self.generate_with_metadata(
            label=label,
            length=length,
            condition_features=condition_features,
            noise_scale=noise_scale,
            random_state=random_state,
        )["raw"]

    def generate_with_metadata(
        self,
        *,
        label: Any,
        length: Optional[int] = None,
        condition_features: Optional[
            np.ndarray | Sequence[float] | Sequence[Sequence[float]]
        ] = None,
        noise_scale: float = 0.0,
        random_state: Optional[int] = None,
    ) -> Dict[str, Any]:
        """Generate a raw sequence and return conditioning audit metadata."""
        if not hasattr(self, "raw_prototypes_"):
            raise RuntimeError("PrototypeRawGenerator must be fit before generate")
        if label not in self.raw_prototypes_:
            raise ValueError(f"no raw prototype found for label={label!r}")

        out_length = self.default_lengths_[label] if length is None else int(length)
        if out_length <= 0:
            raise ValueError("length must be positive")
        scale = float(noise_scale)
        if not np.isfinite(scale) or scale < 0.0:
            raise ValueError("noise_scale must be non-negative")

        prototype = self.raw_prototypes_[label][self.raw_prototype_masks_[label]]
        generated = _interpolate_sequence(prototype, out_length)
        neighbor_index: Optional[int] = None
        neighbor_distance: Optional[float] = None

        if condition_features is not None:
            summary = self._condition_summary(condition_features)
            assert self.training_feature_summaries_ is not None
            candidate_indices = np.flatnonzero(self.training_labels_ == label)
            candidate_summaries = self.training_feature_summaries_[candidate_indices]
            distances = _finite_euclidean_distances(candidate_summaries, summary)
            local_index = int(np.argmin(distances))
            neighbor_index = int(candidate_indices[local_index])
            neighbor_distance = float(distances[local_index])
            residual = _interpolate_sequence(self.training_raw_residuals_[neighbor_index], out_length)
            generated = generated + residual

        if scale > 0.0:
            rng = np.random.default_rng(random_state)
            generated = generated + rng.normal(
                0.0,
                self.residual_scales_[label] * scale,
                size=generated.shape,
            )
        if not np.all(np.isfinite(generated)):
            raise RuntimeError("generated raw sequence contains NaN or inf")

        return {
            "raw": generated.astype(float, copy=False),
            "label": label,
            "length": out_length,
            "conditioned": condition_features is not None,
            "selected_neighbor_index": neighbor_index,
            "selected_neighbor_distance": neighbor_distance,
            "noise_scale": scale,
        }


__all__ = ["PrototypeRawGenerator"]
