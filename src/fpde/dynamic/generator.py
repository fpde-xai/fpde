"""Baseline raw-sequence generator for future conditional generation hooks."""

from __future__ import annotations

from typing import Any, Dict, Optional, Sequence

import numpy as np

from .._array import _as_label_array
from .legacy import _labels_unique
from .preprocessing import validate_sequence_inputs


def _interp_matrix(values: np.ndarray, length: int) -> np.ndarray:
    if length <= 0:
        raise ValueError("length must be positive")
    if values.shape[0] == length:
        return values.astype(float, copy=True)
    if values.shape[0] == 1:
        return np.repeat(values, length, axis=0).astype(float, copy=False)
    source = np.linspace(0.0, 1.0, values.shape[0])
    target = np.linspace(0.0, 1.0, int(length))
    out = np.empty((int(length), values.shape[1]), dtype=float)
    for channel in range(values.shape[1]):
        out[:, channel] = np.interp(target, source, values[:, channel])
    return out


class PrototypeRawGenerator:
    """Lightweight label-conditioned raw generator baseline.

    This stores label-wise raw prototypes and interpolates them to the requested
    length. It is intentionally a simple interface placeholder for future
    conditional VAE, diffusion, or seq2seq raw generators.
    """

    def fit(
        self,
        *,
        raw: np.ndarray | Sequence[np.ndarray | Sequence[Sequence[float]]],
        y: Sequence[Any],
        features: Optional[np.ndarray | Sequence[np.ndarray | Sequence[Sequence[float]]]] = None,
        mask: Optional[np.ndarray | Sequence[Sequence[bool]]] = None,
    ) -> "PrototypeRawGenerator":
        """Fit label-wise raw prototypes and residual scales."""
        batch = validate_sequence_inputs(raw, features=features, mask=mask)
        labels = _as_label_array(y)
        if labels.shape[0] != batch.raw.shape[0]:
            raise ValueError(f"number of raw samples and labels differ: {batch.raw.shape[0]} vs {labels.shape[0]}")
        classes, inverse = _labels_unique(labels)
        prototypes: Dict[Any, np.ndarray] = {}
        residual_scales: Dict[Any, np.ndarray] = {}
        lengths: Dict[Any, int] = {}
        for class_idx, label in enumerate(classes.tolist()):
            selected = inverse == class_idx
            weights = batch.mask[selected].astype(float)
            denom = np.sum(weights, axis=0)
            numer = np.sum(batch.raw[selected] * weights[:, :, None], axis=0)
            proto = np.zeros((batch.raw.shape[1], batch.raw.shape[2]), dtype=float)
            valid = denom > 0.0
            proto[valid] = numer[valid] / denom[valid, None]
            prototypes[label] = proto
            lengths[label] = int(np.max(np.sum(batch.mask[selected], axis=1)))

            residuals = []
            for sample, sample_mask in zip(batch.raw[selected], batch.mask[selected]):
                residuals.append(sample[sample_mask] - proto[sample_mask])
            if residuals:
                stacked = np.concatenate(residuals, axis=0)
                residual_scales[label] = np.std(stacked, axis=0)
            else:
                residual_scales[label] = np.zeros(batch.raw.shape[2], dtype=float)

        self.prototype_labels_ = classes.copy()
        self.raw_prototypes_ = prototypes
        self.residual_scales_ = residual_scales
        self.default_lengths_ = lengths
        return self

    def generate(
        self,
        *,
        label: Any,
        length: Optional[int] = None,
        condition_features: Optional[np.ndarray | Sequence[Sequence[float]]] = None,
        noise_scale: float = 0.0,
        random_state: Optional[int] = None,
    ) -> np.ndarray:
        """Generate one raw sequence from a label prototype.

        ``condition_features`` is accepted for API stability, but v1 does not
        condition on it yet.
        """
        if not hasattr(self, "raw_prototypes_"):
            raise RuntimeError("PrototypeRawGenerator must be fit before generate")
        if label not in self.raw_prototypes_:
            raise ValueError(f"no raw prototype found for label={label!r}")
        if condition_features is not None:
            condition_arr = np.asarray(condition_features, dtype=float)
            if not np.all(np.isfinite(condition_arr)):
                raise ValueError("condition_features contains NaN or inf")
        out_length = self.default_lengths_[label] if length is None else int(length)
        generated = _interp_matrix(self.raw_prototypes_[label], out_length)
        scale = float(noise_scale)
        if not np.isfinite(scale) or scale < 0.0:
            raise ValueError("noise_scale must be non-negative")
        if scale > 0.0:
            rng = np.random.default_rng(random_state)
            generated = generated + rng.normal(0.0, self.residual_scales_[label] * scale, size=generated.shape)
        return generated.astype(float, copy=False)


__all__ = ["PrototypeRawGenerator"]
