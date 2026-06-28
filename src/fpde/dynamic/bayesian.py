"""Bootstrap posterior diagnostics for native-time RawFeat Dynamic-FPDE.

This module decomposes target-versus-rival prototype evidence.  It does not
estimate causal effects or ground-truth feature importance.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any, Dict, Literal, Sequence

import numpy as np

from .._array import _as_label_array
from .legacy import (
    _labels_unique,
    _validate_eps,
    _validate_lambda_hyb,
    native_dynamic_cos_fpde,
    native_dynamic_diff_fpde,
    native_dynamic_hyb_fpde,
)


@dataclass(frozen=True)
class RawFeatSequence:
    """One native-time RawFeat sample.

    ``raw`` and ``features`` are two-dimensional, while ``dt`` and ``mask``
    are one-dimensional arrays sharing the same time length.
    """

    raw: np.ndarray
    features: np.ndarray
    dt: np.ndarray
    mask: np.ndarray


RawFeatScalingMode = Literal["none", "group_l1", "group_l2", "standard"]
RawFeatFrameWeighting = Literal["frame_equal", "sample_equal"]


@dataclass(frozen=True)
class RawFeatScaling:
    """Training-fitted affine scaling for concatenated RawFeat coordinates."""

    mode: RawFeatScalingMode
    offset: np.ndarray
    scale: np.ndarray


@dataclass(frozen=True)
class BayesianRawFeatPrototypePosterior:
    """Bootstrap draws of one feature-vector prototype per class."""

    prototypes: np.ndarray
    prototype_labels: np.ndarray
    feature_slices: Dict[str, slice]
    prototype_stat: str
    class_counts: Dict[Any, int]
    scaling: RawFeatScaling
    frame_weighting: RawFeatFrameWeighting

    @property
    def n_samples(self) -> int:
        return int(self.prototypes.shape[0])


@dataclass(frozen=True)
class PosteriorScalarSummary:
    """Posterior diagnostics for a scalar quantity."""

    mean: float
    credible_interval: tuple[float, float]
    probability_positive: float
    sign_stability: float
    posterior: np.ndarray


@dataclass(frozen=True)
class BayesianRawFeatMethodSummary:
    """Posterior attribution diagnostics for one Dynamic-FPDE method."""

    posterior_mean: np.ndarray
    credible_interval_lower: np.ndarray
    credible_interval_upper: np.ndarray
    probability_positive: np.ndarray
    sign_stability: np.ndarray
    attribution_posterior: np.ndarray
    evidence: PosteriorScalarSummary
    exactness_residuals: np.ndarray
    raw_group_attribution: PosteriorScalarSummary
    feature_group_attribution: PosteriorScalarSummary
    dt_group_attribution: PosteriorScalarSummary

    @property
    def credible_interval(self) -> tuple[np.ndarray, np.ndarray]:
        """Return the lower and upper coordinate-wise credible bounds."""
        return self.credible_interval_lower, self.credible_interval_upper

    @property
    def evidence_posterior(self) -> np.ndarray:
        """Return the per-draw evidence values."""
        return self.evidence.posterior


@dataclass(frozen=True)
class BayesianRawFeatDynamicFPDEResult:
    """Dynamic-Diff, Dynamic-Cos, and Dynamic-Hyb posterior summaries."""

    target_label: Any
    rival_label: Any
    mask: np.ndarray
    feature_slices: Dict[str, slice]
    diff: BayesianRawFeatMethodSummary
    cos: BayesianRawFeatMethodSummary
    hyb: BayesianRawFeatMethodSummary
    lambda_hyb_posterior: np.ndarray

    @property
    def dynamic_diff(self) -> BayesianRawFeatMethodSummary:
        return self.diff

    @property
    def dynamic_cos(self) -> BayesianRawFeatMethodSummary:
        return self.cos

    @property
    def dynamic_hyb(self) -> BayesianRawFeatMethodSummary:
        return self.hyb


def build_rawfeat_matrix(
    sequence: RawFeatSequence,
    scaling: RawFeatScaling | None = None,
) -> np.ndarray:
    """Validate and concatenate one RawFeat sample without resampling time.

    Non-finite values are rejected on valid frames.  Invalid frames are
    represented by zeros so they cannot leak into later calculations.
    """
    if not isinstance(sequence, RawFeatSequence):
        raise TypeError("sequence must be a RawFeatSequence")
    raw = np.asarray(sequence.raw, dtype=float)
    features = np.asarray(sequence.features, dtype=float)
    dt = np.asarray(sequence.dt, dtype=float)
    mask = np.asarray(sequence.mask, dtype=bool)
    if raw.ndim != 2:
        raise ValueError(f"raw must be 2D, got shape={raw.shape}")
    if features.ndim != 2:
        raise ValueError(f"features must be 2D, got shape={features.shape}")
    if dt.ndim != 1:
        raise ValueError(f"dt must be 1D, got shape={dt.shape}")
    if mask.ndim != 1:
        raise ValueError(f"mask must be 1D, got shape={mask.shape}")
    if raw.shape[0] == 0:
        raise ValueError("RawFeat sequences must contain at least one frame")
    if raw.shape[1] == 0 or features.shape[1] == 0:
        raise ValueError("raw and features must contain at least one channel")
    lengths = (raw.shape[0], features.shape[0], dt.shape[0], mask.shape[0])
    if len(set(lengths)) != 1:
        raise ValueError(f"all RawFeat time dimensions must match, got {lengths}")
    valid = mask
    if not np.all(np.isfinite(raw[valid])):
        raise ValueError("raw contains NaN or inf on valid frames")
    if not np.all(np.isfinite(features[valid])):
        raise ValueError("features contains NaN or inf on valid frames")
    if not np.all(np.isfinite(dt[valid])):
        raise ValueError("dt contains NaN or inf on valid frames")
    matrix = np.concatenate((raw, features, dt[:, None]), axis=1)
    matrix = matrix.astype(float, copy=True)
    matrix[~valid] = 0.0
    if scaling is not None:
        if not isinstance(scaling, RawFeatScaling):
            raise TypeError("scaling must be a RawFeatScaling or None")
        if scaling.offset.shape != (matrix.shape[1],) or scaling.scale.shape != (matrix.shape[1],):
            raise ValueError(
                f"scaling dimension mismatch: expected {(matrix.shape[1],)}, "
                f"got offset={scaling.offset.shape}, scale={scaling.scale.shape}"
            )
        if not np.all(np.isfinite(scaling.offset)) or not np.all(np.isfinite(scaling.scale)):
            raise ValueError("scaling parameters must be finite")
        if np.any(scaling.scale <= 0.0):
            raise ValueError("scaling scale values must be positive")
        matrix[valid] = (matrix[valid] - scaling.offset) / scaling.scale
        matrix[~valid] = 0.0
    return matrix


def _feature_slices(sequence: RawFeatSequence) -> Dict[str, slice]:
    c_raw = int(np.asarray(sequence.raw).shape[1])
    c_features = int(np.asarray(sequence.features).shape[1])
    return {
        "raw": slice(0, c_raw),
        "features": slice(c_raw, c_raw + c_features),
        "dt": slice(c_raw + c_features, c_raw + c_features + 1),
    }


def fit_rawfeat_scaling(
    sequences: Sequence[RawFeatSequence],
    scaling: RawFeatScalingMode = "none",
) -> RawFeatScaling:
    """Fit RawFeat scaling parameters from training valid frames only.

    ``group_l1`` and ``group_l2`` give each RawFeat block unit mean per-frame
    norm. ``standard`` performs coordinate-wise centering and standardization.
    Degenerate scales are replaced by one to keep constant inputs finite.
    """
    if scaling not in ("none", "group_l1", "group_l2", "standard"):
        raise ValueError("scaling must be 'none', 'group_l1', 'group_l2', or 'standard'")
    items = list(sequences)
    if not items:
        raise ValueError("sequences must contain at least one sample")
    matrices = [build_rawfeat_matrix(item) for item in items]
    masks = [np.asarray(item.mask, dtype=bool) for item in items]
    slices = _feature_slices(items[0])
    dimension = int(matrices[0].shape[1])
    valid_parts = []
    for i, (item, matrix, mask) in enumerate(zip(items, matrices, masks)):
        if matrix.shape[1] != dimension or _feature_slices(item) != slices:
            raise ValueError(f"RawFeat group dimensions differ at sequences[{i}]")
        if not np.any(mask):
            raise ValueError(f"sequences[{i}] has no valid frames")
        valid_parts.append(matrix[mask])
    valid_frames = np.concatenate(valid_parts, axis=0)
    offset = np.zeros(dimension, dtype=float)
    scale = np.ones(dimension, dtype=float)
    if scaling == "standard":
        offset = np.mean(valid_frames, axis=0)
        scale = np.std(valid_frames, axis=0)
        scale[scale <= np.finfo(float).eps] = 1.0
    elif scaling in ("group_l1", "group_l2"):
        for group_slice in slices.values():
            group = valid_frames[:, group_slice]
            if scaling == "group_l1":
                norms = np.sum(np.abs(group), axis=1)
            else:
                norms = np.linalg.norm(group, axis=1)
            group_scale = float(np.mean(norms))
            if not np.isfinite(group_scale) or group_scale <= np.finfo(float).eps:
                group_scale = 1.0
            scale[group_slice] = group_scale
    return RawFeatScaling(mode=scaling, offset=offset, scale=scale)


def fit_bayesian_rawfeat_prototypes(
    sequences: Sequence[RawFeatSequence],
    labels: Sequence[Any],
    n_samples: int,
    prototype_stat: Literal["median", "mean"] = "median",
    random_state: int | np.random.Generator | None = None,
    *,
    scaling: RawFeatScalingMode = "none",
    frame_weighting: RawFeatFrameWeighting = "frame_equal",
) -> BayesianRawFeatPrototypePosterior:
    """Fit a class-prototype posterior by bootstrapping training samples.

    Within each posterior draw, each class's training samples are sampled with
    replacement. ``frame_equal`` pools their valid frames, while
    ``sample_equal`` first summarizes each sampled sequence so every sampled
    training item has equal weight regardless of native duration.
    """
    items = list(sequences)
    if not items:
        raise ValueError("sequences must contain at least one sample")
    n_draws = int(n_samples)
    if n_draws <= 0:
        raise ValueError("n_samples must be positive")
    if prototype_stat not in ("median", "mean"):
        raise ValueError("prototype_stat must be 'median' or 'mean'")
    if frame_weighting not in ("frame_equal", "sample_equal"):
        raise ValueError("frame_weighting must be 'frame_equal' or 'sample_equal'")
    y = _as_label_array(labels)
    if y.shape[0] != len(items):
        raise ValueError(f"number of sequences and labels differ: {len(items)} vs {y.shape[0]}")

    scaling_parameters = fit_rawfeat_scaling(items, scaling=scaling)
    matrices = [build_rawfeat_matrix(item, scaling=scaling_parameters) for item in items]
    masks = [np.asarray(item.mask, dtype=bool) for item in items]
    slices = _feature_slices(items[0])
    dimension = int(matrices[0].shape[1])
    for i, (item, matrix, mask) in enumerate(zip(items, matrices, masks)):
        if matrix.shape[1] != dimension:
            raise ValueError(f"RawFeat dimension mismatch at sequences[{i}]: expected {dimension}, got {matrix.shape[1]}")
        if _feature_slices(item) != slices:
            raise ValueError(f"RawFeat group dimensions differ at sequences[{i}]")
        if not np.any(mask):
            raise ValueError(f"sequences[{i}] has no valid frames")

    classes, inverse = _labels_unique(y)
    if classes.shape[0] < 2:
        raise ValueError("at least two labels are required")
    rng = random_state if isinstance(random_state, np.random.Generator) else np.random.default_rng(random_state)
    prototypes = np.empty((n_draws, classes.shape[0], dimension), dtype=float)
    class_counts: Dict[Any, int] = {}
    reducer = np.median if prototype_stat == "median" else np.mean
    for class_idx, label in enumerate(classes.tolist()):
        indices = np.flatnonzero(inverse == class_idx)
        class_counts[label] = int(indices.size)
        for draw in range(n_draws):
            sampled = rng.choice(indices, size=indices.size, replace=True)
            sampled_frames = [matrices[i][masks[i]] for i in sampled.tolist()]
            if frame_weighting == "frame_equal":
                prototype_values = np.concatenate(sampled_frames, axis=0)
            else:
                prototype_values = np.stack([reducer(frames, axis=0) for frames in sampled_frames], axis=0)
            prototypes[draw, class_idx] = reducer(prototype_values, axis=0)

    return BayesianRawFeatPrototypePosterior(
        prototypes=prototypes,
        prototype_labels=classes.copy(),
        feature_slices=dict(slices),
        prototype_stat=prototype_stat,
        class_counts=class_counts,
        scaling=scaling_parameters,
        frame_weighting=frame_weighting,
    )


def _label_index(posterior: BayesianRawFeatPrototypePosterior, label: Any, name: str) -> int:
    matches = np.flatnonzero(posterior.prototype_labels == label)
    if matches.size == 0:
        raise ValueError(f"no posterior prototype found for {name}={label!r}")
    return int(matches[0])


def _sign_stability(values: np.ndarray, axis: int = 0) -> np.ndarray:
    return np.maximum(np.mean(values > 0.0, axis=axis), np.mean(values < 0.0, axis=axis))


def _scalar_summary(values: np.ndarray) -> PosteriorScalarSummary:
    arr = np.asarray(values, dtype=float)
    interval = np.quantile(arr, [0.025, 0.975])
    return PosteriorScalarSummary(
        mean=float(np.mean(arr)),
        credible_interval=(float(interval[0]), float(interval[1])),
        probability_positive=float(np.mean(arr > 0.0)),
        sign_stability=float(_sign_stability(arr)),
        posterior=arr.copy(),
    )


def _method_summary(
    attributions: np.ndarray,
    evidence: np.ndarray,
    feature_slices: Dict[str, slice],
) -> BayesianRawFeatMethodSummary:
    lower, upper = np.quantile(attributions, [0.025, 0.975], axis=0)
    group_draws = {
        name: np.sum(attributions[:, :, group_slice], axis=(1, 2))
        for name, group_slice in feature_slices.items()
    }
    return BayesianRawFeatMethodSummary(
        posterior_mean=np.mean(attributions, axis=0),
        credible_interval_lower=lower,
        credible_interval_upper=upper,
        probability_positive=np.mean(attributions > 0.0, axis=0),
        sign_stability=_sign_stability(attributions),
        attribution_posterior=attributions.copy(),
        evidence=_scalar_summary(evidence),
        exactness_residuals=evidence - np.sum(attributions, axis=(1, 2)),
        raw_group_attribution=_scalar_summary(group_draws["raw"]),
        feature_group_attribution=_scalar_summary(group_draws["features"]),
        dt_group_attribution=_scalar_summary(group_draws["dt"]),
    )


def bayesian_rawfeat_dynamic_fpde_explain_one(
    sequence: RawFeatSequence,
    posterior: BayesianRawFeatPrototypePosterior,
    *,
    target_label: Any,
    rival_label: Any,
    lambda_hyb: float | Sequence[float] | np.ndarray = 0.5,
    normalize: Literal["l1", "none"] = "l1",
    eps: float = 1e-12,
) -> BayesianRawFeatDynamicFPDEResult:
    """Explain one native-time sample over every prototype posterior draw.

    ``lambda_hyb`` may be one selected value or one value per posterior draw;
    the latter propagates lambda-selection uncertainty into Dynamic-Hyb.
    """
    if not isinstance(posterior, BayesianRawFeatPrototypePosterior):
        raise TypeError("posterior must be a BayesianRawFeatPrototypePosterior")
    eps_value = _validate_eps(eps)
    X = build_rawfeat_matrix(sequence)
    if X.shape[1] != posterior.prototypes.shape[2]:
        raise ValueError(
            f"RawFeat dimension mismatch: posterior has {posterior.prototypes.shape[2]}, input has {X.shape[1]}"
        )
    if _feature_slices(sequence) != posterior.feature_slices:
        raise ValueError("input RawFeat group dimensions differ from the fitted posterior")
    X = build_rawfeat_matrix(sequence, scaling=posterior.scaling)
    if target_label == rival_label:
        raise ValueError("target_label and rival_label must differ")
    target_idx = _label_index(posterior, target_label, "target_label")
    rival_idx = _label_index(posterior, rival_label, "rival_label")
    mask = np.asarray(sequence.mask, dtype=bool)
    valid_X = X[mask]

    lambdas = np.asarray(lambda_hyb, dtype=float)
    if lambdas.ndim == 0:
        lambdas = np.full(posterior.n_samples, _validate_lambda_hyb(float(lambdas)), dtype=float)
    elif lambdas.ndim == 1 and lambdas.shape[0] == posterior.n_samples:
        lambdas = np.array([_validate_lambda_hyb(value) for value in lambdas], dtype=float)
    else:
        raise ValueError(f"lambda_hyb must be scalar or have shape {(posterior.n_samples,)}, got {lambdas.shape}")

    shape = (posterior.n_samples, X.shape[0], X.shape[1])
    draws = {name: np.zeros(shape, dtype=float) for name in ("diff", "cos", "hyb")}
    evidences = {name: np.zeros(posterior.n_samples, dtype=float) for name in draws}
    if not np.any(mask):
        summaries = {
            name: _method_summary(draws[name], evidences[name], posterior.feature_slices)
            for name in draws
        }
        return BayesianRawFeatDynamicFPDEResult(
            target_label=target_label,
            rival_label=rival_label,
            mask=mask.copy(),
            feature_slices=dict(posterior.feature_slices),
            diff=summaries["diff"],
            cos=summaries["cos"],
            hyb=summaries["hyb"],
            lambda_hyb_posterior=lambdas,
        )
    for draw in range(posterior.n_samples):
        target = posterior.prototypes[draw, target_idx]
        rival = posterior.prototypes[draw, rival_idx]
        diff_attr, diff_evidence = native_dynamic_diff_fpde(valid_X, target, rival)
        cos_attr, cos_evidence = native_dynamic_cos_fpde(valid_X, target, rival, eps=eps_value)
        hyb_attr, hyb_evidence, _ = native_dynamic_hyb_fpde(
            valid_X,
            target,
            rival,
            lambda_hyb=float(lambdas[draw]),
            normalize=normalize,
            eps=eps_value,
        )
        for name, attr, evidence in (
            ("diff", diff_attr, diff_evidence),
            ("cos", cos_attr, cos_evidence),
            ("hyb", hyb_attr, hyb_evidence),
        ):
            draws[name][draw, mask] = attr
            evidences[name][draw] = evidence

    summaries = {
        name: _method_summary(draws[name], evidences[name], posterior.feature_slices)
        for name in draws
    }
    return BayesianRawFeatDynamicFPDEResult(
        target_label=target_label,
        rival_label=rival_label,
        mask=mask.copy(),
        feature_slices=dict(posterior.feature_slices),
        diff=summaries["diff"],
        cos=summaries["cos"],
        hyb=summaries["hyb"],
        lambda_hyb_posterior=lambdas,
    )


__all__ = [
    "RawFeatSequence",
    "RawFeatScalingMode",
    "RawFeatFrameWeighting",
    "RawFeatScaling",
    "BayesianRawFeatPrototypePosterior",
    "PosteriorScalarSummary",
    "BayesianRawFeatMethodSummary",
    "BayesianRawFeatDynamicFPDEResult",
    "build_rawfeat_matrix",
    "fit_rawfeat_scaling",
    "fit_bayesian_rawfeat_prototypes",
    "bayesian_rawfeat_dynamic_fpde_explain_one",
]
