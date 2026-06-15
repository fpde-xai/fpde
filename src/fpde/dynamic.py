"""Dynamic-FPDE for frame-level time-series feature matrices."""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any, Dict, Literal, Optional, Sequence, Tuple

import numpy as np

from ._array import _as_label_array, _safe_auc


@dataclass(frozen=True)
class DynamicFPDEContext:
    """Reusable Dynamic-FPDE state built from variable-length sequences."""

    prototypes: np.ndarray
    prototype_labels: np.ndarray
    prototype_length: int
    n_features: int
    mean_anchor: np.ndarray
    zero_anchor: np.ndarray
    alignment: str
    details: Dict[str, Any]


@dataclass(frozen=True)
class DynamicFPDEExplanation:
    """Result object for one Dynamic-FPDE sequence explanation."""

    mode: str
    evidence: float
    attributions: np.ndarray
    time_importance: np.ndarray
    feature_importance: np.ndarray
    positive_score: float
    negative_score: float
    target_label: Any
    rival_label: Any
    exactness_residual: float
    details: Dict[str, Any]


def _as_dynamic_matrix(name: str, X: np.ndarray | Sequence[Sequence[float]]) -> np.ndarray:
    arr = np.asarray(X, dtype=float)
    if arr.ndim != 2:
        raise ValueError(f"{name} must be a 2D array, got shape={arr.shape}")
    if arr.shape[0] == 0:
        raise ValueError(f"{name} must contain at least one time frame")
    if arr.shape[1] == 0:
        raise ValueError(f"{name} must contain at least one feature")
    if not np.all(np.isfinite(arr)):
        raise ValueError(f"{name} contains NaN or inf")
    return arr


def _check_same_dynamic_shape(*arrays: np.ndarray) -> None:
    if not arrays:
        return
    shape = arrays[0].shape
    for i, arr in enumerate(arrays):
        if arr.shape != shape:
            raise ValueError(f"shape mismatch at array {i}: expected {shape}, got {arr.shape}")


def _validate_positive_int(name: str, value: int) -> int:
    out = int(value)
    if out <= 0:
        raise ValueError(f"{name} must be positive")
    return out


def _validate_eps(eps: float) -> float:
    value = float(eps)
    if not np.isfinite(value) or value <= 0.0:
        raise ValueError("eps must be positive")
    return value


def _validate_lambda_hyb(lambda_hyb: float) -> float:
    value = float(lambda_hyb)
    if not np.isfinite(value):
        raise ValueError("lambda_hyb must be finite")
    if value < 0.0 or value > 1.0:
        raise ValueError("lambda_hyb must be in [0, 1]")
    return value


def _l1_normalized(values: np.ndarray, eps: float) -> Tuple[np.ndarray, float]:
    scale = float(np.sum(np.abs(values)) + eps)
    return values / scale, scale


def _scale_value(value: float, scale: float) -> float:
    return float(value / scale) if scale > 0.0 else 0.0


def _regularized_matrix_norm(values: np.ndarray, eps: float) -> float:
    return float(np.sqrt(np.sum(values * values) + eps))


def _regularized_matrix_cosine(a: np.ndarray, b: np.ndarray, eps: float) -> float:
    return float(np.sum(a * b) / (_regularized_matrix_norm(a, eps) * _regularized_matrix_norm(b, eps)))


def _labels_unique(labels: np.ndarray) -> Tuple[np.ndarray, np.ndarray]:
    try:
        classes, inverse = np.unique(labels, return_inverse=True)
        return classes, inverse.astype(np.intp, copy=False)
    except TypeError:
        classes = np.array(sorted(set(labels.tolist())), dtype=object)
        class_to_idx = {label: i for i, label in enumerate(classes.tolist())}
        inverse = np.fromiter((class_to_idx[label] for label in labels.tolist()), dtype=np.intp, count=labels.shape[0])
        return classes, inverse


def _prototype_index(labels: np.ndarray, label: Any, name: str) -> int:
    matches = np.where(labels == label)[0]
    if matches.size == 0:
        raise ValueError(f"no prototype found for {name}={label!r}")
    return int(matches[0])


def _validate_dynamic_context(context: DynamicFPDEContext) -> DynamicFPDEContext:
    if not isinstance(context, DynamicFPDEContext):
        raise TypeError("context must be a DynamicFPDEContext")
    if context.prototypes.ndim != 3:
        raise ValueError("context prototypes must have shape (C, K, F)")
    c, k, f = context.prototypes.shape
    if c == 0 or k == 0 or f == 0:
        raise ValueError("context prototypes must be non-empty")
    if context.prototype_labels.ndim != 1 or context.prototype_labels.shape[0] != c:
        raise ValueError("context prototype labels have an incompatible shape")
    if int(context.prototype_length) != k:
        raise ValueError("context prototype_length does not match prototypes")
    if int(context.n_features) != f:
        raise ValueError("context n_features does not match prototypes")
    if context.mean_anchor.shape != (k, f) or context.zero_anchor.shape != (k, f):
        raise ValueError("context anchors have an incompatible shape")
    if not np.all(np.isfinite(context.prototypes)):
        raise ValueError("context prototypes contain NaN or inf")
    if not np.all(np.isfinite(context.mean_anchor)) or not np.all(np.isfinite(context.zero_anchor)):
        raise ValueError("context anchors contain NaN or inf")
    return context


def resample_time_series_linear(X: np.ndarray | Sequence[Sequence[float]], target_length: int) -> np.ndarray:
    """Linearly resample a ``(T, F)`` time-series feature matrix to ``target_length`` frames."""
    X_arr = _as_dynamic_matrix("X", X)
    target = _validate_positive_int("target_length", target_length)
    if X_arr.shape[0] == target:
        return X_arr.copy()
    if X_arr.shape[0] == 1:
        return np.repeat(X_arr, target, axis=0).astype(float, copy=False)

    source_pos = np.linspace(0.0, 1.0, X_arr.shape[0])
    target_pos = np.linspace(0.0, 1.0, target)
    out = np.empty((target, X_arr.shape[1]), dtype=float)
    for feature_idx in range(X_arr.shape[1]):
        out[:, feature_idx] = np.interp(target_pos, source_pos, X_arr[:, feature_idx])
    return out


def prepare_dynamic_fpde_context(
    X_train: Sequence[np.ndarray | Sequence[Sequence[float]]],
    y_train: Sequence[Any],
    *,
    prototype_length: int = 128,
    alignment: str = "linear",
    baseline: str = "mean",
) -> DynamicFPDEContext:
    """Build class-mean temporal prototypes for Dynamic-FPDE."""
    if alignment != "linear":
        raise NotImplementedError("only alignment='linear' is supported")
    if baseline not in ("mean", "zero"):
        raise ValueError("baseline must be 'mean' or 'zero'")
    target_length = _validate_positive_int("prototype_length", prototype_length)
    X_items = list(X_train)
    if len(X_items) == 0:
        raise ValueError("X_train must contain at least one sample")
    y_arr = _as_label_array(y_train)
    if y_arr.shape[0] != len(X_items):
        raise ValueError(f"number of train samples and labels differ: {len(X_items)} vs {y_arr.shape[0]}")

    matrices = [_as_dynamic_matrix(f"X_train[{i}]", sample) for i, sample in enumerate(X_items)]
    n_features = int(matrices[0].shape[1])
    for i, sample in enumerate(matrices):
        if sample.shape[1] != n_features:
            raise ValueError(
                f"inconsistent feature dimension at X_train[{i}]: expected {n_features}, got {sample.shape[1]}"
            )

    resampled = np.stack([resample_time_series_linear(sample, target_length) for sample in matrices], axis=0)
    labels, inverse = _labels_unique(y_arr)
    if labels.size == 0:
        raise ValueError("y_train must contain at least one class")

    prototypes = np.empty((labels.shape[0], target_length, n_features), dtype=float)
    counts = np.bincount(inverse, minlength=labels.shape[0]).astype(float)
    for class_idx in range(labels.shape[0]):
        if counts[class_idx] <= 0.0:
            raise ValueError(f"class {labels[class_idx]!r} has no samples")
        prototypes[class_idx] = np.mean(resampled[inverse == class_idx], axis=0)

    mean_anchor = np.mean(resampled, axis=0)
    zero_anchor = np.zeros_like(mean_anchor)
    return DynamicFPDEContext(
        prototypes=prototypes,
        prototype_labels=labels.copy(),
        prototype_length=target_length,
        n_features=n_features,
        mean_anchor=mean_anchor,
        zero_anchor=zero_anchor,
        alignment=alignment,
        details={
            "baseline": baseline,
            "n_train_samples": len(X_items),
            "train_lengths": tuple(int(sample.shape[0]) for sample in matrices),
            "class_counts": {labels[i].item() if hasattr(labels[i], "item") else labels[i]: int(counts[i]) for i in range(labels.shape[0])},
        },
    )


def dynamic_diff_fpde(
    X: np.ndarray | Sequence[Sequence[float]],
    P_target: np.ndarray | Sequence[Sequence[float]],
    P_rival: np.ndarray | Sequence[Sequence[float]],
) -> Tuple[np.ndarray, float]:
    """Compute Dynamic-Diff-FPDE for one target/rival prototype pair."""
    X_arr = _as_dynamic_matrix("X", X)
    target_arr = _as_dynamic_matrix("P_target", P_target)
    rival_arr = _as_dynamic_matrix("P_rival", P_rival)
    _check_same_dynamic_shape(X_arr, target_arr, rival_arr)

    attr = (X_arr - rival_arr) ** 2 - (X_arr - target_arr) ** 2
    return attr, float(np.sum(attr))


def dynamic_cos_fpde(
    X: np.ndarray | Sequence[Sequence[float]],
    P_target: np.ndarray | Sequence[Sequence[float]],
    P_rival: np.ndarray | Sequence[Sequence[float]],
    *,
    anchor: Optional[np.ndarray | Sequence[Sequence[float]]] = None,
    eps: float = 1e-12,
) -> Tuple[np.ndarray, float]:
    """Compute Dynamic-Cos-FPDE as a coordinate decomposition of cosine contrast."""
    eps_value = _validate_eps(eps)
    X_arr = _as_dynamic_matrix("X", X)
    target_arr = _as_dynamic_matrix("P_target", P_target)
    rival_arr = _as_dynamic_matrix("P_rival", P_rival)
    _check_same_dynamic_shape(X_arr, target_arr, rival_arr)
    if anchor is None:
        anchor_arr = np.zeros_like(X_arr, dtype=float)
    else:
        anchor_arr = _as_dynamic_matrix("anchor", anchor)
        _check_same_dynamic_shape(X_arr, anchor_arr)

    z = X_arr - anchor_arr
    q_target = target_arr - anchor_arr
    q_rival = rival_arr - anchor_arr
    target_part = (z * q_target) / (_regularized_matrix_norm(z, eps_value) * _regularized_matrix_norm(q_target, eps_value))
    rival_part = (z * q_rival) / (_regularized_matrix_norm(z, eps_value) * _regularized_matrix_norm(q_rival, eps_value))
    attr = target_part - rival_part
    return attr, float(np.sum(attr))


def dynamic_hyb_fpde(
    X: np.ndarray | Sequence[Sequence[float]],
    P_target: np.ndarray | Sequence[Sequence[float]],
    P_rival: np.ndarray | Sequence[Sequence[float]],
    *,
    lambda_hyb: float = 0.5,
    normalize: str = "l1",
    anchor: Optional[np.ndarray | Sequence[Sequence[float]]] = None,
    eps: float = 1e-12,
) -> Tuple[np.ndarray, float, Dict[str, Any]]:
    """Compute Dynamic-Hyb-FPDE by mixing Dynamic-Diff and Dynamic-Cos attributions."""
    eps_value = _validate_eps(eps)
    lambda_value = _validate_lambda_hyb(lambda_hyb)
    if normalize not in ("l1", "none"):
        raise ValueError("normalize must be either 'l1' or 'none'")

    diff_attr, diff_evidence = dynamic_diff_fpde(X, P_target, P_rival)
    cos_attr, cos_evidence = dynamic_cos_fpde(X, P_target, P_rival, anchor=anchor, eps=eps_value)
    if normalize == "l1":
        diff_part, diff_scale = _l1_normalized(diff_attr, eps_value)
        cos_part, cos_scale = _l1_normalized(cos_attr, eps_value)
    else:
        diff_part, cos_part = diff_attr, cos_attr
        diff_scale = cos_scale = 1.0

    attr = lambda_value * diff_part + (1.0 - lambda_value) * cos_part
    evidence = float(np.sum(attr))
    return (
        attr,
        evidence,
        {
            "diff_evidence": float(diff_evidence),
            "cos_evidence": float(cos_evidence),
            "lambda_hyb": float(lambda_value),
            "normalize": normalize,
            "diff_scale": float(diff_scale),
            "cos_scale": float(cos_scale),
        },
    )


def dynamic_fpde_explain_one(
    X: np.ndarray | Sequence[Sequence[float]],
    context: DynamicFPDEContext,
    *,
    target_label: Any,
    rival_label: Optional[Any] = None,
    mode: str = "dynamic_hyb",
    lambda_hyb: float = 0.5,
    normalize: str = "l1",
    anchor_strategy: str = "mean",
    eps: float = 1e-12,
) -> DynamicFPDEExplanation:
    """Explain one variable-length sequence with Dynamic-FPDE."""
    X_arr = _as_dynamic_matrix("X", X)
    ctx = _validate_dynamic_context(context)
    if X_arr.shape[1] != ctx.n_features:
        raise ValueError(f"feature dimension mismatch: context has {ctx.n_features}, X has {X_arr.shape[1]}")
    if mode not in ("dynamic_diff", "dynamic_cos", "dynamic_hyb"):
        raise ValueError("mode must be 'dynamic_diff', 'dynamic_cos', or 'dynamic_hyb'")
    if anchor_strategy not in ("mean", "zero", "none"):
        raise ValueError("anchor_strategy must be 'mean', 'zero', or 'none'")

    target_idx = _prototype_index(ctx.prototype_labels, target_label, "target_label")
    target_proto = resample_time_series_linear(ctx.prototypes[target_idx], X_arr.shape[0])
    if rival_label is None:
        candidate_indices = np.where(ctx.prototype_labels != target_label)[0]
        if candidate_indices.size == 0:
            raise ValueError("rival_label is None, but no non-target prototypes exist")
        distances = []
        for idx in candidate_indices:
            candidate = resample_time_series_linear(ctx.prototypes[int(idx)], X_arr.shape[0])
            distances.append(float(np.sum((X_arr - candidate) ** 2)))
        rival_idx = int(candidate_indices[int(np.argmin(np.asarray(distances, dtype=float)))])
    else:
        rival_idx = _prototype_index(ctx.prototype_labels, rival_label, "rival_label")
        if ctx.prototype_labels[rival_idx] == target_label:
            raise ValueError("rival_label must differ from target_label")
    resolved_rival_label = ctx.prototype_labels[rival_idx]
    rival_proto = resample_time_series_linear(ctx.prototypes[rival_idx], X_arr.shape[0])

    if anchor_strategy == "mean":
        anchor = resample_time_series_linear(ctx.mean_anchor, X_arr.shape[0])
    else:
        anchor = np.zeros_like(X_arr, dtype=float)

    diff_attr, diff_evidence = dynamic_diff_fpde(X_arr, target_proto, rival_proto)
    diff_positive_score = -float(np.sum((X_arr - target_proto) ** 2))
    diff_negative_score = -float(np.sum((X_arr - rival_proto) ** 2))
    cos_attr, cos_evidence = dynamic_cos_fpde(X_arr, target_proto, rival_proto, anchor=anchor, eps=eps)
    z = X_arr - anchor
    q_target = target_proto - anchor
    q_rival = rival_proto - anchor
    cos_positive_score = _regularized_matrix_cosine(z, q_target, _validate_eps(eps))
    cos_negative_score = _regularized_matrix_cosine(z, q_rival, _validate_eps(eps))

    details: Dict[str, Any] = {
        "target_prototype_index": int(target_idx),
        "rival_prototype_index": int(rival_idx),
        "anchor_strategy": anchor_strategy,
        "eps": float(eps),
        "diff_evidence": float(diff_evidence),
        "cos_evidence": float(cos_evidence),
        "diff_positive_score": float(diff_positive_score),
        "diff_negative_score": float(diff_negative_score),
        "cos_positive_score": float(cos_positive_score),
        "cos_negative_score": float(cos_negative_score),
    }

    if mode == "dynamic_diff":
        attributions = diff_attr
        evidence = float(diff_evidence)
        positive_score = diff_positive_score
        negative_score = diff_negative_score
    elif mode == "dynamic_cos":
        attributions = cos_attr
        evidence = float(cos_evidence)
        positive_score = cos_positive_score
        negative_score = cos_negative_score
    else:
        lambda_value = _validate_lambda_hyb(lambda_hyb)
        attributions, evidence, hyb_details = dynamic_hyb_fpde(
            X_arr,
            target_proto,
            rival_proto,
            lambda_hyb=lambda_value,
            normalize=normalize,
            anchor=anchor,
            eps=eps,
        )
        details.update(hyb_details)
        if normalize == "none":
            positive_score = lambda_value * diff_positive_score + (1.0 - lambda_value) * cos_positive_score
            negative_score = lambda_value * diff_negative_score + (1.0 - lambda_value) * cos_negative_score
            details["hyb_score_definition"] = "raw weighted Diff/Cos positive and negative scores"
        elif normalize == "l1":
            diff_scale = float(hyb_details["diff_scale"])
            cos_scale = float(hyb_details["cos_scale"])
            positive_score = lambda_value * _scale_value(diff_positive_score, diff_scale)
            positive_score += (1.0 - lambda_value) * _scale_value(cos_positive_score, cos_scale)
            negative_score = lambda_value * _scale_value(diff_negative_score, diff_scale)
            negative_score += (1.0 - lambda_value) * _scale_value(cos_negative_score, cos_scale)
            details["hyb_score_definition"] = "weighted Diff/Cos scores divided by each component attribution L1 scale"
        else:
            raise ValueError("normalize must be either 'l1' or 'none'")
        details["hyb_positive_score"] = float(positive_score)
        details["hyb_negative_score"] = float(negative_score)

    time_importance = np.sum(attributions, axis=1)
    feature_importance = np.sum(attributions, axis=0)
    residual = float(evidence - np.sum(attributions))
    return DynamicFPDEExplanation(
        mode=mode,
        evidence=float(evidence),
        attributions=attributions.astype(float, copy=True),
        time_importance=time_importance.astype(float, copy=True),
        feature_importance=feature_importance.astype(float, copy=True),
        positive_score=float(positive_score),
        negative_score=float(negative_score),
        target_label=target_label,
        rival_label=resolved_rival_label,
        exactness_residual=residual,
        details=details,
    )


def dynamic_fpde_explain_batch(
    X_list: Sequence[np.ndarray | Sequence[Sequence[float]]],
    context: DynamicFPDEContext,
    *,
    target_labels: Sequence[Any],
    rival_labels: Optional[Sequence[Any]] = None,
    mode: str = "dynamic_hyb",
    lambda_hyb: float = 0.5,
    normalize: str = "l1",
    anchor_strategy: str = "mean",
    eps: float = 1e-12,
) -> list[DynamicFPDEExplanation]:
    """Explain a batch of variable-length sequences."""
    X_items = list(X_list)
    targets = list(target_labels)
    if len(X_items) != len(targets):
        raise ValueError(f"number of samples and target_labels differ: {len(X_items)} vs {len(targets)}")
    rivals = None if rival_labels is None else list(rival_labels)
    if rivals is not None and len(rivals) != len(X_items):
        raise ValueError(f"number of samples and rival_labels differ: {len(X_items)} vs {len(rivals)}")
    return [
        dynamic_fpde_explain_one(
            sample,
            context,
            target_label=targets[i],
            rival_label=None if rivals is None else rivals[i],
            mode=mode,
            lambda_hyb=lambda_hyb,
            normalize=normalize,
            anchor_strategy=anchor_strategy,
            eps=eps,
        )
        for i, sample in enumerate(X_items)
    ]


def _dynamic_diff_evidence_for_labels(
    X: np.ndarray,
    context: DynamicFPDEContext,
    target_label: Any,
    rival_label: Any,
) -> float:
    target_idx = _prototype_index(context.prototype_labels, target_label, "target_label")
    rival_idx = _prototype_index(context.prototype_labels, rival_label, "rival_label")
    target_proto = resample_time_series_linear(context.prototypes[target_idx], X.shape[0])
    rival_proto = resample_time_series_linear(context.prototypes[rival_idx], X.shape[0])
    _, evidence = dynamic_diff_fpde(X, target_proto, rival_proto)
    return float(evidence)


def temporal_deletion_insertion_curves(
    X: np.ndarray | Sequence[Sequence[float]],
    explanation: DynamicFPDEExplanation,
    context: DynamicFPDEContext,
    *,
    target_label: Any,
    rival_label: Any,
    steps: int = 20,
    baseline_strategy: str = "mean",
    rank_by: Literal["positive", "signed", "absolute"] = "positive",
    eps: float = 1e-12,
) -> Dict[str, Any]:
    """Compute normalized prototype-evidence deletion and insertion curves."""
    eps_value = _validate_eps(eps)
    X_arr = _as_dynamic_matrix("X", X)
    ctx = _validate_dynamic_context(context)
    if X_arr.shape[1] != ctx.n_features:
        raise ValueError(f"feature dimension mismatch: context has {ctx.n_features}, X has {X_arr.shape[1]}")
    if not isinstance(explanation, DynamicFPDEExplanation):
        raise TypeError("explanation must be a DynamicFPDEExplanation")
    if explanation.time_importance.shape != (X_arr.shape[0],):
        raise ValueError("explanation time_importance must match X time length")
    step_count = _validate_positive_int("steps", steps)
    if baseline_strategy == "mean":
        baseline = resample_time_series_linear(ctx.mean_anchor, X_arr.shape[0])
    elif baseline_strategy in ("zero", "none"):
        baseline = np.zeros_like(X_arr, dtype=float)
    else:
        raise ValueError("baseline_strategy must be 'mean', 'zero', or 'none'")
    if rank_by not in ("positive", "signed", "absolute"):
        raise ValueError("rank_by must be 'positive', 'signed', or 'absolute'")

    fractions = np.linspace(0.0, 1.0, min(step_count, X_arr.shape[0]) + 1)
    counts = np.unique(np.rint(fractions * X_arr.shape[0]).astype(np.intp))
    if counts[0] != 0:
        counts = np.concatenate([[0], counts])
    if counts[-1] != X_arr.shape[0]:
        counts = np.concatenate([counts, [X_arr.shape[0]]])
    fractions = counts.astype(float) / float(X_arr.shape[0])
    if rank_by == "positive":
        rank_scores = np.maximum(explanation.time_importance, 0.0)
    elif rank_by == "signed":
        rank_scores = explanation.time_importance
    else:
        rank_scores = np.abs(explanation.time_importance)
    order = np.argsort(rank_scores)[::-1]

    deletion_curve = []
    insertion_curve = []
    original_evidence = _dynamic_diff_evidence_for_labels(X_arr, ctx, target_label, rival_label)
    baseline_evidence = _dynamic_diff_evidence_for_labels(baseline, ctx, target_label, rival_label)
    for count in counts:
        selected = order[: int(count)]
        deletion = X_arr.copy()
        deletion[selected] = baseline[selected]
        insertion = baseline.copy()
        insertion[selected] = X_arr[selected]
        deletion_curve.append(_dynamic_diff_evidence_for_labels(deletion, ctx, target_label, rival_label))
        insertion_curve.append(_dynamic_diff_evidence_for_labels(insertion, ctx, target_label, rival_label))

    deletion_arr = np.asarray(deletion_curve, dtype=float)
    insertion_arr = np.asarray(insertion_curve, dtype=float)
    scale = float(abs(original_evidence - baseline_evidence) + eps_value)
    deletion_drop_curve = (original_evidence - deletion_arr) / scale
    insertion_gain_curve = (insertion_arr - insertion_arr[0]) / scale
    deletion_drop_auc = _safe_auc(fractions, deletion_drop_curve)
    insertion_gain_auc = _safe_auc(fractions, insertion_gain_curve)
    combined_score = float(0.5 * (deletion_drop_auc + insertion_gain_auc))
    return {
        "fractions": fractions.tolist(),
        "deletion_curve": deletion_arr.tolist(),
        "insertion_curve": insertion_arr.tolist(),
        "deletion_drop_curve": deletion_drop_curve.tolist(),
        "insertion_gain_curve": insertion_gain_curve.tolist(),
        "original_evidence": float(original_evidence),
        "baseline_evidence": float(baseline_evidence),
        "evidence_scale": scale,
        "deletion_drop_auc": float(deletion_drop_auc),
        "insertion_gain_auc": float(insertion_gain_auc),
        "insertion_auc": float(insertion_gain_auc),
        "combined_score": combined_score,
        "rank_by": rank_by,
    }


def select_dynamic_lambda(
    X_val: Sequence[np.ndarray | Sequence[Sequence[float]]],
    y_val: Sequence[Any],
    context: DynamicFPDEContext,
    *,
    lambda_grid: Optional[Sequence[float]] = None,
    mode: str = "dynamic_hyb",
    normalize: str = "l1",
    anchor_strategy: str = "mean",
    steps: int = 20,
    rank_by: Literal["positive", "signed", "absolute"] = "positive",
    eps: float = 1e-12,
) -> Dict[str, Any]:
    """Select a Dynamic-Hyb lambda using prototype-driven temporal curves."""
    if mode != "dynamic_hyb":
        raise ValueError("select_dynamic_lambda currently supports mode='dynamic_hyb' only")
    X_items = list(X_val)
    y_arr = _as_label_array(y_val)
    if len(X_items) == 0:
        raise ValueError("X_val must contain at least one sample")
    if y_arr.shape[0] != len(X_items):
        raise ValueError(f"number of validation samples and labels differ: {len(X_items)} vs {y_arr.shape[0]}")
    grid = [0.0, 0.25, 0.5, 0.75, 1.0] if lambda_grid is None else [float(v) for v in lambda_grid]
    if not grid:
        raise ValueError("lambda_grid must contain at least one value")
    lambdas = [_validate_lambda_hyb(value) for value in grid]

    rows = []
    for candidate_id, lambda_value in enumerate(lambdas):
        scores = []
        deletion_scores = []
        insertion_scores = []
        evidences = []
        for sample, target in zip(X_items, y_arr.tolist()):
            explanation = dynamic_fpde_explain_one(
                sample,
                context,
                target_label=target,
                rival_label=None,
                mode=mode,
                lambda_hyb=lambda_value,
                normalize=normalize,
                anchor_strategy=anchor_strategy,
                eps=eps,
            )
            curves = temporal_deletion_insertion_curves(
                sample,
                explanation,
                context,
                target_label=target,
                rival_label=explanation.rival_label,
                steps=steps,
                baseline_strategy=anchor_strategy,
                rank_by=rank_by,
                eps=eps,
            )
            scores.append(float(curves["combined_score"]))
            deletion_scores.append(float(curves["deletion_drop_auc"]))
            insertion_scores.append(float(curves["insertion_gain_auc"]))
            evidences.append(float(explanation.evidence))
        rows.append(
            {
                "candidate_id": int(candidate_id),
                "lambda_hyb": float(lambda_value),
                "status": "ok",
                "score": float(np.mean(scores)),
                "mean_combined_score": float(np.mean(scores)),
                "mean_deletion_drop_auc": float(np.mean(deletion_scores)),
                "mean_insertion_gain_auc": float(np.mean(insertion_scores)),
                "mean_insertion_auc": float(np.mean(insertion_scores)),
                "mean_evidence": float(np.mean(evidences)),
                "n_eval_samples": len(X_items),
                "normalize": normalize,
                "anchor_strategy": anchor_strategy,
                "rank_by": rank_by,
                "metric_source": "normalized_prototype_evidence_curves",
            }
        )

    best_row = max(
        rows,
        key=lambda row: (
            float(row["score"]),
            float(row["mean_insertion_gain_auc"]),
            -abs(float(row["lambda_hyb"]) - 0.5),
        ),
    )
    return {
        "best_lambda": float(best_row["lambda_hyb"]),
        "rows": rows,
        "metric_means": {
            str(row["lambda_hyb"]): {
                "combined_score": float(row["mean_combined_score"]),
                "deletion_drop_auc": float(row["mean_deletion_drop_auc"]),
                "insertion_gain_auc": float(row["mean_insertion_gain_auc"]),
                "insertion_auc": float(row["mean_insertion_auc"]),
                "evidence": float(row["mean_evidence"]),
            }
            for row in rows
        },
        "best_row": dict(best_row),
    }


def _pyplot():
    try:
        import matplotlib.pyplot as plt
    except ImportError as exc:
        raise ImportError(
            "Dynamic-FPDE plotting requires matplotlib. Install it with "
            "`python -m pip install fpde[plot]` or pass an existing matplotlib Axes."
        ) from exc
    return plt


def _axes(ax: Optional[Any], *, figsize: Tuple[float, float]) -> Any:
    if ax is not None:
        return ax
    _, new_ax = _pyplot().subplots(figsize=figsize)
    return new_ax


def plot_dynamic_time_importance(
    explanation: DynamicFPDEExplanation,
    *,
    ax: Optional[Any] = None,
    title: Optional[str] = None,
    positive_color: str = "#2563eb",
    negative_color: str = "#dc2626",
) -> Any:
    """Plot signed frame-level importance for a Dynamic-FPDE explanation."""
    if not isinstance(explanation, DynamicFPDEExplanation):
        raise TypeError("explanation must be a DynamicFPDEExplanation")
    values = _as_dynamic_matrix("attributions", explanation.attributions)
    time_importance = np.sum(values, axis=1)
    colors = [positive_color if value >= 0.0 else negative_color for value in time_importance]
    ax = _axes(ax, figsize=(7.0, 3.0))
    ax.bar(np.arange(time_importance.shape[0]), time_importance, color=colors)
    ax.axhline(0.0, color="#111827", linewidth=0.8)
    ax.set_xlabel("Frame")
    ax.set_ylabel("Attribution")
    if title is not None:
        ax.set_title(title)
    return ax


def plot_dynamic_attribution_heatmap(
    explanation: DynamicFPDEExplanation,
    *,
    ax: Optional[Any] = None,
    title: Optional[str] = None,
    cmap: str = "coolwarm",
    colorbar: bool = True,
    symmetric: bool = True,
) -> Any:
    """Plot a Dynamic-FPDE ``(T, F)`` attribution matrix as a heatmap."""
    if not isinstance(explanation, DynamicFPDEExplanation):
        raise TypeError("explanation must be a DynamicFPDEExplanation")
    values = _as_dynamic_matrix("attributions", explanation.attributions)
    if symmetric:
        vmax = float(np.max(np.abs(values)))
        vmin = -vmax if vmax > 0.0 else None
        vmax_arg = vmax if vmax > 0.0 else None
    else:
        vmin = vmax_arg = None
    ax = _axes(ax, figsize=(6.4, 3.8))
    image = ax.imshow(values, aspect="auto", cmap=cmap, vmin=vmin, vmax=vmax_arg, interpolation="nearest")
    ax.set_xlabel("Feature")
    ax.set_ylabel("Frame")
    if title is not None:
        ax.set_title(title)
    if colorbar:
        ax.figure.colorbar(image, ax=ax, fraction=0.046, pad=0.04)
    return ax


__all__ = [
    "DynamicFPDEContext",
    "DynamicFPDEExplanation",
    "resample_time_series_linear",
    "prepare_dynamic_fpde_context",
    "dynamic_diff_fpde",
    "dynamic_cos_fpde",
    "dynamic_hyb_fpde",
    "dynamic_fpde_explain_one",
    "dynamic_fpde_explain_batch",
    "temporal_deletion_insertion_curves",
    "select_dynamic_lambda",
    "plot_dynamic_time_importance",
    "plot_dynamic_attribution_heatmap",
]
