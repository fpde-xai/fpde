"""Metrics, probability helpers, and perturbation utilities used by FPDE."""

from __future__ import annotations

from typing import Any, Dict, Optional, Sequence, Tuple

import numpy as np

from ._array import (
    _as_1d_float,
    _canonical_fraction_array,
    _check_same_dim,
    _class_to_probability_index,
    _predict_proba_matrix,
    _regularized_norm,
    _rounded_feature_counts,
    _safe_auc,
)
from ._batch import _topk_masks_for_counts

def regularized_cosine(u: np.ndarray, v: np.ndarray, eps: float = 1e-12) -> float:
    """Cosine similarity with epsilon-regularized norms."""
    u = _as_1d_float("u", u)
    v = _as_1d_float("v", v)
    _check_same_dim(u, v)
    if eps <= 0.0:
        raise ValueError("eps must be positive")
    return float(np.dot(u, v) / (_regularized_norm(u, eps) * _regularized_norm(v, eps)))


def predict_proba_for_label(model: Any, X: np.ndarray, label: Any) -> np.ndarray:
    """Return predict_proba(X) column corresponding to label."""
    if not hasattr(model, "predict_proba"):
        raise ValueError("model must implement predict_proba")
    proba = np.asarray(model.predict_proba(X), dtype=float)
    if proba.ndim != 2:
        raise ValueError(f"predict_proba must return a 2D array, got shape={proba.shape}")
    if not np.all(np.isfinite(proba)):
        raise ValueError("predict_proba contains NaN or inf")
    class_to_idx = _class_to_probability_index(model)
    if label not in class_to_idx:
        raise ValueError(f"label {label!r} is not in model.classes_={getattr(model, 'classes_', None)!r}")
    return proba[:, int(class_to_idx[label])]


def top_two_labels(model: Any, x: np.ndarray | Sequence[float]) -> Tuple[Any, Optional[Any], np.ndarray]:
    """Return predicted label, rival label, and probability vector for one sample."""
    x_arr = _as_1d_float("x", x)
    if not hasattr(model, "predict_proba"):
        raise ValueError("model must implement predict_proba")
    proba = np.asarray(model.predict_proba(x_arr.reshape(1, -1))[0], dtype=float)
    if proba.ndim != 1 or proba.size == 0:
        raise ValueError("predict_proba for one sample must return a non-empty vector")
    if not np.all(np.isfinite(proba)):
        raise ValueError("predict_proba contains NaN or inf")
    classes = np.asarray(model.classes_, dtype=object)
    if classes.shape[0] != proba.shape[0]:
        raise ValueError("model.classes_ length differs from predict_proba width")
    positive_idx = int(np.argmax(proba))
    if proba.size == 2 and proba[1] >= proba[0]:
        positive_idx = 1
    positive_label = classes[positive_idx]
    if proba.size >= 2:
        if proba.size == 2:
            negative_idx = 1 - positive_idx
        else:
            pair = np.argpartition(proba, kth=proba.size - 2)[-2:]
            negative_idx = int(pair[1] if int(pair[0]) == positive_idx else pair[0])
        negative_label = classes[int(negative_idx)]
    else:
        negative_label = None
    return positive_label, negative_label, proba


def perturbation_curves(
    model: Any,
    x: np.ndarray | Sequence[float],
    attributions: np.ndarray | Sequence[float],
    target_label: Any,
    baseline: np.ndarray | Sequence[float],
    fractions: Sequence[float] = (0.0, 0.05, 0.1, 0.2, 0.3, 0.5, 0.7, 1.0),
) -> Dict[str, Any]:
    """Compute deletion and insertion curves for one explanation vector.

    Important features are ranked by signed positive attribution, descending.
    Deletion replaces selected features with the baseline; insertion starts from
    the baseline and restores selected features.
    """
    x_arr = _as_1d_float("x", x)
    baseline_arr = _as_1d_float("baseline", baseline)
    attr = _as_1d_float("attributions", attributions)
    if x_arr.shape != baseline_arr.shape or x_arr.shape != attr.shape:
        raise ValueError("x, baseline, and attributions must have the same shape")

    d = x_arr.shape[0]

    frac_arr = _canonical_fraction_array(fractions)
    ks, ks_inverse = _rounded_feature_counts(frac_arr, d)
    inner_mask = (ks > 0) & (ks < d)
    inner_ks = ks[inner_mask]
    inner_positions = np.flatnonzero(inner_mask)
    m_inner = int(inner_ks.shape[0])
    include_endpoints = bool(np.any(~inner_mask))
    offset = 2 if include_endpoints else 0

    all_eval = np.empty((offset + 2 * m_inner, d), dtype=float)
    if include_endpoints:
        all_eval[0] = x_arr
        all_eval[1] = baseline_arr
    if m_inner > 0:
        mask = _topk_masks_for_counts(attr.reshape(1, -1), inner_ks)[0]

        deletion_X_arr = all_eval[offset : offset + m_inner]
        insertion_X_arr = all_eval[offset + m_inner :]
        deletion_X_arr[...] = x_arr[None, :]
        np.copyto(deletion_X_arr, baseline_arr[None, :], where=mask)
        insertion_X_arr[...] = baseline_arr[None, :]
        np.copyto(insertion_X_arr, x_arr[None, :], where=mask)

    all_proba = _predict_proba_matrix(model, all_eval)
    class_to_idx = _class_to_probability_index(model)
    if target_label not in class_to_idx:
        raise ValueError(f"label {target_label!r} is not in model.classes_={getattr(model, 'classes_', None)!r}")
    label_idx = int(class_to_idx[target_label])
    if label_idx >= all_proba.shape[1]:
        raise ValueError("target label index exceeds predict_proba width")
    all_prob = all_proba[:, label_idx]

    deletion_prob_unique = np.empty(ks.shape[0], dtype=float)
    insertion_prob_unique = np.empty(ks.shape[0], dtype=float)
    for pos, k in enumerate(ks):
        if int(k) == 0:
            deletion_prob_unique[pos] = all_prob[0]
            insertion_prob_unique[pos] = all_prob[1]
        elif int(k) == d:
            deletion_prob_unique[pos] = all_prob[1]
            insertion_prob_unique[pos] = all_prob[0]
    if m_inner > 0:
        deletion_prob_unique[inner_positions] = all_prob[offset : offset + m_inner]
        insertion_prob_unique[inner_positions] = all_prob[offset + m_inner :]

    deletion_prob = deletion_prob_unique[ks_inverse]
    insertion_prob = insertion_prob_unique[ks_inverse]

    deletion_auc = _safe_auc(frac_arr, deletion_prob)
    insertion_auc = _safe_auc(frac_arr, insertion_prob)
    p0 = float(deletion_prob[0])
    deletion_drop_auc = p0 - deletion_auc
    combined_score = 0.5 * (deletion_drop_auc + insertion_auc)

    return {
        "fractions": frac_arr.tolist(),
        "deletion_prob": deletion_prob.tolist(),
        "insertion_prob": insertion_prob.tolist(),
        "p0": p0,
        "deletion_auc": deletion_auc,
        "deletion_drop_auc": deletion_drop_auc,
        "insertion_auc": insertion_auc,
        "combined_score": combined_score,
    }


__all__ = [
    "regularized_cosine",
    "predict_proba_for_label",
    "top_two_labels",
    "perturbation_curves",
]
