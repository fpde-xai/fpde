"""Prototype construction, selection, and reusable context helpers."""

from __future__ import annotations

from typing import Any, Optional, Sequence, Tuple

import numpy as np

from ._array import _as_1d_float, _as_2d_float, _as_label_array, _check_same_dim, _regularized_norm
from .types import AnchorStrategy, FPDEContext, GridMode

def select_prototype_pair(
    x: np.ndarray | Sequence[float],
    prototypes: np.ndarray | Sequence[Sequence[float]],
    prototype_labels: Sequence[Any],
    *,
    positive_label: Any,
    negative_label: Optional[Any] = None,
    mode: GridMode = "diff",
    anchor: Optional[np.ndarray | Sequence[float]] = None,
    eps: float = 1e-12,
) -> Tuple[int, int]:
    """Select a positive and negative prototype for a local contrast.

    Diff mode:
        Select nearest positive and nearest negative prototype by squared distance.

    Cos mode:
        Select most cosine-similar positive and negative prototype.

    Hyb-FPDE mode:
        Use the Diff selection rule by default. The lambda-grid hybridization is
        applied to the attribution formula, not to prototype selection.
    """
    x_arr = _as_1d_float("x", x)
    P = _as_2d_float("prototypes", prototypes)
    labels = _as_label_array(prototype_labels)

    if P.shape[0] != labels.shape[0]:
        raise ValueError(f"number of prototypes and labels differ: {P.shape[0]} vs {labels.shape[0]}")
    if P.shape[1] != x_arr.shape[0]:
        raise ValueError(f"prototype dimension mismatch: expected {x_arr.shape[0]}, got {P.shape[1]}")

    pos_indices = np.where(labels == positive_label)[0]
    if pos_indices.size == 0:
        raise ValueError(f"no prototype found for positive_label={positive_label!r}")

    if negative_label is None:
        neg_indices = np.where(labels != positive_label)[0]
        if neg_indices.size == 0:
            raise ValueError("negative_label is None, but no non-positive prototypes exist")
    else:
        neg_indices = np.where(labels == negative_label)[0]
        if neg_indices.size == 0:
            raise ValueError(f"no prototype found for negative_label={negative_label!r}")

    if mode in ("diff", "hyb_grid"):
        x_sq = float(np.dot(x_arr, x_arr))
        P_pos = P[pos_indices]
        P_neg = P[neg_indices]
        pos_d = np.einsum("ij,ij->i", P_pos, P_pos) - 2.0 * (P_pos @ x_arr) + x_sq
        neg_d = np.einsum("ij,ij->i", P_neg, P_neg) - 2.0 * (P_neg @ x_arr) + x_sq
        pos_idx = int(pos_indices[int(np.argmin(pos_d))])
        neg_idx = int(neg_indices[int(np.argmin(neg_d))])
        return pos_idx, neg_idx

    if mode == "cos":
        if eps <= 0.0:
            raise ValueError("eps must be positive")
        if anchor is None:
            anchor_arr = np.zeros_like(x_arr, dtype=float)
        else:
            anchor_arr = _as_1d_float("anchor", anchor)
            _check_same_dim(x_arr, anchor_arr)

        z = x_arr - anchor_arr
        n_z = _regularized_norm(z, eps)
        q_pos = P[pos_indices] - anchor_arr[None, :]
        q_neg = P[neg_indices] - anchor_arr[None, :]
        pos_norms = np.sqrt(np.einsum("ij,ij->i", q_pos, q_pos) + eps * eps)
        neg_norms = np.sqrt(np.einsum("ij,ij->i", q_neg, q_neg) + eps * eps)
        pos_scores = (q_pos @ z) / (pos_norms * n_z)
        neg_scores = (q_neg @ z) / (neg_norms * n_z)

        pos_idx = int(pos_indices[int(np.argmax(pos_scores))])
        neg_idx = int(neg_indices[int(np.argmax(neg_scores))])
        return pos_idx, neg_idx

    raise ValueError(f"unknown mode={mode!r}")


def class_mean_prototypes(
    X: np.ndarray | Sequence[Sequence[float]],
    y: Sequence[Any],
) -> Tuple[np.ndarray, np.ndarray]:
    """Build one mean prototype per class.

    CPU-optimized implementation: the class labels are mapped once and all
    feature sums are accumulated in a single pass over X, avoiding one full
    boolean slice of X per class.
    """
    X_arr = _as_2d_float("X", X)
    y_arr = _as_label_array(y)
    if X_arr.shape[0] == 0:
        raise ValueError("X must contain at least one sample")
    if X_arr.shape[0] != y_arr.shape[0]:
        raise ValueError(f"number of samples and labels differ: {X_arr.shape[0]} vs {y_arr.shape[0]}")

    try:
        classes, inverse = np.unique(y_arr, return_inverse=True)
        inverse = inverse.astype(np.intp, copy=False)
    except TypeError:
        classes = np.array(sorted(set(y_arr.tolist())), dtype=object)
        if classes.size == 0:
            raise ValueError("y must contain at least one class")
        class_to_idx = {c: i for i, c in enumerate(classes.tolist())}
        inverse = np.fromiter((class_to_idx[v] for v in y_arr.tolist()), dtype=np.intp, count=y_arr.shape[0])

    if classes.size == 0:
        raise ValueError("y must contain at least one class")
    counts = np.bincount(inverse, minlength=classes.shape[0]).astype(float)
    if np.any(counts <= 0.0):
        missing = classes[np.where(counts <= 0.0)[0][0]]
        raise ValueError(f"class {missing!r} has no samples")

    n_classes = int(classes.shape[0])
    dense_indicator_size = X_arr.shape[0] * n_classes
    if X_arr.shape[1] >= 30 and dense_indicator_size <= 8_000_000:
        indicator = np.zeros((X_arr.shape[0], n_classes), dtype=float)
        indicator[np.arange(X_arr.shape[0]), inverse] = 1.0
        prototypes = indicator.T @ X_arr
    else:
        prototypes = np.empty((n_classes, X_arr.shape[1]), dtype=float)
        for j in range(X_arr.shape[1]):
            prototypes[:, j] = np.bincount(inverse, weights=X_arr[:, j], minlength=n_classes)
    prototypes /= counts[:, None]
    return prototypes, classes


def prepare_fpde_context(
    X_train: np.ndarray | Sequence[Sequence[float]],
    y_train: Sequence[Any],
    *,
    baseline: Optional[np.ndarray | Sequence[float]] = None,
) -> FPDEContext:
    """Precompute reusable prototypes, anchors, and baseline for repeated FPDE calls."""
    X_train_arr = _as_2d_float("X_train", X_train)
    y_train_arr = _as_label_array(y_train)
    if X_train_arr.shape[0] != y_train_arr.shape[0]:
        raise ValueError(
            f"number of train samples and labels differ: {X_train_arr.shape[0]} vs {y_train_arr.shape[0]}"
        )
    prototypes, labels = class_mean_prototypes(X_train_arr, y_train_arr)
    mean_anchor = np.mean(X_train_arr, axis=0)
    baseline_arr = mean_anchor.copy() if baseline is None else _as_1d_float("baseline", baseline)
    if baseline_arr.shape[0] != X_train_arr.shape[1]:
        raise ValueError("baseline dimension differs from X_train feature dimension")
    return FPDEContext(
        prototypes=prototypes.copy(),
        prototype_labels=labels.copy(),
        mean_anchor=mean_anchor,
        zero_anchor=np.zeros(X_train_arr.shape[1], dtype=float),
        baseline=baseline_arr.copy(),
        n_features=int(X_train_arr.shape[1]),
    )


def _validate_fpde_context(context: FPDEContext, n_features: int) -> FPDEContext:
    if int(context.n_features) != int(n_features):
        raise ValueError(f"context feature dimension mismatch: expected {n_features}, got {context.n_features}")
    if context.prototypes.ndim != 2 or context.prototypes.shape[1] != n_features:
        raise ValueError("context prototypes have an incompatible shape")
    if context.prototype_labels.ndim != 1 or context.prototype_labels.shape[0] != context.prototypes.shape[0]:
        raise ValueError("context prototype labels have an incompatible shape")
    if context.mean_anchor.shape != (n_features,) or context.zero_anchor.shape != (n_features,):
        raise ValueError("context anchors have an incompatible shape")
    if context.baseline.shape != (n_features,):
        raise ValueError("context baseline has an incompatible shape")
    return context


def _context_from_training(
    X_train_arr: np.ndarray,
    y_train_arr: np.ndarray,
    *,
    baseline: Optional[np.ndarray | Sequence[float]] = None,
    context: Optional[FPDEContext] = None,
) -> FPDEContext:
    if context is not None:
        ctx = _validate_fpde_context(context, X_train_arr.shape[1])
        if X_train_arr.shape[0] != y_train_arr.shape[0]:
            raise ValueError(
                f"number of train samples and labels differ: {X_train_arr.shape[0]} vs {y_train_arr.shape[0]}"
            )
        if baseline is not None:
            baseline_arr = _as_1d_float("baseline", baseline)
            if baseline_arr.shape[0] != X_train_arr.shape[1]:
                raise ValueError("baseline dimension differs from X_train feature dimension")
        return ctx
    return prepare_fpde_context(X_train_arr, y_train_arr, baseline=baseline)


def _anchor_from_context(context: FPDEContext, anchor_strategy: AnchorStrategy) -> np.ndarray:
    if anchor_strategy == "mean":
        return context.mean_anchor
    if anchor_strategy in ("zero", "none"):
        return context.zero_anchor
    raise ValueError(f"unknown anchor_strategy={anchor_strategy!r}")


def _nearest_label_from_prototypes(
    x: np.ndarray,
    prototypes: np.ndarray,
    prototype_labels: np.ndarray,
) -> Any:
    d = np.sum((prototypes - x[None, :]) ** 2, axis=1)
    return prototype_labels[int(np.argmin(d))]


__all__ = [
    "class_mean_prototypes",
    "select_prototype_pair",
    "prepare_fpde_context",
]
