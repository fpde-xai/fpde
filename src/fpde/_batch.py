"""Vectorized FPDE component and perturbation helpers."""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any, Dict, Literal, Optional, Tuple

import numpy as np

from ._array import (
    _predict_proba_matrix,
    _rounded_feature_counts,
)
from ._backend import as_numpy_array, as_numpy_dict
from .types import NormalizeMode


@dataclass(frozen=True)
class _PerturbationChunkPlan:
    axis: Literal["none", "sample", "lambda"]
    chunk_size: int


def _plan_single_perturbation_chunks(
    *,
    n_samples: int,
    bytes_per_row: int,
    max_working_bytes: int,
) -> _PerturbationChunkPlan:
    if n_samples * bytes_per_row <= max_working_bytes:
        return _PerturbationChunkPlan(axis="none", chunk_size=n_samples)
    chunk_size = int(max_working_bytes // max(bytes_per_row, 1))
    if chunk_size < 1:
        raise MemoryError("perturbation batch would exceed memory budget")
    return _PerturbationChunkPlan(axis="sample", chunk_size=chunk_size)


def _plan_lambda_grid_perturbation_chunks(
    *,
    n_samples: int,
    n_lambdas: int,
    bytes_per_lambda: int,
    bytes_per_sample_all_lambdas: int,
    max_working_bytes: int,
) -> _PerturbationChunkPlan:
    estimated_bytes = n_lambdas * bytes_per_lambda
    if estimated_bytes <= max_working_bytes:
        return _PerturbationChunkPlan(axis="none", chunk_size=n_lambdas)

    sample_chunk_size = int(max_working_bytes // max(bytes_per_sample_all_lambdas, 1))
    lambda_chunk_size = int(max_working_bytes // max(bytes_per_lambda, 1))
    sample_chunks = (n_samples + sample_chunk_size - 1) // sample_chunk_size if sample_chunk_size >= 1 else np.inf
    lambda_chunks = (n_lambdas + lambda_chunk_size - 1) // lambda_chunk_size if lambda_chunk_size >= 1 else np.inf

    if sample_chunks <= lambda_chunks and sample_chunk_size < n_samples:
        return _PerturbationChunkPlan(axis="sample", chunk_size=sample_chunk_size)
    if lambda_chunk_size < 1:
        raise MemoryError("lambda-grid perturbation batch would exceed memory budget")
    return _PerturbationChunkPlan(axis="lambda", chunk_size=lambda_chunk_size)


def _truth(value: Any) -> bool:
    return bool(value.item()) if hasattr(value, "item") else bool(value)


def _full_topk_mask(scores: np.ndarray, ks: np.ndarray, *, array_namespace: Any = np) -> np.ndarray:
    """Return masks equivalent to ranking features by descending scores."""
    xp = array_namespace
    n, d = scores.shape
    order = xp.argsort(-scores, axis=1)
    ranks = xp.empty_like(order, dtype=np.intp)
    ranks[xp.arange(n)[:, None], order] = xp.arange(d, dtype=np.intp)[None, :]
    return ranks[:, None, :] < ks[None, :, None]


def _topk_masks_for_counts(scores: np.ndarray, ks: np.ndarray, *, array_namespace: Any = np) -> np.ndarray:
    """Build top-k masks, avoiding full feature sorts when requested k is small."""
    xp = array_namespace
    scores_arr = xp.asarray(scores, dtype=float)
    ks_values = [int(k) for k in np.asarray(ks, dtype=np.intp).tolist()]
    ks_arr = xp.asarray(ks_values, dtype=np.intp)
    if scores_arr.ndim != 2:
        raise ValueError(f"scores must be a 2D array, got shape={scores_arr.shape}")
    n, d = scores_arr.shape
    if ks_arr.ndim != 1:
        raise ValueError("ks must be a 1D array")
    if ks_arr.size == 0:
        return xp.zeros((n, 0, d), dtype=bool)
    max_k = max(ks_values)
    if max_k <= 0:
        return xp.zeros((n, ks_arr.shape[0], d), dtype=bool)
    if d <= 128 or max_k >= d or max_k * 2 >= d:
        return _full_topk_mask(scores_arr, ks_arr, array_namespace=xp)

    mask = xp.zeros((n, ks_arr.shape[0], d), dtype=bool)
    candidate_idx = xp.argpartition(scores_arr, kth=d - max_k, axis=1)[:, -max_k:]
    candidate_scores = xp.take_along_axis(scores_arr, candidate_idx, axis=1)
    candidate_order = xp.argsort(-candidate_scores, axis=1)
    sorted_idx = xp.take_along_axis(candidate_idx, candidate_order, axis=1)
    sorted_scores = xp.take_along_axis(candidate_scores, candidate_order, axis=1)

    tied_rows = xp.zeros(n, dtype=bool)
    for k_int in ks_values:
        if k_int <= 0:
            continue
        threshold = sorted_scores[:, k_int - 1]
        greater = xp.sum(scores_arr > threshold[:, None], axis=1)
        equal = xp.sum(scores_arr == threshold[:, None], axis=1)
        tied_rows |= (greater < k_int) & (k_int < greater + equal)

    fast_rows = ~tied_rows
    if _truth(xp.any(fast_rows)):
        fast_out = mask[fast_rows]
        fast_sorted_idx = sorted_idx[fast_rows]
        row = xp.arange(fast_sorted_idx.shape[0])[:, None]
        for pos, k_int in enumerate(ks_values):
            if k_int > 0:
                fast_out[row, pos, fast_sorted_idx[:, :k_int]] = True
        mask[fast_rows] = fast_out
    if _truth(xp.any(tied_rows)):
        mask[tied_rows] = _full_topk_mask(scores_arr[tied_rows], ks_arr, array_namespace=xp)
    return mask


def _topk_mask_working_bytes_per_row(n_features: int, ks: np.ndarray) -> int:
    if ks.size == 0:
        return 0
    max_k = int(np.max(ks))
    mask_bytes = int(ks.shape[0]) * int(n_features)
    if max_k <= 0:
        return mask_bytes
    if n_features <= 128 or max_k >= n_features or max_k * 2 >= n_features:
        return mask_bytes + int(n_features) * np.dtype(np.intp).itemsize
    return mask_bytes + 2 * max_k * np.dtype(np.intp).itemsize + max_k * np.dtype(float).itemsize


def _scale_attribution_batch(
    attr: np.ndarray,
    evidence: np.ndarray,
    positive_score: np.ndarray,
    negative_score: np.ndarray,
    *,
    normalize: NormalizeMode,
    eps: float,
    array_namespace: Any = np,
) -> Tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
    """Vectorized counterpart of _scaled_explanation_parts for many rows."""
    xp = array_namespace
    if normalize == "none":
        scale = xp.ones(attr.shape[0], dtype=float)
        return (
            attr.astype(float, copy=False),
            evidence.astype(float, copy=False),
            positive_score.astype(float, copy=False),
            negative_score.astype(float, copy=False),
            scale,
        )
    if normalize != "l1":
        raise ValueError(f"unknown normalize={normalize!r}; expected 'none' or 'l1'")

    scale = xp.sum(xp.abs(attr), axis=1).astype(float, copy=False)
    ok = (scale > eps) & xp.isfinite(scale)
    out_attr = attr.astype(float, copy=False)
    out_ev = xp.zeros_like(evidence, dtype=float)
    out_pos = xp.zeros_like(positive_score, dtype=float)
    out_neg = xp.zeros_like(negative_score, dtype=float)
    if _truth(xp.any(ok)):
        out_attr[ok] /= scale[ok, None]
        out_ev[ok] = evidence[ok] / scale[ok]
        out_pos[ok] = positive_score[ok] / scale[ok]
        out_neg[ok] = negative_score[ok] / scale[ok]
    if _truth(xp.any(~ok)):
        out_attr[~ok] = 0.0
    scale = xp.where(ok, scale, 0.0)
    return out_attr, out_ev, out_pos, out_neg, scale


def _scale_attribution_array_batch(
    attr: np.ndarray,
    *,
    normalize: NormalizeMode,
    eps: float,
    array_namespace: Any = np,
) -> np.ndarray:
    """Scale only attribution rows, skipping evidence/score outputs."""
    xp = array_namespace
    if normalize == "none":
        return attr.astype(float, copy=False)
    if normalize != "l1":
        raise ValueError(f"unknown normalize={normalize!r}; expected 'none' or 'l1'")

    scale = xp.sum(xp.abs(attr), axis=1).astype(float, copy=False)
    ok = (scale > eps) & xp.isfinite(scale)
    out = attr.astype(float, copy=False)
    if _truth(xp.any(ok)):
        out[ok] /= scale[ok, None]
    if _truth(xp.any(~ok)):
        out[~ok] = 0.0
    return out


def _hyb_components_for_prototype_indices_batch(
    X: np.ndarray,
    prototypes: np.ndarray,
    pos_idx: np.ndarray,
    neg_idx: np.ndarray,
    *,
    anchor: np.ndarray,
    normalize: NormalizeMode,
    eps: float,
    include_scores: bool = True,
    component_mode: Literal["both", "diff", "cos"] = "both",
    array_namespace: Any = np,
) -> Dict[str, np.ndarray]:
    """Precompute normalized Diff/Cos components for many selected prototype pairs."""
    xp = array_namespace
    X = xp.asarray(X, dtype=float)
    prototypes = xp.asarray(prototypes, dtype=float)
    anchor = xp.asarray(anchor, dtype=float)
    if component_mode not in ("both", "diff", "cos"):
        raise ValueError("component_mode must be 'both', 'diff', or 'cos'")
    pos_idx = xp.asarray(pos_idx, dtype=np.intp)
    neg_idx = xp.asarray(neg_idx, dtype=np.intp)
    if pos_idx.shape != (X.shape[0],) or neg_idx.shape != (X.shape[0],):
        raise ValueError("prototype index arrays must have shape (n_samples,)")

    p_pos = prototypes[pos_idx]
    p_neg = prototypes[neg_idx]

    out: Dict[str, np.ndarray] = {"pos_idx": pos_idx, "neg_idx": neg_idx}

    if component_mode in ("both", "diff"):
        pos_sqdist = X - p_pos
        neg_sqdist = X - p_neg
        xp.square(pos_sqdist, out=pos_sqdist)
        xp.square(neg_sqdist, out=neg_sqdist)
        diff_attr_raw = neg_sqdist
        diff_attr_raw -= pos_sqdist
        if include_scores:
            diff_pos_raw = -xp.sum(pos_sqdist, axis=1)
            diff_neg_raw = -xp.sum(neg_sqdist, axis=1)
            diff_ev_raw = xp.sum(diff_attr_raw, axis=1)
            diff_attr, diff_ev, diff_pos, diff_neg, diff_scale = _scale_attribution_batch(
                diff_attr_raw,
                diff_ev_raw,
                diff_pos_raw,
                diff_neg_raw,
                normalize=normalize,
                eps=eps,
                array_namespace=xp,
            )
            out.update(
                {
                    "diff_attr": diff_attr,
                    "diff_attr_sum": diff_ev if normalize == "none" else xp.sum(diff_attr, axis=1),
                    "diff_ev": diff_ev,
                    "diff_pos": diff_pos,
                    "diff_neg": diff_neg,
                    "diff_scale": diff_scale,
                    "diff_ev_raw": diff_ev_raw,
                }
            )
        else:
            out["diff_attr"] = _scale_attribution_array_batch(
                diff_attr_raw,
                normalize=normalize,
                eps=eps,
                array_namespace=xp,
            )

    if component_mode in ("both", "cos"):
        z = X - anchor[None, :]
        q_pos = p_pos - anchor[None, :]
        q_neg = p_neg - anchor[None, :]

        n_z = xp.sqrt(xp.sum(z * z, axis=1) + eps * eps)
        n_pos = xp.sqrt(xp.sum(q_pos * q_pos, axis=1) + eps * eps)
        n_neg = xp.sqrt(xp.sum(q_neg * q_neg, axis=1) + eps * eps)

        pos_by_feature = z * q_pos
        pos_by_feature /= (n_z * n_pos)[:, None]
        neg_by_feature = z * q_neg
        neg_by_feature /= (n_z * n_neg)[:, None]
        if include_scores:
            cos_pos_raw = xp.sum(pos_by_feature, axis=1)
            cos_neg_raw = xp.sum(neg_by_feature, axis=1)
        cos_attr_raw = pos_by_feature
        cos_attr_raw -= neg_by_feature
        if include_scores:
            cos_ev_raw = xp.sum(cos_attr_raw, axis=1)
            cos_attr, cos_ev, cos_pos, cos_neg, cos_scale = _scale_attribution_batch(
                cos_attr_raw,
                cos_ev_raw,
                cos_pos_raw,
                cos_neg_raw,
                normalize=normalize,
                eps=eps,
                array_namespace=xp,
            )
            out.update(
                {
                    "cos_attr": cos_attr,
                    "cos_attr_sum": cos_ev if normalize == "none" else xp.sum(cos_attr, axis=1),
                    "cos_ev": cos_ev,
                    "cos_pos": cos_pos,
                    "cos_neg": cos_neg,
                    "cos_scale": cos_scale,
                    "cos_ev_raw": cos_ev_raw,
                }
            )
        else:
            out["cos_attr"] = _scale_attribution_array_batch(
                cos_attr_raw,
                normalize=normalize,
                eps=eps,
                array_namespace=xp,
            )
    return as_numpy_dict(out, xp)


def _batched_perturbation_scores(
    model: Any,
    X: np.ndarray,
    attributions: np.ndarray,
    target_label_indices: np.ndarray,
    baseline: np.ndarray,
    frac_arr: np.ndarray,
    *,
    max_working_bytes: int = 256 * 1024 * 1024,
    base_prob: Optional[np.ndarray] = None,
    baseline_prob: Optional[np.ndarray] = None,
    array_namespace: Any = np,
) -> Dict[str, np.ndarray]:
    """Compute deletion/insertion metrics for many explanations with two model calls.

    The public perturbation_curves API is kept unchanged.  This private helper is
    used by validation selection to batch all validation samples for a lambda
    candidate and avoid one predict_proba call per sample and per curve.
    """
    xp = array_namespace
    n, d = X.shape
    m = int(frac_arr.shape[0])
    if attributions.shape != X.shape:
        raise ValueError("attributions must have the same shape as X")
    if baseline.shape[0] != d:
        raise ValueError("baseline dimension differs from X feature dimension")
    if target_label_indices.shape[0] != n:
        raise ValueError("target label index length differs from X rows")

    ks, ks_inverse = _rounded_feature_counts(frac_arr, d)
    m_work = int(ks.shape[0])
    inner_mask = (ks > 0) & (ks < d)
    inner_ks = ks[inner_mask]
    inner_positions = np.flatnonzero(inner_mask)
    m_inner = int(inner_ks.shape[0])

    bytes_per_row = 0
    if m_inner > 0:
        bytes_per_row = 2 * m_inner * d * np.dtype(float).itemsize
        bytes_per_row += _topk_mask_working_bytes_per_row(d, inner_ks)
    chunk_plan = _plan_single_perturbation_chunks(
        n_samples=n,
        bytes_per_row=bytes_per_row,
        max_working_bytes=max_working_bytes,
    )
    if chunk_plan.axis == "sample":
        chunk_size = chunk_plan.chunk_size
        chunks = [
            _batched_perturbation_scores(
                model,
                X[start : start + chunk_size],
                attributions[start : start + chunk_size],
                target_label_indices[start : start + chunk_size],
                baseline,
                frac_arr,
                max_working_bytes=max_working_bytes,
                base_prob=None if base_prob is None else base_prob[start : start + chunk_size],
                baseline_prob=None if baseline_prob is None else baseline_prob[start : start + chunk_size],
                array_namespace=xp,
            )
            for start in range(0, n, chunk_size)
        ]
        return {name: np.concatenate([chunk[name] for chunk in chunks], axis=0) for name in chunks[0]}

    row_idx = np.arange(n)
    if base_prob is None:
        base_proba = _predict_proba_matrix(model, X)
        if int(np.max(target_label_indices)) >= base_proba.shape[1]:
            raise ValueError("target label index exceeds predict_proba width")
        base_prob_arr = base_proba[row_idx, target_label_indices].astype(float, copy=False)
    else:
        base_prob_arr = np.asarray(base_prob, dtype=float)
        if base_prob_arr.shape != (n,):
            raise ValueError("base_prob must have shape (n_samples,)")
        if not np.all(np.isfinite(base_prob_arr)):
            raise ValueError("base_prob contains NaN or inf")

    if baseline_prob is None:
        baseline_proba = _predict_proba_matrix(model, baseline.reshape(1, -1))
        if int(np.max(target_label_indices)) >= baseline_proba.shape[1]:
            raise ValueError("target label index exceeds predict_proba width")
        baseline_prob_arr = baseline_proba[0, target_label_indices].astype(float, copy=False)
    else:
        baseline_prob_arr = np.asarray(baseline_prob, dtype=float)
        if baseline_prob_arr.shape != (n,):
            raise ValueError("baseline_prob must have shape (n_samples,)")
        if not np.all(np.isfinite(baseline_prob_arr)):
            raise ValueError("baseline_prob contains NaN or inf")

    deletion_prob_unique = np.empty((n, m_work), dtype=float)
    insertion_prob_unique = np.empty((n, m_work), dtype=float)
    for pos, k in enumerate(ks):
        if int(k) == 0:
            deletion_prob_unique[:, pos] = base_prob_arr
            insertion_prob_unique[:, pos] = baseline_prob_arr
        elif int(k) == d:
            deletion_prob_unique[:, pos] = baseline_prob_arr
            insertion_prob_unique[:, pos] = base_prob_arr

    if m_inner > 0:
        X_gpu = xp.asarray(X, dtype=float)
        baseline_gpu = xp.asarray(baseline, dtype=float)
        mask = _topk_masks_for_counts(attributions, inner_ks, array_namespace=xp)

        all_perturbed = xp.empty((2 * n * m_inner, d), dtype=float)
        deletion = all_perturbed[: n * m_inner].reshape(n, m_inner, d)
        insertion = all_perturbed[n * m_inner :].reshape(n, m_inner, d)
        deletion[...] = X_gpu[:, None, :]
        xp.copyto(deletion, baseline_gpu[None, None, :], where=mask)
        insertion[...] = baseline_gpu[None, None, :]
        xp.copyto(insertion, X_gpu[:, None, :], where=mask)

        all_proba = _predict_proba_matrix(model, as_numpy_array(all_perturbed, xp))
        if int(np.max(target_label_indices)) >= all_proba.shape[1]:
            raise ValueError("target label index exceeds predict_proba width")

        deletion_full = all_proba[: n * m_inner].reshape(n, m_inner, all_proba.shape[1])
        insertion_full = all_proba[n * m_inner :].reshape(n, m_inner, all_proba.shape[1])
        col = target_label_indices[:, None, None]
        deletion_prob_unique[:, inner_positions] = np.take_along_axis(deletion_full, col, axis=2).squeeze(2)
        insertion_prob_unique[:, inner_positions] = np.take_along_axis(insertion_full, col, axis=2).squeeze(2)

    deletion_prob = deletion_prob_unique[:, ks_inverse]
    insertion_prob = insertion_prob_unique[:, ks_inverse]

    try:
        deletion_auc = np.trapezoid(deletion_prob, frac_arr, axis=1)
        insertion_auc = np.trapezoid(insertion_prob, frac_arr, axis=1)
    except AttributeError:  # NumPy < 2.0
        deletion_auc = np.trapz(deletion_prob, frac_arr, axis=1)
        insertion_auc = np.trapz(insertion_prob, frac_arr, axis=1)

    p0 = deletion_prob[:, 0]
    deletion_drop_auc = p0 - deletion_auc
    combined_score = 0.5 * (deletion_drop_auc + insertion_auc)
    return {
        "p0": p0.astype(float, copy=False),
        "deletion_auc": deletion_auc.astype(float, copy=False),
        "deletion_drop_auc": deletion_drop_auc.astype(float, copy=False),
        "insertion_auc": insertion_auc.astype(float, copy=False),
        "combined_score": combined_score.astype(float, copy=False),
    }


def _batched_perturbation_scores_for_lambda_grid(
    model: Any,
    X: np.ndarray,
    diff_attr: np.ndarray,
    cos_attr: np.ndarray,
    lambdas: np.ndarray,
    target_label_indices: np.ndarray,
    baseline: np.ndarray,
    frac_arr: np.ndarray,
    *,
    max_working_bytes: int = 256 * 1024 * 1024,
    base_prob: Optional[np.ndarray] = None,
    baseline_prob: Optional[np.ndarray] = None,
    array_namespace: Any = np,
) -> Dict[str, np.ndarray]:
    """Compute perturbation metrics for all lambda candidates in one model call."""
    xp = array_namespace
    n, d = X.shape
    l_count = int(lambdas.shape[0])
    ks, ks_inverse = _rounded_feature_counts(frac_arr, d)
    m_work = int(ks.shape[0])
    inner_mask = (ks > 0) & (ks < d)
    inner_ks = ks[inner_mask]
    inner_positions = np.flatnonzero(inner_mask)
    m_inner = int(inner_ks.shape[0])
    if diff_attr.shape != X.shape or cos_attr.shape != X.shape:
        raise ValueError("diff_attr and cos_attr must have the same shape as X")
    if target_label_indices.shape[0] != n:
        raise ValueError("target label index length differs from X rows")

    n_perturbed = 2 * l_count * n * m_inner
    bytes_per_lambda = 0
    bytes_per_sample_all_lambdas = 0
    if m_inner > 0:
        bytes_per_lambda = 2 * n * m_inner * d * np.dtype(float).itemsize
        bytes_per_lambda += n * d * np.dtype(float).itemsize
        bytes_per_lambda += n * _topk_mask_working_bytes_per_row(d, inner_ks)
        bytes_per_sample_all_lambdas = 2 * l_count * m_inner * d * np.dtype(float).itemsize
        bytes_per_sample_all_lambdas += l_count * d * np.dtype(float).itemsize
        bytes_per_sample_all_lambdas += l_count * _topk_mask_working_bytes_per_row(d, inner_ks)
    chunk_plan = _plan_lambda_grid_perturbation_chunks(
        n_samples=n,
        n_lambdas=l_count,
        bytes_per_lambda=bytes_per_lambda,
        bytes_per_sample_all_lambdas=bytes_per_sample_all_lambdas,
        max_working_bytes=max_working_bytes,
    )
    if chunk_plan.axis != "none":
        if chunk_plan.axis == "sample":
            sample_chunk_size = chunk_plan.chunk_size
            chunks = [
                _batched_perturbation_scores_for_lambda_grid(
                    model,
                    X[start : start + sample_chunk_size],
                    diff_attr[start : start + sample_chunk_size],
                    cos_attr[start : start + sample_chunk_size],
                    lambdas,
                    target_label_indices[start : start + sample_chunk_size],
                    baseline,
                    frac_arr,
                    max_working_bytes=max_working_bytes,
                    base_prob=None if base_prob is None else base_prob[start : start + sample_chunk_size],
                    baseline_prob=None if baseline_prob is None else baseline_prob[start : start + sample_chunk_size],
                    array_namespace=xp,
                )
                for start in range(0, n, sample_chunk_size)
            ]
            return {name: np.concatenate([chunk[name] for chunk in chunks], axis=1) for name in chunks[0]}

        lambda_chunk_size = chunk_plan.chunk_size
        chunks = [
            _batched_perturbation_scores_for_lambda_grid(
                model,
                X,
                diff_attr,
                cos_attr,
                lambdas[start : start + lambda_chunk_size],
                target_label_indices,
                baseline,
                frac_arr,
                max_working_bytes=max_working_bytes,
                base_prob=base_prob,
                baseline_prob=baseline_prob,
                array_namespace=xp,
            )
            for start in range(0, l_count, lambda_chunk_size)
        ]
        return {name: np.concatenate([chunk[name] for chunk in chunks], axis=0) for name in chunks[0]}

    row_idx = np.arange(n)
    if base_prob is None:
        base_proba = _predict_proba_matrix(model, X)
        if int(np.max(target_label_indices)) >= base_proba.shape[1]:
            raise ValueError("target label index exceeds predict_proba width")
        base_prob_arr = base_proba[row_idx, target_label_indices].astype(float, copy=False)
    else:
        base_prob_arr = np.asarray(base_prob, dtype=float)
        if base_prob_arr.shape != (n,):
            raise ValueError("base_prob must have shape (n_samples,)")
        if not np.all(np.isfinite(base_prob_arr)):
            raise ValueError("base_prob contains NaN or inf")

    if baseline_prob is None:
        baseline_proba = _predict_proba_matrix(model, baseline.reshape(1, -1))
        if int(np.max(target_label_indices)) >= baseline_proba.shape[1]:
            raise ValueError("target label index exceeds predict_proba width")
        baseline_prob_arr = baseline_proba[0, target_label_indices].astype(float, copy=False)
    else:
        baseline_prob_arr = np.asarray(baseline_prob, dtype=float)
        if baseline_prob_arr.shape != (n,):
            raise ValueError("baseline_prob must have shape (n_samples,)")
        if not np.all(np.isfinite(baseline_prob_arr)):
            raise ValueError("baseline_prob contains NaN or inf")

    deletion_prob_unique = np.empty((l_count, n, m_work), dtype=float)
    insertion_prob_unique = np.empty((l_count, n, m_work), dtype=float)
    for pos, k in enumerate(ks):
        if int(k) == 0:
            deletion_prob_unique[:, :, pos] = base_prob_arr[None, :]
            insertion_prob_unique[:, :, pos] = baseline_prob_arr[None, :]
        elif int(k) == d:
            deletion_prob_unique[:, :, pos] = baseline_prob_arr[None, :]
            insertion_prob_unique[:, :, pos] = base_prob_arr[None, :]

    if m_inner > 0:
        X_gpu = xp.asarray(X, dtype=float)
        baseline_gpu = xp.asarray(baseline, dtype=float)
        diff_attr_gpu = xp.asarray(diff_attr, dtype=float)
        cos_attr_gpu = xp.asarray(cos_attr, dtype=float)
        lam = xp.asarray(lambdas, dtype=float)[:, None, None]
        attr = lam * diff_attr_gpu[None, :, :] + (1.0 - lam) * cos_attr_gpu[None, :, :]
        mask = _topk_masks_for_counts(attr.reshape(l_count * n, d), inner_ks, array_namespace=xp).reshape(
            l_count,
            n,
            m_inner,
            d,
        )

        all_perturbed = xp.empty((n_perturbed, d), dtype=float)
        deletion = all_perturbed[: l_count * n * m_inner].reshape(l_count, n, m_inner, d)
        insertion = all_perturbed[l_count * n * m_inner :].reshape(l_count, n, m_inner, d)
        deletion[...] = X_gpu[None, :, None, :]
        xp.copyto(deletion, baseline_gpu[None, None, None, :], where=mask)
        insertion[...] = baseline_gpu[None, None, None, :]
        xp.copyto(insertion, X_gpu[None, :, None, :], where=mask)

        all_proba = _predict_proba_matrix(model, as_numpy_array(all_perturbed, xp))
        if int(np.max(target_label_indices)) >= all_proba.shape[1]:
            raise ValueError("target label index exceeds predict_proba width")

        deletion_full = all_proba[: l_count * n * m_inner].reshape(l_count, n, m_inner, all_proba.shape[1])
        insertion_full = all_proba[l_count * n * m_inner :].reshape(l_count, n, m_inner, all_proba.shape[1])
        col = target_label_indices[None, :, None, None]
        deletion_prob_unique[:, :, inner_positions] = np.take_along_axis(deletion_full, col, axis=3).squeeze(3)
        insertion_prob_unique[:, :, inner_positions] = np.take_along_axis(insertion_full, col, axis=3).squeeze(3)

    deletion_prob = deletion_prob_unique[:, :, ks_inverse]
    insertion_prob = insertion_prob_unique[:, :, ks_inverse]

    try:
        deletion_auc = np.trapezoid(deletion_prob, frac_arr, axis=2)
        insertion_auc = np.trapezoid(insertion_prob, frac_arr, axis=2)
    except AttributeError:  # NumPy < 2.0
        deletion_auc = np.trapz(deletion_prob, frac_arr, axis=2)
        insertion_auc = np.trapz(insertion_prob, frac_arr, axis=2)

    p0 = deletion_prob[:, :, 0]
    deletion_drop_auc = p0 - deletion_auc
    combined_score = 0.5 * (deletion_drop_auc + insertion_auc)
    return {
        "p0": p0.astype(float, copy=False),
        "deletion_auc": deletion_auc.astype(float, copy=False),
        "deletion_drop_auc": deletion_drop_auc.astype(float, copy=False),
        "insertion_auc": insertion_auc.astype(float, copy=False),
        "combined_score": combined_score.astype(float, copy=False),
    }

def _hyb_metric_components_for_prototype_indices_batch(
    X: np.ndarray,
    prototypes: np.ndarray,
    pos_idx: np.ndarray,
    neg_idx: np.ndarray,
    *,
    anchor: np.ndarray,
    eps: float,
    include_l1: bool = True,
    array_namespace: Any = np,
) -> Dict[str, np.ndarray]:
    """Precompute only scalar batch components needed for grid-search scoring."""
    xp = array_namespace
    X = xp.asarray(X, dtype=float)
    prototypes = xp.asarray(prototypes, dtype=float)
    anchor = xp.asarray(anchor, dtype=float)
    pos_idx = xp.asarray(pos_idx, dtype=np.intp)
    neg_idx = xp.asarray(neg_idx, dtype=np.intp)
    if pos_idx.shape != (X.shape[0],) or neg_idx.shape != (X.shape[0],):
        raise ValueError("prototype index arrays must have shape (n_samples,)")

    p_pos = prototypes[pos_idx]
    p_neg = prototypes[neg_idx]

    pos_delta = X - p_pos
    neg_delta = X - p_neg
    xp.square(pos_delta, out=pos_delta)
    xp.square(neg_delta, out=neg_delta)
    pos_sqdist = xp.sum(pos_delta, axis=1)
    neg_sqdist = xp.sum(neg_delta, axis=1)
    diff_ev_raw = neg_sqdist - pos_sqdist

    z = X - anchor[None, :]
    q_pos = p_pos - anchor[None, :]
    q_neg = p_neg - anchor[None, :]
    n_z = xp.sqrt(xp.sum(z * z, axis=1) + eps * eps)
    n_pos = xp.sqrt(xp.sum(q_pos * q_pos, axis=1) + eps * eps)
    n_neg = xp.sqrt(xp.sum(q_neg * q_neg, axis=1) + eps * eps)

    if include_l1:
        pos_by_feature = z * q_pos
        pos_by_feature /= (n_z * n_pos)[:, None]
        neg_by_feature = z * q_neg
        neg_by_feature /= (n_z * n_neg)[:, None]
        cos_pos_raw = xp.sum(pos_by_feature, axis=1)
        cos_neg_raw = xp.sum(neg_by_feature, axis=1)
    else:
        cos_pos_raw = xp.einsum("ij,ij->i", z, q_pos) / (n_z * n_pos)
        cos_neg_raw = xp.einsum("ij,ij->i", z, q_neg) / (n_z * n_neg)
    cos_ev_raw = cos_pos_raw - cos_neg_raw

    out = {
        "diff_ev_raw": diff_ev_raw,
        "cos_ev_raw": cos_ev_raw,
    }
    if include_l1:
        diff_attr_raw = neg_delta
        diff_attr_raw -= pos_delta
        diff_ev_l1_raw = xp.sum(diff_attr_raw, axis=1)
        diff_scale = xp.sum(xp.abs(diff_attr_raw), axis=1)
        diff_ok = (diff_scale > eps) & xp.isfinite(diff_scale)
        diff_ev_l1 = xp.zeros_like(diff_ev_raw, dtype=float)
        diff_ev_l1[diff_ok] = diff_ev_l1_raw[diff_ok] / diff_scale[diff_ok]

        cos_attr_raw = pos_by_feature
        cos_attr_raw -= neg_by_feature
        cos_ev_l1_raw = xp.sum(cos_attr_raw, axis=1)
        cos_scale = xp.sum(xp.abs(cos_attr_raw), axis=1)
        cos_ok = (cos_scale > eps) & xp.isfinite(cos_scale)
        cos_ev_l1 = xp.zeros_like(cos_ev_raw, dtype=float)
        cos_ev_l1[cos_ok] = cos_ev_l1_raw[cos_ok] / cos_scale[cos_ok]
        out["diff_ev_l1"] = diff_ev_l1
        out["cos_ev_l1"] = cos_ev_l1
    return as_numpy_dict(out, xp)


__all__: list[str] = []
