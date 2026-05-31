"""Public explainer functions for FPDE."""

from __future__ import annotations

from typing import Any, Optional, Sequence

import numpy as np

from ._array import (
    _as_1d_float,
    _as_2d_float,
    _as_label_array,
    _check_same_dim,
    _regularized_norm,
    _scaled_explanation_parts,
)
from .metrics import regularized_cosine
from .prototypes import select_prototype_pair
from .types import (
    FPDEExplanation,
    GridMode,
    Mode,
    NormalizeMode,
)

def diff_fpde(
    x: np.ndarray | Sequence[float],
    p_pos: np.ndarray | Sequence[float],
    p_neg: np.ndarray | Sequence[float],
    *,
    positive_label: Any = "positive",
    negative_label: Any = "negative",
    positive_prototype_index: int = -1,
    negative_prototype_index: int = -1,
) -> FPDEExplanation:
    """Strict Diff-FPDE for one positive/negative prototype pair.

    Evidence:

        E_diff = ||x - p_neg||^2 - ||x - p_pos||^2

    Feature attribution:

        phi_j = (x_j - p_neg_j)^2 - (x_j - p_pos_j)^2
    """
    x_arr = _as_1d_float("x", x)
    p_pos_arr = _as_1d_float("p_pos", p_pos)
    p_neg_arr = _as_1d_float("p_neg", p_neg)
    _check_same_dim(x_arr, p_pos_arr, p_neg_arr)

    pos_sqdist_by_feature = (x_arr - p_pos_arr) ** 2
    neg_sqdist_by_feature = (x_arr - p_neg_arr) ** 2

    attributions = neg_sqdist_by_feature - pos_sqdist_by_feature

    pos_score = -float(np.sum(pos_sqdist_by_feature))
    neg_score = -float(np.sum(neg_sqdist_by_feature))
    evidence = float(np.sum(attributions))

    direct_evidence = float(np.sum(neg_sqdist_by_feature) - np.sum(pos_sqdist_by_feature))
    residual = float(evidence - direct_evidence)

    return FPDEExplanation(
        mode="diff",
        evidence=evidence,
        attributions=attributions,
        positive_score=pos_score,
        negative_score=neg_score,
        positive_label=positive_label,
        negative_label=negative_label,
        positive_prototype_index=int(positive_prototype_index),
        negative_prototype_index=int(negative_prototype_index),
        exactness_residual=residual,
        details={
            "positive_squared_distance": float(np.sum(pos_sqdist_by_feature)),
            "negative_squared_distance": float(np.sum(neg_sqdist_by_feature)),
            "definition": "E_diff = ||x-p_neg||^2 - ||x-p_pos||^2",
        },
    )


def cos_fpde(
    x: np.ndarray | Sequence[float],
    p_pos: np.ndarray | Sequence[float],
    p_neg: np.ndarray | Sequence[float],
    *,
    anchor: Optional[np.ndarray | Sequence[float]] = None,
    eps: float = 1e-12,
    positive_label: Any = "positive",
    negative_label: Any = "negative",
    positive_prototype_index: int = -1,
    negative_prototype_index: int = -1,
) -> FPDEExplanation:
    """Strict Cos-FPDE for one positive/negative prototype pair.

    Let z = x - anchor, q_pos = p_pos - anchor, q_neg = p_neg - anchor.

    Evidence:

        E_cos = cos_eps(z, q_pos) - cos_eps(z, q_neg)

    Feature attribution:

        phi_j = z_j q_pos_j / (N_z N_pos) - z_j q_neg_j / (N_z N_neg)

    This is an exact coordinate decomposition of the cosine contrast itself.
    It is not a leave-one-feature-out causal effect because the cosine norm
    denominator couples all coordinates.
    """
    if eps <= 0.0:
        raise ValueError("eps must be positive")

    x_arr = _as_1d_float("x", x)
    p_pos_arr = _as_1d_float("p_pos", p_pos)
    p_neg_arr = _as_1d_float("p_neg", p_neg)
    _check_same_dim(x_arr, p_pos_arr, p_neg_arr)

    if anchor is None:
        anchor_arr = np.zeros_like(x_arr, dtype=float)
    else:
        anchor_arr = _as_1d_float("anchor", anchor)
        _check_same_dim(x_arr, anchor_arr)

    z = x_arr - anchor_arr
    q_pos = p_pos_arr - anchor_arr
    q_neg = p_neg_arr - anchor_arr

    n_z = _regularized_norm(z, eps)
    n_pos = _regularized_norm(q_pos, eps)
    n_neg = _regularized_norm(q_neg, eps)

    pos_by_feature = (z * q_pos) / (n_z * n_pos)
    neg_by_feature = (z * q_neg) / (n_z * n_neg)

    attributions = pos_by_feature - neg_by_feature

    pos_score = float(np.sum(pos_by_feature))
    neg_score = float(np.sum(neg_by_feature))
    evidence = float(np.sum(attributions))

    direct_evidence = regularized_cosine(z, q_pos, eps=eps) - regularized_cosine(z, q_neg, eps=eps)
    residual = float(evidence - direct_evidence)

    return FPDEExplanation(
        mode="cos",
        evidence=evidence,
        attributions=attributions,
        positive_score=pos_score,
        negative_score=neg_score,
        positive_label=positive_label,
        negative_label=negative_label,
        positive_prototype_index=int(positive_prototype_index),
        negative_prototype_index=int(negative_prototype_index),
        exactness_residual=residual,
        details={
            "anchor": anchor_arr.copy(),
            "eps": float(eps),
            "norm_x_anchor": float(n_z),
            "norm_positive_anchor": float(n_pos),
            "norm_negative_anchor": float(n_neg),
            "definition": "E_cos = cos_eps(x-a,p_pos-a) - cos_eps(x-a,p_neg-a)",
        },
    )


def _hyb_fpde_grid_candidate(
    x: np.ndarray | Sequence[float],
    p_pos: np.ndarray | Sequence[float],
    p_neg: np.ndarray | Sequence[float],
    *,
    anchor: Optional[np.ndarray | Sequence[float]] = None,
    lambda_hyb: float = 0.5,
    normalize: NormalizeMode = "l1",
    eps: float = 1e-12,
    positive_label: Any = "positive",
    negative_label: Any = "negative",
    positive_prototype_index: int = -1,
    negative_prototype_index: int = -1,
) -> FPDEExplanation:
    """Internal Hyb-FPDE candidate for one positive/negative prototype pair.

    Standalone fixed-lambda Hyb-FPDE is intentionally not exposed as a public
    method. This helper exists only so Hyb-FPDE can evaluate each fixed
    lambda_hyb candidate.

    The grid candidate combines Diff-FPDE and Cos-FPDE at the attribution level:

        phi_grid_j = lambda_hyb * phi_diff_j' + (1 - lambda_hyb) * phi_cos_j'

    If normalize="l1", each component attribution vector is divided by its L1
    norm before mixing. If normalize="none", raw component attributions are
    mixed directly.
    """
    if eps <= 0.0:
        raise ValueError("eps must be positive")
    if not np.isfinite(lambda_hyb):
        raise ValueError("lambda_hyb must be finite")
    if lambda_hyb < 0.0 or lambda_hyb > 1.0:
        raise ValueError("lambda_hyb must be in [0, 1]")
    if normalize not in ("none", "l1"):
        raise ValueError("normalize must be either 'none' or 'l1'")

    diff_exp = diff_fpde(
        x,
        p_pos,
        p_neg,
        positive_label=positive_label,
        negative_label=negative_label,
        positive_prototype_index=positive_prototype_index,
        negative_prototype_index=negative_prototype_index,
    )
    cos_exp = cos_fpde(
        x,
        p_pos,
        p_neg,
        anchor=anchor,
        eps=eps,
        positive_label=positive_label,
        negative_label=negative_label,
        positive_prototype_index=positive_prototype_index,
        negative_prototype_index=negative_prototype_index,
    )

    diff_attr, diff_ev, diff_pos, diff_neg, diff_scale = _scaled_explanation_parts(
        diff_exp,
        normalize=normalize,
        eps=eps,
    )
    cos_attr, cos_ev, cos_pos, cos_neg, cos_scale = _scaled_explanation_parts(
        cos_exp,
        normalize=normalize,
        eps=eps,
    )

    w_diff = float(lambda_hyb)
    w_cos = float(1.0 - lambda_hyb)

    attributions = w_diff * diff_attr + w_cos * cos_attr
    evidence = float(np.sum(attributions))

    positive_score = float(w_diff * diff_pos + w_cos * cos_pos)
    negative_score = float(w_diff * diff_neg + w_cos * cos_neg)
    direct_grid_evidence = float(w_diff * diff_ev + w_cos * cos_ev)
    residual = float(evidence - direct_grid_evidence)

    return FPDEExplanation(
        mode="hyb_grid",
        evidence=evidence,
        attributions=attributions,
        positive_score=positive_score,
        negative_score=negative_score,
        positive_label=positive_label,
        negative_label=negative_label,
        positive_prototype_index=int(positive_prototype_index),
        negative_prototype_index=int(negative_prototype_index),
        exactness_residual=residual,
        details={
            "method_family": "hyb_fpde_grid",
            "lambda_hyb": float(lambda_hyb),
            "normalize": normalize,
            "eps": float(eps),
            "definition": "phi_grid_j = lambda_hyb * phi_diff_j' + (1-lambda_hyb) * phi_cos_j'",
            "diff_evidence_raw": float(diff_exp.evidence),
            "cos_evidence_raw": float(cos_exp.evidence),
            "diff_evidence_scaled": float(diff_ev),
            "cos_evidence_scaled": float(cos_ev),
            "diff_scale": float(diff_scale),
            "cos_scale": float(cos_scale),
            "diff_exactness_residual": float(diff_exp.exactness_residual),
            "cos_exactness_residual": float(cos_exp.exactness_residual),
            "selection_note": "Prototype indices are supplied by the caller; Hyb-FPDE uses diff selection for lambda candidates by default.",
        },
    )


def explain_with_selected_prototypes(
    x: np.ndarray | Sequence[float],
    prototypes: np.ndarray | Sequence[Sequence[float]],
    prototype_labels: Sequence[Any],
    *,
    positive_label: Any,
    negative_label: Optional[Any] = None,
    mode: Mode = "diff",
    anchor: Optional[np.ndarray | Sequence[float]] = None,
    eps: float = 1e-12,
) -> FPDEExplanation:
    """Select prototypes and compute a public Diff-FPDE or Cos-FPDE explanation."""
    if mode not in ("diff", "cos"):
        raise ValueError("mode must be either 'diff' or 'cos'; use FPDEEngine for Hyb-FPDE")
    return _explain_with_selected_prototypes_for_grid(
        x,
        prototypes,
        prototype_labels,
        positive_label=positive_label,
        negative_label=negative_label,
        mode=mode,
        anchor=anchor,
        lambda_hyb=0.5,
        normalize="l1",
        eps=eps,
    )


def _explain_with_selected_prototypes_for_grid(
    x: np.ndarray | Sequence[float],
    prototypes: np.ndarray | Sequence[Sequence[float]],
    prototype_labels: Sequence[Any],
    *,
    positive_label: Any,
    negative_label: Optional[Any] = None,
    mode: GridMode = "diff",
    anchor: Optional[np.ndarray | Sequence[float]] = None,
    lambda_hyb: float = 0.5,
    normalize: NormalizeMode = "l1",
    eps: float = 1e-12,
) -> FPDEExplanation:
    """Internal dispatcher for Diff, Cos, and Hyb-FPDE lambda candidates."""
    P = _as_2d_float("prototypes", prototypes)
    labels = _as_label_array(prototype_labels)

    pos_idx, neg_idx = select_prototype_pair(
        x,
        P,
        labels,
        positive_label=positive_label,
        negative_label=negative_label,
        mode=mode,
        anchor=anchor,
        eps=eps,
    )

    neg_label_value = labels[neg_idx]

    if mode == "diff":
        return diff_fpde(
            x,
            P[pos_idx],
            P[neg_idx],
            positive_label=positive_label,
            negative_label=neg_label_value,
            positive_prototype_index=pos_idx,
            negative_prototype_index=neg_idx,
        )

    if mode == "cos":
        return cos_fpde(
            x,
            P[pos_idx],
            P[neg_idx],
            anchor=anchor,
            eps=eps,
            positive_label=positive_label,
            negative_label=neg_label_value,
            positive_prototype_index=pos_idx,
            negative_prototype_index=neg_idx,
        )

    if mode == "hyb_grid":
        return _hyb_fpde_grid_candidate(
            x,
            P[pos_idx],
            P[neg_idx],
            anchor=anchor,
            lambda_hyb=lambda_hyb,
            normalize=normalize,
            eps=eps,
            positive_label=positive_label,
            negative_label=neg_label_value,
            positive_prototype_index=pos_idx,
            negative_prototype_index=neg_idx,
        )

__all__ = [
    "diff_fpde",
    "cos_fpde",
    "explain_with_selected_prototypes",
]
