"""Grid-search and validation-based Hyb-FPDE selection."""

from __future__ import annotations

from dataclasses import replace
from typing import Any, Dict, List, Optional, Sequence, Tuple

import numpy as np

from ._array import (
    _default_lambda_hyb_grid,
    _get_predict_proba_vector,
    _probability_for_label,
    _top_two_probability_columns,
)
from .prototypes import _nearest_label_from_prototypes
from .types import (
    AnchorStrategy,
    FPDEExplanation,
    GridMode,
    GridObjective,
    NormalizeMode,
)
from .explainers import _explain_with_selected_prototypes_for_grid

def _hyb_fpde_grid_candidate_dicts(
    *,
    fpde_mode_grid: Sequence[GridMode] = ("diff", "cos", "hyb_grid"),
    normalize_grid: Sequence[NormalizeMode] = ("l1",),
    lambda_hyb_grid: Sequence[float] = _default_lambda_hyb_grid(0.1),
    anchor_strategy_grid: Sequence[AnchorStrategy] = ("mean",),
    include_explicit_diff_cos: bool = True,
) -> List[Dict[str, Any]]:
    """Create exhaustive candidate dictionaries for Diff/Cos/Hyb-FPDE search."""
    lambdas: List[float] = []
    for lam in lambda_hyb_grid:
        value = float(lam)
        if not np.isfinite(value) or value < 0.0 or value > 1.0:
            raise ValueError("lambda_hyb_grid values must be finite and in [0, 1]")
        lambdas.append(value)
    if not lambdas:
        raise ValueError("lambda_hyb_grid must contain at least one value")

    requested_modes = tuple(fpde_mode_grid)
    for mode in requested_modes:
        if mode not in ("diff", "cos", "hyb_grid"):
            raise ValueError("fpde_mode_grid values must be 'diff', 'cos', or 'hyb_grid'")

    candidates: List[Dict[str, Any]] = []
    for normalize in normalize_grid:
        if normalize not in ("none", "l1"):
            raise ValueError("normalize_grid values must be 'none' or 'l1'")
        for anchor_strategy in anchor_strategy_grid:
            if anchor_strategy not in ("mean", "zero", "none"):
                raise ValueError("anchor_strategy_grid values must be 'mean', 'zero', or 'none'")
            if include_explicit_diff_cos:
                if "diff" in requested_modes:
                    candidates.append(
                        {
                            "fpde_mode": "diff",
                            "method_variant": "diff_fpde",
                            "normalize": normalize,
                            "lambda_hyb": None,
                            "lambda_grid_mode": "not_applicable",
                            "anchor_strategy": anchor_strategy,
                        }
                    )
                if "cos" in requested_modes:
                    candidates.append(
                        {
                            "fpde_mode": "cos",
                            "method_variant": "cos_fpde",
                            "normalize": normalize,
                            "lambda_hyb": None,
                            "lambda_grid_mode": "not_applicable",
                            "anchor_strategy": anchor_strategy,
                        }
                    )
            if "hyb_grid" in requested_modes:
                for lambda_hyb in lambdas:
                    if lambda_hyb == 0.0:
                        variant = "hyb_fpde_grid_lambda_0_cos_endpoint"
                    elif lambda_hyb == 1.0:
                        variant = "hyb_fpde_grid_lambda_1_diff_endpoint"
                    else:
                        variant = "hyb_fpde_grid"
                    candidates.append(
                        {
                            "fpde_mode": "hyb_grid",
                            "method_variant": variant,
                            "normalize": normalize,
                            "lambda_hyb": float(lambda_hyb),
                            "lambda_grid_mode": "grid_candidate_lambda_hyb",
                            "anchor_strategy": anchor_strategy,
                        }
                    )
    return candidates


def _label_contrast_for_grid_sample(
    x: np.ndarray,
    *,
    prototypes: np.ndarray,
    prototype_labels: np.ndarray,
    classes: np.ndarray,
    predictor: Optional[Any],
) -> Tuple[Any, Optional[Any], Optional[np.ndarray], Optional[np.ndarray]]:
    """Choose positive and rival labels for one grid-search sample."""
    probabilities, probability_labels = _get_predict_proba_vector(predictor, x)
    if probabilities is None:
        return _nearest_label_from_prototypes(x, prototypes, prototype_labels), None, None, None

    labels_for_prob = probability_labels if probability_labels is not None else classes
    if labels_for_prob.shape[0] != probabilities.shape[0]:
        raise ValueError("cannot infer labels: probability label length mismatch")
    if probabilities.shape[0] < 2:
        return labels_for_prob[int(np.argmax(probabilities))], None, probabilities, probability_labels

    prob_row = probabilities.reshape(1, -1)
    pos_cols, fallback_rival_cols = _top_two_probability_columns(prob_row)
    pos_idx = int(pos_cols[0])
    pos = labels_for_prob[pos_idx]
    available = np.isin(labels_for_prob, prototype_labels)
    if bool(np.all(available)):
        neg: Optional[Any] = labels_for_prob[int(fallback_rival_cols[0])]
    else:
        rival_scores = np.where(available, probabilities, -np.inf)
        rival_scores[pos_idx] = -np.inf
        rival_idx = int(np.argmax(rival_scores))
        neg = labels_for_prob[rival_idx] if np.isfinite(rival_scores[rival_idx]) else None
    return pos, neg, probabilities, probability_labels


def _explain_single_prototype_candidate_from_contrast(
    x: np.ndarray,
    *,
    prototypes: np.ndarray,
    prototype_labels: np.ndarray,
    classes: np.ndarray,
    anchor: np.ndarray,
    cfg: Dict[str, Any],
    eps: float,
    pos: Any,
    neg: Optional[Any],
    probabilities: Optional[np.ndarray],
    probability_labels: Optional[np.ndarray],
) -> FPDEExplanation:
    """Explain one sample after its positive/rival labels have been selected."""
    exp = _explain_with_selected_prototypes_for_grid(
        x,
        prototypes,
        prototype_labels,
        positive_label=pos,
        negative_label=neg,
        mode=cfg["fpde_mode"],
        anchor=anchor,
        lambda_hyb=0.5 if cfg.get("lambda_hyb") is None else float(cfg["lambda_hyb"]),
        normalize=cfg["normalize"],
        eps=eps,
    )
    details = dict(exp.details)
    details.update(
        {
            "positive_probability": _probability_for_label(pos, probabilities, probability_labels, classes),
            "negative_probability": _probability_for_label(exp.negative_label, probabilities, probability_labels, classes),
            "probabilities": None if probabilities is None else probabilities.copy(),
            "probability_labels": None if probability_labels is None else probability_labels.copy(),
        }
    )
    return replace(exp, details=details)


def _score_hyb_fpde_explanations(
    explanations: Sequence[FPDEExplanation],
    *,
    objective: GridObjective,
) -> Dict[str, float]:
    """Compute scalar grid-search metrics from Diff/Cos/Hyb-FPDE explanations."""
    if len(explanations) == 0:
        raise ValueError("explanations must be non-empty")

    evidences = np.asarray([float(exp.evidence) for exp in explanations], dtype=float)
    residuals = np.asarray([float(exp.exactness_residual) for exp in explanations], dtype=float)
    positive = np.maximum(evidences, 0.0)
    agreement = np.asarray(evidences > 0.0, dtype=float)

    if objective == "blackbox_agreement":
        score = float(np.mean(agreement))
    elif objective == "mean_positive_evidence":
        score = float(np.mean(positive))
    elif objective == "mean_margin_weighted_evidence":
        weights = []
        for exp in explanations:
            p_pos = exp.details.get("positive_probability", None)
            p_neg = exp.details.get("negative_probability", None)
            if p_pos is None or p_neg is None:
                weights.append(1.0)
            else:
                weights.append(max(0.0, float(p_pos) - float(p_neg)))
        w = np.asarray(weights, dtype=float)
        score = float(np.mean(positive * w))
    else:
        raise ValueError(
            "objective must be 'blackbox_agreement', 'mean_positive_evidence', "
            "or 'mean_margin_weighted_evidence'"
        )

    lambdas = []
    for exp in explanations:
        value = exp.details.get("lambda_hyb", np.nan)
        try:
            lambdas.append(float(value))
        except (TypeError, ValueError):
            lambdas.append(float("nan"))
    lambda_arr = np.asarray(lambdas, dtype=float)
    finite_lambdas = lambda_arr[np.isfinite(lambda_arr)]

    return {
        "score": score,
        "agreement_rate": float(np.mean(agreement)),
        "mean_evidence": float(np.mean(evidences)),
        "median_evidence": float(np.median(evidences)),
        "mean_positive_evidence": float(np.mean(positive)),
        "mean_abs_evidence": float(np.mean(np.abs(evidences))),
        "mean_exactness_abs_residual": float(np.mean(np.abs(residuals))),
        "max_exactness_abs_residual": float(np.max(np.abs(residuals))),
        "mean_lambda_hyb": float(np.mean(finite_lambdas)) if finite_lambdas.size else float("nan"),
        "min_lambda_hyb": float(np.min(finite_lambdas)) if finite_lambdas.size else float("nan"),
        "max_lambda_hyb": float(np.max(finite_lambdas)) if finite_lambdas.size else float("nan"),
    }


def _score_hyb_fpde_arrays(
    *,
    evidences: np.ndarray,
    residuals: Optional[np.ndarray],
    objective: GridObjective,
    lambda_hyb: Optional[float],
    positive_probabilities: Optional[np.ndarray],
    negative_probabilities: Optional[np.ndarray],
) -> Dict[str, float]:
    """Compute grid-search metrics directly from vectorized explanation arrays."""
    if evidences.size == 0:
        raise ValueError("evidences must be non-empty")

    positive = np.maximum(evidences, 0.0)
    agreement_rate = float(np.mean(evidences > 0.0))

    if objective == "blackbox_agreement":
        score = agreement_rate
    elif objective == "mean_positive_evidence":
        score = float(np.mean(positive))
    elif objective == "mean_margin_weighted_evidence":
        if positive_probabilities is None or negative_probabilities is None:
            weights = np.ones_like(evidences, dtype=float)
        else:
            weights = np.maximum(positive_probabilities - negative_probabilities, 0.0)
        score = float(np.mean(positive * weights))
    else:
        raise ValueError(
            "objective must be 'blackbox_agreement', 'mean_positive_evidence', "
            "or 'mean_margin_weighted_evidence'"
        )

    lambda_value = float("nan") if lambda_hyb is None else float(lambda_hyb)
    if residuals is None:
        mean_abs_residual = 0.0
        max_abs_residual = 0.0
    else:
        abs_residuals = np.abs(residuals)
        mean_abs_residual = float(np.mean(abs_residuals))
        max_abs_residual = float(np.max(abs_residuals))
    return {
        "score": score,
        "agreement_rate": agreement_rate,
        "mean_evidence": float(np.mean(evidences)),
        "median_evidence": float(np.median(evidences)),
        "mean_positive_evidence": float(np.mean(positive)),
        "mean_abs_evidence": float(np.mean(np.abs(evidences))),
        "mean_exactness_abs_residual": mean_abs_residual,
        "max_exactness_abs_residual": max_abs_residual,
        "mean_lambda_hyb": lambda_value,
        "min_lambda_hyb": lambda_value,
        "max_lambda_hyb": lambda_value,
    }


def _grid_candidate_metrics_from_components(
    cfg: Dict[str, Any],
    comp: Dict[str, np.ndarray],
    *,
    objective: GridObjective,
    positive_probabilities: Optional[np.ndarray],
    negative_probabilities: Optional[np.ndarray],
) -> Dict[str, float]:
    """Score one grid-search candidate from precomputed batch components."""
    mode = cfg["fpde_mode"]
    if mode == "diff":
        evidences = comp["diff_ev_raw"]
        residuals = None
        lambda_hyb = None
    elif mode == "cos":
        evidences = comp["cos_ev_raw"]
        residuals = None
        lambda_hyb = None
    elif mode == "hyb_grid":
        lambda_hyb = float(cfg["lambda_hyb"])
        w_diff = lambda_hyb
        w_cos = 1.0 - lambda_hyb
        if cfg["normalize"] == "none":
            evidences = w_diff * comp["diff_ev_raw"] + w_cos * comp["cos_ev_raw"]
        elif cfg["normalize"] == "l1":
            evidences = w_diff * comp["diff_ev_l1"] + w_cos * comp["cos_ev_l1"]
        else:
            raise ValueError("normalize must be either 'none' or 'l1'")
        residuals = None
    else:
        raise ValueError(f"unknown mode={mode!r}")

    return _score_hyb_fpde_arrays(
        evidences=np.asarray(evidences, dtype=float),
        residuals=None if residuals is None else np.asarray(residuals, dtype=float),
        objective=objective,
        lambda_hyb=lambda_hyb,
        positive_probabilities=positive_probabilities,
        negative_probabilities=negative_probabilities,
    )


__all__: list[str] = []
