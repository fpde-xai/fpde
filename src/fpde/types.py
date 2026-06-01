"""Shared public types for FPDE."""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any, Dict, List, Literal, Tuple

import numpy as np

Mode = Literal["diff", "cos"]
GridMode = Literal["diff", "cos", "hyb_grid"]
NormalizeMode = Literal["none", "l1"]
AnchorStrategy = Literal["mean", "zero", "none"]
GridObjective = Literal[
    "blackbox_agreement",
    "mean_positive_evidence",
    "mean_margin_weighted_evidence",
]


@dataclass(frozen=True)
class FPDEExplanation:
    """Result object for a strict FPDE explanation."""

    mode: GridMode
    evidence: float
    attributions: np.ndarray
    positive_score: float
    negative_score: float
    positive_label: Any
    negative_label: Any
    positive_prototype_index: int
    negative_prototype_index: int
    exactness_residual: float
    details: Dict[str, Any]

    @property
    def normalized_attributions(self) -> np.ndarray:
        """Return an L1-normalized attribution vector for visualization only."""
        denom = float(np.sum(np.abs(self.attributions)))
        if denom <= 0.0 or not np.isfinite(denom):
            return np.zeros_like(self.attributions, dtype=float)
        return self.attributions / denom


@dataclass(frozen=True)
class FPDEContext:
    """Reusable training-side FPDE state for repeated explanations."""

    prototypes: np.ndarray
    prototype_labels: np.ndarray
    mean_anchor: np.ndarray
    zero_anchor: np.ndarray
    baseline: np.ndarray
    n_features: int

@dataclass(frozen=True)
class HybFPDEGridSearchResult:
    """Result of an exhaustive grid search over Diff/Cos/Hyb-FPDE settings.

    Report this condition as "Hyb-FPDE" or "Hyb-FPDE (lambda grid)".
    Do not describe it as Adaptive-Hyb-FPDE, because lambda_hyb is selected by
    grid search rather than computed sample-wise.
    """

    best_config: Dict[str, Any]
    best_score: float
    rows: Tuple[Dict[str, Any], ...]
    objective: GridObjective
    n_candidates: int
    n_eval_samples: int

    def sorted_rows(self) -> List[Dict[str, Any]]:
        """Return candidate rows sorted from best to worst."""
        return sorted(
            (dict(row) for row in self.rows),
            key=lambda r: (float(r["score"]), float(r.get("mean_evidence", 0.0))),
            reverse=True,
        )

@dataclass(frozen=True)
class HybFPDEValidationSelectionResult:
    """Result of validation-based lambda_hyb selection for Hyb-FPDE.

    The rows field contains one dictionary per lambda candidate.  The selected
    lambda maximizes the held-out validation combined score:

        combined_score = 0.5 * (deletion_drop_auc + insertion_auc)

    This result object is intentionally separate from HybFPDEGridSearchResult,
    because this validation-based selection evaluates perturbation curves rather
    than evidence-only FPDE grid-search objectives.
    """

    best_lambda: float
    best_config: Dict[str, Any]
    rows: Tuple[Dict[str, Any], ...]
    n_eval_samples: int

    def sorted_rows(self) -> List[Dict[str, Any]]:
        """Return lambda candidates sorted from best to worst."""
        return sorted(
            (dict(row) for row in self.rows),
            key=lambda r: (
                float(r.get("score", float("-inf"))),
                float(r.get("mean_insertion_auc", 0.0)),
                -abs(float(r.get("lambda_hyb", 0.5)) - 0.5),
            ),
            reverse=True,
        )


@dataclass(frozen=True)
class BayesianFPDELambdaSelectionResult:
    """Bayesian posterior over Hyb-FPDE lambda candidates.

    The posterior is defined on the finite lambda grid supplied to
    FPDEEngine.select_bayesian_lambda.  The posterior mean lambda can be used
    as a Bayesian model-averaged Hyb-FPDE mixture weight.
    """

    posterior_mean_lambda: float
    map_lambda: float
    credible_interval: Tuple[float, float]
    posterior_rows: Tuple[Dict[str, Any], ...]
    prior_alpha: float
    prior_beta: float
    temperature: float
    normalize: NormalizeMode
    anchor_strategy: AnchorStrategy
    eps: float
    n_eval_samples: int

    def sorted_rows(self) -> List[Dict[str, Any]]:
        """Return posterior rows sorted from highest to lowest posterior mass."""
        return sorted(
            (dict(row) for row in self.posterior_rows),
            key=lambda r: (
                float(r.get("posterior_probability", 0.0)),
                float(r.get("score", float("-inf"))),
            ),
            reverse=True,
        )


__all__ = [
    "Mode",
    "GridMode",
    "NormalizeMode",
    "AnchorStrategy",
    "GridObjective",
    "FPDEExplanation",
    "FPDEContext",
    "HybFPDEGridSearchResult",
    "HybFPDEValidationSelectionResult",
    "BayesianFPDELambdaSelectionResult",
]
