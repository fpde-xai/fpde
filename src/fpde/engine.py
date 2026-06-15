"""Stateful FPDE engine for repeated explanations and selection workflows."""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any, Dict, List, Literal, Optional, Sequence, Tuple

import numpy as np

from ._array import (
    _as_1d_float,
    _as_2d_float,
    _as_label_array,
    _canonical_fraction_array,
    _default_lambda_hyb_grid,
    _predict_proba_matrix,
    _predictor_probability_context,
    _top_two_probability_columns,
    _top_two_probability_labels,
    parse_float_grid,
)
from ._batch import (
    _batched_perturbation_scores,
    _batched_perturbation_scores_for_lambda_grid,
    _hyb_components_for_prototype_indices_batch,
    _hyb_metric_components_for_prototype_indices_batch,
)
from ._backend import DeviceMode, resolve_array_backend
from .prototypes import (
    _anchor_from_context,
    _context_from_training,
)
from .types import (
    AnchorStrategy,
    BayesianFPDELambdaSelectionResult,
    FPDEContext,
    GridMode,
    GridObjective,
    HybFPDEGridSearchResult,
    HybFPDEValidationSelectionResult,
    NormalizeMode,
)


@dataclass(frozen=True)
class _BatchHybArrays:
    attributions: np.ndarray
    positive_labels: np.ndarray
    negative_labels: np.ndarray
    target_probabilities: np.ndarray
    evidence: Optional[np.ndarray] = None
    direct_evidence: Optional[np.ndarray] = None
    positive_score: Optional[np.ndarray] = None
    negative_score: Optional[np.ndarray] = None


def _validate_lambda_hyb(lambda_hyb: float) -> float:
    value = float(lambda_hyb)
    if not np.isfinite(value):
        raise ValueError("lambda_hyb must be finite")
    if value < 0.0 or value > 1.0:
        raise ValueError("lambda_hyb must be in [0, 1]")
    return value


def _validate_normalize(normalize: NormalizeMode) -> NormalizeMode:
    if normalize not in ("none", "l1"):
        raise ValueError("normalize must be either 'none' or 'l1'")
    return normalize


def _validate_anchor_strategy(anchor_strategy: AnchorStrategy) -> AnchorStrategy:
    if anchor_strategy not in ("mean", "zero", "none"):
        raise ValueError("anchor_strategy must be 'mean', 'zero', or 'none'")
    return anchor_strategy


def _validate_eps(eps: float) -> float:
    value = float(eps)
    if value <= 0.0:
        raise ValueError("eps must be positive")
    return value


def _validate_positive_float(name: str, value: float) -> float:
    out = float(value)
    if not np.isfinite(out) or out <= 0.0:
        raise ValueError(f"{name} must be positive")
    return out


def _validate_credible_mass(credible_mass: float) -> float:
    value = float(credible_mass)
    if not np.isfinite(value) or value <= 0.0 or value >= 1.0:
        raise ValueError("credible_mass must be in (0, 1)")
    return value


def _stable_softmax(log_weights: np.ndarray) -> np.ndarray:
    if log_weights.ndim != 1 or log_weights.size == 0:
        raise ValueError("log_weights must be a non-empty 1D array")
    finite = np.isfinite(log_weights)
    if not np.any(finite):
        raise RuntimeError("all Bayesian-FPDE lambda candidates failed")
    shifted = np.full_like(log_weights, fill_value=float("-inf"), dtype=float)
    max_log = float(np.max(log_weights[finite]))
    shifted[finite] = log_weights[finite] - max_log
    weights = np.exp(shifted)
    total = float(np.sum(weights))
    if total <= 0.0 or not np.isfinite(total):
        raise RuntimeError("could not normalize Bayesian-FPDE posterior weights")
    return weights / total


def _weighted_quantile(values: np.ndarray, weights: np.ndarray, q: float) -> float:
    order = np.argsort(values)
    sorted_values = values[order]
    sorted_weights = weights[order]
    cumulative = np.cumsum(sorted_weights)
    idx = int(np.searchsorted(cumulative, float(q), side="left"))
    idx = min(max(idx, 0), sorted_values.shape[0] - 1)
    return float(sorted_values[idx])


def _credible_interval(values: np.ndarray, weights: np.ndarray, credible_mass: float) -> Tuple[float, float]:
    tail = 0.5 * (1.0 - credible_mass)
    return (
        _weighted_quantile(values, weights, tail),
        _weighted_quantile(values, weights, 1.0 - tail),
    )


def _component_mode_for_lambda(lambda_hyb: float) -> Literal["both", "diff", "cos"]:
    if lambda_hyb == 1.0:
        return "diff"
    if lambda_hyb == 0.0:
        return "cos"
    return "both"


def _component_mode_for_lambda_grid(lambdas: np.ndarray) -> Literal["both", "diff", "cos"]:
    if bool(np.all(lambdas == 1.0)):
        return "diff"
    if bool(np.all(lambdas == 0.0)):
        return "cos"
    return "both"


def _mixed_hyb_component(comp: Dict[str, np.ndarray], lambda_hyb: float, diff_key: str, cos_key: str) -> np.ndarray:
    if lambda_hyb == 1.0:
        return comp[diff_key]
    if lambda_hyb == 0.0:
        return comp[cos_key]
    return lambda_hyb * comp[diff_key] + (1.0 - lambda_hyb) * comp[cos_key]


def _bayesian_detail_metadata(selection: BayesianFPDELambdaSelectionResult) -> Dict[str, Any]:
    probabilities = np.asarray(
        [float(row.get("posterior_probability", 0.0)) for row in selection.posterior_rows],
        dtype=float,
    )
    positive = probabilities[probabilities > 0.0]
    entropy = float(-np.sum(positive * np.log(positive))) if positive.size else 0.0
    return {
        "lambda_source": "bayesian_posterior_mean",
        "posterior_mean_lambda": float(selection.posterior_mean_lambda),
        "map_lambda": float(selection.map_lambda),
        "credible_interval": tuple(float(v) for v in selection.credible_interval),
        "posterior_entropy": entropy,
        "effective_candidates": float(np.exp(entropy)),
    }


def _validation_row_base(
    *,
    candidate_id: int,
    lambda_hyb: float,
    normalize: NormalizeMode,
    anchor_strategy: AnchorStrategy,
    n_eval_samples: int,
) -> Dict[str, Any]:
    return {
        "candidate_id": int(candidate_id),
        "fpde_mode": "hyb_grid",
        "method_variant": "hyb_fpde_grid_validation",
        "normalize": normalize,
        "lambda_hyb": float(lambda_hyb),
        "lambda_grid_mode": "grid_candidate_lambda_hyb",
        "anchor_strategy": anchor_strategy,
        "selection_objective": "deletion_insertion_validation",
        "validation_metric_source": "heldout_deletion_insertion",
        "n_eval_samples": int(n_eval_samples),
    }


def _ok_validation_row(
    *,
    candidate_id: int,
    lambda_hyb: float,
    normalize: NormalizeMode,
    anchor_strategy: AnchorStrategy,
    n_eval_samples: int,
    p0: np.ndarray,
    deletion_auc: np.ndarray,
    deletion_drop_auc: np.ndarray,
    insertion_auc: np.ndarray,
    combined_score: np.ndarray,
    evidence: np.ndarray,
    residual: np.ndarray,
) -> Dict[str, Any]:
    row = _validation_row_base(
        candidate_id=candidate_id,
        lambda_hyb=lambda_hyb,
        normalize=normalize,
        anchor_strategy=anchor_strategy,
        n_eval_samples=n_eval_samples,
    )
    row.update(
        {
            "status": "ok",
            "score": float(np.mean(combined_score)),
            "mean_combined_score": float(np.mean(combined_score)),
            "mean_deletion_drop_auc": float(np.mean(deletion_drop_auc)),
            "mean_insertion_auc": float(np.mean(insertion_auc)),
            "mean_deletion_auc": float(np.mean(deletion_auc)),
            "mean_p0": float(np.mean(p0)),
            "mean_evidence": float(np.mean(evidence)),
            "mean_exactness_abs_residual": float(np.nanmean(np.abs(residual))),
            "n_success": int(n_eval_samples),
            "n_error": 0,
        }
    )
    return row


def _error_validation_row(
    *,
    candidate_id: int,
    lambda_hyb: float,
    normalize: NormalizeMode,
    anchor_strategy: AnchorStrategy,
    n_eval_samples: int,
    error: Exception,
) -> Dict[str, Any]:
    row = _validation_row_base(
        candidate_id=candidate_id,
        lambda_hyb=lambda_hyb,
        normalize=normalize,
        anchor_strategy=anchor_strategy,
        n_eval_samples=n_eval_samples,
    )
    row.update(
        {
            "status": "error",
            "score": float("-inf"),
            "error": f"{type(error).__name__}: {error}",
            "n_success": 0,
            "n_error": int(n_eval_samples),
        }
    )
    return row


class FPDEEngine:
    """Reusable FPDE state for fast repeated explanations.

    The engine owns the fitted class-mean prototype context, optional model,
    model class metadata, anchors, baseline, and a cached prototype-label lookup.
    Legacy function APIs remain available, but repeated workflows should prefer
    this stateful API.
    """

    def __init__(
        self,
        X_train: np.ndarray | Sequence[Sequence[float]],
        y_train: Sequence[Any],
        model: Optional[Any] = None,
        baseline: Optional[np.ndarray | Sequence[float]] = None,
        *,
        context: Optional[FPDEContext] = None,
        device: DeviceMode = "cpu",
    ) -> None:
        self._array_backend = resolve_array_backend(device)
        self.device = self._array_backend.name
        X_train_arr = _as_2d_float("X_train", X_train)
        y_train_arr = _as_label_array(y_train)
        if X_train_arr.shape[0] != y_train_arr.shape[0]:
            raise ValueError(
                f"number of train samples and labels differ: {X_train_arr.shape[0]} vs {y_train_arr.shape[0]}"
            )
        self.context = _context_from_training(
            X_train_arr,
            y_train_arr,
            baseline=baseline,
            context=context,
        )
        self.model = model
        self.model_classes = None if model is None or not hasattr(model, "classes_") else np.asarray(model.classes_, dtype=object)
        self.prototypes = self.context.prototypes
        self.prototype_labels = self.context.prototype_labels
        self.classes = self.prototype_labels
        self.n_features = int(self.context.n_features)
        if baseline is None:
            self.baseline = self.context.baseline
        else:
            baseline_arr = _as_1d_float("baseline", baseline)
            if baseline_arr.shape[0] != self.n_features:
                raise ValueError("baseline dimension differs from X_train feature dimension")
            self.baseline = baseline_arr
        self._label_to_index = self._build_label_index(self.prototype_labels)

    @classmethod
    def fit(
        cls,
        X_train: np.ndarray | Sequence[Sequence[float]],
        y_train: Sequence[Any],
        model: Optional[Any] = None,
        baseline: Optional[np.ndarray | Sequence[float]] = None,
        device: DeviceMode = "cpu",
    ) -> "FPDEEngine":
        """Fit a reusable FPDE engine from training data."""
        return cls(X_train, y_train, model=model, baseline=baseline, device=device)

    @staticmethod
    def _build_label_index(labels: np.ndarray) -> Optional[Dict[Any, int]]:
        try:
            out: Dict[Any, int] = {}
            for i, label in enumerate(labels.tolist()):
                out.setdefault(label, i)
            return out
        except TypeError:
            return None

    def _indices_for_labels(self, query_labels: Sequence[Any]) -> np.ndarray:
        query = list(query_labels)
        if self._label_to_index is not None:
            try:
                return np.fromiter(
                    (self._label_to_index[label] for label in query),
                    dtype=np.intp,
                    count=len(query),
                )
            except KeyError as exc:
                raise ValueError(f"label {exc.args[0]!r} is not available") from None

        out = np.empty(len(query), dtype=np.intp)
        for i, label in enumerate(query):
            matches = np.where(self.prototype_labels == label)[0]
            if matches.size == 0:
                raise ValueError(f"label {label!r} is not available")
            out[i] = int(matches[0])
        return out

    def _check_X(self, name: str, X: np.ndarray | Sequence[Sequence[float]]) -> np.ndarray:
        X_arr = _as_2d_float(name, X)
        if X_arr.shape[0] == 0:
            raise ValueError(f"{name} must contain at least one sample")
        if X_arr.shape[1] != self.n_features:
            raise ValueError(
                f"feature dimension mismatch: engine has {self.n_features}, {name} has {X_arr.shape[1]}"
            )
        return X_arr

    def _check_x(self, x: np.ndarray | Sequence[float]) -> np.ndarray:
        x_arr = _as_1d_float("x", x)
        if x_arr.shape[0] != self.n_features:
            raise ValueError(f"feature dimension mismatch: engine has {self.n_features}, x has {x_arr.shape[0]}")
        return x_arr

    def _model_or(self, model: Optional[Any]) -> Any:
        resolved = self.model if model is None else model
        if resolved is None:
            raise ValueError("a model with predict_proba is required")
        return resolved

    def _model_classes_for(self, model: Any) -> np.ndarray:
        if not hasattr(model, "classes_"):
            raise ValueError("model must expose classes_ when using predict_proba")
        return np.asarray(model.classes_, dtype=object)

    def _anchor(self, anchor_strategy: AnchorStrategy) -> np.ndarray:
        return _anchor_from_context(self.context, _validate_anchor_strategy(anchor_strategy))

    def _batch_arrays(
        self,
        X_arr: np.ndarray,
        *,
        model: Any,
        lambda_hyb: float,
        normalize: NormalizeMode,
        anchor_strategy: AnchorStrategy,
        eps: float,
        include_scores: bool,
    ) -> _BatchHybArrays:
        model_classes = self._model_classes_for(model)
        proba = _predict_proba_matrix(model, X_arr)
        if model_classes.shape[0] != proba.shape[1]:
            raise ValueError("model.classes_ length differs from predict_proba width")
        if proba.shape[1] < 2:
            raise ValueError("predict_proba must contain at least two classes for Hyb-FPDE")

        target_idx, rival_idx = _top_two_probability_columns(proba)
        positive_labels = model_classes[target_idx]
        negative_labels = model_classes[rival_idx]
        pos_proto_idx = self._indices_for_labels(positive_labels)
        neg_proto_idx = self._indices_for_labels(negative_labels)

        component_mode = _component_mode_for_lambda(lambda_hyb)

        anchor = self.context.zero_anchor if component_mode == "diff" else self._anchor(anchor_strategy)
        comp = _hyb_components_for_prototype_indices_batch(
            X_arr,
            self.prototypes,
            pos_proto_idx,
            neg_proto_idx,
            anchor=anchor,
            normalize=normalize,
            eps=eps,
            include_scores=include_scores,
            component_mode=component_mode,
            array_namespace=self._array_backend.xp,
        )

        attr = _mixed_hyb_component(comp, lambda_hyb, "diff_attr", "cos_attr")

        target_probability = proba[np.arange(X_arr.shape[0]), target_idx]
        rival_labels = self.prototype_labels[comp["neg_idx"]]
        if not include_scores:
            return _BatchHybArrays(
                attributions=attr.astype(float, copy=False),
                positive_labels=positive_labels,
                negative_labels=rival_labels,
                target_probabilities=target_probability,
            )

        evidence = _mixed_hyb_component(comp, lambda_hyb, "diff_attr_sum", "cos_attr_sum")
        direct_evidence = _mixed_hyb_component(comp, lambda_hyb, "diff_ev", "cos_ev")
        positive_score = _mixed_hyb_component(comp, lambda_hyb, "diff_pos", "cos_pos")
        negative_score = _mixed_hyb_component(comp, lambda_hyb, "diff_neg", "cos_neg")

        return _BatchHybArrays(
            attributions=attr.astype(float, copy=False),
            positive_labels=positive_labels,
            negative_labels=rival_labels,
            target_probabilities=target_probability,
            evidence=evidence,
            direct_evidence=direct_evidence,
            positive_score=positive_score,
            negative_score=negative_score,
        )

    def explain_one(
        self,
        x: np.ndarray | Sequence[float],
        *,
        lambda_hyb: float,
        normalize: NormalizeMode = "l1",
        anchor_strategy: AnchorStrategy = "mean",
        eps: float = 1e-12,
        model: Optional[Any] = None,
    ) -> Tuple[np.ndarray, Dict[str, Any]]:
        """Explain one sample with a fixed Hyb-FPDE lambda."""
        x_arr = self._check_x(x)
        attr, details = self.explain_batch(
            x_arr.reshape(1, -1),
            lambda_hyb=lambda_hyb,
            normalize=normalize,
            anchor_strategy=anchor_strategy,
            eps=eps,
            include_details=True,
            model=model,
        )
        return attr[0], details[0]

    def explain_batch(
        self,
        X: np.ndarray | Sequence[Sequence[float]],
        *,
        lambda_hyb: float,
        normalize: NormalizeMode = "l1",
        anchor_strategy: AnchorStrategy = "mean",
        include_details: bool = True,
        eps: float = 1e-12,
        model: Optional[Any] = None,
    ) -> Tuple[np.ndarray, List[Dict[str, Any]]]:
        """Explain many samples with a fixed Hyb-FPDE lambda."""
        X_arr = self._check_X("X", X)
        lambda_value = _validate_lambda_hyb(lambda_hyb)
        normalize = _validate_normalize(normalize)
        _validate_anchor_strategy(anchor_strategy)
        eps = _validate_eps(eps)
        resolved_model = self._model_or(model)

        arrays = self._batch_arrays(
            X_arr,
            model=resolved_model,
            lambda_hyb=lambda_value,
            normalize=normalize,
            anchor_strategy=anchor_strategy,
            eps=eps,
            include_scores=include_details,
        )
        if not include_details:
            return arrays.attributions, []

        assert arrays.evidence is not None
        assert arrays.direct_evidence is not None
        assert arrays.positive_score is not None
        assert arrays.negative_score is not None
        residual = arrays.evidence - arrays.direct_evidence
        details = []
        for i in range(X_arr.shape[0]):
            details.append(
                {
                    "target_label": arrays.positive_labels[i],
                    "rival_label": arrays.negative_labels[i],
                    "target_probability": float(arrays.target_probabilities[i]),
                    "lambda_hyb": float(lambda_value),
                    "evidence": float(arrays.evidence[i]),
                    "exactness_residual": float(residual[i]),
                    "positive_score": float(arrays.positive_score[i]),
                    "negative_score": float(arrays.negative_score[i]),
                }
            )
        return arrays.attributions, details

    def explain_matrix(
        self,
        X: np.ndarray | Sequence[Sequence[float]],
        *,
        lambda_hyb: float,
        normalize: NormalizeMode = "l1",
        anchor_strategy: AnchorStrategy = "mean",
        eps: float = 1e-12,
        model: Optional[Any] = None,
    ) -> np.ndarray:
        """Return only the fixed-lambda Hyb-FPDE attribution matrix."""
        attr, _ = self.explain_batch(
            X,
            lambda_hyb=lambda_hyb,
            normalize=normalize,
            anchor_strategy=anchor_strategy,
            include_details=False,
            eps=eps,
            model=model,
        )
        return attr

    @staticmethod
    def _check_bayesian_selection(selection: BayesianFPDELambdaSelectionResult) -> BayesianFPDELambdaSelectionResult:
        if not isinstance(selection, BayesianFPDELambdaSelectionResult):
            raise TypeError("selection must be a BayesianFPDELambdaSelectionResult")
        return selection

    def explain_one_bayesian(
        self,
        x: np.ndarray | Sequence[float],
        selection: BayesianFPDELambdaSelectionResult,
        *,
        model: Optional[Any] = None,
    ) -> Tuple[np.ndarray, Dict[str, Any]]:
        """Explain one sample using the Bayesian posterior mean lambda."""
        x_arr = self._check_x(x)
        attr, details = self.explain_batch_bayesian(
            x_arr.reshape(1, -1),
            selection,
            include_details=True,
            model=model,
        )
        return attr[0], details[0]

    def explain_batch_bayesian(
        self,
        X: np.ndarray | Sequence[Sequence[float]],
        selection: BayesianFPDELambdaSelectionResult,
        *,
        include_details: bool = True,
        model: Optional[Any] = None,
    ) -> Tuple[np.ndarray, List[Dict[str, Any]]]:
        """Explain many samples using the Bayesian posterior mean lambda."""
        selection = self._check_bayesian_selection(selection)
        attr, details = self.explain_batch(
            X,
            lambda_hyb=selection.posterior_mean_lambda,
            normalize=selection.normalize,
            anchor_strategy=selection.anchor_strategy,
            include_details=include_details,
            eps=selection.eps,
            model=model,
        )
        if include_details:
            metadata = _bayesian_detail_metadata(selection)
            for row in details:
                row.update(metadata)
        return attr, details

    def explain_matrix_bayesian(
        self,
        X: np.ndarray | Sequence[Sequence[float]],
        selection: BayesianFPDELambdaSelectionResult,
        *,
        model: Optional[Any] = None,
    ) -> np.ndarray:
        """Return only the Bayesian-FPDE attribution matrix."""
        attr, _ = self.explain_batch_bayesian(
            X,
            selection,
            include_details=False,
            model=model,
        )
        return attr

    def grid_search(
        self,
        X_eval: np.ndarray | Sequence[Sequence[float]],
        *,
        predictor: Optional[Any] = None,
        objective: GridObjective = "blackbox_agreement",
        fpde_mode_grid: Sequence[GridMode] = ("diff", "cos", "hyb_grid"),
        normalize_grid: Sequence[NormalizeMode] = ("l1",),
        lambda_hyb_grid: Sequence[float] = _default_lambda_hyb_grid(0.1),
        anchor_strategy_grid: Sequence[AnchorStrategy] = ("mean",),
        include_explicit_diff_cos: bool = True,
        max_eval_samples: Optional[int] = None,
        eps: float = 1e-12,
        verbose: bool = False,
    ) -> HybFPDEGridSearchResult:
        """Exhaustively search Diff/Cos/Hyb-FPDE settings for this engine."""
        from .selection import (
            _explain_single_prototype_candidate_from_contrast,
            _grid_candidate_metrics_from_components,
            _hyb_fpde_grid_candidate_dicts,
            _label_contrast_for_grid_sample,
            _score_hyb_fpde_explanations,
        )

        X_eval_arr = self._check_X("X_eval", X_eval)
        eps = _validate_eps(eps)
        if max_eval_samples is not None:
            m = int(max_eval_samples)
            if m < 1:
                raise ValueError("max_eval_samples must be positive when provided")
            X_eval_arr = X_eval_arr[: min(m, X_eval_arr.shape[0])]
        predictor = self.model if predictor is None else predictor

        candidates = _hyb_fpde_grid_candidate_dicts(
            fpde_mode_grid=fpde_mode_grid,
            normalize_grid=normalize_grid,
            lambda_hyb_grid=lambda_hyb_grid,
            anchor_strategy_grid=anchor_strategy_grid,
            include_explicit_diff_cos=include_explicit_diff_cos,
        )
        if not candidates:
            raise ValueError("grid produced no valid candidates")

        anchor_strategy_names = tuple(dict.fromkeys(str(cfg["anchor_strategy"]) for cfg in candidates))
        anchor_by_strategy: Dict[str, np.ndarray] = {}
        for anchor_strategy in anchor_strategy_names:
            anchor_by_strategy[anchor_strategy] = self._anchor(anchor_strategy)  # type: ignore[arg-type]

        sample_contexts = None
        component_by_strategy: Dict[str, Dict[str, np.ndarray]] = {}
        positive_probabilities = None
        negative_probabilities = None
        include_l1_components = any(cfg["fpde_mode"] == "hyb_grid" and cfg["normalize"] == "l1" for cfg in candidates)

        try:
            probabilities, probability_labels = _predictor_probability_context(predictor, X_eval_arr, self.classes)
            if probabilities is not None and probability_labels is not None:
                pos_labels, neg_labels, positive_probabilities, negative_probabilities = _top_two_probability_labels(
                    probabilities,
                    probability_labels,
                    self.prototype_labels,
                )
                pos_idx = self._indices_for_labels(pos_labels)
                neg_idx = self._indices_for_labels(neg_labels)
                for anchor_strategy, anchor in anchor_by_strategy.items():
                    component_by_strategy[anchor_strategy] = _hyb_metric_components_for_prototype_indices_batch(
                        X_eval_arr,
                        self.prototypes,
                        pos_idx,
                        neg_idx,
                        anchor=anchor,
                        eps=eps,
                        include_l1=include_l1_components,
                        array_namespace=self._array_backend.xp,
                    )
        except Exception:
            component_by_strategy = {}

        rows: List[Dict[str, Any]] = []
        best_row: Optional[Dict[str, Any]] = None
        best_config: Optional[Dict[str, Any]] = None

        for candidate_id, cfg in enumerate(candidates):
            try:
                anchor = anchor_by_strategy[cfg["anchor_strategy"]]
                comp = component_by_strategy.get(cfg["anchor_strategy"])
                if comp is None:
                    if sample_contexts is None:
                        sample_contexts = [
                            _label_contrast_for_grid_sample(
                                X_eval_arr[i],
                                prototypes=self.prototypes,
                                prototype_labels=self.prototype_labels,
                                classes=self.classes,
                                predictor=predictor,
                            )
                            for i in range(X_eval_arr.shape[0])
                        ]
                    explanations = [
                        _explain_single_prototype_candidate_from_contrast(
                            X_eval_arr[i],
                            prototypes=self.prototypes,
                            prototype_labels=self.prototype_labels,
                            classes=self.classes,
                            anchor=anchor,
                            cfg=cfg,
                            eps=eps,
                            pos=sample_contexts[i][0],
                            neg=sample_contexts[i][1],
                            probabilities=sample_contexts[i][2],
                            probability_labels=sample_contexts[i][3],
                        )
                        for i in range(X_eval_arr.shape[0])
                    ]
                    metrics = _score_hyb_fpde_explanations(explanations, objective=objective)
                else:
                    metrics = _grid_candidate_metrics_from_components(
                        cfg,
                        comp,
                        objective=objective,
                        positive_probabilities=positive_probabilities,
                        negative_probabilities=negative_probabilities,
                    )
                row: Dict[str, Any] = {
                    "candidate_id": int(candidate_id),
                    "status": "ok",
                    **cfg,
                    **metrics,
                    "n_eval_samples": int(X_eval_arr.shape[0]),
                    "n_fitted_prototypes": int(self.prototypes.shape[0]),
                }
            except Exception as exc:
                row = {
                    "candidate_id": int(candidate_id),
                    "status": "error",
                    **cfg,
                    "score": float("-inf"),
                    "error": f"{type(exc).__name__}: {exc}",
                    "n_eval_samples": int(X_eval_arr.shape[0]),
                }

            rows.append(row)
            if verbose:
                print(
                    f"[hyb-grid] candidate={candidate_id + 1}/{len(candidates)} "
                    f"status={row['status']} score={row['score']} cfg={cfg}"
                )

            if row["status"] == "ok":
                if best_row is None:
                    best_row = row
                    best_config = dict(cfg)
                else:
                    current_key = (float(row["score"]), float(row.get("mean_evidence", 0.0)))
                    best_key = (float(best_row["score"]), float(best_row.get("mean_evidence", 0.0)))
                    if current_key > best_key:
                        best_row = row
                        best_config = dict(cfg)

        if best_row is None or best_config is None:
            first_error = next((r.get("error", "unknown error") for r in rows if r["status"] == "error"), "unknown error")
            raise RuntimeError(f"all Hyb-FPDE grid candidates failed; first error: {first_error}")

        return HybFPDEGridSearchResult(
            best_config=best_config,
            best_score=float(best_row["score"]),
            rows=tuple(rows),
            objective=objective,
            n_candidates=len(candidates),
            n_eval_samples=int(X_eval_arr.shape[0]),
        )

    def select_lambda(
        self,
        X_val: np.ndarray | Sequence[Sequence[float]],
        *,
        lambda_hyb_grid: Sequence[float] = _default_lambda_hyb_grid(0.1),
        fractions: Sequence[float] = (0.0, 0.05, 0.1, 0.2, 0.3, 0.5, 0.7, 1.0),
        normalize: NormalizeMode = "l1",
        anchor_strategy: AnchorStrategy = "mean",
        eps: float = 1e-12,
        max_working_bytes: int = 256 * 1024 * 1024,
        model: Optional[Any] = None,
    ) -> HybFPDEValidationSelectionResult:
        """Select lambda_hyb by held-out deletion/insertion validation."""
        X_val_arr = self._check_X("X_val", X_val)
        normalize = _validate_normalize(normalize)
        _validate_anchor_strategy(anchor_strategy)
        eps = _validate_eps(eps)
        max_working_bytes = int(max_working_bytes)
        if max_working_bytes < 1:
            raise ValueError("max_working_bytes must be positive")

        model = self._model_or(model)
        model_classes = self._model_classes_for(model)
        lambdas = parse_float_grid(lambda_hyb_grid)
        for lam in lambdas:
            if lam < 0.0 or lam > 1.0:
                raise ValueError("lambda_hyb_grid values must be in [0, 1]")
        unique_lambdas = tuple(dict.fromkeys(float(lam) for lam in lambdas))
        unique_index = {lam: i for i, lam in enumerate(unique_lambdas)}

        frac_arr = _canonical_fraction_array(fractions)
        anchor = self._anchor(anchor_strategy)

        validation_context = np.empty((X_val_arr.shape[0] + 1, X_val_arr.shape[1]), dtype=float)
        validation_context[:-1] = X_val_arr
        validation_context[-1] = self.baseline
        validation_proba = _predict_proba_matrix(model, validation_context)
        val_proba = validation_proba[:-1]
        baseline_proba = validation_proba[-1]
        if model_classes.shape[0] != val_proba.shape[1]:
            raise ValueError("model.classes_ length differs from predict_proba width")
        if val_proba.shape[1] < 2:
            raise ValueError("predict_proba must contain at least two classes for Hyb-FPDE validation selection")

        target_label_indices, negative_label_indices = _top_two_probability_columns(val_proba)
        positive_labels = model_classes[target_label_indices]
        negative_labels = model_classes[negative_label_indices]
        row_idx = np.arange(X_val_arr.shape[0])
        base_prob = val_proba[row_idx, target_label_indices]
        baseline_prob = baseline_proba[target_label_indices]
        pos_proto_idx = self._indices_for_labels(positive_labels)
        neg_proto_idx = self._indices_for_labels(negative_labels)

        unique_lambdas_arr = np.asarray(unique_lambdas, dtype=float)
        component_mode = _component_mode_for_lambda_grid(unique_lambdas_arr)

        comp = _hyb_components_for_prototype_indices_batch(
            X_val_arr,
            self.prototypes,
            pos_proto_idx,
            neg_proto_idx,
            anchor=anchor,
            normalize=normalize,
            eps=eps,
            component_mode=component_mode,
            array_namespace=self._array_backend.xp,
        )

        all_curves: Optional[Dict[str, np.ndarray]]
        try:
            if component_mode == "diff":
                curves = _batched_perturbation_scores(
                    model,
                    X_val_arr,
                    comp["diff_attr"],
                    target_label_indices,
                    self.baseline,
                    frac_arr,
                    max_working_bytes=max_working_bytes,
                    base_prob=base_prob,
                    baseline_prob=baseline_prob,
                    array_namespace=self._array_backend.xp,
                )
                all_curves = {
                    name: np.broadcast_to(values, (unique_lambdas_arr.shape[0], values.shape[0]))
                    for name, values in curves.items()
                }
            elif component_mode == "cos":
                curves = _batched_perturbation_scores(
                    model,
                    X_val_arr,
                    comp["cos_attr"],
                    target_label_indices,
                    self.baseline,
                    frac_arr,
                    max_working_bytes=max_working_bytes,
                    base_prob=base_prob,
                    baseline_prob=baseline_prob,
                    array_namespace=self._array_backend.xp,
                )
                all_curves = {
                    name: np.broadcast_to(values, (unique_lambdas_arr.shape[0], values.shape[0]))
                    for name, values in curves.items()
                }
            else:
                all_curves = _batched_perturbation_scores_for_lambda_grid(
                    model,
                    X_val_arr,
                    comp["diff_attr"],
                    comp["cos_attr"],
                    unique_lambdas_arr,
                    target_label_indices,
                    self.baseline,
                    frac_arr,
                    max_working_bytes=max_working_bytes,
                    base_prob=base_prob,
                    baseline_prob=baseline_prob,
                    array_namespace=self._array_backend.xp,
                )
        except Exception:
            all_curves = None

        unique_rows: List[Dict[str, Any]] = []
        for candidate_id, lam in enumerate(unique_lambdas):
            lam_f = float(lam)
            evidence = _mixed_hyb_component(comp, lam_f, "diff_attr_sum", "cos_attr_sum")
            direct_evidence = _mixed_hyb_component(comp, lam_f, "diff_ev", "cos_ev")
            residual = evidence - direct_evidence

            try:
                if all_curves is None:
                    attr = _mixed_hyb_component(comp, lam_f, "diff_attr", "cos_attr")
                    curves = _batched_perturbation_scores(
                        model,
                        X_val_arr,
                        attr,
                        target_label_indices,
                        self.baseline,
                        frac_arr,
                        max_working_bytes=max_working_bytes,
                        base_prob=base_prob,
                        baseline_prob=baseline_prob,
                        array_namespace=self._array_backend.xp,
                    )
                    p0 = curves["p0"]
                    deletion_auc = curves["deletion_auc"]
                    deletion_drop_auc = curves["deletion_drop_auc"]
                    insertion_auc = curves["insertion_auc"]
                    combined_score = curves["combined_score"]
                else:
                    p0 = all_curves["p0"][candidate_id]
                    deletion_auc = all_curves["deletion_auc"][candidate_id]
                    deletion_drop_auc = all_curves["deletion_drop_auc"][candidate_id]
                    insertion_auc = all_curves["insertion_auc"][candidate_id]
                    combined_score = all_curves["combined_score"][candidate_id]
                row = _ok_validation_row(
                    candidate_id=candidate_id,
                    lambda_hyb=lam_f,
                    normalize=normalize,
                    anchor_strategy=anchor_strategy,
                    n_eval_samples=X_val_arr.shape[0],
                    p0=p0,
                    deletion_auc=deletion_auc,
                    deletion_drop_auc=deletion_drop_auc,
                    insertion_auc=insertion_auc,
                    combined_score=combined_score,
                    evidence=evidence,
                    residual=residual,
                )
            except Exception as exc:
                row = _error_validation_row(
                    candidate_id=candidate_id,
                    lambda_hyb=lam_f,
                    normalize=normalize,
                    anchor_strategy=anchor_strategy,
                    n_eval_samples=X_val_arr.shape[0],
                    error=exc,
                )
            unique_rows.append(row)

        rows: List[Dict[str, Any]] = []
        for candidate_id, lam in enumerate(lambdas):
            row = dict(unique_rows[unique_index[float(lam)]])
            row["candidate_id"] = int(candidate_id)
            rows.append(row)

        ok_rows = [r for r in rows if r.get("status") == "ok"]
        if not ok_rows:
            first_error = next((r.get("error", "unknown error") for r in rows), "unknown error")
            raise RuntimeError(f"all lambda candidates failed; first error: {first_error}")

        best_row = max(
            ok_rows,
            key=lambda r: (
                float(r["score"]),
                float(r.get("mean_insertion_auc", 0.0)),
                -abs(float(r["lambda_hyb"]) - 0.5),
            ),
        )
        best_lambda = float(best_row["lambda_hyb"])
        best_config = {
            "fpde_mode": "hyb_grid",
            "method_variant": "hyb_fpde_grid_validation",
            "normalize": normalize,
            "lambda_hyb": best_lambda,
            "lambda_grid_mode": "grid_candidate_lambda_hyb",
            "anchor_strategy": anchor_strategy,
            "selection_objective": "deletion_insertion_validation",
            "validation_metric_source": "heldout_deletion_insertion",
            "validation_score": float(best_row["score"]),
            "validation_mean_deletion_drop_auc": float(best_row.get("mean_deletion_drop_auc", float("nan"))),
            "validation_mean_insertion_auc": float(best_row.get("mean_insertion_auc", float("nan"))),
            "n_grid_eval_samples": int(X_val_arr.shape[0]),
        }
        return HybFPDEValidationSelectionResult(
            best_lambda=best_lambda,
            best_config=best_config,
            rows=tuple(rows),
            n_eval_samples=int(X_val_arr.shape[0]),
        )

    def select_bayesian_lambda(
        self,
        X_val: np.ndarray | Sequence[Sequence[float]],
        *,
        lambda_hyb_grid: Sequence[float] = _default_lambda_hyb_grid(0.1),
        fractions: Sequence[float] = (0.0, 0.05, 0.1, 0.2, 0.3, 0.5, 0.7, 1.0),
        normalize: NormalizeMode = "l1",
        anchor_strategy: AnchorStrategy = "mean",
        eps: float = 1e-12,
        max_working_bytes: int = 256 * 1024 * 1024,
        alpha: float = 1.0,
        beta: float = 1.0,
        temperature: float = 1.0,
        credible_mass: float = 0.95,
        model: Optional[Any] = None,
    ) -> BayesianFPDELambdaSelectionResult:
        """Build a Bayesian posterior over validation-scored lambda_hyb candidates."""
        alpha_value = _validate_positive_float("alpha", alpha)
        beta_value = _validate_positive_float("beta", beta)
        temperature_value = _validate_positive_float("temperature", temperature)
        credible_mass_value = _validate_credible_mass(credible_mass)

        lambdas = parse_float_grid(lambda_hyb_grid)
        for lam in lambdas:
            if lam < 0.0 or lam > 1.0:
                raise ValueError("lambda_hyb_grid values must be in [0, 1]")
        unique_lambdas = tuple(dict.fromkeys(float(lam) for lam in lambdas))

        validation = self.select_lambda(
            X_val,
            lambda_hyb_grid=unique_lambdas,
            fractions=fractions,
            normalize=normalize,
            anchor_strategy=anchor_strategy,
            eps=eps,
            max_working_bytes=max_working_bytes,
            model=model,
        )

        lambda_values = np.asarray([float(row["lambda_hyb"]) for row in validation.rows], dtype=float)
        log_weights = np.full(lambda_values.shape, fill_value=float("-inf"), dtype=float)
        clip_eps = np.finfo(float).eps

        for i, row in enumerate(validation.rows):
            if row.get("status") != "ok":
                continue
            score = float(row["score"])
            if not np.isfinite(score):
                continue
            lam = float(row["lambda_hyb"])
            clipped_lam = float(np.clip(lam, clip_eps, 1.0 - clip_eps))
            clipped_one_minus = float(np.clip(1.0 - lam, clip_eps, 1.0 - clip_eps))
            log_likelihood = validation.n_eval_samples * score / temperature_value
            log_prior = (alpha_value - 1.0) * np.log(clipped_lam)
            log_prior += (beta_value - 1.0) * np.log(clipped_one_minus)
            log_weights[i] = float(log_likelihood + log_prior)

        probabilities = _stable_softmax(log_weights)
        posterior_mean = float(np.sum(probabilities * lambda_values))
        map_idx = int(np.argmax(probabilities))
        interval = _credible_interval(lambda_values, probabilities, credible_mass_value)

        posterior_rows: List[Dict[str, Any]] = []
        for i, row in enumerate(validation.rows):
            out = dict(row)
            out["candidate_id"] = int(i)
            out["posterior_probability"] = float(probabilities[i])
            if np.isfinite(log_weights[i]):
                out["log_posterior_unnormalized"] = float(log_weights[i])
                out["log_likelihood"] = float(validation.n_eval_samples * float(row["score"]) / temperature_value)
                lam = float(row["lambda_hyb"])
                clipped_lam = float(np.clip(lam, clip_eps, 1.0 - clip_eps))
                clipped_one_minus = float(np.clip(1.0 - lam, clip_eps, 1.0 - clip_eps))
                out["log_prior"] = float(
                    (alpha_value - 1.0) * np.log(clipped_lam)
                    + (beta_value - 1.0) * np.log(clipped_one_minus)
                )
            else:
                out["log_posterior_unnormalized"] = float("-inf")
                out["log_likelihood"] = float("-inf")
                out["log_prior"] = float("-inf")
            posterior_rows.append(out)

        return BayesianFPDELambdaSelectionResult(
            posterior_mean_lambda=posterior_mean,
            map_lambda=float(lambda_values[map_idx]),
            credible_interval=interval,
            posterior_rows=tuple(posterior_rows),
            prior_alpha=alpha_value,
            prior_beta=beta_value,
            temperature=temperature_value,
            normalize=normalize,
            anchor_strategy=anchor_strategy,
            eps=eps,
            n_eval_samples=validation.n_eval_samples,
        )


__all__ = ["FPDEEngine"]
