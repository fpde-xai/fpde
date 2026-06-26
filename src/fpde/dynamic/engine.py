"""Stateful RawFeat Dynamic-FPDE engine."""

from __future__ import annotations

from typing import Any, Callable, Dict, Optional, Sequence

import numpy as np

from .._array import _as_label_array
from .legacy import _labels_unique, _validate_eps, _validate_lambda_hyb
from .preprocessing import DynamicInputBatch, validate_sequence_inputs
from .types import DynamicFPDEResult, DynamicMethod


def _method_name(method: str) -> DynamicMethod:
    aliases = {
        "diff": "diff",
        "cos": "cos",
        "hyb": "hyb",
        "dynamic_diff": "diff",
        "dynamic_cos": "cos",
        "dynamic_hyb": "hyb",
    }
    try:
        return aliases[str(method)]
    except KeyError:
        raise ValueError("method must be 'diff', 'cos', or 'hyb'") from None


def _l1_or_zero(values: np.ndarray, eps: float) -> np.ndarray:
    scale = float(np.sum(np.abs(values)))
    if scale <= eps:
        return np.zeros_like(values, dtype=float)
    return values / scale


def split_representation(
    representation: np.ndarray | Sequence[Sequence[float]] | Sequence[Sequence[Sequence[float]]],
    feature_slices: Dict[str, slice],
) -> tuple[np.ndarray, Optional[np.ndarray], Optional[np.ndarray]]:
    """Split a concatenated RawFeat representation into raw/features/dt arrays."""
    rep = np.asarray(representation, dtype=float)
    if rep.ndim not in (2, 3):
        raise ValueError(f"representation must be a 2D or 3D array, got shape={rep.shape}")
    raw = rep[..., feature_slices["raw"]]
    features = rep[..., feature_slices["features"]] if "features" in feature_slices else None
    dt = rep[..., feature_slices["dt"]] if "dt" in feature_slices else None
    return raw, features, dt


class DynamicFPDEEngine:
    """RawFeat Dynamic-FPDE for raw sequences plus optional feature sequences.

    The first implementation uses the concatenated input
    ``representation_ = concat(raw, features, dt)`` directly. This keeps the
    attribution identity exact while leaving an internal representation hook for
    future encoders.
    """

    def __init__(self, *, lambda_hyb: float = 0.5, eps: float = 1e-12, tolerance: float = 1e-9) -> None:
        self.lambda_hyb = _validate_lambda_hyb(lambda_hyb)
        self.eps = _validate_eps(eps)
        self.tolerance = float(tolerance)
        if not np.isfinite(self.tolerance) or self.tolerance < 0.0:
            raise ValueError("tolerance must be non-negative")

    def fit(
        self,
        *,
        raw: np.ndarray | Sequence[np.ndarray | Sequence[Sequence[float]]],
        y: Sequence[Any],
        features: Optional[np.ndarray | Sequence[np.ndarray | Sequence[Sequence[float]]]] = None,
        dt: Optional[np.ndarray | Sequence[np.ndarray | Sequence[Sequence[float]]]] = None,
        mask: Optional[np.ndarray | Sequence[Sequence[bool]] | Sequence[bool]] = None,
    ) -> "DynamicFPDEEngine":
        """Fit mask-weighted temporal prototypes."""
        batch = validate_sequence_inputs(raw, features=features, dt=dt, mask=mask)
        labels = _as_label_array(y)
        if labels.shape[0] != batch.raw.shape[0]:
            raise ValueError(f"number of raw samples and labels differ: {batch.raw.shape[0]} vs {labels.shape[0]}")

        classes, inverse = _labels_unique(labels)
        if classes.shape[0] < 2:
            raise ValueError("DynamicFPDEEngine requires at least two classes")

        representation = batch.representation
        prototypes = np.zeros((classes.shape[0], representation.shape[1], representation.shape[2]), dtype=float)
        class_masks = np.zeros((classes.shape[0], representation.shape[1]), dtype=bool)
        class_counts: Dict[Any, int] = {}
        for class_idx, label in enumerate(classes.tolist()):
            selected = inverse == class_idx
            weights = batch.mask[selected].astype(float)
            denom = np.sum(weights, axis=0)
            numer = np.sum(representation[selected] * weights[:, :, None], axis=0)
            valid = denom > 0.0
            prototypes[class_idx, valid, :] = numer[valid, :] / denom[valid, None]
            class_masks[class_idx] = valid
            class_counts[label] = int(np.sum(selected))

        weights_all = batch.mask.astype(float)
        denom_all = np.sum(weights_all, axis=0)
        numer_all = np.sum(representation * weights_all[:, :, None], axis=0)
        anchor = np.zeros((representation.shape[1], representation.shape[2]), dtype=float)
        valid_all = denom_all > 0.0
        anchor[valid_all] = numer_all[valid_all] / denom_all[valid_all, None]

        self.input_ = batch
        self.representation_ = representation
        self.prototypes_ = prototypes
        self.prototype_masks_ = class_masks
        self.prototype_labels_ = classes.copy()
        self.classes_ = self.prototype_labels_
        self.mean_anchor_ = anchor
        self.feature_slices_ = dict(batch.feature_slices)
        self.class_counts_ = class_counts
        return self

    def _check_fitted(self) -> None:
        if not hasattr(self, "prototypes_"):
            raise RuntimeError("DynamicFPDEEngine must be fit before explaining")

    def _prototype_index(self, label: Any, name: str) -> int:
        matches = np.where(self.prototype_labels_ == label)[0]
        if matches.size == 0:
            raise ValueError(f"no prototype found for {name}={label!r}")
        return int(matches[0])

    def _coerce_one(
        self,
        raw: np.ndarray | Sequence[Sequence[float]],
        *,
        features: Optional[np.ndarray | Sequence[Sequence[float]]],
        dt: Optional[np.ndarray | Sequence[Sequence[float]]],
        mask: Optional[np.ndarray | Sequence[bool]],
    ) -> DynamicInputBatch:
        batch = validate_sequence_inputs(raw, features=features, dt=dt, mask=mask)
        if batch.raw.shape[0] != 1:
            raise ValueError("explain_one expects exactly one sample")
        if batch.raw.shape[1] > self.prototypes_.shape[1]:
            raise ValueError(
                f"input length {batch.raw.shape[1]} exceeds fitted maximum length {self.prototypes_.shape[1]}"
            )
        if batch.representation.shape[2] != self.prototypes_.shape[2]:
            raise ValueError(
                f"representation dimension mismatch: engine has {self.prototypes_.shape[2]}, input has {batch.representation.shape[2]}"
            )
        return batch

    def _pad_to_fit(self, representation: np.ndarray, mask: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
        t_fit = int(self.prototypes_.shape[1])
        if representation.shape[1] == t_fit:
            return representation[0].copy(), mask[0].copy()
        padded = np.zeros((t_fit, representation.shape[2]), dtype=float)
        padded_mask = np.zeros(t_fit, dtype=bool)
        t = int(representation.shape[1])
        padded[:t] = representation[0]
        padded_mask[:t] = mask[0]
        return padded, padded_mask

    def _nearest_indices(self, u: np.ndarray, mask: np.ndarray) -> np.ndarray:
        distances = np.empty(self.prototypes_.shape[0], dtype=float)
        for idx in range(self.prototypes_.shape[0]):
            valid = (mask & self.prototype_masks_[idx])[:, None]
            distances[idx] = float(np.sum(np.where(valid, (u - self.prototypes_[idx]) ** 2, 0.0)))
            if not np.any(valid):
                distances[idx] = float("inf")
        return np.argsort(distances)

    def _resolve_target_rival(
        self,
        u: np.ndarray,
        mask: np.ndarray,
        *,
        target_class: Any,
        rival_class: Any,
        predict_proba: Optional[Sequence[float]],
        class_names: Optional[Sequence[Any]],
    ) -> tuple[Any, Any]:
        if target_class is not None and rival_class is not None:
            if target_class == rival_class:
                raise ValueError("target_class and rival_class must differ")
            self._prototype_index(target_class, "target_class")
            self._prototype_index(rival_class, "rival_class")
            return target_class, rival_class

        if predict_proba is not None:
            proba = np.asarray(predict_proba, dtype=float)
            if proba.ndim != 1:
                raise ValueError("predict_proba for explain_one must be a 1D vector")
            if proba.shape[0] < 2:
                raise ValueError("predict_proba must contain at least two classes")
            names = self.prototype_labels_ if class_names is None else np.asarray(class_names, dtype=object)
            if names.shape[0] != proba.shape[0]:
                raise ValueError("class_names length must match predict_proba")
            order = np.argsort(proba)[::-1]
            resolved_target = names[int(order[0])]
            resolved_rival = names[int(order[1])]
            self._prototype_index(resolved_target, "target_class")
            self._prototype_index(resolved_rival, "rival_class")
            return resolved_target, resolved_rival

        if target_class is not None:
            self._prototype_index(target_class, "target_class")
            order = self._nearest_indices(u, mask)
            for idx in order.tolist():
                candidate = self.prototype_labels_[idx]
                if candidate != target_class:
                    return target_class, candidate
            raise ValueError("could not resolve a non-target rival class")

        order = self._nearest_indices(u, mask)
        if order.shape[0] < 2:
            raise ValueError("at least two prototypes are required")
        return self.prototype_labels_[int(order[0])], self.prototype_labels_[int(order[1])]

    def _attributions(
        self,
        u: np.ndarray,
        mask: np.ndarray,
        target_idx: int,
        rival_idx: int,
        method: DynamicMethod,
    ) -> tuple[np.ndarray, Dict[str, Any]]:
        target = self.prototypes_[target_idx]
        rival = self.prototypes_[rival_idx]
        target_valid = self.prototype_masks_[target_idx]
        rival_valid = self.prototype_masks_[rival_idx]
        effective_mask = mask & target_valid & rival_valid
        valid = effective_mask[:, None].astype(float)

        diff = valid * ((u - rival) ** 2 - (u - target) ** 2)
        z = u - self.mean_anchor_
        q_target = target - self.mean_anchor_
        q_rival = rival - self.mean_anchor_
        z_norm = np.sqrt(np.sum(z * z, axis=1, keepdims=True) + self.eps**2)
        target_norm = np.sqrt(np.sum(q_target * q_target, axis=1, keepdims=True) + self.eps**2)
        rival_norm = np.sqrt(np.sum(q_rival * q_rival, axis=1, keepdims=True) + self.eps**2)
        cos = valid * ((z * q_target) / (z_norm * target_norm) - (z * q_rival) / (z_norm * rival_norm))

        details: Dict[str, Any] = {
            "diff_evidence": float(np.sum(diff)),
            "cos_evidence": float(np.sum(cos)),
            "diff_l1": float(np.sum(np.abs(diff))),
            "cos_l1": float(np.sum(np.abs(cos))),
            "valid_time_count": int(np.sum(effective_mask)),
            "masked_time_count": int(np.sum(~mask)),
            "prototype_invalid_time_count": int(np.sum(mask & ~(target_valid & rival_valid))),
            "target_prototype_valid_count": int(np.sum(target_valid)),
            "rival_prototype_valid_count": int(np.sum(rival_valid)),
            "effective_mask": effective_mask.copy(),
        }
        if details["valid_time_count"] == 0:
            details["warning"] = "no_valid_time"

        if method == "diff":
            return diff, details
        if method == "cos":
            return cos, details

        diff_part = _l1_or_zero(diff, self.eps)
        cos_part = _l1_or_zero(cos, self.eps)
        attr = self.lambda_hyb * diff_part + (1.0 - self.lambda_hyb) * cos_part
        details["lambda_hyb"] = float(self.lambda_hyb)
        return attr, details

    def explain_one(
        self,
        *,
        raw: np.ndarray | Sequence[Sequence[float]],
        features: Optional[np.ndarray | Sequence[Sequence[float]]] = None,
        dt: Optional[np.ndarray | Sequence[Sequence[float]]] = None,
        mask: Optional[np.ndarray | Sequence[bool]] = None,
        method: str = "hyb",
        target_class: Any = None,
        rival_class: Any = None,
        predict_proba: Optional[Sequence[float]] = None,
        class_names: Optional[Sequence[Any]] = None,
    ) -> DynamicFPDEResult:
        """Explain one raw/feature sequence."""
        self._check_fitted()
        method_value = _method_name(method)
        batch = self._coerce_one(raw, features=features, dt=dt, mask=mask)
        u, mask_one = self._pad_to_fit(batch.representation, batch.mask)
        target, rival = self._resolve_target_rival(
            u,
            mask_one,
            target_class=target_class,
            rival_class=rival_class,
            predict_proba=predict_proba,
            class_names=class_names,
        )
        target_idx = self._prototype_index(target, "target_class")
        rival_idx = self._prototype_index(rival, "rival_class")
        attr, attr_details = self._attributions(u, mask_one, target_idx, rival_idx, method_value)
        effective_mask = attr_details.pop("effective_mask")
        attr[~effective_mask] = 0.0
        evidence = float(np.sum(attr))

        slices = dict(self.feature_slices_)
        raw_attr = attr[:, slices["raw"]].copy()
        feature_attr = attr[:, slices["features"]].copy() if "features" in slices else None
        dt_attr = attr[:, slices["dt"]].copy() if "dt" in slices else None
        groups = {
            "raw": float(np.sum(raw_attr)),
            "features": 0.0 if feature_attr is None else float(np.sum(feature_attr)),
            "dt": 0.0 if dt_attr is None else float(np.sum(dt_attr)),
        }
        attribution_sum = float(np.sum(attr))
        abs_error = float(abs(evidence - attribution_sum))
        return DynamicFPDEResult(
            method=method_value,
            target_class=target,
            rival_class=rival,
            evidence=evidence,
            attributions=attr.copy(),
            raw_attributions=raw_attr,
            feature_attributions=feature_attr,
            dt_attributions=dt_attr,
            time_attributions=np.sum(attr, axis=1),
            group_attributions=groups,
            mask=mask_one.copy(),
            feature_slices=slices,
            audit={
                "attribution_sum": attribution_sum,
                "evidence": evidence,
                "abs_error": abs_error,
                "passed": abs_error <= self.tolerance,
                **attr_details,
            },
        )

    def explain_batch(
        self,
        *,
        raw: np.ndarray | Sequence[np.ndarray | Sequence[Sequence[float]]],
        features: Optional[np.ndarray | Sequence[np.ndarray | Sequence[Sequence[float]]]] = None,
        dt: Optional[np.ndarray | Sequence[np.ndarray | Sequence[Sequence[float]]]] = None,
        mask: Optional[np.ndarray | Sequence[Sequence[bool]]] = None,
        target_classes: Optional[Sequence[Any]] = None,
        rival_classes: Optional[Sequence[Any]] = None,
        predict_proba: Optional[np.ndarray | Sequence[Sequence[float]]] = None,
        class_names: Optional[Sequence[Any]] = None,
        method: str = "hyb",
    ) -> list[DynamicFPDEResult]:
        """Explain a batch of raw/feature sequences."""
        self._check_fitted()
        batch = validate_sequence_inputs(raw, features=features, dt=dt, mask=mask)
        n = int(batch.raw.shape[0])
        targets = [None] * n if target_classes is None else list(target_classes)
        rivals = [None] * n if rival_classes is None else list(rival_classes)
        if len(targets) != n:
            raise ValueError(f"number of target_classes differs from raw: {len(targets)} vs {n}")
        if len(rivals) != n:
            raise ValueError(f"number of rival_classes differs from raw: {len(rivals)} vs {n}")
        proba_arr = None if predict_proba is None else np.asarray(predict_proba, dtype=float)
        if proba_arr is not None:
            if proba_arr.ndim != 2:
                raise ValueError("predict_proba for explain_batch must be a 2D array with shape (N, K)")
            if proba_arr.shape[0] != n:
                raise ValueError(f"predict_proba first dimension must match raw samples: {proba_arr.shape[0]} vs {n}")
            expected_classes = self.prototype_labels_ if class_names is None else np.asarray(class_names, dtype=object)
            if expected_classes.shape[0] != proba_arr.shape[1]:
                raise ValueError("predict_proba class dimension must match class_names or fitted prototype labels")

        results = []
        for i in range(n):
            sample_features = None if batch.features is None else batch.features[i]
            sample_dt = None if batch.dt is None else batch.dt[i]
            sample_proba = None if proba_arr is None else proba_arr[i]
            results.append(
                self.explain_one(
                    raw=batch.raw[i],
                    features=sample_features,
                    dt=sample_dt,
                    mask=batch.mask[i],
                    method=method,
                    target_class=targets[i],
                    rival_class=rivals[i],
                    predict_proba=sample_proba,
                    class_names=class_names,
                )
            )
        return results

    def select_lambda(
        self,
        *,
        raw: np.ndarray | Sequence[np.ndarray | Sequence[Sequence[float]]],
        features: Optional[np.ndarray | Sequence[np.ndarray | Sequence[Sequence[float]]]] = None,
        dt: Optional[np.ndarray | Sequence[np.ndarray | Sequence[Sequence[float]]]] = None,
        mask: Optional[np.ndarray | Sequence[Sequence[bool]]] = None,
        predict_proba: Any,
        lambdas: Optional[Sequence[float]] = None,
        target_classes: Optional[Sequence[Any]] = None,
        rival_classes: Optional[Sequence[Any]] = None,
        class_names: Optional[Sequence[Any]] = None,
        baseline: str | np.ndarray = "mean",
        steps: int = 20,
    ) -> Dict[str, Any]:
        """Select ``lambda_hyb`` with RawFeat Dynamic-FPDE validation perturbations."""
        return select_lambda_dynamic(
            engine=self,
            raw=raw,
            features=features,
            dt=dt,
            mask=mask,
            predict_proba=predict_proba,
            lambdas=lambdas,
            target_classes=target_classes,
            rival_classes=rival_classes,
            class_names=class_names,
            baseline=baseline,
            steps=steps,
        )


def _lambda_values(lambdas: Optional[Sequence[float]], lambda_grid: Optional[Sequence[float]]) -> list[float]:
    values = lambdas if lambdas is not None else lambda_grid
    if values is None:
        values = [i / 10.0 for i in range(11)]
    out = [_validate_lambda_hyb(float(value)) for value in values]
    if not out:
        raise ValueError("lambdas must contain at least one value")
    return out


def _placeholder_lambda_selection(lambdas: Optional[Sequence[float]], lambda_grid: Optional[Sequence[float]]) -> Dict[str, Any]:
    values = _lambda_values(lambdas, lambda_grid)
    best = min(values, key=lambda value: abs(value - 0.5))
    return {
        "best_lambda": float(best),
        "scores": {float(value): float("nan") for value in values},
        "deletion_drop_auc": {float(value): float("nan") for value in values},
        "insertion_auc": {float(value): float("nan") for value in values},
        "lambdas": [float(value) for value in values],
        "n_validation": 0,
        "steps": 0,
        "baseline": "none",
        "metric": "dynamic_deletion_insertion",
        "status": "placeholder",
        "rows": [
            {
                "candidate_id": int(i),
                "lambda_hyb": float(value),
                "status": "placeholder",
                "score": float("nan"),
                "metric_source": "not_evaluated",
            }
            for i, value in enumerate(values)
        ],
        "details": {
            "status": "placeholder",
            "message": "Pass engine, validation data, and predict_proba to evaluate dynamic lambda candidates.",
        },
    }


def _canonical_steps(steps: int) -> int:
    value = int(steps)
    if value < 2:
        raise ValueError("steps must be at least 2")
    return value


def _pad_batch_to_fit(engine: DynamicFPDEEngine, batch: DynamicInputBatch) -> tuple[np.ndarray, np.ndarray]:
    if batch.representation.shape[2] != engine.prototypes_.shape[2]:
        raise ValueError(
            f"representation dimension mismatch: engine has {engine.prototypes_.shape[2]}, validation input has {batch.representation.shape[2]}"
        )
    if batch.representation.shape[1] > engine.prototypes_.shape[1]:
        raise ValueError(
            f"validation input length {batch.representation.shape[1]} exceeds fitted maximum length {engine.prototypes_.shape[1]}"
        )
    if batch.representation.shape[1] == engine.prototypes_.shape[1]:
        return batch.representation.copy(), batch.mask.copy()
    n_samples, t_current, n_features = batch.representation.shape
    t_fit = int(engine.prototypes_.shape[1])
    rep = np.zeros((n_samples, t_fit, n_features), dtype=float)
    mask = np.zeros((n_samples, t_fit), dtype=bool)
    rep[:, :t_current, :] = batch.representation
    mask[:, :t_current] = batch.mask
    return rep, mask


def _baseline_representation(engine: DynamicFPDEEngine, baseline: str | np.ndarray) -> tuple[np.ndarray, str]:
    if isinstance(baseline, str):
        if baseline == "mean":
            return engine.mean_anchor_.astype(float, copy=True), "mean"
        if baseline == "zero":
            return np.zeros_like(engine.mean_anchor_, dtype=float), "zero"
        raise ValueError("baseline must be 'mean', 'zero', or an ndarray")
    arr = np.asarray(baseline, dtype=float)
    if not np.all(np.isfinite(arr)):
        raise ValueError("baseline contains NaN or inf")
    if arr.ndim == 1:
        if arr.shape[0] != engine.prototypes_.shape[2]:
            raise ValueError(f"baseline feature dimension mismatch: expected {engine.prototypes_.shape[2]}, got {arr.shape[0]}")
        return np.broadcast_to(arr, engine.mean_anchor_.shape).astype(float, copy=True), "custom"
    if arr.shape != engine.mean_anchor_.shape:
        raise ValueError(f"baseline must have shape {engine.mean_anchor_.shape} or ({engine.prototypes_.shape[2]},), got {arr.shape}")
    return arr.astype(float, copy=True), "custom"


def _as_probability_matrix(proba: Any, *, expected_rows: int) -> np.ndarray:
    arr = np.asarray(proba, dtype=float)
    if arr.ndim == 1 and expected_rows == 1:
        arr = arr.reshape(1, -1)
    if arr.ndim != 2:
        raise ValueError(f"predict_proba must return a 2D array, got shape={arr.shape}")
    if arr.shape[0] != expected_rows:
        raise ValueError(f"predict_proba row count mismatch: expected {expected_rows}, got {arr.shape[0]}")
    if arr.shape[1] < 2:
        raise ValueError("predict_proba must contain at least two classes")
    if not np.all(np.isfinite(arr)):
        raise ValueError("predict_proba contains NaN or inf")
    return arr


def _call_predict_proba(
    engine: DynamicFPDEEngine,
    predict_proba: Callable[..., Any],
    representation: np.ndarray,
    mask: np.ndarray,
) -> np.ndarray:
    raw, features, dt = split_representation(representation, engine.feature_slices_)
    try:
        proba = predict_proba(raw=raw, features=features, dt=dt, mask=mask)
    except TypeError as keyword_error:
        try:
            proba = predict_proba(representation)
        except Exception as fallback_error:
            raise ValueError(
                "predict_proba callable must accept either keyword raw/features/dt/mask inputs "
                "or one representation argument"
            ) from fallback_error
        if proba is None:
            raise ValueError("predict_proba returned None") from keyword_error
    return _as_probability_matrix(proba, expected_rows=representation.shape[0])


def _validation_probabilities(
    engine: DynamicFPDEEngine,
    predict_proba: Any,
    representation: np.ndarray,
    mask: np.ndarray,
) -> tuple[np.ndarray, str]:
    if predict_proba is None:
        raise ValueError("predict_proba is required for dynamic lambda selection")
    if isinstance(predict_proba, np.ndarray):
        return _as_probability_matrix(predict_proba, expected_rows=representation.shape[0]), "precomputed"
    if callable(predict_proba):
        return _call_predict_proba(engine, predict_proba, representation, mask), "callable"
    raise ValueError("predict_proba must be a callable or a probability ndarray")


def _probabilities_for_perturbations(
    engine: DynamicFPDEEngine,
    predict_proba: Any,
    representation: np.ndarray,
    mask: np.ndarray,
    *,
    fallback_row: np.ndarray,
) -> np.ndarray:
    if isinstance(predict_proba, np.ndarray):
        return np.repeat(fallback_row.reshape(1, -1), representation.shape[0], axis=0)
    return _call_predict_proba(engine, predict_proba, representation, mask)


def _class_names(engine: DynamicFPDEEngine, class_names: Optional[Sequence[Any]], n_classes: int) -> np.ndarray:
    names = engine.prototype_labels_ if class_names is None else np.asarray(class_names, dtype=object)
    if names.shape[0] != n_classes:
        raise ValueError("class_names length must match predict_proba class dimension")
    return names


def _class_probability_index(names: np.ndarray, label: Any) -> int:
    matches = np.where(names == label)[0]
    if matches.size == 0:
        raise ValueError(f"target class {label!r} is not present in class_names")
    return int(matches[0])


def _resolve_selection_target_rival(
    engine: DynamicFPDEEngine,
    u: np.ndarray,
    mask: np.ndarray,
    *,
    target_class: Any,
    rival_class: Any,
    predict_proba: np.ndarray,
    class_names: np.ndarray,
) -> tuple[Any, Any]:
    if target_class is not None and rival_class is not None:
        if target_class == rival_class:
            raise ValueError("target_class and rival_class must differ")
        engine._prototype_index(target_class, "target_class")
        engine._prototype_index(rival_class, "rival_class")
        return target_class, rival_class
    if target_class is not None:
        return engine._resolve_target_rival(
            u,
            mask,
            target_class=target_class,
            rival_class=None,
            predict_proba=None,
            class_names=class_names,
        )
    if rival_class is not None:
        order = np.argsort(predict_proba)[::-1]
        for idx in order.tolist():
            candidate = class_names[int(idx)]
            if candidate != rival_class:
                engine._prototype_index(candidate, "target_class")
                engine._prototype_index(rival_class, "rival_class")
                return candidate, rival_class
        raise ValueError("could not resolve a target class distinct from rival_class")
    return engine._resolve_target_rival(
        u,
        mask,
        target_class=None,
        rival_class=None,
        predict_proba=predict_proba,
        class_names=class_names,
    )


def _auc(values: np.ndarray) -> float:
    if values.shape[0] == 1:
        return float(values[0])
    fractions = np.linspace(0.0, 1.0, values.shape[0])
    return float(np.trapezoid(values, fractions))


def _rank_coordinates(attr: np.ndarray, effective_mask: np.ndarray) -> np.ndarray:
    valid = np.broadcast_to(effective_mask[:, None], attr.shape).reshape(-1)
    valid_indices = np.flatnonzero(valid)
    if valid_indices.size == 0:
        return valid_indices
    scores = np.nan_to_num(attr.reshape(-1)[valid_indices], nan=-np.inf, posinf=-np.inf, neginf=-np.inf)
    order = np.argsort(scores)[::-1]
    return valid_indices[order]


def _perturbation_representations(
    original: np.ndarray,
    baseline: np.ndarray,
    order: np.ndarray,
    *,
    steps: int,
) -> tuple[np.ndarray, np.ndarray]:
    fractions = np.linspace(0.0, 1.0, steps)
    deletion = []
    insertion = []
    total = int(order.shape[0])
    flat_original = original.reshape(-1)
    flat_baseline = baseline.reshape(-1)
    for fraction in fractions:
        count = int(round(float(fraction) * total))
        chosen = order[:count]
        del_flat = flat_original.copy()
        ins_flat = flat_baseline.copy()
        if chosen.size:
            del_flat[chosen] = flat_baseline[chosen]
            ins_flat[chosen] = flat_original[chosen]
        deletion.append(del_flat.reshape(original.shape))
        insertion.append(ins_flat.reshape(original.shape))
    return np.stack(deletion, axis=0), np.stack(insertion, axis=0)


def select_lambda_dynamic(
    *,
    engine: Optional[DynamicFPDEEngine] = None,
    raw: Optional[np.ndarray | Sequence[np.ndarray | Sequence[Sequence[float]]]] = None,
    features: Optional[np.ndarray | Sequence[np.ndarray | Sequence[Sequence[float]]]] = None,
    dt: Optional[np.ndarray | Sequence[np.ndarray | Sequence[Sequence[float]]]] = None,
    mask: Optional[np.ndarray | Sequence[Sequence[bool]]] = None,
    predict_proba: Any = None,
    lambdas: Optional[Sequence[float]] = None,
    lambda_grid: Optional[Sequence[float]] = None,
    target_classes: Optional[Sequence[Any]] = None,
    rival_classes: Optional[Sequence[Any]] = None,
    class_names: Optional[Sequence[Any]] = None,
    baseline: str | np.ndarray = "mean",
    steps: int = 20,
) -> Dict[str, Any]:
    """Select ``lambda_hyb`` with coordinate-level deletion/insertion validation."""
    if engine is None:
        return _placeholder_lambda_selection(lambdas, lambda_grid)
    if raw is None:
        raise ValueError("raw validation data is required when engine is provided")
    engine._check_fitted()
    lambda_values = _lambda_values(lambdas, lambda_grid)
    step_count = _canonical_steps(steps)
    batch = validate_sequence_inputs(raw, features=features, dt=dt, mask=mask)
    representation, input_mask = _pad_batch_to_fit(engine, batch)
    n_validation = int(representation.shape[0])
    baseline_rep, baseline_name = _baseline_representation(engine, baseline)
    validation_proba, proba_source = _validation_probabilities(engine, predict_proba, representation, input_mask)
    names = _class_names(engine, class_names, validation_proba.shape[1])

    targets = [None] * n_validation if target_classes is None else list(target_classes)
    rivals = [None] * n_validation if rival_classes is None else list(rival_classes)
    if len(targets) != n_validation:
        raise ValueError(f"number of target_classes differs from validation samples: {len(targets)} vs {n_validation}")
    if len(rivals) != n_validation:
        raise ValueError(f"number of rival_classes differs from validation samples: {len(rivals)} vs {n_validation}")

    per_lambda_scores: Dict[float, list[float]] = {float(value): [] for value in lambda_values}
    per_lambda_deletion: Dict[float, list[float]] = {float(value): [] for value in lambda_values}
    per_lambda_insertion: Dict[float, list[float]] = {float(value): [] for value in lambda_values}
    per_sample_scores: list[Dict[str, Any]] = []
    resolved_targets = []
    resolved_rivals = []

    for sample_idx in range(n_validation):
        target, rival = _resolve_selection_target_rival(
            engine,
            representation[sample_idx],
            input_mask[sample_idx],
            target_class=targets[sample_idx],
            rival_class=rivals[sample_idx],
            predict_proba=validation_proba[sample_idx],
            class_names=names,
        )
        target_idx = engine._prototype_index(target, "target_class")
        rival_idx = engine._prototype_index(rival, "rival_class")
        target_column = _class_probability_index(names, target)
        resolved_targets.append(target)
        resolved_rivals.append(rival)

        sample_rows: Dict[str, Any] = {
            "sample_index": int(sample_idx),
            "target_class": target,
            "rival_class": rival,
            "scores": {},
            "deletion_drop_auc": {},
            "insertion_auc": {},
        }
        for lambda_value in lambda_values:
            old_lambda = engine.lambda_hyb
            engine.lambda_hyb = float(lambda_value)
            try:
                attr, details = engine._attributions(
                    representation[sample_idx],
                    input_mask[sample_idx],
                    target_idx,
                    rival_idx,
                    "hyb",
                )
            finally:
                engine.lambda_hyb = old_lambda
            effective_mask = details["effective_mask"]
            order = _rank_coordinates(attr, effective_mask)
            deletion_rep, insertion_rep = _perturbation_representations(
                representation[sample_idx],
                baseline_rep,
                order,
                steps=step_count,
            )
            deletion_rep[:, ~input_mask[sample_idx], :] = 0.0
            insertion_rep[:, ~input_mask[sample_idx], :] = 0.0
            repeated_mask = np.repeat(input_mask[sample_idx][None, :], step_count, axis=0)
            deletion_proba = _probabilities_for_perturbations(
                engine,
                predict_proba,
                deletion_rep,
                repeated_mask,
                fallback_row=validation_proba[sample_idx],
            )
            insertion_proba = _probabilities_for_perturbations(
                engine,
                predict_proba,
                insertion_rep,
                repeated_mask,
                fallback_row=validation_proba[sample_idx],
            )
            deletion_curve = deletion_proba[:, target_column]
            insertion_curve = insertion_proba[:, target_column]
            deletion_drop_auc = float(deletion_curve[0] - _auc(deletion_curve))
            insertion_auc = _auc(insertion_curve)
            combined = float(0.5 * (deletion_drop_auc + insertion_auc))
            key = float(lambda_value)
            per_lambda_scores[key].append(combined)
            per_lambda_deletion[key].append(deletion_drop_auc)
            per_lambda_insertion[key].append(insertion_auc)
            sample_rows["scores"][key] = combined
            sample_rows["deletion_drop_auc"][key] = deletion_drop_auc
            sample_rows["insertion_auc"][key] = insertion_auc
        per_sample_scores.append(sample_rows)

    scores = {key: float(np.mean(values)) for key, values in per_lambda_scores.items()}
    deletion_scores = {key: float(np.mean(values)) for key, values in per_lambda_deletion.items()}
    insertion_scores = {key: float(np.mean(values)) for key, values in per_lambda_insertion.items()}
    best_lambda = min(lambda_values, key=lambda value: (-scores[float(value)], float(value)))
    rows = [
        {
            "candidate_id": int(i),
            "lambda_hyb": float(value),
            "status": "evaluated",
            "score": scores[float(value)],
            "deletion_drop_auc": deletion_scores[float(value)],
            "insertion_auc": insertion_scores[float(value)],
            "metric_source": "dynamic_deletion_insertion",
        }
        for i, value in enumerate(lambda_values)
    ]
    return {
        "best_lambda": float(best_lambda),
        "scores": scores,
        "deletion_drop_auc": deletion_scores,
        "insertion_auc": insertion_scores,
        "lambdas": [float(value) for value in lambda_values],
        "rows": rows,
        "per_sample_scores": per_sample_scores,
        "target_classes": resolved_targets,
        "rival_classes": resolved_rivals,
        "n_validation": n_validation,
        "steps": step_count,
        "baseline": baseline_name,
        "metric": "dynamic_deletion_insertion",
        "status": "evaluated",
        "tie_break": "smallest_lambda",
        "predict_proba_source": proba_source,
        "granularity": "coordinate",
    }


__all__ = ["DynamicFPDEEngine", "select_lambda_dynamic", "split_representation"]
