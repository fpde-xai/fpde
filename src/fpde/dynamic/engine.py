"""Stateful RawFeat Dynamic-FPDE engine."""

from __future__ import annotations

from typing import Any, Dict, Optional, Sequence

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
        valid = mask[:, None]
        distances = np.sum(np.where(valid, (u[None, :, :] - self.prototypes_) ** 2, 0.0), axis=(1, 2))
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

    def _attributions(self, u: np.ndarray, mask: np.ndarray, target_idx: int, rival_idx: int, method: DynamicMethod) -> tuple[np.ndarray, Dict[str, Any]]:
        target = self.prototypes_[target_idx]
        rival = self.prototypes_[rival_idx]
        valid = mask[:, None].astype(float)

        diff = valid * ((u - rival) ** 2 - (u - target) ** 2)
        z = u - self.mean_anchor_
        q_target = target - self.mean_anchor_
        q_rival = rival - self.mean_anchor_
        z_norm = np.sqrt(np.sum(z * z, axis=1, keepdims=True) + self.eps**2)
        target_norm = np.sqrt(np.sum(q_target * q_target, axis=1, keepdims=True) + self.eps**2)
        rival_norm = np.sqrt(np.sum(q_rival * q_rival, axis=1, keepdims=True) + self.eps**2)
        cos = valid * ((z * q_target) / (z_norm * target_norm) - (z * q_rival) / (z_norm * rival_norm))

        if method == "diff":
            return diff, {"diff_evidence": float(np.sum(diff)), "cos_evidence": float(np.sum(cos))}
        if method == "cos":
            return cos, {"diff_evidence": float(np.sum(diff)), "cos_evidence": float(np.sum(cos))}

        diff_part = _l1_or_zero(diff, self.eps)
        cos_part = _l1_or_zero(cos, self.eps)
        attr = self.lambda_hyb * diff_part + (1.0 - self.lambda_hyb) * cos_part
        return attr, {
            "diff_evidence": float(np.sum(diff)),
            "cos_evidence": float(np.sum(cos)),
            "lambda_hyb": float(self.lambda_hyb),
            "diff_l1": float(np.sum(np.abs(diff))),
            "cos_l1": float(np.sum(np.abs(cos))),
        }

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
        attr, _ = self._attributions(u, mask_one, target_idx, rival_idx, method_value)
        attr[~mask_one] = 0.0
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
        if proba_arr is not None and proba_arr.shape[0] != n:
            raise ValueError(f"predict_proba first dimension must match raw samples: {proba_arr.shape[0]} vs {n}")

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


def select_lambda_dynamic(*args: Any, **kwargs: Any) -> Dict[str, Any]:
    """Placeholder hook for future RawFeat Dynamic-Hyb validation selection."""
    lambda_grid = kwargs.get("lambda_grid", None)
    values = [0.5] if lambda_grid is None else [float(value) for value in lambda_grid]
    if not values:
        raise ValueError("lambda_grid must contain at least one value")
    lambdas = [_validate_lambda_hyb(value) for value in values]
    best = min(lambdas, key=lambda value: abs(value - 0.5))
    return {
        "best_lambda": float(best),
        "rows": [
            {
                "candidate_id": int(i),
                "lambda_hyb": float(value),
                "status": "not_evaluated",
                "score": float("nan"),
                "metric_source": "placeholder",
            }
            for i, value in enumerate(lambdas)
        ],
        "details": {
            "status": "placeholder",
            "message": "RawFeat Dynamic-Hyb validation selection is reserved for a future release.",
        },
    }


__all__ = ["DynamicFPDEEngine", "select_lambda_dynamic"]
