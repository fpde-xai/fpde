"""Low-level array, probability, and grid helpers for FPDE."""

from __future__ import annotations

from typing import Any, Dict, Optional, Sequence, Tuple

import numpy as np

from .types import FPDEExplanation, NormalizeMode

def _as_1d_float(name: str, x: np.ndarray | Sequence[float]) -> np.ndarray:
    arr = np.asarray(x, dtype=float)
    if arr.ndim != 1:
        raise ValueError(f"{name} must be a 1D vector, got shape={arr.shape}")
    if not np.all(np.isfinite(arr)):
        raise ValueError(f"{name} contains NaN or inf")
    return arr


def _as_2d_float(name: str, x: np.ndarray | Sequence[Sequence[float]]) -> np.ndarray:
    arr = np.asarray(x, dtype=float)
    if arr.ndim != 2:
        raise ValueError(f"{name} must be a 2D array, got shape={arr.shape}")
    if not np.all(np.isfinite(arr)):
        raise ValueError(f"{name} contains NaN or inf")
    return arr


def _as_label_array(labels: Sequence[Any]) -> np.ndarray:
    """Return labels as a 1D array without list-copying ndarray inputs."""
    if isinstance(labels, np.ndarray):
        arr = np.asarray(labels)
    else:
        arr = np.asarray(list(labels), dtype=object)
    if arr.ndim != 1:
        raise ValueError(f"labels must be a 1D sequence, got shape={arr.shape}")
    return arr


def _check_same_dim(*vectors: np.ndarray) -> None:
    if not vectors:
        return
    d = vectors[0].shape[0]
    for i, v in enumerate(vectors):
        if v.shape[0] != d:
            raise ValueError(f"dimension mismatch at vector {i}: expected {d}, got {v.shape[0]}")


def _regularized_norm(v: np.ndarray, eps: float) -> float:
    return float(np.sqrt(np.dot(v, v) + eps * eps))


def _l1_denom(v: np.ndarray, eps: float) -> float:
    denom = float(np.sum(np.abs(v)))
    if denom <= eps or not np.isfinite(denom):
        return 0.0
    return denom


def _scaled_explanation_parts(
    exp: FPDEExplanation,
    *,
    normalize: NormalizeMode,
    eps: float,
) -> Tuple[np.ndarray, float, float, float, float]:
    """Return scaled attribution/evidence/scores/scale for hybridization."""
    if normalize == "none":
        return (
            exp.attributions.astype(float, copy=True),
            float(exp.evidence),
            float(exp.positive_score),
            float(exp.negative_score),
            1.0,
        )

    if normalize == "l1":
        scale = _l1_denom(exp.attributions, eps=eps)
        if scale == 0.0:
            zeros = np.zeros_like(exp.attributions, dtype=float)
            return zeros, 0.0, 0.0, 0.0, 0.0
        return (
            exp.attributions / scale,
            float(exp.evidence / scale),
            float(exp.positive_score / scale),
            float(exp.negative_score / scale),
            scale,
        )

    raise ValueError(f"unknown normalize={normalize!r}; expected 'none' or 'l1'")


def _get_predict_proba_vector(
    predictor: Optional[Any],
    x: np.ndarray,
) -> Tuple[Optional[np.ndarray], Optional[np.ndarray]]:
    """Evaluate a predict_proba-like object or callable for one sample."""
    if predictor is None:
        return None, None

    if hasattr(predictor, "predict_proba"):
        raw = predictor.predict_proba(x.reshape(1, -1))
        labels = getattr(predictor, "classes_", None)
    elif callable(predictor):
        raw = predictor(x.reshape(1, -1))
        labels = getattr(predictor, "classes_", None)
    else:
        raise ValueError("predictor must be None, a predict_proba callable, or an object with predict_proba")

    arr = np.asarray(raw, dtype=float)
    if arr.ndim == 2:
        if arr.shape[0] != 1:
            raise ValueError(f"predict_proba for one sample must have one row, got shape={arr.shape}")
        arr = arr[0]
    elif arr.ndim != 1:
        raise ValueError(f"predict_proba output must be 1D or 2D, got shape={arr.shape}")
    if arr.size == 0 or not np.all(np.isfinite(arr)):
        raise ValueError("predict_proba output is empty or contains NaN/inf")

    if labels is None:
        return arr.astype(float, copy=False), None
    labels_arr = np.asarray(labels, dtype=object)
    if labels_arr.shape[0] != arr.shape[0]:
        raise ValueError(
            f"predictor.classes_ length and predict_proba width differ: {labels_arr.shape[0]} vs {arr.shape[0]}"
        )
    return arr.astype(float, copy=False), labels_arr


def _probability_for_label(
    label: Any,
    probabilities: Optional[np.ndarray],
    probability_labels: Optional[np.ndarray],
    fallback_labels: np.ndarray,
) -> Optional[float]:
    if probabilities is None:
        return None
    labels = probability_labels if probability_labels is not None else fallback_labels
    if labels.shape[0] != probabilities.shape[0]:
        return None
    matches = np.where(labels == label)[0]
    if matches.size == 0:
        return None
    return float(probabilities[int(matches[0])])


def _default_lambda_hyb_grid(step: float = 0.1) -> Tuple[float, ...]:
    """Return a deterministic inclusive lambda_hyb grid from 0.0 to 1.0."""
    step = float(step)
    if not np.isfinite(step) or step <= 0.0 or step > 1.0:
        raise ValueError("step must be finite and in (0, 1]")
    n_steps = int(round(1.0 / step))
    values = [i * step for i in range(n_steps + 1)]
    values[0] = 0.0
    values[-1] = 1.0
    return tuple(float(round(v, 12)) for v in values)


def _predictor_probability_context(
    predictor: Optional[Any],
    X: np.ndarray,
    classes: np.ndarray,
) -> Tuple[Optional[np.ndarray], Optional[np.ndarray]]:
    """Evaluate a predictor for many samples when a batched call is available."""
    if predictor is None:
        return None, None

    if hasattr(predictor, "predict_proba"):
        raw = predictor.predict_proba(X)
        labels = getattr(predictor, "classes_", None)
    elif callable(predictor):
        raw = predictor(X)
        labels = getattr(predictor, "classes_", None)
    else:
        raise ValueError("predictor must be None, a predict_proba callable, or an object with predict_proba")

    probabilities = np.asarray(raw, dtype=float)
    if probabilities.ndim != 2 or probabilities.shape[0] != X.shape[0]:
        raise ValueError(f"batched predict_proba output must have shape (n_samples, n_classes), got {probabilities.shape}")
    if probabilities.shape[1] == 0 or not np.all(np.isfinite(probabilities)):
        raise ValueError("predict_proba output is empty or contains NaN/inf")

    if labels is None:
        if classes.shape[0] != probabilities.shape[1]:
            raise ValueError("cannot infer labels: class count differs from predict_proba width")
        return probabilities.astype(float, copy=False), classes

    labels_arr = np.asarray(labels, dtype=object)
    if labels_arr.shape[0] != probabilities.shape[1]:
        raise ValueError(
            f"predictor.classes_ length and predict_proba width differ: {labels_arr.shape[0]} vs {probabilities.shape[1]}"
        )
    return probabilities.astype(float, copy=False), labels_arr


def _top_two_probability_columns(
    probabilities: np.ndarray,
    *,
    available: Optional[np.ndarray] = None,
) -> Tuple[np.ndarray, np.ndarray]:
    """Return highest and rival probability columns without a full row sort."""
    n, k = probabilities.shape
    if k < 2:
        raise ValueError("at least two probability columns are required")

    if available is None or bool(np.all(available)):
        if k == 2:
            pos_cols = (probabilities[:, 1] >= probabilities[:, 0]).astype(np.intp, copy=False)
            rival_cols = 1 - pos_cols
            return pos_cols, rival_cols.astype(np.intp, copy=False)

        pair = np.argpartition(probabilities, kth=k - 2, axis=1)[:, -2:]
        pair_scores = np.take_along_axis(probabilities, pair, axis=1)
        first_is_best = pair_scores[:, 0] >= pair_scores[:, 1]
        pos_cols = np.where(first_is_best, pair[:, 0], pair[:, 1]).astype(np.intp, copy=False)
        rival_cols = np.where(first_is_best, pair[:, 1], pair[:, 0]).astype(np.intp, copy=False)
        return pos_cols, rival_cols

    scores = np.where(available[None, :], probabilities, -np.inf)
    pos_cols = np.argmax(scores, axis=1).astype(np.intp, copy=False)
    scores[np.arange(n), pos_cols] = -np.inf
    rival_cols = np.argmax(scores, axis=1).astype(np.intp, copy=False)
    if not np.all(np.isfinite(scores[np.arange(n), rival_cols])):
        raise ValueError("could not find two available probability labels")
    return pos_cols, rival_cols


def _top_two_probability_labels(
    probabilities: np.ndarray,
    probability_labels: np.ndarray,
    prototype_labels: np.ndarray,
) -> Tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
    """Choose target/rival labels for a batched probability matrix."""
    pos_cols, fallback_rival_cols = _top_two_probability_columns(probabilities)
    available = np.isin(probability_labels, prototype_labels)
    pos_labels = probability_labels[pos_cols]
    if bool(np.all(available)):
        rival_cols = fallback_rival_cols
    else:
        rival_scores = np.where(available[None, :], probabilities, -np.inf)
        rival_scores[np.arange(probabilities.shape[0]), pos_cols] = -np.inf
        rival_cols = np.argmax(rival_scores, axis=1).astype(np.intp, copy=False)
        if not np.all(np.isfinite(rival_scores[np.arange(probabilities.shape[0]), rival_cols])):
            raise ValueError("could not find a rival probability label with a prototype")
    neg_labels = probability_labels[rival_cols]
    row_idx = np.arange(probabilities.shape[0])
    return (
        pos_labels,
        neg_labels,
        probabilities[row_idx, pos_cols].astype(float, copy=False),
        probabilities[row_idx, rival_cols].astype(float, copy=False),
    )


def parse_float_grid(text_or_values: str | Sequence[float]) -> Tuple[float, ...]:
    """Parse a comma-separated grid string or numeric sequence."""
    if isinstance(text_or_values, str):
        values = []
        for part in text_or_values.split(","):
            part = part.strip()
            if part:
                values.append(float(part))
    else:
        values = [float(v) for v in text_or_values]

    if not values:
        raise ValueError("grid must contain at least one value")
    for value in values:
        if not np.isfinite(value):
            raise ValueError(f"non-finite grid value: {value!r}")
    return tuple(values)


def _safe_auc(x: np.ndarray, y: np.ndarray) -> float:
    """Trapezoidal AUC compatible with NumPy 1.x and 2.x."""
    try:
        return float(np.trapezoid(y, x))
    except AttributeError:  # NumPy < 2.0
        return float(np.trapz(y, x))


def _canonical_fraction_array(fractions: Sequence[float]) -> np.ndarray:
    """Validate and canonicalize perturbation fractions once."""
    frac_arr = np.asarray(sorted(set(float(f) for f in fractions)), dtype=float)
    if frac_arr.size == 0:
        raise ValueError("fractions must contain at least one value")
    if not np.all(np.isfinite(frac_arr)):
        raise ValueError("fractions contains NaN or inf")
    if np.any(frac_arr < 0.0) or np.any(frac_arr > 1.0):
        raise ValueError("fractions must be in [0, 1]")
    if frac_arr[0] > 0.0:
        frac_arr = np.concatenate([[0.0], frac_arr])
    if frac_arr[-1] < 1.0:
        frac_arr = np.concatenate([frac_arr, [1.0]])
    return np.clip(frac_arr, 0.0, 1.0)


def _rounded_feature_counts(frac_arr: np.ndarray, n_features: int) -> Tuple[np.ndarray, np.ndarray]:
    """Return unique rounded feature counts and inverse indices for fractions."""
    return np.unique(np.rint(frac_arr * n_features).astype(np.intp), return_inverse=True)


def _predict_proba_matrix(model: Any, X: np.ndarray) -> np.ndarray:
    """Return a finite 2D predict_proba matrix."""
    if not hasattr(model, "predict_proba"):
        raise ValueError("model must implement predict_proba")
    proba = np.asarray(model.predict_proba(X), dtype=float)
    if proba.ndim != 2:
        raise ValueError(f"predict_proba must return a 2D array, got shape={proba.shape}")
    if not np.all(np.isfinite(proba)):
        raise ValueError("predict_proba contains NaN or inf")
    return proba


def _indices_for_labels(all_labels: np.ndarray, query_labels: Sequence[Any]) -> np.ndarray:
    """Map labels to integer indices using NumPy equality semantics."""
    labels_list = all_labels.tolist()
    query_list = list(query_labels)
    try:
        label_to_idx: Dict[Any, int] = {}
        for i, label in enumerate(labels_list):
            label_to_idx.setdefault(label, i)
        return np.fromiter((label_to_idx[label] for label in query_list), dtype=np.intp, count=len(query_list))
    except TypeError:
        pass
    except KeyError as exc:
        raise ValueError(f"label {exc.args[0]!r} is not available") from None

    out = np.empty(len(query_list), dtype=np.intp)
    for i, label in enumerate(query_list):
        matches = np.where(all_labels == label)[0]
        if matches.size == 0:
            raise ValueError(f"label {label!r} is not available")
        out[i] = int(matches[0])
    return out

def _class_to_probability_index(model: Any) -> Dict[Any, int]:
    if not hasattr(model, "classes_"):
        raise ValueError("model must expose classes_ when using predict_proba")
    return {c: i for i, c in enumerate(np.asarray(model.classes_, dtype=object).tolist())}


__all__ = [
    "parse_float_grid",
]
