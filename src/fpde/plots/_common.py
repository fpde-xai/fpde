"""Shared helpers for the high-level FPDE plotting API."""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any, Literal, Optional, Sequence, Tuple

import numpy as np

OrderMode = Literal["mean_abs", "max_abs", "mean", "sum_abs"]


@dataclass(frozen=True)
class FPDEPlotExplanation:
    """Lightweight plotting container for FPDE contribution arrays."""

    values: np.ndarray
    base_values: Optional[np.ndarray | float] = None
    data: Optional[np.ndarray] = None
    feature_names: Optional[Sequence[str]] = None
    output_names: Optional[Sequence[str]] = None
    predictions: Optional[np.ndarray | float] = None


def _pyplot() -> Any:
    try:
        import matplotlib.pyplot as plt
    except ImportError as exc:
        raise ImportError(
            "FPDE plots require matplotlib. Install it with "
            "`python -m pip install fpde[plot]`."
        ) from exc
    return plt


def _safe_array(x: Any, name: str) -> np.ndarray:
    try:
        return np.asarray(x, dtype=float)
    except (TypeError, ValueError) as exc:
        raise ValueError(f"{name} must be convertible to a numeric array") from exc


def _values_from_input(explanation: Any = None, values: Any = None) -> np.ndarray:
    if explanation is not None and values is not None:
        raise ValueError("pass either explanation or values, not both")
    if values is not None:
        return _safe_array(values, "values")
    if explanation is None:
        raise ValueError("either explanation or values is required")
    if hasattr(explanation, "values"):
        return _safe_array(getattr(explanation, "values"), "explanation.values")
    if hasattr(explanation, "attributions"):
        return _safe_array(getattr(explanation, "attributions"), "explanation.attributions")
    raise ValueError("explanation must expose values or attributions")


def _data_from_input(explanation: Any = None, data: Any = None) -> Optional[np.ndarray]:
    if data is not None:
        return _safe_array(data, "data")
    if explanation is not None and hasattr(explanation, "data"):
        raw = getattr(explanation, "data")
        if raw is not None:
            return _safe_array(raw, "explanation.data")
    return None


def _feature_names_from_input(
    explanation: Any = None,
    feature_names: Optional[Sequence[Any]] = None,
) -> Optional[Sequence[Any]]:
    if feature_names is not None:
        return feature_names
    if explanation is not None and hasattr(explanation, "feature_names"):
        return getattr(explanation, "feature_names")
    return None


def _scalar_from_input(value: Any, name: str) -> Optional[float]:
    if value is None:
        return None
    arr = _safe_array(value, name)
    if arr.ndim == 0:
        result = float(arr)
    elif arr.size == 1:
        result = float(arr.reshape(-1)[0])
    else:
        raise ValueError(f"{name} must be a scalar or length-one array")
    if not np.isfinite(result):
        raise ValueError(f"{name} must be finite")
    return result


def _base_value_from_input(explanation: Any = None, base_value: Any = None) -> Optional[float]:
    if base_value is not None:
        return _scalar_from_input(base_value, "base_value")
    if explanation is not None and hasattr(explanation, "base_values"):
        return _scalar_from_input(getattr(explanation, "base_values"), "explanation.base_values")
    return None


def _prediction_from_input(explanation: Any = None, prediction: Any = None) -> Optional[float]:
    if prediction is not None:
        return _scalar_from_input(prediction, "prediction")
    if explanation is not None and hasattr(explanation, "predictions"):
        return _scalar_from_input(getattr(explanation, "predictions"), "explanation.predictions")
    return None


def _as_2d_values(values: Any) -> np.ndarray:
    arr = _safe_array(values, "values")
    if arr.ndim == 1:
        if arr.shape[0] == 0:
            raise ValueError("values must contain at least one feature")
        return arr.reshape(1, -1)
    if arr.ndim == 2:
        if arr.shape[0] == 0:
            raise ValueError("values must contain at least one sample")
        if arr.shape[1] == 0:
            raise ValueError("values must contain at least one feature")
        return arr
    raise ValueError(f"values must be 1D or 2D, got shape={arr.shape}")


def _as_single_values(values: Any) -> np.ndarray:
    arr = _safe_array(values, "values")
    if arr.ndim == 1:
        if arr.shape[0] == 0:
            raise ValueError("values must contain at least one feature")
        return arr
    if arr.ndim == 2 and arr.shape[0] == 1:
        if arr.shape[1] == 0:
            raise ValueError("values must contain at least one feature")
        return arr[0]
    if arr.ndim == 2:
        raise ValueError("waterfall accepts a single sample; got multiple rows")
    raise ValueError(f"values must be 1D or a single-row 2D array, got shape={arr.shape}")


def _require_finite(values: np.ndarray, name: str) -> np.ndarray:
    if not np.all(np.isfinite(values)):
        raise ValueError(f"{name} contains NaN or inf")
    return values


def _feature_names(feature_names: Optional[Sequence[Any]], n_features: int) -> np.ndarray:
    if n_features < 1:
        raise ValueError("values must contain at least one feature")
    if feature_names is None:
        return np.asarray([f"f{i}" for i in range(n_features)], dtype=object)
    labels = np.asarray(list(feature_names), dtype=object)
    if labels.ndim != 1:
        raise ValueError(f"feature_names must be a 1D sequence, got shape={labels.shape}")
    if labels.shape[0] != n_features:
        raise ValueError(f"feature_names length must match values: {labels.shape[0]} vs {n_features}")
    return labels


def _top_indices(values: np.ndarray, max_display: int, order: OrderMode = "mean_abs") -> np.ndarray:
    max_count = int(max_display)
    if max_count < 1:
        raise ValueError("max_display must be positive")
    if values.ndim == 1:
        matrix = values.reshape(1, -1)
    elif values.ndim == 2:
        matrix = values
    else:
        raise ValueError(f"values must be 1D or 2D, got shape={values.shape}")
    if matrix.shape[1] < 1:
        raise ValueError("values must contain at least one feature")

    if order == "mean_abs":
        scores = np.mean(np.abs(matrix), axis=0)
    elif order == "max_abs":
        scores = np.max(np.abs(matrix), axis=0)
    elif order == "mean":
        scores = np.mean(matrix, axis=0)
    elif order == "sum_abs":
        scores = np.sum(np.abs(matrix), axis=0)
    else:
        raise ValueError("order must be 'mean_abs', 'max_abs', 'mean', or 'sum_abs'")

    selected = np.argsort(scores)[-min(max_count, scores.shape[0]) :]
    return selected[np.argsort(scores[selected])[::-1]]


def _axes(ax: Optional[Any], *, figsize: Optional[Tuple[float, float]], default: Tuple[float, float]) -> Any:
    if ax is not None:
        return ax
    _, new_ax = _pyplot().subplots(figsize=figsize or default)
    return new_ax


def _finalize(ax: Any, show: bool) -> Any:
    if show:
        _pyplot().show()
    return ax


def _finite_row_mask(*arrays: Optional[np.ndarray]) -> np.ndarray:
    first = next((arr for arr in arrays if arr is not None), None)
    if first is None:
        raise ValueError("at least one array is required")
    mask = np.ones(first.shape[0], dtype=bool)
    for arr in arrays:
        if arr is None:
            continue
        if arr.shape[0] != first.shape[0]:
            raise ValueError(f"array row count mismatch: {arr.shape[0]} vs {first.shape[0]}")
        if arr.ndim == 1:
            mask &= np.isfinite(arr)
        else:
            mask &= np.all(np.isfinite(arr), axis=tuple(range(1, arr.ndim)))
    return mask
