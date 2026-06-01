"""Feature-value scatter plot for FPDE contributions."""

from __future__ import annotations

import warnings
from typing import Any, Optional, Sequence, Tuple

import numpy as np

from ._common import (
    _as_2d_values,
    _axes,
    _data_from_input,
    _feature_names,
    _feature_names_from_input,
    _finalize,
    _safe_array,
    _values_from_input,
)


def _feature_index(feature: int | str | None, labels: np.ndarray, n_features: int, name: str) -> int:
    if feature is None:
        raise ValueError(f"{name} is required")
    if isinstance(feature, str):
        matches = np.where(labels == feature)[0]
        if matches.size == 0:
            raise ValueError(f"{name}={feature!r} is not in feature_names")
        return int(matches[0])
    idx = int(feature)
    if idx < 0 or idx >= n_features:
        raise ValueError(f"{name} index must be in [0, {n_features - 1}]")
    return idx


def scatter(
    explanation: Any = None,
    values: Any = None,
    data: Any = None,
    feature: int | str | None = None,
    feature_names: Optional[Sequence[Any]] = None,
    color_feature: int | str | None = None,
    alpha: float = 0.7,
    dot_size: float = 16,
    jitter: float = 0.0,
    ax: Optional[Any] = None,
    show: bool = True,
    title: Optional[str] = None,
    figsize: Optional[Tuple[float, float]] = None,
) -> Any:
    """Plot one feature's value against its FPDE contribution."""
    matrix = _as_2d_values(_values_from_input(explanation, values))
    labels = _feature_names(_feature_names_from_input(explanation, feature_names), matrix.shape[1])
    feature_idx = _feature_index(feature, labels, matrix.shape[1], "feature")

    data_arr = _data_from_input(explanation, data)
    if data_arr is not None:
        data_arr = _safe_array(data_arr, "data")
        if data_arr.shape != matrix.shape:
            raise ValueError(f"data shape must match values: {data_arr.shape} vs {matrix.shape}")
        x = data_arr[:, feature_idx].astype(float, copy=True)
        xlabel = str(labels[feature_idx])
    else:
        warnings.warn("data is not available; using sample index on the x-axis", RuntimeWarning, stacklevel=2)
        x = np.arange(matrix.shape[0], dtype=float)
        xlabel = "Sample index"

    y = matrix[:, feature_idx]
    color_values = None
    if color_feature is not None:
        if data_arr is None:
            raise ValueError("color_feature requires data")
        color_idx = _feature_index(color_feature, labels, matrix.shape[1], "color_feature")
        color_values = data_arr[:, color_idx]

    jitter_value = float(jitter)
    if not np.isfinite(jitter_value) or jitter_value < 0.0:
        raise ValueError("jitter must be a non-negative finite value")
    if jitter_value > 0.0:
        x = x + np.random.default_rng(0).uniform(-jitter_value, jitter_value, size=x.shape[0])

    mask = np.isfinite(x) & np.isfinite(y)
    if color_values is not None:
        mask &= np.isfinite(color_values)

    ax = _axes(ax, figsize=figsize, default=(5.8, 3.8))
    if color_values is None:
        ax.scatter(x[mask], y[mask], s=dot_size, color="#64748b", alpha=alpha, edgecolors="none")
    else:
        scatter_obj = ax.scatter(
            x[mask],
            y[mask],
            s=dot_size,
            c=color_values[mask],
            cmap="coolwarm",
            alpha=alpha,
            edgecolors="none",
        )
        ax.figure.colorbar(scatter_obj, ax=ax, fraction=0.046, pad=0.04, label=str(labels[color_idx]))
    ax.axhline(0.0, color="#111827", linewidth=0.8)
    ax.set_xlabel(xlabel)
    ax.set_ylabel(f"{labels[feature_idx]} contribution")
    if title is not None:
        ax.set_title(title)
    return _finalize(ax, show)
