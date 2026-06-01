"""Distribution summary plot for FPDE contribution values."""

from __future__ import annotations

from typing import Any, Optional, Sequence, Tuple

import numpy as np

from ._common import (
    OrderMode,
    _as_2d_values,
    _axes,
    _data_from_input,
    _feature_names,
    _feature_names_from_input,
    _finalize,
    _safe_array,
    _top_indices,
    _values_from_input,
)


def beeswarm(
    explanation: Any = None,
    values: Any = None,
    data: Any = None,
    feature_names: Optional[Sequence[Any]] = None,
    max_display: int = 10,
    order: OrderMode = "mean_abs",
    color_by_value: bool = True,
    alpha: float = 0.7,
    dot_size: float = 16,
    jitter: float = 0.25,
    ax: Optional[Any] = None,
    show: bool = True,
    title: Optional[str] = None,
    figsize: Optional[Tuple[float, float]] = None,
) -> Any:
    """Plot per-sample FPDE contributions grouped by feature."""
    matrix = _as_2d_values(_values_from_input(explanation, values))
    labels = _feature_names(_feature_names_from_input(explanation, feature_names), matrix.shape[1])
    selected = _top_indices(np.nan_to_num(matrix, nan=0.0, posinf=0.0, neginf=0.0), max_display, order)

    data_arr = _data_from_input(explanation, data)
    if data_arr is not None:
        data_arr = _safe_array(data_arr, "data")
        if data_arr.shape != matrix.shape:
            raise ValueError(f"data shape must match values: {data_arr.shape} vs {matrix.shape}")

    jitter_value = float(jitter)
    if not np.isfinite(jitter_value) or jitter_value < 0.0:
        raise ValueError("jitter must be a non-negative finite value")
    rng = np.random.default_rng(0)

    ax = _axes(ax, figsize=figsize, default=(7.0, max(2.8, 0.34 * selected.shape[0] + 1.4)))
    last_scatter = None
    for row, feature_idx in enumerate(selected):
        x = matrix[:, feature_idx]
        y = np.full(matrix.shape[0], row, dtype=float)
        color_values = data_arr[:, feature_idx] if data_arr is not None and color_by_value else None
        mask = np.isfinite(x)
        if color_values is not None:
            mask &= np.isfinite(color_values)
        if not np.any(mask):
            continue
        if jitter_value > 0.0:
            y += rng.uniform(-jitter_value, jitter_value, size=matrix.shape[0])
        if color_values is None:
            last_scatter = ax.scatter(x[mask], y[mask], s=dot_size, color="#64748b", alpha=alpha, edgecolors="none")
        else:
            last_scatter = ax.scatter(
                x[mask],
                y[mask],
                s=dot_size,
                c=color_values[mask],
                cmap="coolwarm",
                alpha=alpha,
                edgecolors="none",
            )

    ax.axvline(0.0, color="#111827", linewidth=0.8)
    ax.set_yticks(np.arange(selected.shape[0]))
    ax.set_yticklabels([str(label) for label in labels[selected]])
    ax.invert_yaxis()
    ax.set_xlabel("Contribution")
    if title is not None:
        ax.set_title(title)
    if data_arr is not None and color_by_value and last_scatter is not None:
        ax.figure.colorbar(last_scatter, ax=ax, fraction=0.046, pad=0.04, label="Feature value")
    return _finalize(ax, show)
