"""Bar plot for FPDE contribution values."""

from __future__ import annotations

from typing import Any, Optional, Sequence, Tuple

import numpy as np

from ._common import (
    OrderMode,
    _as_2d_values,
    _axes,
    _feature_names,
    _feature_names_from_input,
    _finalize,
    _require_finite,
    _top_indices,
    _values_from_input,
)


def bar(
    explanation: Any = None,
    values: Any = None,
    feature_names: Optional[Sequence[Any]] = None,
    max_display: int = 10,
    order: OrderMode = "mean_abs",
    ax: Optional[Any] = None,
    show: bool = True,
    title: Optional[str] = None,
    figsize: Optional[Tuple[float, float]] = None,
) -> Any:
    """Plot local or global FPDE contribution importance as horizontal bars."""
    raw_values = _require_finite(_values_from_input(explanation, values), "values")
    matrix = _as_2d_values(raw_values)
    labels = _feature_names(_feature_names_from_input(explanation, feature_names), matrix.shape[1])
    selected = _top_indices(matrix, max_display, order)

    if matrix.shape[0] == 1 and order == "mean_abs":
        importance = np.abs(matrix[0])
    elif order == "mean_abs":
        importance = np.mean(np.abs(matrix), axis=0)
    elif order == "max_abs":
        importance = np.max(np.abs(matrix), axis=0)
    elif order == "mean":
        importance = np.mean(matrix, axis=0)
    elif order == "sum_abs":
        importance = np.sum(np.abs(matrix), axis=0)
    else:
        raise ValueError("order must be 'mean_abs', 'max_abs', 'mean', or 'sum_abs'")

    display_values = importance[selected]
    display_labels = labels[selected]

    ax = _axes(ax, figsize=figsize, default=(6.5, max(2.5, 0.34 * selected.shape[0] + 1.2)))
    y = np.arange(selected.shape[0])
    ax.barh(y, display_values, color="#64748b")
    ax.set_yticks(y)
    ax.set_yticklabels([str(label) for label in display_labels])
    ax.invert_yaxis()
    ax.set_xlabel("Mean absolute contribution" if matrix.shape[0] > 1 else "Absolute contribution")
    if title is not None:
        ax.set_title(title)
    return _finalize(ax, show)
