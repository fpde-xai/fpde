"""Waterfall plot for one FPDE contribution vector."""

from __future__ import annotations

import warnings
from typing import Any, Optional, Sequence, Tuple

import numpy as np

from ._common import (
    _as_single_values,
    _axes,
    _base_value_from_input,
    _feature_names,
    _feature_names_from_input,
    _finalize,
    _prediction_from_input,
    _require_finite,
    _top_indices,
    _values_from_input,
)


def waterfall(
    explanation: Any = None,
    values: Any = None,
    base_value: Any = None,
    prediction: Any = None,
    data: Any = None,
    feature_names: Optional[Sequence[Any]] = None,
    max_display: int = 10,
    ax: Optional[Any] = None,
    show: bool = True,
    title: Optional[str] = None,
    figsize: Optional[Tuple[float, float]] = None,
) -> Any:
    """Plot a single FPDE explanation from baseline to prediction."""
    data  # reserved for a future label-with-feature-value enhancement
    vector = _require_finite(_as_single_values(_values_from_input(explanation, values)), "values")
    labels = _feature_names(_feature_names_from_input(explanation, feature_names), vector.shape[0])
    base = _base_value_from_input(explanation, base_value)
    if base is None:
        base = 0.0
    predicted = _prediction_from_input(explanation, prediction)
    additive_prediction = float(base + np.sum(vector))
    if predicted is None:
        predicted = additive_prediction

    tolerance = 1e-6 + 1e-6 * abs(float(predicted))
    if abs(float(predicted) - additive_prediction) > tolerance:
        warnings.warn(
            "prediction differs from base_value + sum(values) beyond tolerance",
            RuntimeWarning,
            stacklevel=2,
        )

    selected = _top_indices(vector, max_display, "mean_abs")
    selected_mask = np.zeros(vector.shape[0], dtype=bool)
    selected_mask[selected] = True
    shown_values = vector[selected].astype(float, copy=True)
    shown_labels = labels[selected].astype(object, copy=True)
    if np.any(~selected_mask):
        shown_values = np.concatenate([shown_values, [float(np.sum(vector[~selected_mask]))]])
        shown_labels = np.concatenate([shown_labels, ["other features"]])

    starts = float(base) + np.concatenate([[0.0], np.cumsum(shown_values[:-1])])
    colors = ["#2563eb" if value >= 0.0 else "#dc2626" for value in shown_values]

    ax = _axes(ax, figsize=figsize, default=(7.5, max(2.8, 0.34 * shown_values.shape[0] + 1.4)))
    y = np.arange(shown_values.shape[0])
    ax.barh(y, shown_values, left=starts, color=colors)
    ax.axvline(float(base), color="#6b7280", linestyle="--", linewidth=0.9, label="base")
    ax.axvline(float(predicted), color="#111827", linewidth=1.0, label="prediction")
    ax.set_yticks(y)
    ax.set_yticklabels([str(label) for label in shown_labels])
    ax.invert_yaxis()
    ax.set_xlabel("Model output")
    ax.legend()
    ax.text(
        0.99,
        0.02,
        f"base={float(base):.4g}; prediction={float(predicted):.4g}; sum={additive_prediction:.4g}",
        transform=ax.transAxes,
        ha="right",
        va="bottom",
        fontsize=8,
        color="#374151",
    )
    if title is not None:
        ax.set_title(title)
    return _finalize(ax, show)
