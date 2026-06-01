from __future__ import annotations

import numpy as np
import pytest

matplotlib = pytest.importorskip("matplotlib")

matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402

from fpde import diff_fpde  # noqa: E402
from fpde.plots import FPDEPlotExplanation, bar, beeswarm, scatter, waterfall  # noqa: E402


def _values() -> np.ndarray:
    return np.array(
        [
            [0.2, -0.1, 0.05],
            [0.1, -0.2, 0.03],
            [-0.05, 0.3, -0.02],
        ],
        dtype=float,
    )


def _data() -> np.ndarray:
    return np.array(
        [
            [1.0, 2.0, 3.0],
            [1.5, 2.5, 3.5],
            [0.5, 1.5, 2.5],
        ],
        dtype=float,
    )


def test_plots_public_imports():
    assert FPDEPlotExplanation(values=_values()).values.shape == (3, 3)
    assert callable(bar)
    assert callable(beeswarm)
    assert callable(waterfall)
    assert callable(scatter)


def test_show_false_does_not_call_pyplot_show(monkeypatch):
    called = False

    def fake_show():
        nonlocal called
        called = True

    monkeypatch.setattr(plt, "show", fake_show)
    ax = bar(values=_values(), show=False)

    assert ax.__class__.__name__ == "Axes"
    assert not called
    plt.close(ax.figure)


def test_bar_accepts_1d_and_2d_values():
    ax_1d = bar(values=[0.2, -0.1, 0.05], feature_names=["a", "b", "c"], show=False)
    ax_2d = bar(values=_values(), feature_names=["a", "b", "c"], show=False)

    assert ax_1d.__class__.__name__ == "Axes"
    assert ax_2d.__class__.__name__ == "Axes"
    assert len(ax_1d.patches) == 3
    assert len(ax_2d.patches) == 3
    plt.close(ax_1d.figure)
    plt.close(ax_2d.figure)


def test_bar_accepts_existing_fpde_explanation_object():
    explanation = diff_fpde(
        np.array([1.0, 2.0]),
        np.array([0.5, 1.5]),
        np.array([1.5, 1.0]),
    )

    ax = bar(explanation=explanation, show=False)

    assert ax.__class__.__name__ == "Axes"
    assert len(ax.patches) == 2
    plt.close(ax.figure)


def test_invalid_feature_names_and_shapes_raise_value_error():
    with pytest.raises(ValueError, match="feature_names length"):
        bar(values=_values(), feature_names=["only"], show=False)
    with pytest.raises(ValueError, match="1D or 2D"):
        bar(values=np.zeros((1, 2, 3)), show=False)
    with pytest.raises(ValueError, match="pass either explanation or values"):
        bar(explanation=FPDEPlotExplanation(values=_values()), values=_values(), show=False)


def test_beeswarm_with_and_without_data_and_nan_filtering():
    dirty_values = _values()
    dirty_values[1, 0] = np.nan
    dirty_data = _data()
    dirty_data[2, 1] = np.inf

    ax_with_data = beeswarm(
        values=dirty_values,
        data=dirty_data,
        feature_names=["a", "b", "c"],
        show=False,
    )
    ax_without_data = beeswarm(values=dirty_values, feature_names=["a", "b", "c"], show=False)

    assert ax_with_data.__class__.__name__ == "Axes"
    assert ax_without_data.__class__.__name__ == "Axes"
    assert ax_with_data.collections
    assert ax_without_data.collections
    plt.close(ax_with_data.figure)
    plt.close(ax_without_data.figure)


def test_waterfall_prediction_defaults_and_rejects_multiple_samples():
    ax = waterfall(values=[0.2, -0.1, 0.05], base_value=0.5, feature_names=["a", "b", "c"], show=False)

    assert ax.__class__.__name__ == "Axes"
    assert len(ax.patches) == 3
    with pytest.raises(ValueError, match="single sample"):
        waterfall(values=_values(), show=False)
    plt.close(ax.figure)


def test_waterfall_warns_on_prediction_mismatch():
    with pytest.warns(RuntimeWarning, match="prediction differs"):
        ax = waterfall(values=[0.2, -0.1], base_value=0.0, prediction=10.0, show=False)
    plt.close(ax.figure)


def test_scatter_accepts_int_str_color_and_no_data_fallback():
    ax_int = scatter(values=_values(), data=_data(), feature=0, feature_names=["a", "b", "c"], show=False)
    ax_str = scatter(
        values=_values(),
        data=_data(),
        feature="a",
        feature_names=["a", "b", "c"],
        color_feature="b",
        show=False,
    )
    with pytest.warns(RuntimeWarning, match="using sample index"):
        ax_no_data = scatter(values=_values(), feature=1, feature_names=["a", "b", "c"], show=False)

    assert ax_int.__class__.__name__ == "Axes"
    assert ax_str.__class__.__name__ == "Axes"
    assert ax_no_data.__class__.__name__ == "Axes"
    plt.close(ax_int.figure)
    plt.close(ax_str.figure)
    plt.close(ax_no_data.figure)


def test_scatter_filters_non_finite_points():
    dirty_values = _values()
    dirty_values[0, 0] = np.nan
    dirty_data = _data()
    dirty_data[1, 0] = np.inf

    ax = scatter(values=dirty_values, data=dirty_data, feature=0, show=False)

    offsets = ax.collections[0].get_offsets()
    assert offsets.shape[0] == 1
    plt.close(ax.figure)
