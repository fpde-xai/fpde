from __future__ import annotations

import numpy as np
import pytest

from fpde import (
    DynamicFPDEExplanation,
    dynamic_cos_fpde,
    dynamic_diff_fpde,
    dynamic_fpde_explain_batch,
    dynamic_fpde_explain_one,
    dynamic_hyb_fpde,
    plot_dynamic_attribution_heatmap,
    plot_dynamic_time_importance,
    prepare_dynamic_fpde_context,
    resample_time_series_linear,
    select_dynamic_lambda,
    temporal_deletion_insertion_curves,
)


def _training_sequences():
    return [
        np.array([[0.0, 0.0], [0.2, 0.1], [0.4, 0.0]], dtype=float),
        np.array([[0.1, 0.0], [0.3, 0.2], [0.5, 0.1], [0.6, 0.2]], dtype=float),
        np.array([[2.0, 1.0], [2.2, 1.2]], dtype=float),
        np.array([[2.1, 1.1], [2.3, 1.3], [2.5, 1.4]], dtype=float),
        np.array([[-1.0, 2.0], [-0.8, 2.2], [-0.6, 2.4], [-0.4, 2.6], [-0.2, 2.8]], dtype=float),
        np.array([[-1.1, 2.1], [-0.7, 2.3], [-0.3, 2.7]], dtype=float),
    ]


def _context():
    return prepare_dynamic_fpde_context(_training_sequences(), ["a", "a", "b", "b", "c", "c"], prototype_length=6)


def test_linear_resampling_shape():
    X = np.arange(30, dtype=float).reshape(10, 3)

    out = resample_time_series_linear(X, 20)

    assert out.shape == (20, 3)
    np.testing.assert_allclose(out[0], X[0])
    np.testing.assert_allclose(out[-1], X[-1])


def test_single_frame_resampling_repeats_frame():
    X = np.array([[1.0, 2.0, 3.0]], dtype=float)

    out = resample_time_series_linear(X, 5)

    assert out.shape == (5, 3)
    np.testing.assert_allclose(out, np.repeat(X, 5, axis=0))


def test_context_creation_with_variable_length_arrays():
    context = _context()

    assert context.prototypes.shape == (3, 6, 2)
    assert context.mean_anchor.shape == (6, 2)
    assert context.zero_anchor.shape == (6, 2)
    assert set(context.prototype_labels.tolist()) == {"a", "b", "c"}


def test_dynamic_diff_exactness():
    X = np.array([[1.0, 2.0], [3.0, 4.0]], dtype=float)
    target = X + 0.1
    rival = X + 1.0

    attr, evidence = dynamic_diff_fpde(X, target, rival)

    assert evidence == pytest.approx(float(np.sum(attr)))


def test_dynamic_cos_exactness_and_zero_stability():
    X = np.array([[1.0, 2.0], [3.0, 4.0]], dtype=float)
    target = X + 0.1
    rival = X + 1.0

    attr, evidence = dynamic_cos_fpde(X, target, rival)

    assert evidence == pytest.approx(float(np.sum(attr)))
    assert np.all(np.isfinite(attr))

    zero_attr, zero_evidence = dynamic_cos_fpde(np.zeros((3, 2)), np.zeros((3, 2)), np.zeros((3, 2)))
    assert np.all(np.isfinite(zero_attr))
    assert np.isfinite(zero_evidence)
    np.testing.assert_allclose(zero_attr, np.zeros((3, 2)))


def test_dynamic_hyb_endpoints_match_components_without_normalization():
    X = np.array([[0.0, 1.0], [2.0, 3.0]], dtype=float)
    target = X + 0.25
    rival = X - 0.5
    diff_attr, _ = dynamic_diff_fpde(X, target, rival)
    cos_attr, _ = dynamic_cos_fpde(X, target, rival)

    hyb_diff, _, _ = dynamic_hyb_fpde(X, target, rival, lambda_hyb=1.0, normalize="none")
    hyb_cos, _, _ = dynamic_hyb_fpde(X, target, rival, lambda_hyb=0.0, normalize="none")

    np.testing.assert_allclose(hyb_diff, diff_attr)
    np.testing.assert_allclose(hyb_cos, cos_attr)


def test_explain_one_shapes_and_scores():
    context = _context()
    X = np.array([[0.05, 0.0], [0.25, 0.1], [0.45, 0.1], [0.55, 0.2]], dtype=float)

    explanation = dynamic_fpde_explain_one(X, context, target_label="a", rival_label="b", mode="dynamic_diff")

    assert explanation.attributions.shape == X.shape
    assert explanation.time_importance.shape == (X.shape[0],)
    assert explanation.feature_importance.shape == (X.shape[1],)
    assert explanation.evidence == pytest.approx(float(np.sum(explanation.attributions)))
    assert explanation.evidence == pytest.approx(explanation.positive_score - explanation.negative_score)
    assert abs(explanation.exactness_residual) < 1e-12


def test_automatic_rival_selection_uses_non_target_label():
    context = _context()
    X = np.array([[0.05, 0.0], [0.25, 0.1], [0.45, 0.1], [0.55, 0.2]], dtype=float)

    explanation = dynamic_fpde_explain_one(X, context, target_label="a", rival_label=None)

    assert explanation.rival_label != "a"
    assert explanation.rival_label in {"b", "c"}


def test_dynamic_explain_batch_validates_lengths_and_runs():
    context = _context()
    X_list = _training_sequences()[:2]

    explanations = dynamic_fpde_explain_batch(X_list, context, target_labels=["a", "a"])

    assert len(explanations) == 2
    assert all(exp.attributions.shape == X_list[i].shape for i, exp in enumerate(explanations))
    with pytest.raises(ValueError, match="target_labels"):
        dynamic_fpde_explain_batch(X_list, context, target_labels=["a"])


def test_invalid_inputs_raise_clear_errors():
    context = _context()
    with pytest.raises(ValueError, match="at least one sample"):
        prepare_dynamic_fpde_context([], [])
    with pytest.raises(ValueError, match="inconsistent feature dimension"):
        prepare_dynamic_fpde_context([np.ones((2, 2)), np.ones((2, 3))], [0, 1])
    with pytest.raises(ValueError, match="2D"):
        resample_time_series_linear(np.ones((2, 2, 1)), 5)
    with pytest.raises(NotImplementedError, match="linear"):
        prepare_dynamic_fpde_context([np.ones((2, 2))], [0], alignment="dtw")
    with pytest.raises(ValueError, match="lambda_hyb"):
        dynamic_hyb_fpde(np.ones((2, 2)), np.ones((2, 2)), np.zeros((2, 2)), lambda_hyb=1.5)
    with pytest.raises(ValueError, match="mode"):
        dynamic_fpde_explain_one(np.ones((2, 2)), context, target_label="a", mode="raw_waveform")
    with pytest.raises(ValueError, match="anchor_strategy"):
        dynamic_fpde_explain_one(np.ones((2, 2)), context, target_label="a", anchor_strategy="bad")
    with pytest.raises(ValueError, match="eps"):
        dynamic_cos_fpde(np.ones((2, 2)), np.ones((2, 2)), np.ones((2, 2)), eps=0.0)


def test_numerical_stability_for_constant_and_zero_arrays():
    constant = np.full((4, 3), 2.0)
    zero = np.zeros((4, 3))

    for X, target, rival in ((constant, constant, constant), (zero, zero, zero)):
        diff_attr, diff_evidence = dynamic_diff_fpde(X, target, rival)
        cos_attr, cos_evidence = dynamic_cos_fpde(X, target, rival)
        hyb_attr, hyb_evidence, _ = dynamic_hyb_fpde(X, target, rival)
        assert np.all(np.isfinite(diff_attr))
        assert np.all(np.isfinite(cos_attr))
        assert np.all(np.isfinite(hyb_attr))
        assert np.isfinite(diff_evidence)
        assert np.isfinite(cos_evidence)
        assert np.isfinite(hyb_evidence)


def test_temporal_deletion_insertion_curves_and_lambda_selection():
    context = _context()
    X = _training_sequences()[0]
    explanation = dynamic_fpde_explain_one(X, context, target_label="a", rival_label="b")

    curves = temporal_deletion_insertion_curves(
        X,
        explanation,
        context,
        target_label="a",
        rival_label="b",
        steps=3,
    )

    assert set(curves) >= {"deletion_curve", "insertion_curve", "deletion_drop_auc", "insertion_auc", "combined_score"}
    assert len(curves["deletion_curve"]) == len(curves["insertion_curve"])
    assert np.isfinite(curves["combined_score"])

    selection = select_dynamic_lambda(_training_sequences()[:4], ["a", "a", "b", "b"], context, lambda_grid=[0.0, 0.5, 1.0], steps=3)
    assert selection["best_lambda"] in (0.0, 0.5, 1.0)
    assert len(selection["rows"]) == 3


class _FakeFigure:
    def __init__(self):
        self.colorbar_calls = []

    def colorbar(self, *args, **kwargs):
        self.colorbar_calls.append((args, kwargs))


class _FakeAxes:
    def __init__(self):
        self.calls = []
        self.figure = _FakeFigure()

    def __getattr__(self, name):
        def recorder(*args, **kwargs):
            self.calls.append((name, args, kwargs))
            return "image" if name == "imshow" else None

        return recorder


def test_dynamic_plotting_helpers_accept_existing_axes():
    explanation = DynamicFPDEExplanation(
        mode="dynamic_diff",
        evidence=0.0,
        attributions=np.array([[1.0, -0.5], [0.25, -0.25]], dtype=float),
        time_importance=np.array([0.5, 0.0], dtype=float),
        feature_importance=np.array([1.25, -0.75], dtype=float),
        positive_score=0.0,
        negative_score=0.0,
        target_label="a",
        rival_label="b",
        exactness_residual=0.0,
        details={},
    )
    time_ax = _FakeAxes()
    heatmap_ax = _FakeAxes()

    assert plot_dynamic_time_importance(explanation, ax=time_ax) is time_ax
    assert [name for name, _, _ in time_ax.calls].count("bar") == 1
    assert plot_dynamic_attribution_heatmap(explanation, ax=heatmap_ax) is heatmap_ax
    assert [name for name, _, _ in heatmap_ax.calls].count("imshow") == 1
    assert len(heatmap_ax.figure.colorbar_calls) == 1
