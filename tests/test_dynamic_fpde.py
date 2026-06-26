from __future__ import annotations

import numpy as np
import pytest

from fpde import (
    DynamicFPDEEngine,
    DynamicFPDEExplanation,
    DynamicFPDEResult,
    NativeTimeDynamicFPDEExplanation,
    PrototypeRawGenerator,
    dynamic_cos_fpde,
    dynamic_diff_fpde,
    dynamic_fpde_explain_batch,
    dynamic_fpde_explain_one,
    dynamic_hyb_fpde,
    native_dynamic_cos_fpde,
    native_dynamic_diff_fpde,
    native_dynamic_fpde_explain_batch,
    native_dynamic_fpde_explain_one,
    native_dynamic_hyb_fpde,
    pad_sequences,
    plot_dynamic_attribution_heatmap,
    plot_dynamic_time_importance,
    prepare_dynamic_fpde_context,
    resample_time_series_linear,
    select_dynamic_lambda,
    select_lambda_dynamic,
    split_representation,
    temporal_deletion_insertion_curves,
    validate_sequence_inputs,
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


def test_dynamic_hyb_scores_are_finite():
    context = _context()
    X = np.array([[0.05, 0.0], [0.25, 0.1], [0.45, 0.1], [0.55, 0.2]], dtype=float)

    for normalize in ("none", "l1"):
        explanation = dynamic_fpde_explain_one(
            X,
            context,
            target_label="a",
            rival_label="b",
            mode="dynamic_hyb",
            lambda_hyb=0.25,
            normalize=normalize,
        )

        assert np.isfinite(explanation.positive_score)
        assert np.isfinite(explanation.negative_score)
        assert explanation.details["hyb_positive_score"] == pytest.approx(explanation.positive_score)
        assert explanation.details["hyb_negative_score"] == pytest.approx(explanation.negative_score)


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


def _prototype_for_label(context, label, length):
    idx = int(np.where(context.prototype_labels == label)[0][0])
    return resample_time_series_linear(context.prototypes[idx], length)


def _manual_diff_evidence(context, X, target_label, rival_label):
    target = _prototype_for_label(context, target_label, X.shape[0])
    rival = _prototype_for_label(context, rival_label, X.shape[0])
    _, evidence = dynamic_diff_fpde(X, target, rival)
    return evidence


def test_temporal_deletion_insertion_curves_return_normalized_metrics():
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

    assert set(curves) >= {
        "deletion_curve",
        "insertion_curve",
        "deletion_drop_curve",
        "insertion_gain_curve",
        "deletion_drop_auc",
        "insertion_gain_auc",
        "insertion_auc",
        "combined_score",
    }
    assert len(curves["deletion_curve"]) == len(curves["insertion_curve"])
    assert len(curves["deletion_drop_curve"]) == len(curves["deletion_curve"])
    assert len(curves["insertion_gain_curve"]) == len(curves["insertion_curve"])
    scale = abs(curves["original_evidence"] - curves["baseline_evidence"]) + 1e-12
    np.testing.assert_allclose(
        curves["deletion_drop_curve"],
        (curves["original_evidence"] - np.asarray(curves["deletion_curve"])) / scale,
    )
    np.testing.assert_allclose(
        curves["insertion_gain_curve"],
        (np.asarray(curves["insertion_curve"]) - curves["insertion_curve"][0]) / scale,
    )
    assert curves["insertion_auc"] == pytest.approx(curves["insertion_gain_auc"])
    assert np.isfinite(curves["combined_score"])


def test_temporal_deletion_insertion_rank_by_modes():
    context = _context()
    X = _training_sequences()[0]
    explanation = DynamicFPDEExplanation(
        mode="dynamic_hyb",
        evidence=0.0,
        attributions=np.zeros_like(X, dtype=float),
        time_importance=np.array([0.0, -10.0, 1.0], dtype=float),
        feature_importance=np.zeros(X.shape[1], dtype=float),
        positive_score=0.0,
        negative_score=0.0,
        target_label="a",
        rival_label="b",
        exactness_residual=0.0,
        details={},
    )
    baseline = resample_time_series_linear(context.mean_anchor, X.shape[0])

    for rank_by, expected_first_idx in (("positive", 2), ("signed", 2), ("absolute", 1)):
        curves = temporal_deletion_insertion_curves(
            X,
            explanation,
            context,
            target_label="a",
            rival_label="b",
            steps=3,
            rank_by=rank_by,
        )
        deletion = X.copy()
        deletion[expected_first_idx] = baseline[expected_first_idx]
        assert curves["deletion_curve"][1] == pytest.approx(_manual_diff_evidence(context, deletion, "a", "b"))
        assert curves["rank_by"] == rank_by

    with pytest.raises(ValueError, match="rank_by"):
        temporal_deletion_insertion_curves(
            X,
            explanation,
            context,
            target_label="a",
            rival_label="b",
            rank_by="unknown",
        )


def test_select_dynamic_lambda_uses_normalized_combined_score():
    context = _context()
    X_val = _training_sequences()[:2]
    y_val = ["a", "a"]

    selection = select_dynamic_lambda(X_val, y_val, context, lambda_grid=[0.5], steps=3)

    manual_scores = []
    for X, y in zip(X_val, y_val):
        explanation = dynamic_fpde_explain_one(X, context, target_label=y, lambda_hyb=0.5)
        curves = temporal_deletion_insertion_curves(
            X,
            explanation,
            context,
            target_label=y,
            rival_label=explanation.rival_label,
            steps=3,
        )
        manual_scores.append(curves["combined_score"])

    assert selection["rows"][0]["metric_source"] == "normalized_prototype_evidence_curves"
    assert selection["rows"][0]["score"] == pytest.approx(float(np.mean(manual_scores)))
    assert selection["rows"][0]["mean_insertion_gain_auc"] == pytest.approx(selection["rows"][0]["mean_insertion_auc"])
    assert selection["metric_means"]["0.5"]["insertion_gain_auc"] == pytest.approx(
        selection["metric_means"]["0.5"]["insertion_auc"]
    )


def test_temporal_deletion_insertion_curves_and_lambda_selection():
    context = _context()
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


def test_native_dynamic_shape_preservation_and_metadata():
    X = np.random.default_rng(0).normal(size=(137, 20))
    p_target = np.zeros(20)
    p_rival = np.ones(20)

    exp = native_dynamic_fpde_explain_one(
        X,
        p_target=p_target,
        p_rival=p_rival,
        mode="dynamic_diff",
        feature_names=[f"f{i}" for i in range(20)],
        timestamps_sec=np.arange(137, dtype=float),
    )

    assert isinstance(exp, NativeTimeDynamicFPDEExplanation)
    assert exp.attributions.shape == X.shape
    assert exp.time_importance.shape == (137,)
    assert exp.feature_importance.shape == (20,)
    assert exp.time_mode == "native"
    assert exp.temporal_resampling is False
    assert exp.temporal_pooling is False
    assert exp.details["time_mode"] == "native"
    assert exp.details["temporal_resampling"] is False
    assert exp.details["temporal_pooling"] is False
    assert exp.details["prototype_kind"] == "feature_vector"
    assert exp.details["input_shape"] == X.shape
    assert exp.details["output_shape"] == X.shape


def test_native_dynamic_variable_length_batch_preserves_each_length():
    rng = np.random.default_rng(1)
    X_list = [
        rng.normal(size=(100, 20)),
        rng.normal(size=(777, 20)),
        rng.normal(size=(2400, 20)),
    ]
    p_target = np.zeros(20)
    p_rival = np.ones(20)

    explanations = native_dynamic_fpde_explain_batch(
        X_list,
        p_targets=p_target,
        p_rivals=[p_rival, p_rival + 1.0, p_rival + 2.0],
        target_labels=["a", "b", "c"],
        rival_labels=["x", "y", "z"],
    )

    assert [exp.attributions.shape for exp in explanations] == [X.shape for X in X_list]
    assert [exp.time_importance.shape for exp in explanations] == [(100,), (777,), (2400,)]


def test_native_dynamic_diff_has_no_temporal_spreading_from_resampling():
    X = np.zeros((101, 3), dtype=float)
    X[50, 1] = 100.0
    p_target = np.zeros(3, dtype=float)
    p_rival = np.ones(3, dtype=float)

    attr, evidence = native_dynamic_diff_fpde(X, p_target, p_rival)
    expected = (X - p_rival) ** 2 - (X - p_target) ** 2

    assert attr.shape == (101, 3)
    np.testing.assert_allclose(attr, expected)
    assert attr[50, 1] == pytest.approx(expected[50, 1])
    assert attr[49, 1] == pytest.approx(expected[49, 1])
    assert attr[51, 1] == pytest.approx(expected[51, 1])
    assert attr[49, 1] == pytest.approx(attr[51, 1])
    assert evidence == pytest.approx(float(np.sum(attr)))


def test_native_dynamic_diff_exactness():
    rng = np.random.default_rng(2)
    X = rng.normal(size=(17, 5))
    p_target = rng.normal(size=5)
    p_rival = rng.normal(size=5)

    attr, evidence = native_dynamic_diff_fpde(X, p_target, p_rival)

    assert evidence == pytest.approx(float(np.sum(attr)))
    assert abs(evidence - np.sum(attr)) <= 1e-9


def test_native_dynamic_cos_numerical_stability():
    cases = [
        (np.zeros((11, 4)), np.zeros(4), np.zeros(4), np.zeros(4)),
        (np.full((11, 4), 1e-300), np.full(4, 1e-300), np.full(4, -1e-300), np.zeros(4)),
    ]

    for X, p_target, p_rival, anchor in cases:
        attr, evidence = native_dynamic_cos_fpde(X, p_target, p_rival, anchor=anchor)
        assert attr.shape == X.shape
        assert np.all(np.isfinite(attr))
        assert np.isfinite(evidence)


def test_native_dynamic_hyb_endpoints_match_components_without_normalization():
    rng = np.random.default_rng(3)
    X = rng.normal(size=(31, 6))
    p_target = rng.normal(size=6)
    p_rival = rng.normal(size=6)
    diff_attr, _ = native_dynamic_diff_fpde(X, p_target, p_rival)
    cos_attr, _ = native_dynamic_cos_fpde(X, p_target, p_rival)

    hyb_diff, _, _ = native_dynamic_hyb_fpde(X, p_target, p_rival, lambda_hyb=1.0, normalize="none")
    hyb_cos, _, _ = native_dynamic_hyb_fpde(X, p_target, p_rival, lambda_hyb=0.0, normalize="none")

    np.testing.assert_allclose(hyb_diff, diff_attr)
    np.testing.assert_allclose(hyb_cos, cos_attr)


def test_native_dynamic_invalid_inputs_raise_clear_errors():
    X = np.ones((5, 3))
    p_target = np.zeros(3)
    p_rival = np.ones(3)

    with pytest.raises(ValueError, match="2D"):
        native_dynamic_fpde_explain_one(np.ones((5, 3, 1)), p_target=p_target, p_rival=p_rival)
    with pytest.raises(ValueError, match="p_target"):
        native_dynamic_fpde_explain_one(X, p_target=np.zeros(2), p_rival=p_rival)
    with pytest.raises(ValueError, match="p_rival"):
        native_dynamic_fpde_explain_one(X, p_target=p_target, p_rival=np.ones(2))
    with pytest.raises(ValueError, match="anchor"):
        native_dynamic_fpde_explain_one(X, p_target=p_target, p_rival=p_rival, anchor=np.zeros(2))
    with pytest.raises(ValueError, match="NaN or inf"):
        native_dynamic_fpde_explain_one(np.array([[np.nan, 1.0, 2.0]]), p_target=p_target, p_rival=p_rival)
    with pytest.raises(ValueError, match="NaN or inf"):
        native_dynamic_fpde_explain_one(np.array([[np.inf, 1.0, 2.0]]), p_target=p_target, p_rival=p_rival)
    with pytest.raises(ValueError, match="timestamps_sec"):
        native_dynamic_fpde_explain_one(X, p_target=p_target, p_rival=p_rival, timestamps_sec=[0.0, 1.0])
    with pytest.raises(ValueError, match="feature_names"):
        native_dynamic_fpde_explain_one(X, p_target=p_target, p_rival=p_rival, feature_names=["a", "b"])
    with pytest.raises(ValueError, match="lambda_hyb"):
        native_dynamic_hyb_fpde(X, p_target, p_rival, lambda_hyb=-0.1)
    with pytest.raises(ValueError, match="lambda_hyb"):
        native_dynamic_hyb_fpde(X, p_target, p_rival, lambda_hyb=1.1)


def test_dynamic_plotting_helpers_accept_native_explanations():
    exp = native_dynamic_fpde_explain_one(
        np.array([[0.0, 1.0], [2.0, 3.0]], dtype=float),
        p_target=np.zeros(2),
        p_rival=np.ones(2),
        mode="dynamic_diff",
    )
    time_ax = _FakeAxes()
    heatmap_ax = _FakeAxes()

    assert plot_dynamic_time_importance(exp, ax=time_ax) is time_ax
    assert [name for name, _, _ in time_ax.calls].count("bar") == 1
    assert plot_dynamic_attribution_heatmap(exp, ax=heatmap_ax) is heatmap_ax
    assert [name for name, _, _ in heatmap_ax.calls].count("imshow") == 1


def test_rawfeat_pad_sequences_returns_boolean_mask():
    padded, mask = pad_sequences([np.ones((2, 3)), np.full((4, 3), 2.0)])

    assert padded.shape == (2, 4, 3)
    assert mask.dtype == bool
    np.testing.assert_array_equal(mask[0], [True, True, False, False])
    np.testing.assert_allclose(padded[0, 2:], 0.0)


def test_rawfeat_dynamic_fixed_length_raw_only_diff_exactness():
    rng = np.random.default_rng(10)
    raw = rng.normal(size=(6, 5, 2))
    y = np.array(["a", "a", "a", "b", "b", "b"], dtype=object)

    engine = DynamicFPDEEngine()
    returned = engine.fit(raw=raw, y=y)
    result = engine.explain_one(raw=raw[0], method="diff", target_class="a", rival_class="b")

    assert returned is engine
    assert isinstance(result, DynamicFPDEResult)
    assert result.method == "diff"
    assert result.attributions.shape == (5, 2)
    assert result.raw_attributions.shape == (5, 2)
    assert result.feature_attributions is None
    assert result.evidence == pytest.approx(float(np.sum(result.attributions)))
    assert result.audit["passed"] is True


def test_rawfeat_dynamic_fixed_length_raw_and_features_group_exactness():
    rng = np.random.default_rng(11)
    raw = rng.normal(size=(8, 4, 2))
    features = rng.normal(size=(8, 4, 3))
    y = np.array([0, 1] * 4)

    engine = DynamicFPDEEngine(lambda_hyb=0.25).fit(raw=raw, features=features, y=y)
    result = engine.explain_one(raw=raw[0], features=features[0], method="hyb", target_class=0, rival_class=1)

    assert result.raw_attributions.shape == (4, 2)
    assert result.feature_attributions is not None
    assert result.feature_attributions.shape == (4, 3)
    assert result.dt_attributions is None
    assert result.group_attributions["raw"] + result.group_attributions["features"] == pytest.approx(result.evidence)
    assert result.audit["attribution_sum"] == pytest.approx(result.evidence)


def test_rawfeat_dynamic_variable_length_padding_attributions_are_zero():
    raw = [
        np.array([[0.0], [0.2]], dtype=float),
        np.array([[0.1], [0.3], [0.4], [0.5]], dtype=float),
        np.array([[2.0], [2.1], [2.2]], dtype=float),
        np.array([[1.9], [2.0], [2.1], [2.2]], dtype=float),
    ]
    y = np.array(["a", "a", "b", "b"], dtype=object)

    engine = DynamicFPDEEngine().fit(raw=raw, y=y)
    result = engine.explain_one(raw=raw[0], method="diff", target_class="a", rival_class="b")

    assert result.attributions.shape == (4, 1)
    np.testing.assert_array_equal(result.mask, [True, True, False, False])
    np.testing.assert_allclose(result.attributions[~result.mask], 0.0)
    np.testing.assert_allclose(result.time_attributions[~result.mask], 0.0)


def test_rawfeat_dynamic_cos_and_hyb_exactness_and_zero_l1_stability():
    raw = np.zeros((4, 3, 2), dtype=float)
    raw[2:] = 1.0
    y = np.array([0, 0, 1, 1])

    engine = DynamicFPDEEngine(lambda_hyb=0.5).fit(raw=raw, y=y)
    cos = engine.explain_one(raw=raw[0], method="cos", target_class=0, rival_class=1)
    hyb = engine.explain_one(raw=raw[0], method="hyb", target_class=0, rival_class=1)
    zero_hyb = DynamicFPDEEngine().fit(raw=np.zeros((4, 3, 2)), y=y).explain_one(
        raw=np.zeros((3, 2)),
        method="hyb",
        target_class=0,
        rival_class=1,
    )

    assert cos.evidence == pytest.approx(float(np.sum(cos.attributions)))
    assert hyb.evidence == pytest.approx(float(np.sum(hyb.attributions)))
    assert np.all(np.isfinite(zero_hyb.attributions))
    np.testing.assert_allclose(zero_hyb.attributions, 0.0)
    assert zero_hyb.evidence == pytest.approx(0.0)


def test_rawfeat_dynamic_explain_batch_and_explicit_target_rival():
    rng = np.random.default_rng(12)
    raw = rng.normal(size=(6, 4, 2))
    y = np.array(["a", "b", "a", "b", "a", "b"], dtype=object)

    engine = DynamicFPDEEngine().fit(raw=raw, y=y)
    results = engine.explain_batch(
        raw=raw[:3],
        method="diff",
        target_classes=["a", "b", "a"],
        rival_classes=["b", "a", "b"],
    )

    assert len(results) == 3
    assert [result.target_class for result in results] == ["a", "b", "a"]
    assert [result.rival_class for result in results] == ["b", "a", "b"]
    assert all(result.evidence == pytest.approx(float(np.sum(result.attributions))) for result in results)


def test_rawfeat_dynamic_predict_proba_priority_and_lambda_placeholder():
    raw = np.array(
        [
            [[0.0], [0.1]],
            [[1.0], [1.1]],
            [[2.0], [2.1]],
        ],
        dtype=float,
    )
    engine = DynamicFPDEEngine().fit(raw=raw, y=["a", "b", "c"])

    result = engine.explain_one(
        raw=raw[0],
        method="dynamic_diff",
        predict_proba=[0.1, 0.8, 0.3],
        class_names=["a", "b", "c"],
    )
    selection = select_lambda_dynamic(lambda_grid=[0.0, 0.5, 1.0])

    assert result.method == "diff"
    assert result.target_class == "b"
    assert result.rival_class == "c"
    assert selection["best_lambda"] == pytest.approx(0.5)
    assert [row["status"] for row in selection["rows"]] == ["placeholder", "placeholder", "placeholder"]
    assert {row["metric_source"] for row in selection["rows"]} == {"not_evaluated"}


def test_prototype_raw_generator_generates_requested_length():
    raw = [
        np.array([[0.0, 0.1], [0.2, 0.3]], dtype=float),
        np.array([[0.1, 0.0], [0.3, 0.2], [0.5, 0.4]], dtype=float),
        np.array([[1.0, 1.1], [1.2, 1.3]], dtype=float),
    ]

    gen = PrototypeRawGenerator().fit(raw=raw, y=["a", "a", "b"])
    generated = gen.generate(label="a", length=5, noise_scale=0.05, random_state=0)

    assert generated.shape == (5, 2)
    assert np.all(np.isfinite(generated))


def test_rawfeat_prototype_invalid_times_are_zero_for_both_directions():
    raw = [
        np.array([[0.0], [0.1], [0.2]], dtype=float),
        np.array([[0.1], [0.2], [0.3]], dtype=float),
        np.array([[2.0], [2.1], [2.2], [2.3], [2.4], [2.5]], dtype=float),
        np.array([[2.1], [2.2], [2.3], [2.4], [2.5], [2.6]], dtype=float),
    ]
    y = np.array([0, 0, 1, 1])
    engine = DynamicFPDEEngine().fit(raw=raw, y=y)

    short_target = engine.explain_one(raw=raw[2], method="diff", target_class=0, rival_class=1)
    long_target = engine.explain_one(raw=raw[2], method="cos", target_class=1, rival_class=0)

    for result in (short_target, long_target):
        np.testing.assert_allclose(result.attributions[3:], 0.0)
        np.testing.assert_allclose(result.time_attributions[3:], 0.0)
        assert result.audit["prototype_invalid_time_count"] == 3
        assert result.audit["valid_time_count"] == 3
        assert result.audit["target_prototype_valid_count"] in (3, 6)
        assert result.audit["rival_prototype_valid_count"] in (3, 6)
        assert result.audit["passed"] is True
        assert result.evidence == pytest.approx(float(np.sum(result.attributions)))


def test_rawfeat_tuple_variable_length_input_and_single_2d_sample():
    raw_tuple = (
        np.array([[0.0, 0.1], [0.2, 0.3]], dtype=float),
        np.array([[1.0, 1.1], [1.2, 1.3], [1.4, 1.5]], dtype=float),
    )
    feat_tuple = (
        np.ones((2, 1), dtype=float),
        np.ones((3, 1), dtype=float),
    )

    batch = validate_sequence_inputs(raw_tuple, features=feat_tuple)
    single = validate_sequence_inputs(raw_tuple[0])

    assert batch.raw.shape == (2, 3, 2)
    assert batch.features is not None
    assert batch.features.shape == (2, 3, 1)
    np.testing.assert_array_equal(batch.mask[0], [True, True, False])
    assert single.raw.shape == (1, 2, 2)
    np.testing.assert_array_equal(single.mask, [[True, True]])


def test_rawfeat_all_false_and_no_effective_valid_time_return_zero_audit_passed():
    raw = np.array([[[0.0], [0.1], [0.2]], [[1.0], [1.1], [1.2]]], dtype=float)
    engine = DynamicFPDEEngine().fit(raw=raw, y=[0, 1])
    variable_engine = DynamicFPDEEngine().fit(
        raw=[
            np.array([[0.0], [0.1]], dtype=float),
            np.array([[1.0], [1.1], [1.2]], dtype=float),
        ],
        y=[0, 1],
    )

    all_false = engine.explain_one(
        raw=raw[0],
        mask=np.array([False, False, False]),
        method="hyb",
        target_class=0,
        rival_class=1,
    )
    no_effective = variable_engine.explain_one(
        raw=np.array([[1.0], [1.1], [1.2]], dtype=float),
        mask=np.array([False, False, True]),
        method="cos",
        target_class=0,
        rival_class=1,
    )

    for result in (all_false, no_effective):
        np.testing.assert_allclose(result.attributions, 0.0)
        np.testing.assert_allclose(result.time_attributions, 0.0)
        assert result.evidence == pytest.approx(0.0)
        assert result.audit["valid_time_count"] == 0
        assert result.audit["warning"] == "no_valid_time"
        assert result.audit["passed"] is True


def test_rawfeat_hyb_l1_edge_cases_are_finite_and_audited():
    raw = np.array(
        [
            [[1.0, 0.0]],
            [[0.0, 1.0]],
        ],
        dtype=float,
    )
    engine = DynamicFPDEEngine(lambda_hyb=0.5).fit(raw=raw, y=[0, 1])

    engine.mean_anchor_ = np.zeros_like(engine.mean_anchor_)
    diff_zero = engine.explain_one(raw=np.array([[0.5, 0.5]]), method="hyb", target_class=0, rival_class=1)
    engine.mean_anchor_ = np.array([[0.25, 0.25]], dtype=float)
    cos_zero = engine.explain_one(raw=np.array([[0.25, 0.25]]), method="hyb", target_class=0, rival_class=1)
    both_zero = DynamicFPDEEngine().fit(raw=np.zeros((2, 1, 2)), y=[0, 1]).explain_one(
        raw=np.zeros((1, 2)),
        method="hyb",
        target_class=0,
        rival_class=1,
    )

    assert diff_zero.audit["diff_l1"] == pytest.approx(0.0)
    assert diff_zero.audit["cos_l1"] > 0.0
    assert cos_zero.audit["diff_l1"] > 0.0
    assert cos_zero.audit["cos_l1"] == pytest.approx(0.0)
    assert both_zero.audit["diff_l1"] == pytest.approx(0.0)
    assert both_zero.audit["cos_l1"] == pytest.approx(0.0)
    for result in (diff_zero, cos_zero, both_zero):
        assert np.all(np.isfinite(result.attributions))
        assert result.evidence == pytest.approx(float(np.sum(result.attributions)))
        assert result.audit["passed"] is True


def test_rawfeat_explain_batch_predict_proba_shape_validation():
    raw = np.array(
        [
            [[0.0], [0.1]],
            [[1.0], [1.1]],
            [[2.0], [2.1]],
        ],
        dtype=float,
    )
    engine = DynamicFPDEEngine().fit(raw=raw, y=["a", "b", "c"])

    results = engine.explain_batch(
        raw=raw[:2],
        predict_proba=np.array([[0.7, 0.2, 0.1], [0.1, 0.8, 0.2]], dtype=float),
        class_names=["a", "b", "c"],
        method="diff",
    )

    assert [result.target_class for result in results] == ["a", "b"]
    with pytest.raises(ValueError, match="2D"):
        engine.explain_batch(raw=raw[:2], predict_proba=np.array([0.7, 0.2, 0.1]))
    with pytest.raises(ValueError, match="class dimension"):
        engine.explain_batch(raw=raw[:2], predict_proba=np.ones((2, 2)), class_names=["a", "b", "c"])


def test_rawfeat_result_shape_contract_all_methods_with_dt():
    rng = np.random.default_rng(42)
    raw = rng.normal(size=(6, 4, 2))
    features = rng.normal(size=(6, 4, 3))
    dt = np.ones((6, 4, 1), dtype=float)
    y = np.array([0, 1] * 3)
    engine = DynamicFPDEEngine().fit(raw=raw, features=features, dt=dt, y=y)

    for method in ("diff", "cos", "hyb"):
        result = engine.explain_one(
            raw=raw[0],
            features=features[0],
            dt=dt[0],
            method=method,
            target_class=0,
            rival_class=1,
        )

        assert result.attributions.shape == (4, 6)
        assert result.raw_attributions.shape == (4, 2)
        assert result.feature_attributions is not None
        assert result.feature_attributions.shape == (4, 3)
        assert result.dt_attributions is not None
        assert result.dt_attributions.shape == (4, 1)
        assert result.time_attributions.shape == (4,)
        assert (
            result.group_attributions["raw"]
            + result.group_attributions["features"]
            + result.group_attributions["dt"]
        ) == pytest.approx(result.evidence)
        assert result.audit["passed"] is True


def test_prototype_raw_generator_hardening_and_reproducible_noise():
    raw = [
        np.array([[0.0, 0.1], [0.2, 0.3]], dtype=float),
        np.array([[0.1, 0.0], [0.3, 0.2]], dtype=float),
        np.array([[1.0, 1.1], [1.2, 1.3]], dtype=float),
    ]

    with pytest.raises(RuntimeError, match="fit"):
        PrototypeRawGenerator().generate(label="a")

    gen = PrototypeRawGenerator().fit(raw=raw, y=["a", "a", "b"])
    first = gen.generate(label="a", length=4, noise_scale=0.2, random_state=123)
    second = gen.generate(label="a", length=4, noise_scale=0.2, random_state=123)

    np.testing.assert_allclose(first, second)
    with pytest.raises(ValueError, match="label"):
        gen.generate(label="missing")
    with pytest.raises(ValueError, match="length"):
        gen.generate(label="a", length=0)
    with pytest.raises(ValueError, match="noise_scale"):
        gen.generate(label="a", noise_scale=-0.1)
    with pytest.raises(ValueError, match="condition_features"):
        gen.generate(label="a", condition_features=np.array([[np.nan]]))


def _probabilities_from_raw_mean(raw, features=None, dt=None, mask=None):
    values = np.asarray(raw, dtype=float)
    if mask is None:
        means = np.mean(values, axis=(1, 2))
    else:
        weights = np.asarray(mask, dtype=float)
        denom = np.maximum(np.sum(weights, axis=1), 1.0)
        means = np.sum(values[:, :, 0] * weights, axis=1) / denom
    p1 = 1.0 / (1.0 + np.exp(-4.0 * (means - 0.5)))
    return np.stack([1.0 - p1, p1], axis=1)


def _probabilities_from_representation_mean(representation):
    raw, _, _ = split_representation(representation, {"raw": slice(0, 1)})
    return _probabilities_from_raw_mean(raw)


def test_select_lambda_dynamic_evaluates_fixed_length_validation():
    raw = np.array(
        [
            [[0.0], [0.1], [0.2]],
            [[0.1], [0.2], [0.3]],
            [[0.8], [0.9], [1.0]],
            [[0.7], [0.8], [0.9]],
        ],
        dtype=float,
    )
    engine = DynamicFPDEEngine().fit(raw=raw, y=[0, 0, 1, 1])

    selection = select_lambda_dynamic(
        engine=engine,
        raw=raw[2:],
        predict_proba=_probabilities_from_raw_mean,
        lambdas=[0.0, 0.5, 1.0],
        target_classes=[1, 1],
        rival_classes=[0, 0],
        steps=5,
    )

    assert selection["status"] == "evaluated"
    assert selection["metric"] == "dynamic_deletion_insertion"
    assert selection["best_lambda"] in [0.0, 0.5, 1.0]
    assert set(selection["scores"]) == {0.0, 0.5, 1.0}
    assert selection["n_validation"] == 2
    assert selection["steps"] == 5
    assert selection["granularity"] == "coordinate"
    assert all(np.isfinite(value) for value in selection["scores"].values())


def test_engine_select_lambda_method_delegates_to_function():
    raw = np.array(
        [
            [[0.0], [0.1]],
            [[0.2], [0.3]],
            [[0.8], [0.9]],
            [[1.0], [1.1]],
        ],
        dtype=float,
    )
    engine = DynamicFPDEEngine().fit(raw=raw, y=[0, 0, 1, 1])

    selection = engine.select_lambda(
        raw=raw[:2],
        predict_proba=_probabilities_from_raw_mean,
        lambdas=[0.0, 1.0],
        target_classes=[0, 0],
        rival_classes=[1, 1],
        steps=3,
    )

    assert selection["status"] == "evaluated"
    assert selection["lambdas"] == [0.0, 1.0]


def test_select_lambda_dynamic_accepts_precomputed_predict_proba():
    raw = np.array(
        [
            [[0.0], [0.1]],
            [[1.0], [1.1]],
            [[0.2], [0.3]],
            [[0.9], [1.0]],
        ],
        dtype=float,
    )
    engine = DynamicFPDEEngine().fit(raw=raw[:2], y=[0, 1])

    selection = select_lambda_dynamic(
        engine=engine,
        raw=raw[2:],
        predict_proba=np.array([[0.8, 0.2], [0.1, 0.9]], dtype=float),
        lambdas=[0.0, 0.5, 1.0],
        steps=4,
    )

    assert selection["status"] == "evaluated"
    assert selection["predict_proba_source"] == "precomputed"
    assert selection["best_lambda"] == pytest.approx(0.0)


def test_select_lambda_dynamic_callable_raw_features_api():
    raw = np.array(
        [
            [[0.0], [0.1], [0.2]],
            [[1.0], [1.1], [1.2]],
            [[0.2], [0.3], [0.4]],
            [[0.8], [0.9], [1.0]],
        ],
        dtype=float,
    )
    features = raw + 0.5
    engine = DynamicFPDEEngine().fit(raw=raw[:2], features=features[:2], y=[0, 1])
    calls = []

    def predict_proba(*, raw=None, features=None, dt=None, mask=None):
        calls.append((raw.shape, None if features is None else features.shape, mask.shape))
        return _probabilities_from_raw_mean(raw, features=features, mask=mask)

    selection = select_lambda_dynamic(
        engine=engine,
        raw=raw[2:],
        features=features[2:],
        predict_proba=predict_proba,
        lambdas=[0.0, 0.5],
        target_classes=[0, 1],
        rival_classes=[1, 0],
        baseline="zero",
        steps=3,
    )

    assert selection["status"] == "evaluated"
    assert calls
    assert selection["baseline"] == "zero"


def test_select_lambda_dynamic_callable_representation_fallback():
    raw = np.array(
        [
            [[0.0], [0.1]],
            [[1.0], [1.1]],
            [[0.2], [0.3]],
            [[0.8], [0.9]],
        ],
        dtype=float,
    )
    engine = DynamicFPDEEngine().fit(raw=raw[:2], y=[0, 1])

    selection = select_lambda_dynamic(
        engine=engine,
        raw=raw[2:],
        predict_proba=_probabilities_from_representation_mean,
        lambdas=[0.0, 1.0],
        target_classes=[0, 1],
        rival_classes=[1, 0],
        steps=3,
    )

    assert selection["status"] == "evaluated"
    assert selection["predict_proba_source"] == "callable"


def test_select_lambda_dynamic_variable_length_validation_keeps_padding_zero():
    train_raw = [
        np.array([[0.0], [0.1]], dtype=float),
        np.array([[1.0], [1.1], [1.2], [1.3]], dtype=float),
    ]
    val_raw = [
        np.array([[0.2], [0.3]], dtype=float),
        np.array([[0.8], [0.9], [1.0], [1.1]], dtype=float),
    ]
    engine = DynamicFPDEEngine().fit(raw=train_raw, y=[0, 1])

    def predict_proba(*, raw=None, features=None, dt=None, mask=None):
        assert np.all(raw[~mask] == 0.0)
        return _probabilities_from_raw_mean(raw, mask=mask)

    selection = select_lambda_dynamic(
        engine=engine,
        raw=val_raw,
        predict_proba=predict_proba,
        lambdas=[0.0, 0.5, 1.0],
        target_classes=[0, 1],
        rival_classes=[1, 0],
        steps=4,
    )

    assert selection["status"] == "evaluated"
    assert selection["n_validation"] == 2


def test_select_lambda_dynamic_all_zero_attribution_case_is_finite():
    raw = np.zeros((4, 3, 1), dtype=float)
    engine = DynamicFPDEEngine().fit(raw=raw, y=[0, 0, 1, 1])

    selection = select_lambda_dynamic(
        engine=engine,
        raw=raw[:2],
        predict_proba=np.array([[0.5, 0.5], [0.5, 0.5]], dtype=float),
        lambdas=[0.0, 0.5, 1.0],
        target_classes=[0, 0],
        rival_classes=[1, 1],
        steps=3,
    )

    assert selection["status"] == "evaluated"
    assert selection["best_lambda"] in [0.0, 0.5, 1.0]
    assert all(np.isfinite(value) for value in selection["scores"].values())


def test_select_lambda_dynamic_invalid_inputs():
    raw = np.array([[[0.0]], [[1.0]]], dtype=float)
    engine = DynamicFPDEEngine().fit(raw=raw, y=[0, 1])

    with pytest.raises(ValueError, match="lambdas"):
        select_lambda_dynamic(engine=engine, raw=raw, predict_proba=np.ones((2, 2)), lambdas=[])
    with pytest.raises(ValueError, match="lambda_hyb"):
        select_lambda_dynamic(engine=engine, raw=raw, predict_proba=np.ones((2, 2)), lambdas=[-0.1])
    with pytest.raises(ValueError, match="steps"):
        select_lambda_dynamic(engine=engine, raw=raw, predict_proba=np.ones((2, 2)), lambdas=[0.5], steps=1)
    with pytest.raises(ValueError, match="row count"):
        select_lambda_dynamic(engine=engine, raw=raw, predict_proba=np.ones((1, 2)), lambdas=[0.5])
    with pytest.raises(ValueError, match="predict_proba"):
        select_lambda_dynamic(engine=engine, raw=raw, predict_proba=lambda bad: None, lambdas=[0.5])
