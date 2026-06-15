from __future__ import annotations

import importlib
import importlib.metadata
import pkgutil

import numpy as np
import pytest
from sklearn.datasets import make_classification
from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import train_test_split

from fpde import (
    BayesianFPDELambdaSelectionResult,
    FPDEEngine,
    DynamicFPDEContext,
    DynamicFPDEExplanation,
    class_mean_prototypes,
    compute_prototype_similarities,
    dynamic_diff_fpde,
    dynamic_fpde_explain_one,
    explain_with_selected_prototypes,
    prepare_local_contribution_data,
    plot_attribution_image,
    plot_attribution_summary,
    plot_attribution_waterfall,
    plot_attributions,
    plot_local_contributions,
    plot_perturbation_curves,
    plot_prototype_similarity_distribution,
    plot_similarity_distribution,
    prepare_dynamic_fpde_context,
    perturbation_curves,
    prepare_fpde_context,
    resample_time_series_linear,
    top_two_labels,
)


def test_installed_package_metadata_supports_python_312_plus():
    metadata = importlib.metadata.metadata("fpde")

    assert metadata["Requires-Python"] == ">=3.12"


def test_all_fpde_modules_are_importable():
    package = importlib.import_module("fpde")

    assert package.__path__
    for module in pkgutil.walk_packages(package.__path__, prefix=f"{package.__name__}."):
        importlib.import_module(module.name)


def test_public_module_import_paths_remain_available():
    expected = {
        "fpde": [
            "FPDEEngine",
            "BayesianFPDELambdaSelectionResult",
            "DynamicFPDEContext",
            "DynamicFPDEExplanation",
            "diff_fpde",
            "dynamic_diff_fpde",
            "dynamic_fpde_explain_one",
            "class_mean_prototypes",
            "top_two_labels",
            "compute_prototype_similarities",
            "prepare_local_contribution_data",
            "prepare_dynamic_fpde_context",
            "resample_time_series_linear",
            "plot_attributions",
            "plot_attribution_waterfall",
            "plot_attribution_summary",
            "plot_local_contributions",
            "plot_prototype_similarity_distribution",
            "plot_similarity_distribution",
        ],
        "fpde.core": [
            "FPDEEngine",
            "BayesianFPDELambdaSelectionResult",
            "DynamicFPDEContext",
            "DynamicFPDEExplanation",
            "diff_fpde",
            "dynamic_diff_fpde",
            "dynamic_fpde_explain_one",
            "class_mean_prototypes",
            "top_two_labels",
            "compute_prototype_similarities",
            "prepare_local_contribution_data",
            "prepare_dynamic_fpde_context",
            "resample_time_series_linear",
            "plot_attributions",
            "plot_attribution_waterfall",
            "plot_attribution_summary",
            "plot_local_contributions",
            "plot_prototype_similarity_distribution",
            "plot_similarity_distribution",
        ],
        "fpde.types": ["BayesianFPDELambdaSelectionResult"],
        "fpde.explainers": ["diff_fpde", "cos_fpde", "explain_with_selected_prototypes"],
        "fpde.prototypes": ["class_mean_prototypes", "select_prototype_pair", "prepare_fpde_context"],
        "fpde.metrics": ["regularized_cosine", "top_two_labels", "perturbation_curves"],
        "fpde.engine": ["FPDEEngine"],
        "fpde.utils": ["parse_float_grid"],
        "fpde.dynamic": [
            "DynamicFPDEContext",
            "DynamicFPDEExplanation",
            "resample_time_series_linear",
            "prepare_dynamic_fpde_context",
            "dynamic_diff_fpde",
            "dynamic_cos_fpde",
            "dynamic_hyb_fpde",
            "dynamic_fpde_explain_one",
            "dynamic_fpde_explain_batch",
            "temporal_deletion_insertion_curves",
            "select_dynamic_lambda",
            "plot_dynamic_time_importance",
            "plot_dynamic_attribution_heatmap",
        ],
        "fpde.plotting": [
            "plot_attributions",
            "plot_attribution_waterfall",
            "plot_attribution_summary",
            "plot_attribution_image",
            "plot_local_contributions",
            "plot_perturbation_curves",
            "prepare_local_contribution_data",
            "plot_prototype_similarity_distribution",
            "plot_similarity_distribution",
        ],
    }
    for module_name, names in expected.items():
        module = importlib.import_module(module_name)
        for name in names:
            assert hasattr(module, name), f"{module_name}.{name} is missing"

    removed = {
        "fpde": [
            "hyb_fpde_grid_attribution",
            "hyb_fpde_grid_attributions",
            "hyb_fpde_grid_attribution_matrix",
            "grid_search_hyb_fpde_grid",
            "select_lambda_by_deletion_insertion_validation",
            "select_hyb_fpde_grid_lambda",
            "explain_with_validation_selected_lambda",
            "explain_with_validation_selected_lambda_batch",
            "explain_hyb_fpde_grid",
        ],
        "fpde.core": [
            "hyb_fpde_grid_attribution",
            "hyb_fpde_grid_attributions",
            "hyb_fpde_grid_attribution_matrix",
            "grid_search_hyb_fpde_grid",
            "select_lambda_by_deletion_insertion_validation",
            "select_hyb_fpde_grid_lambda",
            "explain_with_validation_selected_lambda",
            "explain_with_validation_selected_lambda_batch",
            "explain_hyb_fpde_grid",
        ],
        "fpde.explainers": [
            "hyb_fpde_grid_attribution",
            "hyb_fpde_grid_attributions",
            "hyb_fpde_grid_attribution_matrix",
            "explain_hyb_fpde_grid",
        ],
        "fpde.selection": [
            "grid_search_hyb_fpde_grid",
            "select_lambda_by_deletion_insertion_validation",
            "select_hyb_fpde_grid_lambda",
            "explain_with_validation_selected_lambda",
            "explain_with_validation_selected_lambda_batch",
        ],
    }
    for module_name, names in removed.items():
        module = importlib.import_module(module_name)
        for name in names:
            assert not hasattr(module, name), f"{module_name}.{name} should have been removed"


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
            return "image" if name == "imshow" else "collection" if name == "scatter" else None

        return recorder


def test_plotting_helpers_accept_existing_axes_without_matplotlib():
    ax = _FakeAxes()
    returned = plot_attributions(
        [0.2, -0.6, 0.1],
        feature_names=["a", "b", "c"],
        top_k=2,
        ax=ax,
        title="FPDE",
    )

    assert returned is ax
    assert [name for name, _, _ in ax.calls].count("barh") == 1
    assert any(name == "axvline" for name, _, _ in ax.calls)
    labels_call = next(args for name, args, _ in ax.calls if name == "set_yticklabels")
    assert labels_call[0] == ["b", "a"]

    interval_ax = _FakeAxes()
    assert (
        plot_attributions(
            [0.2, -0.6, 0.1],
            feature_names=["a", "b", "c"],
            top_k=2,
            interval_low=[0.1, -0.8, 0.0],
            interval_high=[0.3, -0.4, 0.2],
            ax=interval_ax,
        )
        is interval_ax
    )
    assert [name for name, _, _ in interval_ax.calls].count("errorbar") == 1
    assert [name for name, _, _ in interval_ax.calls].count("legend") == 1

    curves_ax = _FakeAxes()
    curves = {
        "fractions": [0.0, 0.5, 1.0],
        "deletion_prob": [0.9, 0.5, 0.2],
        "insertion_prob": [0.1, 0.6, 0.9],
    }
    assert plot_perturbation_curves(curves, ax=curves_ax) is curves_ax
    assert [name for name, _, _ in curves_ax.calls].count("plot") == 2

    waterfall_ax = _FakeAxes()
    assert plot_attribution_waterfall([0.5, -0.25, 0.1], feature_names=["a", "b", "c"], ax=waterfall_ax) is waterfall_ax
    assert [name for name, _, _ in waterfall_ax.calls].count("barh") == 1

    summary_ax = _FakeAxes()
    attr_matrix = np.array([[0.2, -0.1, 0.4], [0.1, -0.3, 0.2]])
    values = np.array([[1.0, 2.0, 3.0], [1.5, 2.5, 3.5]])
    assert plot_attribution_summary(attr_matrix, feature_values=values, ax=summary_ax) is summary_ax
    assert [name for name, _, _ in summary_ax.calls].count("scatter") == 3
    assert len(summary_ax.figure.colorbar_calls) == 1

    similarity_ax = _FakeAxes()
    assert plot_similarity_distribution([0.1, 0.2, 0.5], target_similarity=0.4, ax=similarity_ax) is similarity_ax
    assert [name for name, _, _ in similarity_ax.calls].count("hist") == 1
    assert any(name == "axvline" for name, _, _ in similarity_ax.calls)

    prototype_ax = _FakeAxes()
    assert (
        plot_prototype_similarity_distribution(
            [[1.0, 0.0], [0.0, 1.0]],
            [1.0, 0.0],
            rival_prototype=[0.0, 1.0],
            x=[1.0, 0.0],
            ax=prototype_ax,
        )
        is prototype_ax
    )
    assert [name for name, _, _ in prototype_ax.calls].count("hist") == 2
    assert [name for name, _, _ in prototype_ax.calls].count("axvline") == 2

    local_ax = _FakeAxes()
    assert plot_local_contributions(["a", "b", "c"], [0.2, -0.5, 0.0], values=[1, 2, 3], ax=local_ax) is local_ax
    assert [name for name, _, _ in local_ax.calls].count("barh") == 1
    assert any(name == "axvline" for name, _, _ in local_ax.calls)


def test_plot_attribution_image_accepts_flat_vector_and_shape():
    ax = _FakeAxes()

    assert plot_attribution_image([1.0, -1.0, 0.5, -0.5], shape=(2, 2), ax=ax) is ax
    image_call = next(args for name, args, _ in ax.calls if name == "imshow")
    assert image_call[0].shape == (2, 2)
    assert len(ax.figure.colorbar_calls) == 1


def test_plotting_helpers_validate_inputs():
    with pytest.raises(ValueError, match="feature_names length"):
        plot_attributions([1.0, 2.0], feature_names=["only-one"], ax=_FakeAxes())
    with pytest.raises(ValueError, match="top_k"):
        plot_attributions([1.0, 2.0], top_k=0, ax=_FakeAxes())
    with pytest.raises(ValueError, match="provided together"):
        plot_attributions([1.0, 2.0], interval_low=[0.5, 1.5], ax=_FakeAxes())
    with pytest.raises(ValueError, match="interval_low length"):
        plot_attributions([1.0, 2.0], interval_low=[0.5], interval_high=[1.5], ax=_FakeAxes())
    with pytest.raises(ValueError, match="less than or equal"):
        plot_attributions([1.0, 2.0], interval_low=[1.5, 1.0], interval_high=[1.0, 3.0], ax=_FakeAxes())
    with pytest.raises(ValueError, match="normalize=True"):
        plot_attributions([1.0, 2.0], normalize=True, interval_low=[0.5, 1.5], interval_high=[1.5, 2.5], ax=_FakeAxes())
    with pytest.raises(ValueError, match="shape is required"):
        plot_attribution_image([1.0, 2.0], ax=_FakeAxes())
    with pytest.raises(ValueError, match="missing required keys"):
        plot_perturbation_curves({"fractions": [0.0]}, ax=_FakeAxes())
    with pytest.raises(ValueError, match="feature_values shape"):
        plot_attribution_summary([[1.0, 2.0]], feature_values=[[1.0]], ax=_FakeAxes())
    with pytest.raises(ValueError, match="base_value"):
        plot_attribution_waterfall([1.0], base_value=float("nan"), ax=_FakeAxes())
    with pytest.raises(ValueError, match="bins"):
        plot_similarity_distribution([0.1, 0.2], bins=0, ax=_FakeAxes())
    with pytest.raises(ValueError, match="at least one value"):
        plot_similarity_distribution([], ax=_FakeAxes())


def test_prepare_local_contribution_data_sorts_topk_by_absolute_contribution():
    rows = prepare_local_contribution_data(
        ["a", "b", "c", "d"],
        [0.2, -0.8, 0.1, 0.5],
        top_k=2,
    )

    assert [row["feature"] for row in rows] == ["b", "d"]
    assert [row["contribution"] for row in rows] == pytest.approx([-0.8, 0.5])


def test_prepare_local_contribution_data_sorts_by_signed_value():
    rows = prepare_local_contribution_data(
        ["a", "b", "c"],
        [0.2, -0.8, 0.5],
        top_k=3,
        sort_by="value",
    )

    assert [row["feature"] for row in rows] == ["c", "a", "b"]


def test_prepare_local_contribution_data_direction_and_value_display():
    rows = prepare_local_contribution_data(
        ["tempo", "centroid", "zcr"],
        [0.2, -0.1, 0.0],
        values=[132.0, 2410.5, 0.07],
        top_k=3,
    )

    assert {row["direction"] for row in rows} == {"positive", "negative", "zero"}
    assert rows[0]["display_name"] == "tempo = 132.0"
    assert rows[1]["direction_label"] == "opposes prediction"
    assert rows[2]["direction_label"] == "zero contribution"


def test_prepare_local_contribution_data_validates_inputs():
    with pytest.raises(ValueError, match="feature_names length"):
        prepare_local_contribution_data(["a"], [0.1, 0.2])
    with pytest.raises(ValueError, match="values length"):
        prepare_local_contribution_data(["a", "b"], [0.1, 0.2], values=[1])
    with pytest.raises(ValueError, match="NaN or inf"):
        prepare_local_contribution_data(["a"], [float("nan")])
    with pytest.raises(ValueError, match="top_k"):
        prepare_local_contribution_data(["a"], [0.1], top_k=0)
    with pytest.raises(ValueError, match="sort_by"):
        prepare_local_contribution_data(["a"], [0.1], sort_by="unknown")
    with pytest.raises(ValueError, match="at least one value"):
        prepare_local_contribution_data([], [])


def test_plot_local_contributions_accepts_existing_axes_and_all_zero_contributions():
    ax = _FakeAxes()

    assert plot_local_contributions(["a", "b"], [0.0, 0.0], ax=ax, show_values=False) is ax
    assert [name for name, _, _ in ax.calls].count("barh") == 1
    labels_call = next(args for name, args, _ in ax.calls if name == "set_yticklabels")
    assert labels_call[0] == ["b", "a"]


def test_plot_local_contributions_without_matplotlib_raises_clear_import_error_when_missing():
    if importlib.util.find_spec("matplotlib") is not None:
        return

    with pytest.raises(ImportError, match=r"fpde\[plot\]"):
        plot_local_contributions(["a"], [0.1])


def test_compute_prototype_similarities_cosine_basic_case():
    similarities = compute_prototype_similarities(
        [[1.0, 0.0], [0.0, 1.0], [1.0, 1.0]],
        [1.0, 0.0],
        "cosine",
    )

    np.testing.assert_allclose(similarities, [1.0, 0.0, 1.0 / np.sqrt(2.0)])


def test_compute_prototype_similarities_cosine_zero_vectors_are_finite():
    similarities = compute_prototype_similarities(
        [[0.0, 0.0], [1.0, 0.0]],
        [0.0, 0.0],
        "cosine",
    )

    np.testing.assert_allclose(similarities, [0.0, 0.0])
    assert np.all(np.isfinite(similarities))


def test_compute_prototype_similarities_negative_euclidean_basic_case():
    similarities = compute_prototype_similarities(
        [[1.0, 0.0], [1.0, 2.0]],
        [1.0, 0.0],
        "negative_euclidean",
    )

    np.testing.assert_allclose(similarities, [0.0, -2.0])


def test_compute_prototype_similarities_validates_inputs():
    with pytest.raises(ValueError, match="feature dimension"):
        compute_prototype_similarities([[1.0, 2.0]], [1.0], "cosine")
    with pytest.raises(ValueError, match="at least one sample"):
        compute_prototype_similarities(np.empty((0, 2)), [1.0, 0.0], "cosine")
    with pytest.raises(ValueError, match="at least one feature"):
        compute_prototype_similarities(np.empty((2, 0)), [], "cosine")
    with pytest.raises(ValueError, match="NaN or inf"):
        compute_prototype_similarities([[1.0, np.nan]], [1.0, 0.0], "cosine")
    with pytest.raises(ValueError, match="metric"):
        compute_prototype_similarities([[1.0, 0.0]], [1.0, 0.0], "unknown")


def test_plot_prototype_similarity_distribution_accepts_target_only():
    ax = _FakeAxes()

    assert plot_prototype_similarity_distribution([[1.0, 0.0], [0.0, 1.0]], [1.0, 0.0], ax=ax) is ax
    assert [name for name, _, _ in ax.calls].count("hist") == 1


def test_plot_prototype_similarity_distribution_validates_shapes():
    with pytest.raises(ValueError, match="feature dimension"):
        plot_prototype_similarity_distribution([[1.0, 2.0]], [1.0], ax=_FakeAxes())


def _fit_classifier(*, n_classes: int, random_state: int = 13):
    X, y = make_classification(
        n_samples=90,
        n_features=6,
        n_informative=4,
        n_redundant=0,
        n_classes=n_classes,
        n_clusters_per_class=1,
        class_sep=1.4,
        random_state=random_state,
    )
    X_train, X_test, y_train, _ = train_test_split(
        X,
        y,
        test_size=0.25,
        random_state=random_state,
        stratify=y,
    )
    clf = RandomForestClassifier(n_estimators=25, random_state=random_state)
    clf.fit(X_train, y_train)
    return X_train, y_train, X_test, clf


def _explain_first_sample(n_classes: int):
    X_train, y_train, X_test, clf = _fit_classifier(n_classes=n_classes)
    prototypes, labels = class_mean_prototypes(X_train, y_train)
    target, rival, _ = top_two_labels(clf, X_test[0])
    return explain_with_selected_prototypes(
        X_test[0],
        prototypes,
        labels,
        positive_label=target,
        negative_label=rival,
        mode="diff",
    )


def test_deterministic_outputs_with_fixed_seed():
    exp_a = _explain_first_sample(n_classes=2)
    exp_b = _explain_first_sample(n_classes=2)

    np.testing.assert_allclose(exp_a.attributions, exp_b.attributions)
    assert exp_a.evidence == pytest.approx(exp_b.evidence)


def test_attribution_shape_and_finite_values():
    exp = _explain_first_sample(n_classes=2)

    assert exp.attributions.shape == (6,)
    assert np.all(np.isfinite(exp.attributions))
    assert np.isfinite(exp.evidence)


def test_binary_classification_works():
    exp = _explain_first_sample(n_classes=2)

    assert exp.positive_label in (0, 1)
    assert exp.negative_label in (0, 1)


def test_multiclass_classification_works():
    exp = _explain_first_sample(n_classes=3)

    assert exp.attributions.shape == (6,)
    assert exp.positive_label != exp.negative_label


def test_hybrid_grid_explanation_is_finite():
    X_train, y_train, X_test, clf = _fit_classifier(n_classes=3)
    engine = FPDEEngine.fit(X_train, y_train, model=clf)

    for lambda_hyb in (0.0, 0.5, 1.0):
        attr, detail = engine.explain_one(X_test[0], lambda_hyb=lambda_hyb)
        assert attr.shape == (6,)
        assert np.all(np.isfinite(attr))
        assert np.isfinite(detail["evidence"])


def test_zero_variance_feature_does_not_crash():
    X_train, y_train, X_test, clf = _fit_classifier(n_classes=2)
    X_train = np.c_[X_train, np.ones(X_train.shape[0])]
    x = np.r_[X_test[0], 1.0]
    clf.fit(X_train, y_train)
    prototypes, labels = class_mean_prototypes(X_train, y_train)
    target, rival, _ = top_two_labels(clf, x)

    exp = explain_with_selected_prototypes(
        x,
        prototypes,
        labels,
        positive_label=target,
        negative_label=rival,
        mode="cos",
    )

    assert exp.attributions.shape == (7,)
    assert np.all(np.isfinite(exp.attributions))


def test_empty_inputs_raise_clear_value_error():
    with pytest.raises(ValueError, match="at least one sample"):
        class_mean_prototypes(np.empty((0, 3)), [])


def test_validation_lambda_selection_returns_grid_candidate():
    X_train, y_train, X_test, clf = _fit_classifier(n_classes=3)
    engine = FPDEEngine.fit(X_train, y_train, model=clf)

    selection = engine.select_lambda(
        X_test[:8],
        lambda_hyb_grid=(0.0, 0.5, 1.0),
        fractions=(0.0, 0.5, 1.0),
    )

    assert selection.best_lambda in (0.0, 0.5, 1.0)
    assert len(selection.rows) == 3
    assert all(row["status"] == "ok" for row in selection.rows)


def test_validation_lambda_selection_chunking_matches_full_batch():
    X_train, y_train, X_test, clf = _fit_classifier(n_classes=3)
    engine = FPDEEngine.fit(X_train, y_train, model=clf)
    kwargs = {
        "lambda_hyb_grid": (0.0, 0.5, 1.0),
        "fractions": (0.0, 0.5, 1.0),
    }

    full = engine.select_lambda(
        X_test[:8],
        **kwargs,
    )
    chunked = engine.select_lambda(
        X_test[:8],
        max_working_bytes=2500,
        **kwargs,
    )

    assert chunked.best_lambda == pytest.approx(full.best_lambda)
    for full_row, chunked_row in zip(full.rows, chunked.rows):
        assert chunked_row["status"] == full_row["status"] == "ok"
        assert chunked_row["score"] == pytest.approx(full_row["score"])
        assert chunked_row["mean_deletion_drop_auc"] == pytest.approx(full_row["mean_deletion_drop_auc"])


def test_validation_lambda_selection_reuses_endpoint_perturbations():
    X_train, y_train, X_test, clf = _fit_classifier(n_classes=3)

    class CountingModel:
        def __init__(self, model):
            self.model = model
            self.classes_ = model.classes_
            self.rows = 0
            self.calls = 0

        def predict_proba(self, X):
            self.calls += 1
            self.rows += X.shape[0]
            return self.model.predict_proba(X)

    wrapped = CountingModel(clf)
    engine = FPDEEngine.fit(X_train, y_train, model=wrapped)
    n_val = 6
    n_lambdas = 3
    engine.select_lambda(
        X_test[:n_val],
        lambda_hyb_grid=(0.0, 0.5, 1.0),
        fractions=(0.0, 0.5, 1.0),
    )

    old_endpoint_repeated_rows = n_val + 2 * n_lambdas * n_val * 3
    assert wrapped.rows < old_endpoint_repeated_rows
    assert wrapped.calls == 2

    raw_engine = FPDEEngine.fit(X_train, y_train, model=clf)
    endpoint_only = raw_engine.select_lambda(
        X_test[:n_val],
        lambda_hyb_grid=(0.0, 1.0),
        fractions=(0.0, 1.0),
        max_working_bytes=1,
    )
    assert endpoint_only.best_lambda in (0.0, 1.0)

    diff_only = raw_engine.select_lambda(
        X_test[:n_val],
        lambda_hyb_grid=(1.0,),
        fractions=(0.0, 0.5, 1.0),
    )
    cos_only = raw_engine.select_lambda(
        X_test[:n_val],
        lambda_hyb_grid=(0.0,),
        fractions=(0.0, 0.5, 1.0),
    )
    assert diff_only.best_lambda == pytest.approx(1.0)
    assert cos_only.best_lambda == pytest.approx(0.0)


def test_endpoint_lambdas_skip_unused_component_batches(monkeypatch):
    X_train, y_train, X_test, clf = _fit_classifier(n_classes=3)

    import fpde.engine as engine_module

    attribution_modes = []
    original_attr_components = engine_module._hyb_components_for_prototype_indices_batch

    def tracking_attr_components(*args, **kwargs):
        attribution_modes.append(kwargs.get("component_mode"))
        return original_attr_components(*args, **kwargs)

    monkeypatch.setattr(engine_module, "_hyb_components_for_prototype_indices_batch", tracking_attr_components)
    engine = FPDEEngine.fit(X_train, y_train, model=clf)
    engine.explain_matrix(X_test[:3], lambda_hyb=1.0)
    engine.explain_matrix(X_test[:3], lambda_hyb=0.0)
    assert attribution_modes == ["diff", "cos"]

    selection_modes = []

    def tracking_selection_components(*args, **kwargs):
        selection_modes.append(kwargs.get("component_mode"))
        return original_attr_components(*args, **kwargs)

    monkeypatch.setattr(engine_module, "_hyb_components_for_prototype_indices_batch", tracking_selection_components)
    engine.select_lambda(
        X_test[:3],
        lambda_hyb_grid=(1.0,),
        fractions=(0.0, 1.0),
    )
    engine.select_lambda(
        X_test[:3],
        lambda_hyb_grid=(0.0,),
        fractions=(0.0, 1.0),
    )
    assert selection_modes == ["diff", "cos"]


def test_context_matches_training_inputs_across_fast_apis():
    X_train, y_train, X_test, clf = _fit_classifier(n_classes=3)
    context = prepare_fpde_context(X_train, y_train)
    engine_ctx = FPDEEngine(X_train, y_train, model=clf, context=context)
    engine_raw = FPDEEngine.fit(X_train, y_train, model=clf)

    attr_ctx, detail_ctx = engine_ctx.explain_one(X_test[0], lambda_hyb=0.5)
    attr_raw, detail_raw = engine_raw.explain_one(X_test[0], lambda_hyb=0.5)
    np.testing.assert_allclose(attr_ctx, attr_raw)
    assert detail_ctx["evidence"] == pytest.approx(detail_raw["evidence"])

    batch_ctx, details_ctx = engine_ctx.explain_batch(X_test[:5], lambda_hyb=0.5)
    batch_raw, details_raw = engine_raw.explain_batch(X_test[:5], lambda_hyb=0.5)
    np.testing.assert_allclose(batch_ctx, batch_raw)
    assert [row["evidence"] for row in details_ctx] == pytest.approx([row["evidence"] for row in details_raw])

    matrix_ctx = engine_ctx.explain_matrix(X_test[:5], lambda_hyb=0.5)
    np.testing.assert_allclose(matrix_ctx, batch_raw)

    grid_ctx = engine_ctx.grid_search(
        X_test[:8],
        lambda_hyb_grid=(0.0, 0.5, 1.0),
    )
    grid_raw = engine_raw.grid_search(
        X_test[:8],
        lambda_hyb_grid=(0.0, 0.5, 1.0),
    )
    assert grid_ctx.best_config == grid_raw.best_config
    assert [row["score"] for row in grid_ctx.rows] == pytest.approx([row["score"] for row in grid_raw.rows])

    selection_ctx = engine_ctx.select_lambda(
        X_test[:8],
        lambda_hyb_grid=(0.0, 0.5, 1.0),
        fractions=(0.0, 0.5, 1.0),
    )
    selection_raw = engine_raw.select_lambda(
        X_test[:8],
        lambda_hyb_grid=(0.0, 0.5, 1.0),
        fractions=(0.0, 0.5, 1.0),
    )
    assert selection_ctx.best_lambda == pytest.approx(selection_raw.best_lambda)
    assert [row["score"] for row in selection_ctx.rows] == pytest.approx([row["score"] for row in selection_raw.rows])

    selected_ctx, _ = engine_ctx.explain_batch(X_test[:4], lambda_hyb=selection_ctx.best_lambda)
    selected_raw, _ = engine_raw.explain_batch(X_test[:4], lambda_hyb=selection_ctx.best_lambda)
    np.testing.assert_allclose(selected_ctx, selected_raw)


def test_engine_explain_batch_and_matrix_are_internally_consistent():
    X_train, y_train, X_test, clf = _fit_classifier(n_classes=3)
    engine = FPDEEngine.fit(X_train, y_train, model=clf)

    batch_attr, batch_details = engine.explain_batch(
        X_test[:5],
        lambda_hyb=0.5,
        normalize="l1",
        include_details=True,
    )
    single_attr, single_detail = engine.explain_one(X_test[0], lambda_hyb=0.5, normalize="l1")
    np.testing.assert_allclose(batch_attr[0], single_attr)
    assert batch_details[0]["evidence"] == pytest.approx(single_detail["evidence"])

    matrix = engine.explain_matrix(X_test[:5], lambda_hyb=0.5, normalize="l1")
    attr_no_details, details = engine.explain_batch(
        X_test[:5],
        lambda_hyb=0.5,
        normalize="l1",
        include_details=False,
    )
    np.testing.assert_allclose(matrix, attr_no_details)
    np.testing.assert_allclose(matrix, batch_attr)
    assert details == []


def test_engine_grid_search_and_select_lambda_are_deterministic():
    X_train, y_train, X_test, clf = _fit_classifier(n_classes=3)
    engine = FPDEEngine.fit(X_train, y_train, model=clf)

    grid_a = engine.grid_search(
        X_test[:8],
        lambda_hyb_grid=(0.0, 0.5, 1.0),
        normalize_grid=("l1", "none"),
    )
    grid_b = engine.grid_search(
        X_test[:8],
        lambda_hyb_grid=(0.0, 0.5, 1.0),
        normalize_grid=("l1", "none"),
    )
    assert grid_a.best_config == grid_b.best_config
    assert [row["score"] for row in grid_a.rows] == pytest.approx([row["score"] for row in grid_b.rows])

    selection_a = engine.select_lambda(
        X_test[:8],
        lambda_hyb_grid=(0.0, 0.5, 1.0),
        fractions=(0.0, 0.5, 1.0),
    )
    selection_b = engine.select_lambda(
        X_test[:8],
        lambda_hyb_grid=(0.0, 0.5, 1.0),
        fractions=(0.0, 0.5, 1.0),
    )
    assert selection_a.best_lambda == pytest.approx(selection_b.best_lambda)
    assert [row["score"] for row in selection_a.rows] == pytest.approx([row["score"] for row in selection_b.rows])


def test_engine_validation_selection_deduplicates_lambdas_but_preserves_rows():
    X_train, y_train, X_test, clf = _fit_classifier(n_classes=3)
    engine = FPDEEngine.fit(X_train, y_train, model=clf)

    selection = engine.select_lambda(
        X_test[:6],
        lambda_hyb_grid=(0.0, 0.5, 0.5, 1.0),
        fractions=(0.0, 0.5, 0.5, 1.0),
    )

    assert len(selection.rows) == 4
    assert selection.rows[1]["lambda_hyb"] == pytest.approx(0.5)
    assert selection.rows[2]["lambda_hyb"] == pytest.approx(0.5)
    assert selection.rows[1]["score"] == pytest.approx(selection.rows[2]["score"])


def test_bayesian_lambda_selection_is_deterministic_and_normalized():
    X_train, y_train, X_test, clf = _fit_classifier(n_classes=3)
    engine = FPDEEngine.fit(X_train, y_train, model=clf)
    kwargs = {
        "lambda_hyb_grid": (0.0, 0.5, 1.0),
        "fractions": (0.0, 0.5, 1.0),
    }

    selection_a = engine.select_bayesian_lambda(X_test[:6], **kwargs)
    selection_b = engine.select_bayesian_lambda(X_test[:6], **kwargs)

    assert isinstance(selection_a, BayesianFPDELambdaSelectionResult)
    assert selection_a.posterior_mean_lambda == pytest.approx(selection_b.posterior_mean_lambda)
    assert selection_a.map_lambda == pytest.approx(selection_b.map_lambda)
    assert selection_a.credible_interval == pytest.approx(selection_b.credible_interval)
    probabilities = [row["posterior_probability"] for row in selection_a.posterior_rows]
    assert sum(probabilities) == pytest.approx(1.0)
    assert 0.0 <= selection_a.posterior_mean_lambda <= 1.0
    assert 0.0 <= selection_a.credible_interval[0] <= selection_a.credible_interval[1] <= 1.0


def test_bayesian_lambda_uniform_prior_map_matches_best_validation_score():
    X_train, y_train, X_test, clf = _fit_classifier(n_classes=3)
    engine = FPDEEngine.fit(X_train, y_train, model=clf)

    selection = engine.select_bayesian_lambda(
        X_test[:6],
        lambda_hyb_grid=(0.0, 0.5, 1.0),
        fractions=(0.0, 0.5, 1.0),
        alpha=1.0,
        beta=1.0,
    )

    best_score = max(row["score"] for row in selection.posterior_rows)
    best_lambdas = {
        row["lambda_hyb"]
        for row in selection.posterior_rows
        if row["score"] == pytest.approx(best_score)
    }
    assert selection.map_lambda in best_lambdas


def test_bayesian_lambda_selection_deduplicates_grid_values():
    X_train, y_train, X_test, clf = _fit_classifier(n_classes=3)
    engine = FPDEEngine.fit(X_train, y_train, model=clf)
    kwargs = {"fractions": (0.0, 0.5, 1.0)}

    unique = engine.select_bayesian_lambda(X_test[:6], lambda_hyb_grid=(0.0, 0.5, 1.0), **kwargs)
    duplicated = engine.select_bayesian_lambda(X_test[:6], lambda_hyb_grid=(0.0, 0.5, 0.5, 1.0), **kwargs)

    assert len(duplicated.posterior_rows) == 3
    assert [row["lambda_hyb"] for row in duplicated.posterior_rows] == pytest.approx([0.0, 0.5, 1.0])
    assert duplicated.posterior_mean_lambda == pytest.approx(unique.posterior_mean_lambda)
    assert [row["posterior_probability"] for row in duplicated.posterior_rows] == pytest.approx(
        [row["posterior_probability"] for row in unique.posterior_rows]
    )


def test_bayesian_explanations_match_posterior_mean_lambda_api():
    X_train, y_train, X_test, clf = _fit_classifier(n_classes=3)
    engine = FPDEEngine.fit(X_train, y_train, model=clf)
    selection = engine.select_bayesian_lambda(
        X_test[:6],
        lambda_hyb_grid=(0.0, 0.5, 1.0),
        fractions=(0.0, 0.5, 1.0),
    )

    bayes_one, bayes_detail = engine.explain_one_bayesian(X_test[0], selection)
    fixed_one, fixed_detail = engine.explain_one(
        X_test[0],
        lambda_hyb=selection.posterior_mean_lambda,
        normalize=selection.normalize,
        anchor_strategy=selection.anchor_strategy,
        eps=selection.eps,
    )
    np.testing.assert_allclose(bayes_one, fixed_one)
    assert bayes_detail["evidence"] == pytest.approx(fixed_detail["evidence"])
    assert bayes_detail["lambda_source"] == "bayesian_posterior_mean"
    assert bayes_detail["posterior_mean_lambda"] == pytest.approx(selection.posterior_mean_lambda)

    bayes_batch, bayes_details = engine.explain_batch_bayesian(X_test[:4], selection)
    fixed_batch, _ = engine.explain_batch(
        X_test[:4],
        lambda_hyb=selection.posterior_mean_lambda,
        normalize=selection.normalize,
        anchor_strategy=selection.anchor_strategy,
        eps=selection.eps,
    )
    np.testing.assert_allclose(bayes_batch, fixed_batch)
    assert all(detail["map_lambda"] == pytest.approx(selection.map_lambda) for detail in bayes_details)
    np.testing.assert_allclose(engine.explain_matrix_bayesian(X_test[:4], selection), bayes_batch)


def test_bayesian_lambda_selection_validates_hyperparameters():
    X_train, y_train, X_test, clf = _fit_classifier(n_classes=3)
    engine = FPDEEngine.fit(X_train, y_train, model=clf)
    kwargs = {
        "lambda_hyb_grid": (0.0, 0.5, 1.0),
        "fractions": (0.0, 0.5, 1.0),
    }

    with pytest.raises(ValueError, match="alpha must be positive"):
        engine.select_bayesian_lambda(X_test[:4], alpha=0.0, **kwargs)
    with pytest.raises(ValueError, match="beta must be positive"):
        engine.select_bayesian_lambda(X_test[:4], beta=0.0, **kwargs)
    with pytest.raises(ValueError, match="temperature must be positive"):
        engine.select_bayesian_lambda(X_test[:4], temperature=0.0, **kwargs)
    with pytest.raises(ValueError, match="credible_mass must be in"):
        engine.select_bayesian_lambda(X_test[:4], credible_mass=1.0, **kwargs)


def test_perturbation_curves_reuses_endpoint_samples():
    X_train, y_train, X_test, clf = _fit_classifier(n_classes=3)

    class CountingModel:
        def __init__(self, model):
            self.model = model
            self.classes_ = model.classes_
            self.rows = 0
            self.calls = 0

        def predict_proba(self, X):
            self.calls += 1
            self.rows += X.shape[0]
            return self.model.predict_proba(X)

    engine = FPDEEngine.fit(X_train, y_train, model=clf)
    attr, detail = engine.explain_one(X_test[0], lambda_hyb=0.5)
    wrapped = CountingModel(clf)
    curves = perturbation_curves(
        wrapped,
        X_test[0],
        attr,
        detail["target_label"],
        np.mean(X_train, axis=0),
        fractions=(0.0, 0.5, 1.0),
    )

    assert len(curves["deletion_prob"]) == 3
    assert wrapped.calls == 1
    assert wrapped.rows == 4


def test_partial_topk_masks_match_full_sort_and_ties():
    from fpde.core import _full_topk_mask, _topk_masks_for_counts

    scores = np.array(
        [
            [0.9, 0.1, 0.8, 0.2, 0.7, 0.3],
            [0.1, 0.6, 0.2, 0.5, 0.3, 0.4],
        ],
        dtype=float,
    )
    ks = np.array([1, 2], dtype=np.intp)
    np.testing.assert_array_equal(_topk_masks_for_counts(scores, ks), _full_topk_mask(scores, ks))

    tied_scores = np.array(
        [
            [1.0, 1.0, 0.5, 0.4, 0.3, 0.2],
            [0.9, 0.7, 0.7, 0.6, 0.5, 0.4],
        ],
        dtype=float,
    )
    np.testing.assert_array_equal(_topk_masks_for_counts(tied_scores, ks), _full_topk_mask(tied_scores, ks))


def test_grid_search_with_predictor_scores_all_candidates():
    X_train, y_train, X_test, clf = _fit_classifier(n_classes=3)
    engine = FPDEEngine.fit(X_train, y_train, model=clf)

    result = engine.grid_search(
        X_test[:8],
        lambda_hyb_grid=(0.0, 0.5, 1.0),
        normalize_grid=("l1", "none"),
    )

    assert result.n_candidates == 10
    assert result.best_config["fpde_mode"] in {"diff", "cos", "hyb_grid"}
    assert all(row["status"] == "ok" for row in result.rows)
    assert {"l1", "none"} == {row["normalize"] for row in result.rows}


def test_engine_explain_one_matches_batch_first_row():
    X_train, y_train, X_test, clf = _fit_classifier(n_classes=3)
    engine = FPDEEngine.fit(X_train, y_train, model=clf)

    for normalize in ("l1", "none"):
        attr, detail = engine.explain_one(X_test[0], lambda_hyb=0.5, normalize=normalize)
        batch_attr, batch_details = engine.explain_batch(X_test[:1], lambda_hyb=0.5, normalize=normalize)

        np.testing.assert_allclose(attr, batch_attr[0])
        assert detail["evidence"] == pytest.approx(batch_details[0]["evidence"])
        assert detail["exactness_residual"] == pytest.approx(batch_details[0]["exactness_residual"])


def test_batched_hyb_attributions_match_single_sample_api():
    X_train, y_train, X_test, clf = _fit_classifier(n_classes=3)
    engine = FPDEEngine.fit(X_train, y_train, model=clf)

    for normalize in ("l1", "none"):
        for lambda_hyb in (0.0, 0.5, 1.0):
            batch_attr, batch_details = engine.explain_batch(
                X_test[:5],
                lambda_hyb=lambda_hyb,
                normalize=normalize,
            )

            assert batch_attr.shape == (5, X_train.shape[1])
            assert len(batch_details) == 5
            for i, x in enumerate(X_test[:5]):
                attr, detail = engine.explain_one(
                    x,
                    lambda_hyb=lambda_hyb,
                    normalize=normalize,
                )
                np.testing.assert_allclose(batch_attr[i], attr)
                assert batch_details[i]["target_label"] == detail["target_label"]
                assert batch_details[i]["rival_label"] == detail["rival_label"]
                assert batch_details[i]["evidence"] == pytest.approx(detail["evidence"])

            matrix = engine.explain_matrix(
                X_test[:5],
                lambda_hyb=lambda_hyb,
                normalize=normalize,
            )
            np.testing.assert_allclose(matrix, batch_attr)

            no_detail_attr, no_details = engine.explain_batch(
                X_test[:5],
                lambda_hyb=lambda_hyb,
                normalize=normalize,
                include_details=False,
            )
            np.testing.assert_allclose(no_detail_attr, batch_attr)
            assert no_details == []


def test_validation_selected_lambda_batch_matches_selection():
    X_train, y_train, X_test, clf = _fit_classifier(n_classes=3)
    engine = FPDEEngine.fit(X_train, y_train, model=clf)
    selection = engine.select_lambda(
        X_test[:8],
        lambda_hyb_grid=(0.0, 0.5, 1.0),
        fractions=(0.0, 0.5, 1.0),
    )

    attr, details = engine.explain_batch(
        X_test[:4],
        lambda_hyb=selection.best_lambda,
    )

    assert attr.shape == (4, X_train.shape[1])
    assert len(details) == 4
    assert all(detail["lambda_hyb"] == pytest.approx(selection.best_lambda) for detail in details)

    attr_only, no_details = engine.explain_batch(
        X_test[:4],
        lambda_hyb=selection.best_lambda,
        include_details=False,
    )
    np.testing.assert_allclose(attr_only, attr)
    assert no_details == []
