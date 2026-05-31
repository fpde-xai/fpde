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
    FPDEEngine,
    class_mean_prototypes,
    explain_with_selected_prototypes,
    perturbation_curves,
    prepare_fpde_context,
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
        "fpde": ["FPDEEngine", "diff_fpde", "class_mean_prototypes", "top_two_labels"],
        "fpde.core": ["FPDEEngine", "diff_fpde", "class_mean_prototypes", "top_two_labels"],
        "fpde.explainers": ["diff_fpde", "cos_fpde", "explain_with_selected_prototypes"],
        "fpde.prototypes": ["class_mean_prototypes", "select_prototype_pair", "prepare_fpde_context"],
        "fpde.metrics": ["regularized_cosine", "top_two_labels", "perturbation_curves"],
        "fpde.engine": ["FPDEEngine"],
        "fpde.utils": ["parse_float_grid"],
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
