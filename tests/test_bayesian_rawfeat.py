import numpy as np
import pytest

from fpde import (
    RawFeatSequence,
    bayesian_rawfeat_dynamic_fpde_explain_one,
    build_rawfeat_matrix,
    fit_bayesian_rawfeat_prototypes,
    fit_rawfeat_scaling,
)


def _sequence(offset: float, *, mask=None) -> RawFeatSequence:
    raw = np.arange(8, dtype=float).reshape(4, 2) + offset
    features = np.arange(12, dtype=float).reshape(4, 3) * 0.1 + offset
    dt = np.array([0.0, 0.01, 0.01, 0.01])
    if mask is None:
        mask = np.ones(4, dtype=bool)
    return RawFeatSequence(raw=raw, features=features, dt=dt, mask=np.asarray(mask))


def _posterior():
    sequences = [_sequence(0.0), _sequence(0.2), _sequence(2.0), _sequence(2.2)]
    return fit_bayesian_rawfeat_prototypes(sequences, ["like", "like", "dislike", "dislike"], 16, random_state=2)


def test_build_rawfeat_matrix_preserves_native_shape_and_masks_invalid_frames():
    sequence = _sequence(0.0, mask=[True, False, True, False])
    sequence.raw[1] = np.nan
    X = build_rawfeat_matrix(sequence)

    assert X.shape == (4, 6)
    np.testing.assert_allclose(X[~sequence.mask], 0.0)
    np.testing.assert_allclose(X[sequence.mask, -1], sequence.dt[sequence.mask])


def test_bayesian_rawfeat_posterior_summary_shapes_groups_and_exactness():
    posterior = _posterior()
    result = bayesian_rawfeat_dynamic_fpde_explain_one(
        _sequence(0.1, mask=[True, True, False, True]),
        posterior,
        target_label="like",
        rival_label="dislike",
        lambda_hyb=np.linspace(0.2, 0.8, posterior.n_samples),
    )

    assert posterior.prototypes.shape == (16, 2, 6)
    for summary in (result.diff, result.cos, result.hyb):
        assert summary.posterior_mean.shape == (4, 6)
        assert summary.credible_interval_lower.shape == (4, 6)
        assert summary.credible_interval_upper.shape == (4, 6)
        assert summary.probability_positive.shape == (4, 6)
        assert summary.sign_stability.shape == (4, 6)
        assert summary.evidence.posterior.shape == (16,)
        np.testing.assert_allclose(summary.exactness_residuals, 0.0, atol=1e-12)
        np.testing.assert_allclose(summary.attribution_posterior[:, 2], 0.0)
        group_sum = (
            summary.raw_group_attribution.posterior
            + summary.feature_group_attribution.posterior
            + summary.dt_group_attribution.posterior
        )
        np.testing.assert_allclose(group_sum, summary.evidence.posterior, atol=1e-12)


@pytest.mark.parametrize("mode", ["none", "group_l1", "group_l2", "standard"])
def test_rawfeat_scaling_is_fitted_on_training_valid_frames_and_reused(mode):
    train = [_sequence(0.0, mask=[True, False, True, True]), _sequence(2.0)]
    scaling = fit_rawfeat_scaling(train, scaling=mode)
    posterior = fit_bayesian_rawfeat_prototypes(train, [0, 1], 3, scaling=mode, random_state=0)
    explained = _sequence(1000.0, mask=[True, False, True, True])

    np.testing.assert_allclose(posterior.scaling.offset, scaling.offset)
    np.testing.assert_allclose(posterior.scaling.scale, scaling.scale)
    transformed = build_rawfeat_matrix(explained, scaling=posterior.scaling)
    expected = build_rawfeat_matrix(explained)
    expected[explained.mask] = (
        expected[explained.mask] - scaling.offset
    ) / scaling.scale
    np.testing.assert_allclose(transformed, expected)
    np.testing.assert_allclose(transformed[~explained.mask], 0.0)
    assert transformed.shape == (4, 6)


def test_group_scaling_equalizes_training_mean_group_norms():
    train = [_sequence(0.0), _sequence(2.0)]
    for mode, order in (("group_l1", 1), ("group_l2", 2)):
        scaling = fit_rawfeat_scaling(train, scaling=mode)
        transformed = np.concatenate([build_rawfeat_matrix(item, scaling=scaling) for item in train])
        for group_slice in (slice(0, 2), slice(2, 5), slice(5, 6)):
            if order == 1:
                norms = np.sum(np.abs(transformed[:, group_slice]), axis=1)
            else:
                norms = np.linalg.norm(transformed[:, group_slice], axis=1)
            assert np.mean(norms) == pytest.approx(1.0)


def test_lambda_hyb_scalar_and_per_draw_arrays():
    posterior = _posterior()
    sequence = _sequence(0.1)
    scalar = bayesian_rawfeat_dynamic_fpde_explain_one(
        sequence, posterior, target_label="like", rival_label="dislike", lambda_hyb=0.3
    )
    per_draw = bayesian_rawfeat_dynamic_fpde_explain_one(
        sequence,
        posterior,
        target_label="like",
        rival_label="dislike",
        lambda_hyb=np.full(posterior.n_samples, 0.3),
    )

    np.testing.assert_allclose(scalar.lambda_hyb_posterior, 0.3)
    np.testing.assert_allclose(per_draw.lambda_hyb_posterior, 0.3)
    np.testing.assert_allclose(scalar.hyb.attribution_posterior, per_draw.hyb.attribution_posterior)


def test_invalid_lambda_labels_and_input_dimensions_are_rejected():
    posterior = _posterior()
    sequence = _sequence(0.1)
    with pytest.raises(ValueError, match="lambda_hyb must be scalar or have shape"):
        bayesian_rawfeat_dynamic_fpde_explain_one(
            sequence, posterior, target_label="like", rival_label="dislike", lambda_hyb=np.ones(2)
        )
    with pytest.raises(ValueError, match="must differ"):
        bayesian_rawfeat_dynamic_fpde_explain_one(
            sequence, posterior, target_label="like", rival_label="like"
        )
    wrong_dimension = RawFeatSequence(
        raw=np.ones((4, 3)),
        features=np.ones((4, 3)),
        dt=np.ones(4),
        mask=np.ones(4, dtype=bool),
    )
    with pytest.raises(ValueError, match="RawFeat dimension mismatch"):
        bayesian_rawfeat_dynamic_fpde_explain_one(
            wrong_dimension, posterior, target_label="like", rival_label="dislike"
        )


def test_all_false_explanation_mask_returns_finite_zero_summaries():
    posterior = _posterior()
    sequence = _sequence(0.0, mask=np.zeros(4, dtype=bool))
    sequence.raw[:] = np.nan
    result = bayesian_rawfeat_dynamic_fpde_explain_one(
        sequence, posterior, target_label="like", rival_label="dislike"
    )

    for summary in (result.diff, result.cos, result.hyb):
        assert np.all(np.isfinite(summary.attribution_posterior))
        assert np.all(np.isfinite(summary.evidence_posterior))
        np.testing.assert_allclose(summary.attribution_posterior, 0.0)
        np.testing.assert_allclose(summary.evidence_posterior, 0.0)
        np.testing.assert_allclose(summary.exactness_residuals, 0.0)


def test_bayesian_rawfeat_zero_constant_inputs_are_finite():
    zero = RawFeatSequence(
        raw=np.zeros((3, 2)),
        features=np.zeros((3, 1)),
        dt=np.zeros(3),
        mask=np.ones(3, dtype=bool),
    )
    posterior = fit_bayesian_rawfeat_prototypes([zero, zero], [0, 1], 5, random_state=0)
    result = bayesian_rawfeat_dynamic_fpde_explain_one(zero, posterior, target_label=0, rival_label=1)

    for summary in (result.diff, result.cos, result.hyb):
        assert np.all(np.isfinite(summary.attribution_posterior))
        assert np.all(np.isfinite(summary.evidence.posterior))
        np.testing.assert_allclose(summary.evidence.posterior, summary.attribution_posterior.sum(axis=(1, 2)))


def test_sample_equal_prevents_long_sequences_from_dominating_prototype():
    def constant_sequence(value, length):
        return RawFeatSequence(
            raw=np.full((length, 1), value, dtype=float),
            features=np.zeros((length, 1), dtype=float),
            dt=np.zeros(length, dtype=float),
            mask=np.ones(length, dtype=bool),
        )

    sequences = [
        constant_sequence(0.0, 1),
        constant_sequence(10.0, 9),
        constant_sequence(20.0, 1),
        constant_sequence(20.0, 1),
    ]
    labels = [0, 0, 1, 1]
    frame_equal = fit_bayesian_rawfeat_prototypes(
        sequences,
        labels,
        1,
        prototype_stat="mean",
        random_state=1,
        frame_weighting="frame_equal",
    )
    sample_equal = fit_bayesian_rawfeat_prototypes(
        sequences,
        labels,
        1,
        prototype_stat="mean",
        random_state=1,
        frame_weighting="sample_equal",
    )

    assert frame_equal.prototypes[0, 0, 0] == pytest.approx(9.0)
    assert sample_equal.prototypes[0, 0, 0] == pytest.approx(5.0)


@pytest.mark.parametrize(
    ("field", "value", "message"),
    [
        ("raw", np.ones(3), "raw must be 2D"),
        ("features", np.ones(3), "features must be 2D"),
        ("dt", np.ones((3, 1)), "dt must be 1D"),
        ("mask", np.ones((3, 1)), "mask must be 1D"),
        ("dt", np.ones(2), "time dimensions must match"),
    ],
)
def test_build_rawfeat_matrix_rejects_invalid_shapes(field, value, message):
    data = dict(raw=np.ones((3, 2)), features=np.ones((3, 1)), dt=np.ones(3), mask=np.ones(3, dtype=bool))
    data[field] = value
    with pytest.raises(ValueError, match=message):
        build_rawfeat_matrix(RawFeatSequence(**data))


def test_build_rawfeat_matrix_rejects_nonfinite_values_only_when_valid():
    allowed = _sequence(0.0, mask=[True, False, True, True])
    allowed.features[1, 0] = np.inf
    build_rawfeat_matrix(allowed)

    rejected = _sequence(0.0)
    rejected.dt[2] = np.nan
    with pytest.raises(ValueError, match="dt contains"):
        build_rawfeat_matrix(rejected)
