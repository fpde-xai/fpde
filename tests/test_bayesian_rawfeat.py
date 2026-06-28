import numpy as np
import pytest

from fpde import (
    RawFeatSequence,
    bayesian_rawfeat_dynamic_fpde_explain_one,
    build_rawfeat_matrix,
    fit_bayesian_rawfeat_prototypes,
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
