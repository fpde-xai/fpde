from __future__ import annotations

import numpy as np
import pytest

from fpde import dynamic_cos_fpde, dynamic_diff_fpde, dynamic_hyb_fpde
from fpde.dynamic_cuda import dynamic_cos_fpde_gpu, dynamic_diff_fpde_gpu, dynamic_hyb_fpde_gpu


def _cupy_or_skip():
    cp = pytest.importorskip("cupy")
    try:
        if int(cp.cuda.runtime.getDeviceCount()) < 1:
            pytest.skip("CUDA device is unavailable")
    except Exception as exc:
        pytest.skip(f"CUDA runtime is unavailable: {exc}")
    return cp


def _sample_inputs():
    X = np.array([[0.0, 1.0], [2.0, 3.0], [4.0, 5.0]], dtype=float)
    target = X + np.array([[0.1, -0.2], [0.3, 0.1], [-0.2, 0.4]], dtype=float)
    rival = X + np.array([[1.0, 0.5], [-0.5, 1.2], [0.7, -0.8]], dtype=float)
    anchor = np.full_like(X, 0.25, dtype=float)
    return X, target, rival, anchor


def _batched_inputs():
    X, target, rival, anchor = _sample_inputs()
    X_batch = np.stack([X, X + 0.5], axis=0)
    target_batch = np.stack([target, target + 0.25], axis=0)
    rival_batch = np.stack([rival, rival - 0.1], axis=0)
    anchor_batch = np.stack([anchor, anchor + 0.05], axis=0)
    return X_batch, target_batch, rival_batch, anchor_batch


def test_dynamic_diff_gpu_matches_cpu_for_matrix_and_batch():
    _cupy_or_skip()
    X, target, rival, _ = _sample_inputs()

    cpu_attr, cpu_evidence = dynamic_diff_fpde(X, target, rival)
    gpu_attr, gpu_evidence = dynamic_diff_fpde_gpu(X, target, rival)

    np.testing.assert_allclose(gpu_attr, cpu_attr)
    assert gpu_evidence == pytest.approx(cpu_evidence)

    X_batch, target_batch, rival_batch, _ = _batched_inputs()
    gpu_batch_attr, gpu_batch_evidence = dynamic_diff_fpde_gpu(X_batch, target_batch, rival_batch)

    expected_attr = []
    expected_evidence = []
    for i in range(X_batch.shape[0]):
        attr, evidence = dynamic_diff_fpde(X_batch[i], target_batch[i], rival_batch[i])
        expected_attr.append(attr)
        expected_evidence.append(evidence)
    np.testing.assert_allclose(gpu_batch_attr, np.stack(expected_attr, axis=0))
    np.testing.assert_allclose(gpu_batch_evidence, np.asarray(expected_evidence, dtype=float))


def test_dynamic_cos_gpu_matches_cpu_for_matrix_and_batch():
    _cupy_or_skip()
    X, target, rival, anchor = _sample_inputs()

    cpu_attr, cpu_evidence = dynamic_cos_fpde(X, target, rival, anchor=anchor)
    gpu_attr, gpu_evidence = dynamic_cos_fpde_gpu(X, target, rival, anchor=anchor)

    np.testing.assert_allclose(gpu_attr, cpu_attr)
    assert gpu_evidence == pytest.approx(cpu_evidence)

    X_batch, target_batch, rival_batch, anchor_batch = _batched_inputs()
    gpu_batch_attr, gpu_batch_evidence = dynamic_cos_fpde_gpu(
        X_batch,
        target_batch,
        rival_batch,
        anchor=anchor_batch,
    )

    expected_attr = []
    expected_evidence = []
    for i in range(X_batch.shape[0]):
        attr, evidence = dynamic_cos_fpde(X_batch[i], target_batch[i], rival_batch[i], anchor=anchor_batch[i])
        expected_attr.append(attr)
        expected_evidence.append(evidence)
    np.testing.assert_allclose(gpu_batch_attr, np.stack(expected_attr, axis=0))
    np.testing.assert_allclose(gpu_batch_evidence, np.asarray(expected_evidence, dtype=float))


def test_dynamic_hyb_gpu_matches_cpu_for_matrix_and_batch():
    _cupy_or_skip()
    X, target, rival, anchor = _sample_inputs()

    cpu_attr, cpu_evidence, cpu_details = dynamic_hyb_fpde(
        X,
        target,
        rival,
        lambda_hyb=0.35,
        anchor=anchor,
    )
    gpu_attr, gpu_evidence, gpu_details = dynamic_hyb_fpde_gpu(
        X,
        target,
        rival,
        lambda_hyb=0.35,
        anchor=anchor,
    )

    np.testing.assert_allclose(gpu_attr, cpu_attr)
    assert gpu_evidence == pytest.approx(cpu_evidence)
    assert gpu_details["diff_scale"] == pytest.approx(cpu_details["diff_scale"])
    assert gpu_details["cos_scale"] == pytest.approx(cpu_details["cos_scale"])

    X_batch, target_batch, rival_batch, anchor_batch = _batched_inputs()
    gpu_batch_attr, gpu_batch_evidence, gpu_batch_details = dynamic_hyb_fpde_gpu(
        X_batch,
        target_batch,
        rival_batch,
        lambda_hyb=0.35,
        anchor=anchor_batch,
    )

    expected_attr = []
    expected_evidence = []
    expected_diff_scale = []
    expected_cos_scale = []
    for i in range(X_batch.shape[0]):
        attr, evidence, details = dynamic_hyb_fpde(
            X_batch[i],
            target_batch[i],
            rival_batch[i],
            lambda_hyb=0.35,
            anchor=anchor_batch[i],
        )
        expected_attr.append(attr)
        expected_evidence.append(evidence)
        expected_diff_scale.append(details["diff_scale"])
        expected_cos_scale.append(details["cos_scale"])
    np.testing.assert_allclose(gpu_batch_attr, np.stack(expected_attr, axis=0))
    np.testing.assert_allclose(gpu_batch_evidence, np.asarray(expected_evidence, dtype=float))
    np.testing.assert_allclose(gpu_batch_details["diff_scale"], np.asarray(expected_diff_scale, dtype=float))
    np.testing.assert_allclose(gpu_batch_details["cos_scale"], np.asarray(expected_cos_scale, dtype=float))


def test_dynamic_hyb_gpu_matches_cpu_without_normalization_and_broadcasts_prototypes():
    _cupy_or_skip()
    X, target, rival, anchor = _sample_inputs()
    X_batch = np.stack([X, X + 0.5], axis=0)

    gpu_attr, gpu_evidence, _ = dynamic_hyb_fpde_gpu(
        X_batch,
        target,
        rival,
        lambda_hyb=0.8,
        normalize="none",
        anchor=anchor,
    )

    expected_attr = []
    expected_evidence = []
    for i in range(X_batch.shape[0]):
        attr, evidence, _ = dynamic_hyb_fpde(
            X_batch[i],
            target,
            rival,
            lambda_hyb=0.8,
            normalize="none",
            anchor=anchor,
        )
        expected_attr.append(attr)
        expected_evidence.append(evidence)
    np.testing.assert_allclose(gpu_attr, np.stack(expected_attr, axis=0))
    np.testing.assert_allclose(gpu_evidence, np.asarray(expected_evidence, dtype=float))


def test_dynamic_cuda_rejects_invalid_shape_nan_inf_and_lambda():
    _cupy_or_skip()
    X, target, rival, _ = _sample_inputs()

    with pytest.raises(ValueError, match="shape"):
        dynamic_diff_fpde_gpu(np.ones((1, 2, 3, 4)), target, rival)
    with pytest.raises(ValueError, match="shape mismatch"):
        dynamic_diff_fpde_gpu(X, target[:-1], rival)
    with pytest.raises(ValueError, match="NaN or inf"):
        dynamic_diff_fpde_gpu(np.array([[np.nan, 1.0], [2.0, 3.0]]), target[:2], rival[:2])
    with pytest.raises(ValueError, match="NaN or inf"):
        dynamic_cos_fpde_gpu(X, target, np.full_like(rival, np.inf))
    with pytest.raises(ValueError, match="lambda_hyb"):
        dynamic_hyb_fpde_gpu(X, target, rival, lambda_hyb=1.5)
    with pytest.raises(ValueError, match="lambda_hyb"):
        dynamic_hyb_fpde_gpu(X, target, rival, lambda_hyb=float("nan"))


def test_dynamic_cuda_return_numpy_false_returns_cupy_arrays():
    cp = _cupy_or_skip()
    X, target, rival, anchor = _sample_inputs()

    attr, evidence = dynamic_diff_fpde_gpu(cp.asarray(X), cp.asarray(target), cp.asarray(rival), return_numpy=False)
    assert isinstance(attr, cp.ndarray)
    assert isinstance(evidence, cp.ndarray)
    assert attr.shape == X.shape
    assert evidence.shape == ()

    cos_attr, cos_evidence = dynamic_cos_fpde_gpu(
        cp.asarray(X),
        cp.asarray(target),
        cp.asarray(rival),
        anchor=cp.asarray(anchor),
        return_numpy=False,
    )
    assert isinstance(cos_attr, cp.ndarray)
    assert isinstance(cos_evidence, cp.ndarray)
    assert cos_attr.shape == X.shape
    assert cos_evidence.shape == ()

    X_batch, target_batch, rival_batch, anchor_batch = _batched_inputs()
    hyb_attr, hyb_evidence, details = dynamic_hyb_fpde_gpu(
        cp.asarray(X_batch),
        cp.asarray(target_batch),
        cp.asarray(rival_batch),
        anchor=cp.asarray(anchor_batch),
        return_numpy=False,
    )
    assert isinstance(hyb_attr, cp.ndarray)
    assert isinstance(hyb_evidence, cp.ndarray)
    assert isinstance(details["diff_scale"], cp.ndarray)
    assert hyb_attr.shape == X_batch.shape
    assert hyb_evidence.shape == (X_batch.shape[0],)
