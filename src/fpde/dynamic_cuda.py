"""Optional CuPy implementations for Dynamic-FPDE tensor operations.

Feature extraction and temporal resampling remain CPU-side.  This module is
intended for already-resampled Dynamic-FPDE tensors with shape ``(T, F)`` or
batched tensors with shape ``(N, T, F)``.
"""

from __future__ import annotations

from typing import Any, Dict, Optional, Sequence, Tuple

import numpy as np


def _cupy() -> Any:
    try:
        import cupy as cp  # type: ignore[import-not-found]
    except ImportError as exc:
        raise ImportError(
            "CuPy is required for Dynamic-FPDE CUDA functions. Install "
            "`fpde[cuda12]`, `fpde[cuda13]`, or a CUDA-matched CuPy package."
        ) from exc

    try:
        device_count = int(cp.cuda.runtime.getDeviceCount())
    except Exception as exc:  # pragma: no cover - depends on local CUDA driver state
        raise RuntimeError("CuPy is installed, but no usable CUDA runtime/device was found.") from exc
    if device_count < 1:
        raise RuntimeError("CuPy is installed, but no CUDA devices were found.")
    return cp


def _truth(value: Any) -> bool:
    return bool(value.item()) if hasattr(value, "item") else bool(value)


def _validate_eps(eps: float) -> float:
    value = float(eps)
    if not np.isfinite(value) or value <= 0.0:
        raise ValueError("eps must be positive")
    return value


def _validate_lambda_hyb(lambda_hyb: float) -> float:
    value = float(lambda_hyb)
    if not np.isfinite(value):
        raise ValueError("lambda_hyb must be finite")
    if value < 0.0 or value > 1.0:
        raise ValueError("lambda_hyb must be in [0, 1]")
    return value


def _as_gpu_tensor(name: str, x: Any, cp: Any) -> Any:
    arr = cp.asarray(x, dtype=float)
    if arr.ndim not in (2, 3):
        raise ValueError(f"{name} must have shape (T, F) or (N, T, F), got shape={arr.shape}")
    if any(int(dim) == 0 for dim in arr.shape):
        raise ValueError(f"{name} must be non-empty in every dimension")
    if not _truth(cp.all(cp.isfinite(arr))):
        raise ValueError(f"{name} contains NaN or inf")
    return arr


def _match_dynamic_shape(name: str, x: Any, reference: Any, cp: Any) -> Any:
    arr = _as_gpu_tensor(name, x, cp)
    if arr.shape == reference.shape:
        return arr
    if reference.ndim == 3 and arr.ndim == 2 and arr.shape == reference.shape[1:]:
        return cp.broadcast_to(arr[None, :, :], reference.shape)
    raise ValueError(f"shape mismatch for {name}: expected {reference.shape}, got {arr.shape}")


def _maybe_numpy(value: Any, cp: Any, *, return_numpy: bool) -> Any:
    if not return_numpy:
        return value
    if getattr(value, "ndim", None) == 0:
        return float(value.item())
    return cp.asnumpy(value)


def _maybe_numpy_details(details: Dict[str, Any], cp: Any, *, return_numpy: bool) -> Dict[str, Any]:
    if not return_numpy:
        return details
    out: Dict[str, Any] = {}
    for key, value in details.items():
        if hasattr(value, "shape"):
            out[key] = _maybe_numpy(value, cp, return_numpy=True)
        else:
            out[key] = value
    return out


def _sum_dynamic(values: Any, cp: Any) -> Any:
    if values.ndim == 2:
        return cp.sum(values)
    return cp.sum(values, axis=(1, 2))


def _regularized_norm(values: Any, cp: Any, eps: float) -> Any:
    if values.ndim == 2:
        return cp.sqrt(cp.sum(values * values) + eps)
    return cp.sqrt(cp.sum(values * values, axis=(1, 2)) + eps)[:, None, None]


def _l1_normalized(values: Any, cp: Any, eps: float) -> Tuple[Any, Any]:
    if values.ndim == 2:
        scale = cp.sum(cp.abs(values)) + eps
        return values / scale, scale
    scale = cp.sum(cp.abs(values), axis=(1, 2)) + eps
    return values / scale[:, None, None], scale


def dynamic_diff_fpde_gpu(
    X: Any,
    P_target: Any,
    P_rival: Any,
    *,
    return_numpy: bool = True,
) -> Tuple[Any, Any]:
    """Compute Dynamic-Diff-FPDE on CUDA for ``(T, F)`` or ``(N, T, F)`` tensors."""
    cp = _cupy()
    X_arr = _as_gpu_tensor("X", X, cp)
    target_arr = _match_dynamic_shape("P_target", P_target, X_arr, cp)
    rival_arr = _match_dynamic_shape("P_rival", P_rival, X_arr, cp)

    attr = (X_arr - rival_arr) ** 2 - (X_arr - target_arr) ** 2
    evidence = _sum_dynamic(attr, cp)
    return (
        _maybe_numpy(attr, cp, return_numpy=return_numpy),
        _maybe_numpy(evidence, cp, return_numpy=return_numpy),
    )


def dynamic_cos_fpde_gpu(
    X: Any,
    P_target: Any,
    P_rival: Any,
    *,
    anchor: Optional[Any] = None,
    eps: float = 1e-12,
    return_numpy: bool = True,
) -> Tuple[Any, Any]:
    """Compute Dynamic-Cos-FPDE on CUDA for ``(T, F)`` or ``(N, T, F)`` tensors."""
    cp = _cupy()
    eps_value = _validate_eps(eps)
    X_arr = _as_gpu_tensor("X", X, cp)
    target_arr = _match_dynamic_shape("P_target", P_target, X_arr, cp)
    rival_arr = _match_dynamic_shape("P_rival", P_rival, X_arr, cp)
    if anchor is None:
        anchor_arr = cp.zeros_like(X_arr, dtype=float)
    else:
        anchor_arr = _match_dynamic_shape("anchor", anchor, X_arr, cp)

    z = X_arr - anchor_arr
    q_target = target_arr - anchor_arr
    q_rival = rival_arr - anchor_arr
    target_part = (z * q_target) / (_regularized_norm(z, cp, eps_value) * _regularized_norm(q_target, cp, eps_value))
    rival_part = (z * q_rival) / (_regularized_norm(z, cp, eps_value) * _regularized_norm(q_rival, cp, eps_value))
    attr = target_part - rival_part
    evidence = _sum_dynamic(attr, cp)
    return (
        _maybe_numpy(attr, cp, return_numpy=return_numpy),
        _maybe_numpy(evidence, cp, return_numpy=return_numpy),
    )


def dynamic_hyb_fpde_gpu(
    X: Any,
    P_target: Any,
    P_rival: Any,
    *,
    lambda_hyb: float = 0.5,
    normalize: str = "l1",
    anchor: Optional[Any] = None,
    eps: float = 1e-12,
    return_numpy: bool = True,
) -> Tuple[Any, Any, Dict[str, Any]]:
    """Compute Dynamic-Hyb-FPDE on CUDA for ``(T, F)`` or ``(N, T, F)`` tensors."""
    cp = _cupy()
    eps_value = _validate_eps(eps)
    lambda_value = _validate_lambda_hyb(lambda_hyb)
    if normalize not in ("l1", "none"):
        raise ValueError("normalize must be either 'l1' or 'none'")

    X_arr = _as_gpu_tensor("X", X, cp)
    target_arr = _match_dynamic_shape("P_target", P_target, X_arr, cp)
    rival_arr = _match_dynamic_shape("P_rival", P_rival, X_arr, cp)
    if anchor is None:
        anchor_arr = cp.zeros_like(X_arr, dtype=float)
    else:
        anchor_arr = _match_dynamic_shape("anchor", anchor, X_arr, cp)

    diff_attr = (X_arr - rival_arr) ** 2 - (X_arr - target_arr) ** 2
    diff_evidence = _sum_dynamic(diff_attr, cp)

    z = X_arr - anchor_arr
    q_target = target_arr - anchor_arr
    q_rival = rival_arr - anchor_arr
    target_part = (z * q_target) / (_regularized_norm(z, cp, eps_value) * _regularized_norm(q_target, cp, eps_value))
    rival_part = (z * q_rival) / (_regularized_norm(z, cp, eps_value) * _regularized_norm(q_rival, cp, eps_value))
    cos_attr = target_part - rival_part
    cos_evidence = _sum_dynamic(cos_attr, cp)

    if normalize == "l1":
        diff_part, diff_scale = _l1_normalized(diff_attr, cp, eps_value)
        cos_part, cos_scale = _l1_normalized(cos_attr, cp, eps_value)
    else:
        diff_part, cos_part = diff_attr, cos_attr
        if X_arr.ndim == 2:
            diff_scale = cp.asarray(1.0, dtype=float)
            cos_scale = cp.asarray(1.0, dtype=float)
        else:
            diff_scale = cp.ones((X_arr.shape[0],), dtype=float)
            cos_scale = cp.ones((X_arr.shape[0],), dtype=float)

    attr = lambda_value * diff_part + (1.0 - lambda_value) * cos_part
    evidence = _sum_dynamic(attr, cp)
    details = {
        "diff_evidence": diff_evidence,
        "cos_evidence": cos_evidence,
        "lambda_hyb": float(lambda_value),
        "normalize": normalize,
        "diff_scale": diff_scale,
        "cos_scale": cos_scale,
    }
    return (
        _maybe_numpy(attr, cp, return_numpy=return_numpy),
        _maybe_numpy(evidence, cp, return_numpy=return_numpy),
        _maybe_numpy_details(details, cp, return_numpy=return_numpy),
    )


__all__ = [
    "dynamic_diff_fpde_gpu",
    "dynamic_cos_fpde_gpu",
    "dynamic_hyb_fpde_gpu",
]
