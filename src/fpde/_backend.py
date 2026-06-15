"""Optional array backend resolution for CPU and CUDA execution."""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any, Literal

import numpy as np

DeviceMode = Literal["cpu", "cuda", "gpu", "cupy", "auto"]


@dataclass(frozen=True)
class ArrayBackend:
    """Resolved array namespace used by vectorized FPDE internals."""

    name: Literal["cpu", "cuda"]
    xp: Any

    @property
    def is_cuda(self) -> bool:
        return self.name == "cuda"


def _load_cupy() -> Any:
    try:
        import cupy as cp  # type: ignore[import-not-found]
    except ImportError as exc:
        raise ImportError(
            "CuPy is required for device='cuda'. Install a CUDA-matched CuPy "
            "package such as cupy-cuda13x, or use device='cpu'."
        ) from exc

    try:
        device_count = int(cp.cuda.runtime.getDeviceCount())
    except Exception as exc:  # pragma: no cover - depends on local CUDA driver state
        raise RuntimeError("CuPy is installed, but no usable CUDA runtime/device was found.") from exc
    if device_count < 1:
        raise RuntimeError("CuPy is installed, but no CUDA devices were found.")
    return cp


def resolve_array_backend(device: DeviceMode = "cpu") -> ArrayBackend:
    """Resolve a user-facing device option to an array namespace."""
    if device == "cpu":
        return ArrayBackend(name="cpu", xp=np)
    if device in ("cuda", "gpu", "cupy"):
        return ArrayBackend(name="cuda", xp=_load_cupy())
    if device == "auto":
        try:
            return ArrayBackend(name="cuda", xp=_load_cupy())
        except (ImportError, RuntimeError):
            return ArrayBackend(name="cpu", xp=np)
    raise ValueError("device must be 'cpu', 'cuda', 'gpu', 'cupy', or 'auto'")


def is_numpy_namespace(xp: Any) -> bool:
    return xp is np or getattr(xp, "__name__", "") == "numpy"


def as_numpy_array(value: Any, xp: Any) -> np.ndarray:
    """Convert an array namespace value back to a NumPy array when needed."""
    if is_numpy_namespace(xp):
        return value
    return xp.asnumpy(value)


def as_numpy_dict(values: dict[str, Any], xp: Any) -> dict[str, Any]:
    """Convert array values in a dictionary back to NumPy arrays."""
    if is_numpy_namespace(xp):
        return values
    return {name: as_numpy_array(value, xp) for name, value in values.items()}


__all__ = ["ArrayBackend", "DeviceMode", "resolve_array_backend"]
