"""Input validation and padding helpers for RawFeat Dynamic-FPDE."""

from __future__ import annotations

from dataclasses import dataclass
from typing import Dict, Optional, Sequence

import numpy as np


@dataclass(frozen=True)
class DynamicInputBatch:
    """Normalized RawFeat sequence inputs."""

    raw: np.ndarray
    features: Optional[np.ndarray]
    dt: Optional[np.ndarray]
    mask: np.ndarray
    feature_slices: Dict[str, slice]

    @property
    def representation(self) -> np.ndarray:
        parts = [self.raw]
        if self.features is not None:
            parts.append(self.features)
        if self.dt is not None:
            parts.append(self.dt)
        return np.concatenate(parts, axis=2)


def _as_sequence_array(name: str, value: object) -> np.ndarray:
    arr = np.asarray(value, dtype=float)
    if arr.ndim != 2:
        raise ValueError(f"{name} entries must be 2D arrays, got shape={arr.shape}")
    if arr.shape[0] == 0:
        raise ValueError(f"{name} entries must contain at least one time step")
    if arr.shape[1] == 0:
        raise ValueError(f"{name} entries must contain at least one channel")
    if not np.all(np.isfinite(arr)):
        raise ValueError(f"{name} contains NaN or inf")
    return arr


def _is_sequence_collection(value: object) -> bool:
    return isinstance(value, (list, tuple))


def pad_sequences(sequences: Sequence[np.ndarray | Sequence[Sequence[float]]], value: float = 0.0) -> tuple[np.ndarray, np.ndarray]:
    """Pad a list of ``(T_i, C)`` arrays to ``(N, T_max, C)``.

    Returns the padded values and a boolean mask with shape ``(N, T_max)``.
    """
    items = list(sequences)
    if not items:
        raise ValueError("sequences must contain at least one sequence")
    arrays = [_as_sequence_array(f"sequences[{i}]", item) for i, item in enumerate(items)]
    n_channels = int(arrays[0].shape[1])
    for i, arr in enumerate(arrays):
        if arr.shape[1] != n_channels:
            raise ValueError(
                f"inconsistent channel dimension at sequences[{i}]: expected {n_channels}, got {arr.shape[1]}"
            )
    max_length = max(int(arr.shape[0]) for arr in arrays)
    padded = np.full((len(arrays), max_length, n_channels), float(value), dtype=float)
    mask = np.zeros((len(arrays), max_length), dtype=bool)
    for i, arr in enumerate(arrays):
        length = int(arr.shape[0])
        padded[i, :length, :] = arr
        mask[i, :length] = True
    return padded, mask


def _fixed_array(name: str, value: object) -> np.ndarray:
    arr = np.asarray(value, dtype=float)
    if arr.ndim == 2:
        arr = arr[None, :, :]
    if arr.ndim != 3:
        raise ValueError(f"{name} must be a 3D array or one 2D sample, got shape={arr.shape}")
    if arr.shape[0] == 0 or arr.shape[1] == 0 or arr.shape[2] == 0:
        raise ValueError(f"{name} must have non-empty sample, time, and channel dimensions")
    if not np.all(np.isfinite(arr)):
        raise ValueError(f"{name} contains NaN or inf")
    return arr.astype(float, copy=True)


def _optional_array(
    name: str,
    value: object,
    *,
    n_samples: int,
    n_steps: int,
) -> Optional[np.ndarray]:
    if value is None:
        return None
    arr = _fixed_array(name, value)
    if arr.shape[0] != n_samples or arr.shape[1] != n_steps:
        raise ValueError(
            f"{name} shape mismatch: expected first two dimensions {(n_samples, n_steps)}, got {arr.shape[:2]}"
        )
    return arr


def _optional_list(
    name: str,
    value: object,
    raw_lengths: list[int],
) -> Optional[np.ndarray]:
    if value is None:
        return None
    if not _is_sequence_collection(value):
        raise ValueError(f"{name} must be a list or tuple when raw is a list or tuple")
    items = list(value)  # type: ignore[arg-type]
    if len(items) != len(raw_lengths):
        raise ValueError(f"number of {name} sequences differs from raw: {len(items)} vs {len(raw_lengths)}")
    arrays = [_as_sequence_array(f"{name}[{i}]", item) for i, item in enumerate(items)]
    for i, (arr, length) in enumerate(zip(arrays, raw_lengths)):
        if arr.shape[0] != length:
            raise ValueError(f"{name}[{i}] length mismatch: expected {length}, got {arr.shape[0]}")
    padded, _ = pad_sequences(arrays)
    return padded


def _mask_array(mask: object, *, n_samples: int, n_steps: int) -> np.ndarray:
    arr = np.asarray(mask, dtype=bool)
    if arr.ndim == 1 and n_samples == 1:
        arr = arr[None, :]
    if arr.shape != (n_samples, n_steps):
        raise ValueError(f"mask must have shape {(n_samples, n_steps)}, got {arr.shape}")
    return arr.astype(bool, copy=True)


def _feature_slices(c_raw: int, c_features: int, c_dt: int) -> Dict[str, slice]:
    start = 0
    slices: Dict[str, slice] = {"raw": slice(start, start + c_raw)}
    start += c_raw
    if c_features:
        slices["features"] = slice(start, start + c_features)
        start += c_features
    if c_dt:
        slices["dt"] = slice(start, start + c_dt)
    return slices


def validate_sequence_inputs(
    raw: np.ndarray | Sequence[np.ndarray | Sequence[Sequence[float]]],
    features: Optional[np.ndarray | Sequence[np.ndarray | Sequence[Sequence[float]]]] = None,
    dt: Optional[np.ndarray | Sequence[np.ndarray | Sequence[Sequence[float]]]] = None,
    mask: Optional[np.ndarray | Sequence[Sequence[bool]] | Sequence[bool]] = None,
) -> DynamicInputBatch:
    """Normalize fixed-length arrays and variable-length sequence collections.

    Fixed-length input accepts ``(N, T, C)`` arrays and a single ``(T, C)``
    sample. Variable-length input accepts lists or tuples of ``(T_i, C)``
    arrays.
    """
    if _is_sequence_collection(raw):
        raw_items = list(raw)  # type: ignore[arg-type]
        raw_arr, inferred_mask = pad_sequences(raw_items)
        raw_lengths = [int(np.asarray(item).shape[0]) for item in raw_items]
        features_arr = _optional_list("features", features, raw_lengths)
        dt_arr = _optional_list("dt", dt, raw_lengths)
        mask_arr = inferred_mask if mask is None else _mask_array(mask, n_samples=raw_arr.shape[0], n_steps=raw_arr.shape[1])
    else:
        raw_arr = _fixed_array("raw", raw)
        features_arr = _optional_array("features", features, n_samples=raw_arr.shape[0], n_steps=raw_arr.shape[1])
        dt_arr = _optional_array("dt", dt, n_samples=raw_arr.shape[0], n_steps=raw_arr.shape[1])
        mask_arr = (
            np.ones((raw_arr.shape[0], raw_arr.shape[1]), dtype=bool)
            if mask is None
            else _mask_array(mask, n_samples=raw_arr.shape[0], n_steps=raw_arr.shape[1])
        )

    if dt_arr is not None and dt_arr.shape[2] != 1:
        raise ValueError(f"dt must have one channel, got {dt_arr.shape[2]}")

    raw_arr = raw_arr.copy()
    raw_arr[~mask_arr] = 0.0
    if features_arr is not None:
        features_arr = features_arr.copy()
        features_arr[~mask_arr] = 0.0
    if dt_arr is not None:
        dt_arr = dt_arr.copy()
        dt_arr[~mask_arr] = 0.0

    slices = _feature_slices(
        int(raw_arr.shape[2]),
        0 if features_arr is None else int(features_arr.shape[2]),
        0 if dt_arr is None else int(dt_arr.shape[2]),
    )
    return DynamicInputBatch(
        raw=raw_arr,
        features=features_arr,
        dt=dt_arr,
        mask=mask_arr,
        feature_slices=slices,
    )


__all__ = ["DynamicInputBatch", "pad_sequences", "validate_sequence_inputs"]
