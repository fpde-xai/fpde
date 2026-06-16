"""Raw-waveform Dynamic-FPDE.

This module keeps raw audio samples as the explanation domain. It does not
extract acoustic features, spectrograms, MFCCs, or normalize waveform
amplitudes.
"""

from __future__ import annotations

import csv
import json
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Callable, Dict, Mapping, Optional, Sequence, Tuple

import numpy as np

from ._backend import DeviceMode, as_numpy_array, resolve_array_backend
from .dynamic import _labels_unique, _validate_eps, _validate_lambda_hyb


RawGenerator = Callable[[Any, float, np.ndarray, int, str, Mapping[str, Any]], np.ndarray]


@dataclass(frozen=True)
class RawWaveformFPDEContext:
    """Reusable raw segment banks and medoid prototypes for Raw-Waveform Dynamic-FPDE."""

    segment_banks: Dict[Any, np.ndarray]
    segment_masks: Dict[Any, np.ndarray]
    prototype_labels: np.ndarray
    prototypes: Dict[Any, np.ndarray]
    prototype_masks: Dict[Any, np.ndarray]
    prototype_indices: Dict[Any, int]
    target_sr: int
    segment_length: int
    hop_length: int
    segment_sec: float
    hop_sec: float
    details: Dict[str, Any]


@dataclass(frozen=True)
class RawWaveformFPDEExplanation:
    """Result object for one raw-waveform Dynamic-FPDE explanation."""

    mode: str
    target_label: Any
    rival_label: Any
    waveform: np.ndarray
    sample_rate: int
    lambda_results: Dict[float, Dict[str, Any]]
    best_lambda: Optional[float]
    details: Dict[str, Any]
    time_mode: str = "raw_waveform"
    temporal_resampling: bool = False
    waveform_normalization: bool = False


def _validate_sample_rate(name: str, sample_rate: int) -> int:
    value = int(sample_rate)
    if value <= 0:
        raise ValueError(f"{name} must be positive")
    return value


def _as_raw_waveform(name: str, waveform: np.ndarray | Sequence[float] | Sequence[Sequence[float]]) -> np.ndarray:
    arr = np.asarray(waveform, dtype=float)
    if arr.ndim == 2:
        if 1 in arr.shape:
            arr = arr.reshape(-1)
        elif arr.shape[1] <= 8:
            arr = np.mean(arr, axis=1)
        elif arr.shape[0] <= 8:
            arr = np.mean(arr, axis=0)
        else:
            raise ValueError(f"{name} stereo waveform must have a recognizable channel axis")
    if arr.ndim != 1:
        raise ValueError(f"{name} must be a 1D raw waveform or 2D stereo waveform, got shape={arr.shape}")
    if arr.shape[0] == 0:
        raise ValueError(f"{name} must contain at least one sample")
    if not np.all(np.isfinite(arr)):
        raise ValueError(f"{name} contains NaN or inf")
    return arr.astype(float, copy=True)


def _resample_raw_waveform(waveform: np.ndarray, source_sr: int, target_sr: int) -> np.ndarray:
    source = _validate_sample_rate("source_sr", source_sr)
    target = _validate_sample_rate("target_sr", target_sr)
    if source == target:
        return waveform.astype(float, copy=True)
    target_length = max(1, int(round(waveform.shape[0] * float(target) / float(source))))
    old_pos = np.arange(waveform.shape[0], dtype=float) / float(source)
    new_pos = np.arange(target_length, dtype=float) / float(target)
    right = float(waveform[-1])
    return np.interp(new_pos, old_pos, waveform, right=right).astype(float, copy=False)


def _validate_time_params(target_sr: int, segment_sec: float, hop_sec: float) -> Tuple[int, int, float, float]:
    sr = _validate_sample_rate("target_sr", target_sr)
    seg = float(segment_sec)
    hop = float(hop_sec)
    if not np.isfinite(seg) or seg <= 0.0:
        raise ValueError("segment_sec must be positive")
    if not np.isfinite(hop) or hop <= 0.0:
        raise ValueError("hop_sec must be positive")
    segment_length = int(round(seg * sr))
    hop_length = int(round(hop * sr))
    if segment_length <= 0:
        raise ValueError("segment_length must be positive")
    if hop_length <= 0:
        raise ValueError("hop_length must be positive")
    return segment_length, hop_length, seg, hop


def _raw_windows(waveform: np.ndarray, segment_length: int, hop_length: int) -> Tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
    n_samples = int(waveform.shape[0])
    if n_samples < segment_length:
        window = np.zeros(segment_length, dtype=float)
        mask = np.zeros(segment_length, dtype=bool)
        window[:n_samples] = waveform
        mask[:n_samples] = True
        return window[None, :], mask[None, :], np.array([0], dtype=np.intp), np.array([n_samples], dtype=np.intp)

    starts = list(range(0, n_samples - segment_length + 1, hop_length))
    final_start = n_samples - segment_length
    if starts[-1] != final_start:
        starts.append(final_start)
    starts_arr = np.asarray(starts, dtype=np.intp)
    windows = np.stack([waveform[start : start + segment_length] for start in starts_arr], axis=0)
    masks = np.ones_like(windows, dtype=bool)
    lengths = np.full(starts_arr.shape, segment_length, dtype=np.intp)
    return windows.astype(float, copy=True), masks, starts_arr, lengths


def _masked_l2(a: np.ndarray, a_mask: np.ndarray, b: np.ndarray, b_mask: np.ndarray) -> float:
    valid = a_mask & b_mask
    if not np.any(valid):
        return float("inf")
    diff = a[valid] - b[valid]
    return float(np.sum(diff * diff))


def _choose_medoid(windows: np.ndarray, masks: np.ndarray) -> int:
    if windows.shape[0] == 1:
        return 0
    totals = np.zeros(windows.shape[0], dtype=float)
    for i in range(windows.shape[0]):
        total = 0.0
        for j in range(windows.shape[0]):
            if i == j:
                continue
            total += _masked_l2(windows[i], masks[i], windows[j], masks[j])
        totals[i] = total
    return int(np.argmin(totals))


def _lambda_grid(lambda_grid: Optional[Sequence[float]]) -> Tuple[float, ...]:
    values = [i / 10.0 for i in range(11)] if lambda_grid is None else [float(value) for value in lambda_grid]
    if not values:
        raise ValueError("lambda_grid must contain at least one value")
    return tuple(_validate_lambda_hyb(value) for value in values)


def _component_scale(values: np.ndarray, mask: np.ndarray, eps: float) -> float:
    return float(np.sum(np.abs(values[mask])) + eps)


def raw_diff_fpde(
    window: np.ndarray | Sequence[float],
    p_target: np.ndarray | Sequence[float],
    p_rival: np.ndarray | Sequence[float],
    *,
    mask: Optional[np.ndarray | Sequence[bool]] = None,
    target_mask: Optional[np.ndarray | Sequence[bool]] = None,
    rival_mask: Optional[np.ndarray | Sequence[bool]] = None,
) -> Tuple[np.ndarray, float]:
    """Compute raw squared-distance prototype evidence for one window."""
    w = _as_raw_waveform("window", window)
    target = _as_raw_waveform("p_target", p_target)
    rival = _as_raw_waveform("p_rival", p_rival)
    if target.shape != w.shape or rival.shape != w.shape:
        raise ValueError("window, p_target, and p_rival must have the same shape")
    valid = np.ones(w.shape, dtype=bool) if mask is None else np.asarray(mask, dtype=bool)
    t_valid = np.ones(w.shape, dtype=bool) if target_mask is None else np.asarray(target_mask, dtype=bool)
    r_valid = np.ones(w.shape, dtype=bool) if rival_mask is None else np.asarray(rival_mask, dtype=bool)
    if valid.shape != w.shape or t_valid.shape != w.shape or r_valid.shape != w.shape:
        raise ValueError("mask shapes must match the window shape")
    valid = valid & t_valid & r_valid
    attr = np.zeros_like(w, dtype=float)
    attr[valid] = (w[valid] - rival[valid]) ** 2 - (w[valid] - target[valid]) ** 2
    return attr, float(np.sum(attr[valid]))


def raw_cos_fpde(
    window: np.ndarray | Sequence[float],
    p_target: np.ndarray | Sequence[float],
    p_rival: np.ndarray | Sequence[float],
    *,
    mask: Optional[np.ndarray | Sequence[bool]] = None,
    target_mask: Optional[np.ndarray | Sequence[bool]] = None,
    rival_mask: Optional[np.ndarray | Sequence[bool]] = None,
    eps: float = 1e-12,
) -> Tuple[np.ndarray, float]:
    """Compute raw cosine-contrast coordinate evidence for one window."""
    eps_value = _validate_eps(eps)
    w = _as_raw_waveform("window", window)
    target = _as_raw_waveform("p_target", p_target)
    rival = _as_raw_waveform("p_rival", p_rival)
    if target.shape != w.shape or rival.shape != w.shape:
        raise ValueError("window, p_target, and p_rival must have the same shape")
    valid = np.ones(w.shape, dtype=bool) if mask is None else np.asarray(mask, dtype=bool)
    t_valid = np.ones(w.shape, dtype=bool) if target_mask is None else np.asarray(target_mask, dtype=bool)
    r_valid = np.ones(w.shape, dtype=bool) if rival_mask is None else np.asarray(rival_mask, dtype=bool)
    if valid.shape != w.shape or t_valid.shape != w.shape or r_valid.shape != w.shape:
        raise ValueError("mask shapes must match the window shape")
    valid = valid & t_valid & r_valid
    attr = np.zeros_like(w, dtype=float)
    target_norm = float(np.linalg.norm(target[valid]))
    rival_norm = float(np.linalg.norm(rival[valid]))
    window_norm = float(np.linalg.norm(w[valid]))
    if np.any(valid):
        attr[valid] = (w[valid] * target[valid]) / ((window_norm * target_norm) + eps_value)
        attr[valid] -= (w[valid] * rival[valid]) / ((window_norm * rival_norm) + eps_value)
    attr[~valid] = 0.0
    return attr, float(np.sum(attr[valid]))


def raw_hyb_fpde(
    window: np.ndarray | Sequence[float],
    p_target: np.ndarray | Sequence[float],
    p_rival: np.ndarray | Sequence[float],
    *,
    lambda_hyb: float = 0.5,
    mask: Optional[np.ndarray | Sequence[bool]] = None,
    target_mask: Optional[np.ndarray | Sequence[bool]] = None,
    rival_mask: Optional[np.ndarray | Sequence[bool]] = None,
    eps: float = 1e-12,
) -> Tuple[np.ndarray, float, Dict[str, Any]]:
    """Mix Raw-Diff and Raw-Cos after component-wise valid-mask L1 scaling."""
    eps_value = _validate_eps(eps)
    lambda_value = _validate_lambda_hyb(lambda_hyb)
    diff_attr, diff_evidence = raw_diff_fpde(
        window,
        p_target,
        p_rival,
        mask=mask,
        target_mask=target_mask,
        rival_mask=rival_mask,
    )
    cos_attr, cos_evidence = raw_cos_fpde(
        window,
        p_target,
        p_rival,
        mask=mask,
        target_mask=target_mask,
        rival_mask=rival_mask,
        eps=eps_value,
    )
    valid = np.ones(diff_attr.shape, dtype=bool) if mask is None else np.asarray(mask, dtype=bool)
    diff_scale = _component_scale(diff_attr, valid, eps_value)
    cos_scale = _component_scale(cos_attr, valid, eps_value)
    attr = np.zeros_like(diff_attr, dtype=float)
    attr[valid] = lambda_value * (diff_attr[valid] / diff_scale)
    attr[valid] += (1.0 - lambda_value) * (cos_attr[valid] / cos_scale)
    evidence = float(np.sum(attr[valid]))
    return (
        attr,
        evidence,
        {
            "diff_evidence": float(diff_evidence),
            "cos_evidence": float(cos_evidence),
            "lambda_hyb": float(lambda_value),
            "normalize": "valid_l1",
            "diff_scale": float(diff_scale),
            "cos_scale": float(cos_scale),
        },
    )


def prepare_raw_waveform_fpde_context(
    waveforms: Sequence[np.ndarray | Sequence[float] | Sequence[Sequence[float]]],
    labels: Sequence[Any],
    *,
    sample_rates: Sequence[int] | int,
    target_sr: int = 16000,
    segment_sec: float = 0.5,
    hop_sec: float = 0.1,
) -> RawWaveformFPDEContext:
    """Build label-specific raw segment banks and label-medoid prototypes."""
    segment_length, hop_length, seg, hop = _validate_time_params(target_sr, segment_sec, hop_sec)
    waveform_items = list(waveforms)
    if not waveform_items:
        raise ValueError("waveforms must contain at least one sample")
    labels_arr = np.asarray(labels, dtype=object)
    if labels_arr.ndim != 1 or labels_arr.shape[0] != len(waveform_items):
        raise ValueError(f"number of waveforms and labels differ: {len(waveform_items)} vs {labels_arr.shape[0]}")
    if isinstance(sample_rates, (int, np.integer)):
        rate_items = [int(sample_rates)] * len(waveform_items)
    else:
        rate_items = [int(value) for value in sample_rates]
    if len(rate_items) != len(waveform_items):
        raise ValueError(f"number of waveforms and sample_rates differ: {len(waveform_items)} vs {len(rate_items)}")

    processed = []
    bank_windows: Dict[Any, list[np.ndarray]] = {}
    bank_masks: Dict[Any, list[np.ndarray]] = {}
    input_lengths = []
    resampled_lengths = []
    for i, (waveform, label, source_sr) in enumerate(zip(waveform_items, labels_arr.tolist(), rate_items)):
        raw = _as_raw_waveform(f"waveforms[{i}]", waveform)
        resampled = _resample_raw_waveform(raw, source_sr, target_sr)
        windows, masks, _, _ = _raw_windows(resampled, segment_length, hop_length)
        bank_windows.setdefault(label, []).extend([windows[j].copy() for j in range(windows.shape[0])])
        bank_masks.setdefault(label, []).extend([masks[j].copy() for j in range(masks.shape[0])])
        processed.append(resampled)
        input_lengths.append(int(raw.shape[0]))
        resampled_lengths.append(int(resampled.shape[0]))

    prototype_labels, _ = _labels_unique(labels_arr)
    segment_banks: Dict[Any, np.ndarray] = {}
    segment_masks: Dict[Any, np.ndarray] = {}
    prototypes: Dict[Any, np.ndarray] = {}
    prototype_masks: Dict[Any, np.ndarray] = {}
    prototype_indices: Dict[Any, int] = {}
    class_counts: Dict[Any, int] = {}
    for label in prototype_labels.tolist():
        windows = np.stack(bank_windows[label], axis=0).astype(float, copy=False)
        masks = np.stack(bank_masks[label], axis=0).astype(bool, copy=False)
        medoid_idx = _choose_medoid(windows, masks)
        segment_banks[label] = windows
        segment_masks[label] = masks
        prototypes[label] = windows[medoid_idx].copy()
        prototype_masks[label] = masks[medoid_idx].copy()
        prototype_indices[label] = int(medoid_idx)
        class_counts[label] = int(np.sum(labels_arr == label))

    return RawWaveformFPDEContext(
        segment_banks=segment_banks,
        segment_masks=segment_masks,
        prototype_labels=prototype_labels.copy(),
        prototypes=prototypes,
        prototype_masks=prototype_masks,
        prototype_indices=prototype_indices,
        target_sr=int(target_sr),
        segment_length=int(segment_length),
        hop_length=int(hop_length),
        segment_sec=float(seg),
        hop_sec=float(hop),
        details={
            "time_mode": "raw_waveform",
            "uses_acoustic_features": False,
            "uses_spectrogram": False,
            "uses_mfcc": False,
            "waveform_normalization": False,
            "prototype_kind": "label_medoid_raw_segment",
            "n_train_samples": len(waveform_items),
            "input_lengths": tuple(input_lengths),
            "resampled_lengths": tuple(resampled_lengths),
            "class_counts": class_counts,
        },
    )


def _validate_context(context: RawWaveformFPDEContext) -> RawWaveformFPDEContext:
    if not isinstance(context, RawWaveformFPDEContext):
        raise TypeError("context must be a RawWaveformFPDEContext")
    if context.segment_length <= 0 or context.hop_length <= 0:
        raise ValueError("context segment_length and hop_length must be positive")
    if context.prototype_labels.ndim != 1 or context.prototype_labels.shape[0] == 0:
        raise ValueError("context prototype_labels must be a non-empty 1D array")
    for label in context.prototype_labels.tolist():
        if label not in context.prototypes or label not in context.prototype_masks:
            raise ValueError(f"context is missing prototype for label {label!r}")
        if context.prototypes[label].shape != (context.segment_length,):
            raise ValueError(f"context prototype for label {label!r} has an incompatible shape")
        if context.prototype_masks[label].shape != (context.segment_length,):
            raise ValueError(f"context prototype mask for label {label!r} has an incompatible shape")
    return context


def _prototype_for_label(context: RawWaveformFPDEContext, label: Any, name: str) -> Tuple[np.ndarray, np.ndarray]:
    if label not in context.prototypes:
        raise ValueError(f"no raw prototype found for {name}={label!r}")
    return context.prototypes[label], context.prototype_masks[label]


def _select_rival_label(windows: np.ndarray, masks: np.ndarray, context: RawWaveformFPDEContext, target_label: Any) -> Any:
    candidate_labels = [label for label in context.prototype_labels.tolist() if label != target_label]
    if not candidate_labels:
        raise ValueError("rival_label is None, but no non-target raw prototypes exist")
    scores = []
    for label in candidate_labels:
        proto = context.prototypes[label]
        proto_mask = context.prototype_masks[label]
        distances = [_masked_l2(windows[i], masks[i], proto, proto_mask) for i in range(windows.shape[0])]
        scores.append(float(np.mean(distances)))
    return candidate_labels[int(np.argmin(np.asarray(scores, dtype=float)))]


def _overlap_add(window_attrs: np.ndarray, masks: np.ndarray, starts: np.ndarray, n_samples: int) -> np.ndarray:
    values = np.zeros(n_samples, dtype=float)
    counts = np.zeros(n_samples, dtype=float)
    for attr, mask, start in zip(window_attrs, masks, starts):
        valid_idx = np.where(mask)[0]
        sample_idx = valid_idx + int(start)
        sample_idx = sample_idx[sample_idx < n_samples]
        valid_idx = valid_idx[: sample_idx.shape[0]]
        values[sample_idx] += attr[valid_idx]
        counts[sample_idx] += 1.0
    out = np.zeros(n_samples, dtype=float)
    covered = counts > 0.0
    out[covered] = values[covered] / counts[covered]
    return out


def _top_segments(
    waveform: np.ndarray,
    starts: np.ndarray,
    lengths: np.ndarray,
    evidence: np.ndarray,
    *,
    top_k: int,
    positive: bool,
) -> list[Dict[str, Any]]:
    if top_k <= 0:
        return []
    order = np.argsort(evidence)
    if positive:
        order = order[::-1]
        keep = [idx for idx in order.tolist() if evidence[idx] > 0.0]
        role = "positive"
    else:
        keep = [idx for idx in order.tolist() if evidence[idx] < 0.0]
        role = "negative"
    rows = []
    for rank, idx in enumerate(keep[:top_k], start=1):
        start = int(starts[idx])
        length = int(lengths[idx])
        end = min(start + length, waveform.shape[0])
        rows.append(
            {
                "rank": int(rank),
                "window_index": int(idx),
                "role": role,
                "start_sample": start,
                "end_sample": int(end),
                "evidence": float(evidence[idx]),
                "segment": waveform[start:end].astype(float, copy=True),
            }
        )
    return rows


def _call_generator(
    generator: Optional[RawGenerator],
    *,
    label: Any,
    lambda_hyb: float,
    segment: np.ndarray,
    sample_rate: int,
    role: str,
    metadata: Mapping[str, Any],
) -> Tuple[Optional[np.ndarray], str]:
    if generator is None:
        return None, "skipped"
    generated = generator(label, float(lambda_hyb), segment.copy(), int(sample_rate), role, metadata)
    return _as_raw_waveform(f"generated_{role}", generated), "ok"


def _raw_hyb_windows_backend(
    windows: np.ndarray,
    masks: np.ndarray,
    p_target: np.ndarray,
    target_mask: np.ndarray,
    p_rival: np.ndarray,
    rival_mask: np.ndarray,
    *,
    lambda_hyb: float,
    eps: float,
    xp: Any,
) -> Tuple[np.ndarray, np.ndarray, Dict[str, Any]]:
    w = xp.asarray(windows, dtype=float)
    target = xp.asarray(p_target, dtype=float)
    rival = xp.asarray(p_rival, dtype=float)
    valid = xp.asarray(masks & target_mask[None, :] & rival_mask[None, :], dtype=bool)
    lambda_value = float(lambda_hyb)

    diff_attr = xp.where(valid, (w - rival[None, :]) ** 2 - (w - target[None, :]) ** 2, 0.0)
    diff_evidence = xp.sum(diff_attr, axis=1)

    target_sq = xp.where(valid, target[None, :] * target[None, :], 0.0)
    rival_sq = xp.where(valid, rival[None, :] * rival[None, :], 0.0)
    window_sq = xp.where(valid, w * w, 0.0)
    target_norm = xp.sqrt(xp.sum(target_sq, axis=1))
    rival_norm = xp.sqrt(xp.sum(rival_sq, axis=1))
    window_norm = xp.sqrt(xp.sum(window_sq, axis=1))
    target_part = (w * target[None, :]) / ((window_norm * target_norm)[:, None] + eps)
    rival_part = (w * rival[None, :]) / ((window_norm * rival_norm)[:, None] + eps)
    cos_attr = xp.where(valid, target_part - rival_part, 0.0)
    cos_evidence = xp.sum(cos_attr, axis=1)

    diff_scale = xp.sum(xp.abs(diff_attr), axis=1) + eps
    cos_scale = xp.sum(xp.abs(cos_attr), axis=1) + eps
    attr = xp.where(
        valid,
        lambda_value * (diff_attr / diff_scale[:, None]) + (1.0 - lambda_value) * (cos_attr / cos_scale[:, None]),
        0.0,
    )
    evidence = xp.sum(attr, axis=1)
    return (
        as_numpy_array(attr, xp).astype(float, copy=False),
        as_numpy_array(evidence, xp).astype(float, copy=False),
        {
            "diff_evidence": as_numpy_array(diff_evidence, xp).astype(float, copy=False).tolist(),
            "cos_evidence": as_numpy_array(cos_evidence, xp).astype(float, copy=False).tolist(),
            "diff_scale": as_numpy_array(diff_scale, xp).astype(float, copy=False).tolist(),
            "cos_scale": as_numpy_array(cos_scale, xp).astype(float, copy=False).tolist(),
        },
    )


def raw_waveform_fpde_explain_one(
    waveform: np.ndarray | Sequence[float] | Sequence[Sequence[float]],
    context: RawWaveformFPDEContext,
    *,
    sample_rate: int,
    target_label: Any,
    rival_label: Optional[Any] = None,
    lambda_grid: Optional[Sequence[float]] = None,
    top_k_segments: int = 1,
    generator: Optional[RawGenerator] = None,
    eps: float = 1e-12,
    device: DeviceMode = "cpu",
    details: Optional[Dict[str, Any]] = None,
) -> RawWaveformFPDEExplanation:
    """Explain one raw waveform across a lambda grid."""
    eps_value = _validate_eps(eps)
    backend = resolve_array_backend(device)
    ctx = _validate_context(context)
    if top_k_segments < 0:
        raise ValueError("top_k_segments must be non-negative")
    waveform_arr = _resample_raw_waveform(_as_raw_waveform("waveform", waveform), sample_rate, ctx.target_sr)
    windows, masks, starts, lengths = _raw_windows(waveform_arr, ctx.segment_length, ctx.hop_length)
    p_target, target_mask = _prototype_for_label(ctx, target_label, "target_label")
    if rival_label is None:
        resolved_rival = _select_rival_label(windows, masks, ctx, target_label)
    else:
        resolved_rival = rival_label
        if resolved_rival == target_label:
            raise ValueError("rival_label must differ from target_label")
    p_rival, rival_mask = _prototype_for_label(ctx, resolved_rival, "rival_label")
    effective_masks = masks & target_mask[None, :] & rival_mask[None, :]

    lambdas = _lambda_grid(lambda_grid)
    lambda_results: Dict[float, Dict[str, Any]] = {}
    for lambda_value in lambdas:
        window_attrs_arr, window_evidence_arr, component_details = _raw_hyb_windows_backend(
            windows,
            masks,
            p_target,
            target_mask,
            p_rival,
            rival_mask,
            lambda_hyb=lambda_value,
            eps=eps_value,
            xp=backend.xp,
        )
        phi = _overlap_add(window_attrs_arr, effective_masks, starts, waveform_arr.shape[0])
        if phi.shape != waveform_arr.shape:
            raise RuntimeError(f"raw waveform attribution shape mismatch: expected {waveform_arr.shape}, got {phi.shape}")

        top_positive = _top_segments(
            waveform_arr,
            starts,
            lengths,
            window_evidence_arr,
            top_k=top_k_segments,
            positive=True,
        )
        top_negative = _top_segments(
            waveform_arr,
            starts,
            lengths,
            window_evidence_arr,
            top_k=top_k_segments,
            positive=False,
        )
        generation_status = {"target": "skipped", "rival": "skipped"}
        generated_target = None
        generated_rival = None
        if top_positive:
            metadata = {key: value for key, value in top_positive[0].items() if key != "segment"}
            generated_target, generation_status["target"] = _call_generator(
                generator,
                label=target_label,
                lambda_hyb=lambda_value,
                segment=top_positive[0]["segment"],
                sample_rate=ctx.target_sr,
                role="target",
                metadata=metadata,
            )
        if top_negative:
            metadata = {key: value for key, value in top_negative[0].items() if key != "segment"}
            generated_rival, generation_status["rival"] = _call_generator(
                generator,
                label=resolved_rival,
                lambda_hyb=lambda_value,
                segment=top_negative[0]["segment"],
                sample_rate=ctx.target_sr,
                role="rival",
                metadata=metadata,
            )

        lambda_results[float(lambda_value)] = {
            "phi": phi.astype(float, copy=True),
            "window_attributions": window_attrs_arr.astype(float, copy=True),
            "window_evidence": window_evidence_arr,
            "window_starts": starts.copy(),
            "window_lengths": lengths.copy(),
            "window_masks": masks.copy(),
            "effective_window_masks": effective_masks.copy(),
            "evidence": float(np.sum(window_evidence_arr)),
            "top_positive_segments": top_positive,
            "top_negative_segments": top_negative,
            "generated_target": generated_target,
            "generated_rival": generated_rival,
            "generation_status": generation_status,
            "details": {
                "lambda_hyb": float(lambda_value),
                "diff_evidence": component_details["diff_evidence"],
                "cos_evidence": component_details["cos_evidence"],
                "diff_scale": component_details["diff_scale"],
                "cos_scale": component_details["cos_scale"],
                "device": backend.name,
                "input_shape": tuple(waveform_arr.shape),
                "output_shape": tuple(phi.shape),
            },
        }

    best_lambda = max(lambdas, key=lambda value: float(lambda_results[float(value)]["evidence"])) if lambdas else None
    merged_details = {} if details is None else dict(details)
    merged_details.update(
        {
            "time_mode": "raw_waveform",
            "uses_acoustic_features": False,
            "uses_spectrogram": False,
            "uses_mfcc": False,
            "waveform_normalization": False,
            "target_sr": int(ctx.target_sr),
            "segment_length": int(ctx.segment_length),
            "hop_length": int(ctx.hop_length),
            "lambda_grid": tuple(float(value) for value in lambdas),
            "device": backend.name,
            "target_label": target_label,
            "rival_label": resolved_rival,
        }
    )
    return RawWaveformFPDEExplanation(
        mode="raw_hyb",
        target_label=target_label,
        rival_label=resolved_rival,
        waveform=waveform_arr.astype(float, copy=True),
        sample_rate=int(ctx.target_sr),
        lambda_results=lambda_results,
        best_lambda=None if best_lambda is None else float(best_lambda),
        details=merged_details,
    )


def _pyplot():
    try:
        import matplotlib.pyplot as plt
    except ImportError as exc:
        raise ImportError(
            "Raw-Waveform Dynamic-FPDE plotting requires matplotlib. Install it with "
            "`python -m pip install fpde[plot]` or pass save_plots=False."
        ) from exc
    return plt


def _write_wav(path: Path, data: np.ndarray, sample_rate: int) -> None:
    try:
        import soundfile as sf
    except ImportError as exc:
        raise ImportError("Saving WAV files requires soundfile. Install it with `python -m pip install fpde[audio]`.") from exc
    sf.write(str(path), np.asarray(data, dtype=float), int(sample_rate))


def _write_waveform_plot(path: Path, waveform: np.ndarray, phi: np.ndarray, sample_rate: int, *, title: str) -> None:
    plt = _pyplot()
    time = np.arange(waveform.shape[0], dtype=float) / float(sample_rate)
    fig, axes = plt.subplots(2, 1, figsize=(8.0, 4.0), sharex=True)
    axes[0].plot(time, waveform, color="#111827", linewidth=0.8)
    axes[0].set_ylabel("waveform")
    axes[0].set_title(title)
    axes[1].plot(time, phi, color="#2563eb", linewidth=0.8)
    axes[1].axhline(0.0, color="#111827", linewidth=0.6)
    axes[1].set_xlabel("Seconds")
    axes[1].set_ylabel("phi_hyb")
    fig.tight_layout()
    fig.savefig(path, dpi=150)
    plt.close(fig)


def _write_comparison_plot(path: Path, segment: np.ndarray, generated: Optional[np.ndarray], sample_rate: int, *, title: str) -> None:
    plt = _pyplot()
    fig, ax = plt.subplots(figsize=(8.0, 2.5))
    seg_time = np.arange(segment.shape[0], dtype=float) / float(sample_rate)
    ax.plot(seg_time, segment, color="#2563eb", linewidth=0.8, label="segment")
    if generated is not None:
        gen_time = np.arange(generated.shape[0], dtype=float) / float(sample_rate)
        ax.plot(gen_time, generated, color="#dc2626", linewidth=0.8, alpha=0.8, label="generated")
        ax.legend()
    ax.set_title(title)
    ax.set_xlabel("Seconds")
    ax.set_ylabel("Amplitude")
    fig.tight_layout()
    fig.savefig(path, dpi=150)
    plt.close(fig)


def _json_safe(value: Any) -> Any:
    if isinstance(value, np.ndarray):
        return value.tolist()
    if isinstance(value, np.generic):
        return value.item()
    if isinstance(value, dict):
        return {str(key): _json_safe(item) for key, item in value.items()}
    if isinstance(value, (list, tuple)):
        return [_json_safe(item) for item in value]
    return value


def _lambda_name(lambda_value: float) -> str:
    rounded_tenth = round(float(lambda_value), 1)
    if abs(float(lambda_value) - rounded_tenth) <= 1e-12:
        return f"{rounded_tenth:.1f}"
    return f"{float(lambda_value):.6g}"


def save_raw_waveform_fpde_results(
    explanation: RawWaveformFPDEExplanation,
    output_dir: str | Path,
    *,
    save_plots: bool = True,
) -> Dict[str, Any]:
    """Save lambda-wise Raw-Waveform Dynamic-FPDE artifacts."""
    if not isinstance(explanation, RawWaveformFPDEExplanation):
        raise TypeError("explanation must be a RawWaveformFPDEExplanation")
    root = Path(output_dir)
    root.mkdir(parents=True, exist_ok=True)
    summary_rows = []
    for lambda_value, result in sorted(explanation.lambda_results.items(), key=lambda item: item[0]):
        lambda_name = _lambda_name(lambda_value)
        lambda_dir = root / f"raw_hyb_lambda_{lambda_name}"
        lambda_dir.mkdir(parents=True, exist_ok=True)
        phi = np.asarray(result["phi"], dtype=float)
        if save_plots:
            _write_waveform_plot(
                lambda_dir / "waveform_phi_hyb.png",
                explanation.waveform,
                phi,
                explanation.sample_rate,
                title=f"Raw-Hyb lambda={lambda_name}",
            )

        window_csv = lambda_dir / "window_evidence.csv"
        with window_csv.open("w", newline="", encoding="utf-8") as handle:
            writer = csv.DictWriter(
                handle,
                fieldnames=["window_index", "start_sample", "end_sample", "evidence"],
            )
            writer.writeheader()
            for idx, (start, length, evidence) in enumerate(
                zip(result["window_starts"], result["window_lengths"], result["window_evidence"])
            ):
                writer.writerow(
                    {
                        "window_index": int(idx),
                        "start_sample": int(start),
                        "end_sample": int(start + length),
                        "evidence": float(evidence),
                    }
                )

        for role, rows in (
            ("positive", result["top_positive_segments"]),
            ("negative", result["top_negative_segments"]),
        ):
            if rows:
                segment = rows[0]["segment"]
                _write_wav(lambda_dir / f"top_{role}_segment.wav", segment, explanation.sample_rate)
                generated_key = "generated_target" if role == "positive" else "generated_rival"
                generated_role = "target" if role == "positive" else "rival"
                generated = result.get(generated_key)
                if generated is not None:
                    _write_wav(lambda_dir / f"generated_{generated_role}_lambda_{lambda_name}.wav", generated, explanation.sample_rate)
                if save_plots:
                    _write_comparison_plot(
                        lambda_dir / f"comparison_{role}.png",
                        segment,
                        generated,
                        explanation.sample_rate,
                        title=f"{role} segment comparison lambda={lambda_name}",
                    )

        metrics = {
            "lambda_hyb": float(lambda_value),
            "evidence": float(result["evidence"]),
            "target_label": explanation.target_label,
            "rival_label": explanation.rival_label,
            "sample_rate": int(explanation.sample_rate),
            "input_shape": tuple(explanation.waveform.shape),
            "phi_shape": tuple(phi.shape),
            "generation_status": result["generation_status"],
            "top_positive_segments": [
                {key: value for key, value in row.items() if key != "segment"} for row in result["top_positive_segments"]
            ],
            "top_negative_segments": [
                {key: value for key, value in row.items() if key != "segment"} for row in result["top_negative_segments"]
            ],
        }
        with (lambda_dir / "metrics.json").open("w", encoding="utf-8") as handle:
            json.dump(_json_safe(metrics), handle, indent=2, sort_keys=True)
            handle.write("\n")
        summary_rows.append(
            {
                "lambda_hyb": float(lambda_value),
                "evidence": float(result["evidence"]),
                "target_generation_status": result["generation_status"]["target"],
                "rival_generation_status": result["generation_status"]["rival"],
                "n_windows": int(np.asarray(result["window_evidence"]).shape[0]),
                "phi_shape": str(tuple(phi.shape)),
            }
        )

    summary_csv = root / "summary.csv"
    with summary_csv.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(
            handle,
            fieldnames=[
                "lambda_hyb",
                "evidence",
                "target_generation_status",
                "rival_generation_status",
                "n_windows",
                "phi_shape",
            ],
        )
        writer.writeheader()
        writer.writerows(summary_rows)
    return {"output_dir": str(root), "summary_csv": str(summary_csv), "n_lambdas": len(summary_rows)}


__all__ = [
    "RawWaveformFPDEContext",
    "RawWaveformFPDEExplanation",
    "prepare_raw_waveform_fpde_context",
    "raw_waveform_fpde_explain_one",
    "raw_diff_fpde",
    "raw_cos_fpde",
    "raw_hyb_fpde",
    "save_raw_waveform_fpde_results",
]
