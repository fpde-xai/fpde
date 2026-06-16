from __future__ import annotations

import json
import sys
import types
from pathlib import Path

import numpy as np
import pytest

from fpde import (
    RawWaveformFPDEContext,
    RawWaveformFPDEExplanation,
    prepare_raw_waveform_fpde_context,
    raw_cos_fpde,
    raw_diff_fpde,
    raw_hyb_fpde,
    raw_waveform_fpde_explain_one,
    save_raw_waveform_fpde_results,
)


def _context():
    return prepare_raw_waveform_fpde_context(
        [
            np.array([1.0, 1.0], dtype=float),
            np.array([1.0, 0.9], dtype=float),
            np.array([-1.0, -1.0], dtype=float),
            np.array([-1.0, -0.9], dtype=float),
        ],
        ["a", "a", "b", "b"],
        sample_rates=[10, 10, 10, 10],
        target_sr=10,
        segment_sec=0.2,
        hop_sec=0.2,
    )


def test_raw_waveform_validation_mono_conversion_and_no_normalization():
    stereo = np.array([[2.0, 4.0], [6.0, 8.0]], dtype=float)

    context = prepare_raw_waveform_fpde_context(
        [stereo, np.array([-2.0, -4.0])],
        ["a", "b"],
        sample_rates=10,
        target_sr=10,
        segment_sec=0.2,
        hop_sec=0.1,
    )

    assert isinstance(context, RawWaveformFPDEContext)
    np.testing.assert_allclose(context.segment_banks["a"][0], [3.0, 7.0])
    assert np.max(np.abs(context.segment_banks["a"][0])) > 1.0
    assert context.details["waveform_normalization"] is False
    assert context.details["uses_spectrogram"] is False
    assert context.details["uses_mfcc"] is False

    with pytest.raises(ValueError, match="at least one sample"):
        prepare_raw_waveform_fpde_context([np.array([])], ["a"], sample_rates=10)
    with pytest.raises(ValueError, match="NaN or inf"):
        prepare_raw_waveform_fpde_context([np.array([np.nan])], ["a"], sample_rates=10)
    with pytest.raises(ValueError, match="NaN or inf"):
        prepare_raw_waveform_fpde_context([np.array([np.inf])], ["a"], sample_rates=10)


def test_raw_windowing_padding_mask_and_end_aligned_coverage():
    short_context = prepare_raw_waveform_fpde_context(
        [np.array([1.0, 2.0]), np.array([-1.0, -2.0])],
        ["a", "b"],
        sample_rates=10,
        target_sr=10,
        segment_sec=0.4,
        hop_sec=0.2,
    )

    np.testing.assert_allclose(short_context.segment_banks["a"][0], [1.0, 2.0, 0.0, 0.0])
    np.testing.assert_array_equal(short_context.segment_masks["a"][0], [True, True, False, False])
    short_explanation = raw_waveform_fpde_explain_one(
        np.array([1.0, 2.0, 3.0, 4.0]),
        short_context,
        sample_rate=10,
        target_label="a",
        rival_label="b",
        lambda_grid=[0.5],
    )
    np.testing.assert_array_equal(short_explanation.lambda_results[0.5]["effective_window_masks"][0], [True, True, False, False])
    np.testing.assert_allclose(short_explanation.lambda_results[0.5]["phi"][2:], [0.0, 0.0])

    context = prepare_raw_waveform_fpde_context(
        [np.arange(8.0), -np.arange(8.0)],
        ["a", "b"],
        sample_rates=10,
        target_sr=10,
        segment_sec=0.4,
        hop_sec=0.3,
    )
    explanation = raw_waveform_fpde_explain_one(
        np.arange(8.0),
        context,
        sample_rate=10,
        target_label="a",
        rival_label="b",
        lambda_grid=[0.5],
    )

    result = explanation.lambda_results[0.5]
    np.testing.assert_array_equal(result["window_starts"], [0, 3, 4])
    assert result["phi"].shape == (8,)
    assert np.all(np.isfinite(result["phi"]))


def test_raw_evidence_exactness_endpoints_and_masking():
    window = np.array([1.0, 2.0, 99.0])
    target = np.array([1.0, 1.0, 0.0])
    rival = np.array([0.0, 0.0, 0.0])
    mask = np.array([True, True, False])

    diff_attr, diff_evidence = raw_diff_fpde(window, target, rival, mask=mask)
    expected_diff = np.array([1.0, 3.0, 0.0])
    np.testing.assert_allclose(diff_attr, expected_diff)
    assert diff_evidence == pytest.approx(float(np.sum(expected_diff)))

    cos_attr, cos_evidence = raw_cos_fpde(window, target, rival, mask=mask)
    assert np.all(np.isfinite(cos_attr))
    assert cos_attr[2] == pytest.approx(0.0)
    assert cos_evidence == pytest.approx(float(np.sum(cos_attr)))

    hyb_cos, _, cos_details = raw_hyb_fpde(window, target, rival, lambda_hyb=0.0, mask=mask)
    hyb_diff, _, diff_details = raw_hyb_fpde(window, target, rival, lambda_hyb=1.0, mask=mask)
    np.testing.assert_allclose(hyb_cos[mask], cos_attr[mask] / cos_details["cos_scale"])
    np.testing.assert_allclose(hyb_diff[mask], diff_attr[mask] / diff_details["diff_scale"])
    assert np.all(np.isfinite(hyb_cos))
    assert np.all(np.isfinite(hyb_diff))


def test_raw_lambda_grid_shapes_and_generator_hook():
    context = _context()

    default_explanation = raw_waveform_fpde_explain_one(
        np.array([1.0, 1.0, -1.0, -1.0]),
        context,
        sample_rate=10,
        target_label="a",
    )

    assert isinstance(default_explanation, RawWaveformFPDEExplanation)
    assert len(default_explanation.lambda_results) == 11
    assert default_explanation.rival_label == "b"
    for lambda_value, result in default_explanation.lambda_results.items():
        assert 0.0 <= lambda_value <= 1.0
        assert result["phi"].shape == default_explanation.waveform.shape
        assert result["generation_status"] == {"target": "skipped", "rival": "skipped"}

    calls = []

    def generator(label, lambda_hyb, segment, sample_rate, role, metadata):
        calls.append((label, lambda_hyb, role, metadata["evidence"]))
        return segment * 0.5

    generated = raw_waveform_fpde_explain_one(
        np.array([1.0, 1.0, -1.0, -1.0]),
        context,
        sample_rate=10,
        target_label="a",
        rival_label="b",
        lambda_grid=[0.5],
        generator=generator,
    )

    assert [call[2] for call in calls] == ["target", "rival"]
    assert generated.lambda_results[0.5]["generation_status"] == {"target": "ok", "rival": "ok"}
    assert generated.lambda_results[0.5]["generated_target"] is not None
    assert generated.lambda_results[0.5]["generated_rival"] is not None


def test_raw_waveform_device_auto_matches_cpu_backend():
    context = _context()
    kwargs = {
        "sample_rate": 10,
        "target_label": "a",
        "rival_label": "b",
        "lambda_grid": [0.0, 0.5, 1.0],
    }

    cpu = raw_waveform_fpde_explain_one(
        np.array([1.0, 1.0, -1.0, -1.0]),
        context,
        device="cpu",
        **kwargs,
    )
    auto = raw_waveform_fpde_explain_one(
        np.array([1.0, 1.0, -1.0, -1.0]),
        context,
        device="auto",
        **kwargs,
    )

    assert auto.details["device"] in {"cpu", "cuda"}
    for lambda_value in kwargs["lambda_grid"]:
        np.testing.assert_allclose(auto.lambda_results[lambda_value]["phi"], cpu.lambda_results[lambda_value]["phi"])
        np.testing.assert_allclose(
            auto.lambda_results[lambda_value]["window_evidence"],
            cpu.lambda_results[lambda_value]["window_evidence"],
        )

    with pytest.raises(ValueError, match="device"):
        raw_waveform_fpde_explain_one(
            np.array([1.0, 1.0, -1.0, -1.0]),
            context,
            device="tpu",
            **kwargs,
        )


def test_label_medoid_and_automatic_rival_selection_are_deterministic():
    context = prepare_raw_waveform_fpde_context(
        [
            np.array([0.0, 0.0]),
            np.array([1.0, 1.0]),
            np.array([9.0, 9.0]),
            np.array([2.0, 2.0]),
            np.array([10.0, 10.0]),
        ],
        ["a", "a", "a", "b", "c"],
        sample_rates=10,
        target_sr=10,
        segment_sec=0.2,
        hop_sec=0.2,
    )

    assert context.prototype_indices["a"] == 1
    np.testing.assert_allclose(context.prototypes["a"], [1.0, 1.0])

    explanation = raw_waveform_fpde_explain_one(
        np.array([2.0, 2.0]),
        context,
        sample_rate=10,
        target_label="a",
        rival_label=None,
        lambda_grid=[0.5],
    )

    assert explanation.rival_label == "b"


def test_save_raw_waveform_fpde_results_writes_minimum_artifacts(tmp_path, monkeypatch):
    context = _context()

    def generator(label, lambda_hyb, segment, sample_rate, role, metadata):
        return segment

    explanation = raw_waveform_fpde_explain_one(
        np.array([1.0, 1.0, -1.0, -1.0]),
        context,
        sample_rate=10,
        target_label="a",
        rival_label="b",
        lambda_grid=[0.5],
        generator=generator,
    )

    fake_soundfile = types.SimpleNamespace(write=lambda path, data, sample_rate: Path(path).write_text("wav"))
    monkeypatch.setitem(sys.modules, "soundfile", fake_soundfile)

    saved = save_raw_waveform_fpde_results(explanation, tmp_path / "results", save_plots=False)

    lambda_dir = tmp_path / "results" / "raw_hyb_lambda_0.5"
    assert saved["n_lambdas"] == 1
    assert (tmp_path / "results" / "summary.csv").exists()
    assert (lambda_dir / "window_evidence.csv").exists()
    assert (lambda_dir / "metrics.json").exists()
    assert (lambda_dir / "top_positive_segment.wav").exists()
    assert (lambda_dir / "top_negative_segment.wav").exists()
    assert (lambda_dir / "generated_target_lambda_0.5.wav").exists()
    assert (lambda_dir / "generated_rival_lambda_0.5.wav").exists()

    metrics = json.loads((lambda_dir / "metrics.json").read_text())
    assert metrics["generation_status"] == {"target": "ok", "rival": "ok"}
    assert metrics["phi_shape"] == [4]
