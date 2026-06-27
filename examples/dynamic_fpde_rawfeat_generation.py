#!/usr/bin/env python3
"""End-to-end RawFeat Dynamic-FPDE generation and audit example."""

from __future__ import annotations

import numpy as np

from fpde.dynamic import DynamicFPDEEngine, PrototypeRawGenerator


def feature_extractor(raw: np.ndarray) -> np.ndarray:
    """Build simple frame-level features with the same time axis as raw."""
    values = np.asarray(raw, dtype=float)
    channel_mean = np.mean(values, axis=1)
    change = np.diff(channel_mean, prepend=channel_mean[0])
    energy = np.sqrt(np.mean(values * values, axis=1))
    return np.column_stack([channel_mean, change, energy])


def _synthetic_raw(label: int, length: int, phase: float) -> np.ndarray:
    time = np.linspace(0.0, 1.0, length)
    offset = 1.4 * label
    return np.column_stack(
        [
            offset + np.sin(2.0 * np.pi * time + phase),
            0.5 * offset + np.cos(2.0 * np.pi * time + 0.5 * phase),
        ]
    )


def main() -> None:
    train_y = np.array([0, 0, 0, 1, 1, 1])
    train_raw = [
        _synthetic_raw(int(label), length, phase)
        for label, length, phase in zip(
            train_y,
            [6, 8, 7, 7, 9, 8],
            [0.0, 0.3, 0.6, 0.1, 0.4, 0.7],
        )
    ]
    train_features = [feature_extractor(sample) for sample in train_raw]

    engine = DynamicFPDEEngine().fit(
        raw=train_raw,
        features=train_features,
        y=train_y,
    )
    observed = engine.explain_one(
        raw=train_raw[0],
        features=train_features[0],
        target_class=0,
        rival_class=1,
        method="hyb",
    )

    generator = PrototypeRawGenerator(summary_scaling="standard").fit(
        raw=train_raw,
        features=train_features,
        y=train_y,
    )
    condition = train_features[4]
    condition_mask = np.ones(condition.shape[0], dtype=bool)
    condition_mask[0] = False
    generation = generator.generate_with_metadata(
        label=1,
        length=7,
        condition_features=condition,
        condition_mask=condition_mask,
        noise_scale=0.02,
        random_state=0,
    )

    generated_raw = generation["raw"]
    generated_features = feature_extractor(generated_raw)
    generated_audit = engine.explain_one(
        raw=generated_raw,
        features=generated_features,
        target_class=1,
        rival_class=0,
        method="hyb",
    )
    top_time = int(np.argmax(np.abs(generated_audit.time_attributions)))

    print(f"Observed evidence: {observed.evidence:.6f}")
    print(f"Generated evidence: {generated_audit.evidence:.6f}")
    print(f"Generated audit: {generated_audit.audit}")
    print(f"Top generated time attribution index: {top_time}")
    print(
        "Generation metadata: "
        f"neighbor={generation['selected_neighbor_index']}, "
        f"distance={generation['selected_neighbor_summary_distance']:.6f}, "
        f"scaling={generation['summary_scaling']}, "
        f"generated_norm={generation['generated_norm']:.6f}"
    )


if __name__ == "__main__":
    main()
