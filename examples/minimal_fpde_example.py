#!/usr/bin/env python3
"""Minimal FPDE example on a small scikit-learn dataset."""

from __future__ import annotations

import numpy as np
from sklearn.datasets import load_breast_cancer
from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler

from fpde import FPDEEngine


def main() -> None:
    data = load_breast_cancer()
    X_train, X_test, y_train, _ = train_test_split(
        data.data,
        data.target,
        test_size=0.25,
        random_state=7,
        stratify=data.target,
    )

    scaler = StandardScaler()
    X_train_scaled = scaler.fit_transform(X_train)
    X_test_scaled = scaler.transform(X_test)

    clf = LogisticRegression(max_iter=2000, random_state=7)
    clf.fit(X_train_scaled, y_train)

    x = X_test_scaled[0]
    engine = FPDEEngine.fit(X_train_scaled, y_train, model=clf)
    attributions, details = engine.explain_one(
        x,
        lambda_hyb=0.5,
    )

    attributions = np.asarray(attributions, dtype=float)
    feature_names = np.asarray(data.feature_names, dtype=object)
    order = np.argsort(attributions)

    print(
        f"Predicted class: {details['target_label']} "
        f"with probability {float(details['target_probability']):.3f}"
    )
    print("\nTop positive feature contributions:")
    for idx in order[-5:][::-1]:
        print(f"  {feature_names[idx]}: {attributions[idx]: .4f}")

    print("\nTop negative feature contributions:")
    for idx in order[:5]:
        print(f"  {feature_names[idx]}: {attributions[idx]: .4f}")


if __name__ == "__main__":
    main()
