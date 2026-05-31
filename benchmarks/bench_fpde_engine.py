from __future__ import annotations

import gc
import sys
import time
from pathlib import Path
from statistics import mean

import numpy as np

ROOT = Path(__file__).resolve().parents[1]
SRC = ROOT / "src"
if str(SRC) not in sys.path:
    sys.path.insert(0, str(SRC))

from fpde import (  # noqa: E402
    FPDEEngine,
)


class LinearSoftmaxModel:
    def __init__(self, n_features: int, n_classes: int, *, seed: int = 123) -> None:
        rng = np.random.default_rng(seed)
        self.classes_ = np.arange(n_classes, dtype=int)
        self.coef_ = rng.normal(scale=0.08, size=(n_features, n_classes))
        self.bias_ = rng.normal(scale=0.1, size=n_classes)

    def predict_proba(self, X: np.ndarray) -> np.ndarray:
        logits = np.asarray(X, dtype=float) @ self.coef_ + self.bias_
        logits -= np.max(logits, axis=1, keepdims=True)
        exp = np.exp(logits)
        return exp / np.sum(exp, axis=1, keepdims=True)


def time_call(label: str, fn, *, repeats: int = 5, warmups: int = 1) -> None:
    for _ in range(warmups):
        fn()
    timings = []
    for _ in range(repeats):
        gc.collect()
        start = time.perf_counter()
        fn()
        timings.append(time.perf_counter() - start)
    print(f"{label:36s} min={min(timings):.4f}s mean={mean(timings):.4f}s repeats={repeats}")


def main() -> None:
    rng = np.random.default_rng(42)
    n_train = 300
    n_eval = 64
    n_val = 64
    n_features = 1000
    n_classes = 3
    lambda_grid = tuple(np.linspace(0.0, 1.0, 11))
    fractions = (0.0, 0.05, 0.1, 0.2, 0.3, 0.5, 0.7, 1.0)

    X_train = rng.normal(size=(n_train, n_features))
    y_train = rng.integers(0, n_classes, size=n_train)
    X_eval = rng.normal(size=(n_eval, n_features))
    X_val = rng.normal(size=(n_val, n_features))
    model = LinearSoftmaxModel(n_features, n_classes)
    engine = FPDEEngine.fit(X_train, y_train, model=model)

    print("FPDE engine benchmark")
    print(
        f"dims: n_train={n_train} n_eval={n_eval} n_val={n_val} "
        f"n_features={n_features} n_classes={n_classes} "
        f"lambdas={len(lambda_grid)} fractions={len(fractions)}"
    )

    time_call(
        "batch explanation",
        lambda: engine.explain_matrix(X_eval, lambda_hyb=0.5),
    )
    time_call(
        "grid search",
        lambda: engine.grid_search(X_eval, lambda_hyb_grid=lambda_grid),
        repeats=3,
    )
    time_call(
        "validation selection",
        lambda: engine.select_lambda(
            X_val,
            lambda_hyb_grid=lambda_grid,
            fractions=fractions,
            max_working_bytes=128 * 1024 * 1024,
        ),
        repeats=3,
    )


if __name__ == "__main__":
    main()
