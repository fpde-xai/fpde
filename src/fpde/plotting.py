"""Optional matplotlib plotting helpers for FPDE outputs."""

from __future__ import annotations

from typing import Any, Literal, Mapping, Optional, Sequence, Tuple

import numpy as np

from ._array import _as_1d_float, _as_2d_float
from .types import FPDEExplanation


_POSITIVE_COLOR = "#2563eb"
_NEGATIVE_COLOR = "#dc2626"
_RIVAL_COLOR = "#f97316"


def _pyplot():
    try:
        import matplotlib.pyplot as plt
    except ImportError as exc:
        raise ImportError(
            "FPDE plotting requires matplotlib. Install it with "
            "`python -m pip install fpde[plot]` or pass an existing matplotlib Axes."
        ) from exc
    return plt


def _axes(ax: Optional[Any], *, figsize: Tuple[float, float]) -> Any:
    if ax is not None:
        return ax
    _, new_ax = _pyplot().subplots(figsize=figsize)
    return new_ax


def _attribution_vector(attributions: FPDEExplanation | np.ndarray | Sequence[float]) -> np.ndarray:
    if isinstance(attributions, FPDEExplanation):
        values = attributions.attributions
    else:
        values = attributions
    return _as_1d_float("attributions", values).astype(float, copy=False)


def _attribution_matrix(attributions: np.ndarray | Sequence[Sequence[float]]) -> np.ndarray:
    return _as_2d_float("attributions", attributions).astype(float, copy=False)


def _feature_labels(feature_names: Optional[Sequence[Any]], n_features: int) -> np.ndarray:
    if feature_names is None:
        return np.asarray([f"x{i}" for i in range(n_features)], dtype=object)
    labels = np.asarray(list(feature_names), dtype=object)
    if labels.ndim != 1:
        raise ValueError(f"feature_names must be a 1D sequence, got shape={labels.shape}")
    if labels.shape[0] != n_features:
        raise ValueError(f"feature_names length must match attributions: {labels.shape[0]} vs {n_features}")
    return labels


def _normalized(values: np.ndarray) -> np.ndarray:
    denom = float(np.sum(np.abs(values)))
    if denom <= 0.0 or not np.isfinite(denom):
        return np.zeros_like(values, dtype=float)
    return values / denom


def _top_feature_indices(values: np.ndarray, top_k: Optional[int]) -> np.ndarray:
    if values.ndim == 1:
        scores = np.abs(values)
    else:
        scores = np.mean(np.abs(values), axis=0)

    if top_k is None:
        selected = np.arange(scores.shape[0])
    else:
        k = int(top_k)
        if k < 1:
            raise ValueError("top_k must be positive or None")
        selected = np.argsort(scores)[-min(k, scores.shape[0]) :]
    return selected[np.argsort(scores[selected])[::-1]]


def plot_attributions(
    attributions: FPDEExplanation | np.ndarray | Sequence[float],
    *,
    feature_names: Optional[Sequence[Any]] = None,
    top_k: Optional[int] = 20,
    normalize: bool = False,
    sort: bool = True,
    ax: Optional[Any] = None,
    title: Optional[str] = None,
    positive_color: str = _POSITIVE_COLOR,
    negative_color: str = _NEGATIVE_COLOR,
) -> Any:
    """Plot a signed horizontal bar chart for an FPDE attribution vector.

    Parameters
    ----------
    attributions:
        A 1D attribution vector, or an ``FPDEExplanation``.
    feature_names:
        Optional display labels with the same length as ``attributions``.
    top_k:
        Number of largest absolute attributions to show. Use ``None`` to show
        all features.
    normalize:
        If true, L1-normalize values for display. Raw attributions are not
        modified.
    sort:
        If true, order displayed features by absolute attribution magnitude.
    ax:
        Existing matplotlib Axes. If omitted, a new figure and axes are created.
    title:
        Optional axes title.

    Returns
    -------
    matplotlib.axes.Axes
        The axes containing the bar chart.
    """
    values = _attribution_vector(attributions)
    labels = _feature_labels(feature_names, values.shape[0])
    display_values = _normalized(values) if normalize else values.copy()

    if top_k is not None:
        k = int(top_k)
        if k < 1:
            raise ValueError("top_k must be positive or None")
        selected = np.argsort(np.abs(display_values))[-min(k, display_values.shape[0]) :]
    else:
        selected = np.arange(display_values.shape[0])

    if sort:
        selected = selected[np.argsort(np.abs(display_values[selected]))[::-1]]

    selected_values = display_values[selected]
    selected_labels = labels[selected]
    colors = [positive_color if value >= 0.0 else negative_color for value in selected_values]

    ax = _axes(ax, figsize=(7.0, max(2.5, 0.32 * selected.shape[0] + 1.2)))
    y = np.arange(selected.shape[0])
    ax.barh(y, selected_values, color=colors)
    ax.axvline(0.0, color="#111827", linewidth=0.8)
    ax.set_yticks(y)
    ax.set_yticklabels([str(label) for label in selected_labels])
    ax.invert_yaxis()
    ax.set_xlabel("L1-normalized attribution" if normalize else "Attribution")
    if title is not None:
        ax.set_title(title)
    return ax


def plot_attribution_waterfall(
    attributions: FPDEExplanation | np.ndarray | Sequence[float],
    *,
    feature_names: Optional[Sequence[Any]] = None,
    top_k: Optional[int] = 10,
    base_value: float = 0.0,
    ax: Optional[Any] = None,
    title: Optional[str] = None,
    positive_color: str = _POSITIVE_COLOR,
    negative_color: str = _NEGATIVE_COLOR,
    other_label: str = "other features",
) -> Any:
    """Plot a cumulative waterfall for one FPDE attribution vector.

    The bars accumulate from ``base_value`` to ``base_value + sum(attributions)``.
    If ``top_k`` hides features, their remaining signed sum is shown as one
    ``other features`` bar.
    """
    values = _attribution_vector(attributions)
    labels = _feature_labels(feature_names, values.shape[0])
    base = float(base_value)
    if not np.isfinite(base):
        raise ValueError("base_value must be finite")

    selected = _top_feature_indices(values, top_k)
    selected_mask = np.zeros(values.shape[0], dtype=bool)
    selected_mask[selected] = True
    shown_values = values[selected].astype(float, copy=True)
    shown_labels = labels[selected].astype(object, copy=True)
    if top_k is not None and np.any(~selected_mask):
        shown_values = np.concatenate([shown_values, [float(np.sum(values[~selected_mask]))]])
        shown_labels = np.concatenate([shown_labels, [other_label]])

    starts = base + np.concatenate([[0.0], np.cumsum(shown_values[:-1])])
    final_value = float(base + np.sum(values))
    colors = [positive_color if value >= 0.0 else negative_color for value in shown_values]

    ax = _axes(ax, figsize=(7.5, max(2.8, 0.34 * shown_values.shape[0] + 1.4)))
    y = np.arange(shown_values.shape[0])
    ax.barh(y, shown_values, left=starts, color=colors)
    ax.axvline(base, color="#6b7280", linestyle="--", linewidth=0.9)
    ax.axvline(final_value, color="#111827", linewidth=1.0)
    ax.set_yticks(y)
    ax.set_yticklabels([str(label) for label in shown_labels])
    ax.invert_yaxis()
    ax.set_xlabel("Cumulative attribution")
    if title is not None:
        ax.set_title(title)
    return ax


def plot_attribution_summary(
    attributions: np.ndarray | Sequence[Sequence[float]],
    *,
    feature_values: Optional[np.ndarray | Sequence[Sequence[float]]] = None,
    feature_names: Optional[Sequence[Any]] = None,
    top_k: Optional[int] = 20,
    ax: Optional[Any] = None,
    title: Optional[str] = None,
    cmap: str = "coolwarm",
    alpha: float = 0.75,
    jitter: float = 0.18,
    random_state: int = 0,
    colorbar: bool = True,
) -> Any:
    """Plot a batch contribution summary for an attribution matrix.

    Features are ordered by mean absolute attribution. When ``feature_values``
    is provided, points are colored by the corresponding original feature value.
    """
    values = _attribution_matrix(attributions)
    labels = _feature_labels(feature_names, values.shape[1])
    selected = _top_feature_indices(values, top_k)

    feature_value_arr = None
    if feature_values is not None:
        feature_value_arr = _as_2d_float("feature_values", feature_values)
        if feature_value_arr.shape != values.shape:
            raise ValueError(f"feature_values shape must match attributions: {feature_value_arr.shape} vs {values.shape}")

    jitter_value = float(jitter)
    if not np.isfinite(jitter_value) or jitter_value < 0.0:
        raise ValueError("jitter must be a non-negative finite value")
    rng = np.random.default_rng(int(random_state))

    ax = _axes(ax, figsize=(7.0, max(2.8, 0.34 * selected.shape[0] + 1.4)))
    last_scatter = None
    for row, feature_idx in enumerate(selected):
        y = np.full(values.shape[0], row, dtype=float)
        if jitter_value > 0.0:
            y += rng.uniform(-jitter_value, jitter_value, size=values.shape[0])
        x = values[:, feature_idx]
        if feature_value_arr is None:
            point_colors = np.where(x >= 0.0, _POSITIVE_COLOR, _NEGATIVE_COLOR)
            last_scatter = ax.scatter(x, y, c=point_colors, alpha=alpha, edgecolors="none")
        else:
            last_scatter = ax.scatter(
                x,
                y,
                c=feature_value_arr[:, feature_idx],
                cmap=cmap,
                alpha=alpha,
                edgecolors="none",
            )

    ax.axvline(0.0, color="#111827", linewidth=0.8)
    ax.set_yticks(np.arange(selected.shape[0]))
    ax.set_yticklabels([str(label) for label in labels[selected]])
    ax.invert_yaxis()
    ax.set_xlabel("Attribution")
    if title is not None:
        ax.set_title(title)
    if feature_value_arr is not None and colorbar and last_scatter is not None:
        ax.figure.colorbar(last_scatter, ax=ax, fraction=0.046, pad=0.04, label="Feature value")
    return ax


def plot_attribution_image(
    attributions: FPDEExplanation | np.ndarray | Sequence[float],
    shape: Optional[Tuple[int, int]] = None,
    *,
    ax: Optional[Any] = None,
    title: Optional[str] = None,
    cmap: str = "coolwarm",
    colorbar: bool = True,
    symmetric: bool = True,
    interpolation: str = "nearest",
) -> Any:
    """Plot an attribution vector as a 2D heatmap.

    ``shape`` is required for 1D vectors and optional for already-2D arrays.
    The returned value is the matplotlib Axes.
    """
    if isinstance(attributions, FPDEExplanation):
        raw = np.asarray(attributions.attributions, dtype=float)
    else:
        raw = np.asarray(attributions, dtype=float)
    if not np.all(np.isfinite(raw)):
        raise ValueError("attributions contains NaN or inf")

    if raw.ndim == 1:
        if shape is None:
            raise ValueError("shape is required when attributions is a 1D vector")
        values = raw.reshape(shape)
    elif raw.ndim == 2:
        if shape is not None and raw.shape != tuple(shape):
            raise ValueError(f"shape {shape!r} does not match 2D attributions shape {raw.shape}")
        values = raw
    else:
        raise ValueError(f"attributions must be 1D or 2D, got shape={raw.shape}")

    if symmetric:
        vmax = float(np.max(np.abs(values))) if values.size else 0.0
        if vmax > 0.0:
            vmin = -vmax
        else:
            vmin = vmax = None
    else:
        vmin = vmax = None

    ax = _axes(ax, figsize=(3.6, 3.2))
    image = ax.imshow(values, cmap=cmap, vmin=vmin, vmax=vmax, interpolation=interpolation)
    ax.set_xticks([])
    ax.set_yticks([])
    if title is not None:
        ax.set_title(title)
    if colorbar:
        ax.figure.colorbar(image, ax=ax, fraction=0.046, pad=0.04)
    return ax


def plot_perturbation_curves(
    curves: Mapping[str, Sequence[float]],
    *,
    ax: Optional[Any] = None,
    title: Optional[str] = None,
    deletion_label: str = "deletion",
    insertion_label: str = "insertion",
) -> Any:
    """Plot deletion and insertion probabilities returned by ``perturbation_curves``."""
    required = ("fractions", "deletion_prob", "insertion_prob")
    missing = [name for name in required if name not in curves]
    if missing:
        raise ValueError(f"curves is missing required keys: {', '.join(missing)}")

    fractions = _as_1d_float("curves['fractions']", curves["fractions"])
    deletion = _as_1d_float("curves['deletion_prob']", curves["deletion_prob"])
    insertion = _as_1d_float("curves['insertion_prob']", curves["insertion_prob"])
    if fractions.shape != deletion.shape or fractions.shape != insertion.shape:
        raise ValueError("fractions, deletion_prob, and insertion_prob must have the same length")

    ax = _axes(ax, figsize=(5.8, 3.6))
    ax.plot(fractions, deletion, marker="o", label=deletion_label)
    ax.plot(fractions, insertion, marker="o", label=insertion_label)
    ax.set_xlabel("Fraction of features perturbed")
    ax.set_ylabel("Target probability")
    ax.set_xlim(0.0, 1.0)
    ax.set_ylim(0.0, 1.0)
    ax.legend()
    if title is not None:
        ax.set_title(title)
    return ax


def prepare_local_contribution_data(
    feature_names: Sequence[Any],
    contributions: np.ndarray | Sequence[float],
    values: Optional[Sequence[Any]] = None,
    *,
    top_k: int = 10,
    sort_by: str = "abs",
    positive_label: str = "supports prediction",
    negative_label: str = "opposes prediction",
) -> list[dict[str, Any]]:
    """Prepare local FPDE contributions for a signed bar plot.

    This function formats FPDE attribution values only. It does not implement
    perturbation or local surrogate model algorithms.
    """
    contribution_arr = _as_1d_float("contributions", contributions)
    if contribution_arr.size == 0:
        raise ValueError("contributions must contain at least one value")

    k = int(top_k)
    if k < 1:
        raise ValueError("top_k must be positive")
    if sort_by not in {"abs", "value"}:
        raise ValueError("sort_by must be 'abs' or 'value'")

    labels = _feature_labels(feature_names, contribution_arr.shape[0])
    value_labels = None
    if values is not None:
        value_labels = np.asarray(list(values), dtype=object)
        if value_labels.ndim != 1:
            raise ValueError(f"values must be a 1D sequence, got shape={value_labels.shape}")
        if value_labels.shape[0] != contribution_arr.shape[0]:
            raise ValueError(f"values length must match contributions: {value_labels.shape[0]} vs {contribution_arr.shape[0]}")

    if sort_by == "abs":
        order = np.argsort(np.abs(contribution_arr))[::-1]
    else:
        order = np.argsort(contribution_arr)[::-1]
    selected = order[: min(k, contribution_arr.shape[0])]

    rows: list[dict[str, Any]] = []
    for idx in selected:
        contribution = float(contribution_arr[int(idx)])
        if contribution > 0.0:
            direction = "positive"
            direction_label = positive_label
        elif contribution < 0.0:
            direction = "negative"
            direction_label = negative_label
        else:
            direction = "zero"
            direction_label = "zero contribution"

        feature = str(labels[int(idx)])
        if value_labels is None:
            display_name = feature
        else:
            display_name = f"{feature} = {value_labels[int(idx)]}"
        rows.append(
            {
                "feature": feature,
                "display_name": display_name,
                "contribution": contribution,
                "abs_contribution": abs(contribution),
                "direction": direction,
                "direction_label": direction_label,
            }
        )
    return rows


def plot_local_contributions(
    feature_names: Sequence[Any],
    contributions: np.ndarray | Sequence[float],
    values: Optional[Sequence[Any]] = None,
    *,
    top_k: int = 10,
    sort_by: str = "abs",
    ax: Optional[Any] = None,
    title: Optional[str] = None,
    xlabel: str = "Contribution",
    positive_label: str = "supports prediction",
    negative_label: str = "opposes prediction",
    show_values: bool = True,
    zero_line: bool = True,
) -> Any:
    """Plot FPDE local contributions as a signed horizontal bar chart."""
    rows = prepare_local_contribution_data(
        feature_names,
        contributions,
        values if show_values else None,
        top_k=top_k,
        sort_by=sort_by,
        positive_label=positive_label,
        negative_label=negative_label,
    )

    ax = _axes(ax, figsize=(7.0, max(2.5, 0.34 * len(rows) + 1.2)))
    y = np.arange(len(rows))
    contribution_values = [row["contribution"] for row in rows]
    colors = [
        _POSITIVE_COLOR if row["direction"] == "positive" else _NEGATIVE_COLOR if row["direction"] == "negative" else "#64748b"
        for row in rows
    ]
    ax.barh(y, contribution_values, color=colors)
    if zero_line:
        ax.axvline(0.0, color="#111827", linewidth=0.8)
    ax.set_yticks(y)
    ax.set_yticklabels([row["display_name"] for row in rows])
    ax.invert_yaxis()
    ax.set_xlabel(xlabel)
    if title is not None:
        ax.set_title(title)
    return ax


def plot_similarity_distribution(
    similarities: np.ndarray | Sequence[float],
    *,
    target_similarity: Optional[float] = None,
    ax: Optional[Any] = None,
    title: Optional[str] = None,
    bins: int = 30,
    color: str = "#64748b",
    target_color: str = _POSITIVE_COLOR,
    xlabel: str = "Similarity",
) -> Any:
    """Plot a representative-instance similarity distribution."""
    values = _as_1d_float("similarities", similarities)
    if values.size == 0:
        raise ValueError("similarities must contain at least one value")
    bin_count = int(bins)
    if bin_count < 1:
        raise ValueError("bins must be positive")

    ax = _axes(ax, figsize=(5.8, 3.6))
    ax.hist(values, bins=bin_count, color=color, alpha=0.85)
    if target_similarity is not None:
        target = float(target_similarity)
        if not np.isfinite(target):
            raise ValueError("target_similarity must be finite")
        ax.axvline(target, color=target_color, linewidth=1.4, label="target")
        ax.legend()
    ax.set_xlabel(xlabel)
    ax.set_ylabel("Count")
    if title is not None:
        ax.set_title(title)
    return ax


def compute_prototype_similarities(
    X: np.ndarray | Sequence[Sequence[float]],
    prototype: np.ndarray | Sequence[float],
    metric: Literal["cosine", "negative_euclidean"] = "cosine",
) -> np.ndarray:
    """Compute row-wise similarities between samples and one prototype."""
    X_arr = _as_2d_float("X", X)
    prototype_arr = _as_1d_float("prototype", prototype)
    if X_arr.shape[0] == 0:
        raise ValueError("X must contain at least one sample")
    if X_arr.shape[1] == 0:
        raise ValueError("X must contain at least one feature")
    if prototype_arr.size == 0:
        raise ValueError("prototype must contain at least one feature")
    if X_arr.shape[1] != prototype_arr.shape[0]:
        raise ValueError(f"prototype length must match X feature dimension: {prototype_arr.shape[0]} vs {X_arr.shape[1]}")

    if metric == "cosine":
        prototype_norm = float(np.linalg.norm(prototype_arr))
        row_norms = np.linalg.norm(X_arr, axis=1)
        denom = row_norms * prototype_norm
        similarities = np.zeros(X_arr.shape[0], dtype=float)
        nonzero = denom > 0.0
        if np.any(nonzero):
            similarities[nonzero] = (X_arr[nonzero] @ prototype_arr) / denom[nonzero]
        return similarities

    if metric == "negative_euclidean":
        return -np.linalg.norm(X_arr - prototype_arr, axis=1)

    raise ValueError("metric must be 'cosine' or 'negative_euclidean'")


def plot_prototype_similarity_distribution(
    X: np.ndarray | Sequence[Sequence[float]],
    target_prototype: np.ndarray | Sequence[float],
    *,
    rival_prototype: Optional[np.ndarray | Sequence[float]] = None,
    x: Optional[np.ndarray | Sequence[float]] = None,
    metric: Literal["cosine", "negative_euclidean"] = "cosine",
    bins: int = 30,
    ax: Optional[Any] = None,
    title: Optional[str] = None,
    target_label: str = "target prototype",
    rival_label: str = "rival prototype",
    sample_label: str = "explained sample",
) -> Any:
    """Plot FPDE target/rival prototype similarity distributions."""
    bin_count = int(bins)
    if bin_count < 1:
        raise ValueError("bins must be positive")

    target_similarities = compute_prototype_similarities(X, target_prototype, metric)
    rival_similarities = None
    if rival_prototype is not None:
        rival_similarities = compute_prototype_similarities(X, rival_prototype, metric)

    sample_arr = None
    if x is not None:
        sample_arr = _as_1d_float("x", x)
        target_arr = _as_1d_float("target_prototype", target_prototype)
        if sample_arr.shape[0] != target_arr.shape[0]:
            raise ValueError(
                f"x length must match target_prototype length: {sample_arr.shape[0]} vs "
                f"{target_arr.shape[0]}"
            )

    ax = _axes(ax, figsize=(6.2, 3.8))
    ax.hist(target_similarities, bins=bin_count, color=_POSITIVE_COLOR, alpha=0.55, label=target_label)
    if rival_similarities is not None:
        ax.hist(rival_similarities, bins=bin_count, color=_RIVAL_COLOR, alpha=0.45, label=rival_label)

    if sample_arr is not None:
        target_sample = compute_prototype_similarities(sample_arr.reshape(1, -1), target_prototype, metric)[0]
        ax.axvline(
            float(target_sample),
            color=_POSITIVE_COLOR,
            linewidth=1.5,
            label=f"{sample_label} vs {target_label}",
        )
        if rival_prototype is not None:
            rival_sample = compute_prototype_similarities(sample_arr.reshape(1, -1), rival_prototype, metric)[0]
            ax.axvline(
                float(rival_sample),
                color=_RIVAL_COLOR,
                linestyle="--",
                linewidth=1.5,
                label=f"{sample_label} vs {rival_label}",
            )

    ax.set_xlabel("Cosine similarity" if metric == "cosine" else "Negative Euclidean distance")
    ax.set_ylabel("Count")
    ax.legend()
    if title is not None:
        ax.set_title(title)
    return ax


__all__ = [
    "compute_prototype_similarities",
    "prepare_local_contribution_data",
    "plot_attributions",
    "plot_attribution_waterfall",
    "plot_attribution_summary",
    "plot_attribution_image",
    "plot_local_contributions",
    "plot_perturbation_curves",
    "plot_similarity_distribution",
    "plot_prototype_similarity_distribution",
]
