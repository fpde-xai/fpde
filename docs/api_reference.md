# API Reference

Import the public package as `fpde`.

```python
from fpde import FPDEEngine, class_mean_prototypes, diff_fpde
```

This page documents the public API exported by `fpde` and `fpde.core`.

## Engine API

Use `FPDEEngine` for repeated explanations, batch explanations, Hyb-FPDE, grid
search, validation-based lambda selection, and experimental Bayesian-FPDE
lambda posterior selection.

### `FPDEEngine.fit`

```python
FPDEEngine.fit(X_train, y_train, model=None, baseline=None)
```

Fits reusable FPDE state from training data.

| Parameter | Description |
| --- | --- |
| `X_train` | Training feature matrix with shape `(n_samples, n_features)`. |
| `y_train` | Training labels with one label per row in `X_train`. |
| `model` | Optional classifier. Required later when an operation needs `predict_proba`. |
| `baseline` | Optional replacement vector for perturbation curves. Defaults to the training mean. |

Returns an `FPDEEngine`.

### `engine.explain_one`

```python
engine.explain_one(
    x,
    *,
    lambda_hyb,
    normalize="l1",
    anchor_strategy="mean",
    eps=1e-12,
    model=None,
)
```

Explains one sample with fixed-lambda Hyb-FPDE.

| Parameter | Description |
| --- | --- |
| `x` | One feature vector with length `n_features`. |
| `lambda_hyb` | Diff/Cos mixture weight in `[0, 1]`. |
| `normalize` | `"l1"` to normalize components before mixing, or `"none"` for raw scales. |
| `anchor_strategy` | `"mean"`, `"zero"`, or `"none"` for the Cos-FPDE anchor. |
| `eps` | Positive regularization value for cosine norms. |
| `model` | Optional model override. Uses the engine model when omitted. |

Returns `(attributions, details)`.

The `details` dictionary includes `target_label`, `rival_label`,
`target_probability`, `lambda_hyb`, `evidence`, `exactness_residual`,
`positive_score`, and `negative_score`.

### `engine.explain_batch`

```python
engine.explain_batch(
    X,
    *,
    lambda_hyb,
    normalize="l1",
    anchor_strategy="mean",
    include_details=True,
    eps=1e-12,
    model=None,
)
```

Explains many samples with fixed-lambda Hyb-FPDE.

Returns `(attribution_matrix, details)`. If `include_details=False`, `details`
is an empty list.

### `engine.explain_matrix`

```python
engine.explain_matrix(
    X,
    *,
    lambda_hyb,
    normalize="l1",
    anchor_strategy="mean",
    eps=1e-12,
    model=None,
)
```

Returns only the attribution matrix for a batch.

### `engine.grid_search`

```python
engine.grid_search(
    X_eval,
    *,
    predictor=None,
    objective="blackbox_agreement",
    fpde_mode_grid=("diff", "cos", "hyb_grid"),
    normalize_grid=("l1",),
    lambda_hyb_grid=...,
    anchor_strategy_grid=("mean",),
    include_explicit_diff_cos=True,
    max_eval_samples=None,
    eps=1e-12,
    verbose=False,
)
```

Searches Diff-FPDE, Cos-FPDE, and Hyb-FPDE candidate settings.

Supported objectives:

| Objective | Description |
| --- | --- |
| `"blackbox_agreement"` | Scores whether FPDE evidence agrees with the model-selected target/rival contrast. |
| `"mean_positive_evidence"` | Scores the mean positive evidence across evaluation samples. |
| `"mean_margin_weighted_evidence"` | Weights positive evidence by the model probability margin when probabilities are available. |

Returns a `HybFPDEGridSearchResult`.

### `engine.select_lambda`

```python
engine.select_lambda(
    X_val,
    *,
    lambda_hyb_grid=...,
    fractions=(0.0, 0.05, 0.1, 0.2, 0.3, 0.5, 0.7, 1.0),
    normalize="l1",
    anchor_strategy="mean",
    eps=1e-12,
    max_working_bytes=268435456,
    model=None,
)
```

Selects `lambda_hyb` by held-out deletion and insertion validation.

Returns a `HybFPDEValidationSelectionResult` with `best_lambda`,
`best_config`, `rows`, and `n_eval_samples`.

### `engine.select_bayesian_lambda`

```python
engine.select_bayesian_lambda(
    X_val,
    *,
    lambda_hyb_grid=...,
    fractions=(0.0, 0.05, 0.1, 0.2, 0.3, 0.5, 0.7, 1.0),
    normalize="l1",
    anchor_strategy="mean",
    eps=1e-12,
    max_working_bytes=268435456,
    alpha=1.0,
    beta=1.0,
    temperature=1.0,
    credible_mass=0.95,
    model=None,
)
```

Builds a Bayesian posterior over unique `lambda_hyb` candidates using the same
held-out deletion and insertion validation score as `select_lambda`.

This API models uncertainty over the finite `lambda_hyb` grid only. It does
not implement prototype posterior sampling, feature-level credible intervals,
or black-box model uncertainty.

| Parameter | Description |
| --- | --- |
| `alpha` | Positive alpha parameter for the Beta prior on `lambda_hyb`. |
| `beta` | Positive beta parameter for the Beta prior on `lambda_hyb`. |
| `temperature` | Positive likelihood temperature. Smaller values make the posterior sharper. |
| `credible_mass` | Credible interval mass in `(0, 1)`. |

Returns a `BayesianFPDELambdaSelectionResult`.

### `engine.explain_one_bayesian`

```python
engine.explain_one_bayesian(x, selection, *, model=None)
```

Explains one sample using `selection.posterior_mean_lambda`, where `selection`
is a `BayesianFPDELambdaSelectionResult`.

Because the posterior is over `lambda_hyb`, returned attributions are computed
with the posterior mean lambda. They are not sampled-prototype attribution
summaries.

Returns `(attributions, details)`. The `details` dictionary includes the usual
fixed-lambda fields plus `lambda_source`, `posterior_mean_lambda`,
`map_lambda`, `credible_interval`, `posterior_entropy`, and
`effective_candidates`.

### `engine.explain_batch_bayesian`

```python
engine.explain_batch_bayesian(
    X,
    selection,
    *,
    include_details=True,
    model=None,
)
```

Explains many samples with the Bayesian posterior mean lambda. Returns
`(attribution_matrix, details)`.

### `engine.explain_matrix_bayesian`

```python
engine.explain_matrix_bayesian(X, selection, *, model=None)
```

Returns only the Bayesian-FPDE attribution matrix.

## Prototype Helpers

### `class_mean_prototypes`

```python
class_mean_prototypes(X, y)
```

Builds one mean prototype per class.

Returns `(prototypes, labels)`, where `prototypes` has shape
`(n_classes, n_features)` and `labels` contains the class label for each
prototype row.

### `select_prototype_pair`

```python
select_prototype_pair(
    x,
    prototypes,
    prototype_labels,
    *,
    positive_label,
    negative_label=None,
    mode="diff",
    anchor=None,
    eps=1e-12,
)
```

Selects the positive and negative prototype indices for a local contrast.

Selection behavior:

- `"diff"`: nearest positive and nearest negative prototypes by squared
  distance.
- `"cos"`: most cosine-similar positive and negative prototypes.
- `"hyb_grid"`: Diff prototype selection.

Returns `(positive_index, negative_index)`.

### `prepare_fpde_context`

```python
prepare_fpde_context(X_train, y_train, *, baseline=None)
```

Precomputes reusable prototypes, anchors, baseline, and feature metadata.

Returns an `FPDEContext`.

## Dynamic-FPDE API

Dynamic-FPDE explains frame-level time-series feature matrices with shape
`(T, F)` and returns attribution matrices with the same shape. It is prototype
evidence decomposition, not a causal explanation. Dynamic-FPDE evidence values
and validation curves are prototype evidence scores, not probabilities.

### `prepare_dynamic_fpde_context`

```python
prepare_dynamic_fpde_context(
    X_train,
    y_train,
    *,
    prototype_length=128,
    alignment="linear",
    baseline="mean",
)
```

Builds one class-mean temporal prototype per label from a sequence of
variable-length `(T_i, F)` arrays. Each training sequence is linearly resampled
to `prototype_length`.

Returns a `DynamicFPDEContext` with prototypes shaped `(n_classes,
prototype_length, n_features)`, labels, mean and zero anchors, and metadata.
Only `alignment="linear"` is supported in v0.1.

### `resample_time_series_linear`

```python
resample_time_series_linear(X, target_length)
```

Linearly resamples a finite 2D sequence matrix to `(target_length, F)`.
Single-frame inputs are repeated.

### `dynamic_diff_fpde`

```python
dynamic_diff_fpde(X, P_target, P_rival)
```

Computes:

```text
Phi_diff[t, f] = (X[t, f] - P_rival[t, f])**2
               - (X[t, f] - P_target[t, f])**2
```

Returns `(Phi_diff, E_diff)`, where `E_diff = Phi_diff.sum()`.

### `dynamic_cos_fpde`

```python
dynamic_cos_fpde(X, P_target, P_rival, *, anchor=None, eps=1e-12)
```

Computes a coordinate decomposition of the cosine contrast between
`X - anchor`, `P_target - anchor`, and `P_rival - anchor`. Returns
`(Phi_cos, E_cos)`.

### `dynamic_hyb_fpde`

```python
dynamic_hyb_fpde(
    X,
    P_target,
    P_rival,
    *,
    lambda_hyb=0.5,
    normalize="l1",
    anchor=None,
    eps=1e-12,
)
```

Mixes Dynamic-Diff and Dynamic-Cos attribution matrices. `lambda_hyb=1.0` is
the Dynamic-Diff endpoint, and `lambda_hyb=0.0` is the Dynamic-Cos endpoint.
`normalize` may be `"l1"` or `"none"`.

Returns `(Phi_hyb, E_hyb, details)`.

### `dynamic_fpde_explain_one`

```python
dynamic_fpde_explain_one(
    X,
    context,
    *,
    target_label,
    rival_label=None,
    mode="dynamic_hyb",
    lambda_hyb=0.5,
    normalize="l1",
    anchor_strategy="mean",
    eps=1e-12,
)
```

Explains one sequence. If `rival_label` is omitted, the closest non-target
prototype is selected after resampling prototypes to the sample length.

Returns a `DynamicFPDEExplanation` with `attributions`, `time_importance`,
`feature_importance`, evidence, target/rival labels, residual, and details.

### `dynamic_fpde_explain_batch`

```python
dynamic_fpde_explain_batch(
    X_list,
    context,
    *,
    target_labels,
    rival_labels=None,
    mode="dynamic_hyb",
    lambda_hyb=0.5,
    normalize="l1",
    anchor_strategy="mean",
    eps=1e-12,
)
```

Explains many variable-length sequences by calling
`dynamic_fpde_explain_one` for each sample.

### `temporal_deletion_insertion_curves`

```python
temporal_deletion_insertion_curves(
    X,
    explanation,
    context,
    *,
    target_label,
    rival_label,
    steps=20,
    baseline_strategy="mean",
    rank_by="positive",
    eps=1e-12,
)
```

Ranks frames by `explanation.time_importance`, then computes prototype-driven
deletion and insertion curves using Dynamic-Diff evidence. `rank_by` controls
the frame ordering:

- `"positive"`: rank by `max(time_importance, 0)`, descending. This is the
  default for target-supporting-frame evaluation.
- `"signed"`: rank by signed `time_importance`, descending.
- `"absolute"`: rank by `abs(time_importance)`, descending.

The raw `deletion_curve` and `insertion_curve` contain unbounded prototype
evidence scores. The AUC metrics are computed from normalized curves:

```text
scale = abs(original_evidence - baseline_evidence) + eps
deletion_drop_curve = (original_evidence - deletion_curve) / scale
insertion_gain_curve = (insertion_curve - insertion_curve[0]) / scale
```

Returns raw curves, normalized curves, `deletion_drop_auc`,
`insertion_gain_auc`, `combined_score`, and metadata. `insertion_auc` is kept
as a backward-compatible alias for `insertion_gain_auc`; prefer
`insertion_gain_auc` in new code.

### `select_dynamic_lambda`

```python
select_dynamic_lambda(
    X_val,
    y_val,
    context,
    *,
    lambda_grid=None,
    mode="dynamic_hyb",
    normalize="l1",
    anchor_strategy="mean",
    steps=20,
    rank_by="positive",
    eps=1e-12,
)
```

Evaluates Dynamic-Hyb lambda candidates with temporal deletion/insertion
metrics from normalized prototype-evidence curves. The default lambda grid is
`[0.0, 0.25, 0.5, 0.75, 1.0]`.

Returns a dictionary with `best_lambda`, candidate `rows`, metric means, and
the best row.

### Dynamic Plotting Helpers

```python
plot_dynamic_time_importance(explanation, *, ax=None, title=None)
plot_dynamic_attribution_heatmap(explanation, *, ax=None, title=None)
```

These optional helpers require matplotlib only when no existing axes object is
provided.

## Explanation Functions

Use these functions when you want direct control over prototypes and labels.

### `diff_fpde`

```python
diff_fpde(
    x,
    p_pos,
    p_neg,
    *,
    positive_label="positive",
    negative_label="negative",
    positive_prototype_index=-1,
    negative_prototype_index=-1,
)
```

Computes a Diff-FPDE explanation for one target/rival prototype pair.

### `cos_fpde`

```python
cos_fpde(
    x,
    p_pos,
    p_neg,
    *,
    anchor=None,
    eps=1e-12,
    positive_label="positive",
    negative_label="negative",
    positive_prototype_index=-1,
    negative_prototype_index=-1,
)
```

Computes a Cos-FPDE explanation for one target/rival prototype pair.

### `explain_with_selected_prototypes`

```python
explain_with_selected_prototypes(
    x,
    prototypes,
    prototype_labels,
    *,
    positive_label,
    negative_label=None,
    mode="diff",
    anchor=None,
    eps=1e-12,
)
```

Selects prototypes and computes a public Diff-FPDE or Cos-FPDE explanation.
Use `FPDEEngine` for Hyb-FPDE.

## Metrics And Probability Helpers

### `regularized_cosine`

```python
regularized_cosine(u, v, eps=1e-12)
```

Returns cosine similarity with epsilon-regularized norms.

### `top_two_labels`

```python
top_two_labels(model, x)
```

Returns `(target_label, rival_label, probability_vector)` for one sample.
`model` must implement `predict_proba` and expose `classes_`.

### `predict_proba_for_label`

```python
predict_proba_for_label(model, X, label)
```

Returns the `predict_proba(X)` column for `label`.

### `perturbation_curves`

```python
perturbation_curves(
    model,
    x,
    attributions,
    target_label,
    baseline,
    fractions=(0.0, 0.05, 0.1, 0.2, 0.3, 0.5, 0.7, 1.0),
)
```

Computes deletion and insertion curves for one attribution vector. Features are
ranked by signed positive attribution in descending order.

Returns a dictionary containing:

- `fractions`
- `deletion_prob`
- `insertion_prob`
- `p0`
- `deletion_auc`
- `deletion_drop_auc`
- `insertion_auc`
- `combined_score`

## Plotting Helpers

The plotting helpers are exported from `fpde` and `fpde.core`, and are also
available from `fpde.plotting`. They require matplotlib only when called without
an existing axes object. The local bar and heatmap cover the compact
local contribution view, while the waterfall and summary helpers provide
cumulative and batch-level contribution plots. The similarity helpers provide
FPDE-native views of representative sample and target/rival prototype
similarity distributions.

Install the optional plotting dependency with:

```bash
python -m pip install "fpde[plot]"
```

### `plot_attributions`

```python
plot_attributions(
    attributions,
    *,
    feature_names=None,
    top_k=20,
    normalize=False,
    sort=True,
    ax=None,
    title=None,
    interval_low=None,
    interval_high=None,
    interval_label="Bayesian lambda range",
)
```

Plots a signed horizontal bar chart for a 1D attribution vector or an
`FPDEExplanation`. Positive values support the target class; negative values
support the rival class.

Pass `interval_low` and `interval_high` to draw horizontal error bars for each
feature. For Bayesian-FPDE, compute those bounds from the attribution vectors
at the lower and upper `selection.credible_interval` lambda values. The range
is display-only and represents uncertainty over `lambda_hyb`, not sampled
prototype uncertainty.

### `plot_attribution_waterfall`

```python
plot_attribution_waterfall(
    attributions,
    *,
    feature_names=None,
    top_k=10,
    base_value=0.0,
    ax=None,
    title=None,
)
```

Plots a cumulative local explanation from `base_value` to
`base_value + sum(attributions)`. If `top_k` hides features, their signed
remainder is shown as an `"other features"` bar.

### `plot_attribution_summary`

```python
plot_attribution_summary(
    attributions,
    *,
    feature_values=None,
    feature_names=None,
    top_k=20,
    ax=None,
    title=None,
)
```

Plots a batch attribution matrix as a distribution summary ordered by mean
absolute attribution. Pass `feature_values` with the same shape to color points
by original feature value.

### `plot_attribution_image`

```python
plot_attribution_image(
    attributions,
    shape=None,
    *,
    ax=None,
    title=None,
    cmap="coolwarm",
    colorbar=True,
    symmetric=True,
)
```

Plots attributions as a 2D heatmap. Pass `shape=(height, width)` for flat image
vectors.

### `plot_perturbation_curves`

```python
plot_perturbation_curves(curves, *, ax=None, title=None)
```

Plots the deletion and insertion probability curves returned by
`perturbation_curves`.

### `prepare_local_contribution_data`

```python
prepare_local_contribution_data(
    feature_names,
    contributions,
    values=None,
    *,
    top_k=10,
    sort_by="abs",
)
```

Formats one FPDE local attribution vector for a signed local contribution view.
The returned rows include the feature name, display name, signed contribution,
absolute contribution, direction, and direction label.

### `plot_local_contributions`

```python
plot_local_contributions(
    feature_names,
    contributions,
    values=None,
    *,
    top_k=10,
    sort_by="abs",
    ax=None,
    title=None,
    xlabel="Contribution",
)
```

Plots a signed horizontal bar chart for one FPDE local attribution vector.
Positive and negative contributions are color-separated, and `x=0` is marked by
default. The function returns the matplotlib Axes and does not call
`plt.show()` or save files.

### `plot_similarity_distribution`

```python
plot_similarity_distribution(
    similarities,
    *,
    target_similarity=None,
    ax=None,
    title=None,
    bins=30,
)
```

Plots a histogram of representative-instance similarities. Use
`target_similarity` to mark the explained sample or selected prototype.

### `compute_prototype_similarities`

```python
compute_prototype_similarities(X, prototype, metric="cosine")
```

Computes row-wise similarities between a sample matrix and one prototype.

Supported metrics:

- `"cosine"`: cosine similarity. Rows or prototypes with zero norm receive
  similarity `0.0`.
- `"negative_euclidean"`: negative Euclidean distance, so larger values are
  closer to the prototype.

Returns a 1D NumPy array with one similarity value per row in `X`.

### `plot_prototype_similarity_distribution`

```python
plot_prototype_similarity_distribution(
    X,
    target_prototype,
    *,
    rival_prototype=None,
    x=None,
    metric="cosine",
    bins=30,
    ax=None,
    title=None,
)
```

Plots the distribution of similarities from rows in `X` to a target prototype.
Pass `rival_prototype` to overlay the rival-prototype distribution, and pass
`x` to mark the explained sample's position against the target and rival
prototypes. The returned value is the matplotlib Axes.

## `fpde.plots` Contribution Plot API

`fpde.plots` provides a compact matplotlib API for FPDE contribution arrays.
Use `show=False` to receive an Axes without displaying it, then save with
`ax.figure.savefig(...)`.

```python
from fpde.plots import FPDEPlotExplanation, bar, beeswarm, scatter, waterfall

exp = FPDEPlotExplanation(
    values=attribution_matrix,
    data=X_eval,
    feature_names=feature_names,
)

ax = beeswarm(exp, show=False)
ax.figure.savefig("fpde_beeswarm.png", dpi=150, bbox_inches="tight")
```

### `FPDEPlotExplanation`

```python
FPDEPlotExplanation(
    values,
    base_values=None,
    data=None,
    feature_names=None,
    output_names=None,
    predictions=None,
)
```

Lightweight plotting container for FPDE contribution matrices. Existing FPDE
result objects with `.attributions` can also be passed directly to plot
functions as a single-sample explanation.

### `bar`

```python
bar(
    explanation=None,
    values=None,
    feature_names=None,
    max_display=10,
    order="mean_abs",
    ax=None,
    show=True,
    title=None,
    figsize=None,
)
```

Plots local or global contribution importance. A 1D value vector is treated as
one local explanation; a 2D matrix is summarized by feature importance.

### `beeswarm`

```python
beeswarm(
    explanation=None,
    values=None,
    data=None,
    feature_names=None,
    max_display=10,
    order="mean_abs",
    color_by_value=True,
    ax=None,
    show=True,
    title=None,
    figsize=None,
)
```

Plots per-sample contributions grouped by feature. When feature data with the
same shape as `values` is available, points are colored by the corresponding
feature value.

### `waterfall`

```python
waterfall(
    explanation=None,
    values=None,
    base_value=None,
    prediction=None,
    feature_names=None,
    max_display=10,
    ax=None,
    show=True,
    title=None,
    figsize=None,
)
```

Plots one contribution vector from `base_value` to `prediction`. If prediction
is omitted, it is computed as `base_value + sum(values)`. Hidden features are
grouped into `"other features"`.

### `scatter`

```python
scatter(
    explanation=None,
    values=None,
    data=None,
    feature=None,
    feature_names=None,
    color_feature=None,
    ax=None,
    show=True,
    title=None,
    figsize=None,
)
```

Plots one feature's original value against its FPDE contribution. `feature` and
`color_feature` may be integer indices or names from `feature_names`. If data
is unavailable, the x-axis falls back to sample index.

## Result Objects

### `FPDEExplanation`

Result object returned by direct explanation functions.

| Field | Description |
| --- | --- |
| `mode` | `"diff"`, `"cos"`, or `"hyb_grid"`. |
| `evidence` | Scalar target-versus-rival contrast. |
| `attributions` | Per-feature contribution vector. |
| `positive_score` | Target-side score. |
| `negative_score` | Rival-side score. |
| `positive_label` | Target class label. |
| `negative_label` | Rival class label. |
| `positive_prototype_index` | Selected target prototype index. |
| `negative_prototype_index` | Selected rival prototype index. |
| `exactness_residual` | Numerical residual between summed and direct evidence. |
| `details` | Method-specific metadata. |

`normalized_attributions` returns an L1-normalized copy for visualization.

### `FPDEContext`

Reusable training-side state with prototypes, prototype labels, mean anchor,
zero anchor, baseline, and feature count.

### `DynamicFPDEContext`

Reusable Dynamic-FPDE state with temporal prototypes, prototype labels,
prototype length, feature count, mean anchor, zero anchor, alignment, and
metadata.

### `DynamicFPDEExplanation`

Result object for one Dynamic-FPDE explanation. The `attributions` field has
shape `(T, F)`, `time_importance` has shape `(T,)`, and
`feature_importance` has shape `(F,)`. For Dynamic-Hyb, `positive_score` and
`negative_score` are deterministic weighted component scores; with
`normalize="l1"`, component scores are divided by each component attribution
L1 scale before mixing. Hyb evidence remains the sum of the mixed attribution
matrix.

### `HybFPDEGridSearchResult`

Contains `best_config`, `best_score`, `rows`, `objective`, `n_candidates`, and
`n_eval_samples`. Use `sorted_rows()` to inspect candidates from best to worst.

### `HybFPDEValidationSelectionResult`

Contains `best_lambda`, `best_config`, `rows`, and `n_eval_samples`. Use
`sorted_rows()` to inspect lambda candidates from best to worst.

### `BayesianFPDELambdaSelectionResult`

Contains `posterior_mean_lambda`, `map_lambda`, `credible_interval`,
`posterior_rows`, `prior_alpha`, `prior_beta`, `temperature`, `normalize`,
`anchor_strategy`, `eps`, and `n_eval_samples`.

The `credible_interval` field is an interval over lambda candidates, not a
feature-attribution interval.

Use `sorted_rows()` to inspect lambda candidates from highest to lowest
posterior probability.

## Common Errors

| Error | Cause | Fix |
| --- | --- | --- |
| `model must implement predict_proba` | A model-dependent operation was called without probability support. | Pass a fitted classifier with `predict_proba`. |
| `model must expose classes_` | The model does not expose class labels. | Use a scikit-learn-compatible classifier or add compatible `classes_` metadata. |
| `feature dimension mismatch` | Input vectors do not match the fitted training feature count. | Apply the same preprocessing pipeline to training, validation, and explanation data. |
| `lambda_hyb must be in [0, 1]` | The Hyb-FPDE mixture weight is outside the valid range. | Pass a finite value from 0.0 to 1.0. |
| `eps must be positive` | Cosine regularization was zero or negative. | Use a positive `eps`, such as `1e-12`. |
