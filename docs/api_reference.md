# API Reference

Import the public package as `fpde`.

```python
from fpde import FPDEEngine, class_mean_prototypes, diff_fpde
```

This page documents the public API exported by `fpde` and `fpde.core`.

## Engine API

Use `FPDEEngine` for repeated explanations, batch explanations, Hyb-FPDE, grid
search, and validation-based lambda selection.

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
- `"hyb_grid"`: Diff-style prototype selection.

Returns `(positive_index, negative_index)`.

### `prepare_fpde_context`

```python
prepare_fpde_context(X_train, y_train, *, baseline=None)
```

Precomputes reusable prototypes, anchors, baseline, and feature metadata.

Returns an `FPDEContext`.

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

### `HybFPDEGridSearchResult`

Contains `best_config`, `best_score`, `rows`, `objective`, `n_candidates`, and
`n_eval_samples`. Use `sorted_rows()` to inspect candidates from best to worst.

### `HybFPDEValidationSelectionResult`

Contains `best_lambda`, `best_config`, `rows`, and `n_eval_samples`. Use
`sorted_rows()` to inspect lambda candidates from best to worst.

## Common Errors

| Error | Cause | Fix |
| --- | --- | --- |
| `model must implement predict_proba` | A model-dependent operation was called without probability support. | Pass a fitted classifier with `predict_proba`. |
| `model must expose classes_` | The model does not expose class labels. | Use a scikit-learn-style classifier or add compatible `classes_` metadata. |
| `feature dimension mismatch` | Input vectors do not match the fitted training feature count. | Apply the same preprocessing pipeline to training, validation, and explanation data. |
| `lambda_hyb must be in [0, 1]` | The Hyb-FPDE mixture weight is outside the valid range. | Pass a finite value from 0.0 to 1.0. |
| `eps must be positive` | Cosine regularization was zero or negative. | Use a positive `eps`, such as `1e-12`. |
