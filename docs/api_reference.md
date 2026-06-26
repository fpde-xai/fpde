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
FPDEEngine.fit(X_train, y_train, model=None, baseline=None, device="cpu")
```

Fits reusable FPDE state from training data.

| Parameter | Description |
| --- | --- |
| `X_train` | Training feature matrix with shape `(n_samples, n_features)`. |
| `y_train` | Training labels with one label per row in `X_train`. |
| `model` | Optional classifier. Required later when an operation needs `predict_proba`. |
| `baseline` | Optional replacement vector for perturbation curves. Defaults to the training mean. |
| `device` | `"cpu"` by default. Use `"cuda"`/`"gpu"`/`"cupy"` to require CuPy on an NVIDIA CUDA device, or `"auto"` to use CuPy when available and otherwise fall back to CPU. |

Returns an `FPDEEngine`.

When `device` resolves to CUDA, FPDE uses CuPy for vectorized attribution
components and validation perturbation-array construction. Install
`fpde[cuda12]` for CUDA 12.x or `fpde[cuda13]` for CUDA 13.x. The `fpde[gpu]`
extra is a convenience alias that currently points to CUDA 13. Model calls
such as scikit-learn `predict_proba` remain on the model backend, so arrays
are converted back to NumPy at that boundary.

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

Dynamic-FPDE explains frame-level acoustic feature matrices with shape
`(T, F)` and returns prototype-evidence attribution matrices with shape
`(T, F)`. It is prototype evidence decomposition, not a causal explanation.
Dynamic-FPDE evidence values and validation curves are prototype evidence
scores, not probabilities.

Native-Time Dynamic-FPDE is the intended formulation for variable-length music
and cover-song analysis. The older context-based API remains available as a
legacy resampled-time / benchmark-oriented variant.

Raw-Waveform Dynamic-FPDE is a separate API for raw-sample explanations. It
uses raw waveform arrays and labels only; it does not extract acoustic
features, spectrograms, or MFCCs, and it does not normalize waveform
amplitudes.

### `DynamicFPDEEngine`

```python
from fpde.dynamic import DynamicFPDEEngine

engine = DynamicFPDEEngine(lambda_hyb=0.5, eps=1e-12, tolerance=1e-9)
engine.fit(raw=raw, y=y, features=features, dt=dt, mask=mask)
result = engine.explain_one(
    raw=raw[0],
    features=features[0],
    method="hyb",
    target_class=0,
    rival_class=1,
)
```

RawFeat Dynamic-FPDE accepts raw sequences with shape `(N, T, C_raw)`, one
sample with shape `(T, C_raw)`, or a list/tuple of `(T_i, C_raw)` arrays.
Optional `features` and `dt` use the same time axis. Version 1 has no deep
encoder. The engine builds:

```text
u_t = concat(raw_t, features_t, dt_t)
```

and stores it as `representation_`. Variable-length lists are padded to
`T_max`; the mask is stored as boolean values and padding positions always
produce zero attribution.

Attribution is also zero when either selected prototype is invalid at a time
step:

```text
valid_t = input_mask_t
          AND prototype_masks_[target_class, t]
          AND prototype_masks_[rival_class, t]
```

If no valid time step remains, the result has zero evidence,
`audit["passed"] == True`, and `audit["warning"] == "no_valid_time"`.

`explain_one` accepts `method="diff"`, `"cos"`, or `"hyb"` and returns a
`DynamicFPDEResult`. The result includes:

- `attributions` with shape `(T_max, C_raw + C_feat + C_dt)`
- `raw_attributions`, `feature_attributions`, and `dt_attributions`
- `time_attributions = attributions.sum(axis=1)`
- `group_attributions` for `raw`, `features`, and `dt`
- `audit`, where `evidence == audit["attribution_sum"]` within `tolerance`
  plus valid-time counts, prototype-invalid counts, and component L1 scales

Target/rival resolution uses explicit classes first, then the top two entries
of a one-dimensional `predict_proba`, then nearest non-target prototypes.

### `pad_sequences`

```python
from fpde.dynamic import pad_sequences

padded, mask = pad_sequences([raw_a, raw_b])
```

Pads a list of `(T_i, C)` arrays to `(N, T_max, C)` and returns a boolean mask
with shape `(N, T_max)`.

### `validate_sequence_inputs`

```python
from fpde.dynamic import validate_sequence_inputs

batch = validate_sequence_inputs(raw, features=features, dt=dt, mask=mask)
```

Normalizes fixed-length arrays and variable-length lists or tuples into padded
raw, feature, `dt`, and boolean mask arrays. Invalid or padded time steps are
zero-filled.

### `PrototypeRawGenerator`

```python
from fpde.dynamic import PrototypeRawGenerator

gen = PrototypeRawGenerator().fit(raw=raw_list, y=y)
generated = gen.generate(label=0, length=100, noise_scale=0.05, random_state=0)
```

Stores label-wise mask-weighted raw prototypes and interpolates them to a
requested length. This is a lightweight baseline interface for future
conditional raw generation models; it is not invoked by `DynamicFPDEEngine`.
`condition_features` is accepted and validated for future compatibility, but
the current baseline generator does not condition on it.

### `select_lambda_dynamic`

```python
from fpde.dynamic import DynamicFPDEEngine, select_lambda_dynamic

engine = DynamicFPDEEngine().fit(raw=train_raw, features=train_features, y=train_y)
selection = select_lambda_dynamic(
    engine=engine,
    raw=val_raw,
    features=val_features,
    predict_proba=predict_proba,
    lambdas=[0.0, 0.5, 1.0],
    baseline="mean",
    steps=20,
)
```

Selects RawFeat Dynamic-Hyb `lambda_hyb` with a coordinate-level validation
perturbation score. For each candidate lambda, valid coordinates are ranked by
signed positive attribution. Deletion replaces top-ranked coordinates with a
baseline; insertion starts from the baseline and restores top-ranked
coordinates. The score is:

```text
0.5 * (deletion_drop_auc + insertion_auc)
```

The return dictionary includes `best_lambda`, `scores`,
`deletion_drop_auc`, `insertion_auc`, `lambdas`, `rows`, `n_validation`,
`steps`, `baseline`, `metric="dynamic_deletion_insertion"`, and
`status="evaluated"`. Ties are broken by the smallest lambda. The engine also
provides `engine.select_lambda(...)`, which delegates to this function.

`predict_proba` may be a callable accepting `raw`, `features`, `dt`, and
`mask` keywords, a callable accepting one concatenated representation array, or
a precomputed probability matrix. Use a callable for meaningful perturbation
curves because precomputed probabilities cannot change under perturbation.

Calling `select_lambda_dynamic(lambda_grid=[...])` without an engine remains a
backward-compatible placeholder mode for lambda validation only.

### `split_representation`

```python
from fpde.dynamic import split_representation

raw, features, dt = split_representation(representation, feature_slices)
```

Splits a concatenated RawFeat representation back into raw, optional feature,
and optional `dt` arrays using a `DynamicFPDEResult.feature_slices` or fitted
engine `feature_slices_` mapping.

### `native_dynamic_diff_fpde`

```python
native_dynamic_diff_fpde(X, p_target, p_rival)
```

Accepts `X` with shape `(T, F)` and feature-vector prototypes `p_target` and
`p_rival` with shape `(F,)`. Computes:

```text
Phi_diff[t, f] = (X[t, f] - p_rival[f])**2
               - (X[t, f] - p_target[f])**2
```

Returns `(Phi_diff, E_diff)`, where `Phi_diff.shape == X.shape`.

### `native_dynamic_cos_fpde`

```python
native_dynamic_cos_fpde(X, p_target, p_rival, *, anchor=None, eps=1e-12)
```

Computes a native-time coordinate decomposition of the cosine contrast. The
input norm is computed per frame, prototype norms are feature-vector norms,
and no temporal resampling or pooling is performed.

### `native_dynamic_hyb_fpde`

```python
native_dynamic_hyb_fpde(
    X,
    p_target,
    p_rival,
    *,
    lambda_hyb=0.5,
    normalize="none",
    anchor=None,
    eps=1e-12,
)
```

Mixes Native-Time Dynamic-Diff and Dynamic-Cos attribution matrices.
`lambda_hyb=1.0` is the Diff endpoint, and `lambda_hyb=0.0` is the Cos
endpoint. `normalize="none"` is the default. `normalize="l1"` normalizes each
full attribution matrix component and does not change the time axis.

Returns `(Phi_hyb, E_hyb, details)`.

### `native_dynamic_fpde_explain_one`

```python
native_dynamic_fpde_explain_one(
    X,
    *,
    p_target,
    p_rival,
    target_label=None,
    rival_label=None,
    mode="dynamic_hyb",
    lambda_hyb=0.5,
    normalize="none",
    anchor=None,
    feature_names=None,
    timestamps_sec=None,
    eps=1e-12,
    details=None,
)
```

Explains one native-time sequence with feature-vector prototypes. The returned
`NativeTimeDynamicFPDEExplanation` preserves `X.shape` exactly. Its `details`
include `time_mode="native"`, `temporal_resampling=False`,
`temporal_pooling=False`, `prototype_kind="feature_vector"`, `input_shape`,
and `output_shape`.

### `native_dynamic_fpde_explain_batch`

```python
native_dynamic_fpde_explain_batch(
    X_list,
    *,
    p_targets,
    p_rivals,
    target_labels=None,
    rival_labels=None,
    mode="dynamic_hyb",
    lambda_hyb=0.5,
    normalize="none",
    anchor=None,
    feature_names=None,
    timestamps_list=None,
    eps=1e-12,
)
```

Explains a list of variable-length `(T_i, F)` arrays without padding,
resampling, or dense tensor conversion. `p_targets` and `p_rivals` may be a
single `(F,)` vector broadcast to all samples or one vector per sample.

### Raw-Waveform Dynamic-FPDE API

```python
prepare_raw_waveform_fpde_context(
    waveforms,
    labels,
    *,
    sample_rates,
    target_sr=16000,
    segment_sec=0.5,
    hop_sec=0.1,
)
```

Builds raw segment banks per label. Inputs may be mono 1D arrays or stereo 2D
arrays. Stereo arrays are downmixed to mono. Each waveform is resampled to
`target_sr`, but no fixed duration, peak normalization, RMS normalization, or
loudness normalization is applied.

The function uses sliding windows with `segment_length=int(round(segment_sec *
target_sr))` and `hop_length=int(round(hop_sec * target_sr))`. Waveforms shorter
than one segment are zero padded and tracked with a mask. Padding is excluded
from prototype distances, evidence, aggregation, and exported segments. Longer
waveforms keep full windows only, with an end-aligned final window when needed
to cover the original sample axis.

Returns a `RawWaveformFPDEContext` with label-specific segment banks and
label-medoid raw prototypes.

```python
raw_waveform_fpde_explain_one(
    waveform,
    context,
    *,
    sample_rate,
    target_label,
    rival_label=None,
    lambda_grid=None,
    top_k_segments=1,
    generator=None,
    eps=1e-12,
    device="cpu",
    details=None,
)
```

Explains one raw waveform. If `rival_label` is omitted, the closest non-target
label medoid is used. The default `lambda_grid` is `[i / 10 for i in
range(11)]`.

Pass `device="cuda"` after installing `fpde[cuda13]` to run Raw-Diff,
Raw-Cos, and Raw-Hyb window-evidence computation with CuPy on CUDA 13. Pass
`device="auto"` to use CUDA when available and otherwise fall back to CPU.
Waveform validation, sample-rate conversion, sliding-window construction, and
artifact export remain CPU-side.

For every lambda, the returned `RawWaveformFPDEExplanation.lambda_results`
contains:

- `phi`: sample-level Raw-Hyb attribution vector with the same shape as the
  resampled waveform.
- `window_evidence`: one scalar evidence value per sliding window.
- `top_positive_segments` and `top_negative_segments`: original waveform
  segments that most strongly support the target or rival label.
- `generated_target` and `generated_rival`: optional generator-hook outputs.
- `generation_status`: `"ok"` or `"skipped"` per generated role.

The optional generator hook has this signature:

```python
generator(label, lambda_hyb, segment, sample_rate, role, metadata)
```

It is called only after Raw-Hyb evidence has been computed and top segments
have been selected. FPDE does not include a built-in label-conditioned raw
audio generator.

```python
raw_diff_fpde(window, p_target, p_rival, *, mask=None, target_mask=None, rival_mask=None)
raw_cos_fpde(window, p_target, p_rival, *, mask=None, target_mask=None, rival_mask=None, eps=1e-12)
raw_hyb_fpde(window, p_target, p_rival, *, lambda_hyb=0.5, mask=None, target_mask=None, rival_mask=None, eps=1e-12)
```

These tensor-level helpers compute Raw-Diff, Raw-Cos, and valid-mask L1-scaled
Raw-Hyb evidence for one raw window and one target/rival prototype pair.

```python
save_raw_waveform_fpde_results(explanation, output_dir, *, save_plots=True)
```

Writes lambda-wise directories such as `raw_hyb_lambda_0.0`, window evidence
CSVs, top segment WAV files, optional generated WAV files, `metrics.json`, and
root-level `summary.csv`. WAV writing requires:

```bash
python -m pip install "fpde[audio]"
```

### Legacy Resampled-Time API

The following names remain available for existing code and benchmark-oriented
fixed-length comparisons:

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

Builds one class-mean temporal prototype per label from variable-length
`(T_i, F)` arrays. Each training sequence is linearly resampled to
`prototype_length`. Returns a `DynamicFPDEContext` with prototypes shaped
`(n_classes, prototype_length, n_features)`.

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

Explains one sequence with the legacy resampled-time context. If
`rival_label` is omitted, the closest non-target prototype is selected after
resampling temporal prototypes to the sample length.

```python
dynamic_fpde_explain_batch(...)
resample_time_series_linear(X, target_length)
dynamic_diff_fpde(X, P_target, P_rival)
dynamic_cos_fpde(X, P_target, P_rival, *, anchor=None, eps=1e-12)
dynamic_hyb_fpde(X, P_target, P_rival, *, lambda_hyb=0.5, normalize="l1", anchor=None, eps=1e-12)
```

These legacy tensor-level functions use temporal prototype tensors with shape
`(T, F)` and remain backward-compatible.

### Dynamic-FPDE CUDA helpers

```python
from fpde.dynamic_cuda import (
    dynamic_diff_fpde_gpu,
    dynamic_cos_fpde_gpu,
    dynamic_hyb_fpde_gpu,
)
```

These optional CuPy helpers accept either one already-resampled tensor with
shape `(T, F)` or a batch with shape `(N, T, F)`. CUDA acceleration is
intended for batched, already-resampled Dynamic-FPDE tensor operations.

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
deletion and insertion curves using the legacy resampled-time context.

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

Evaluates legacy Dynamic-Hyb lambda candidates with temporal
deletion/insertion metrics from normalized prototype-evidence curves.

### Dynamic Plotting Helpers

```python
plot_dynamic_time_importance(explanation, *, ax=None, title=None)
plot_dynamic_attribution_heatmap(explanation, *, ax=None, title=None)
```

These helpers accept either `DynamicFPDEExplanation` or
`NativeTimeDynamicFPDEExplanation`.

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

Reusable state for the legacy resampled-time Dynamic-FPDE variant, with
temporal prototypes, prototype labels, prototype length, feature count, mean
anchor, zero anchor, alignment, and metadata.

### `DynamicFPDEResult`

Result object returned by `DynamicFPDEEngine.explain_one` and
`DynamicFPDEEngine.explain_batch`. It stores the selected `method`,
`target_class`, `rival_class`, scalar `evidence`, full concatenated
`attributions`, group-specific raw/feature/`dt` attribution matrices,
`time_attributions`, `group_attributions`, the boolean `mask`,
`feature_slices`, and an `audit` dictionary for the attribution-sum identity,
valid-time accounting, prototype-invalid accounting, and Diff/Cos L1 scales.

### `DynamicFPDEExplanation`

Result object for one legacy resampled-time Dynamic-FPDE explanation. The
`attributions` field has shape `(T, F)`, `time_importance` has shape `(T,)`,
and `feature_importance` has shape `(F,)`. For Dynamic-Hyb, `positive_score`
and `negative_score` are deterministic weighted component scores; with
`normalize="l1"`, component scores are divided by each component attribution
L1 scale before mixing. Hyb evidence remains the sum of the mixed attribution
matrix.

### `NativeTimeDynamicFPDEExplanation`

Result object for one Native-Time Dynamic-FPDE explanation. The `attributions`
field has exactly the same shape as the input feature matrix, `time_importance`
has shape `(T,)`, and `feature_importance` has shape `(F,)`. The object and
its `details` metadata record `time_mode="native"`,
`temporal_resampling=False`, and `temporal_pooling=False`.

### `RawWaveformFPDEContext`

Reusable state for Raw-Waveform Dynamic-FPDE. It contains label-specific raw
segment banks, masks, label-medoid raw prototypes, `target_sr`,
`segment_length`, `hop_length`, and metadata recording that acoustic feature
extraction and waveform normalization are not used.

### `RawWaveformFPDEExplanation`

Result object for one Raw-Waveform Dynamic-FPDE explanation. It contains the
resampled raw waveform, target/rival labels, lambda-wise results, optional
generator outputs, and metadata recording `time_mode="raw_waveform"`,
`temporal_resampling=False`, and `waveform_normalization=False`. Each
lambda-wise `phi` vector has the same shape as the explanation waveform.

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
