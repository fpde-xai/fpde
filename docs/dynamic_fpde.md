# Dynamic-FPDE

Dynamic-FPDE extends FPDE from fixed feature vectors to frame-level
time-series feature matrices.

Native-Time Dynamic-FPDE is the intended Dynamic-FPDE formulation for
variable-length music and cover-song analysis. It preserves each input clip's
native frame-level time axis and computes a time-feature prototype-evidence
matrix without temporal pooling, fixed-length resampling, or temporal
aggregation.

Use Native-Time Dynamic-FPDE when each sample is a frame-level feature matrix:

```text
X shape = (T, F)
```

where `T` is the number of native frames and `F` is the number of frame-level
acoustic features. The returned attribution matrix has exactly the same shape:

```text
Phi shape = (T, F)
```

`Phi[t, f]` is prototype evidence for frame `t` and feature `f`. Positive
values support the target prototype relative to the rival prototype. Negative
values support the rival prototype relative to the target prototype. Evidence
scores are prototype-contrast scores, not probabilities.

## Native-Time Dynamic-Diff

Native-Time Dynamic-Diff uses feature-vector prototypes:

```text
X_i      in R^(T_i x F)
p_target in R^F
p_rival  in R^F
```

It decomposes the squared-distance contrast at every native frame:

```text
Phi_diff[t, f] = (X[t, f] - p_rival[f])**2
               - (X[t, f] - p_target[f])**2

E_diff = sum(Phi_diff)
```

A positive element is closer to the target prototype than to the rival
prototype under this squared-distance contrast.

## Native-Time Dynamic-Cos

Native-Time Dynamic-Cos decomposes a cosine-similarity contrast after
subtracting a feature-vector anchor:

```text
z[t]     = X[t, :] - anchor
q_target = p_target - anchor
q_rival  = p_rival  - anchor

Phi_cos[t, f] =
    z[t, f] * q_target[f] / ((||z[t]||_2 + eps) * (||q_target||_2 + eps))
  - z[t, f] * q_rival[f]  / ((||z[t]||_2 + eps) * (||q_rival||_2 + eps))
```

The input norm `||z[t]||_2` is computed per frame. Prototype norms are
feature-vector norms. No time-axis resampling, pooling, or aggregation is
performed.

## Native-Time Dynamic-Hyb

Native-Time Dynamic-Hyb mixes Native-Time Dynamic-Diff and Dynamic-Cos
attribution matrices:

```text
Phi_hyb = lambda_hyb * Phi_diff + (1 - lambda_hyb) * Phi_cos
```

`lambda_hyb=1.0` is the Dynamic-Diff endpoint, and `lambda_hyb=0.0` is the
Dynamic-Cos endpoint. The default `normalize="none"` mixes raw component
matrices. If `normalize="l1"` is used, it normalizes the full attribution
matrix for each component only; it does not resample, pool, or otherwise
change the time axis.

## Native-Time Usage

```python
from fpde import native_dynamic_fpde_explain_one

explanation = native_dynamic_fpde_explain_one(
    X_sample,              # shape (T, F)
    p_target=target_proto, # shape (F,)
    p_rival=rival_proto,   # shape (F,)
    target_label="target",
    rival_label="rival",
    mode="dynamic_hyb",
    lambda_hyb=0.5,
)

print(explanation.attributions.shape)      # exactly X_sample.shape
print(explanation.time_importance.shape)   # (T,)
print(explanation.feature_importance.shape) # (F,)
print(explanation.details["time_mode"])    # "native"
```

For variable-length batches, use `native_dynamic_fpde_explain_batch`. It keeps
the input as a list of 2D arrays and does not pad or resample:

```python
from fpde import native_dynamic_fpde_explain_batch

explanations = native_dynamic_fpde_explain_batch(
    [X1, X2, X3],
    p_targets=target_proto,
    p_rivals=rival_proto,
)
```

If `X1.shape == (100, F)`, `X2.shape == (777, F)`, and
`X3.shape == (2400, F)`, the returned attribution shapes are `(100, F)`,
`(777, F)`, and `(2400, F)`.

## Raw-Waveform Dynamic-FPDE

Raw-Waveform Dynamic-FPDE is a separate raw-sample API for experiments that
must use only a waveform and its label. It does not extract acoustic features,
spectrograms, or MFCCs, and it does not apply peak, RMS, loudness, or other
waveform normalization. Stereo inputs are downmixed to mono, and sample rates
are converted to `target_sr`; clip durations remain variable.

```python
from fpde import prepare_raw_waveform_fpde_context, raw_waveform_fpde_explain_one

context = prepare_raw_waveform_fpde_context(
    train_waveforms,
    train_labels,
    sample_rates=train_sample_rates,
    target_sr=16000,
    segment_sec=0.5,
    hop_sec=0.1,
)

explanation = raw_waveform_fpde_explain_one(
    waveform,
    context,
    sample_rate=source_sample_rate,
    target_label=label,
    device="cuda",
)
```

The context builds a raw segment bank per label with sliding windows. If a
waveform is shorter than one segment, it is zero padded and tracked with a
mask; padding is excluded from distance, evidence, aggregation, and exported
segments. For longer waveforms, an end-aligned final window is added when
needed so overlap-add attribution covers the original sample axis.

Raw prototypes are label medoids from the segment banks. If `rival_label` is
omitted, the closest non-target medoid label is used for the explanation.

For each lambda, the API computes Raw-Diff and Raw-Cos window evidence, scales
each component by its valid-mask L1 scale, and mixes them:

```text
Phi_hyb(lambda) = lambda * scale(Phi_diff)
                + (1 - lambda) * scale(Phi_cos)
```

The default lambda grid is:

```text
0.0, 0.1, 0.2, 0.3, 0.4, 0.5,
0.6, 0.7, 0.8, 0.9, 1.0
```

Each lambda result includes a sample-level `phi` vector produced by
overlap-add averaging, and every `phi.shape` matches the resampled raw
waveform shape exactly.

Install `fpde[cuda13]` and pass `device="cuda"` to run the Raw-Diff,
Raw-Cos, and Raw-Hyb window-evidence computation with CuPy on CUDA 13. Use
`device="auto"` to use CUDA when available and otherwise fall back to CPU.
Waveform validation, mono conversion, resampling, window creation, and artifact
export remain CPU-side.

Label-conditioned RAW generation is intentionally a post-evidence verification
hook rather than a built-in model:

```python
def generator(label, lambda_hyb, segment, sample_rate, role, metadata):
    return generated_waveform

explanation = raw_waveform_fpde_explain_one(
    waveform,
    context,
    sample_rate=source_sample_rate,
    target_label=label,
    generator=generator,
)
```

The hook is called after top positive and negative segments have been selected.
Without a hook, generation is recorded as `"skipped"`.

Use `save_raw_waveform_fpde_results` to write lambda-wise result directories
with window evidence, top segment WAV files, optional generated WAV files,
metrics, summary CSV, and plots. WAV export requires the optional audio extra:

```bash
python -m pip install "fpde[audio]"
```

## Legacy Resampled-Time Variant

The existing resampled-time formulation can be useful for controlled
fixed-length benchmark comparisons, but it may smooth, stretch, or distort
short transient events such as drum attacks. Therefore, it should be treated
as a legacy or benchmark-oriented variant rather than the primary formulation
for music-level Dynamic-FPDE explanations.

The legacy API remains available:

```python
from fpde import prepare_dynamic_fpde_context, dynamic_fpde_explain_one

context = prepare_dynamic_fpde_context(
    X_train_sequences,
    y_train,
    prototype_length=128,
)

explanation = dynamic_fpde_explain_one(
    X_sample,
    context,
    target_label="target",
    rival_label=None,
    mode="dynamic_hyb",
    lambda_hyb=0.5,
)
```

`prepare_dynamic_fpde_context` builds class-mean temporal prototypes after
linearly resampling each training sequence to `prototype_length`.
`dynamic_fpde_explain_one` then resamples those stored temporal prototypes to
the input length.

## Optional CUDA Tensor Operations

Dynamic-FPDE provides optional CuPy helpers for already-resampled tensors:

```python
from fpde.dynamic_cuda import dynamic_hyb_fpde_gpu

attr, evidence, details = dynamic_hyb_fpde_gpu(
    X_resampled_batch,        # shape (N, T, F), or one sample (T, F)
    target_proto_resampled,   # shape (N, T, F) or broadcastable (T, F)
    rival_proto_resampled,
    lambda_hyb=0.5,
)
```

For batched input, CUDA helpers return attributions with shape `(N, T, F)` and
evidence with shape `(N,)`. Pass `return_numpy=False` to keep returned arrays
on the GPU as CuPy arrays.

Feature extraction and temporal resampling remain CPU-side. CUDA acceleration
is intended for batched, already-resampled Dynamic-FPDE tensor operations and
does not apply to Raw-Waveform Dynamic-FPDE.

## Temporal Deletion And Insertion

`temporal_deletion_insertion_curves` evaluates frame rankings with
prototype-evidence curves for the legacy resampled-time context. It does not
evaluate class probabilities. Frames are ranked with `rank_by`:

- `"positive"` ranks by `max(time_importance, 0)`, descending. This is the
  default because the metric is intended to evaluate target-supporting frames.
- `"signed"` ranks by signed `time_importance`, descending.
- `"absolute"` ranks by `abs(time_importance)`, descending.

The function computes raw Dynamic-Diff evidence curves, then normalizes them
against the original-to-baseline evidence range:

```text
scale = abs(original_evidence - baseline_evidence) + eps

deletion_drop_curve = (original_evidence - deletion_curve) / scale
insertion_gain_curve = (insertion_curve - insertion_curve[0]) / scale
```

The reported `deletion_drop_auc`, `insertion_gain_auc`, and `combined_score`
come from these normalized curves. The returned `insertion_auc` key is
retained as an alias for `insertion_gain_auc`; prefer `insertion_gain_auc` in
new code.

## Limitations

- Native-Time Dynamic-FPDE explains frame-level acoustic feature matrices.
- Raw-Waveform Dynamic-FPDE is the raw-sample variant and does not extract
  acoustic features, spectrograms, or MFCCs.
- Native-Time Dynamic-FPDE does not perform verse/chorus alignment.
- Native-Time Dynamic-FPDE does not perform DTW.
- Dynamic-FPDE is prototype evidence decomposition, not a causal explanation.
- Dynamic-FPDE does not claim sampling-rate invariance.
- Sampling-rate differences should be controlled before feature extraction by
  converting audio to a common sample rate. Raw-Waveform Dynamic-FPDE converts
  input waveforms to `target_sr` before sliding-window evidence.
- Attribution signs depend on the target/rival prototype pair.
- CUDA helpers currently target already-resampled tensor operations, not
  Native-Time feature-vector prototypes or Raw-Waveform Dynamic-FPDE.
