# Dynamic-FPDE

Dynamic-FPDE extends FPDE from fixed feature vectors to variable-length
frame-level time-series feature matrices.

Use Dynamic-FPDE when each sample is a matrix:

```text
X shape = (T, F)
```

where `T` is the number of time frames and `F` is the number of frame-level
features. The returned attribution matrix has the same shape:

```text
Phi shape = (T, F)
```

`Phi[t, f]` is prototype evidence for frame `t` and feature `f`. Positive
values support the target class prototype relative to the rival prototype.
Negative values support the rival class prototype relative to the target
prototype. The sign always depends on the chosen target/rival pair.

## Dynamic-Diff

Dynamic-Diff decomposes the squared-distance contrast elementwise:

```text
Phi_diff[t, f] = (X[t, f] - P_rival[t, f])**2
               - (X[t, f] - P_target[t, f])**2

E_diff = sum(Phi_diff)
```

A positive element is closer to the target prototype than to the rival
prototype under this squared-distance contrast.

## Dynamic-Cos

Dynamic-Cos decomposes a cosine-similarity contrast after subtracting an
anchor:

```text
Z        = X - anchor
Q_target = P_target - anchor
Q_rival  = P_rival - anchor

Phi_cos[t, f] = Z[t, f] * Q_target[t, f] / (norm(Z) * norm(Q_target))
              - Z[t, f] * Q_rival[t, f] / (norm(Z) * norm(Q_rival))

E_cos = sum(Phi_cos)
```

This is a coordinate decomposition of the cosine contrast. It is not a causal
removal effect.

## Dynamic-Hyb

Dynamic-Hyb mixes Dynamic-Diff and Dynamic-Cos attribution matrices:

```text
Phi_hyb = lambda_hyb * Phi_diff_norm + (1 - lambda_hyb) * Phi_cos_norm
```

`lambda_hyb=1.0` is the Dynamic-Diff endpoint, and `lambda_hyb=0.0` is the
Dynamic-Cos endpoint. With `normalize="l1"`, each component matrix is divided
by `sum(abs(Phi)) + eps` before mixing. With `normalize="none"`, raw component
matrices are mixed directly.

## Basic Usage

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

print(explanation.attributions.shape)
print(explanation.time_importance)
print(explanation.feature_importance)
```

If `rival_label=None`, Dynamic-FPDE chooses the closest non-target prototype
after resampling prototypes to the sample length.

## Limitations

- Dynamic-FPDE is prototype evidence decomposition, not a causal explanation.
- Attribution signs depend on the target/rival prototype pair.
- v0.1 uses linear temporal resampling only.
- Inputs are frame-level feature matrices only.
- Raw waveform direct explanation, DTW alignment, Delta-Dynamic-FPDE,
  AIME/SHAP/LIME comparisons, and recommender-specific logic are out of scope.
