# Feature Prototype Direction Explainer (FPDE)

[![PyPI version](https://img.shields.io/pypi/v/fpde.svg)](https://pypi.org/project/fpde/)
[![Python versions](https://img.shields.io/pypi/pyversions/fpde.svg)](https://pypi.org/project/fpde/)
[![License](https://img.shields.io/badge/License-MIT%20OR%20Apache--2.0-blue.svg)](https://github.com/fpde-xai/fpde/blob/main/LICENSE)

Feature Prototype Direction Explainer (FPDE) is a Python package for
prototype-contrast feature attribution. It explains a classification result by
comparing an input with a prototype for the target class and a prototype for a
rival class, then decomposing that contrast into per-feature contributions.

Use FPDE when you want a lightweight, post-hoc explanation method for
tabular feature vectors and black-box classifiers that expose class
probabilities.

## What You Can Do

- Build class-mean prototypes from training data.
- Explain one sample with Diff-FPDE, Cos-FPDE, or a fixed Hyb-FPDE mixture.
- Explain batches while reusing fitted prototype state.
- Search Diff, Cos, and Hyb-FPDE candidate settings.
- Select `lambda_hyb` with held-out deletion and insertion validation.
- Build an experimental Bayesian posterior over Hyb-FPDE `lambda_hyb`
  candidates.
- Explain variable-length frame-level feature matrices with Dynamic-FPDE.
- Compute deletion and insertion perturbation curves for an attribution vector.
- Plot attribution bars, cumulative waterfalls and contribution summaries, attribution
  heatmaps, FPDE-native prototype similarity distributions, and perturbation
  curves.

## Install FPDE

FPDE requires Python 3.12 or newer.

```bash
python -m pip install fpde
```

Plotting helpers use matplotlib as an optional dependency:

```bash
python -m pip install "fpde[plot]"
```

For local development, clone the repository and install it in editable mode:

```bash
python -m pip install -e .
```

The PyPI distribution name and Python import package are both `fpde`.

## Quick Start

This example trains a scikit-learn classifier, fits an FPDE engine on the
training data, and explains one test sample.

```python
import numpy as np
from sklearn.datasets import load_breast_cancer
from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler

from fpde import FPDEEngine

data = load_breast_cancer()
X_train, X_test, y_train, _ = train_test_split(
    data.data,
    data.target,
    test_size=0.25,
    random_state=7,
    stratify=data.target,
)

scaler = StandardScaler()
X_train = scaler.fit_transform(X_train)
X_test = scaler.transform(X_test)

model = LogisticRegression(max_iter=2000, random_state=7)
model.fit(X_train, y_train)

engine = FPDEEngine.fit(X_train, y_train, model=model)
attributions, details = engine.explain_one(X_test[0], lambda_hyb=0.5)

print(np.asarray(attributions))
print(details["target_label"], details["rival_label"], details["evidence"])
```

Positive attribution values support the target class relative to the rival
class. Negative values support the rival class relative to the target class.

### Optional NVIDIA GPU acceleration

FPDE can use CuPy for vectorized attribution and validation tensor operations.
Install the CuPy wheel that matches your CUDA stack:

```bash
pip install "fpde[cuda12]"  # CUDA 12.x
pip install "fpde[cuda13]"  # CUDA 13.x
```

The `fpde[gpu]` extra is kept as a convenience alias and currently points to
the CUDA 13 wheel.

For fixed-length tabular FPDE, request CUDA on the engine:

```python
engine = FPDEEngine.fit(X_train, y_train, model=model, device="cuda")
attributions, details = engine.explain_batch(X_test[:128], lambda_hyb=0.5)
```

Use `device="auto"` to use CuPy when a CUDA device is available and otherwise
fall back to CPU. Model calls such as scikit-learn `predict_proba` still run
through the model's own backend; FPDE converts arrays back to NumPy at that
boundary.

Dynamic-FPDE also has explicit CUDA helpers for already-resampled tensors:

```python
from fpde.dynamic_cuda import dynamic_hyb_fpde_gpu

attr, evidence, details = dynamic_hyb_fpde_gpu(
    X_resampled_batch,        # shape (N, T, F), or one sample (T, F)
    target_proto_resampled,   # shape (N, T, F) or broadcastable (T, F)
    rival_proto_resampled,
    lambda_hyb=0.5,
)
```

Feature extraction and temporal resampling remain CPU-side. CUDA acceleration
is intended for batched, already-resampled Dynamic-FPDE tensor operations.

To visualize an explanation, install the optional plotting extra and pass
feature names when available:

```python
from fpde import plot_attribution_waterfall, plot_attributions

plot_attributions(
    attributions,
    feature_names=data.feature_names,
    top_k=10,
    title="Top FPDE feature contributions",
)

plot_attribution_waterfall(
    attributions,
    feature_names=data.feature_names,
    title="Cumulative FPDE evidence",
)
```

For Bayesian-FPDE plots, `plot_attributions` also accepts `interval_low` and
`interval_high` so you can show the attribution range induced by the
`lambda_hyb` credible interval.

For a local contribution view, use `plot_local_contributions`. This displays FPDE local
attributions as a signed feature bar chart:

```python
from fpde import plot_local_contributions

ax = plot_local_contributions(
    data.feature_names,
    attributions,
    values=X_test[0],
    top_k=10,
    title="FPDE local explanation",
)
ax.figure.savefig("fpde_local_contributions.png", dpi=160, bbox_inches="tight")
```

For a compact plotting namespace for FPDE contributions, use `fpde.plots`.
These helpers visualize FPDE attribution arrays directly:

```python
from fpde.plots import FPDEPlotExplanation, bar, beeswarm, scatter, waterfall

batch_attributions, _ = engine.explain_batch(X_test[:20], lambda_hyb=0.5)
plot_exp = FPDEPlotExplanation(
    values=batch_attributions,
    data=X_test[:20],
    feature_names=data.feature_names,
)

ax = beeswarm(plot_exp, show=False, title="FPDE contribution summary")
ax.figure.savefig("fpde_beeswarm.png", dpi=150, bbox_inches="tight")

bar(plot_exp, show=False)
scatter(plot_exp, feature=data.feature_names[0], show=False)
waterfall(
    values=attributions,
    base_value=0.0,
    prediction=float(np.sum(attributions)),
    feature_names=data.feature_names,
    show=False,
)
```

You can also compare how training samples and the explained sample relate to
target and rival prototypes:

```python
from fpde import plot_prototype_similarity_distribution

target_idx = np.where(engine.prototype_labels == details["target_label"])[0][0]
rival_idx = np.where(engine.prototype_labels == details["rival_label"])[0][0]

plot_prototype_similarity_distribution(
    X_train,
    engine.prototypes[target_idx],
    rival_prototype=engine.prototypes[rival_idx],
    x=X_test[0],
    metric="cosine",
    title="FPDE prototype similarity distribution",
)
```

To select a Hyb-FPDE mixture weight with Bayesian-FPDE, use held-out samples to
build a posterior over lambda candidates, then explain with the posterior mean:

```python
selection = engine.select_bayesian_lambda(
    X_test[:16],
    lambda_hyb_grid=(0.0, 0.25, 0.5, 0.75, 1.0),
)
attributions, details = engine.explain_one_bayesian(X_test[0], selection)

print(selection.posterior_mean_lambda, selection.map_lambda)
print(selection.credible_interval)
```

In v0.1.0, Bayesian-FPDE uncertainty is limited to the finite grid of
`lambda_hyb` candidates. It does not sample class prototypes, estimate
feature-level prototype uncertainty, or model black-box classifier uncertainty.

## Dynamic-FPDE

Dynamic-FPDE explains variable-length frame-level feature matrices with shape
`(T, F)`, where `T` is the number of time frames and `F` is the number of
frame-level features. It returns an attribution matrix with the same shape.
Positive values support the target prototype over the rival prototype, and
negative values support the rival prototype over the target prototype.

```python
from fpde import prepare_dynamic_fpde_context, dynamic_fpde_explain_one

context = prepare_dynamic_fpde_context(X_train_sequences, y_train, prototype_length=128)
explanation = dynamic_fpde_explain_one(
    X_sample,
    context,
    target_label=target_label,
    mode="dynamic_hyb",
    lambda_hyb=0.5,
)

print(explanation.attributions.shape)
print(explanation.time_importance.shape)
```

Dynamic-FPDE is prototype evidence decomposition, not a causal explanation. In
v0.1, it supports linear temporal resampling for frame-level features only.

## Run The Example

```bash
python examples/minimal_fpde_example.py
```

The script prints the predicted class and the largest positive and negative
feature contributions for one sample from the breast cancer dataset bundled
with scikit-learn.

For the experimental Bayesian-FPDE lambda posterior workflow, see
`examples/bayesian_fpde_example.ipynb`.

## Documentation

- [Method overview](https://github.com/fpde-xai/fpde/blob/main/docs/method_overview.md): core FPDE concepts and variants.
- [Dynamic-FPDE](https://github.com/fpde-xai/fpde/blob/main/docs/dynamic_fpde.md): frame-level time-series feature attribution.
- [API reference](https://github.com/fpde-xai/fpde/blob/main/docs/api_reference.md): public functions, classes, parameters,
  and result objects.
- [Reproducibility checklist](https://github.com/fpde-xai/fpde/blob/main/docs/reproducibility_checklist.md): what to record
  when reporting FPDE experiments.
- [Data and code availability](https://github.com/fpde-xai/fpde/blob/main/docs/data_and_code_availability_statement.md):
  repository and dataset availability statement.
- [Repository metadata](https://github.com/fpde-xai/fpde/blob/main/docs/repository_metadata.md): project URL, description,
  topics, and important paths.
- [Release notes](https://github.com/fpde-xai/fpde/blob/main/RELEASE_NOTES.md): package history.

## Test The Package

Install the development dependencies, then run the test suite:

```bash
python -m pip install -e .
python -m pytest
```

You can also run the repository in Docker:

```bash
docker build -t fpde .
docker run --rm fpde
```

The Docker command runs the test suite and the minimal example.

## Cite FPDE

If you use FPDE in academic work, cite the software or method as appropriate.
Citation metadata is available in [CITATION.cff](https://github.com/fpde-xai/fpde/blob/main/CITATION.cff).

## License

FPDE is distributed under a dual license: MIT OR Apache-2.0. You may choose either license. See [LICENSE](https://github.com/fpde-xai/fpde/blob/main/LICENSE), [LICENSE-MIT](https://github.com/fpde-xai/fpde/blob/main/LICENSE-MIT), and [LICENSE-APACHE](https://github.com/fpde-xai/fpde/blob/main/LICENSE-APACHE).
