# Feature Prototype Direction Explainer (FPDE)

[![PyPI version](https://img.shields.io/pypi/v/fpde.svg)](https://pypi.org/project/fpde/)
[![Python versions](https://img.shields.io/pypi/pyversions/fpde.svg)](https://pypi.org/project/fpde/)
[![License](https://img.shields.io/badge/License-Apache_2.0-blue.svg)](LICENSE)

Feature Prototype Direction Explainer (FPDE) is a Python package for
prototype-contrast feature attribution. It explains a classification result by
comparing an input with a prototype for the target class and a prototype for a
rival class, then decomposing that contrast into per-feature contributions.

Use FPDE when you want a lightweight, post-hoc explanation method for
tabular-style feature vectors and black-box classifiers that expose class
probabilities.

## What You Can Do

- Build class-mean prototypes from training data.
- Explain one sample with Diff-FPDE, Cos-FPDE, or a fixed Hyb-FPDE mixture.
- Explain batches while reusing fitted prototype state.
- Search Diff, Cos, and Hyb-FPDE candidate settings.
- Select `lambda_hyb` with held-out deletion and insertion validation.
- Compute deletion and insertion perturbation curves for an attribution vector.

## Install FPDE

FPDE requires Python 3.12 or newer.

```bash
python -m pip install fpde
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

## Run The Example

```bash
python examples/minimal_fpde_example.py
```

The script prints the predicted class and the largest positive and negative
feature contributions for one sample from the breast cancer dataset bundled
with scikit-learn.

## Documentation

- [Method overview](docs/method_overview.md): core FPDE concepts and variants.
- [API reference](docs/api_reference.md): public functions, classes, parameters,
  and result objects.
- [Reproducibility checklist](docs/reproducibility_checklist.md): what to record
  when reporting FPDE experiments.
- [Data and code availability](docs/data_and_code_availability_statement.md):
  repository and dataset availability statement.
- [Repository metadata](docs/repository_metadata.md): project URL, description,
  topics, and important paths.
- [Release notes](RELEASE_NOTES.md): package history.

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
Citation metadata is available in [CITATION.cff](CITATION.cff).

## License

FPDE is distributed under the Apache License 2.0. See [LICENSE](LICENSE).
