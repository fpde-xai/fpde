# Reproducibility Checklist

Use this checklist when reporting FPDE experiments, publishing results, or
comparing attribution settings across runs.

## Package And Environment

- [ ] Record the FPDE package version.
- [ ] Record the Python version. FPDE requires Python 3.12 or newer.
- [ ] Record dependency versions, especially NumPy and scikit-learn.
- [ ] State whether FPDE was installed from PyPI or from a local checkout.
- [ ] Preserve the commit hash when using a local checkout.

## Data And Preprocessing

- [ ] Identify the dataset and any dataset version or download date.
- [ ] Record the train, validation, and test split procedure.
- [ ] Record all random seeds used for splitting, model fitting, and sampling.
- [ ] Describe feature preprocessing, including scaling, encoding, imputation,
  and feature selection.
- [ ] Confirm that FPDE receives data in the same feature space used by the
  classifier.
- [ ] Record feature names and feature order.

## Model

- [ ] Name the classifier and implementation library.
- [ ] Record model hyperparameters.
- [ ] Record the fitted model version or artifact location.
- [ ] Confirm that the model exposes `predict_proba` and `classes_` when using
  `FPDEEngine`, grid search, lambda selection, or perturbation curves.
- [ ] Report model performance on the evaluation split used for explanations.

## FPDE Configuration

- [ ] Record the FPDE mode: Diff-FPDE, Cos-FPDE, or Hyb-FPDE.
- [ ] Record `lambda_hyb` for Hyb-FPDE.
- [ ] Record whether `lambda_hyb` was fixed manually or selected with
  `FPDEEngine.select_lambda`.
- [ ] Record the `lambda_hyb_grid` used for validation selection.
- [ ] Record `normalize`, usually `"l1"` or `"none"`.
- [ ] Record `anchor_strategy`, usually `"mean"`, `"zero"`, or `"none"`.
- [ ] Record `eps` for cosine computations.
- [ ] Record the baseline vector used for deletion and insertion curves.
- [ ] Record the validation fractions used by `select_lambda` or
  `perturbation_curves`.

## Prototype State

- [ ] Record how prototypes were built. This package currently provides
  class-mean prototypes.
- [ ] Record the training data used to build prototypes.
- [ ] Record class labels and label ordering.
- [ ] If using `prepare_fpde_context`, preserve the context-generation inputs.

## Explanation Outputs

- [ ] Save raw attribution vectors before visualization normalization.
- [ ] Save target and rival labels for each explanation.
- [ ] Save evidence values and exactness residuals.
- [ ] Save `positive_score` and `negative_score` when comparing contrasts.
- [ ] Save grid-search rows or validation-selection rows when using automated
  selection.

## Validation And Sanity Checks

- [ ] Run the test suite with `python -m pytest`.
- [ ] Verify that attribution shapes match the number of features.
- [ ] Check that attribution values are finite.
- [ ] Check that target and rival labels are different for multiclass
  explanations.
- [ ] Inspect deletion and insertion curves for a representative sample.
- [ ] Confirm that repeated runs with fixed seeds are deterministic.

## Minimal Reporting Template

```text
FPDE version:
Python version:
Model:
Dataset:
Preprocessing:
Train/validation/test split:
Prototype construction:
FPDE mode:
lambda_hyb:
normalize:
anchor_strategy:
eps:
baseline:
Validation fractions:
Random seeds:
Repository commit:
```
