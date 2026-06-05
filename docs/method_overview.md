# Method Overview

Feature Prototype Direction Explainer (FPDE) is a post-hoc feature attribution
method for classification. It explains one input by contrasting the input with
a prototype for a target class and a prototype for a rival class.

The output is an attribution vector with one value per feature. Positive values
support the target class relative to the rival class. Negative values support
the rival class relative to the target class.

## When To Use FPDE

Use FPDE when you have:

- A feature matrix where each column has a consistent meaning.
- Class labels for training samples.
- A classification model, usually one that exposes `predict_proba`.
- A need to explain target-versus-rival evidence at the feature level.

FPDE does not retrain or modify the classifier. It uses training data to build
prototypes, then uses those prototypes to explain individual inputs.

## Core Terms

| Term | Meaning |
| --- | --- |
| Prototype | A representative feature vector for a class. This package currently builds one class-mean prototype per class. |
| Target class | The class being explained, often the classifier's predicted class. |
| Rival class | The contrast class, often the second-highest-probability class. |
| Evidence | The scalar target-versus-rival contrast decomposed by FPDE. |
| Attribution | A per-feature contribution to the evidence value. |
| Anchor | A reference vector used by Cos-FPDE before computing cosine contrasts. |
| Baseline | A replacement vector used for deletion and insertion perturbation curves. |

## Prototype Construction

`class_mean_prototypes(X, y)` computes one prototype per class by averaging all
training rows with that class label. `FPDEEngine.fit(X_train, y_train, model)`
builds the same prototype state and stores it for repeated explanations.

For repeated workflows, prefer `FPDEEngine` because it reuses:

- Class-mean prototypes.
- Prototype labels.
- Mean and zero anchors.
- The baseline vector.
- Label-to-prototype lookup state.

## Target And Rival Selection

When you use `FPDEEngine` with a model, FPDE chooses the local contrast from
`predict_proba`:

1. The target class is the highest-probability class.
2. The rival class is the second-highest-probability class.
3. Both labels are mapped to their fitted class prototypes.

When you call lower-level functions directly, you can pass `positive_label` and
`negative_label` yourself. If `negative_label` is omitted, FPDE selects a
non-target prototype according to the mode.

## Diff-FPDE

Diff-FPDE decomposes the difference in squared distances from the input to the
rival and target prototypes:

```text
E_diff = ||x - p_neg||^2 - ||x - p_pos||^2
phi_j  = (x_j - p_neg_j)^2 - (x_j - p_pos_j)^2
```

The attribution values sum to the Diff-FPDE evidence. A positive feature value
means that feature moves the input closer to the target prototype than to the
rival prototype under this squared-distance contrast.

## Cos-FPDE

Cos-FPDE decomposes a regularized cosine-similarity contrast:

```text
E_cos = cos_eps(x - anchor, p_pos - anchor)
        - cos_eps(x - anchor, p_neg - anchor)
```

Cos-FPDE is an exact coordinate decomposition of the cosine contrast. It is not
a leave-one-feature-out causal effect, because the cosine denominator depends
on all coordinates.

## Hyb-FPDE

Hyb-FPDE mixes Diff-FPDE and Cos-FPDE attribution vectors:

```text
phi_hyb_j = lambda_hyb * phi_diff_j + (1 - lambda_hyb) * phi_cos_j
```

`lambda_hyb` must be in `[0, 1]`:

- `lambda_hyb=1.0` uses the Diff-FPDE endpoint.
- `lambda_hyb=0.0` uses the Cos-FPDE endpoint.
- Intermediate values blend both attribution vectors.

By default, Hyb-FPDE uses L1-normalized component attribution vectors before
mixing. Set `normalize="none"` when you want to mix the raw component scales.

## Validation-Based Lambda Selection

`FPDEEngine.select_lambda` evaluates a grid of `lambda_hyb` candidates on
held-out samples. For each candidate it computes deletion and insertion curves,
then scores the candidate with:

```text
combined_score = 0.5 * (deletion_drop_auc + insertion_auc)
```

The selected lambda is the candidate with the best validation score. Use this
when you have a validation set and want a reproducible, data-driven fixed
lambda for later explanations.

## Bayesian-FPDE

`FPDEEngine.select_bayesian_lambda` evaluates the same validation score as
`select_lambda`, but converts the finite lambda grid into a posterior
distribution. The default prior is `Beta(1, 1)`, which is uniform over the
candidate grid.

This experimental v0.1.0 Bayesian-FPDE layer is intentionally narrow:
uncertainty is over `lambda_hyb` only. It does not place a posterior over
class prototypes, does not sample prototype means, and does not model
uncertainty in the black-box classifier.

For each candidate:

```text
log_likelihood = n_eval_samples * validation_score / temperature
log_prior      = (alpha - 1) * log(lambda_hyb)
                 + (beta - 1) * log(1 - lambda_hyb)
```

The normalized posterior weights produce:

- `posterior_mean_lambda`: the Bayesian model-averaged mixture weight.
- `map_lambda`: the highest-posterior lambda candidate.
- `credible_interval`: an equal-tail credible interval on the grid.
- `posterior_rows`: candidate rows with validation metrics and posterior mass.

`engine.explain_one_bayesian` and `engine.explain_batch_bayesian` use
`posterior_mean_lambda`. Because Hyb-FPDE is linear in `lambda_hyb`, this is
equivalent to the expected attribution vector under the lambda posterior.
The returned attribution vector is therefore an expected Hyb-FPDE attribution
under the lambda posterior, not a per-feature credible interval from sampled
prototypes.

## Interpreting Results

An FPDE explanation contains:

- `attributions`: one contribution per feature.
- `evidence`: the sum of the attribution vector for the selected contrast.
- `positive_score` and `negative_score`: the target and rival side scores.
- `exactness_residual`: numerical difference between summed attribution and
  direct evidence.
- Prototype labels and prototype indices.

For visualization, `FPDEExplanation.normalized_attributions` returns an
L1-normalized attribution vector. Use it for plotting, not as the raw evidence
scale.

## Practical Notes

- Scale features before using distance-based explanations when feature units
  differ meaningfully.
- Keep the training data, validation data, preprocessing pipeline, and model
  aligned. FPDE expects all inputs to share the same feature space.
- Record `lambda_hyb`, `normalize`, `anchor_strategy`, `eps`, and baseline
  choices when reporting results.
- Treat FPDE as a local contrast explanation: the rival class matters.
- Treat Bayesian-FPDE intervals as intervals over `lambda_hyb` candidates, not
  as feature-level uncertainty intervals.
