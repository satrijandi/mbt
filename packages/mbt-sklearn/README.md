# mbt-sklearn

scikit-learn training adapter for [mbt](../../README.md): declare a
LogisticRegression, Ridge, RandomForest, or HistGradientBoosting model as YAML
and let mbt handle the splits, gates, registry, and reproducibility.

```yaml
# models/churn_classifier.yml
models:
  - name: churn_classifier
    task: binary_classification
    adapter: sklearn
    owner: growth-ds@company.com
    dataset: ref('churn_training_set')
    target: churned_90d
    hyperparameters:
      estimator: logistic          # or random_forest / hist_gradient_boosting
      C: 0.5
      class_weight: "{{ auto }}"   # -> sklearn's "balanced"
    evaluation:
      protocol: {split: temporal}
      metrics: [pr_auc, roc_auc]
    seed: 42
```

## Why this adapter

scikit-learn is the most common tabular modelling stack there is, and mbt
already depends on it: `mbt-adapter-base[metrics]` pulls scikit-learn in to
compute PR-AUC and friends, so **every** mbt install that evaluates a model
already has it. This adapter therefore adds no new dependency to a typical
install - it just makes the framework you already have declarable.

## Estimators

One spec field picks the estimator; the rest of the block is that estimator's
own hyperparameters, validated at parse time with `extra='forbid'`.

| `estimator` | binary | regression | sklearn class |
|---|:--:|:--:|---|
| `logistic` | yes | - | `LogisticRegression` |
| `linear` | - | yes | `Ridge` (`alpha: 0` recovers OLS) |
| `random_forest` | yes | yes | `RandomForest{Classifier,Regressor}` |
| `hist_gradient_boosting` | yes | yes | `HistGradientBoosting{Classifier,Regressor}` |

Naming a hyperparameter that belongs to a *different* estimator is a parse-time
error with a field path, not a `TypeError` inside a training subprocess:

```
Value error, hyperparameter(s) C are not valid for estimator 'random_forest';
valid: class_weight, max_depth, max_features, min_samples_leaf,
min_samples_split, n_estimators, n_jobs
```

## Guarantees

- **Determinism: exact.** Every estimator is seeded from the manifest's
  `seed`, and `n_jobs` defaults to 1 because sklearn's threaded paths reduce in
  nondeterministic order. Raising `n_jobs` trades determinism for speed and
  emits a warning at parse time.
- **Categoricals**: the encoding follows the *estimator family*, not the data.
  Trees (`random_forest`, `hist_gradient_boosting`) take ordinal codes, which
  they can split arbitrarily. The linear estimators (`logistic`, `linear`) take
  one-hot columns, because a linear model reads an ordinal code as a magnitude
  and can therefore only fit a categorical whose code order happens to track
  the label. Either way the train-time levels persist in the artifact, and an
  unseen level becomes the `-1` sentinel (trees) or an all-zero row (one-hot)
  at prediction time.
- **Calibration**: `supports_calibration` is true, so `calibration:` in a spec
  fits a post-hoc calibrator on the dedicated calibration slice (ADR-18 / F17)
  and every downstream metric, gate, and prediction sees calibrated scores.
- **Feature importance**: `feature_importances_` for the tree estimators,
  `|coef_|` for the linear ones, with a one-hot expanded categorical reported
  as the sum over its levels so the answer is per feature, not per encoded
  column. `HistGradientBoosting*` exposes neither attribute, so it returns `{}`
  - the contract's documented escape hatch for a model that cannot attribute -
  rather than a row of zeros dressed up as a ranking.
- **Artifacts**: `joblib`, sklearn's own documented persistence format, with a
  JSON envelope for mbt's metadata. joblib is a pickle, so an artifact is only
  loadable by a compatible environment - which is what the manifest's
  `env_digest` already pins (ADR-19).

## Optional capabilities not implemented

Stated here rather than left to be discovered, since the compliance suite skips
them silently:

| capability | status | why |
|---|---|---|
| `shap_importance`, `explain` | not implemented | would add a `shap` dependency, which is the one thing this adapter is designed to avoid. `feature_importance` still populates the model card. |
| `train_with_report` | not implemented | sklearn has no uniform per-iteration callback across these estimators. Optuna tuning still works; trials just run to completion instead of being pruned early, and mbt logs that it is doing so. |

## Extensibility

Like `mbt-lightgbm`, this package imports **only** `mbt-adapter-base` - never
`mbt-core` internals - and passes the shared compliance suite. See
[`docs/adapter-authoring.md`](../../docs/adapter-authoring.md) to write your
own.
