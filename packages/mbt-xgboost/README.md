# mbt-xgboost

The XGBoost training adapter for [mbt](https://github.com/satrijandi/mbt), the declarative build tool for machine learning models.
Declare a gradient-boosted model in YAML, and mbt handles the splits, tuning, gates, registry, and reproducibility.

```bash
pip install mbt-xgboost        # plus mbt-core
```

mbt is not on PyPI yet; see [Installation](https://satrijandi.github.io/mbt/installation/) for installing from a release tag.

## Use it

```yaml
# models/churn_classifier.yml
models:
  - name: churn_classifier
    task: binary_classification          # or regression
    adapter: xgboost
    owner: growth-ds@company.com
    dataset: ref('churn_training_set')
    target: churned_90d
    features:
      exclude: [user_id]
    hyperparameters:
      max_depth: 6
      learning_rate: 0.1
      n_estimators: 300
      scale_pos_weight: "{{ auto }}"     # resolved from the training split's class balance
    evaluation:
      protocol: {split: temporal}
      metrics: [pr_auc, roc_auc, logloss]
      gates:
        - {metric: pr_auc, compare_to: production, min_delta: 0.005}
    registration:
      name: churn_classifier
      stage_on_pass: staging
    seed: 42
```

## Hyperparameters

Validated at `mbt parse` time against a strict parameter model, so a misspelled name is an error before anything trains.

| Name | Default | Notes |
|---|---|---|
| `n_estimators` | 100 | Boosting rounds |
| `max_depth` | 6 | |
| `learning_rate` | 0.3 | |
| `min_child_weight` | 1.0 | |
| `subsample` | 1.0 | |
| `colsample_bytree` | 1.0 | |
| `gamma` | 0.0 | |
| `reg_alpha` | 0.0 | L1 |
| `reg_lambda` | 1.0 | L2 |
| `tree_method` | `hist` | `hist`, `exact`, or `approx`; only `hist` is in the exact determinism tier |
| `device` | `cpu` | `cpu` or `cuda`; `cuda` is not bit-reproducible |
| `early_stopping_rounds` | none | Stops when the `validation` split stops improving, and the model scores with its best iteration. Declare `split.validation` on the dataset: without one there is nothing to watch, every round trains, and the adapter says so in the build output |
| `scale_pos_weight` | none | Binary classification only; `"{{ auto }}"` computes it from the class balance |

## Guarantees

- **Determinism: exact.**
  Training runs single-threaded with the seed mbt derives from the spec, so the same manifest reproduces metrics bit for bit (`mbt run --manifest`).
  Settings outside that tier (`tree_method` other than `hist`, `device: cuda`) produce a parse-time warning.
- **Categoricals.**
  String features train as native XGBoost categoricals, with the train-time levels persisted in the artifact; a level first seen at scoring time is treated as missing.
  `features.categorical` can declare integer-coded categoricals too, and supports `levels`, `min_frequency`, `max_levels`, and `null_as_level`.
- **Feature treatment.**
  `features.transforms` and `features.monotonic` are supported; monotone constraints reach the booster directly.
- **Calibration.**
  `calibration: isotonic` or `sigmoid` fits a calibrator on a slice carved from the training window, and every metric, gate, and prediction then sees calibrated scores.
- **Explainability.**
  Model cards rank features by mean absolute SHAP value, and a scoring pipeline's `output.explain_top_k` attaches each row's top SHAP contributions.
- **Tuning.**
  The adapter reports per-round validation progress, so an Optuna search with `pruner: median` stops weak trials early.
- **Artifacts.**
  Models are stored in XGBoost's native UBJSON format (`xgboost_ubj`).

## ONNX export

The `onnx` extra (`pip install 'mbt-xgboost[onnx]'`) adds `export(model, "onnx", store)` for programmatic callers of the adapter.
It does not support categorical features yet, and `mbt build` always stores the native format.

## Learn more

- [Spec reference](https://satrijandi.github.io/mbt/spec-reference/) for every model field
- [Adapters](https://satrijandi.github.io/mbt/adapters/) to compare training adapters

## License

Apache License 2.0.
