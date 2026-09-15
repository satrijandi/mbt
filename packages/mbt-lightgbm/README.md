# mbt-lightgbm

The LightGBM training adapter for [mbt](https://github.com/satrijandi/mbt), the declarative build tool for machine learning models.

It is also mbt's extensibility proof: it imports only the public adapter contract in `mbt-adapter-base`, never `mbt-core` internals, and passes the same compliance suite a third-party adapter would.
If you are writing your own adapter, this package is a compact reference.

```bash
pip install mbt-lightgbm       # plus mbt-core
```

mbt is not on PyPI yet; see [Installation](https://satrijandi.github.io/mbt/installation/) for installing from a release tag.

## Use it

```yaml
models:
  - name: churn_lgbm
    task: binary_classification          # or regression
    adapter: lightgbm
    owner: growth-ds@company.com
    dataset: ref('churn_training_set')
    target: churned_90d
    features:
      exclude: [user_id]
    hyperparameters:
      num_leaves: 31
      learning_rate: 0.05
      n_estimators: 400
      scale_pos_weight: "{{ auto }}"
    evaluation:
      protocol: {split: temporal}
      metrics: [pr_auc, roc_auc]
      gates:
        - {metric: pr_auc, threshold: 0.4}
    registration:
      name: churn_lgbm
      stage_on_pass: staging
    seed: 42
```

## Hyperparameters

Validated at `mbt parse` time against a strict parameter model.

| Name | Default | Notes |
|---|---|---|
| `n_estimators` | 100 | Boosting rounds |
| `num_leaves` | 31 | |
| `max_depth` | -1 | -1 means no limit |
| `learning_rate` | 0.1 | |
| `min_child_samples` | 20 | |
| `subsample` | 1.0 | Row bagging; below 1.0 the adapter also sets `bagging_freq: 1`, without which LightGBM ignores it |
| `colsample_bytree` | 1.0 | |
| `reg_alpha` | 0.0 | L1 |
| `reg_lambda` | 0.0 | L2 |
| `num_threads` | 1 | Above 1, training is no longer bit-reproducible, and parse warns |
| `early_stopping_rounds` | none | Stops when the `validation` split stops improving. Declare `split.validation` on the dataset: without one every round trains, and the adapter says so in the build output |
| `scale_pos_weight` | none | Binary classification only; `"{{ auto }}"` computes it from the class balance |

## Guarantees

- **Determinism: exact.**
  Training runs with LightGBM's `deterministic` and `force_row_wise` settings and the seed mbt derives from the spec, so a stored manifest reproduces metrics bit for bit.
- **Categoricals.**
  String features train as native LightGBM categoricals, with train-time levels persisted in the artifact; a level first seen at scoring time is treated as missing.
  `features.categorical` supports `levels`, `min_frequency`, `max_levels`, and `null_as_level`.
- **Feature treatment.**
  `features.transforms` and `features.monotonic` are supported; monotone constraints reach the booster.
- **Calibration.**
  `calibration: isotonic` or `sigmoid` fits on a slice carved from the training window.
- **Explainability.**
  Model cards rank features by mean absolute SHAP value, and scoring can attach per-row SHAP contributions with `output.explain_top_k`.
- **Tuning.**
  The adapter reports per-round validation progress, so Optuna's median pruner can stop weak trials early.
- **Artifacts.**
  Models are stored as LightGBM's text model (`lightgbm_json`), holding only the best iteration after early stopping, so a reloaded model scores exactly like the one that was trained.

## Learn more

- [Adapter authoring](https://satrijandi.github.io/mbt/adapter-authoring/) - this package is the worked example
- [Adapters](https://satrijandi.github.io/mbt/adapters/) - compare training adapters

## License

Apache License 2.0.
