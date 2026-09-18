# mbt-h2o

The H2O AutoML training adapter for [mbt](https://github.com/satrijandi/mbt), the declarative build tool for machine learning models.
The spec declares the *search*; the AutoML leader is the artifact, stored as a single self-contained MOJO zip.

```bash
pip install mbt-h2o                  # plus mbt-core; needs Java 17 wherever training runs
pip install 'mbt-h2o[sparkling]'     # adds the Sparkling Water backend (pins pyspark 3.5)
```

mbt is not on PyPI yet; see [Installation](https://satrijandi.github.io/mbt/installation/) for installing from a release tag.

## Use it

```yaml
models:
  - name: churn_automl
    task: binary_classification          # or regression
    adapter: h2o_automl
    owner: growth-ds@company.com
    dataset: ref('churn_training_set')
    target: churned_90d
    hyperparameters:
      max_models: 20
      include_algos: [GLM, GBM, XGBoost]
      sort_metric: aucpr
      nfolds: 0
    evaluation:
      protocol: {split: temporal}
      metrics: [pr_auc, roc_auc]
      gates:
        - {metric: pr_auc, compare_to: production, min_delta: 0.005}
    seed: 42
```

| Hyperparameter | Default | Notes |
|---|---|---|
| `max_models` | 10 | The search budget; with `seed`, it keeps runs repeatable |
| `max_runtime_secs`, `max_runtime_secs_per_model` | none | Wall-clock budgets make results depend on machine speed, so setting either triggers a nondeterminism warning |
| `include_algos`, `exclude_algos` | all | Any of `GLM`, `GBM`, `XGBoost`, `DRF`, `DeepLearning`, `StackedEnsemble` |
| `sort_metric` | `AUTO` | `AUTO`, `auc`, `aucpr`, `logloss`, or `mean_per_class_error`; leave it at `AUTO` for regression |
| `nfolds` | 0 | 0 skips cross-validation (fast); 2 or more enables stacked ensembles |
| `balance_classes` | `false` | |
| `stopping_metric`, `stopping_rounds`, `stopping_tolerance` | H2O defaults | Early stopping inside the search |

## Rules of the road

- **No `tuning:` block.**
  AutoML is the search, so a `tuning:` block is rejected at parse; tune the search with `max_models` and `include_algos` instead.
- **Both tasks.**
  `binary_classification` scores P(class 1) and `regression` predicts on the target's scale.
  AutoML detects the task from the target, so a numeric target trains a regressor.
- **Determinism: tolerance (0.02).**
  A run bounded by `max_models` and seeded is repeatable within that band; threshold gates widen by it in the model's favor, champion deltas never do.
- **Calibration** is supported.
  `features.monotonic` and `categorical.min_frequency` are not, and fail at `mbt parse`; a scoring pipeline's `output.explain_top_k` is not either, and fails when `mbt score` runs.
  Model cards show H2O's variable importance, with a categorical's levels rolled up into one feature.
- **Champions reload from the MOJO**, so champion gates and `mbt evaluate` need no H2O model registry beyond mbt's own.
- **Data arrives as Parquet paths** rather than in-memory tables, and each training job starts its own local H2O cluster and shuts it down when the job exits.
  On the local backend a job never attaches to a cluster another job is running: jobs trained in parallel (`threads: 2` or more) each get their own, on the first free port from 24321.
  Within a job the session is opened once and reused, so the backend a target declares is the one every later step scores and reloads against.
  Size them with the `h2o_max_mem` target var (default `4G`, per job) and `h2o_nthreads`.

## Distributed training: Sparkling Water

```yaml
# profiles.yml, target vars
vars:
  h2o_backend: sparkling
  spark_master: spark://cluster:7077
  spark_conf: {spark.executor.memory: 8g}
```

The same adapter and the same specs, with H2O running on Spark executors, so AutoML trains on frames that never fit one machine.
The H2O and Spark version matrix is strict: the `sparkling` extra pins pyspark 3.5 and `h2o-pysparkling-3.5`, and the H2O client must match the backend it embeds, so keep the extra's pins.
Because of that pyspark pin, `mbt-h2o[sparkling]` cannot share an environment with `mbt-spark`, which runs on Spark 4.

## The h2o version ceiling

`mbt-h2o` requires `h2o<3.46.0.12`.
That release moved MOJO export behind H2O's commercial tier, and MOJO export is exactly how this adapter stores a model, so a fresh install that resolved it could train but never save a model.
The ceiling lifts once a release restores MOJO export to the open-source tier.

## License

Apache License 2.0.
