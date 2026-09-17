# Adapters

mbt's engine never imports an ML framework, a warehouse client, or a tracking server.
Everything that touches one is an **adapter**: a separately installed package that implements one or more of seven roles against the versioned contract in `mbt-adapter-base`.
This page lists the adapters that ship in this repository and what each one supports.
To write your own, read [Adapter authoring](adapter-authoring.md).

## The seven roles

| Role | Configured in | What it does |
|---|---|---|
| **training** | a model spec's `adapter:` | Fits, predicts, evaluates, and exports one model; declares its determinism tier |
| **data** | a target's `data:` in `profiles.yml` | Pins source snapshots, materializes datasets and scoring batches, and opens the prediction store |
| **tracking** | a target's `tracking:` | Records training runs: parameters, metrics, tags, documents, tuning trials |
| **registry** | a target's `registry:` | Registers versions, resolves champions by stage, moves stages |
| **compute** | a target's `compute:` | Runs a serialized training job somewhere: a subprocess, `spark-submit`, a cluster |
| **tuning** | a model spec's `tuning.engine:` (knobs in a target's `tuning:`) | Proposes hyperparameters for the trial loop the job runs |
| **reporting** | a model spec's `evaluation.report.stability.engine:` | Renders a drift report beside the training report's own tables; never gates |

A target chooses its data, tracking, registry, and compute adapters, and a model chooses its training adapter.
Specs therefore stay portable: the same model runs on DuckDB in development and on Snowflake or Spark in production by switching `--target`.

## Shipped adapters

| Adapter name | Package | Roles |
|---|---|---|
| `local` | `mbt-core` | data (DuckDB over Parquet), compute (subprocess) |
| `xgboost` | `mbt-xgboost` | training |
| `lightgbm` | `mbt-lightgbm` | training |
| `sklearn` | `mbt-sklearn` | training |
| `h2o_automl` | `mbt-h2o` | training |
| `spark` | `mbt-spark` | training (SparkML), data (lakehouse), compute (spark-submit) |
| `snowflake` | `mbt-snowflake` | data |
| `mlflow` | `mbt-mlflow` | tracking, registry |
| `optuna` | `mbt-optuna` | tuning |
| `evidently` | `mbt-evidently` | reporting |
| `fake` | `mbt-testing` | training, tracking, registry, compute (inline), tuning, reporting |

All of them implement adapter contract 1.2, which added the reporting role; a plugin built against 1.1 still loads.
Core loads an adapter whose contract has the same major version and a minor version no newer than its own, and refuses anything else with an upgrade hint.

## Training adapters

| | `xgboost` | `lightgbm` | `sklearn` | `h2o_automl` | `spark` |
|---|---|---|---|---|---|
| Tasks | binary, regression | binary, regression | binary, regression | binary, regression | binary, regression |
| Determinism tier | exact | exact | exact | tolerance, 0.02 | tolerance, 0.01 |
| Artifact | UBJSON booster | text model | joblib | MOJO zip | PipelineModel zip |
| Receives data as | Arrow tables | Arrow tables | Arrow tables | Parquet paths | Parquet paths |
| String features | native categorical | native categorical | ordinal or one-hot, by estimator | native categorical | ordinal index |
| Calibration | yes | yes | yes | yes | yes |
| Monotone constraints | yes | yes | histogram boosting only | no | no |
| Rare-level pooling | yes | yes | yes | no | no |
| Tuning search | yes | yes | yes | no, AutoML searches | yes |
| Pruning of weak trials | yes | yes | no | not applicable | no |
| Card feature importance | SHAP | SHAP | impurity or coefficients | variable importance | model importance |
| Per-row explanations | yes | yes | no | no | no |
| Needs a JVM | no | no | no | yes | yes |

In spec terms: calibration is `calibration:`, monotone constraints are `features.monotonic` (on sklearn only with `estimator: hist_gradient_boosting`), rare-level pooling is `features.categorical.<column>.min_frequency`, the tuning search is a `tuning:` block, and per-row explanations are a scoring pipeline's `output.explain_top_k`.
The artifact formats are recorded as `xgboost_ubj`, `lightgbm_json`, `sklearn_joblib`, `h2o_mojo`, and `sparkml_zip`.

**Determinism tiers.**
An *exact* adapter reproduces metrics bit for bit from the same seed and data, which is what makes `mbt run --manifest` a byte-level reproduction.
A *tolerance* adapter declares a per-metric band; threshold gates widen by it in the model's favor, and champion deltas never do.

**Unsupported features fail at parse.**
Declaring `features.monotonic` on `spark`, `min_frequency` on `h2o_automl`, or `tuning:` on `h2o_automl` is a `mbt parse` error that names the adapter, never a silently unconstrained model.

**Mixing families.**
Spark indexes string features to ordinal codes instead of splitting on categories, so a categorical-heavy champion/challenger comparison between `spark` and a tree adapter is not like for like.
The metric code is shared, so the numbers are computed identically; the models are just different families.

Package details: [mbt-xgboost](https://github.com/satrijandi/mbt/tree/main/packages/mbt-xgboost), [mbt-lightgbm](https://github.com/satrijandi/mbt/tree/main/packages/mbt-lightgbm), [mbt-sklearn](https://github.com/satrijandi/mbt/tree/main/packages/mbt-sklearn), [mbt-h2o](https://github.com/satrijandi/mbt/tree/main/packages/mbt-h2o), [mbt-spark](https://github.com/satrijandi/mbt/tree/main/packages/mbt-spark).

## Data adapters

| | `local` | `snowflake` | `spark` |
|---|---|---|---|
| Reads | Parquet globs (`path:`) | tables and views (`identifier:`) | Parquet or Delta (`path:`), catalog tables (`identifier:`) |
| Where filters, sampling, and splits run | DuckDB, in process | pushed down as Snowflake SQL | pushed down as Spark SQL |
| Default snapshot token | file listing: path, size, mtime | `SYSTEM$LAST_CHANGE_COMMIT_TIME` plus a column fingerprint | file listing (local paths); input-file listing (URIs and catalog tables) |
| `--deep-snapshot` token | file contents | `HASH_AGG(*)` | file contents (local paths); unchanged for URIs, whose listing is already mtime-independent |
| Batch scoring (`mbt score`, `mbt monitor`) | yes | yes | yes |
| Prediction store | Parquet under the project | Parquet staged under `predictions_root` | Parquet staged under `predictions_root` |
| Source-level checks (`unique: {source:}`, `relationships`) | yes | yes | yes |

All three bucket rows for sampling and random splits by the same digest of the `sample_key`, so a given fraction selects the same rows on DuckDB, Snowflake, and Spark, and a model validated on a laptop trains on the same partition in the warehouse.

`predictions_root` defaults to `<tmpdir>/mbt-predictions` on the warehouse adapters, never the project directory, so a scheduled run does not write into its own checkout.
That default is cleared on reboot; set `predictions_root` to a durable location in production.
A store that writes predictions back into warehouse tables is designed in [ADR-23](adr/0023-warehouse-batch-scoring.md) and not yet shipped.

Package details: [mbt-snowflake](https://github.com/satrijandi/mbt/tree/main/packages/mbt-snowflake), [mbt-spark](https://github.com/satrijandi/mbt/tree/main/packages/mbt-spark).

## Tracking and registry: `mlflow`

```yaml
tracking: {adapter: mlflow, config: {uri: "sqlite:///mlflow.db", experiment: wide_v2}}
registry: {adapter: mlflow, config: {uri: "sqlite:///mlflow.db"}}
```

| Key | Default | Meaning |
|---|---|---|
| `uri` | `sqlite:///mlflow.db` | Tracking or registry URI; a relative sqlite path resolves against the project |
| `experiment` | the project name | Tracking only. The experiment is named `<project>__<experiment>` |
| `use_aliases` | `true` | Registry only. mbt stages map to registered-model aliases; set `false` for MLflow servers older than 2.9, which only have the deprecated stage API |

Only training opens tracking runs; `mbt score` and `mbt monitor` write to the prediction store instead ([ADR-28](adr/0028-mlflow-training-only-and-champion-carried-config.md)).
See [Where tracking runs land](spec-reference.md#where-tracking-runs-land).

## Compute adapters

| | `local` | `spark` |
|---|---|---|
| Runs a job as | a `python -m mbt.execute.job` subprocess | a `spark-submit` of the same serialized job |
| `job_timeout_seconds` | yes | yes |
| Use it for | every laptop and CI build | training that needs a driver with cluster-sized memory; any training adapter works unchanged |

Every model trains in its own process, so a crash or out-of-memory kill in a native library fails one node rather than the run ([ADR-3](adr/0003-coordinator-job-split.md)).

## Tuning: `optuna`

```yaml
# a model spec
tuning:
  engine: optuna
  n_trials: 50                     # capped by the target's max_tuning_trials var
  search_space:
    max_depth: {type: int, low: 3, high: 10}
  objective: {metric: pr_auc, direction: maximize}
  pruner: median

# a target in profiles.yml (operational knobs, never part of model identity)
tuning: {adapter: optuna, config: {sampler: tpe, multivariate: false}}
```

| Key | Default | Meaning |
|---|---|---|
| `sampler` | `tpe` | `tpe` or `random`, seeded from the model's `seed + 1` |
| `multivariate` | `false` | Model correlated hyperparameters jointly (TPE only) |
| `group` | `false` | With `multivariate`, model the whole search space as one group |
| `n_startup_trials` | `5` | Median pruner: trials that always run to completion first |
| `n_warmup_steps` | `5` | Median pruner: iterations before a trial can be pruned |

Tuning never sees the test split ([ADR-8](adr/0008-tuning-never-sees-test.md)).

## Reporting: `evidently`

```yaml
# a model spec
evaluation:
  report:
    stability:
      engine: evidently
      feature_top_n: 20       # the most important features, plus the score
      max_html_reports: 12    # the whole after-test window, then the newest months
```

Every training report already compares the months after the test window with the test split using mbt's own PSI and KS ([ADR-30](adr/0030-training-report-and-after-test-window.md)).
With `mbt-evidently` installed, the job also runs Evidently's data-drift preset for the whole window and for each month, and writes its HTML page plus two tables under `stability/evidently/` in the report.
Evidently chooses each column's test by type and sample size, so its verdicts can disagree with mbt's; they are shown, never gated, and a period Evidently cannot render becomes a warning on the report rather than a failed build.
The engine's version stays out of the environment digest, because a report never changes a model.
It needs a dataset with `split.out_of_time`; without one there is nothing after the test window to compare.

## Testing: `fake`

`mbt-testing` provides framework-free adapters for every role, so you can test a project's specs, gates, and CI wiring without installing XGBoost, a JVM, or a tracking server.
The `fake` training adapter's metrics are set by its `fake_metric_value` hyperparameter, which makes gate behaviour easy to drive; its tracking and registry persist under the project's `target/`.
`engine: fake` gives the training report a dependency-free drift report, for testing the reporting wiring.
See [mbt-testing](https://github.com/satrijandi/mbt/tree/main/packages/mbt-testing).
