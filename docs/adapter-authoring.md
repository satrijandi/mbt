# Adapter authoring guide

An mbt adapter is a pip package that exposes an `AdapterPlugin` descriptor through the `mbt.adapters` entry-point group.
It depends on **mbt-adapter-base**, the versioned contract, and not on mbt-core's internals.
One plugin can fill any of seven roles - training, data, tracking, registry, compute, tuning, reporting - and most fill one.

Two training adapters were built exactly this way and are the reference implementations: `packages/mbt-lightgbm` (the original extensibility proof, one estimator) and `packages/mbt-sklearn` (which selects among several estimators in the spec, so it shows how to model per-estimator hyperparameters and how the encoding a model needs can depend on the estimator family rather than on the data).
For the other roles, read the shipped adapters listed on the [Adapters](adapters.md) page; the [Adapter API reference](api-reference.md) is generated from the contract itself.

## 1. Package skeleton

```
mbt-myframework/
├── pyproject.toml
└── src/mbt_myframework/
    ├── __init__.py
    ├── params.py      # Pydantic parameter models - no framework imports
    ├── adapter.py     # the adapter - framework imported lazily
    └── plugin.py      # the descriptor - must stay import-light
```

```toml
[project]
dependencies = ["mbt-adapter-base[metrics]>=0.1,<0.2", "myframework"]

[project.entry-points."mbt.adapters"]
myframework = "mbt_myframework.plugin:PLUGIN"
```

```python
# plugin.py
from mbt_adapter_base import CONTRACT_VERSION, AdapterPlugin
from mbt_myframework.adapter import MyTrainingAdapter

PLUGIN = AdapterPlugin(
    name="myframework",
    contract_version=CONTRACT_VERSION,
    training=MyTrainingAdapter,  # also: data=, tracking=, registry=, compute=, tuning=
    fingerprint_packages=["myframework"],  # joins the manifest's env_digest
)
```

The entry-point name is what users write in YAML (`adapter: myframework`).
`fingerprint_packages` lists the distributions whose versions decide your adapter's numerics: their versions enter the manifest's `env_digest`, so `mbt run --manifest` refuses to reproduce a run on a different version (ADR-19).
`task_schemas` lets a plugin register a new task type without core changes.

Every adapter class is constructed as `MyAdapter(config: dict)`, where `config` is the rendered `config:` mapping from the target (or `{}`).

## 2. Import rules

**Importing `plugin.py` must not import the framework** (ADR-14).
`mbt parse` loads every referenced plugin to validate tasks and hyperparameters within a two-second budget, so frameworks load lazily inside the methods that use them (`train`, `evaluate`, `predict`, `load`, `export`, or a data adapter's `build_dataset`).

**Importing `plugin.py` must not load `mbt-core` either.**
Core internals are not a contract: an adapter that imports them breaks on the next core refactor without any contract version changing.
The compliance suite enforces both rules with subprocess `sys.modules` probes.
The only shipped exceptions are compute adapters that run core's own job entrypoint - `mbt-spark`'s `spark-submit` wrapper and `mbt-testing`'s inline runner - and they import it lazily, inside the method that runs a job.

Every mbt package ships a PEP 561 `py.typed` marker, so the protocols are real types in your checkout: run mypy over your adapter and a drifted signature is an error in your own build.

## 3. Training adapters

### Required surface

See `mbt_adapter_base.protocols.TrainingAdapter`.

| Member | Notes |
|---|---|
| `name`, `contract_version`, `supported_tasks`, `determinism` | Plain attributes |
| `param_model(task)` | A Pydantic model with `extra="forbid"`; validates static hyperparameters at parse time |
| `validate(spec)` | Extra spec-level checks; return `ValidationIssue`s with `severity` `error` or `warning` |
| `resolve_auto(spec, profile)` | Replace `{{ auto }}` sentinels from the dataset profile; must be idempotent |
| `train(spec, data, ctx)` | Return an opaque trained model; seed with `ctx.seed` |
| `evaluate(model, data, split, metrics, slices=None)` | Compute the requested `MetricSpec`s and return `MetricResults` |
| `predict(model, data, split)` | The split's table plus a `prediction` column; must work **without** the target column, because batch scoring passes an unlabeled `score` split (ADR-20) |
| `export(model, format, store)` | Write through `store.put_file(...)` and return the `ArtifactRef`; `mbt build` always asks for `"native"` |
| `load(ref, store)` | Rebuild the model from an artifact, for champion gates and `mbt evaluate` |
| `nondeterminism_warnings(spec)` | Strings describing settings that break your determinism tier; shown at parse |

### Optional capabilities

Core probes for these rather than requiring them, and the model card, scoring, and tuning use whichever you provide.

| Member | Enables |
|---|---|
| `feature_importance(model)` | Normalized per-feature fractions on the model card and in `run_results.json`. Return `{}` when the winning model cannot attribute (an ensemble leader, say) |
| `shap_importance(model, data, split)` | Normalized mean absolute SHAP importance over `split`; the card prefers it to `feature_importance`, since it is additive and not biased toward high-cardinality features |
| `explain(model, data, split, top_k)` | Required if a scoring node sets `output.explain_top_k`: one JSON string per row, `[[feature, contribution], ...]` for the row's top `top_k` features by absolute SHAP value (`training_helpers.top_k_explanations` builds it). Without it, `mbt score` fails with a `ConfigError` |
| `train_with_report(spec, data, ctx, report)` | Train while calling `report(step, value)` each iteration with a higher-is-better validation value, so a tuning pruner can stop weak trials. The callback may raise - let that exception propagate out of your training loop |
| `supports_calibration = True` | Accept a `calibration:` spec. Fit a `mbt_adapter_base.calibration.Calibrator` on the split `training_helpers.calibration_split(data)` names - the dedicated `calibration` slice core carves from train, never the selection split (F17) - persist it in the artifact, and apply it at predict time to both the challenger and a reloaded champion |
| `supports_monotonic_constraints = True` | Accept `features.monotonic` and `transforms.*.monotonic`; `training_helpers.monotone_vector` aligns the directions to your feature order |
| `supports_categorical_pooling = True` | Accept `categorical.*.min_frequency`, which needs the train-fitted level map `mbt_adapter_base.encoding` keeps |
| `data_access = "path"` | Receive splits as Parquet files instead of in-memory Arrow, for JVM and cluster frameworks (ADR-17); `training_helpers.staged_split_path` stages them |

The method capabilities each have a `@runtime_checkable` protocol in `mbt_adapter_base.protocols`: `SupportsFeatureImportance`, `SupportsShapImportance`, `SupportsExplain`, `SupportsTrainWithReport`.
A runtime check only confirms the method exists by name, so pin the signature statically too: add a `_capability_conformance` variable typed with the protocols, as `mbt-xgboost` and `mbt-lightgbm` do, and strict mypy rejects a drifted signature.
A class flag left `False` makes the parser reject a spec that needs the capability, naming your adapter, so a user never trains a silently unconstrained or uncalibrated model.

### What `train` and `evaluate` receive

The tables hold the selected features, the target column, and any declared slice columns; the split time column never reaches you.
Derive features as `columns - {spec.target} - set(spec.evaluation.slices)` - `mbt_adapter_base.encoding.split_feature_columns` does this and separates string columns as categoricals.
If your framework supports categoricals natively, train them that way with deterministic (sorted) train-time levels persisted in the artifact, and map unseen levels to missing; `features.categorical` policies (`levels`, `min_frequency`, `max_levels`, `null_as_level`) are applied by `encoding.train_categories` and `categorical_codes`.
Reject other non-numeric types (timestamps, nested types) with an actionable error rather than encoding them silently.

A `validation` split is present when the dataset declares `split.validation`, or during a tuning trial, where core carves one from train.
If your adapter honours early stopping, call `training_helpers.note_early_stopping_without_validation` so a user who set it without a validation split is told every round trained.

### Determinism

Declare a `DeterminismTier`: `exact` if the same seed and data reproduce metrics bit for bit (fix thread counts), otherwise `tolerance` with per-metric absolute tolerances.
Threshold gates widen by your tolerance in the model's favor; champion deltas never do.

### Metrics

Use `mbt_adapter_base.metrics.compute_results`, through `training_helpers.evaluate_split`, rather than computing metrics yourself.
It implements every builtin metric and the slice group-bys, so champion/challenger comparisons across adapters compute every metric with identical code.
It dispatches on the metric name, so an adapter that supports both tasks needs no metric-side branching: binary classification (`roc_auc`, `pr_auc`, `logloss`, `brier`, `accuracy`, `ece`, and the parameterized `recall_at_precision_*`, `precision_at_recall_*`, `lift_at_*`, `gain_at_*`, `threshold_at_precision_*`, `threshold_at_recall_*`) and regression (`rmse`, `mae`, `r2`, `mape`).

Other shared helpers cover `{{ auto }}` class-weight resolution (`resolve_scale_pos_weight`) and the staged-Parquet fallback for path-based frameworks; prefer them to re-implementing.

## 4. Data adapters

A data adapter turns a source into materialized splits, pins snapshots, and - from contract 1.1 - builds scoring batches and opens the prediction store.
See `mbt_adapter_base.protocols.DataAdapter`; `packages/mbt-snowflake` is a compact SQL-pushdown example and the built-in `local` adapter in mbt-core the simplest one.

| Member | Notes |
|---|---|
| `name` | Plain attribute |
| `snapshot_id(source, deep=False)` | A stable token for the relation's current state, cheap by default and content-based when `deep`. It must move whenever the data or the relation's shape changes, and only then |
| `build_dataset(spec, ctx)` | Read the one relation `ctx.source` names (ADR-29), apply `spec.filters`, sampling, and the split windows in `ctx.resolved_windows`, write one Parquet file per split under `ctx.output_dir`, and return a `DatasetHandle`. Verify the relation still matches `ctx.node.snapshot_id` first, and fail if it moved |
| `from_locator(locator)` | Reopen a materialization inside a job subprocess from its serialized `DatasetLocator` |
| `build_scoring_input(spec, ctx)` | Contract 1.1: one unlabeled `score` split, with the same filters and sampling and the `score` window. Zero rows is a warning, not an error, and a scoring batch is never verified against a pin - it is expected to change every run (R2-10) |
| `open_predictions(output)` | Contract 1.1: a `PredictionStore` for the scoring node's `output` |
| `supported_source_formats` | A frozenset such as `{"parquet"}`; compilation rejects a source declaring any other format before anything runs |
| `count_source_duplicates(source, columns)` | Optional: the number of composite keys appearing more than once in a raw table, nulls ignored - backs `unique: {source: ...}`. Push it down; only a scalar should come back |
| `read_source_distinct(source, column)` | Optional: distinct non-null values as a single `value` column - backs `relationships` |

Reuse `mbt_adapter_base.materialization.MaterializedDatasetHandle` and `write_materialization_metadata` for the on-disk layout, and `mbt_adapter_base.predictions.LocalPredictionStore` for a file-based prediction store; training adapters and `mbt monitor` already read those layouts.
Report progress through `ctx.events.emit("...")` with a plain string - the contract gives adapters no event types - as the built-in adapters do with their row counts.
A data adapter that lacks the optional source-level methods still works; the checks that need them fail with an actionable message rather than passing silently.
Core probes for the contract 1.1 methods, so a 1.0 data adapter keeps training and fails only `mbt score`, with a clear message.

**Sampling and random splits must use the canonical digest (F19)**, so a given fraction and seed select the same rows on every backend:
the unsigned lower 64 bits of the md5 of the `|`-joined key (salt first when present, each column cast to the engine's string type and coalesced to `''`), modulo 1,000,000.
That is Snowflake's `MD5_NUMBER_LOWER64`, Spark's `conv(substring(md5(...), 17, 16), 16, 10)`, and DuckDB's `('0x' || substring(md5(...), 17, 16))::UBIGINT`.
Pin your SQL to the Python reference `int(md5(preimage).hexdigest()[16:32], 16) % 1_000_000` in a test, as the built-in adapters do.
Hash the declared `sample_key`; with no key, refuse rather than hashing every column, because a keyless digest re-buckets every row whenever a column is added (ADR-29).

## 5. Tracking and registry adapters

**Tracking** (`TrackingAdapter`) records training runs only: `mbt score` and `mbt monitor` never open one (ADR-28).
Implement `start_run(node, meta)`, `log(run, params=..., metrics=..., tags=..., artifacts=...)`, `end_run(run, status)`, and `resume(run_id)`.
The training job starts and ends the run; afterwards the coordinator resumes it by id to attach the gate verdicts and the registered version, so `resume` must work from a different process.
Four members are optional and probed: `prepare()` to warm the backend before parallel jobs start, `log_trial(run, index, params, value)` for tuning history, `log_document(run, path)` to upload a file mbt wrote, such as `inference_config.json`, and `log_directory(run, local_dir, artifact_path)` to upload a directory under `artifact_path` with its layout kept.
The training report, the run log, and the config documents need `log_directory` (ADR-30); without it only `report.html`, `summary.json`, and the log reach the run through `log_document`, and the report stays complete in the artifact store.
`log` receives every flattened `model.*` and `dataset.*` parameter at once, often several hundred: batch them within your backend's limits, and cut an over-long value visibly rather than dropping it.
Core composes the experiment name and passes it as `config["experiment"]`; use it as given.

**Registry** (`RegistryAdapter`) implements `register(artifact, name, metadata)`, `get_champion(name, stage)`, `get_version(name, version)`, and `transition(version, stage)`.
Store `metadata` as version tags and return them on read: `mbt promote` reads `mbt.gates_passed`, and scoring reads the artifact, baseline, and inference-config references from them.
The optional `set_version_tags(name, version, tags)` merges tags into an existing version; `mbt evaluate --out-of-time` records its verdict (`mbt.oot_check.*`) with it, and without it the verdict lands on the training run only, so `mbt promote --require-oot-check` cannot see it.
Keep stages exclusive per version, and archive a displaced production champion on promotion, so `mbt rollback` has something to roll back to.
Return `None` only for a model, version, or stage that genuinely does not exist, and raise on anything else: a transient backend error read as "no champion" would silently pass a champion gate (F9).

## 6. Compute adapters

A compute adapter (`ComputeAdapter`) runs a serialized `TrainingJob` somewhere and returns its `JobResult`.
Implement `submit(job)`, returning a handle with a `job_id`, and `wait(handle)`.
The job runs `python -m mbt.execute.job <job.json>`, which writes `<job.json>.result.json` and streams JSON events on stdout; the result file is authoritative, so a job that dies without writing one must come back as an error result.
Forward each event line to the coordinator's bus as it arrives, so logs interleave in real time.
`mbt.adapters.local.compute` exposes `parse_job_line`, `result_path_for`, and `parse_job_timeout` as public helpers for exactly this, which is why a compute adapter may depend on mbt-core.
Implement the optional `terminate(handle, reason)` so `job_timeout_seconds` and `--fail-fast` can stop a running job.

## 7. Tuning engines

A tuning engine (`TuningEngine`) has a `name` and `tune(spec, objective, n_trials, seed)` returning a `TuningResult` (best parameters, best value, trial and pruned counts).
Call `objective(params)` once per trial; the trial loop, data, and metrics belong to the job.
When `spec.pruner` is set, call `objective(params, report=report)` instead, where your `report(step, value)` receives higher-is-better values and may raise to prune the trial.
Seed every source of randomness from `seed`, so the same spec proposes the same trials, and read operational knobs such as sampler choice from `config`, never from the spec: they must not change a model's identity.

## 8. Report engines

A report engine (`ReportingEngine`, contract 1.2) implements `drift_report(reference, current, out_html, *, title)` and returns a `DriftReport`: the share of columns the engine calls drifted, and a `DriftColumn` per column with its method, score, threshold, and verdict.
The job calls it inside the training process for the whole after-test window and each month after it, with the test split as `reference`; both tables hold the model's most important features plus a `prediction` column.
Write the page to `out_html` and give it `title`.
Your verdicts are displayed, never gated, and an exception becomes a warning on the report, so raise rather than return a partial answer.
Leave `fingerprint_packages` empty: a report never changes a model, so the engine's version does not belong in the environment digest.

## 9. Pass the compliance suite

```python
# tests/test_myframework_compliance.py
from mbt_adapter_base.compliance import TrainingAdapterCompliance
from mbt_myframework.adapter import MyTrainingAdapter


class TestMyFrameworkCompliance(TrainingAdapterCompliance):
    adapter_factory = MyTrainingAdapter
    plugin_module = "mbt_myframework.plugin"
    framework_modules = ("myframework",)
    valid_hyperparameters = {"n_estimators": 30}
    auto_hyperparameter = "scale_pos_weight"  # or None
```

Install `mbt-adapter-base[compliance]` for the suite.
For a training adapter it checks contract metadata, both import rules, rejection of unknown parameters, seed determinism within your declared tier, `resolve_auto` idempotence with no leftover sentinels, a stable train, export, load, and evaluate round trip, `predict` with and without the target column, that the model learns from both a numeric and a categorical signal, regression when supported, and every optional capability you claim.

A data adapter that implements batch scoring also subclasses `PredictionStoreCompliance`, which checks idempotent `write_run` by run key, `scored_at` ordering, column projection, and the marker ledger (ADR-21).

Passing the suite is the ship bar.
Then add an end-to-end test that drives a small project through the real CLI against your adapter; the repository's `tests/test_adapter_swap.py` does this for `lightgbm` and `sklearn` by editing only the spec.

## 10. Contract versioning

`mbt-adapter-base` versions the contract separately from its packages.
Pin `contract_version` to the version you built against; core loads an adapter with the same major version and a minor version no newer than its own, and refuses anything else with an upgrade hint.
A deprecation warns for one minor version and is removed at the next major.
Contract 1.1 added the scoring surface (`DataAdapter.build_scoring_input`, `DataAdapter.open_predictions`).
Contract 1.2 added the reporting role (`AdapterPlugin.reporting`, `ReportingEngine`).
