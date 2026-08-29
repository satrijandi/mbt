# Adapter authoring guide

An mbt adapter is a pip package exposing an `AdapterPlugin` descriptor
through the `mbt.adapters` entry-point group. It depends only on
**mbt-adapter-base** - never on mbt-core. The LightGBM adapter
(`packages/mbt-lightgbm`) was built exactly this way as the extensibility
proof; use it as the reference implementation.

## 1. Package skeleton

```
mbt-myframework/
├── pyproject.toml
└── src/mbt_myframework/
    ├── __init__.py
    ├── params.py      # Pydantic param models - no framework imports
    ├── adapter.py     # the TrainingAdapter - framework imported lazily
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
    training=MyTrainingAdapter,
    fingerprint_packages=["myframework"],  # joins the env digest
)
```

## 2. The import hygiene rule (ADR-14)

Importing `plugin.py` (and constructing your adapter class) **must not
import the framework**. `mbt parse` loads every referenced plugin to
validate tasks and hyperparameters inside a 2-second budget; frameworks
load lazily inside `train`/`evaluate`/`predict`/`load`/`export`. The
compliance suite enforces this with a subprocess `sys.modules` probe.

## 3. The TrainingAdapter contract

Construction convention: `MyTrainingAdapter(config: dict)` - the (usually
empty) config dict. Required surface (see `mbt_adapter_base.protocols`):

| Member | Notes |
|---|---|
| `name`, `contract_version`, `supported_tasks`, `determinism` | plain attributes |
| `param_model(task)` | a Pydantic model with `extra="forbid"`; validates static hyperparameters at parse time |
| `validate(spec)` | extra spec-level checks; return `ValidationIssue`s |
| `resolve_auto(spec, profile)` | replace `AUTO` sentinels from the dataset profile; must be idempotent |
| `train(spec, data, ctx)` | return an opaque trained-model object; seed with `ctx.seed` |
| `evaluate(model, data, split, metrics, slices=None)` | compute the requested `MetricSpec`s; return `MetricResults` |
| `predict(model, data, split)` | the split's table + a `prediction` column; must work WITHOUT the target column (batch scoring feeds unlabeled `score` splits, ADR-20) |
| `export(model, format, store)` | write via `store.put_file(...)`; return the `ArtifactRef` |
| `load(ref, store)` | reconstruct the model from an artifact (champion evaluation, `mbt evaluate`) |
| `nondeterminism_warnings(spec)` | strings describing settings that break your determinism tier |
| `feature_importance(model)` | optional: normalized per-feature fractions; rendered in model cards and `run_results.json` (FR-DOCS-02). Return `{}` when the winning model cannot attribute (e.g. an ensemble leader) |
| `shap_importance(model, data, split)` | optional: normalized mean-\|SHAP\| importance over `split` - additive and not cardinality-biased like split-gain, so the card prefers it over `feature_importance` when you expose it (the tree adapters do). Data-grounded, hence the split argument |
| `train_with_report(spec, data, ctx, report)` | optional: train while calling `report(step, value)` per iteration with a HIGHER-IS-BETTER validation value (e.g. AUC on the `validation` split); lets tuning pruners stop weak trials early. The callback may raise - let that exception propagate out of your training loop |
| `explain(model, data, split, top_k)` | optional, but **required** if a scoring node sets `output.explain_top_k`: return one JSON string per row - that row's `top_k` features by \|SHAP\| as `[[feature, contribution], ...]` ordered by descending \|contribution\| (`mbt_adapter_base.training_helpers.top_k_explanations` builds it) - so a consumer can see why each row scored as it did. Core raises a `ConfigError` at score time if `explain_top_k` is set and you do not define this |
| `supports_calibration` (class attribute, default `False`) | optional: set `True` and implement post-hoc probability calibration to accept a `calibration:` spec. When set, fit a `Calibrator` (`mbt_adapter_base.calibration`) on the split named by `mbt_adapter_base.training_helpers.calibration_split(data)` - the dedicated `calibration` slice core carves from train (never the selection split, F17), falling back to `validation` for direct callers, failing loudly when neither exists - persist it in your artifact, and apply it - to both the challenger and the re-loaded champion - at predict time. The parser rejects a `calibration:` spec on an adapter that leaves this `False` |

The optional members are probed with `hasattr` (or, for `supports_calibration`,
read with `getattr(..., False)`); each has a `@runtime_checkable` protocol in
`mbt_adapter_base.protocols` -
`SupportsFeatureImportance`/`SupportsShapImportance`/`SupportsExplain`/`SupportsTrainWithReport`.
`runtime_checkable` verifies only that the method is present by name (that is the
`hasattr` probe); the signature is pinned statically - an adapter that implements
a capability adds a `_capability_conformance` mypy variable (see mbt-xgboost /
mbt-lightgbm) so strict mypy rejects a drifted signature. The compliance suite
also checks `feature_importance` output when present - see the
[Adapter API reference](api-reference.md).

Every mbt package ships a PEP 561 `py.typed` marker, so those protocols are
real types in *your* checkout too: run mypy over your adapter and a drifted
signature is an error in your own build, not a surprise at runtime inside mbt.
(Up to and including v0.1.0 the marker was missing, and a consumer's mypy
silently skipped every `mbt_*` import - if you have an adapter written against
an older release, expect a first strict run to surface real findings.)

Shared implementation helpers
(`mbt_adapter_base.training_helpers`) cover the common `evaluate()` body,
`{{ auto }}` scale_pos_weight resolution, and the staged-parquet fallback
for `data_access="path"` frameworks - prefer them over re-implementing.

### What your `train`/`evaluate` tables contain

selected features + the target column + declared slice columns. The split
time column never reaches you. Derive features as
`columns - {spec.target} - set(spec.evaluation.slices)` - the shared
`mbt_adapter_base.encoding` helpers do this and split off string columns as
categoricals. If your framework supports categoricals natively, train them
that way with deterministic (sorted) train-time levels persisted in the
artifact, and map unseen levels to missing - see `mbt-xgboost` and
`mbt-lightgbm`. Reject other non-numeric types (timestamps, nested) with an
actionable error - do not encode silently.

### Determinism

Declare a `DeterminismTier`: `exact` if same seed + same data reproduce
metrics bit-for-bit (fix thread counts!), else `tolerance` with per-metric
absolute tolerances. Threshold gates widen by your tolerance in the model's
favor; champion deltas never widen.

### Metrics

`mbt_adapter_base.metrics.compute_results` (via the shared
`training_helpers.evaluate_split`) gives you the builtin implementations plus
slice group-bys - use it so champion/challenger comparisons across adapters
compute every metric identically. Split membership is portable too (F19): a
data adapter that implements random splits or sampling must bucket rows by the
canonical cross-adapter digest - the unsigned lower 64 bits of the md5 of the
'|'-joined key (salt first when present, columns COALESCEd to '' and cast to
the engine's string type), modulo 1,000,000 - which is Snowflake's
`MD5_NUMBER_LOWER64`, Spark's `conv(substring(md5(...), 17, 16), 16, 10)`, and
local DuckDB's `('0x' || substring(md5(...), 17, 16))::UBIGINT`. Pin your SQL
to the Python reference `int(md5(preimage).hexdigest()[16:32], 16) % 1_000_000`
in a test, as the built-in adapters do, so the same fraction/seed selects the
same rows on every backend.
It dispatches on the metric name: binary classification
(roc_auc, pr_auc, logloss, brier, accuracy, ece, recall_at_precision_*,
precision_at_recall_*) and regression (rmse, mae, r2, mape) share one entry
point, so an adapter that supports both tasks needs no metric-side branching.

## 4. Pass the compliance suite - the ship bar

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

The suite asserts: contract metadata, plugin import hygiene, unknown-param
rejection, seed determinism within your declared tier, `resolve_auto`
idempotence with no leftover sentinels, train → export → load → evaluate
round-trip stability, `predict` shape (with and without the target column),
and that the model actually learns on a signal-bearing dataset.

DataAdapters that implement batch scoring (contract 1.1: `build_scoring_input`
+ `open_predictions`) subclass `PredictionStoreCompliance` too - it asserts
idempotent `write_run` by run key, `scored_at` ordering, column projection,
and the marker-ledger roundtrip (ADR-21). Reuse
`mbt_adapter_base.predictions.LocalPredictionStore` for file-based layouts.

DataAdapters may also implement the optional source-level check methods
(F2/F21), `hasattr`-probed by the check layer: `count_source_duplicates(source,
columns) -> int` (distinct COMPOSITE keys appearing more than once in the raw
table, nulls ignored - the pre-join `unique: {source: ...}` contract; push it
down, only a scalar returns) and `read_source_distinct(source, column) ->
pa.Table` (DISTINCT non-null values as a single ``value`` column - the parent
side of `relationships`). An adapter lacking them fails those checks with an
actionable message rather than silently passing. Population-spine
`build_dataset` implementations should also record `label_join_coverage`
(spine vs matched counts, before filters/sampling/windows) via
`write_materialization_metadata`, which the `label_join_coverage` check
enforces - the three built-in data adapters show the pattern.

## 5. Contract versioning

`mbt-adapter-base` is SemVer'd independently. Pin `contract_version` to the
version you built against; core accepts the same major with a minor ≤ its
own and refuses otherwise with an upgrade hint. Deprecations warn for one
minor and are removed the next major. Contract 1.1 added the scoring surface
(`DataAdapter.build_scoring_input`, `DataAdapter.open_predictions`); core
probes for it with `hasattr`, so 1.0 data adapters keep training and fail
only `mbt score`, with a clear message.
