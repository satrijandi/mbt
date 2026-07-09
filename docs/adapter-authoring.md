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
    fingerprint_packages=["myframework"],   # joins the env digest
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
| `feature_importance(model)` | optional: normalized per-feature fractions; rendered in model cards and `run_results.json` (FR-DOCS-02) |
| `train_with_report(spec, data, ctx, report)` | optional: train while calling `report(step, value)` per iteration with a HIGHER-IS-BETTER validation value (e.g. AUC on the `validation` split); lets tuning pruners stop weak trials early. The callback may raise - let that exception propagate out of your training loop |

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

For binary classification, `mbt_adapter_base.metrics.compute_binary_results`
gives you the shared implementations (roc_auc, pr_auc, logloss, brier,
accuracy, ece, recall_at_precision_*, precision_at_recall_*) plus slice
group-bys - use it so champion/challenger comparisons across adapters are
apples to apples.

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
    auto_hyperparameter = "scale_pos_weight"   # or None
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

## 5. Contract versioning

`mbt-adapter-base` is SemVer'd independently. Pin `contract_version` to the
version you built against; core accepts the same major with a minor ≤ its
own and refuses otherwise with an upgrade hint. Deprecations warn for one
minor and are removed the next major. Contract 1.1 added the scoring surface
(`DataAdapter.build_scoring_input`, `DataAdapter.open_predictions`); core
probes for it with `hasattr`, so 1.0 data adapters keep training and fail
only `mbt score`, with a clear message.
