# mbt-adapter-base

The versioned contract between [mbt](https://github.com/satrijandi/mbt)'s engine and its adapters.
Every mbt adapter - training, data, tracking, registry, compute, or tuning - depends on this package and on nothing else from mbt, which is what lets one be built, versioned, and shipped independently of the engine.

```bash
pip install mbt-adapter-base                 # the contract
pip install 'mbt-adapter-base[metrics]'      # plus the shared metric engine (numpy, scikit-learn)
pip install 'mbt-adapter-base[compliance]'   # plus the adapter compliance test suite (pytest)
```

mbt is not on PyPI yet; see [Installation](https://satrijandi.github.io/mbt/installation/) for installing from a release tag.
You rarely install it directly: every other mbt package depends on it.

## What is in it

| Module | Contents |
|---|---|
| `mbt_adapter_base.protocols` | The adapter protocols (`TrainingAdapter`, `DataAdapter`, `TrackingAdapter`, `RegistryAdapter`, `ComputeAdapter`, `TuningEngine`), the optional `Supports*` capability protocols, and the `AdapterPlugin` entry-point descriptor |
| `mbt_adapter_base.interchange` | The serializable types that cross the boundary: `TrainingJob`, `JobResult`, `ManifestNode`, `ArtifactRef`, `MetricResults`, `DeterminismTier`, `RunContext`, ... |
| `mbt_adapter_base.specs` | The Pydantic models for every spec a user writes: `DatasetSpec`, `ModelSpec`, `ScoringSpec`, and their parts |
| `mbt_adapter_base.metrics` | The shared metric engine, so champion and challenger compute every metric with identical code whatever adapter trained them |
| `mbt_adapter_base.training_helpers` | Helpers most training adapters need: the `evaluate()` body, `{{ auto }}` class-weight resolution, the calibration split, staged Parquet for path-based frameworks |
| `mbt_adapter_base.encoding` | Feature derivation and deterministic categorical encoding |
| `mbt_adapter_base.calibration` | Post-hoc isotonic and sigmoid calibrators |
| `mbt_adapter_base.materialization`, `predictions` | The shared on-disk layout for materialized datasets and prediction stores |
| `mbt_adapter_base.compliance` | `TrainingAdapterCompliance` and `PredictionStoreCompliance`, the test suites that are the ship bar for an adapter |

The package ships a PEP 561 `py.typed` marker, so these protocols type-check in your own adapter.

## Ship an adapter

```toml
# pyproject.toml of your adapter
[project]
dependencies = ["mbt-adapter-base[metrics]>=0.1,<0.2", "myframework"]

[project.entry-points."mbt.adapters"]
myframework = "mbt_myframework.plugin:PLUGIN"
```

```python
# tests/test_compliance.py
from mbt_adapter_base.compliance import TrainingAdapterCompliance
from mbt_myframework.adapter import MyTrainingAdapter


class TestMyFrameworkCompliance(TrainingAdapterCompliance):
    adapter_factory = MyTrainingAdapter
    plugin_module = "mbt_myframework.plugin"
    framework_modules = ("myframework",)
    valid_hyperparameters = {"n_estimators": 30}
    auto_hyperparameter = None
```

Passing the suite is the ship bar: it checks contract metadata, that importing the plugin loads no ML framework, parameter validation, seed determinism within the declared tier, the train, export, load, and evaluate round trip, prediction on unlabeled data, that the model learns, and every optional capability the adapter claims.
The [adapter authoring guide](https://satrijandi.github.io/mbt/adapter-authoring/) walks through the whole contract, and the [API reference](https://satrijandi.github.io/mbt/api-reference/) is generated from this package.
`mbt-lightgbm` and `mbt-sklearn` are complete reference implementations.

## Versioning

The contract is versioned separately from the packages (`CONTRACT_VERSION`, currently 1.1).
mbt-core loads an adapter whose contract has the same major version and a minor version no newer than its own, and refuses anything else with an upgrade hint.
A deprecation warns for one minor version and is removed at the next major.

## License

Apache License 2.0.
