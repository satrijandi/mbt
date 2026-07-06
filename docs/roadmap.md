# Roadmap

## v0.1 (this release)

One vertical slice done extremely well: declarative **binary classification**
over local Parquet, trained by **XGBoost/LightGBM**, tracked and registered
in **MLflow**, tuned by **Optuna**, with the full PR → CI → registry →
promotion loop, exact reproducibility via stored manifests, and
state-aware retraining.

## v1 candidates (architecture already accommodates)

- **Remote compute** - shipped for Spark (`mbt-spark` compute adapter:
  spark-submit'd jobs); K8s/Ray reuse the same serialized `TrainingJob`
  seam.
- **sklearn / PyTorch adapters** - new packages against the same contract;
  PyTorch declares a tolerance determinism tier.
- **Survival & ranking tasks** - adapters register task schemas via
  `AdapterPlugin.task_schemas`; no core changes.
- **Feast DataAdapter** - `source()` gains a feature-view form behind the
  same `DatasetHandle`/`DatasetLocator`.
- **Ensembles/stacking** - models with `ref()` inputs from other models;
  the DAG and manifest already support model → model edges.
- **`mbt score`** - batch inference from a registered artifact
  (`TrainingAdapter.predict` is already in the contract).
- **Airflow provider** - an operator shelling out to `mbt build` per
  manifest-derived task group.
- **Slice-level gates** - reporting ships in v0.1; gating is schema-ready.
- **Iceberg sources** - snapshot IDs from table metadata via
  `mbt-core[iceberg]`.
