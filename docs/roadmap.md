# Roadmap

## v0.1 (this release)

A tabular vertical done extremely well: declarative **binary classification and regression** on all five training adapters, with the full PR → CI → registry → promotion → batch scoring → ground-truth monitoring loop, exact reproducibility via stored manifests, and state-aware retraining.
Data comes from local Parquet, **Snowflake**, or a **Spark lakehouse**; training adapters cover **XGBoost/LightGBM/scikit-learn** plus **SparkML** and **H2O AutoML** (optionally distributed via Sparkling Water); **MLflow** tracks and registers; **Optuna** tunes.
The dockerized showcase (`examples/showcase`) proves the loop nightly in CI against real services end to end.

## v1 candidates (architecture already accommodates)

- **Remote compute** - shipped for Spark (`mbt-spark` compute adapter:
  spark-submit'd jobs); K8s/Ray reuse the same serialized `TrainingJob`
  seam.
- **sklearn adapter** - shipped (`mbt-sklearn`): LogisticRegression/Ridge,
  RandomForest, and HistGradientBoosting against the same public contract,
  exact determinism tier, zero mbt-core changes. It adds no dependency a
  metric-computing install does not already have.
- **PyTorch adapter** - a new package against the same contract, declaring a
  tolerance determinism tier.
- **Survival & ranking tasks** - adapters register task schemas via
  `AdapterPlugin.task_schemas`; no core changes.
- **Feast DataAdapter** - `source()` gains a feature-view form behind the
  same `DatasetHandle`/`DatasetLocator`.
- **Ensembles/stacking** - models with `ref()` inputs from other models;
  the DAG and manifest already support model → model edges.
- **`mbt score`** - shipped: batch scoring pipelines are a first-class
  `scoring` resource (1 config = 1 serving pipeline) executed by
  `mbt score`, with shift monitors against training-time baselines and
  delayed ground-truth evaluation via `mbt monitor` (ADR-20/21). Online
  serving remains a non-goal; warehouse prediction sinks are follow-ups.
- **Airflow provider** - an operator shelling out to `mbt build` per
  manifest-derived task group. Reference DAGs that run the digest-pinned
  deployable unit with exit-code routing (quality verdicts never retried)
  ship in `examples/showcase`; a first-class provider package remains open.
- **Slice-level gates** - shipped: threshold and champion gates on slices
  are evaluated and block registration. The open piece is statistical:
  champion slice gates compare point deltas, not the ADR-18 paired-bootstrap
  lower bound used for whole-split champion gates.
- **Iceberg sources** - snapshot IDs from table metadata via
  `mbt-core[iceberg]`.
