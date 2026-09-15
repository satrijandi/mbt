# Roadmap

## v0.1 - released

A tabular vertical done thoroughly: declarative **binary classification and regression** on five training adapters, with the whole loop from pull request to CI build, registry, gate-verified promotion, batch scoring, and ground-truth monitoring.
Stored manifests reproduce runs exactly, and state-aware selection retrains only what changed.

- **Data**: local Parquet through DuckDB, **Snowflake**, and a **Spark lakehouse** (Parquet, Delta, catalog tables), each reading one relation per dataset (ADR-29).
- **Training**: **XGBoost**, **LightGBM**, **scikit-learn**, **SparkML**, and **H2O AutoML**, optionally distributed through Sparkling Water.
- **Compute**: a local subprocess per job, or `spark-submit` for cluster-sized drivers.
- **Tracking and registry**: **MLflow**. **Tuning**: **Optuna**, with median pruning.
- **Serving and monitoring**: `scoring` resources run by `mbt score` with PSI and KS shift monitors, and `mbt monitor` for delayed ground-truth evaluation (ADR-20, ADR-21).
- **Quality**: threshold, paired-bootstrap champion, slice, backtest, and disparity gates; declarative data checks; post-hoc calibration; declarative feature treatment (ADR-27).

The [v0.1 status](v0.1-status.md) page carries the evidence, and the dockerized [showcase](showcase.md) runs the loop nightly against real services.

## Next

These build on seams the architecture already has; none needs a change to the adapter contract's major version.

| Item | State |
|---|---|
| **PyPI publication** | The release workflow builds and attests wheels for every package; publishing waits on creating the PyPI projects and their Trusted Publishers (see CONTRIBUTING) |
| **Warehouse-native prediction stores** | Designed in [ADR-23](adr/0023-warehouse-batch-scoring.md): predictions written back into Snowflake or lakehouse tables instead of staged Parquet. Gated on a live verification of the Snowflake serving leg, tracked in [issue #1](https://github.com/satrijandi/mbt/issues/1) |
| **PyTorch adapter** | A new training package against the same contract, declaring a tolerance determinism tier |
| **Survival and ranking tasks** | Adapters register task schemas through `AdapterPlugin.task_schemas`, with no core changes |
| **Multiclass classification** | Not started; binary classification and regression are the two verticals today |
| **Feast data adapter** | `source()` gains a feature-view form behind the same dataset handle and locator |
| **Ensembles and stacking** | Models with `ref()` inputs from other models; the DAG and manifest already support model-to-model edges |
| **Champion slice gates with a confidence bound** | Slice gates block registration today, but champion slice gates compare point deltas rather than the paired-bootstrap lower bound whole-split gates use |
| **Kubernetes and Ray compute** | New compute adapters over the same serialized training-job seam `mbt-spark` already uses |
| **Airflow provider** | The showcase ships reference DAGs that run a digest-pinned image with exit-code routing; a first-class provider package is still open |
| **Iceberg sources** | Snapshot ids read from table metadata |

[MLOps alignment](mlops-alignment.md) lists the practices mbt does not cover yet - fairness metrics beyond the disparity gate, drift-triggered retraining, delivery metrics, and others - and says which of them are candidates and which are out of scope.

## Non-goals

Recorded so the question does not have to be reopened each time it comes up.

- **Online, request/response serving.** mbt's serving surface is batch scoring: every command terminates like a job. Serving infrastructure reacts to registry stages instead (ADR-20).
- **Joining tables, or accepting a SQL query as a dataset.** A dataset reads one relation; the join that builds it belongs upstream in dbt or the warehouse, and `filters:` and `hooks.py` stay the only escape hatches (ADR-29).
- **A feature store or a model catalog product.** mbt integrates with them rather than rebuilding them.
