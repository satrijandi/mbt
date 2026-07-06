# ADR-17: JVM-backed adapters (Spark, H2O) and path data access

**Status:** accepted

## Context

Spark and H2O do not want Arrow tables handed to them by Python - they
ingest files (or distributed frames) natively, and forcing their data
through `DatasetHandle.read()` would round-trip everything through the
coordinator's memory, defeating the point of distributed engines.

## Decisions

1. **`TrainingAdapter.data_access: "arrow" | "path"`.** Arrow remains the
   default. Path adapters are guaranteed a handle whose ``split_path()``
   parquet files already have hooks and feature selection applied: the
   training job materializes the transformed view to disk before handing
   it over, so path adapters see exactly what arrow adapters see. One
   extra parquet write per split is the cost of a uniform contract.

2. **Spark fits at three seams, shipped as one plugin (`mbt-spark`):**
   - *data*: `SparkDataAdapter` builds datasets from parquet/Delta paths or
     catalog tables; joins/filters/sampling/split-windows push down as
     Spark SQL; splits land in the shared materialization so jobs reopen
     them without a session. Snapshots: file listings (local stat-based,
     remote `inputFiles()`); Delta/Iceberg version pinning is the designed
     upgrade.
   - *compute*: `SparkComputeAdapter` spark-submits a wrapper running
     `mbt.execute.job` on the driver - the serialized TrainingJob is
     exactly the remote seam ADR-3 promised. The driver env must have
     mbt-core + the training adapter installed (the K8s image contract).
   - *training*: `SparkMLTrainingAdapter` (GBT pipeline) is the genuinely
     distributed trainer; artifacts are zipped PipelineModel directories.
     Session config comes from target vars (`spark_master`, `spark_conf`),
     so dev runs `local[*]` and prod points at a cluster.

3. **H2O AutoML (`mbt-h2o`) declares the *search*, not the estimator.**
   `max_models` + `seed` bounded runs are repeatable (tolerance tier);
   wall-clock budgets (`max_runtime_secs*`) trigger nondeterminism
   warnings. A model using this adapter must not also declare `tuning:`
   (rejected at parse: it would tune the tuner). Artifacts are MOJOs -
   single self-contained zips - reloaded via `h2o.import_mojo` for
   champion evaluation. Metrics are computed by mbt's shared helpers over
   leader/MOJO probabilities so cross-adapter gate deltas stay honest.
   PySparkling is the same adapter with `h2o_backend: sparkling`
   (`mbt-h2o[sparkling]`, version-locked to the Spark minor).

4. **Training-adapter runtime config flows through target vars**, not a
   new profiles slot: training adapters are constructed bare by contract,
   and `RunContext.vars` already crosses the job boundary. Revisit if
   adapters accumulate real config surfaces.

## Consequences

- The compliance suite runs unchanged against JVM adapters (its in-memory
  fixtures are staged to temp parquet by the adapters' path fallback).
- mbt-spark depends on mbt-core (the compute adapter forwards job events
  through the bus); training/data adapters elsewhere keep the
  adapter-base-only rule - the G4 proof remains LightGBM.
- Spark compute on `master: local[*]` doubles as a memory-isolated local
  runner and makes the whole seam testable without a cluster.
