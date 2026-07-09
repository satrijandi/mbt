# ADR-21: Prediction stores, training-time baselines, and the ground-truth ledger

**Status:** accepted

Contract 1.1 adds two DataAdapter methods: `build_scoring_input` (materialize one unlabeled batch as a single `score` split) and `open_predictions`, which returns a `PredictionStore` owning writes, run scanning, and the evaluation ledger.
A bare `write_predictions -> DatasetLocator` was rejected: it leaves "which runs exist" and "which runs were evaluated" without an owner, and `DatasetLocator` carries snapshot semantics that do not fit predictions.
Core probes for the methods with `hasattr` before any job runs, so a 1.0-era data adapter fails `mbt score` with a clear error instead of mid-run; the registry's minor-version rule keeps 1.0 plugins loading everywhere else.

The shared local layout (`mbt_adapter_base.predictions`, mirroring the materialization module) stores one directory per prediction run: `predictions.parquet`, a `predictions.json` sidecar (scored-at anchor, model name and version, identity hashes), a `_SUCCESS` marker, and ledger markers.
The run key is `sha256(scoring input_hash | resolved windows | champion version)[:16]`: re-running the same manifest against the same champion overwrites its own run idempotently, while new data, a new window, or a new champion partitions fresh.
Rewriting a run clears its markers: a fresh run gets a fresh ledger.

Every training job builds a monitoring baseline and exports it next to the model artifact: per-feature quantile grids (categoricals keep top values plus `__other__`) from the post-hook train split, plus the test-split score distribution.
Registration pins the baseline into registry tags (`mbt.baseline_uri` and friends), so the champion carries its own reference distribution wherever it is loaded; artifact GC keeps them together because they share a run prefix.
Raw-sample baselines were rejected for size and data-retention reasons; quantile grids are small, deterministic, and sufficient for PSI (equal-frequency bins from the deciles) and the KS statistic (ECDF against the grid).
A champion registered before baselines existed cannot be monitored: shift monitors then pass with a loud warning (ADR-10 spirit) until a retrain captures one.

Delayed ground truth runs on its own schedule as `mbt monitor`, declared in the same scoring config (`ground_truth`: label source, join key, maturity duration, builtin metrics, threshold gates).
A prediction run is evaluated once its `scored_at + maturity` lies at or before the monitor run's anchor; the join happens in-process (DuckDB) and realized metrics come from the shared metric engine, so no training adapter or job is involved.
Evaluated runs get a `ground_truth` marker written even when gates fail (evaluated is evaluated; re-alerting belongs to scheduling), while runs with no label coverage or single-class labels are skipped without a marker so they retry when labels arrive.
Realized metrics also log to the tracking adapter, one tracking run per evaluated prediction run, giving MLflow (or any tracker) the production-performance time series.
A central ledger file was rejected: markers living inside each run's directory keep the ledger local to the predictions and safe under concurrent writers.
