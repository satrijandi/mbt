# ADR-15: v0 contract refinements beyond the original TSD sketch

**Status:** accepted

Refinements made while implementing, recorded per the global DoD:

1. **Contracts live in mbt-adapter-base from day one**; mbt.contracts
   re-exports. Skips a painful mid-project extraction (S7-06 done early).
2. **TrainingJob carries more context**: unrendered data/tracking
   AdapterRefs (the job re-resolves env_var() itself, TSD §18), resolved
   MetricSpec list (adapters compute exactly what core compares), dataset
   windows (implicit validation carve), non-secret vars, and a mode flag
   (train | evaluate) so `mbt evaluate`/`mbt test` reuse the same seam.
3. **TrainingAdapter gains predict()** returning the split plus a
   `prediction` column: hook metrics need predictions, and v1 `mbt score`
   reuses it. `load()` takes the ArtifactStore so artifact fetching stays
   out of adapters.
4. **Feature derivation contract**: the table an adapter reads contains
   selected features + target + declared slice columns; features are
   everything except target and slice columns; the split time column never
   reaches adapters. Non-numeric features are an actionable error -
   exclude them or encode via hooks (no hidden encoding in v0).
5. **Polars dropped from v0 dependencies**: DuckDB + PyArrow cover local
   dataset builds; one fewer heavyweight dependency. Revisit if adapters
   want Polars-native interchange.
6. **Events go to stderr; stdout carries command data** so `--output json`
   is machine-parseable; job subprocesses emit events on stdout, which the
   coordinator forwards (they own no user-facing stdout).
