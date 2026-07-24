# ADR-25: Per-table column projection on multi-table inputs

**Status:** accepted

## Context

The wide multi-table shape (ADR-16/ADR-22) joins gold-layer feature tables
that serve the whole company, not one model's cohort.
In practice those tables are not just tall but wide: hundreds to thousands
of columns, of which a given model consumes a handful, plus bookkeeping
columns (load timestamps, batch ids) that must never become features.

mbt already pushes row-side reduction into the source query - filters,
temporal windows, and entity sampling all execute in the warehouse - but the
join itself was `SELECT *`: every column of every joined table was scanned,
transferred, and materialized.
The only column-side controls were the model's `features.include/exclude`
(which selects AFTER materialization, on the laptop) or a DE-owned
projection view in front of the table (which works, but adds an object to
maintain per model-table pair, and shallow snapshot pinning via
`SYSTEM$LAST_CHANGE_COMMIT_TIME` is table-oriented).

## Decisions

1. **Projection rides the feature entry.** A `{source, using}` feature entry
   (dataset AND scoring inputs; the shared `FeatureInput` schema) gains two
   optional fields: `columns` - a keep-list of payload columns (join columns
   are always kept, redundant mentions dedupe) - and `exclude` - a
   drop-list.
   At most one may be set per entry; excluding a join column is a spec
   error; bare-string entries stay unprojected.

2. **Pruning happens inside the source query on every adapter.** Snowflake
   and the local DuckDB adapter wrap the table in a projecting subquery
   (`(SELECT <keep> FROM t)` / `(SELECT * EXCLUDE (<drop>) FROM t)` - the
   `EXCLUDE` syntax is common to both dialects); Spark applies
   `select`/`drop` on the frame before joining, with an explicit existence
   check on drops because Spark's `drop` silently ignores unknown columns.
   Pruned columns are never scanned into the panel, never transferred, and
   never materialized - the workload reduction happens where the data lives.

3. **Normalization at the spec layer.** `feature_entries` returns
   `FeatureEntry` named tuples (`source`, `using`, `columns`, `exclude`)
   consumed by all join builders, so an adapter cannot see a different
   normalization than its peers.

## Consequences

- The training-features contract can now live in the spec (reviewed YAML)
  without a per-table view layer; views remain the right tool for as-of
  cadence alignment, which is join semantics, not projection.
- This is source-side reduction, complementary to the model's
  `features.include/exclude`: the dataset prunes what should never leave the
  warehouse; the model selects among materialized candidates.
- `FeatureInput` gained fields, so mapping-form feature entries serialize
  two new keys into manifest node configs: their `config_hash` changes once
  on upgrade, and `state:modified` selection flags those datasets/scoring
  nodes for one cycle. Bare-string entries and single-`source:` datasets
  are unaffected (the churn_demo golden manifest is byte-identical).
- A keep-list also future-proofs a panel against upstream tables gaining
  columns: new columns do not enter the dataset (and thus cannot flip its
  config or surprise a model) until the spec asks for them.
