# ADR-16: Multi-table dataset inputs, key-based sampling, warehouse adapters

**Status:** accepted

## Context

Real training sets are rarely one table: teams keep feature tables and a
label table joined by an entity key. And at warehouse scale (millions of
rows, thousands of columns), sampling must happen *inside the source query*
and be reproducible - fetching everything and sampling client-side defeats
the purpose, and `SAMPLE ... REPEATABLE` semantics depend on physical
layout, which reclustering silently changes.

## Decisions

1. **`inputs` dataset form.** A dataset declares either a single `source`
   or `inputs: {label, features[], join_key, join}`. The label table is the
   spine (it defines which examples exist); feature tables LEFT JOIN onto it
   by the key (missing features arrive as NULLs; tree adapters handle
   those natively). Column names must be unique across tables apart from
   the join key(s). Every referenced table becomes a DAG edge, and the
   dataset's pinned snapshot is the combination of all its tables'
   snapshots - any input changing marks it `state:modified` (ADR-4).

2. **Key-based deterministic sampling.** `sample_key` (defaulting to
   `inputs.join_key`) names the stable row identity. Sampling keeps rows
   where `md5_number(key) % 1e6 < fraction * 1e6`, pushed down into the
   source query. Properties: same fraction always keeps the same rows;
   smaller fractions are subsets of larger ones; cost is one hash over a
   few key columns instead of digesting every column (the old fallback
   remains for keyless single-table datasets). Seeded random splits hash
   the same key with the split seed as salt.

3. **Shared materialization format.** The parquet-per-split directory
   (metadata + `_SUCCESS`) moved into `mbt-adapter-base` as
   `MaterializedDatasetHandle`, so every DataAdapter - local DuckDB,
   Snowflake, future warehouses - produces and reopens the same layout.
   Training jobs never need warehouse credentials: `from_locator` reads
   the local materialization only.

4. **Snowflake snapshotting.** Compile pins
   `SYSTEM$LAST_CHANGE_COMMIT_TIME(table)` (cheap metadata call, changes on
   any DML); `--deep-snapshot` uses `HASH_AGG(*)` (order-independent
   content fingerprint, scans the table). `MD5_NUMBER_LOWER64` implements
   the sampling hash - deterministic across runs and releases, unlike
   HASH() (not guaranteed stable) or block sampling.

## Consequences

- `DataBuildContext` gained `source_tables`; `DatasetSpec` gained
  `inputs`/`sample_key` (full-dump hashing means this schema addition
  flipped config hashes once - the ADR-7 caveat in action).
- Random splits on warehouses use hash *buckets*: split proportions are
  approximate (each row lands independently), which avoids a full-table
  window sort. The local adapter keeps exact percent_rank fractions.
- Adapter emulation tests run the generated Snowflake SQL through DuckDB
  with shim macros, so push-down logic is exercised without an account.
