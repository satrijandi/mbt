# ADR-22: Population spines, per-table join keys, and label time offsets

**Status:** accepted

## Context

ADR-16's `inputs` form covers the simple shape - a label table as the spine,
every feature table joining by one shared key.
Real feature stores are messier, and the monthly-churn reference scenario
(showcase) makes the gap concrete:

- The examples are defined by a **population** table (one row per
  `customer_id` per month-start `snapshot_date`), which also carries the
  entity crosswalk (`customer_id` to `safe_id`).
- Feature tables join by **different keys**: demographic and login history
  by `(customer_id, snapshot_date)`, transaction history by
  `(safe_id, snapshot_date)` - where `safe_id` only becomes available
  *after* the population join.
- The label is observed one month **after** the prediction snapshot: the
  label row for the 2026-05-01 population snapshot lives at
  `snapshot_date = 2026-06-01`.
  An equi-join cannot express that, and pre-aligning dates upstream hides
  the outcome window - the exact thing a training-set definition should
  state.

## Decisions

1. **Optional `population:` spine.** When present, the population table
   defines which examples exist; the label becomes a joined table like the
   features. Without it, the label table stays the spine (ADR-16 form,
   unchanged). Scoring inputs already model this concept as `spine`; a
   population-spine dataset and its scoring input now declare the same
   table shape, minus the label.

2. **Per-table `on:` join columns.** `features` entries are either a bare
   `source()` string (joins by the dataset-level `join_key`, as before) or
   a mapping `{source, on}`. Joins apply **in declaration order** onto the
   accumulated relation, so a key introduced by an earlier join (the
   population's `safe_id`) is usable by a later one. `on` columns are
   USING-style: same name on both sides, merged in the output.
   `join_key` becomes optional when every joined table declares `on`.

3. **Label joins with an optional calendar `time_offset`.** With a
   population spine, `label` may be a mapping `{source, on, time_offset}`.
   `time_offset` shifts the spine's `split.time_column` when matching the
   label's same-named column: `label.ts = spine.ts + offset`. The grammar
   adds a calendar unit to the duration syntax (`1mo`, `-2mo`) alongside
   `d/w/h`, because "one month later" is a calendar statement, not a fixed
   number of days; adapters render it as native interval arithmetic
   (DuckDB/Spark `INTERVAL n MONTH`, Snowflake `DATEADD`). The label's
   join columns are projected away after the join (the spine's prediction
   date is the one true `time_column`); the label join is always **inner**
   - an example without an observed outcome is not a training example.
   Rows the population defines but the label filters out are the outcome
   coverage: every population-spine build now records it as
   `label_join_coverage` (`spine_rows` vs `matched_rows`, counted before
   filters/sampling/windows), reports it on the event bus, and the
   `label_join_coverage: {min_fraction: ...}` check turns a quiet partial
   drop into an exit-2 failure (F21). An inner join to a table that is
   **unique on the join key** can only shrink the spine, never invent rows;
   but a non-unique label or feature table fans the spine out (an inner join
   to a duplicated key multiplies rows), silently over-weighting that entity.
   mbt does not dedup for you - declare a `unique` dataset check on the join
   key, or its pre-join form `unique: {source: <group.table>, columns: [...]}`
   (the 1:1 join-cardinality contract, checked against the raw table before
   the join can fan anything out), to guard against it (F2).

4. **Sampling identity is unchanged.** `sample_key` still names the stable
   row identity; its default falls back to `join_key`, then to the label's
   `on` columns. The showcase samples by `customer_id` alone - panel
   sampling that keeps every snapshot of a kept customer, which is what
   makes dev-fraction models comparable to full-data models.

5. **No arbitrary SQL in joins.** `on` lists and one calendar offset are
   the whole extension. Expression joins (`ON a.x = f(b.y)`) stay out:
   they would make the training-set definition unhashable-by-intent and
   push mbt toward being a query engine. Anything beyond same-named keys
   plus an outcome offset belongs in the upstream gold layer.

## Consequences

- `DatasetInputs` gained `population`, per-entry `on`, and the label
  mapping form; `ScoringInputs` gained per-entry `on`. Full-dump hashing
  means this schema addition flips config hashes once (the ADR-7 caveat,
  same as ADR-16's own rollout) - golden manifests regenerate.
- All three data adapters (local DuckDB, Spark, Snowflake) implement the
  same join assembly; the Snowflake SQL is exercised through the DuckDB
  emulation shims, the Spark path through the JVM e2e tier.
- The offset join cannot use USING, so the label side is wrapped in a
  rename-project subquery (`SELECT * RENAME (...)`) and its join columns
  are excluded from the output - identical technique in DuckDB and
  Snowflake; the Spark adapter drops the renamed columns after an
  expression join.
- Column names must be unique across joined tables apart from each table's
  own `using` columns; collisions still surface as engine errors wrapped
  in the existing "column names must be unique" hint.
- Building the showcase scenario surfaced a latent cache bug:
  `materialization_key` ignored `sample_fraction`, so a sampled build
  could silently reuse a full build's rows (and vice versa). A non-default
  fraction now partitions its own materialization key; fraction-1.0 keys
  are unchanged, so existing caches stay valid.
