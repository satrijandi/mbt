# ADR-29: A dataset reads exactly one relation; the join belongs to dbt

**Status:** accepted

**Supersedes:** ADR-16 (§1 `inputs` form), ADR-22, ADR-25.
**Amends:** ADR-15 (§4 feature derivation contract), ADR-20 (§ scoring inputs).

## Context

mbt drew this line itself and then crossed it.

ADR-22 §5 states the rule: "Expression joins stay out: they would make the
training-set definition unhashable-by-intent and push mbt toward being a query
engine. Anything beyond same-named keys plus an outcome offset belongs in the
upstream gold layer."
`design-history/PRD.md:45` names that layer: "feature engineering lives in the
dataset source (dbt/warehouse) or `hooks.py`".

What actually happened is that `inputs:` kept growing toward the thing the rule
forbids.
ADR-16 added a label spine and feature joins.
ADR-22 added a population spine, per-table join keys, and a calendar label
offset.
ADR-25 added per-table column projection.
The join is now written three times, once per data adapter, in
`packages/mbt-core/src/mbt/adapters/local/data.py`,
`packages/mbt-snowflake/src/mbt_snowflake/sql.py` and
`packages/mbt-spark/src/mbt_spark/data.py`, and every future data adapter
inherits the obligation to write it a fourth.

Meanwhile the deployment mbt targets already has a tool for this.
Data scientists build the panel in dbt on Snowflake: a population table, a
label table, and feature tables that arrive one at a time as new products ship.
dbt is better at joins than mbt is, it is better at join-cardinality tests than
mbt is, and it is where those teams already work.

## Decisions

1. **A dataset declares one relation.**
   `source:` is the only form.
   `inputs:` and its scoring twin are removed, along with the join assembly in
   all three data adapters.
   Whatever assembles the panel is upstream and mbt does not model it.

2. **`columns:` is the panel contract.**
   A dataset may declare the exact column set it expects from its relation.
   An undeclared column fails the build, and so does a declared column that is
   absent.
   This is the load-bearing half of the decision, not a convenience: without it,
   moving the join upstream would move the record of *what the training set is*
   out of the reviewed repo entirely, and a feature arriving in dbt would be
   indistinguishable from a routine data refresh.
   With it, adding `feature_N+1` remains a reviewed mbt diff sitting next to the
   gates, the seed and the feature list, even though the join that produced the
   column lives in another repo.

3. **`sample_key` is required**, and validated non-empty - a required field
   that can still be `[]` would leave the all-columns fallback reachable, which
   is the whole failure it exists to end. That in turn makes the keyless guards
   in the Snowflake and Spark adapters unreachable for datasets, so they are
   gone; a scoring input's `sample_key` stays optional, and that is where the
   runtime guard still bites.
   With `inputs.join_key` gone there is no fallback but the all-columns digest,
   and `_digest_columns` documents what that costs: adding one column to a
   source re-buckets every row, and a measured ten-row case moved three of them
   across an 80/20 boundary.
   That is exactly the add-a-feature-at-a-time loop this ADR is written for.
   Snowflake and Spark already rejected the keyless path; local only warned, so
   a spec that worked on the dev plane failed on the prod plane.

4. **The outcome window stays declarable, but stops executing.**
   ADR-22 §3 argued that pre-aligning label dates upstream "hides the outcome
   window, the exact thing a training-set definition should state", and that
   argument survives even though its mechanism does not.
   `label.horizon` replaces `label.time_offset`: a duration that joins nothing,
   is rendered on the model card, and cross-checks `split.embargo` and
   `ground_truth.maturity` so all three spellings of one number must agree.
   Without it, moving the offset into dbt would silently switch off mbt's only
   leakage-embargo advice, which was gated on `time_offset` being present.

5. **mbt will never accept a SQL query as a dataset.**
   `filters:` (WHERE fragments) and `hooks.py` remain the only escape hatches,
   unchanged.
   Stating this here closes the question rather than reopening it every time the
   next join-shaped requirement arrives.

## What moves, and where it lands

| Concern | Owner |
|---|---|
| Entity resolution, aggregation, feature computation | dbt (already true) |
| Joining population + label + N feature tables | dbt (moves) |
| Column pruning of wide gold tables | dbt (moves; ADR-25's mechanism retired) |
| Label/outcome alignment | dbt (moves; `label.horizon` declares it) |
| Join-cardinality and outcome-coverage tests | dbt tests (moves; see below) |
| Filters, split windows, sampling fraction | mbt |
| Split policy, embargo, leakage scans, gates | mbt |
| Snapshot pinning, `state:modified`, manifests | mbt |

Three capabilities have no mbt-side replacement.
Recording them here rather than letting them vanish quietly:

- **`label_join_coverage`** (ADR-22 §3, F21) compared spine rows against rows
  surviving the inner label join and gated on `min_fraction`.
  mbt cannot see that ratio once the join is upstream.
  Re-home it as a dbt test on the panel model; mbt's `row_count: {min: N}`
  floor remains the catastrophic-drop backstop.
- **`unique: {source: <table>, columns: [...]}`** was framed as the 1:1
  join-cardinality contract - it blamed the offending table before mbt's own
  join could fan the spine out. That framing is gone with the join, but the
  check is not: it asserts that a raw table's key is unique, which is still
  worth asserting from the consumer side even when someone else does the
  joining, and it costs nothing to keep (`SourceAccess` stays for
  `relationships` regardless). It is now "assert the key uniqueness your panel
  depends on, against the table that owes it", and dbt's own `unique` tests are
  the better place for it when you control that repo.
  The post-materialization `unique: {columns: [...]}` form is unchanged and
  still catches a fan-out that reached the panel.
- **`label.time_offset`** as executable join semantics, replaced by the
  declarative `label.horizon` above.

One capability is gained.
Training and scoring each named the whole join separately, and
`examples/showcase/project/scoring/wide_retention_scoring.yml` carried a comment
warning that "scoring must prune identically or the champion would meet columns
it never saw".
Both sides now name one relation, and the champion's recorded feature columns
are enforced at score time instead of by careful copy-paste.

## Why a dynamic table and not a view

The panel must be a physical relation, and this is a correctness requirement
rather than a performance preference.

`SYSTEM$LAST_CHANGE_COMMIT_TIME` accepts a view, but the token it returns is
derived from the DML of the objects the view references, so it is **DDL-blind**:
re-deploying a view to add a column from a table that is already loaded leaves
the token unchanged.
Combined with a spec whose `config_hash` did not move either, a feature addition
would be invisible to both of mbt's hashes, and a pinned manifest would
re-verify clean against a relation that had changed shape.

A dynamic table has its own rows, so both the commit-time token and
`--deep-snapshot`'s `HASH_AGG(*)` reflect what the panel actually contains.
Its refresh is incremental and shared across every model reading it, which is
also the cheaper answer: a view recomputes the whole join on every `mbt build`,
where a dynamic table computes it once per `TARGET_LAG`.
mbt's `freshness: {max_lag: ...}` check guards the lag the same way it guards
any other upstream.

The requirement is therefore that the panel be a PHYSICAL relation, and a
dynamic table is the production-grade way to keep one fresh rather than the only
shape that qualifies.
A `CREATE TABLE AS SELECT` is the same physical relation without auto-refresh,
and it needs no privilege beyond `CREATE TABLE`.
`examples/showcase` uses that form so its warehouse plane runs on a sandbox role
that cannot create a dynamic table, which costs it nothing: its data is static
and every run pins the same anchor.
A deployment whose panel actually changes wants the dynamic table.

The lake and Spark planes take the same shape as a pre-joined table, so one spec
set still runs across all three planes via `--target`.

Belt and braces: `snapshot_id` now folds a column fingerprint into its digest on
every relation kind, so a shape change moves the token even where the data half
does not.

## Rejected

**Keeping `inputs:` as an escape hatch.**
The tempting middle: leave the multi-table form for teams without dbt and for
the laptop plane.
Rejected because two forms mean two code paths in every data adapter forever,
two shapes in every doc, and no clear answer to "which one should I use" beyond
taste.
The file plane does not need it: a pre-joined parquet is a relation like any
other, and the fixtures already read one.

**A `snapshot_mode:` knob per source table.**
There is nothing to switch on.
`SYSTEM$LAST_CHANGE_COMMIT_TIME` accepts tables, views and dynamic tables
alike, and no mode is DDL-aware, so the knob would not fix the hole that
motivated it.

**Making `columns:` a keep-list that prunes, like ADR-25's.**
ADR-25's projection existed for source-side workload reduction: pruned columns
of a wide gold table were never scanned or transferred.
Under a single relation the panel's author already pruned in dbt, so the
scan-reduction rationale is gone and only the declaration is left.
`columns:` therefore asserts and never selects, which keeps it honest about
being a contract.

**Turning the existing `schema:` check into the closed-schema gate.**
`schema:` is a user-declared assertion that named columns exist with named
types.
Redefining an existing check name to also reject unnamed columns would silently
change the meaning of every project's existing spec.

## Consequences

- **The `inputs`-shaped surface is gone**: `DatasetInputs`, `FeatureInput`,
  `LabelInput`, `FeatureEntry`, `ScoringInputs`, the `source`/`inputs` XOR
  validator, the `sample_key_columns` fallback chain, and the join builders in
  three adapters. `DataBuildContext.source_tables` keeps its dict shape with one
  entry; `SourceAccess` stays, because `relationships` still reads a raw source.
- **Config hashes flip twice, not once.** `config_hash` is a full rendered dump,
  so both halves of this change move it: adding `columns` and `label.horizon`,
  and then REMOVING `inputs` (a dropped key changes the dump exactly as an added
  one does). Landing them in separate commits means two `state:modified` cycles
  rather than one, which is the price of keeping each step independently
  revertible. The ADR-7 caveat, exactly as ADR-16 and ADR-22 recorded it for
  their own rollouts. Golden manifests regenerate for both.
- **Snowflake input hashes flip once**, independently, because `snapshot_id`
  now includes a column fingerprint. This moves `input_hash` rather than
  `config_hash`, so it marks Snowflake-backed nodes modified on the first
  compile after upgrade and never again.
- **The training panel and the scoring panel are two dbt objects that must stay
  in lockstep.** mbt catches divergence twice, at build time via `columns:` and
  at score time via the champion's pinned feature columns, but it cannot prevent
  it. The dbt-side convention is that both build from one shared base model,
  with the scoring model differing only by dropping the label and selecting the
  unlabeled cohort. That is a rule for the other repo, written here so the
  contract exists on both sides of the boundary.
- **`examples/showcase` becomes the reference implementation**, materializing a
  Snowflake panel table and an equivalent pre-joined parquet from the same
  generator, and proving the single-relation spec runs unchanged across all
  three planes.
