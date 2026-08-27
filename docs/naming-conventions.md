# Naming conventions for temporal and entity columns

mbt never hardcodes column names: `time_column`, `using:`, `sample_key`, and friends are all declared per project.
That freedom is exactly why a project needs one written convention, because a wrong or ambiguous date column is how temporal leakage happens.
This page is the convention mbt projects follow; the [showcase](showcase.md)'s batch-monthly wide cadence is its reference implementation.

## The glossary

| Name | Kind | Meaning |
|---|---|---|
| `customer_id`, `safe_id`, `user_id`, ... | entity id | Stable entity identifiers, always suffixed `_id`. A population/spine table may carry several and act as the crosswalk between them (ADR-22). |
| `execution_date` | date | The orchestrator's logical date - what Airflow calls the logical (execution) date of the run. Not a wall-clock time: a backfill run for January 1st has `execution_date` January 1st whenever it actually executes. |
| `target_date` | date | The date a batch job is producing output FOR. Equal to `execution_date` in normal operation; the distinct name exists so job code reads unambiguously. |
| `inference_date` | date | The prediction as-of date: the date a scored row's prediction refers to. Usually `target_date`. Inference "time" is `inference_date` at 00:00 local (UTC+7). This is the join key between the population spine and the label table, and the `split.time_column` of training datasets. |
| `as_of_date` | date | The data-state date: the date a row's balances/aggregates describe. Usually `execution_date - 1 day`, because a batch pipeline running on the logical date can only have complete data through the end of the previous day. A lineage column, never a join key: joins use `inference_date`. |
| `loaded_at_time` | timestamp | Lakehouse audit column: when the row landed in the lake. An ingest-layer concern, never a feature. |

Suffix rule: `_date` columns are calendar dates (midnight, no meaningful time part); `_time` columns are timestamps.
Timezone rule: all dates and timestamps are stored timezone-naive in Jakarta local time (UTC+7); mbt anchors and window expressions are interpreted in that same local convention (the `Z` in an anchor string like `2026-06-30T00:00:00Z` is mbt's canonical timestamp form, not a UTC wall-clock claim).

## How the dates relate in a batch scoring run

```text
execution_date = target_date = inference_date        (the logical date, 00:00 local)
as_of_date     = inference_date - 1 day              (the state the features describe)
label for inference_date matures one outcome-window later
loaded_at_time ~ shortly after 00:00 on execution_date (ingest audit, not a feature)
```

Every joinable gold table shares ONE join key: `inference_date`.
The feature producer aligns each row to the `inference_date` it serves (the balances it describes are as of the previous day - that is metadata, recorded once in the spine's informational `as_of_date` column, not a second join key for every consumer to reason about).
Feature tables and the label all join on `inference_date`; `split.time_column` is `inference_date`; the population spine carries the entity crosswalk (`customer_id` to `safe_id`) plus the `as_of_date` and `loaded_at_time` lineage/audit columns.

## Rules that keep the convention leakage-safe in mbt

- **Label tables are keyed by `inference_date`** (the cohort's own prediction date) and rows appear only once the outcome window has closed - the gold-layer label contract.
  A raw upstream feed keyed by observation date is joined with `time_offset` instead (ADR-22); encode the offset in exactly one place.
- **Keys, lineage, and audit columns never become features.** mbt auto-drops only `split.time_column` (`inference_date`) at train time; everything else the spine carries - the entity ids, `as_of_date`, `loaded_at_time` - must be listed in `features.exclude`, or a trainer will rightly refuse the raw timestamp.
- **Audit columns never become features.** Everything in a joined table lands in the training panel, so `loaded_at_time` belongs in the model's `features.exclude` list (the DS ignored-columns contract, honored by the showcase's selection funnel); mbt's `no_future_columns` check backstops any timestamp that leaks past its split window.
- **Joined gold tables need disjoint non-key column names.** mbt's multi-table joins merge key columns and pass everything else through, so two feature tables both carrying `loaded_at_time` would collide in the panel; keep shared-name audit columns out of tables that get joined together (the showcase carries `loaded_at_time` on the spine only).
- **The orchestrator hands mbt its logical date.** A scheduled DAG passes `execution_date` as `mbt build/score --anchor`; every window (`train:`, `test:`, scoring `window:`) resolves against that anchor, which is what makes backfills and reruns reproducible.

## Adoption status

- The wide batch-monthly cadence (`examples/showcase`, SHOW-19/SHOW-20) implements the convention in full: one uniform `inference_date` join key across all five tables, the entity crosswalk plus the `as_of_date`/`loaded_at_time` lineage columns on the spine, matured labels on `inference_date`, and DAGs that pass the logical date as the anchor.
- The showcase's monthly DuckDB cadence (SHOW-17) uses `inference_date` as its time column.
- The showcase's daily cadence reuses `tests/fixtures/churn_demo`'s tables, a fixture project that predates this convention and is pinned by golden-manifest tests; its `snapshot_date` is an `inference_date` in this vocabulary.
