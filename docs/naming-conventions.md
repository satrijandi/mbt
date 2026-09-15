# Naming conventions for temporal and entity columns

mbt never hardcodes column names: `split.time_column`, `sample_key`, `ground_truth.join_key`, and friends are all declared per project.
That freedom is exactly why a project needs one written convention, because a wrong or ambiguous date column is how temporal leakage happens.
This page is the convention mbt projects follow; the [showcase](showcase.md)'s batch-monthly wide cadence is its reference implementation.

## The glossary

| Name | Kind | Meaning |
|---|---|---|
| `customer_id`, `safe_id`, `user_id`, ... | entity id | Stable entity identifiers, always suffixed `_id`. A population table may carry several and act as the crosswalk between them, which the panel's join needs. |
| `execution_date` | date | The orchestrator's logical date - what Airflow calls the logical (execution) date of the run. Not a wall-clock time: a backfill run for January 1st has `execution_date` January 1st whenever it actually executes. |
| `target_date` | date | The date a batch job is producing output FOR. Equal to `execution_date` in normal operation; the distinct name exists so job code reads unambiguously. |
| `inference_date` | date | The prediction as-of date: the date a scored row's prediction refers to. Usually `target_date`. Inference "time" is `inference_date` at 00:00 local (UTC+7). This is the key the upstream panel is joined on - population to labels to every feature history - and the `split.time_column` of training datasets. |
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
The feature producer aligns each row to the `inference_date` it serves.
The balances a row describes are as of the previous day, but that is metadata, recorded once in the population's informational `as_of_date` column rather than as a second join key every consumer has to reason about.
The population table carries the entity crosswalk (`customer_id` to `safe_id`) plus the `as_of_date` and `loaded_at_time` lineage and audit columns.

Since ADR-29 that join happens upstream - in a dbt model, a warehouse table, or the showcase's generator - and mbt reads the panel it produces.
The rules below still matter to mbt, because the panel's columns are what `split.time_column`, `sample_key`, and `features.exclude` point at.

## Rules that keep the convention leakage-safe

- **Label tables are keyed by `inference_date`**, the cohort's own prediction date, and a row appears only once its outcome window has closed - the gold-layer label contract.
  A raw upstream feed keyed by observation date is realigned inside the panel's join.
  Encode that offset in exactly one place, and state it in the dataset's `label.horizon`, so mbt can check `split.embargo` and the scoring pipeline's `ground_truth.maturity` against it.
- **Keys, lineage, and audit columns never become features.**
  mbt drops only `split.time_column` (`inference_date`) automatically at train time.
  Everything else the population carries - the entity ids, `as_of_date`, `loaded_at_time` - must be listed in the model's `features.exclude`, or a trainer will rightly refuse the raw timestamp.
  `no_future_columns` backstops any timestamp that leaks past its split window.
- **Joined gold tables need disjoint non-key column names.**
  The panel's join merges the key columns and passes everything else through, so two feature tables both carrying `loaded_at_time` would collide.
  Keep shared-name audit columns out of tables that get joined together, or drop them inside the join; the showcase's panel drops each feature table's `etl_loaded_at` in its own subquery and carries `loaded_at_time` from the population only.
- **The orchestrator hands mbt its logical date.**
  A scheduled DAG passes `execution_date` as `mbt build --anchor` or `mbt score --anchor`; every window (`train:`, `test:`, a scoring `window:`) resolves against that anchor, which is what makes backfills and reruns reproducible.

## Adoption status

- The wide batch-monthly cadence (`examples/showcase`, SHOW-19/SHOW-20) implements the convention in full: one uniform `inference_date` join key across all five gold tables, the entity crosswalk plus the `as_of_date`/`loaded_at_time` lineage columns on the population, matured labels on `inference_date`, and DAGs that pass the logical date as the anchor.
- The showcase's monthly DuckDB cadence (SHOW-17) uses `inference_date` as its time column.
- The showcase's daily cadence reuses `tests/fixtures/churn_demo`'s tables, a fixture project that predates this convention and is pinned by golden-manifest tests; its `snapshot_date` is an `inference_date` in this vocabulary.
