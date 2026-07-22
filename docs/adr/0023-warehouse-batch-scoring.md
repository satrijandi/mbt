# ADR-23: Warehouse-native batch scoring (Snowflake), staged vs native prediction stores

**Status:** accepted; the SQL-read half is shipped and hermetically verified, the native prediction store is designed but pending live-credential verification (see "Verification").

ADR-20/21 defined the batch-scoring contract (contract 1.1: `build_scoring_input` and `open_predictions`) but implemented it only for the local (path/parquet) adapter.
A Snowflake-native team could therefore train, gate, register, and promote entirely in-warehouse, yet `mbt score` and `mbt monitor` refused to run against a Snowflake data adapter: `require_scoring_capability` fails the `hasattr` probe before any job runs.
That refusal is the single place the "dbt for ML" value proposition does not close for the warehouse-native audience the tool targets.
This ADR extends contract 1.1 to Snowflake.

## `build_scoring_input`: read the batch straight from the warehouse

The Snowflake adapter now materializes one unlabeled `score` split by generating label-free scoring SQL and streaming the result to `score.parquet` through the same Arrow path training uses.
The SQL is built by pure functions (`scoring_relation`, `scoring_query`) that mirror the training seam (`base_relation`/`split_queries`) but never join a label - a scoring input is unlabeled by design (ADR-20).
Dataset `filters`, the resolved `score` window on `time_column` (the same half-open temporal predicate as training splits), and dev `sample_fraction` all push down into the warehouse query.
Zero rows is a warning, not an error: an empty nightly batch is legitimate, unlike an empty training split.
This half is the high-value, fully verifiable part - it reads production features from Snowflake with no local export step.

## `open_predictions`: staged reuse now, native store designed

ADR-21 already anticipated this: "warehouse adapters can reuse [the local layout] for staged exports or implement the `PredictionStore` protocol natively."
We take both, in sequence.

**v1 (shipped): staged export.**
`open_predictions` returns a `LocalPredictionStore` rooted at `predictions_root / output.path` (a local or mounted-stage directory, `predictions_root` from adapter config; unset, it defaults to an ephemeral `<tmpdir>/mbt-predictions`, never the project dir, so a scheduled run does not write into its checkout - F20).
This reuses the compliance-tested local layout verbatim - per-run directories, idempotent-by-`run_key` writes, `_SUCCESS` and ground-truth markers - so `mbt score`/`mbt monitor` run end to end against a Snowflake input with zero new store code to verify.
Predictions land as parquet next to the run, not back in Snowflake; that is the honest limit of v1.

**v2 (designed here, not yet shipped): native Snowflake tables.**
A warehouse-native store materializes predictions back into the warehouse, which is dbt's defining move.
The design preserves every ADR-21 invariant in SQL:
one predictions table keyed by `run_key` (idempotent rewrite via `DELETE WHERE run_key = ?` then `INSERT`, or `MERGE`), a runs-metadata table holding the `PredictionRunInfo` columns, and a markers table for the ground-truth ledger (a row per `(run_key, marker_name)`, so rewriting a run clears its markers exactly as the local layout does).
`list_runs`/`read`/`read_marker`/`write_marker` become ordinary `SELECT`/`INSERT` statements, all expressible in the same pure-function-plus-DuckDB-emulation style the read half already uses.

## Why staged-first, not native-first

The read half (`build_scoring_input`) is deterministic SQL generation and is verified exactly like the training SQL: the generated statements execute in DuckDB with shim macros for Snowflake-only functions, so joins, windows, and sampling are exercised for real with no warehouse account.
A native, stateful table store is different: its correctness against real Snowflake depends on transaction/`MERGE`/`DELETE` semantics, the Arrow-to-table write path (`write_pandas`/`executemany`), and concurrent-writer behavior that a DuckDB emulation approximates but does not prove.
Shipping that unverified onto a core production path would violate the repo's "reproduce and verify end to end before calling it done" standard.
So v1 ships the verifiable read half plus the sanctioned staged store, and v2's native store is specified and waits for the first credentialed `live_snowflake` run to verify it.

## Verification

The scoring-SQL generation and `build_scoring_input` are covered hermetically (DuckDB execution of the generated SQL; the `mbt-snowflake` package stays at 100% line coverage), and `open_predictions` round-trips through the compliance-tested local store.
What is **not** yet verified: end-to-end `mbt score`/`mbt monitor` against a live Snowflake account (the `live_snowflake` tier has never had a credentialed run), and therefore the native (v2) prediction store, which is designed above but deliberately unimplemented until that verification is possible.

## Spark

Spark warehouse scoring has the same shape (a label-free scoring DataFrame, predictions staged or written to a table) and is deferred to a follow-up so it can be designed against Spark's DataFrame-native write path rather than transliterated from the SQL adapter.

## Consequences

Snowflake teams can now run `mbt score` and `mbt monitor` (features read from Snowflake, predictions staged), and the contract-1.1 refusal no longer fires for the Snowflake adapter.
A later ADR/PR promotes the native table store (v2) once a live Snowflake run verifies it, at which point `predictions_root` becomes optional and `output.path` names a warehouse table.
