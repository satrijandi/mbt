"""The Snowflake DataAdapter (warehouse-native dataset construction).

Built entirely on official Snowflake surfaces:

- `snowflake.connector.connect()`_ for sessions (password, key-pair, or
  ``authenticator`` flows - config keys pass through to the connector);
- ``SYSTEM$LAST_CHANGE_COMMIT_TIME`` for cheap snapshot pinning at compile
  time, ``HASH_AGG(*)`` for ``--deep-snapshot`` content fingerprints;
- ``MD5_NUMBER_LOWER64`` for deterministic push-down sampling and random
  splits (stable across runs and releases, unlike ``SAMPLE ... REPEATABLE``
  which is only defined for block sampling over fixed physical layout);
- ``Cursor.fetch_arrow_batches()`` to stream results as Arrow into one
  parquet file per split - the standard mbt materialization - without ever
  holding a full table in memory.

``import snowflake.connector`` happens lazily inside methods (ADR-14), so
``mbt parse`` stays fast and the plugin module is cheap to import.

.. _snowflake.connector.connect(): https://docs.snowflake.com/en/developer-guide/python-connector/python-connector-api
"""

import threading
from pathlib import Path
from typing import TYPE_CHECKING, Any

import pyarrow as pa
import pyarrow.parquet as pq

from mbt_adapter_base import (
    DatasetLocator,
    DatasetSpec,
    ScoringInputSpec,
    ScoringOutputSpec,
    retry_with_jitter,
)
from mbt_adapter_base.materialization import (
    MaterializationError,
    MaterializedDatasetHandle,
    combine_snapshots,
    write_materialization_metadata,
)
from mbt_adapter_base.predictions import LocalPredictionStore, resolve_predictions_root
from mbt_adapter_base.protocols import DataBuildContext, SourceTableLike
from mbt_snowflake.sql import (
    SnowflakeSQLError,
    base_relation,
    coverage_queries,
    qualify_table,
    sampling_predicate,
    scoring_query,
    split_queries,
    validate_column,
)

if TYPE_CHECKING:
    from snowflake.connector import SnowflakeConnection

#: Connector kwargs accepted directly in the adapter config; anything else
#: goes under ``connect_args`` (all documented snowflake.connector.connect
#: parameters work there, e.g. private_key_file, session_parameters).
_CONNECT_KEYS = (
    "account",
    "user",
    "password",
    "warehouse",
    "database",
    "schema",
    "role",
    "authenticator",
)


class SnowflakeAdapterError(RuntimeError):
    """Snowflake adapter failures with an actionable message."""

    def __init__(self, message: str, hint: str | None = None) -> None:
        if hint:
            message = f"{message}\n  hint: {hint}"
        super().__init__(message)


def _is_transient(exc: BaseException) -> bool:
    """True for a transient warehouse/network blip a retry can clear - an
    ``OperationalError`` (warehouse resuming, a dropped request, a timeout) - as
    opposed to a deterministic ``ProgrammingError`` (bad SQL, missing object,
    insufficient privilege) that fails identically every time. The two are
    siblings under ``DatabaseError``, so this ``isinstance`` retries the former
    and never the latter (F14, R2-2)."""
    from snowflake.connector.errors import OperationalError

    return isinstance(exc, OperationalError)


class SnowflakeDataAdapter:
    """DataAdapter over Snowflake tables (see package README for config)."""

    name = "snowflake"
    #: Source formats this adapter can read; the compiler rejects a referenced
    #: source declaring any other format before anything runs (F23). Snowflake
    #: sources are warehouse tables, so only the default parquet marker (inert
    #: for identifier-based sources) is accepted.
    supported_source_formats = frozenset({"parquet"})

    def __init__(self, config: dict[str, Any] | None = None) -> None:
        self.config = dict(config or {})
        self.database: str | None = self.config.get("database")
        self.schema: str | None = self.config.get("schema")
        self.normalize_case: bool = bool(self.config.get("normalize_case", True))
        self._connection: SnowflakeConnection | None = None
        #: Serializes lazy connection setup. One adapter instance is shared by
        #: the compiler's snapshot-pinning thread pool (one thread per source
        #: table), so an unguarded check-then-connect opens one connection PER
        #: THREAD - and under `authenticator: externalbrowser`, one browser
        #: window per source table, with every connection but the last leaked.
        self._connect_lock = threading.Lock()

    # -- connection -----------------------------------------------------------

    def _connect(self) -> "SnowflakeConnection":
        # Both checks read through a local: testing ``self._connection``
        # directly would let mypy narrow the attribute to None for the rest of
        # the function and call the re-check unreachable, which is exactly the
        # concurrency this guards against.
        connection = self._connection
        if connection is not None:
            return connection
        with self._connect_lock:
            # Re-check under the lock: a thread that raced us here may have
            # completed the connection (and, for SSO, already paid for the
            # single browser prompt) while we were blocked.
            connection = self._connection
            if connection is not None:
                return connection
            return self._open_connection()

    def _open_connection(self) -> "SnowflakeConnection":
        """Connect and memoize. Callers must hold ``_connect_lock``."""
        import snowflake.connector

        kwargs: dict[str, Any] = {
            key: self.config[key] for key in _CONNECT_KEYS if self.config.get(key)
        }
        kwargs.update(self.config.get("connect_args", {}))
        # SSO must survive mbt's execution model: every job subprocess opens
        # its own connection, so without cached tokens `externalbrowser`
        # would pop one browser window PER NODE. Cache by default (an
        # explicit connect_args value still wins); persisting the cache
        # needs keyring - install `mbt-snowflake[sso]`.
        if (
            str(kwargs.get("authenticator", "")).lower() == "externalbrowser"
            and "client_store_temporary_credential" not in kwargs
        ):
            kwargs["client_store_temporary_credential"] = True
        if "account" not in kwargs:
            raise SnowflakeAdapterError(
                "snowflake adapter config needs at least 'account' and 'user'",
                hint="set them in profiles.yml via env_var(), e.g. "
                "account: \"{{ env_var('SNOWFLAKE_ACCOUNT') }}\"",
            )
        try:
            self._connection = snowflake.connector.connect(**kwargs)
        except Exception as exc:
            raise SnowflakeAdapterError(
                f"could not connect to Snowflake: {exc}",
                hint="check account/user/authenticator config and network access",
            ) from exc
        return self._connection

    def _execute_cursor(self, sql: str) -> Any:
        """A fresh cursor with ``sql`` executed, retried on a transient warehouse
        blip (F14). The caller owns closing the returned cursor. ``execute`` is
        the retryable seam - fetching/streaming *after* it is not, because a
        partially written result must never be re-run - so both data-plane paths
        route their execute through here and keep the read that follows outside
        the retry."""

        def _run() -> Any:
            cursor = self._connect().cursor()
            try:
                cursor.execute(sql)
            except BaseException:
                cursor.close()  # do not leak a cursor on a failed/aborted execute
                raise
            return cursor

        return retry_with_jitter(_run, is_transient=_is_transient)

    def _fetch_one(self, sql: str) -> Any:
        cursor = self._execute_cursor(sql)
        try:
            row = cursor.fetchone()
            return row[0] if row else None
        finally:
            cursor.close()

    # -- snapshots (TSD §8.3) ---------------------------------------------------

    def _table_ref(self, source: SourceTableLike) -> str:
        identifier = source.identifier
        if identifier is None:
            raise SnowflakeAdapterError(
                f"source table {source.name!r} has no 'identifier'",
                hint="Snowflake sources use identifier: [DB.][SCHEMA.]TABLE, not path:",
            )
        try:
            return qualify_table(identifier, self.database, self.schema)
        except SnowflakeSQLError as exc:
            raise SnowflakeAdapterError(str(exc)) from exc

    def snapshot_id(self, source: SourceTableLike, deep: bool = False) -> str:
        """Cheap by default: the table's last DML commit token. Deep: an
        order-independent aggregate hash over every row (scans the table)."""
        import hashlib

        ref = self._table_ref(source)
        safe_ref = ref.replace("'", "''")
        if deep:
            token = self._fetch_one(f"SELECT HASH_AGG(*) FROM {ref}")
        else:
            token = self._fetch_one(f"SELECT SYSTEM$LAST_CHANGE_COMMIT_TIME('{safe_ref}')")
        if token is None:
            raise SnowflakeAdapterError(
                f"could not read a snapshot token for {ref}",
                hint="check the table exists and the role can access it",
            )
        digest = hashlib.sha256(f"{ref}|{token}".encode()).hexdigest()
        return f"sha256:{digest}"

    def _verify_snapshot(self, ctx: DataBuildContext) -> None:
        if ctx.node.snapshot_id is None:
            return
        current = combine_snapshots(
            {
                uid: self.snapshot_id(table, deep=ctx.deep_snapshot)
                for uid, table in ctx.source_tables.items()
            }
        )
        if current != ctx.node.snapshot_id:
            raise SnowflakeAdapterError(
                f"Snowflake data changed under the pinned manifest: snapshot "
                f"{current} != pinned {ctx.node.snapshot_id}",
                hint="recompile to pin the new snapshot, or use Time Travel to restore",
            )

    # -- materialization ----------------------------------------------------------

    def build_dataset(self, spec: DatasetSpec, ctx: DataBuildContext) -> MaterializedDatasetHandle:
        self._verify_snapshot(ctx)
        output_dir = ctx.output_dir
        output_dir.mkdir(parents=True, exist_ok=True)
        for stale in output_dir.glob("*"):
            stale.unlink()

        table_refs = {uid: self._table_ref(t) for uid, t in ctx.source_tables.items()}
        where: list[str] = [f"({f})" for f in spec.filters]
        if not 0.0 < ctx.sample_fraction <= 1.0:
            raise SnowflakeAdapterError(
                f"sample_fraction must be in (0, 1], got {ctx.sample_fraction}"
            )
        if ctx.sample_fraction < 1.0:
            keys = spec.sample_key_columns
            if not keys:
                raise SnowflakeAdapterError(
                    "sampling on Snowflake needs a stable row identity",
                    hint="declare sample_key: [<id columns>] on the dataset "
                    "(or use the multi-table inputs form, whose join_key is used)",
                )
            where.append(sampling_predicate(keys, ctx.sample_fraction))

        try:
            relation, exclude = base_relation(spec, table_refs)
            queries = split_queries(spec, relation, where, ctx.resolved_windows, exclude)
        except SnowflakeSQLError as exc:
            raise SnowflakeAdapterError(str(exc)) from exc

        written: dict[str, int] = {}
        for split, sql in queries.items():
            written[split] = self._stream_query_to_parquet(sql, output_dir / f"{split}.parquet")
        for split, count in written.items():
            if count == 0:
                raise SnowflakeAdapterError(
                    f"split {split!r} materialized 0 rows",
                    hint="check the split windows/fractions and filters against the data",
                )
        # Positive-path row counts on the bus (a plain string the EventSink
        # wraps in a LogMessage); mirrors the local adapter.
        ctx.events.emit(
            f"dataset {ctx.node.unique_id}: materialized {sum(written.values())} rows: "
            + ", ".join(f"{split}={count}" for split, count in sorted(written.items()))
        )

        coverage: dict[str, int] | None = None
        pair = coverage_queries(spec, table_refs)
        if pair is not None:
            # Label-join coverage (F21): spine rows vs rows surviving the inner
            # label join, counted in-warehouse before filters/sampling/windows.
            coverage = {
                "spine_rows": int(self._fetch_one(pair[0]) or 0),
                "matched_rows": int(self._fetch_one(pair[1]) or 0),
            }
            if coverage["spine_rows"] > 0:
                fraction = coverage["matched_rows"] / coverage["spine_rows"]
                ctx.events.emit(
                    f"dataset {ctx.node.unique_id}: label join matched "
                    f"{coverage['matched_rows']} of {coverage['spine_rows']} "
                    f"spine rows ({fraction:.1%})"
                )

        write_materialization_metadata(
            output_dir,
            snapshot_id=ctx.node.snapshot_id,
            dataset=spec.name,
            label_column=spec.label.column,
            time_column=spec.split.time_column,
            windows=ctx.resolved_windows,
            sample_fraction=ctx.sample_fraction,
            row_counts=written,
            label_join_coverage=coverage,
        )
        return MaterializedDatasetHandle(output_dir, adapter=self.name)

    # -- source-level checks (F2/F21) -----------------------------------------

    def count_source_duplicates(self, source: SourceTableLike, columns: list[str]) -> int:
        """Distinct COMPOSITE keys appearing more than once in the raw source
        (pre-join, push-down; only a scalar returns): the 1:1 join-cardinality
        contract behind the ``unique`` check's ``source:`` mode (F2)."""
        ref = self._table_ref(source)
        try:
            cols = ", ".join(validate_column(c) for c in columns)
            not_null = " AND ".join(f"{validate_column(c)} IS NOT NULL" for c in columns)
        except SnowflakeSQLError as exc:
            raise SnowflakeAdapterError(str(exc)) from exc
        value = self._fetch_one(
            f"SELECT COUNT(*) FROM (SELECT 1 FROM {ref} WHERE {not_null} "
            f"GROUP BY {cols} HAVING COUNT(*) > 1)"
        )
        return int(value or 0)

    def read_source_distinct(self, source: SourceTableLike, column: str) -> pa.Table:
        """DISTINCT non-null values of one raw source column as a
        single-column ``value`` table - the parent side of the
        ``relationships`` check (F2/F21). DISTINCT runs in-warehouse; size the
        referenced table like a dimension, not a fact table."""
        ref = self._table_ref(source)
        try:
            col = validate_column(column)
        except SnowflakeSQLError as exc:
            raise SnowflakeAdapterError(str(exc)) from exc
        cursor = self._execute_cursor(
            f"SELECT DISTINCT {col} AS VALUE FROM {ref} WHERE {col} IS NOT NULL"
        )
        try:
            tables = [
                batch if isinstance(batch, pa.Table) else pa.Table.from_batches([batch])
                for batch in cursor.fetch_arrow_batches()
            ]
        finally:
            cursor.close()
        if not tables:
            return pa.table({"value": pa.array([], type=pa.string())})
        return pa.concat_tables(tables).rename_columns(["value"])

    def _stream_query_to_parquet(self, sql: str, out: Path) -> int:
        """Stream Arrow batches into a parquet file; returns the row count."""
        writer: pq.ParquetWriter | None = None
        cursor: Any = None
        rows = 0
        try:
            # execute (retried on a transient blip) stays inside this try so a
            # genuine query error still surfaces as the friendly wrapped message.
            cursor = self._execute_cursor(sql)
            for batch in cursor.fetch_arrow_batches():
                table = batch if isinstance(batch, pa.Table) else pa.Table.from_batches([batch])
                if self.normalize_case:
                    # Unquoted Snowflake identifiers arrive UPPERCASE;
                    # normalize to mbt's lowercase spec conventions.
                    table = table.rename_columns([c.lower() for c in table.column_names])
                if writer is None:
                    writer = pq.ParquetWriter(out, table.schema)
                writer.write_table(table)
                rows += table.num_rows
            if writer is None:
                # zero batches: emit an empty file so the 0-row error upstream
                # names the split instead of a missing file
                empty = pa.table({})
                writer = pq.ParquetWriter(out, empty.schema)
        except SnowflakeAdapterError:
            raise
        except Exception as exc:
            raise SnowflakeAdapterError(
                f"Snowflake query failed: {exc}\n  query: {sql[:500]}",
                hint="column names must be unique across joined tables apart "
                "from the join key(s); filters are raw Snowflake SQL",
            ) from exc
        finally:
            if writer is not None:
                writer.close()
            if cursor is not None:
                cursor.close()
        return rows

    # -- reopening -------------------------------------------------------------------

    def from_locator(self, locator: DatasetLocator) -> MaterializedDatasetHandle:
        """Reopen a materialization; needs no Snowflake connection (jobs run
        without warehouse credentials once data is materialized)."""
        path = Path(locator.uri.removeprefix("file://"))
        try:
            handle = MaterializedDatasetHandle(path, adapter=self.name)
        except MaterializationError as exc:
            raise SnowflakeAdapterError(str(exc)) from exc
        if handle.snapshot_id != locator.snapshot_id:
            raise SnowflakeAdapterError(
                "dataset materialization snapshot mismatch: "
                f"{handle.snapshot_id} != {locator.snapshot_id}",
                hint="the manifest pin and the materialized data disagree; recompile",
            )
        return handle

    # -- batch scoring (contract 1.1, ADR-20/21/23) ----------------------------------

    def build_scoring_input(
        self, spec: ScoringInputSpec, ctx: DataBuildContext
    ) -> MaterializedDatasetHandle:
        """Materialize one unlabeled Snowflake batch as a single ``score`` split.

        Reads the scoring input straight from Snowflake (filters and the
        ``score`` window push down); streams it to ``score.parquet`` via the
        same Arrow path training uses. Zero rows is a warning, not an error - an
        empty nightly batch is legitimate (unlike a training split; ADR-20).

        No snapshot verification: the scoring input (and monitor's arriving
        labels, read through this same path) is expected to change every run, so
        a pinned manifest scores the live data instead of hard-failing on drift
        the way a dataset does (R2-10).
        """
        output_dir = ctx.output_dir
        output_dir.mkdir(parents=True, exist_ok=True)
        for stale in output_dir.glob("*"):
            stale.unlink()

        table_refs = {uid: self._table_ref(t) for uid, t in ctx.source_tables.items()}
        where: list[str] = [f"({f})" for f in spec.filters]
        if not 0.0 < ctx.sample_fraction <= 1.0:
            raise SnowflakeAdapterError(
                f"sample_fraction must be in (0, 1], got {ctx.sample_fraction}"
            )
        if ctx.sample_fraction < 1.0:
            keys = spec.sample_key_columns
            if not keys:
                raise SnowflakeAdapterError(
                    "sampling a Snowflake scoring input needs a stable row identity",
                    hint="declare sample_key on the input (or use the inputs form, "
                    "whose join_key is used)",
                )
            where.append(sampling_predicate(keys, ctx.sample_fraction))

        window = ctx.resolved_windows.get("score") if spec.time_column is not None else None
        try:
            sql = scoring_query(spec, table_refs, where, window)
        except SnowflakeSQLError as exc:
            raise SnowflakeAdapterError(str(exc)) from exc
        count = self._stream_query_to_parquet(sql, output_dir / "score.parquet")

        if count == 0:
            # EventSink wraps a plain string in a LogMessage (adapters cannot
            # import core event models); mirrors the local adapter's warning.
            ctx.events.emit(
                f"scoring input {ctx.node.unique_id}: materialized 0 rows; nothing to score"
            )
        else:
            ctx.events.emit(
                f"scoring input {ctx.node.unique_id}: materialized {count} rows to score"
            )
        write_materialization_metadata(
            output_dir,
            snapshot_id=ctx.node.snapshot_id,
            dataset=ctx.node.name,
            label_column="",  # unlabeled by design (ADR-20)
            time_column=spec.time_column,
            windows=ctx.resolved_windows,
            sample_fraction=ctx.sample_fraction,
            row_counts={"score": count},
        )
        return MaterializedDatasetHandle(output_dir, adapter=self.name)

    def open_predictions(self, output: ScoringOutputSpec) -> LocalPredictionStore:
        """Prediction store for a Snowflake scoring pipeline.

        v1 stages prediction runs as parquet under ``predictions_root`` using the
        shared local layout (ADR-21's sanctioned reuse: "warehouse adapters can
        reuse it for staged exports"). A warehouse-native, Snowflake-table-backed
        store is designed in ADR-23 and gated on live-credential verification.
        ``predictions_root`` (adapter config) is joined with the scoring node's
        ``output.path``; unset, it defaults to an ephemeral ``<tmpdir>/
        mbt-predictions`` (never the project dir, so a scheduled run does not
        write into its checkout - F20).
        """
        root = resolve_predictions_root(self.config.get("predictions_root")) / output.path
        return LocalPredictionStore(root)
