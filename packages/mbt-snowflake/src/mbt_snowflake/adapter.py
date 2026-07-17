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

from pathlib import Path
from typing import TYPE_CHECKING, Any

import pyarrow as pa
import pyarrow.parquet as pq

from mbt_adapter_base import DatasetLocator, DatasetSpec
from mbt_adapter_base.materialization import (
    MaterializationError,
    MaterializedDatasetHandle,
    combine_snapshots,
    write_materialization_metadata,
)
from mbt_adapter_base.protocols import DataBuildContext, SourceTableLike
from mbt_snowflake.sql import (
    SnowflakeSQLError,
    base_relation,
    qualify_table,
    sampling_predicate,
    split_queries,
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


class SnowflakeDataAdapter:
    """DataAdapter over Snowflake tables (see package README for config)."""

    name = "snowflake"

    def __init__(self, config: dict[str, Any] | None = None) -> None:
        self.config = dict(config or {})
        self.database: str | None = self.config.get("database")
        self.schema: str | None = self.config.get("schema")
        self.normalize_case: bool = bool(self.config.get("normalize_case", True))
        self._connection: SnowflakeConnection | None = None

    # -- connection -----------------------------------------------------------

    def _connect(self) -> "SnowflakeConnection":
        if self._connection is not None:
            return self._connection
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

    def _fetch_one(self, sql: str) -> Any:
        cursor = self._connect().cursor()
        try:
            cursor.execute(sql)
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

        write_materialization_metadata(
            output_dir,
            snapshot_id=ctx.node.snapshot_id,
            dataset=spec.name,
            label_column=spec.label.column,
            time_column=spec.split.time_column,
            windows=ctx.resolved_windows,
            sample_fraction=ctx.sample_fraction,
            row_counts=written,
        )
        return MaterializedDatasetHandle(output_dir, adapter=self.name)

    def _stream_query_to_parquet(self, sql: str, out: Path) -> int:
        """Stream Arrow batches into a parquet file; returns the row count."""
        cursor = self._connect().cursor()
        writer: pq.ParquetWriter | None = None
        rows = 0
        try:
            cursor.execute(sql)
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
