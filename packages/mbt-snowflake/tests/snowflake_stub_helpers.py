"""Shared Snowflake stub harness (unique module name; imported by both the
adapter tests and the wide-example guard test).

The StubConnection runs the adapter's *generated SQL* in DuckDB with small shim
macros for Snowflake-only functions, so joins, sampling predicates, and split
windows are exercised for real - no warehouse account needed. Snapshot queries
return scriptable tokens.
"""

from dataclasses import dataclass, field
from pathlib import Path
from typing import Any

import duckdb
import pyarrow as pa

from mbt_adapter_base import ManifestNode


class StubCursor:
    def __init__(self, connection: "StubConnection") -> None:
        self._connection = connection
        self._scalar: Any = None
        self._table: pa.Table | None = None

    def execute(self, sql: str) -> "StubCursor":
        self._connection.executed.append(sql)
        if "SYSTEM$LAST_CHANGE_COMMIT_TIME" in sql or "HASH_AGG" in sql:
            self._scalar = self._connection.snapshot_token(sql)
            self._table = None
            return self
        self._scalar = None
        self._table = self._connection.run_in_duckdb(sql)
        return self

    def fetchone(self) -> tuple[Any, ...] | None:
        # Snapshot queries answer with the scripted token; anything else (the
        # coverage/source-check COUNTs) answers from the real DuckDB result.
        if self._table is not None:
            if self._table.num_rows == 0:
                return None
            return tuple(column[0].as_py() for column in self._table.columns)
        return (self._scalar,)

    def fetch_arrow_batches(self):
        assert self._table is not None
        # two chunks to exercise the streaming writer
        half = max(1, self._table.num_rows // 2)
        yield self._table.slice(0, half)
        yield self._table.slice(half)

    def close(self) -> None:
        pass


@dataclass
class StubConnection:
    """Emulates enough of a Snowflake session to run the adapter's SQL."""

    tables: dict[str, pa.Table]  # qualified ref -> data (UPPERCASE columns)
    tokens: dict[str, str] = field(default_factory=dict)
    executed: list[str] = field(default_factory=list)

    def cursor(self) -> StubCursor:
        return StubCursor(self)

    def snapshot_token(self, sql: str) -> str:
        for ref, token in self.tokens.items():
            if ref in sql:
                return token
        return "token-default"

    def run_in_duckdb(self, sql: str) -> pa.Table:
        con = duckdb.connect()
        try:
            # The true Snowflake semantics: the unsigned lower 64 bits of the
            # md5 (the last 16 hex chars as a UBIGINT). This is the canonical
            # cross-adapter digest (F19), so the emulation must match exactly -
            # DuckDB's own md5_number uses a different byte interpretation.
            con.execute(
                "CREATE MACRO MD5_NUMBER_LOWER64(s) AS ('0x' || substring(md5(s), 17, 16))::UBIGINT"
            )
            con.execute("CREATE MACRO TO_TIMESTAMP_NTZ(s) AS CAST(s AS TIMESTAMP)")
            for i, (ref, table) in enumerate(self.tables.items()):
                view = f"stub_table_{i}"
                con.register(view, table)
                sql = sql.replace(ref, view)
            sql = sql.replace("AS TIMESTAMP_NTZ)", "AS TIMESTAMP)")
            # con.sql(...).to_arrow_table() (the Relation API) exists at the
            # duckdb>=1.0 floor and is not deprecated; Connection.fetch_arrow_table
            # warns on current duckdb.
            return con.sql(sql).to_arrow_table()
        finally:
            con.close()


@dataclass
class FakeSourceTable:
    name: str
    identifier: str
    path: str | None = None
    format: str = "snowflake"


class CapturingSink:
    def __init__(self) -> None:
        self.messages: list[Any] = []

    def emit(self, event: Any) -> None:
        self.messages.append(event)


@dataclass
class FakeBuildContext:
    node: ManifestNode
    source: FakeSourceTable
    source_tables: dict[str, FakeSourceTable]
    resolved_windows: dict[str, tuple[str, str]]
    sample_fraction: float
    deep_snapshot: bool
    output_dir: Path
    # The real BuildContext always carries a live sink; default to a capturing
    # one so success-path emits (row counts) have somewhere to go.
    events: Any = field(default_factory=CapturingSink)
