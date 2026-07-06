"""Snowflake adapter tests over a stubbed connection.

The stub executes the adapter's *generated SQL* in DuckDB (with small shim
macros for Snowflake-only functions), so joins, sampling predicates, and
split windows are exercised for real - no warehouse account needed.
Snapshot queries return scriptable tokens.
"""

import subprocess
import sys
from dataclasses import dataclass, field
from datetime import datetime, timedelta
from pathlib import Path
from typing import Any

import duckdb
import pyarrow as pa
import pytest
from mbt_snowflake.adapter import SnowflakeAdapterError, SnowflakeDataAdapter
from mbt_snowflake.sql import (
    SnowflakeSQLError,
    qualify_table,
    sampling_predicate,
    split_queries,
)

from mbt_adapter_base import DatasetSpec, ManifestNode
from mbt_adapter_base.materialization import combine_snapshots

ANCHOR = datetime(2026, 7, 1)
WINDOWS = {
    "train": ("2026-01-02T00:00:00Z", "2026-06-03T00:00:00Z"),
    "test": ("2026-06-03T00:00:00Z", "2026-07-01T00:00:00Z"),
}


# -- the stub connection -------------------------------------------------------


class StubCursor:
    def __init__(self, connection: "StubConnection") -> None:
        self._connection = connection
        self._scalar: Any = None
        self._table: pa.Table | None = None

    def execute(self, sql: str) -> "StubCursor":
        self._connection.executed.append(sql)
        if "SYSTEM$LAST_CHANGE_COMMIT_TIME" in sql or "HASH_AGG" in sql:
            self._scalar = self._connection.snapshot_token(sql)
            return self
        self._table = self._connection.run_in_duckdb(sql)
        return self

    def fetchone(self) -> tuple[Any] | None:
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
            con.execute(
                "CREATE MACRO MD5_NUMBER_LOWER64(s) AS "
                "(md5_number(s) % 9223372036854775807)::BIGINT"
            )
            con.execute("CREATE MACRO TO_TIMESTAMP_NTZ(s) AS CAST(s AS TIMESTAMP)")
            for i, (ref, table) in enumerate(self.tables.items()):
                view = f"stub_table_{i}"
                con.register(view, table)
                sql = sql.replace(ref, view)
            sql = sql.replace("AS TIMESTAMP_NTZ)", "AS TIMESTAMP)")
            return con.execute(sql).to_arrow_table()
        finally:
            con.close()


# -- fixtures ---------------------------------------------------------------------


def _make_tables(n: int = 200) -> dict[str, pa.Table]:
    dates = [ANCHOR - timedelta(days=(i * 179) % 180 + 1) for i in range(n)]
    labels = pa.table(
        {
            "CUSTOMER_ID": list(range(n)),
            "SNAPSHOT_DATE": dates,
            "CHURNED_90D": [1 if (i * 31) % 100 < 25 else 0 for i in range(n)],
        }
    )
    usage = pa.table(
        {
            "CUSTOMER_ID": list(range(n)),
            "SNAPSHOT_DATE": dates,
            "MONTHLY_USAGE": [float(i % 300) for i in range(n)],
        }
    )
    return {
        "ANALYTICS.GOLD.CHURN_LABELS": labels,
        "ANALYTICS.GOLD.USAGE_FEATURES": usage,
    }


@dataclass
class FakeSourceTable:
    name: str
    identifier: str
    path: str | None = None
    format: str = "snowflake"


@dataclass
class FakeBuildContext:
    node: ManifestNode
    source: FakeSourceTable
    source_tables: dict[str, FakeSourceTable]
    resolved_windows: dict[str, tuple[str, str]]
    sample_fraction: float
    deep_snapshot: bool
    output_dir: Path
    events: Any = None


LABEL_UID = "source.p.snowflake.churn_labels"
USAGE_UID = "source.p.snowflake.usage_features"


def _spec(**overrides: Any) -> DatasetSpec:
    base: dict[str, Any] = {
        "name": "churn_training_set",
        "inputs": {
            "label": LABEL_UID,
            "features": [USAGE_UID],
            "join_key": ["customer_id", "snapshot_date"],
        },
        "label": {"column": "churned_90d"},
        "sample_key": ["customer_id"],
        "split": {
            "strategy": "temporal",
            "time_column": "snapshot_date",
            "train": "-180d:-28d",
            "test": "-28d:now",
        },
    }
    base.update(overrides)
    return DatasetSpec.model_validate(base)


def _adapter(stub: StubConnection) -> SnowflakeDataAdapter:
    adapter = SnowflakeDataAdapter({"database": "ANALYTICS", "schema": "GOLD"})
    adapter._connection = stub  # type: ignore[assignment]
    return adapter


def _sources() -> dict[str, FakeSourceTable]:
    return {
        LABEL_UID: FakeSourceTable(name="churn_labels", identifier="CHURN_LABELS"),
        USAGE_UID: FakeSourceTable(name="usage_features", identifier="USAGE_FEATURES"),
    }


def _ctx(
    tmp_path: Path,
    spec: DatasetSpec,
    adapter: SnowflakeDataAdapter,
    sample_fraction: float = 1.0,
    snapshot: str | None = "auto",
) -> FakeBuildContext:
    sources = _sources()
    pinned = None
    if snapshot == "auto":
        pinned = combine_snapshots({uid: adapter.snapshot_id(t) for uid, t in sources.items()})
    elif snapshot is not None:
        pinned = snapshot
    node = ManifestNode(
        unique_id="dataset.p.churn_training_set",
        resource_type="dataset",
        name=spec.name,
        path="datasets/churn_training_set.yml",
        config={},
        snapshot_id=pinned,
    )
    return FakeBuildContext(
        node=node,
        source=sources[LABEL_UID],
        source_tables=sources,
        resolved_windows=WINDOWS,
        sample_fraction=sample_fraction,
        deep_snapshot=False,
        output_dir=tmp_path / "mat",
    )


# -- SQL unit tests ------------------------------------------------------------------


def test_qualify_table_variants() -> None:
    assert qualify_table("T", "DB", "S") == "DB.S.T"
    assert qualify_table("S2.T", "DB", "S") == "DB.S2.T"
    assert qualify_table("DB2.S2.T", "DB", "S") == "DB2.S2.T"
    with pytest.raises(SnowflakeSQLError, match="needs database and schema"):
        qualify_table("T", None, None)
    with pytest.raises(SnowflakeSQLError, match="invalid table identifier"):
        qualify_table("T; DROP TABLE users", "DB", "S")


def test_sampling_predicate_shape_and_injection_guard() -> None:
    predicate = sampling_predicate(["customer_id"], 0.1)
    assert "MD5_NUMBER_LOWER64" in predicate and "< 100000" in predicate
    with pytest.raises(SnowflakeSQLError, match="invalid column identifier"):
        sampling_predicate(["cid; DROP"], 0.1)


def test_temporal_split_queries_push_everything_down() -> None:
    spec = _spec(filters=["is_active = true"])
    queries = split_queries(
        spec, "ANALYTICS.GOLD.CHURN_LABELS AS mbt_label", ["(is_active = true)"], WINDOWS
    )
    assert set(queries) == {"train", "test"}
    assert "TO_TIMESTAMP_NTZ('2026-06-03 00:00:00')" in queries["train"]
    assert "(is_active = true)" in queries["test"]


def test_random_split_requires_a_key() -> None:
    spec = _spec(
        sample_key=None,
        inputs=None,
        source=LABEL_UID,
        split={"strategy": "random", "train": "0.8", "test": "0.2", "seed": 7},
    )
    with pytest.raises(SnowflakeSQLError, match="sample_key"):
        split_queries(spec, "T", [], {})


def test_random_split_buckets_cover_fractions() -> None:
    spec = _spec(
        inputs=None,
        source=LABEL_UID,
        sample_key=["customer_id"],
        split={"strategy": "random", "train": "0.7", "test": "0.3", "seed": 7},
    )
    queries = split_queries(spec, "T", [], {})
    assert ">= 0" in queries["train"] and "< 700000" in queries["train"]
    assert ">= 700000" in queries["test"] and "< 1000000" in queries["test"]


# -- end-to-end over the stub ----------------------------------------------------------


def test_build_dataset_joins_streams_and_normalizes_case(tmp_path: Path) -> None:
    stub = StubConnection(tables=_make_tables())
    adapter = _adapter(stub)
    spec = _spec()
    handle = adapter.build_dataset(spec, _ctx(tmp_path, spec, adapter))

    assert handle.splits() == {"train", "test"}
    train = handle.read("train")
    # joined columns from both tables, lowercased to spec conventions
    assert {"customer_id", "snapshot_date", "churned_90d", "monthly_usage"} <= set(
        train.column_names
    )
    profile = handle.profile()
    assert profile.label_column == "churned_90d"
    assert profile.label_balance and set(profile.label_balance) == {"0", "1"}
    # temporal windows actually applied
    assert profile.n_rows["train"] > profile.n_rows["test"] > 0

    # the SELECTs pushed the join down (single query per split, no client join)
    selects = [q for q in stub.executed if q.startswith("SELECT *")]
    assert len(selects) == 2
    assert all("LEFT JOIN" in q and "USING (customer_id, snapshot_date)" in q for q in selects)


def test_push_down_sampling_is_reproducible_and_monotone(tmp_path: Path) -> None:
    stub = StubConnection(tables=_make_tables())
    adapter = _adapter(stub)
    spec = _spec()

    def ids(fraction: float, out: str) -> set[int]:
        ctx = _ctx(tmp_path / out, spec, adapter, sample_fraction=fraction)
        handle = adapter.build_dataset(spec, ctx)
        rows: set[int] = set()
        for split in ("train", "test"):
            rows |= set(handle.read(split).column("customer_id").to_pylist())
        return rows

    half_a = ids(0.5, "a")
    half_b = ids(0.5, "b")
    assert half_a == half_b  # same fraction -> exactly the same rows
    fifth = ids(0.2, "c")
    assert fifth <= half_a and 0 < len(fifth) < len(half_a) < 200
    # and the predicate went into the warehouse query, not client-side
    assert any("MD5_NUMBER_LOWER64" in q for q in stub.executed if q.startswith("SELECT"))


def test_sampling_without_a_key_is_an_actionable_error(tmp_path: Path) -> None:
    stub = StubConnection(tables=_make_tables())
    adapter = _adapter(stub)
    spec = _spec(inputs=None, source=LABEL_UID, sample_key=None)
    with pytest.raises(SnowflakeAdapterError, match="sample_key"):
        adapter.build_dataset(spec, _ctx(tmp_path, spec, adapter, sample_fraction=0.5))


def test_snapshot_pin_mismatch_fails_loudly(tmp_path: Path) -> None:
    stub = StubConnection(tables=_make_tables())
    adapter = _adapter(stub)
    spec = _spec()
    ctx = _ctx(tmp_path, spec, adapter)
    stub.tokens["CHURN_LABELS"] = "the-table-changed"  # simulate new DML commit
    with pytest.raises(SnowflakeAdapterError, match="changed under the pinned manifest"):
        adapter.build_dataset(spec, ctx)


def test_snapshot_ids_change_with_tokens_and_deep_uses_hash_agg(tmp_path: Path) -> None:
    stub = StubConnection(tables=_make_tables())
    adapter = _adapter(stub)
    table = _sources()[LABEL_UID]
    first = adapter.snapshot_id(table)
    assert first.startswith("sha256:")
    stub.tokens["CHURN_LABELS"] = "new-commit-token"
    assert adapter.snapshot_id(table) != first
    adapter.snapshot_id(table, deep=True)
    assert any("HASH_AGG" in q for q in stub.executed)


def test_from_locator_round_trip_without_connection(tmp_path: Path) -> None:
    stub = StubConnection(tables=_make_tables())
    adapter = _adapter(stub)
    spec = _spec()
    handle = adapter.build_dataset(spec, _ctx(tmp_path, spec, adapter))

    fresh = SnowflakeDataAdapter({})  # no credentials, no connection
    reopened = fresh.from_locator(handle.locator())
    assert reopened.read("train").num_rows == handle.read("train").num_rows

    bad = handle.locator().model_copy(update={"snapshot_id": "sha256:other"})
    with pytest.raises(SnowflakeAdapterError, match="snapshot mismatch"):
        fresh.from_locator(bad)


def test_missing_identifier_is_actionable(tmp_path: Path) -> None:
    adapter = SnowflakeDataAdapter({"database": "DB", "schema": "S"})
    with pytest.raises(SnowflakeAdapterError, match="identifier"):
        adapter.snapshot_id(FakeSourceTable(name="t", identifier=None))  # type: ignore[arg-type]


def test_plugin_import_hygiene() -> None:
    """Importing the plugin must not import snowflake.connector (ADR-14)."""
    probe = (
        "import sys\n"
        "import mbt_snowflake.plugin\n"
        "loaded = [m for m in sys.modules if m.startswith('snowflake')]\n"
        "assert not loaded, f'snowflake modules imported at plugin load: {loaded}'\n"
    )
    subprocess.run([sys.executable, "-c", probe], check=True)
