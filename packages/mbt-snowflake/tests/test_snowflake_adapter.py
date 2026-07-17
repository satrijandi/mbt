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
    base_relation,
    key_hash_expr,
    qualify_table,
    sampling_predicate,
    scoring_query,
    scoring_relation,
    split_queries,
)

from mbt_adapter_base import (
    DatasetLocator,
    DatasetSpec,
    ManifestNode,
    PredictionRunInfo,
    ScoringInputSpec,
    ScoringOutputSpec,
)
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
            # con.sql(...).to_arrow_table() (the Relation API) exists at the
            # duckdb>=1.0 floor and is not deprecated; Connection.fetch_arrow_table
            # warns on current duckdb.
            return con.sql(sql).to_arrow_table()
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


def test_population_spine_with_label_offset_joins_in_duckdb(tmp_path: Path) -> None:
    """ADR-22 through the generated SQL: population spine, per-table using
    columns, and the calendar-month label offset, executed in DuckDB."""
    months = [datetime(2026, m, 1) for m in range(1, 8)]
    population_rows = [(cid, f"sf-{cid}", when) for when in months[:-1] for cid in range(40)]
    population = pa.table(
        {
            "CUSTOMER_ID": [r[0] for r in population_rows],
            "SAFE_ID": [r[1] for r in population_rows],
            "SNAPSHOT_DATE": [r[2] for r in population_rows],
        }
    )
    # outcome for snapshot m lives at m+1, value encodes the SPINE month index
    label_rows = [
        (cid, months[i + 1], (cid + i) % 2) for i in range(len(months) - 1) for cid in range(40)
    ]
    labels = pa.table(
        {
            "CUSTOMER_ID": [r[0] for r in label_rows],
            "SNAPSHOT_DATE": [r[1] for r in label_rows],
            "CHURNED": [r[2] for r in label_rows],
        }
    )
    txn = pa.table(
        {
            "SAFE_ID": [r[1] for r in population_rows],
            "SNAPSHOT_DATE": [r[2] for r in population_rows],
            "TXN_TOTAL": [float(r[0] % 300) for r in population_rows],
        }
    )
    stub = StubConnection(
        tables={
            "ANALYTICS.GOLD.POPULATION": population,
            "ANALYTICS.GOLD.MONTHLY_LABELS": labels,
            "ANALYTICS.GOLD.TXN_FEATURES": txn,
        }
    )
    adapter = _adapter(stub)
    pop_uid = "source.p.snowflake.population"
    lbl_uid = "source.p.snowflake.monthly_labels"
    txn_uid = "source.p.snowflake.txn_features"
    spec = DatasetSpec.model_validate(
        {
            "name": "wide_churn",
            "inputs": {
                "population": pop_uid,
                "label": {
                    "source": lbl_uid,
                    "using": ["customer_id", "snapshot_date"],
                    "time_offset": "1mo",
                },
                "features": [{"source": txn_uid, "using": ["safe_id", "snapshot_date"]}],
            },
            "sample_key": ["customer_id"],
            "label": {"column": "churned"},
            "split": {
                "strategy": "temporal",
                "time_column": "snapshot_date",
                "train": "2026-01-01:2026-05-01",
                "test": "2026-05-01:2026-07-01",
            },
        }
    )
    sources = {
        pop_uid: FakeSourceTable(name="population", identifier="POPULATION"),
        lbl_uid: FakeSourceTable(name="monthly_labels", identifier="MONTHLY_LABELS"),
        txn_uid: FakeSourceTable(name="txn_features", identifier="TXN_FEATURES"),
    }
    pinned = combine_snapshots({uid: adapter.snapshot_id(t) for uid, t in sources.items()})
    node = ManifestNode(
        unique_id="dataset.p.wide_churn",
        resource_type="dataset",
        name=spec.name,
        path="datasets/wide_churn.yml",
        config={},
        snapshot_id=pinned,
    )
    ctx = FakeBuildContext(
        node=node,
        source=sources[pop_uid],
        source_tables=sources,
        resolved_windows={
            "train": ("2026-01-01T00:00:00Z", "2026-05-01T00:00:00Z"),
            "test": ("2026-05-01T00:00:00Z", "2026-07-01T00:00:00Z"),
        },
        sample_fraction=1.0,
        deep_snapshot=False,
        output_dir=tmp_path / "mat",
    )
    handle = adapter.build_dataset(spec, ctx)
    month_index = {when: i for i, when in enumerate(months)}
    for split in ("train", "test"):
        table = handle.read(split)
        # spine + feature + label columns, label join columns projected away
        assert set(table.column_names) == {
            "customer_id",
            "safe_id",
            "snapshot_date",
            "txn_total",
            "churned",
        }
        for row in table.to_pylist():
            expected = (row["customer_id"] + month_index[row["snapshot_date"]]) % 2
            assert row["churned"] == expected
    # the offset join went into the warehouse query, not client-side
    joined = [q for q in stub.executed if "INTERVAL '1 MONTH'" in q]
    assert joined and all("SELECT * RENAME" in q for q in joined)


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


def test_plugin_descriptor_wires_the_data_adapter() -> None:
    from mbt_snowflake.plugin import PLUGIN

    from mbt_adapter_base import CONTRACT_VERSION

    assert PLUGIN.name == "snowflake"
    assert PLUGIN.contract_version == CONTRACT_VERSION
    assert PLUGIN.data is SnowflakeDataAdapter
    assert PLUGIN.fingerprint_packages == ["snowflake-connector-python"]


# -- SQL identifier and split edge cases -----------------------------------------------


def test_qualify_schema_qualified_table_needs_a_database() -> None:
    with pytest.raises(SnowflakeSQLError, match="needs a database"):
        qualify_table("S2.T", None, "S")


def test_key_hash_requires_a_non_empty_key() -> None:
    with pytest.raises(SnowflakeSQLError, match="non-empty key"):
        key_hash_expr([])


def test_base_relation_single_source_is_the_table_ref() -> None:
    spec = _spec(inputs=None, source=LABEL_UID, sample_key=["customer_id"])
    assert base_relation(spec, {LABEL_UID: "ANALYTICS.GOLD.CHURN_LABELS"}) == (
        "ANALYTICS.GOLD.CHURN_LABELS",
        [],
    )


def test_random_split_carves_a_validation_bucket() -> None:
    spec = _spec(
        inputs=None,
        source=LABEL_UID,
        sample_key=["customer_id"],
        split={
            "strategy": "random",
            "train": "0.6",
            "validation": "0.2",
            "test": "0.2",
            "seed": 7,
        },
    )
    queries = split_queries(spec, "T", [], {})
    assert set(queries) == {"train", "validation", "test"}
    assert ">= 600000" in queries["validation"] and "< 800000" in queries["validation"]
    assert ">= 800000" in queries["test"] and "< 1000000" in queries["test"]


# -- connection construction --------------------------------------------------------


def test_connect_without_account_is_actionable() -> None:
    with pytest.raises(SnowflakeAdapterError, match="needs at least 'account'"):
        SnowflakeDataAdapter({"user": "u"})._connect()


def test_connect_passes_config_through_and_caches(monkeypatch: pytest.MonkeyPatch) -> None:
    import snowflake.connector

    session = object()
    calls: list[dict[str, Any]] = []

    def fake_connect(**kwargs: Any) -> object:
        calls.append(kwargs)
        return session

    monkeypatch.setattr(snowflake.connector, "connect", fake_connect)
    adapter = SnowflakeDataAdapter(
        {"account": "acct", "user": "u", "role": "R", "connect_args": {"login_timeout": 5}}
    )
    assert adapter._connect() is session
    assert adapter._connect() is session  # cached: no second connect
    assert calls == [{"account": "acct", "user": "u", "role": "R", "login_timeout": 5}]


def test_connect_externalbrowser_defaults_sso_token_caching(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """Each job subprocess opens its own connection, so externalbrowser
    without cached tokens would prompt once PER NODE - the adapter defaults
    the cache on (an explicit connect_args value still wins)."""
    import snowflake.connector

    calls: list[dict[str, Any]] = []

    def fake_connect(**kwargs: Any) -> object:
        calls.append(kwargs)
        return object()

    monkeypatch.setattr(snowflake.connector, "connect", fake_connect)

    SnowflakeDataAdapter(
        {"account": "acct", "user": "u", "authenticator": "EXTERNALBROWSER"}
    )._connect()
    assert calls[-1]["client_store_temporary_credential"] is True

    SnowflakeDataAdapter(
        {
            "account": "acct",
            "user": "u",
            "authenticator": "externalbrowser",
            "connect_args": {"client_store_temporary_credential": False},
        }
    )._connect()
    assert calls[-1]["client_store_temporary_credential"] is False

    SnowflakeDataAdapter({"account": "acct", "user": "u", "password": "pw"})._connect()
    assert "client_store_temporary_credential" not in calls[-1]


def test_connect_failure_is_actionable(monkeypatch: pytest.MonkeyPatch) -> None:
    import snowflake.connector

    def fake_connect(**kwargs: Any) -> object:
        raise RuntimeError("250001: could not authenticate")

    monkeypatch.setattr(snowflake.connector, "connect", fake_connect)
    adapter = SnowflakeDataAdapter({"account": "acct", "user": "u"})
    with pytest.raises(SnowflakeAdapterError, match="could not connect to Snowflake"):
        adapter._connect()


# -- snapshot and build error paths ---------------------------------------------------


def test_invalid_source_identifier_wraps_the_sql_error() -> None:
    adapter = SnowflakeDataAdapter({"database": "DB", "schema": "S"})
    with pytest.raises(SnowflakeAdapterError, match="invalid table identifier"):
        adapter._table_ref(FakeSourceTable(name="t", identifier="bad-name!"))


def test_missing_snapshot_token_is_actionable() -> None:
    stub = StubConnection(tables=_make_tables(), tokens={"CHURN_LABELS": None})  # type: ignore[dict-item]
    adapter = _adapter(stub)
    with pytest.raises(SnowflakeAdapterError, match="could not read a snapshot token"):
        adapter.snapshot_id(_sources()[LABEL_UID])


def test_verify_snapshot_skipped_without_a_pin(tmp_path: Path) -> None:
    stub = StubConnection(tables=_make_tables())
    adapter = _adapter(stub)
    spec = _spec()
    adapter._verify_snapshot(_ctx(tmp_path, spec, adapter, snapshot=None))
    assert stub.executed == []  # no pin -> no snapshot queries


def test_build_dataset_clears_stale_outputs(tmp_path: Path) -> None:
    stub = StubConnection(tables=_make_tables())
    adapter = _adapter(stub)
    spec = _spec()
    ctx = _ctx(tmp_path, spec, adapter)
    ctx.output_dir.mkdir(parents=True)
    stale = ctx.output_dir / "leftover.parquet"
    stale.write_bytes(b"stale")
    handle = adapter.build_dataset(spec, ctx)
    assert not stale.exists()
    assert handle.splits() == {"train", "test"}


def test_sample_fraction_out_of_range_is_actionable(tmp_path: Path) -> None:
    stub = StubConnection(tables=_make_tables())
    adapter = _adapter(stub)
    spec = _spec()
    with pytest.raises(SnowflakeAdapterError, match=r"sample_fraction must be in \(0, 1\]"):
        adapter.build_dataset(spec, _ctx(tmp_path, spec, adapter, sample_fraction=1.5))


def test_build_dataset_wraps_split_sql_errors(tmp_path: Path) -> None:
    stub = StubConnection(tables=_make_tables())
    adapter = _adapter(stub)
    spec = _spec(
        inputs=None,
        source=LABEL_UID,
        sample_key=None,
        split={"strategy": "random", "train": "0.8", "test": "0.2", "seed": 7},
    )
    with pytest.raises(SnowflakeAdapterError, match="random split on Snowflake needs"):
        adapter.build_dataset(spec, _ctx(tmp_path, spec, adapter))


class _NoBatchCursor(StubCursor):
    def fetch_arrow_batches(self):  # zero result batches
        return iter(())


class _NoBatchConnection(StubConnection):
    def cursor(self) -> StubCursor:
        return _NoBatchCursor(self)


def test_zero_row_split_names_the_split(tmp_path: Path) -> None:
    stub = _NoBatchConnection(tables=_make_tables())
    adapter = _adapter(stub)
    spec = _spec()
    ctx = _ctx(tmp_path, spec, adapter, snapshot=None)
    with pytest.raises(SnowflakeAdapterError, match="materialized 0 rows"):
        adapter.build_dataset(spec, ctx)
    # zero batches still emit an (empty) file so the error names the split
    assert (ctx.output_dir / "train.parquet").is_file()


@dataclass
class _FailingConnection:
    exc: Exception

    def cursor(self) -> "_FailingCursor":
        return _FailingCursor(self.exc)


@dataclass
class _FailingCursor:
    exc: Exception

    def execute(self, sql: str) -> None:
        raise self.exc

    def close(self) -> None:
        pass


def test_stream_query_reraises_adapter_errors_unwrapped(tmp_path: Path) -> None:
    already = SnowflakeAdapterError("already actionable")
    adapter = SnowflakeDataAdapter({})
    adapter._connection = _FailingConnection(already)  # type: ignore[assignment]
    with pytest.raises(SnowflakeAdapterError) as excinfo:
        adapter._stream_query_to_parquet("SELECT 1", tmp_path / "out.parquet")
    assert excinfo.value is already  # not double-wrapped


def test_stream_query_wraps_connector_errors_with_the_query(tmp_path: Path) -> None:
    adapter = SnowflakeDataAdapter({})
    adapter._connection = _FailingConnection(RuntimeError("002003: object does not exist"))  # type: ignore[assignment]
    with pytest.raises(SnowflakeAdapterError, match=r"Snowflake query failed.*\n.*query: SELECT 1"):
        adapter._stream_query_to_parquet("SELECT 1", tmp_path / "out.parquet")


def test_from_locator_wraps_missing_materializations(tmp_path: Path) -> None:
    locator = DatasetLocator(
        adapter="snowflake", uri=f"file://{tmp_path}/nowhere", snapshot_id="sha256:x"
    )
    with pytest.raises(SnowflakeAdapterError, match="no complete dataset materialization"):
        SnowflakeDataAdapter({}).from_locator(locator)


# -- batch scoring (contract 1.1, ADR-20/21/23) ---------------------------------------


class _CapturingSink:
    def __init__(self) -> None:
        self.messages: list[Any] = []

    def emit(self, event: Any) -> None:
        self.messages.append(event)


SCORE_WINDOW = ("2026-06-03T00:00:00Z", "2026-07-01T00:00:00Z")


def _scoring_sources() -> dict[str, FakeSourceTable]:
    return {USAGE_UID: FakeSourceTable(name="usage_features", identifier="USAGE_FEATURES")}


def _scoring_ctx(
    tmp_path: Path,
    adapter: SnowflakeDataAdapter,
    sources: dict[str, FakeSourceTable],
    *,
    sample_fraction: float = 1.0,
    window: tuple[str, str] | None = SCORE_WINDOW,
    events: Any = None,
    verify_snapshot: bool = False,
) -> FakeBuildContext:
    pinned = None
    if verify_snapshot:
        pinned = combine_snapshots({uid: adapter.snapshot_id(t) for uid, t in sources.items()})
    node = ManifestNode(
        unique_id="scoring.p.churn_scoring",
        resource_type="scoring",
        name="churn_scoring",
        path="scoring/churn_scoring.yml",
        config={},
        snapshot_id=pinned,
    )
    return FakeBuildContext(
        node=node,
        source=next(iter(sources.values())),
        source_tables=sources,
        resolved_windows={"score": window} if window else {},
        sample_fraction=sample_fraction,
        deep_snapshot=False,
        output_dir=tmp_path / "score",
        events=events,
    )


def test_scoring_relation_single_source_and_label_free_joins() -> None:
    single = ScoringInputSpec.model_validate({"source": USAGE_UID})
    assert scoring_relation(single, {USAGE_UID: "DB.S.USAGE"}) == "DB.S.USAGE"

    joined = ScoringInputSpec.model_validate(
        {
            "inputs": {
                "spine": LABEL_UID,
                "features": [USAGE_UID],
                "join_key": ["customer_id", "snapshot_date"],
            }
        }
    )
    sql = scoring_relation(joined, {LABEL_UID: "DB.S.SPINE", USAGE_UID: "DB.S.FEAT"})
    assert "DB.S.SPINE AS mbt_spine" in sql
    assert "LEFT JOIN DB.S.FEAT AS mbt_f0 USING (customer_id, snapshot_date)" in sql
    assert "mbt_label" not in sql  # a scoring relation never joins a label


def test_scoring_query_applies_window_only_with_time_column() -> None:
    spec = ScoringInputSpec.model_validate({"source": USAGE_UID, "time_column": "snapshot_date"})
    windowed = scoring_query(spec, {USAGE_UID: "T"}, ["(is_active)"], SCORE_WINDOW)
    assert "TO_TIMESTAMP_NTZ('2026-06-03 00:00:00')" in windowed
    assert "(is_active)" in windowed
    assert scoring_query(spec, {USAGE_UID: "T"}, [], None) == "SELECT * FROM T"


def test_build_scoring_input_streams_windowed_single_source(tmp_path: Path) -> None:
    stub = StubConnection(tables=_make_tables())
    adapter = _adapter(stub)
    sources = _scoring_sources()
    spec = ScoringInputSpec.model_validate({"source": USAGE_UID, "time_column": "snapshot_date"})
    handle = adapter.build_scoring_input(spec, _scoring_ctx(tmp_path, adapter, sources))

    assert handle.splits() == {"score"}
    score = handle.read("score")
    assert "monthly_usage" in score.column_names  # normalized to lowercase
    assert 0 < score.num_rows < 200  # the score window pruned the batch
    selects = [q for q in stub.executed if q.startswith("SELECT *")]
    assert len(selects) == 1 and "TO_TIMESTAMP_NTZ" in selects[0]

    # rebuilding clears stale outputs and re-materializes deterministically
    again = adapter.build_scoring_input(spec, _scoring_ctx(tmp_path, adapter, sources))
    assert again.read("score").num_rows == score.num_rows


def test_build_scoring_input_joins_spine_and_features(tmp_path: Path) -> None:
    stub = StubConnection(tables=_make_tables())
    adapter = _adapter(stub)
    sources = _sources()  # spine + feature tables
    spec = ScoringInputSpec.model_validate(
        {
            "inputs": {
                "spine": LABEL_UID,
                "features": [USAGE_UID],
                "join_key": ["customer_id", "snapshot_date"],
            },
            "time_column": "snapshot_date",
        }
    )
    score = adapter.build_scoring_input(spec, _scoring_ctx(tmp_path, adapter, sources)).read(
        "score"
    )
    assert {"customer_id", "snapshot_date", "monthly_usage"} <= set(score.column_names)
    selects = [q for q in stub.executed if q.startswith("SELECT *")]
    assert "USING (customer_id, snapshot_date)" in selects[0]


def test_build_scoring_input_without_time_column_scores_full_batch(tmp_path: Path) -> None:
    stub = StubConnection(tables=_make_tables())
    adapter = _adapter(stub)
    sources = _scoring_sources()
    spec = ScoringInputSpec.model_validate({"source": USAGE_UID})  # no time_column
    handle = adapter.build_scoring_input(
        spec, _scoring_ctx(tmp_path, adapter, sources, window=None)
    )
    assert handle.read("score").num_rows == 200  # whole table, no window predicate
    selects = [q for q in stub.executed if q.startswith("SELECT *")]
    assert "WHERE" not in selects[0]


def test_build_scoring_input_pushes_down_sampling(tmp_path: Path) -> None:
    stub = StubConnection(tables=_make_tables())
    adapter = _adapter(stub)
    sources = _scoring_sources()
    spec = ScoringInputSpec.model_validate(
        {"source": USAGE_UID, "time_column": "snapshot_date", "sample_key": ["customer_id"]}
    )
    handle = adapter.build_scoring_input(
        spec, _scoring_ctx(tmp_path, adapter, sources, sample_fraction=0.5)
    )
    selects = [q for q in stub.executed if q.startswith("SELECT *")]
    assert "MD5_NUMBER_LOWER64" in selects[0]  # sampling pushed into the warehouse
    assert handle.read("score").num_rows < 200


def test_build_scoring_input_rejects_bad_sample_fraction(tmp_path: Path) -> None:
    adapter = SnowflakeDataAdapter({"database": "ANALYTICS", "schema": "GOLD"})
    spec = ScoringInputSpec.model_validate({"source": USAGE_UID})
    ctx = _scoring_ctx(tmp_path, adapter, _scoring_sources(), sample_fraction=1.5, window=None)
    with pytest.raises(SnowflakeAdapterError, match=r"sample_fraction must be in \(0, 1\]"):
        adapter.build_scoring_input(spec, ctx)


def test_build_scoring_input_sampling_needs_a_key(tmp_path: Path) -> None:
    adapter = SnowflakeDataAdapter({"database": "ANALYTICS", "schema": "GOLD"})
    spec = ScoringInputSpec.model_validate({"source": USAGE_UID})  # no sample_key
    ctx = _scoring_ctx(tmp_path, adapter, _scoring_sources(), sample_fraction=0.5, window=None)
    with pytest.raises(SnowflakeAdapterError, match="stable row identity"):
        adapter.build_scoring_input(spec, ctx)


def test_build_scoring_input_wraps_invalid_identifiers(tmp_path: Path) -> None:
    adapter = SnowflakeDataAdapter({"database": "ANALYTICS", "schema": "GOLD"})
    spec = ScoringInputSpec.model_validate({"source": USAGE_UID, "time_column": "bad; DROP"})
    ctx = _scoring_ctx(
        tmp_path, adapter, _scoring_sources()
    )  # window present -> validates time_col
    with pytest.raises(SnowflakeAdapterError, match="invalid column identifier"):
        adapter.build_scoring_input(spec, ctx)


def test_build_scoring_input_zero_rows_warns_not_errors(tmp_path: Path) -> None:
    stub = StubConnection(tables=_make_tables())
    adapter = _adapter(stub)
    spec = ScoringInputSpec.model_validate({"source": USAGE_UID, "time_column": "snapshot_date"})
    events = _CapturingSink()
    future = ("2030-01-01T00:00:00Z", "2030-02-01T00:00:00Z")  # matches no rows
    ctx = _scoring_ctx(tmp_path, adapter, _scoring_sources(), window=future, events=events)
    handle = adapter.build_scoring_input(spec, ctx)
    assert handle.read("score").num_rows == 0
    assert any("0 rows" in str(m) for m in events.messages)


def test_open_predictions_stages_runs_under_root(tmp_path: Path) -> None:
    adapter = SnowflakeDataAdapter(
        {"database": "ANALYTICS", "schema": "GOLD", "predictions_root": str(tmp_path)}
    )
    output = ScoringOutputSpec.model_validate({"path": "churn_scoring/preds"})
    store = adapter.open_predictions(output)
    assert store.root == tmp_path / "churn_scoring" / "preds"

    table = pa.table({"customer_id": [1, 2], "prediction": [0.1, 0.9]})
    info = PredictionRunInfo(
        run_key="r1",
        uri="",
        scored_at="2026-07-01T00:00:00Z",
        run_id="run1",
        model_name="churn",
        model_version="1",
        row_count=0,
    )
    assert store.write_run(table, info).row_count == 2
    assert [r.run_key for r in store.list_runs()] == ["r1"]
    assert store.read("r1").num_rows == 2
