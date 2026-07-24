"""Show the wide multi-table Snowflake join WITHOUT a Snowflake account.

mbt compiles the population + label + three feature tables (all keyed on
[customer_id, snapshot_date]) into ONE Snowflake query per split: a LEFT JOIN
chain onto the population spine, with the label joined inner and its join
columns projected away. This script feeds synthetic, Snowflake-shaped tables
through the REAL mbt Snowflake adapter, executing that generated SQL in DuckDB
(the same technique the adapter's unit tests use), and prints the generated
SQL plus the joined training/test panels.

    uv run python examples/snowflake_wide/show_wide_join.py

No credentials, no warehouse, no network - just proof the join is correct.
"""

from __future__ import annotations

import re
import tempfile
from dataclasses import dataclass, field
from datetime import datetime
from pathlib import Path
from typing import Any

import duckdb
import pyarrow as pa
import yaml
from mbt_snowflake.adapter import SnowflakeDataAdapter

from mbt_adapter_base import DatasetSpec, ManifestNode

HERE = Path(__file__).resolve().parent
DATABASE, SCHEMA = "ANALYTICS", "GOLD"  # matches the profile's snowflake config


# -- a fake Snowflake session that runs the adapter's generated SQL in DuckDB --
# (mirrors packages/mbt-snowflake/tests/test_snowflake_adapter.py::StubConnection)
class _StubCursor:
    def __init__(self, conn: _StubConnection) -> None:
        self._conn = conn
        self._table: pa.Table | None = None

    def execute(self, sql: str) -> _StubCursor:
        self._conn.executed.append(sql)
        self._table = self._conn.run_in_duckdb(sql)
        return self

    def fetch_arrow_batches(self):
        assert self._table is not None
        yield self._table

    def fetchone(self) -> tuple[Any]:
        return (None,)

    def close(self) -> None:
        pass


@dataclass
class _StubConnection:
    tables: dict[str, pa.Table]  # qualified ref -> data (UPPERCASE columns)
    executed: list[str] = field(default_factory=list)

    def cursor(self) -> _StubCursor:
        return _StubCursor(self)

    def run_in_duckdb(self, sql: str) -> pa.Table:
        con = duckdb.connect()
        try:
            # Shim the Snowflake-only functions the generated SQL may use.
            con.execute("CREATE MACRO TO_TIMESTAMP_NTZ(s) AS CAST(s AS TIMESTAMP)")
            con.execute(
                "CREATE MACRO MD5_NUMBER_LOWER64(s) AS "
                "(md5_number(s) % 9223372036854775807)::BIGINT"
            )
            for i, (ref, table) in enumerate(self.tables.items()):
                view = f"stub_{i}"
                con.register(view, table)
                sql = sql.replace(ref, view)
            sql = sql.replace("AS TIMESTAMP_NTZ)", "AS TIMESTAMP)")
            return con.sql(sql).to_arrow_table()
        finally:
            con.close()


@dataclass
class _FakeSourceTable:
    name: str
    identifier: str
    path: str | None = None
    format: str = "snowflake"


class _PrintSink:
    """Prints the adapter's own event-bus messages (e.g. the row-count log)."""

    def emit(self, event: object) -> None:
        print(f"    [mbt event] {event}")


@dataclass
class _FakeBuildContext:
    node: ManifestNode
    source: Any
    source_tables: dict[str, Any]
    resolved_windows: dict[str, tuple[str, str]]
    sample_fraction: float
    deep_snapshot: bool
    output_dir: Path
    events: Any = field(default_factory=_PrintSink)


# table name -> (snowflake identifier, arrow table); the single source of truth.
# Five Snowflake-shaped tables (UPPERCASE columns), all keyed on
# (CUSTOMER_ID, SNAPSHOT_DATE). The POPULATION covers 60 customers over the
# Jan-Apr 2026 month-starts; the label/feature tables deliberately cover a
# strict SUPERSET (an off-population customer, an off-window month, and a
# mid-month snapshot cadence) - like real gold tables, which serve the whole
# company, not one model's cohort. The join must return spine rows only.
def _synthetic_by_name() -> dict[str, tuple[str, pa.Table]]:
    months = [datetime(2026, m, 1) for m in (1, 2, 3, 4)]
    n = 60
    cid, snap = [], []
    for c in range(n):
        for m in months:
            cid.append(c)
            snap.append(m)
    spine_keys = {"CUSTOMER_ID": list(cid), "SNAPSHOT_DATE": list(snap)}

    # The superset universe the label/feature tables cover.
    for m in months:
        cid.append(999)
        snap.append(m)
    for c in range(n):
        cid.append(c)
        snap.append(datetime(2026, 5, 1))
        for m in months:
            cid.append(c)
            snap.append(datetime(m.year, m.month, 15))
    keys = {"CUSTOMER_ID": cid, "SNAPSHOT_DATE": snap}

    population = pa.table(spine_keys)
    labels = pa.table(
        {**keys, "IS_CHURN": [1 if (c * 7 + i) % 5 == 0 else 0 for i, c in enumerate(cid)]}
    )
    demographic = pa.table(
        {**keys, "AGE": [20 + (c % 50) for c in cid], "TENURE_MONTHS": [c % 36 for c in cid]}
    )
    engagement = pa.table(
        {
            **keys,
            "LOGINS_30D": [(c * 3 + i) % 40 for i, c in enumerate(cid)],
            "AVG_SESSION_MIN": [float((c % 25) + 1) for c in cid],
        }
    )
    billing = pa.table(
        {
            **keys,
            "MONTHLY_SPEND": [float(10 + (c % 90)) for c in cid],
            "PLAN_TIER": [["basic", "pro", "enterprise"][c % 3] for c in cid],
            # Bookkeeping column the spec's per-table `exclude:` prunes
            # inside the generated query (ADR-25) - absent from the panels.
            "ETL_LOADED_AT": [datetime(2026, 6, 1)] * len(cid),
        }
    )
    return {
        "customer_population": ("CUSTOMER_POPULATION", population),
        "churn_labels": ("CHURN_LABELS", labels),
        "demographic_features": ("DEMOGRAPHIC_FEATURES", demographic),
        "engagement_features": ("ENGAGEMENT_FEATURES", engagement),
        "billing_features": ("BILLING_FEATURES", billing),
    }


def _table_name_from_ref(ref: str) -> str:
    """'source('snowflake', 'churn_labels')' -> 'churn_labels'."""
    return re.findall(r"'([^']*)'", ref)[1]


def main() -> None:
    spec_doc = yaml.safe_load((HERE / "datasets" / "wide_churn_training.yml").read_text())
    spec = DatasetSpec.model_validate(spec_doc["datasets"][0])

    synth = _synthetic_by_name()

    # Every source() ref the dataset uses, keyed exactly as the spec holds them.
    refs = [
        spec.inputs.spine,
        spec.inputs.label_source,
        *[entry.source for entry in spec.inputs.feature_entries],
    ]
    source_tables = {}
    for ref in refs:
        name = _table_name_from_ref(ref)
        source_tables[ref] = _FakeSourceTable(name=name, identifier=synth[name][0])
    stub_tables = {f"{DATABASE}.{SCHEMA}.{ident}": tbl for _, (ident, tbl) in synth.items()}

    adapter = SnowflakeDataAdapter({"database": DATABASE, "schema": SCHEMA})
    adapter._connection = _StubConnection(stub_tables)  # type: ignore[assignment]

    node = ManifestNode(
        unique_id="dataset.snowflake_wide.wide_churn_training",
        resource_type="dataset",
        name="wide_churn_training",
        path="datasets/wide_churn_training.yml",
        config={},
        snapshot_id=None,  # skip snapshot verification for the offline demo
    )

    with tempfile.TemporaryDirectory() as tmp:
        ctx = _FakeBuildContext(
            node=node,
            source=source_tables[spec.inputs.spine],
            source_tables=source_tables,
            resolved_windows={
                "train": ("2026-01-01T00:00:00Z", "2026-03-01T00:00:00Z"),
                "test": ("2026-03-01T00:00:00Z", "2026-05-01T00:00:00Z"),
            },
            sample_fraction=1.0,
            deep_snapshot=False,
            output_dir=Path(tmp) / "mat",
        )

        population_rows = synth["customer_population"][1].num_rows
        feature_rows = synth["engagement_features"][1].num_rows
        print("Building the wide dataset through the mbt Snowflake adapter...")
        print(
            f"  population spine: {population_rows} rows; each feature/label table: "
            f"{feature_rows} rows (a superset: extra customer, extra month, mid-month "
            "snapshots).\n  Expect the joined panels to hold the spine's rows ONLY.\n"
        )
        handle = adapter.build_dataset(spec, ctx)

        print("\nGenerated Snowflake SQL (one query per split):\n")
        for sql in adapter._connection.executed:  # type: ignore[attr-defined]
            print("  " + sql.replace("  ", " ") + "\n")

        for split in ("train", "test"):
            table = handle.read(split)
            print(f"== {split}: {table.num_rows} rows x {table.num_columns} cols ==")
            print("   columns:", ", ".join(table.column_names))
            print(table.slice(0, 3).to_pandas().to_string(index=False))
            print()


if __name__ == "__main__":
    main()
