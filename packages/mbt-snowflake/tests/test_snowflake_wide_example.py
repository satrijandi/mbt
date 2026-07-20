"""Guard for the committed examples/snowflake_wide project.

Loads the example's wide dataset spec and builds it through the real Snowflake
adapter (generated SQL run in DuckDB via the shared stub, no warehouse account),
asserting the five-table join is correct - so the example cannot silently rot.
This is the pytest form of examples/snowflake_wide/show_wide_join.py.
"""

import re
from datetime import datetime
from pathlib import Path

import pyarrow as pa
import yaml
from mbt_snowflake.adapter import SnowflakeDataAdapter
from snowflake_stub_helpers import FakeBuildContext, FakeSourceTable, StubConnection

from mbt_adapter_base import DatasetSpec, ManifestNode

EXAMPLE = Path(__file__).resolve().parents[3] / "examples" / "snowflake_wide"
DATABASE, SCHEMA = "ANALYTICS", "GOLD"


def _synthetic() -> dict[str, tuple[str, pa.Table]]:
    """Five Snowflake-shaped tables (UPPERCASE cols) keyed on the demo shape."""
    months = [datetime(2026, m, 1) for m in (1, 2, 3, 4)]
    cid, snap = [], []
    for c in range(20):
        for m in months:
            cid.append(c)
            snap.append(m)
    keys = {"CUSTOMER_ID": cid, "SNAPSHOT_DATE": snap}
    return {
        "customer_population": ("CUSTOMER_POPULATION", pa.table(keys)),
        "churn_labels": (
            "CHURN_LABELS",
            pa.table({**keys, "IS_CHURN": [i % 2 for i in range(len(cid))]}),
        ),
        "demographic_features": (
            "DEMOGRAPHIC_FEATURES",
            pa.table({**keys, "AGE": [20 + c for c in cid]}),
        ),
        "engagement_features": (
            "ENGAGEMENT_FEATURES",
            pa.table({**keys, "LOGINS_30D": list(cid)}),
        ),
        "billing_features": (
            "BILLING_FEATURES",
            pa.table({**keys, "MONTHLY_SPEND": [float(c) for c in cid]}),
        ),
    }


def _table_name(ref: str) -> str:
    """'source('snowflake', 'churn_labels')' -> 'churn_labels'."""
    return re.findall(r"'([^']*)'", ref)[1]


def test_snowflake_wide_example_builds_the_five_table_join(tmp_path: Path) -> None:
    doc = yaml.safe_load((EXAMPLE / "datasets" / "wide_churn_training.yml").read_text())
    spec = DatasetSpec.model_validate(doc["datasets"][0])
    synth = _synthetic()

    refs = [
        spec.inputs.spine,
        spec.inputs.label_source,
        *[src for src, _ in spec.inputs.feature_entries],
    ]
    source_tables = {}
    for ref in refs:
        name = _table_name(ref)
        source_tables[ref] = FakeSourceTable(name=name, identifier=synth[name][0])
    stub = StubConnection(
        tables={f"{DATABASE}.{SCHEMA}.{ident}": tbl for _, (ident, tbl) in synth.items()}
    )
    adapter = SnowflakeDataAdapter({"database": DATABASE, "schema": SCHEMA})
    adapter._connection = stub  # type: ignore[assignment]

    node = ManifestNode(
        unique_id="dataset.snowflake_wide.wide_churn_training",
        resource_type="dataset",
        name="wide_churn_training",
        path="datasets/wide_churn_training.yml",
        config={},
        snapshot_id=None,  # skip snapshot verification
    )
    ctx = FakeBuildContext(
        node=node,
        source=source_tables[spec.inputs.spine],
        source_tables=source_tables,
        resolved_windows={
            "train": ("2026-01-01T00:00:00Z", "2026-03-01T00:00:00Z"),
            "test": ("2026-03-01T00:00:00Z", "2026-05-01T00:00:00Z"),
        },
        sample_fraction=1.0,
        deep_snapshot=False,
        output_dir=tmp_path / "mat",
    )

    handle = adapter.build_dataset(spec, ctx)

    # Join keys merged (from the spine), each feature column present, the label
    # projected in, and the label's join columns projected away.
    assert handle.splits() == {"train", "test"}
    assert set(handle.read("train").column_names) == {
        "customer_id",
        "snapshot_date",
        "age",
        "logins_30d",
        "monthly_spend",
        "is_churn",
    }
    assert handle.read("train").num_rows > 0 and handle.read("test").num_rows > 0

    # One pushed-down query per split, each joining all three feature tables on
    # the shared key (no client-side join).
    selects = [q for q in stub.executed if q.startswith("SELECT *")]
    assert len(selects) == 2
    for query in selects:
        assert query.count("LEFT JOIN") == 3
        assert "USING (customer_id, snapshot_date)" in query

    # the positive-path row-count log reached the bus
    assert any("materialized" in str(m) for m in ctx.events.messages)
