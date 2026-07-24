"""Guard for the committed examples/snowflake_wide project.

Three layers, none needing a warehouse account: the wide dataset spec builds
through the real Snowflake adapter (generated SQL run in DuckDB via the shared
stub), the whole project parses to the DAG its README advertises (training AND
scoring halves), and the seed script's server-side DDL creates exactly the
tables sources.yml points at - so the example cannot silently rot.
The build test is the pytest form of examples/snowflake_wide/show_wide_join.py.
"""

import importlib.util
import os
import re
from datetime import datetime
from pathlib import Path

import pyarrow as pa
import pytest
import yaml
from mbt_snowflake.adapter import SnowflakeDataAdapter
from snowflake_stub_helpers import FakeBuildContext, FakeSourceTable, StubConnection

from mbt_adapter_base import DatasetSpec, ManifestNode

EXAMPLE = Path(__file__).resolve().parents[3] / "examples" / "snowflake_wide"
DATABASE, SCHEMA = "ANALYTICS", "GOLD"

SOURCES = {
    f"source.snowflake_wide.snowflake.{name}"
    for name in (
        "customer_population",
        "churn_labels",
        "demographic_features",
        "engagement_features",
        "billing_features",
    )
}
DATASET = "dataset.snowflake_wide.wide_churn_training"
MODEL = "model.snowflake_wide.churn_wide"
SCORING = "scoring.snowflake_wide.wide_churn_scoring"


#: The population universe: 20 customers x 4 month-starts.
SPINE_CUSTOMERS = 20
SPINE_MONTHS = [datetime(2026, m, 1) for m in (1, 2, 3, 4)]


def _synthetic() -> dict[str, tuple[str, pa.Table]]:
    """Five Snowflake-shaped tables (UPPERCASE cols) keyed on the demo shape.

    The label and feature tables cover a strict SUPERSET of the population:
    an extra customer (999), an extra month (May), and a mid-month snapshot
    cadence - the situation the user's real gold tables are in. The join must
    drop every superset row (the spine decides which rows exist, ADR-22)."""
    cid, snap = [], []
    for c in range(SPINE_CUSTOMERS):
        for m in SPINE_MONTHS:
            cid.append(c)
            snap.append(m)
    spine = {"CUSTOMER_ID": list(cid), "SNAPSHOT_DATE": list(snap)}

    # Superset universe: an off-population customer, an off-window month, and
    # off-cadence mid-month snapshot dates for every (customer, month).
    for m in SPINE_MONTHS:
        cid.append(999)
        snap.append(m)
    for c in range(SPINE_CUSTOMERS):
        cid.append(c)
        snap.append(datetime(2026, 5, 1))
        for m in SPINE_MONTHS:
            cid.append(c)
            snap.append(datetime(m.year, m.month, 15))
    superset = {"CUSTOMER_ID": cid, "SNAPSHOT_DATE": snap}

    return {
        "customer_population": ("CUSTOMER_POPULATION", pa.table(spine)),
        "churn_labels": (
            "CHURN_LABELS",
            pa.table({**superset, "IS_CHURN": [i % 2 for i in range(len(cid))]}),
        ),
        "demographic_features": (
            "DEMOGRAPHIC_FEATURES",
            pa.table({**superset, "AGE": [20 + c for c in cid]}),
        ),
        "engagement_features": (
            "ENGAGEMENT_FEATURES",
            pa.table({**superset, "LOGINS_30D": list(cid)}),
        ),
        "billing_features": (
            "BILLING_FEATURES",
            pa.table({**superset, "MONTHLY_SPEND": [float(c) for c in cid]}),
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
    # Row counts are EXACTLY spine-driven: 20 customers x 2 months per split.
    # The label/feature tables' superset rows (customer 999, month 2026-05,
    # the mid-month snapshot cadence) were all dropped by the join.
    assert handle.read("train").num_rows == SPINE_CUSTOMERS * 2
    assert handle.read("test").num_rows == SPINE_CUSTOMERS * 2
    for split in ("train", "test"):
        ids = set(handle.read(split)["customer_id"].to_pylist())
        days = {d.day for d in handle.read(split)["snapshot_date"].to_pylist()}
        assert 999 not in ids
        assert days == {1}  # month-starts only; the mid-month cadence stayed behind

    # One pushed-down query per split, each joining all three feature tables on
    # the shared key (no client-side join).
    selects = [q for q in stub.executed if q.startswith("SELECT *")]
    assert len(selects) == 2
    for query in selects:
        assert query.count("LEFT JOIN") == 3
        assert "USING (customer_id, snapshot_date)" in query

    # the positive-path row-count log reached the bus
    assert any("materialized" in str(m) for m in ctx.events.messages)


def test_snowflake_wide_example_parses_to_the_full_lifecycle_shape() -> None:
    """The README advertises `mbt parse` and "Parsed 8 resources": 5 warehouse
    sources feeding one dataset feeding one model, plus the scoring node that
    reuses the spine and feature tables (and the label table as ground truth,
    ADR-21/22). The parse-only sibling of the build test above, mirroring
    tests/test_s3_wide_example.py."""
    from mbt.parsing import parse_project

    parsed = parse_project(EXAMPLE)

    assert not parsed.report.errors
    assert set(parsed.sources) == SOURCES
    assert set(parsed.datasets) == {DATASET}
    assert set(parsed.models) == {MODEL}
    assert set(parsed.scoring) == {SCORING}
    assert not parsed.exposures
    assert len(parsed.nodes) + len(parsed.sources) + len(parsed.exposures) == 8

    # Training half: all five tables -> dataset -> model.
    assert set(parsed.datasets[DATASET].depends_on) == SOURCES
    assert parsed.models[MODEL].depends_on == [DATASET]
    # Serving half: the scoring node scores with the registered model over the
    # same spine + feature tables, and joins the label table back in as
    # delayed ground truth - so it depends on the model AND all five sources.
    assert set(parsed.scoring[SCORING].depends_on) == {MODEL} | SOURCES
    for source in SOURCES:
        assert parsed.graph.has_edge(source, DATASET)
    assert parsed.graph.has_edge(DATASET, MODEL)
    assert parsed.graph.has_edge(MODEL, SCORING)


def test_example_profiles_render_with_dev_env_only(
    monkeypatch: pytest.MonkeyPatch, tmp_path: Path
) -> None:
    """profiles.yml is jinja-rendered WHOLE regardless of the selected target,
    so every env_var() in a non-selected target must carry a default. This
    shipped broken once: prod's strict SNOWFLAKE_PRIVATE_KEY_FILE reference
    made plain dev runs fail with 'environment variable ... is not set'."""
    from mbt.config.profiles import load_profiles

    for name in list(os.environ):
        if name.startswith("SNOWFLAKE_"):
            monkeypatch.delenv(name)
    for key in ("ACCOUNT", "USER", "WAREHOUSE", "DATABASE", "SCHEMA"):
        monkeypatch.setenv(f"SNOWFLAKE_{key}", f"dev-{key.lower()}")

    dev = load_profiles("snowflake_wide", EXAMPLE)
    assert dev.target_name == "dev"
    config = dev.target.data.config
    assert config["authenticator"] == "externalbrowser"  # the env_var default
    assert config["account"] == "dev-account"

    # The other target must also select cleanly on the same minimal env; its
    # key-pair var is validated at connect time, not at render time.
    prod = load_profiles("snowflake_wide", EXAMPLE, target_override="prod")
    assert prod.target_name == "prod"
    assert prod.target.data.config["connect_args"]["private_key_file"] == ""


def test_seed_script_creates_exactly_the_example_source_tables() -> None:
    """seed_demo_tables.py and sources.yml must name the same physical tables,
    and every statement must be a single server-side CREATE ... AS SELECT over
    GENERATOR() - the property the README sells (nothing is uploaded, size is
    a knob). Rendering is pure string work, so no connection is involved."""
    spec = importlib.util.spec_from_file_location(
        "seed_demo_tables", EXAMPLE / "seed_demo_tables.py"
    )
    assert spec is not None and spec.loader is not None
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)

    doc = yaml.safe_load((EXAMPLE / "sources.yml").read_text())
    identifiers = {t["identifier"] for t in doc["sources"][0]["tables"]}

    statements = module.render_statements(DATABASE, SCHEMA, customers=1000, months=5)
    assert set(statements) == identifiers == set(module.TABLES)
    feature_customers, feature_months = module.feature_superset(1000, 5)
    assert feature_customers > 1000 and feature_months > 5
    for table, sql in statements.items():
        assert sql.startswith(f"CREATE OR REPLACE TABLE {DATABASE}.{SCHEMA}.{table} AS")
        # Size is a server-side knob; feature tables generate the SUPERSET
        # universe (more customers, an extra month) that the spine join drops.
        expected = 1000 if table in ("CUSTOMER_POPULATION", "CHURN_LABELS") else feature_customers
        assert f"GENERATOR(ROWCOUNT => {expected})" in sql
        # Deterministic seeding: reruns must produce identical tables.
        assert "RANDOM" not in sql.upper()
    # Engagement additionally carries its own mid-month snapshot cadence.
    assert "UNION ALL" in statements["ENGAGEMENT_FEATURES"]
    assert "DATEADD(DAY, 14, snapshot_date)" in statements["ENGAGEMENT_FEATURES"]
