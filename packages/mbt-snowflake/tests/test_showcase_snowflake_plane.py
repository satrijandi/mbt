"""Guard for the showcase's Snowflake data plane (DESIGN.md section 11).

examples/showcase runs one project over three data planes: the SeaweedFS
parquet lake (spark targets), a synced DuckDB copy (prod_score), and Snowflake
(the `snowflake` target). What makes that possible is that every table in
sources.yml declares BOTH a `path:` and an `identifier:`, and each adapter
reads only the field it understands - so the dataset, model, and scoring specs
are literally the same on every plane.

These tests hold that invariant without needing a warehouse account or the
docker stack: the committed wide spec is built through the REAL Snowflake
adapter with its generated SQL executed in DuckDB (the shared stub), and the
identifiers are cross-checked against the seeding script. The credentialed
end-to-end proof lives in tests/test_showcase_snowflake.py.

The wide shape here uses DIFFERENT entity keys per feature history
(transactions match only through safe_id), which a single shared join key
would not cover, plus per-table source pruning (ADR-25).
"""

import re
from datetime import datetime
from pathlib import Path

import pyarrow as pa
import yaml
from mbt_snowflake.adapter import SnowflakeDataAdapter
from snowflake_stub_helpers import FakeBuildContext, FakeSourceTable, StubConnection

from mbt_adapter_base import DatasetSpec, ManifestNode

REPO_ROOT = Path(__file__).resolve().parents[3]
SHOWCASE = REPO_ROOT / "examples" / "showcase"
PROJECT = SHOWCASE / "project"
DATABASE, SCHEMA = "ANALYTICS", "SANDBOX"

#: The spine universe: 10 customers x 3 month-start inference_dates.
CUSTOMERS = 10
MONTHS = [datetime(2026, m, 1) for m in (3, 4, 5)]


def _sources() -> list[dict]:
    doc = yaml.safe_load((PROJECT / "sources.yml").read_text())
    return doc["sources"][0]["tables"]


def _table_name(ref: str) -> str:
    """'source('lake', 'monthly_population')' -> 'monthly_population'."""
    return re.findall(r"'([^']*)'", ref)[1]


def test_every_source_declares_both_plane_addresses() -> None:
    """The invariant the whole three-plane design rests on.

    A table missing `path:` breaks the spark/local targets; one missing
    `identifier:` breaks the Snowflake target's compile - and it breaks it for
    EVERY table, not just the ones a cadence selects, because snapshot pinning
    covers every referenced source regardless of --select.
    """
    for table in _sources():
        assert table.get("path"), f"{table['name']} has no path: (lake planes need it)"
        assert table.get("identifier"), (
            f"{table['name']} has no identifier: (the snowflake target needs it)"
        )


def _seed_module():
    import importlib.util

    spec = importlib.util.spec_from_file_location(
        "showcase_seed_snowflake", SHOWCASE / "scripts" / "seed_snowflake.py"
    )
    assert spec is not None and spec.loader is not None
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


def test_source_identifiers_match_the_seeding_script() -> None:
    """sources.yml and seed_snowflake.py must not drift apart."""
    module = _seed_module()
    declared = {t["name"]: t["identifier"] for t in _sources()}
    seeded = {name: module.table_name(name) for name in module.TABLES}
    assert seeded == declared


def test_seeded_timestamp_columns_are_timestamps_not_numbers() -> None:
    """The regression that broke the first real run.

    The seeder used to let write_pandas(auto_create_table=True) infer types via
    Snowflake's INFER_SCHEMA, which typed the tz-naive datetime64 columns as
    NUMBER: every timestamp arrived as epoch microseconds (2025-07-01 became
    1751328000000000). The load looked clean and the failure surfaced far away,
    as a temporal split that matched no rows.

    Nothing downstream tolerates that - the split predicate, the label join,
    and monitor's maturity arithmetic all treat these as timestamps - so the
    DDL is generated explicitly now and pinned here.
    """
    module = _seed_module()

    assert module.snowflake_type("datetime64[us]") == "TIMESTAMP_NTZ"
    assert module.snowflake_type("datetime64[ns]") == "TIMESTAMP_NTZ"
    # The rest of the mapping, so a future edit cannot quietly widen it.
    assert module.snowflake_type("int64") == "NUMBER(38,0)"
    assert module.snowflake_type("int8") == "NUMBER(38,0)"
    assert module.snowflake_type("float64") == "FLOAT"
    assert module.snowflake_type("bool") == "BOOLEAN"
    assert module.snowflake_type("object") == "VARCHAR"

    # And end to end over the committed data: every datetime column of every
    # seeded table lands as TIMESTAMP_NTZ in the real DDL.
    for name, directory in module.TABLES.items():
        frame = module.read_table(directory)
        ddl = module.create_table_sql("DB", "SC", module.table_name(name), frame)
        for column, dtype in frame.dtypes.items():
            if str(dtype).startswith("datetime64"):
                assert f"{column} TIMESTAMP_NTZ" in ddl, (name, column, ddl)
    # The join key the wide cadence splits on, specifically.
    spine = module.read_table(module.TABLES["monthly_population"])
    assert str(spine.dtypes["inference_date"]).startswith("datetime64")


def _synthetic() -> dict[str, tuple[str, pa.Table]]:
    """The six wide tables, Snowflake-shaped (UPPERCASE columns).

    The spine carries the customer_id-to-safe_id crosswalk plus the two
    lineage columns; transaction_history is keyed ONLY by safe_id, so the
    panel can only be assembled through that crosswalk.
    """
    cid, sid, inf = [], [], []
    for c in range(CUSTOMERS):
        for m in MONTHS:
            cid.append(c)
            sid.append(1000 + c)
            inf.append(m)
    rows = len(cid)

    spine = pa.table(
        {
            "CUSTOMER_ID": cid,
            "SAFE_ID": sid,
            "INFERENCE_DATE": inf,
            # Lineage/audit columns: carried into the panel, never features.
            "AS_OF_DATE": [datetime(m.year, m.month, 1) for m in inf],
            "LOADED_AT_TIME": [datetime(2026, 6, 1)] * rows,
        }
    )
    # Every feature table carries the SAME-named ingest audit column, as gold
    # tables do. Three of them would collide in the panel; the specs' per-table
    # `exclude:` prunes each inside its own subquery (ADR-25), so none arrives.
    audit = {"ETL_LOADED_AT": [datetime(2026, 6, 2)] * rows}
    keys_by_customer = {"CUSTOMER_ID": cid, "INFERENCE_DATE": inf}
    # Feature tables carry the audit column; the label table does not (the real
    # monthly_labels has none, and no `exclude:` would prune it).
    by_customer = {**keys_by_customer, **audit}
    by_safe = {"SAFE_ID": sid, "INFERENCE_DATE": inf, **audit}

    return {
        "monthly_population": ("MBT_SHOWCASE_MONTHLY_POPULATION", spine),
        "monthly_labels": (
            "MBT_SHOWCASE_MONTHLY_LABELS",
            pa.table({**keys_by_customer, "IS_CHURN": [i % 2 for i in range(rows)]}),
        ),
        "demographic_history": (
            "MBT_SHOWCASE_DEMOGRAPHIC_HISTORY",
            pa.table(
                {
                    **by_customer,
                    "AGE_YEARS": [30 + c for c in cid],
                    "CONTRACT_CODE": [c % 4 for c in cid],
                }
            ),
        ),
        "login_history": (
            "MBT_SHOWCASE_LOGIN_HISTORY",
            pa.table({**by_customer, "LOGIN_DAYS_30D": [c % 30 for c in cid]}),
        ),
        "transaction_history": (
            "MBT_SHOWCASE_TRANSACTION_HISTORY",
            pa.table({**by_safe, "TXN_CNT_30D": [s % 7 for s in sid]}),
        ),
    }


def test_showcase_wide_dataset_builds_on_the_snowflake_plane(tmp_path: Path) -> None:
    """The committed wide spec, unmodified, over Snowflake-shaped tables."""
    doc = yaml.safe_load((PROJECT / "datasets" / "wide_churn_training.yml").read_text())
    spec = DatasetSpec.model_validate(doc["datasets"][0])
    synth = _synthetic()

    refs = [
        spec.inputs.spine,
        spec.inputs.label_source,
        *[entry.source for entry in spec.inputs.feature_entries],
    ]
    source_tables = {
        ref: FakeSourceTable(name=_table_name(ref), identifier=synth[_table_name(ref)][0])
        for ref in refs
    }
    stub = StubConnection(
        tables={f"{DATABASE}.{SCHEMA}.{ident}": tbl for ident, tbl in synth.values()}
    )
    adapter = SnowflakeDataAdapter({"database": DATABASE, "schema": SCHEMA})
    adapter._connection = stub  # type: ignore[assignment]

    node = ManifestNode(
        unique_id="dataset.churn_lake.wide_churn_training",
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
        # Narrowed to the synthetic months; the committed windows span a year
        # of data this fixture does not generate.
        resolved_windows={
            "train": ("2026-03-01T00:00:00Z", "2026-05-01T00:00:00Z"),
            "test": ("2026-05-01T00:00:00Z", "2026-06-01T00:00:00Z"),
        },
        sample_fraction=1.0,
        deep_snapshot=False,
        output_dir=tmp_path / "mat",
    )

    handle = adapter.build_dataset(spec, ctx)

    assert handle.splits() == {"train", "test"}
    panel = handle.read("train")
    # The spine's crosswalk and lineage columns survive; each feature history
    # contributes its payload; the label lands and its join columns are
    # projected away (no __mbt_lbl* leakage).
    assert set(panel.column_names) == {
        "customer_id",
        "safe_id",
        "inference_date",
        "as_of_date",
        "loaded_at_time",
        "age_years",
        "contract_code",
        "login_days_30d",
        "txn_cnt_30d",
        "is_churn",
    }
    # ADR-25: all three feature tables carry an identically-named
    # etl_loaded_at, and none of them reaches the panel - each was pruned
    # inside its own source subquery, so they never collided in the first
    # place. A model's features.exclude could not have saved this: the
    # collision would happen during the join, before any model sees it.
    assert "etl_loaded_at" not in panel.column_names
    # Spine-driven row counts: 10 customers x 2 month-starts in the train
    # window, 10 x 1 in test. transaction_history joined through safe_id
    # alone, so a broken crosswalk would show up as nulls or dropped rows.
    assert panel.num_rows == CUSTOMERS * 2
    assert handle.read("test").num_rows == CUSTOMERS
    assert panel.column("txn_cnt_30d").null_count == 0


def test_no_profile_env_var_default_is_a_bare_number() -> None:
    """env_var() taints; redact() censors tainted strings out of ALL serialized
    output by plain substring replacement. A short numeric value is therefore a
    footgun: `env_var('SHOWCASE_MLFLOW_PORT', '5501')` made "5501" a secret,
    and redaction then ate those digits out of the middle of an unrelated PSI
    float, producing `"value":0.1***234` and an unparseable job result.

    It fails far from the cause (the scoring job's monitor_stats carry enough
    high-precision floats to make a collision likely) and only for some runs,
    so a static check is worth more than a runtime one. Compose ports stay in
    compose; profiles reference whole URIs.
    """
    text = (PROJECT / "profiles.yml").read_text()
    defaults = re.findall(r"env_var\('[A-Z_0-9]+',\s*'([^']*)'\)", text)
    numeric = [d for d in defaults if d.strip() and d.strip().isdigit()]
    assert not numeric, (
        f"purely numeric env_var defaults in profiles.yml: {numeric}. "
        "Use a whole URI/path so redaction cannot collide with a float."
    )


def test_wide_tables_is_exactly_what_the_wide_specs_reference() -> None:
    """The seeder loads rows only for the wide cadence and creates the rest
    empty, so WIDE_TABLES must be derived from the specs, not guessed.

    Too small and the wide build reads an empty table - which fails as a
    zero-row split, far from the cause. Too large and the user's sandbox
    collects demo data for cadences this plane never runs.
    """
    module = _seed_module()

    referenced: set[str] = set()
    for spec_file in (
        PROJECT / "datasets" / "wide_churn_training.yml",
        PROJECT / "scoring" / "wide_retention_scoring.yml",
    ):
        referenced |= set(re.findall(r"source\('lake',\s*'([a-z_]+)'\)", spec_file.read_text()))

    assert referenced, "no source() references found - did the spec format change?"
    assert set(module.WIDE_TABLES) == referenced, (
        f"WIDE_TABLES drifted from the wide specs: "
        f"missing={referenced - set(module.WIDE_TABLES)} "
        f"extra={set(module.WIDE_TABLES) - referenced}"
    )
    # And every one of them is a real seedable table.
    assert set(module.WIDE_TABLES) <= set(module.TABLES)


def test_seeder_loads_with_parquet_logical_types(monkeypatch) -> None:
    """The second half of the timestamp fix, and the one with no visible symptom.

    write_pandas stages the frame as parquet and COPYs it through a generated
    FILE FORMAT. Its use_logical_type default (None) leaves USE_LOGICAL_TYPE
    unset, and Snowflake's PARQUET default for that is FALSE - so the
    TIMESTAMP(MICROS) annotation is ignored and the physical INT64 is read.
    Combined with the explicit TIMESTAMP_NTZ DDL that puts epoch integers into
    a timestamp column ("Invalid date" in Snowsight).

    The connector warns about this only for tz-AWARE columns; every datetime
    here is tz-naive, so nothing surfaces. Hence a test rather than a comment.
    """
    import snowflake.connector
    from snowflake.connector import pandas_tools

    module = _seed_module()
    calls: list[dict] = []

    class FakeCursor:
        def execute(self, sql, *a, **k):
            self.sql = sql
            return self

        def fetchall(self):
            return []  # no pre-existing tables

        def close(self):
            pass

    class FakeConnection:
        def cursor(self):
            return FakeCursor()

        def close(self):
            pass

    def fake_write_pandas(conn, df, **kwargs):
        calls.append(kwargs)
        return True, 1, len(df), None

    monkeypatch.setattr(snowflake.connector, "connect", lambda **kw: FakeConnection())
    monkeypatch.setattr(pandas_tools, "write_pandas", fake_write_pandas)
    for name, value in (
        ("SNOWFLAKE_ACCOUNT", "acct"),
        ("SNOWFLAKE_USER", "u"),
        ("SNOWFLAKE_WAREHOUSE", "WH"),
        ("SNOWFLAKE_DATABASE", "DB"),
        ("SNOWFLAKE_SCHEMA", "SC"),
        ("SNOWFLAKE_PASSWORD", "pw"),
    ):
        monkeypatch.setenv(name, value)
    monkeypatch.delenv("SNOWFLAKE_AUTHENTICATOR", raising=False)
    monkeypatch.delenv("SNOWFLAKE_PRIVATE_KEY_FILE", raising=False)

    # Default scope: rows only for the wide cadence, the rest created empty.
    assert module.main([]) == 0
    assert len(calls) == len(module.WIDE_TABLES)
    assert {k["table_name"] for k in calls} == {module.table_name(t) for t in module.WIDE_TABLES}

    # --all-cadences restores the full load for anyone pointing this target at
    # the daily/monthly pipelines.
    calls.clear()
    assert module.main(["--all-cadences"]) == 0
    assert len(calls) == len(module.TABLES)

    for kwargs in calls:
        assert kwargs["use_logical_type"] is True, kwargs
        # The other two settings this load depends on.
        assert kwargs["quote_identifiers"] is False, kwargs
        assert kwargs["auto_create_table"] is False, kwargs


def test_snowflake_target_renders_without_credentials(monkeypatch) -> None:
    """profiles.yml renders WHOLE for whichever target is picked, so every
    env_var() the snowflake target adds needs a default or it breaks the lake
    targets too. AWS_* are the pre-existing exception - the s3a anchor calls
    them with no default, which is why a host-run snowflake command needs them
    even though it never touches s3a.
    """
    from mbt.config.profiles import load_profiles
    from mbt.parsing import parse_project

    monkeypatch.setenv("AWS_ACCESS_KEY_ID", "stub")
    monkeypatch.setenv("AWS_SECRET_ACCESS_KEY", "stub")
    for name in ("SNOWFLAKE_ACCOUNT", "SNOWFLAKE_USER", "SNOWFLAKE_DATABASE"):
        monkeypatch.delenv(name, raising=False)

    parsed = parse_project(PROJECT)
    loaded = load_profiles(
        parsed.project.name, PROJECT, target_override="snowflake", project_vars=parsed.project.vars
    )
    assert loaded.target.data.adapter == "snowflake"
    # Namespaced registration, and its own artifact prefix so the two planes
    # never interleave in the shared store.
    assert loaded.target.vars["plane_suffix"] == "_snowflake"
    assert loaded.target.artifact_store == "s3://mbt-artifacts/churn_snowflake"
    # Published host ports, not in-network service names: this target runs on
    # the host, where `mlflow` does not resolve.
    assert "localhost" in loaded.target.registry.config["uri"]

    # The lake targets keep the empty suffix, so their registered names and
    # config hashes are untouched by this plane's existence.
    for target in ("dev", "ci", "prod", "prod_score"):
        other = load_profiles(
            parsed.project.name, PROJECT, target_override=target, project_vars=parsed.project.vars
        )
        merged = {**parsed.project.vars, **other.target.vars}
        assert merged["plane_suffix"] == "", target


def test_transaction_history_joins_through_safe_id_only() -> None:
    """The heterogeneous-key claim, asserted against the committed spec."""
    doc = yaml.safe_load((PROJECT / "datasets" / "wide_churn_training.yml").read_text())
    spec = DatasetSpec.model_validate(doc["datasets"][0])
    using = {_table_name(e.source): list(e.using) for e in spec.inputs.feature_entries}
    assert using["demographic_history"] == ["customer_id", "inference_date"]
    assert using["login_history"] == ["customer_id", "inference_date"]
    assert using["transaction_history"] == ["safe_id", "inference_date"]
