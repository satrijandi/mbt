"""Live Snowflake integration suite (opt-in, B1 of the integration-test plan).

The unit tests in ``test_snowflake_adapter.py`` execute the adapter's
generated SQL in DuckDB; this module proves the same behavior on a real
account - the dialect surfaces a stand-in cannot vouch for
(``MD5_NUMBER_LOWER64``, ``SYSTEM$LAST_CHANGE_COMMIT_TIME``, ``HASH_AGG``,
Arrow batch streaming, identifier case rules) plus the full data-scientist
scenario: `mbt build` on a laptop training a model straight from warehouse
tables, then reproducing it bit-for-bit from the pinned manifest.

Gating: every test skips unless MBT_LIVE_SNOWFLAKE=1 (so credentials in the
shell can never trigger surprise warehouse traffic or SSO browser prompts);
once opted in, incomplete SNOWFLAKE_* configuration fails loudly instead of
skipping. Auth is whatever the env says: SNOWFLAKE_PASSWORD,
SNOWFLAKE_AUTHENTICATOR=externalbrowser (SSO token caching is enabled so
the whole session needs one browser prompt), or SNOWFLAKE_PRIVATE_KEY_FILE.
See the package README for setup.

The suite creates uniquely named MBT_LIVE_* tables in the configured
database.schema (schema-level CREATE TABLE is the only privilege needed)
and drops them at teardown.
"""

import contextlib
import json
import os
import subprocess
import sys
import uuid
from collections.abc import Iterator
from dataclasses import dataclass, field
from datetime import date, timedelta
from pathlib import Path
from typing import Any

import pyarrow.parquet as pq
import pytest
from mbt_snowflake.adapter import _CONNECT_KEYS, SnowflakeAdapterError, SnowflakeDataAdapter

from mbt_adapter_base import DatasetSpec, ManifestNode
from mbt_adapter_base.materialization import combine_snapshots

pytestmark = [
    pytest.mark.live,
    pytest.mark.live_snowflake,
    pytest.mark.timeout(900),
    pytest.mark.skipif(
        os.environ.get("MBT_LIVE_SNOWFLAKE") != "1",
        reason="live Snowflake tests are opt-in: set MBT_LIVE_SNOWFLAKE=1 "
        "plus SNOWFLAKE_* env vars (packages/mbt-snowflake/README.md)",
    ),
]

_REQUIRED_ENV = (
    "SNOWFLAKE_ACCOUNT",
    "SNOWFLAKE_USER",
    "SNOWFLAKE_WAREHOUSE",
    "SNOWFLAKE_DATABASE",
    "SNOWFLAKE_SCHEMA",
)
_AUTH_ENV = ("SNOWFLAKE_PASSWORD", "SNOWFLAKE_AUTHENTICATOR", "SNOWFLAKE_PRIVATE_KEY_FILE")

#: Matches the seeded snapshot_date spread (2026-01-01 .. 2026-06-28).
ANCHOR = "2026-06-30T00:00:00Z"
WINDOWS = {
    "train": ("2026-01-01T00:00:00Z", "2026-06-02T00:00:00Z"),
    "test": ("2026-06-02T00:00:00Z", "2026-06-30T00:00:00Z"),
}
TEST_WINDOW_START = date(2026, 6, 2)

LABELS_UID = "source.live_snowflake.snowflake.churn_labels"
FEATURES_UID = "source.live_snowflake.snowflake.usage_features"


def _adapter_config() -> dict[str, Any]:
    """Adapter config from SNOWFLAKE_* env vars - the same keys a data
    scientist puts in profiles.yml (account/user/warehouse/database/schema,
    plus whichever of password/role/authenticator/key-pair applies)."""
    config: dict[str, Any] = {
        key: os.environ[f"SNOWFLAKE_{key.upper()}"]
        for key in ("account", "user", "warehouse", "database", "schema")
    }
    for key in ("password", "role", "authenticator"):
        value = os.environ.get(f"SNOWFLAKE_{key.upper()}")
        if value:
            config[key] = value
    connect_args: dict[str, Any] = {}
    if os.environ.get("SNOWFLAKE_PRIVATE_KEY_FILE"):
        connect_args["private_key_file"] = os.environ["SNOWFLAKE_PRIVATE_KEY_FILE"]
        if os.environ.get("SNOWFLAKE_PRIVATE_KEY_FILE_PWD"):
            connect_args["private_key_file_pwd"] = os.environ["SNOWFLAKE_PRIVATE_KEY_FILE_PWD"]
    if config.get("authenticator") == "externalbrowser":
        # The adapter defaults this on, but the suite's own seeding
        # connection calls snowflake.connector.connect directly - inject it
        # there too so ONE browser prompt covers the whole session.
        connect_args["client_store_temporary_credential"] = True
    if connect_args:
        config["connect_args"] = connect_args
    return config


def _seed_rows(n: int = 600) -> list[dict[str, Any]]:
    """Deterministic learnable data, same signal shape as the JVM e2e."""
    base = date(2026, 1, 1)
    rows = []
    for i in range(n):
        signal = ((i * 37) % 100) / 100.0
        noise = ((i * 7919) % 100) / 100.0
        rows.append(
            {
                "customer_id": i,
                "snapshot_date": base + timedelta(days=(i * 131) % 180),
                "monthly_usage": signal,
                "tenure_days": noise,
                "churned_90d": 1 if signal + 0.2 * noise > 0.65 else 0,
            }
        )
    return rows


@dataclass
class LiveWarehouse:
    """Session handle: adapter config, a seeding connection, seeded tables."""

    config: dict[str, Any]
    connection: Any
    database: str
    schema: str
    prefix: str
    labels_table: str
    features_table: str
    rows: list[dict[str, Any]]
    created_tables: list[str] = field(default_factory=list)

    def qualified(self, table: str) -> str:
        return f"{self.database}.{self.schema}.{table}"

    def execute(self, sql: str, rows: list[tuple[Any, ...]] | None = None) -> None:
        cursor = self.connection.cursor()
        try:
            if rows is None:
                cursor.execute(sql)
            else:
                cursor.executemany(sql, rows)
        finally:
            cursor.close()

    def create_table(self, table: str, columns: str) -> str:
        self.execute(f"CREATE TABLE {self.qualified(table)} ({columns})")
        self.created_tables.append(table)
        return table


@pytest.fixture(scope="session")
def live() -> Iterator[LiveWarehouse]:
    missing = [name for name in _REQUIRED_ENV if not os.environ.get(name)]
    if missing:
        pytest.fail(
            f"MBT_LIVE_SNOWFLAKE=1 but env is incomplete: set {', '.join(missing)} "
            "(packages/mbt-snowflake/README.md documents the setup)"
        )
    if not any(os.environ.get(name) for name in _AUTH_ENV):
        pytest.fail(
            "MBT_LIVE_SNOWFLAKE=1 but no auth is configured: set one of "
            f"{', '.join(_AUTH_ENV)} (externalbrowser opens your SSO browser)"
        )
    import snowflake.connector

    config = _adapter_config()
    kwargs = {key: config[key] for key in _CONNECT_KEYS if config.get(key)}
    kwargs.update(config.get("connect_args", {}))
    connection = snowflake.connector.connect(**kwargs)

    prefix = f"MBT_LIVE_{uuid.uuid4().hex[:8].upper()}"
    warehouse = LiveWarehouse(
        config=config,
        connection=connection,
        database=config["database"],
        schema=config["schema"],
        prefix=prefix,
        labels_table=f"{prefix}_LABELS",
        features_table=f"{prefix}_FEATURES",
        rows=_seed_rows(),
    )
    try:
        warehouse.create_table(
            warehouse.labels_table, "CUSTOMER_ID INTEGER, SNAPSHOT_DATE DATE, CHURNED_90D INTEGER"
        )
        warehouse.create_table(
            warehouse.features_table,
            "CUSTOMER_ID INTEGER, SNAPSHOT_DATE DATE, MONTHLY_USAGE FLOAT, TENURE_DAYS FLOAT",
        )
        warehouse.execute(
            f"INSERT INTO {warehouse.qualified(warehouse.labels_table)} VALUES (%s, %s, %s)",
            [(r["customer_id"], r["snapshot_date"], r["churned_90d"]) for r in warehouse.rows],
        )
        warehouse.execute(
            f"INSERT INTO {warehouse.qualified(warehouse.features_table)} VALUES (%s, %s, %s, %s)",
            [
                (r["customer_id"], r["snapshot_date"], r["monthly_usage"], r["tenure_days"])
                for r in warehouse.rows
            ],
        )
        yield warehouse
    finally:
        for table in warehouse.created_tables:
            with contextlib.suppress(Exception):
                warehouse.execute(f"DROP TABLE IF EXISTS {warehouse.qualified(table)}")
        connection.close()


# -- adapter-level context plumbing (mirrors the unit tests) ---------------------------


@dataclass
class SourceTable:
    name: str
    identifier: str
    path: str | None = None
    format: str = "snowflake"


@dataclass
class BuildContext:
    node: ManifestNode
    source: SourceTable
    source_tables: dict[str, SourceTable]
    resolved_windows: dict[str, tuple[str, str]]
    sample_fraction: float
    deep_snapshot: bool
    output_dir: Path
    events: Any = None


def _dataset_spec(**overrides: Any) -> DatasetSpec:
    base: dict[str, Any] = {
        "name": "churn_training_set",
        "inputs": {
            "label": LABELS_UID,
            "features": [FEATURES_UID],
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


def _sources(live: LiveWarehouse) -> dict[str, SourceTable]:
    # bare table names: database/schema qualification comes from the config
    return {
        LABELS_UID: SourceTable(name="churn_labels", identifier=live.labels_table),
        FEATURES_UID: SourceTable(name="usage_features", identifier=live.features_table),
    }


def _ctx(
    adapter: SnowflakeDataAdapter,
    spec: DatasetSpec,
    sources: dict[str, SourceTable],
    output_dir: Path,
    sample_fraction: float = 1.0,
) -> BuildContext:
    pinned = combine_snapshots({uid: adapter.snapshot_id(t) for uid, t in sources.items()})
    spine_uid = spec.inputs.label if spec.inputs is not None else spec.source
    assert spine_uid is not None
    node = ManifestNode(
        unique_id=f"dataset.live_snowflake.{spec.name}",
        resource_type="dataset",
        name=spec.name,
        path=f"datasets/{spec.name}.yml",
        config={},
        snapshot_id=pinned,
    )
    return BuildContext(
        node=node,
        source=sources[spine_uid],
        source_tables=sources,
        resolved_windows=WINDOWS,
        sample_fraction=sample_fraction,
        deep_snapshot=False,
        output_dir=output_dir,
    )


# -- adapter-level live tests ----------------------------------------------------------


def test_multi_table_dataset_round_trip(live: LiveWarehouse, tmp_path: Path) -> None:
    """Join push-down, temporal windows, Arrow streaming, case normalization."""
    adapter = SnowflakeDataAdapter(live.config)
    spec = _dataset_spec()
    ctx = _ctx(adapter, spec, _sources(live), tmp_path / "mat")
    handle = adapter.build_dataset(spec, ctx)

    train = pq.read_table(ctx.output_dir / "train.parquet")
    test = pq.read_table(ctx.output_dir / "test.parquet")
    # unquoted identifiers came back UPPERCASE and were normalized; the join
    # deduplicated the key columns across the two tables
    expected_columns = {
        "customer_id",
        "snapshot_date",
        "churned_90d",
        "monthly_usage",
        "tenure_days",
    }
    assert set(train.column_names) == expected_columns
    assert set(test.column_names) == expected_columns
    # split windows partition the seeded rows exactly as computed locally
    expected_test = sum(1 for r in live.rows if r["snapshot_date"] >= TEST_WINDOW_START)
    assert test.num_rows == expected_test
    assert train.num_rows == len(live.rows) - expected_test
    assert handle.snapshot_id == ctx.node.snapshot_id


def test_push_down_sampling_reproducible_and_monotone_live(
    live: LiveWarehouse, tmp_path: Path
) -> None:
    """MD5_NUMBER_LOWER64 threshold sampling on the real engine: same
    fraction -> same rows; smaller fractions are subsets of larger ones."""
    adapter = SnowflakeDataAdapter(live.config)
    spec = _dataset_spec()
    sources = _sources(live)

    def sampled_ids(fraction: float, name: str) -> set[int]:
        ctx = _ctx(adapter, spec, sources, tmp_path / name, sample_fraction=fraction)
        adapter.build_dataset(spec, ctx)
        ids: set[int] = set()
        for split in ("train", "test"):
            column = pq.read_table(ctx.output_dir / f"{split}.parquet")["customer_id"]
            ids.update(column.to_pylist())
        return ids

    half_a = sampled_ids(0.5, "half_a")
    half_b = sampled_ids(0.5, "half_b")
    quarter = sampled_ids(0.25, "quarter")
    full = sampled_ids(1.0, "full")
    assert half_a == half_b  # reproducible across runs
    assert quarter <= half_a <= full  # monotone subsets
    assert 0 < len(quarter) < len(half_a) < len(full) == len(live.rows)


def test_snapshot_tokens_track_dml_and_guard_pins(live: LiveWarehouse, tmp_path: Path) -> None:
    """SYSTEM$LAST_CHANGE_COMMIT_TIME and HASH_AGG tokens are stable while
    data holds still, move on DML, and a stale manifest pin fails the build."""
    table = live.create_table(
        f"{live.prefix}_SNAP", "CUSTOMER_ID INTEGER, SNAPSHOT_DATE DATE, CHURNED_90D INTEGER"
    )
    live.execute(
        f"INSERT INTO {live.qualified(table)} VALUES (%s, %s, %s)",
        [(r["customer_id"], r["snapshot_date"], r["churned_90d"]) for r in live.rows[:20]],
    )
    uid = "source.live_snowflake.snowflake.snap"
    source = SourceTable(name="snap", identifier=table)
    adapter = SnowflakeDataAdapter(live.config)

    shallow = adapter.snapshot_id(source)
    deep = adapter.snapshot_id(source, deep=True)
    assert shallow == adapter.snapshot_id(source)  # stable while data holds still
    assert deep == adapter.snapshot_id(source, deep=True)
    assert shallow != deep

    spec = _dataset_spec(name="snap_set", inputs=None, source=uid)
    stale = _ctx(adapter, spec, {uid: source}, tmp_path / "stale")  # pins the current tokens

    live.execute(
        f"INSERT INTO {live.qualified(table)} VALUES (%s, %s, %s)", [(9999, date(2026, 3, 1), 1)]
    )
    assert adapter.snapshot_id(source) != shallow  # DML moved both tokens
    assert adapter.snapshot_id(source, deep=True) != deep
    with pytest.raises(SnowflakeAdapterError, match="changed under the pinned manifest"):
        adapter.build_dataset(spec, stale)

    fresh = _ctx(adapter, spec, {uid: source}, tmp_path / "fresh")
    adapter.build_dataset(spec, fresh)  # a fresh pin verifies and builds


# -- the data-scientist scenario, end to end ---------------------------------------------


def _mbt(args: list[str], project: Path, *, timeout: int = 600) -> None:
    proc = subprocess.run(
        [sys.executable, "-m", "mbt.cli.main", *args],
        cwd=project,
        env=os.environ.copy(),
        capture_output=True,
        text=True,
        timeout=timeout,
        stdin=subprocess.DEVNULL,
        check=False,
    )
    assert proc.returncode == 0, (
        f"mbt {' '.join(args)} exited {proc.returncode}\n"
        f"stdout:\n{proc.stdout}\nstderr:\n{proc.stderr}"
    )


def _write_project(live: LiveWarehouse, project: Path) -> Path:
    """A minimal project exactly as a data scientist would write it: the
    snowflake profile via env_var() (secrets never land in files), warehouse
    sources by identifier, and a local xgboost training loop."""
    (project / "datasets").mkdir(parents=True)
    (project / "models").mkdir()
    (project / "mbt_project.yml").write_text('name: live_snowflake\nversion: "0.1.0"\n')

    config_lines = []
    for key in ("account", "user", "warehouse", "database", "schema"):
        config_lines.append(f"          {key}: \"{{{{ env_var('SNOWFLAKE_{key.upper()}') }}}}\"")
    for key in ("password", "role", "authenticator"):
        if os.environ.get(f"SNOWFLAKE_{key.upper()}"):
            config_lines.append(
                f"          {key}: \"{{{{ env_var('SNOWFLAKE_{key.upper()}') }}}}\""
            )
    connect_args = []
    if os.environ.get("SNOWFLAKE_PRIVATE_KEY_FILE"):
        connect_args.append(
            "            private_key_file: \"{{ env_var('SNOWFLAKE_PRIVATE_KEY_FILE') }}\""
        )
    # No client_store_temporary_credential here on purpose: with
    # authenticator externalbrowser the ADAPTER defaults it on, and this
    # generated profile is where the live suite proves that default.
    if connect_args:
        config_lines.append("          connect_args:")
        config_lines.extend(connect_args)

    (project / "profiles.yml").write_text(
        "live_snowflake:\n"
        "  target: live\n"
        "  outputs:\n"
        "    live:\n"
        "      data:\n"
        "        adapter: snowflake\n"
        "        config:\n" + "\n".join(config_lines) + "\n"
        f"      tracking: {{adapter: mlflow, config: {{uri: 'sqlite:///{project}/mlflow.db'}}}}\n"
        f"      registry: {{adapter: mlflow, config: {{uri: 'sqlite:///{project}/mlflow.db'}}}}\n"
        "      compute: {adapter: local}\n"
        f"      artifact_store: file://{project}/target/artifacts\n"
        "      vars: {sample_fraction: 1.0}\n"
    )
    (project / "sources.yml").write_text(
        "sources:\n"
        "  - name: snowflake\n"
        "    tables:\n"
        "      - name: churn_labels\n"
        f"        identifier: {live.labels_table}\n"
        "      - name: usage_features\n"
        f"        identifier: {live.features_table}\n"
    )
    (project / "datasets" / "churn_training_set.yml").write_text(
        "datasets:\n"
        "  - name: churn_training_set\n"
        "    inputs:\n"
        "      label: source('snowflake', 'churn_labels')\n"
        "      features:\n"
        "        - source('snowflake', 'usage_features')\n"
        "      join_key: [customer_id, snapshot_date]\n"
        "    label: {column: churned_90d}\n"
        "    sample_key: [customer_id]\n"
        "    split:\n"
        "      strategy: temporal\n"
        "      time_column: snapshot_date\n"
        '      train: "-180d:-28d"\n'
        '      test: "-28d:now"\n'
    )
    (project / "models" / "churn_classifier.yml").write_text(
        "models:\n"
        "  - name: churn_classifier\n"
        "    task: binary_classification\n"
        "    adapter: xgboost\n"
        "    owner: live@example.com\n"
        "    dataset: ref('churn_training_set')\n"
        "    target: churned_90d\n"
        "    features: {exclude: [customer_id, snapshot_date]}\n"
        "    hyperparameters: {max_depth: 3, n_estimators: 50}\n"
        "    evaluation:\n"
        "      protocol: {split: temporal}\n"
        "      metrics: [pr_auc, roc_auc]\n"
        "      gates:\n"
        "        - {metric: roc_auc, threshold: 0.7}\n"
        "    registration: {name: churn_classifier}\n"
        "    seed: 42\n"
    )
    return project


def test_full_local_training_loop_from_live_snowflake(live: LiveWarehouse, tmp_path: Path) -> None:
    """The scenario a data scientist runs from a laptop: profiles point at
    the production warehouse (SSO or key-pair), `mbt build` materializes the
    joined training set out of Snowflake and trains locally, gates pass, the
    model registers - then `mbt run --manifest` reproduces the metrics
    bit-for-bit, verifying the snapshot pins against the live tables."""
    project = _write_project(live, tmp_path / "live_project")
    _mbt(["build", "--anchor", ANCHOR], project)

    payload = json.loads((project / "target" / "run_results.json").read_text())
    results = {r["unique_id"]: r for r in payload["results"]}
    assert {uid: r["status"] for uid, r in results.items()} == {
        "dataset.live_snowflake.churn_training_set": "success",
        "model.live_snowflake.churn_classifier": "success",
    }
    model = results["model.live_snowflake.churn_classifier"]
    assert model["metrics"]["roc_auc"] > 0.7  # the gate actually gated
    assert model["registration"]["version"] == "1"
    baseline = model["metrics"]

    # the manifest pinned real warehouse snapshots at compile time
    manifest = json.loads((project / "target" / "manifest.json").read_text())
    dataset_node = manifest["nodes"]["dataset.live_snowflake.churn_training_set"]
    assert dataset_node["snapshot_id"], "dataset snapshot was not pinned"

    _mbt(["run", "--manifest", "target/manifest.json"], project)
    rerun = json.loads((project / "target" / "run_results.json").read_text())
    reproduced = {r["unique_id"]: r for r in rerun["results"]}
    assert reproduced["model.live_snowflake.churn_classifier"]["metrics"] == baseline
