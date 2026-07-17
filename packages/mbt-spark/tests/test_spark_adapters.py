"""Spark adapter tests: data plane, SparkML training compliance, and the
spark-submit compute seam. Heavy (JVM); runs under the e2e marker."""

import hashlib
import shutil
import subprocess
import sys
from datetime import datetime, timedelta
from pathlib import Path
from typing import Any, ClassVar

import pyarrow as pa
import pyarrow.parquet as pq
import pytest
from mbt_spark.data import SparkDataAdapter
from mbt_spark.training import SparkMLTrainingAdapter

from mbt_adapter_base import DatasetSpec, ManifestNode
from mbt_adapter_base.compliance import TrainingAdapterCompliance
from mbt_adapter_base.materialization import combine_snapshots

pytestmark = [
    pytest.mark.e2e,
    pytest.mark.skipif(shutil.which("java") is None, reason="Spark needs a JVM"),
]

ANCHOR = datetime(2026, 7, 1)
WINDOWS = {
    "train": ("2026-01-02T00:00:00Z", "2026-06-03T00:00:00Z"),
    "test": ("2026-06-03T00:00:00Z", "2026-07-01T00:00:00Z"),
}
LABEL_UID = "source.p.lake.labels"
USAGE_UID = "source.p.lake.usage"


# -- fixtures -----------------------------------------------------------------------


def _write_tables(root: Path, n: int = 400) -> None:
    dates = [ANCHOR - timedelta(days=(i * 179) % 180 + 1) for i in range(n)]
    signal = [((i * 37) % 100) / 100.0 for i in range(n)]
    pq.write_table(
        pa.table(
            {
                "customer_id": list(range(n)),
                "snapshot_date": dates,
                "churned_90d": [1 if s > 0.55 else 0 for s in signal],
            }
        ),
        _mk(root / "labels") / "part-000.parquet",
    )
    pq.write_table(
        pa.table(
            {
                "customer_id": list(range(n)),
                "snapshot_date": dates,
                "monthly_usage": signal,
                "support_tickets": [i % 7 for i in range(n)],
            }
        ),
        _mk(root / "usage") / "part-000.parquet",
    )


def _mk(path: Path) -> Path:
    path.mkdir(parents=True, exist_ok=True)
    return path


class FakeSourceTable:
    def __init__(self, name: str, path: str) -> None:
        self.name = name
        self.path = path
        self.identifier = None
        self.format = "parquet"


class FakeBuildContext:
    def __init__(
        self,
        node: ManifestNode,
        source_tables: dict[str, Any],
        spine: str,
        output_dir: Path,
        sample_fraction: float = 1.0,
    ) -> None:
        self.node = node
        self.source_tables = source_tables
        self.source = source_tables[spine]
        self.resolved_windows = WINDOWS
        self.sample_fraction = sample_fraction
        self.deep_snapshot = False
        self.output_dir = output_dir
        self.events = None


def _spec(**overrides: Any) -> DatasetSpec:
    base: dict[str, Any] = {
        "name": "churn_spark",
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


@pytest.fixture(scope="module")
def source_root(tmp_path_factory: pytest.TempPathFactory) -> Path:
    root = tmp_path_factory.mktemp("spark-src")
    _write_tables(root)
    return root


def _sources(root: Path) -> dict[str, Any]:
    return {
        LABEL_UID: FakeSourceTable("labels", str(root / "labels" / "*.parquet")),
        USAGE_UID: FakeSourceTable("usage", str(root / "usage" / "*.parquet")),
    }


def _ctx(
    root: Path, out: Path, adapter: SparkDataAdapter, sample_fraction: float = 1.0
) -> FakeBuildContext:
    sources = _sources(root)
    pinned = combine_snapshots({uid: adapter.snapshot_id(t) for uid, t in sources.items()})
    node = ManifestNode(
        unique_id="dataset.p.churn_spark",
        resource_type="dataset",
        name="churn_spark",
        path="datasets/churn_spark.yml",
        config={},
        snapshot_id=pinned,
    )
    return FakeBuildContext(node, sources, LABEL_UID, out, sample_fraction)


# -- data plane -----------------------------------------------------------------------


def test_build_dataset_joins_windows_and_reproducible_sampling(
    source_root: Path, tmp_path: Path
) -> None:
    adapter = SparkDataAdapter({"master": "local[2]"})
    spec = _spec()
    handle = adapter.build_dataset(spec, _ctx(source_root, tmp_path / "full", adapter))
    assert handle.splits() == {"train", "test"}
    train = handle.read("train")
    assert {
        "customer_id",
        "snapshot_date",
        "churned_90d",
        "monthly_usage",
        "support_tickets",
    } <= set(train.column_names)
    profile = handle.profile()
    assert profile.label_balance and set(profile.label_balance) == {"0", "1"}

    def ids(out: str, fraction: float) -> set[int]:
        h = adapter.build_dataset(
            spec, _ctx(source_root, tmp_path / out, adapter, sample_fraction=fraction)
        )
        rows: set[int] = set()
        for split in ("train", "test"):
            rows |= set(h.read(split).column("customer_id").to_pylist())
        return rows

    half_a, half_b, fifth = ids("a", 0.5), ids("b", 0.5), ids("c", 0.2)
    assert half_a == half_b  # same fraction -> same rows
    assert fifth <= half_a and 0 < len(fifth) < len(half_a) < 400

    # reopening needs no Spark session; snapshot pin is enforced
    fresh = SparkDataAdapter({})
    reopened = fresh.from_locator(handle.locator())
    assert reopened.read("test").num_rows == handle.read("test").num_rows


def test_population_spine_with_label_offset_on_spark(tmp_path: Path) -> None:
    """ADR-22 on the Spark plane: population spine, per-table using columns,
    and the calendar-month label offset via an expression join."""
    months = [datetime(2026, m, 1) for m in range(1, 8)]
    root = _mk(tmp_path / "src")
    population_rows = [(cid, f"sf-{cid}", when) for when in months[:-1] for cid in range(40)]
    pq.write_table(
        pa.table(
            {
                "customer_id": [r[0] for r in population_rows],
                "safe_id": [r[1] for r in population_rows],
                "snapshot_date": [r[2] for r in population_rows],
            }
        ),
        _mk(root / "population") / "part-000.parquet",
    )
    label_rows = [
        (cid, months[i + 1], (cid + i) % 2) for i in range(len(months) - 1) for cid in range(40)
    ]
    pq.write_table(
        pa.table(
            {
                "customer_id": [r[0] for r in label_rows],
                "snapshot_date": [r[1] for r in label_rows],
                "churned": [r[2] for r in label_rows],
            }
        ),
        _mk(root / "monthly_labels") / "part-000.parquet",
    )
    pq.write_table(
        pa.table(
            {
                "safe_id": [r[1] for r in population_rows],
                "snapshot_date": [r[2] for r in population_rows],
                "txn_total": [float(r[0] % 300) for r in population_rows],
            }
        ),
        _mk(root / "txn") / "part-000.parquet",
    )
    pop_uid = "source.p.lake.population"
    lbl_uid = "source.p.lake.monthly_labels"
    txn_uid = "source.p.lake.txn"
    sources = {
        pop_uid: FakeSourceTable("population", str(root / "population" / "*.parquet")),
        lbl_uid: FakeSourceTable("monthly_labels", str(root / "monthly_labels" / "*.parquet")),
        txn_uid: FakeSourceTable("txn", str(root / "txn" / "*.parquet")),
    }
    spec = DatasetSpec.model_validate(
        {
            "name": "wide_spark",
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
    adapter = SparkDataAdapter({"master": "local[2]"})
    pinned = combine_snapshots({uid: adapter.snapshot_id(t) for uid, t in sources.items()})
    node = ManifestNode(
        unique_id="dataset.p.wide_spark",
        resource_type="dataset",
        name="wide_spark",
        path="datasets/wide_spark.yml",
        config={},
        snapshot_id=pinned,
    )
    ctx = FakeBuildContext(node, sources, pop_uid, tmp_path / "mat")
    ctx.resolved_windows = {
        "train": ("2026-01-01T00:00:00Z", "2026-05-01T00:00:00Z"),
        "test": ("2026-05-01T00:00:00Z", "2026-07-01T00:00:00Z"),
    }
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
        assert table.num_rows > 0
        for row in table.to_pylist():
            expected = (row["customer_id"] + month_index[row["snapshot_date"]]) % 2
            assert row["churned"] == expected


def test_snapshot_changes_when_source_files_change(source_root: Path) -> None:
    adapter = SparkDataAdapter({})
    table = _sources(source_root)[LABEL_UID]
    before = adapter.snapshot_id(table)
    _write_tables(source_root, n=410)
    assert adapter.snapshot_id(table) != before
    _write_tables(source_root, n=400)  # restore for other tests


def test_snapshot_uri_branch_for_relative_path_under_uri_root() -> None:
    """A relative table path under a URI root is a URI source: the snapshot
    must hash the cluster's file listing, never glob the resolved pattern
    locally (which always finds nothing). Surfaced end-to-end by the
    showcase's SeaweedFS lake (root s3://mbt-lake + path <table>/*.parquet)."""
    adapter = SparkDataAdapter({"root": "s3://lake"})
    table = FakeSourceTable("events", "events/*.parquet")
    listing = ["s3://lake/events/part-001.parquet", "s3://lake/events/part-000.parquet"]

    class FakeFrame:
        def inputFiles(self) -> list[str]:
            return listing

    adapter._read = lambda source: FakeFrame()  # type: ignore[method-assign]
    digest = hashlib.sha256()
    for uri in sorted(listing):
        digest.update(uri.encode())
        digest.update(b"\n")
    assert adapter.snapshot_id(table) == "sha256:" + digest.hexdigest()


# -- training plane: the compliance suite (FR-ADPT-05) ---------------------------------


class TestSparkMLCompliance(TrainingAdapterCompliance):
    adapter_factory = SparkMLTrainingAdapter
    plugin_module = "mbt_spark.plugin"
    framework_modules = ("pyspark",)
    valid_hyperparameters: ClassVar[dict] = {"max_iter": 5, "max_depth": 3}
    auto_hyperparameter = None


# -- compute plane: the spark-submit seam (ADR-3 / TSD §22) ----------------------------


def test_spark_submit_runs_a_training_job(source_root: Path, tmp_path: Path) -> None:
    from mbt_spark.compute import SparkComputeAdapter

    from mbt_adapter_base import AdapterRef, MetricSpec, TrainingJob

    # a real materialization for the fake training adapter to consume
    data_adapter = SparkDataAdapter({"master": "local[1]"})
    spec = _spec()
    handle = data_adapter.build_dataset(spec, _ctx(source_root, tmp_path / "mat", data_adapter))

    node = ManifestNode(
        unique_id="model.p.compute_seam",
        resource_type="model",
        name="compute_seam",
        path="models/compute_seam.yml",
        config={
            "name": "compute_seam",
            "task": "binary_classification",
            "adapter": "fake",
            "owner": "t@example.com",
            "dataset": "ref('churn_spark')",
            "target": "churned_90d",
            "hyperparameters": {"fake_metric_value": 0.66},
            "evaluation": {"protocol": {"split": "temporal"}, "metrics": ["pr_auc"]},
            "seed": 5,
        },
        seed=5,
        adapter="fake",
        task="binary_classification",
    )
    job = TrainingJob(
        run_id="spark-seam-test",
        project_dir=str(tmp_path),
        target_name="dev",
        node=node,
        dataset=handle.locator(),
        data=AdapterRef(adapter="spark"),
        tracking=None,
        metric_specs=[MetricSpec(name="pr_auc", kind="builtin")],
        artifact_store=f"file://{tmp_path}/artifacts",
    )
    compute = SparkComputeAdapter({"master": "local[1]"})
    result = compute.wait(compute.submit(job))
    assert result.status == "success", result.error
    assert result.metrics is not None
    assert result.metrics.metrics["pr_auc"] == pytest.approx(0.66, abs=0.01)
    assert result.artifact is not None and result.artifact.format == "fake_json"


def test_plugin_import_hygiene() -> None:
    probe = (
        "import sys\n"
        "import mbt_spark.plugin\n"
        "loaded = [m for m in sys.modules if m.startswith('pyspark')]\n"
        "assert not loaded, f'pyspark imported at plugin load: {loaded}'\n"
    )
    subprocess.run([sys.executable, "-c", probe], check=True)
