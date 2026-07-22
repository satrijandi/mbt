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
from mbt_spark.data import SparkAdapterError, SparkDataAdapter
from mbt_spark.training import SparkMLTrainingAdapter

from mbt_adapter_base import (
    DatasetSpec,
    ManifestNode,
    RunContext,
    ScoringInputSpec,
    ScoringOutputSpec,
    TaskType,
)
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


class _CapturingSink:
    def __init__(self) -> None:
        self.messages: list[Any] = []

    def emit(self, event: Any) -> None:
        self.messages.append(event)


class FakeBuildContext:
    def __init__(
        self,
        node: ManifestNode,
        source_tables: dict[str, Any],
        spine: str,
        output_dir: Path,
        sample_fraction: float = 1.0,
        resolved_windows: dict[str, tuple[str, str]] | None = None,
    ) -> None:
        self.node = node
        self.source_tables = source_tables
        self.source = source_tables[spine]
        self.resolved_windows = WINDOWS if resolved_windows is None else resolved_windows
        self.sample_fraction = sample_fraction
        self.deep_snapshot = False
        self.output_dir = output_dir
        # The real BuildContext always carries a live sink; capture so
        # success-path emits (row counts) have somewhere to go.
        self.events = _CapturingSink()


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
    ctx = _ctx(source_root, tmp_path / "full", adapter)
    handle = adapter.build_dataset(spec, ctx)
    assert handle.splits() == {"train", "test"}
    # the successful build reports its per-split row counts on the bus
    row_logs = [str(m) for m in ctx.events.messages if "materialized" in str(m)]
    assert len(row_logs) == 1 and "train=" in row_logs[0] and "test=" in row_logs[0]
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
    # F19: sampling uses the canonical cross-adapter digest, so the kept set
    # matches the same Python reference the local/snowflake tests pin to
    import hashlib

    def reference_bucket(preimage: str) -> int:
        return int(hashlib.md5(preimage.encode()).hexdigest()[16:32], 16) % 1_000_000

    sampled_ids = set(half_a)
    all_ids = {
        cid
        for split in ("train", "test")
        for cid in handle.read(split).column("customer_id").to_pylist()
    }
    assert sampled_ids == {cid for cid in all_ids if reference_bucket(str(cid)) < 500_000}
    # and the salted variant (random-split bucketing) matches the reference too
    from mbt_spark.data import key_hash_sql

    spark = adapter._spark()
    probe = spark.range(50).selectExpr("CAST(id AS INT) AS customer_id")
    probe.createOrReplaceTempView("mbt_parity_probe")
    expr = key_hash_sql(["customer_id"], salt="7")
    got = {
        row[0]: row[1]
        for row in spark.sql(f"SELECT customer_id, {expr} FROM mbt_parity_probe").collect()
    }
    assert got == {cid: reference_bucket(f"7|{cid}") for cid in range(50)}

    # reopening needs no Spark session; snapshot pin is enforced
    fresh = SparkDataAdapter({})
    reopened = fresh.from_locator(handle.locator())
    assert reopened.read("test").num_rows == handle.read("test").num_rows


# -- batch scoring (contract 1.1, R2-17) ----------------------------------------------

SCORE_WINDOW = {"score": ("2026-06-03T00:00:00Z", "2026-07-01T00:00:00Z")}
POP_UID = "source.p.lake.population"
SFEAT_UID = "source.p.lake.sfeatures"


def _scoring_sources(root: Path, n: int = 200) -> dict[str, Any]:
    """A population spine (who to score) + a feature table, no label anywhere."""
    dates = [ANCHOR - timedelta(days=(i * 179) % 180 + 1) for i in range(n)]
    pq.write_table(
        pa.table({"customer_id": list(range(n)), "snapshot_date": dates}),
        _mk(root / "population") / "part-000.parquet",
    )
    pq.write_table(
        pa.table(
            {
                "customer_id": list(range(n)),
                "snapshot_date": dates,
                "monthly_usage": [((i * 37) % 100) / 100.0 for i in range(n)],
            }
        ),
        _mk(root / "sfeatures") / "part-000.parquet",
    )
    return {
        POP_UID: FakeSourceTable("population", str(root / "population" / "*.parquet")),
        SFEAT_UID: FakeSourceTable("sfeatures", str(root / "sfeatures" / "*.parquet")),
    }


def _scoring_ctx(
    sources: dict[str, Any], out: Path, sample_fraction: float = 1.0
) -> FakeBuildContext:
    node = ManifestNode(
        unique_id="scoring.p.churn_scoring",
        resource_type="scoring",
        name="churn_scoring",
        path="scoring/churn_scoring.yml",
        config={},
        snapshot_id=None,  # scoring inputs are not snapshot-verified (R2-10)
    )
    return FakeBuildContext(
        node, sources, POP_UID, out, sample_fraction, resolved_windows=SCORE_WINDOW
    )


def _scoring_spec() -> ScoringInputSpec:
    return ScoringInputSpec.model_validate(
        {
            "inputs": {
                "spine": POP_UID,
                "features": [SFEAT_UID],
                "join_key": ["customer_id", "snapshot_date"],
            },
            "time_column": "snapshot_date",
            "window": "-28d:now",
        }
    )


def test_build_scoring_input_joins_windows_and_samples(tmp_path: Path) -> None:
    """R2-17: a Spark team can now materialize a batch for `mbt score` - spine +
    feature joins, the score window, and reproducible key sampling, unlabeled.
    Before this, Spark hard-failed at `mbt score` (no build_scoring_input)."""
    adapter = SparkDataAdapter({"master": "local[2]"})
    sources = _scoring_sources(_mk(tmp_path / "src"))
    spec = _scoring_spec()

    ctx = _scoring_ctx(sources, tmp_path / "score")
    handle = adapter.build_scoring_input(spec, ctx)
    assert handle.splits() == {"score"}
    assert handle.label_column == ""  # unlabeled by design (ADR-20)
    scored = handle.read("score")
    assert {"customer_id", "snapshot_date", "monthly_usage"} <= set(scored.column_names)
    assert 0 < scored.num_rows < 200  # the score window trims 180d of history
    logs = [str(m) for m in ctx.events.messages if "materialized" in str(m)]
    assert len(logs) == 1 and "rows to score" in logs[0]

    # reproducible sampling: same fraction -> same rows, a subset of the batch
    def ids(out: str, fraction: float) -> set[int]:
        h = adapter.build_scoring_input(spec, _scoring_ctx(sources, tmp_path / out, fraction))
        return set(h.read("score").column("customer_id").to_pylist())

    full = set(scored.column("customer_id").to_pylist())
    half_a, half_b = ids("a", 0.5), ids("b", 0.5)
    assert half_a == half_b and half_a < full and len(half_a) > 0

    # reopening needs no Spark session (scoring inputs are not pinned, R2-10)
    reopened = SparkDataAdapter({}).from_locator(handle.locator())
    assert reopened.read("score").num_rows == scored.num_rows


def test_build_scoring_input_empty_batch_is_a_warning_not_an_error(tmp_path: Path) -> None:
    """An empty nightly batch is legitimate (unlike a training split): 0 rows
    materializes with a warning, not a SparkAdapterError."""
    adapter = SparkDataAdapter({"master": "local[2]"})
    sources = _scoring_sources(_mk(tmp_path / "src"))
    ctx = _scoring_ctx(sources, tmp_path / "score")

    # A filter that matches nothing -> an empty score split, no exception.
    empty = _scoring_spec().model_copy(update={"filters": ["customer_id < 0"]})
    handle = adapter.build_scoring_input(empty, ctx)
    assert handle.read("score").num_rows == 0
    assert any("nothing to score" in str(m) for m in ctx.events.messages)


def test_open_predictions_roots_under_predictions_root(tmp_path: Path) -> None:
    """open_predictions stages runs under `predictions_root`/output.path using
    the shared local store (ADR-21 sanctioned reuse), completing the pair the
    coordinator probes before `mbt score`/`mbt monitor` will run on Spark."""
    adapter = SparkDataAdapter({"predictions_root": str(tmp_path / "warehouse")})
    store = adapter.open_predictions(ScoringOutputSpec(path="predictions/churn_scores"))

    from mbt_adapter_base.predictions import LocalPredictionStore, PredictionRunInfo

    assert isinstance(store, LocalPredictionStore)
    info = store.write_run(
        pa.table({"customer_id": [1, 2], "prediction": [0.9, 0.1]}),
        PredictionRunInfo(
            run_key="2026-07-01",
            uri="",  # write_run fills the persisted uri + row_count
            scored_at="2026-07-01T00:00:00Z",
            run_id="run-1",
            model_name="churn_model",
            model_version="1",
            row_count=0,
        ),
    )
    assert info.run_key == "2026-07-01" and info.row_count == 2
    assert store.read("2026-07-01").num_rows == 2
    # landed under predictions_root/output.path, not the cwd
    assert (tmp_path / "warehouse" / "predictions" / "churn_scores").exists()


def test_spark_now_advertises_batch_scoring_capability() -> None:
    """The coordinator gates `mbt score` on both methods being present
    (require_scoring_capability); Spark now passes that probe (R2-17)."""
    adapter = SparkDataAdapter({})
    assert hasattr(adapter, "build_scoring_input")
    assert hasattr(adapter, "open_predictions")


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
    # F21: label-join coverage recorded (every spine month here has labels)
    assert handle.label_join_coverage == {"spine_rows": 240, "matched_rows": 240}
    # F2/F21: the pre-join source-check methods work on the Spark plane
    pop_table = sources[pop_uid]
    assert adapter.count_source_duplicates(pop_table, ["customer_id", "snapshot_date"]) == 0
    values = adapter.read_source_distinct(pop_table, "safe_id")
    assert values.column_names == ["value"]
    assert len(values) == 40


def test_snapshot_changes_when_source_files_change(source_root: Path) -> None:
    adapter = SparkDataAdapter({})
    table = _sources(source_root)[LABEL_UID]
    before = adapter.snapshot_id(table)
    _write_tables(source_root, n=410)
    assert adapter.snapshot_id(table) != before
    _write_tables(source_root, n=400)  # restore for other tests


def test_deep_snapshot_is_mtime_independent(source_root: Path) -> None:
    """--deep-snapshot hashes file CONTENT, so a fresh checkout (which rewrites
    mtimes without touching bytes) does not flag the source as modified - the
    scaffold CI passes --deep-snapshot on every compiling step for exactly this
    reason (ADR-11), and a Spark-backed project on that scaffold used to fail
    because the adapter rejected the flag outright. The shallow (mtime) scheme,
    by contrast, DOES move when only the mtime changes."""
    import os

    adapter = SparkDataAdapter({})
    table = _sources(source_root)[LABEL_UID]
    (label_file,) = (source_root / "labels").glob("*.parquet")

    shallow_before = adapter.snapshot_id(table)
    deep_before = adapter.snapshot_id(table, deep=True)

    stat = label_file.stat()
    os.utime(label_file, ns=(stat.st_atime_ns, stat.st_mtime_ns + 1_000_000_000))

    assert adapter.snapshot_id(table) != shallow_before  # mtime scheme moved
    assert adapter.snapshot_id(table, deep=True) == deep_before  # content did not

    label_file.write_bytes(label_file.read_bytes() + b"\x00")  # real content change
    assert adapter.snapshot_id(table, deep=True) != deep_before
    _write_tables(source_root, n=400)  # restore pristine bytes for other tests


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


# -- training plane: string features + type pre-check (R2-18) --------------------------


def _string_feature_dataset(n: int = 600) -> Any:
    """A dataset with a STRING feature (`plan`) carrying real signal, so the
    adapter must index it rather than let VectorAssembler reject the column."""
    from random import Random

    from mbt_adapter_base.datasets import InMemoryDatasetHandle

    rng = Random(7)
    plans = ["basic", "pro", "enterprise"]
    bonus = {"basic": -0.5, "pro": 0.1, "enterprise": 0.9}
    cols: dict[str, list[Any]] = {"f_num": [], "plan": [], "label": []}
    for _ in range(n):
        plan = plans[rng.randrange(3)]
        num = rng.gauss(0, 1)
        cols["f_num"].append(num)
        cols["plan"].append(plan)
        cols["label"].append(1 if num + bonus[plan] + rng.gauss(0, 0.3) > 0 else 0)
    table = pa.table(cols)
    return InMemoryDatasetHandle(
        {"train": table.slice(0, int(n * 0.8)), "test": table.slice(int(n * 0.8))},
        snapshot_id="sha256:spark-string-feature",
        label_column="label",
    )


def _spark_spec(
    task: TaskType = TaskType.BINARY_CLASSIFICATION,
    metrics: tuple[str, ...] = ("roc_auc",),
    **overrides: Any,
) -> Any:
    from mbt_adapter_base import EvaluationProtocol, EvaluationSpec, ModelSpec

    return ModelSpec(
        name="m",
        task=task,
        adapter="sparkml",
        owner="t@example.com",
        dataset="ref('d')",
        target="label",
        hyperparameters={"max_iter": 8, "max_depth": 3},
        evaluation=EvaluationSpec(protocol=EvaluationProtocol(), metrics=list(metrics)),
        seed=1,
        **overrides,
    )


def _spark_ctx() -> Any:
    return RunContext(
        run_id="t",
        unique_id="model.t.m",
        seed=1,
        target_name="dev",
        project_dir=".",
        vars={},
        events=_CapturingSink(),
    )


# -- post-hoc calibration (R2-8) ------------------------------------------------------


def _spark_calibration_dataset(n: int = 300) -> Any:
    from random import Random

    from mbt_adapter_base.datasets import InMemoryDatasetHandle

    rng = Random(11)

    def tbl(m: int) -> pa.Table:
        xs = [rng.gauss(0, 1) for _ in range(m)]
        ys = [1 if x + rng.gauss(0, 0.4) > 0.3 else 0 for x in xs]
        return pa.table({"x": xs, "label": ys})

    return InMemoryDatasetHandle(
        {"train": tbl(n), "validation": tbl(n // 2), "test": tbl(n // 2)},
        snapshot_id="sha256:spark-calibration",
        label_column="label",
    )


def test_spark_calibration_applies_in_scores_and_survives_save_load(tmp_path: Path) -> None:
    """R2-8: a Spark model can post-hoc calibrate. The calibrator is applied at
    the _scores chokepoint and rides through save/load in the bundle sidecar."""
    import numpy as np

    from mbt_adapter_base.compliance.suite import TempArtifactStore

    adapter = SparkMLTrainingAdapter({})
    data = _spark_calibration_dataset()
    model = adapter.train(_spark_spec(calibration="isotonic"), data, _spark_ctx())
    assert model.calibrator is not None and model.calibrator.method == "isotonic"

    calibrated = adapter._scores(model, data, "test")
    model.calibrator, saved = None, model.calibrator
    raw = adapter._scores(model, data, "test")
    model.calibrator = saved
    assert not np.allclose(calibrated, raw)  # calibration changed the scores...
    np.testing.assert_allclose(calibrated, saved.transform(raw))  # ...via the calibrator
    assert np.all((calibrated >= 0.0) & (calibrated <= 1.0))  # valid probabilities

    store = TempArtifactStore(tmp_path)
    loaded = adapter.load(adapter.export(model, "native", store), store)
    assert loaded.calibrator is not None and loaded.calibrator.method == "isotonic"
    np.testing.assert_allclose(adapter._scores(loaded, data, "test"), calibrated)


def test_spark_calibration_requires_a_holdout_split() -> None:
    # neither a carved 'calibration' slice nor a 'validation' fallback present
    from mbt_adapter_base.datasets import InMemoryDatasetHandle

    adapter = SparkMLTrainingAdapter({})
    tbl = pa.table({"x": [0.1, 0.9, 0.2, 0.8, 0.3, 0.7], "label": [0, 1, 0, 1, 0, 1]})
    data = InMemoryDatasetHandle(
        {"train": tbl, "test": tbl}, snapshot_id="sha256:spark-noval", label_column="label"
    )
    with pytest.raises(SparkAdapterError, match="validation"):
        adapter.train(_spark_spec(calibration="isotonic"), data, _spark_ctx())


def test_string_features_train_via_string_indexer() -> None:
    """A string feature that trains natively on the arrow adapters must also
    train on Spark (R2-18): the adapter indexes it instead of letting the
    VectorAssembler raise a raw JVM IllegalArgumentException, and
    feature_importance stays keyed by the ORIGINAL (un-indexed) feature names."""
    from mbt_adapter_base import MetricSpec

    adapter = SparkMLTrainingAdapter({})
    data = _string_feature_dataset()
    model = adapter.train(_spark_spec(), data, _spark_ctx())

    importance = adapter.feature_importance(model)
    assert set(importance) == {"f_num", "plan"}  # aligned to original feature names
    assert importance["plan"] > 0  # the string feature was actually used by the GBT
    # the fitted StringIndexer is part of the saved pipeline, so scoring applies
    # the same mapping - the learnable plan+num signal clears a real bar
    result = adapter.evaluate(model, data, "test", [MetricSpec(name="roc_auc")])
    assert result.metrics["roc_auc"] > 0.7


def test_walk_forward_backtest_runs_on_spark() -> None:
    """R2-7: the walk-forward backtest works on a path adapter (spark), not just
    the arrow adapters - the per-fold TransformedDatasetHandle stages to parquet
    like any split, so SparkML refits and evaluates each fold."""
    from types import SimpleNamespace

    from mbt.execute.job import _walk_forward_backtest
    from mbt_adapter_base import MetricSpec
    from mbt_adapter_base.datasets import InMemoryDatasetHandle

    n = 60
    anchor = datetime(2026, 1, 1)
    # rows in DESCENDING time order (x jumps around so each time slice has both
    # classes), so the backtest must sort by ts before cutting folds
    signal = [((i * 37) % 100) / 100.0 for i in range(n)]
    table = pa.table(
        {
            "ts": [anchor + timedelta(days=n - 1 - i) for i in range(n)],
            "x": [signal[n - 1 - i] for i in range(n)],
            "label": [1 if signal[n - 1 - i] > 0.5 else 0 for i in range(n)],
        }
    )
    base = InMemoryDatasetHandle({"train": table}, label_column="label", time_column="ts")
    runtime = SimpleNamespace(
        base_handle=base,
        spec=_spark_spec(),
        adapter=SparkMLTrainingAdapter({}),
        ctx=_spark_ctx(),
        base_profile=base.profile(),
        hooks=None,
        builtin_specs=[MetricSpec(name="roc_auc")],
        # the fold logic reads the job's resolved windows (embargo, F6)
        job=SimpleNamespace(dataset_windows={}),
    )
    means, stds = _walk_forward_backtest(runtime, runtime.spec, 3)
    assert "roc_auc" in means and 0.0 <= means["roc_auc"] <= 1.0
    assert stds["roc_auc"] >= 0.0  # a std is reported alongside the mean


def test_unsupported_feature_type_is_actionable() -> None:
    """A non-numeric/-boolean/-string feature (here a timestamp) fails with mbt's
    actionable error, not a raw JVM IllegalArgumentException."""
    from datetime import datetime

    from mbt_adapter_base.datasets import InMemoryDatasetHandle

    table = pa.table(
        {
            "f_num": [0.1, 0.2, 0.3, 0.4],
            "when": [datetime(2026, 1, i + 1) for i in range(4)],  # timestamp feature
            "label": [0, 1, 0, 1],
        }
    )
    data = InMemoryDatasetHandle(
        {"train": table, "test": table}, snapshot_id="sha256:x", label_column="label"
    )
    with pytest.raises(SparkAdapterError, match="cannot train on feature column type"):
        SparkMLTrainingAdapter({}).train(_spark_spec(), data, _spark_ctx())


def _regression_dataset(n: int = 600) -> Any:
    """A continuous target driven by a numeric feature plus a string plan tier -
    exercises Spark regression (GBTRegressor) AND a string feature together."""
    from random import Random

    from mbt_adapter_base.datasets import InMemoryDatasetHandle

    rng = Random(11)
    plans = ["basic", "pro", "enterprise"]
    bonus = {"basic": 0.0, "pro": 8.0, "enterprise": 20.0}
    cols: dict[str, list[Any]] = {"f_num": [], "plan": [], "label": []}
    for _ in range(n):
        plan = plans[rng.randrange(3)]
        num = rng.gauss(0, 1)
        cols["f_num"].append(num)
        cols["plan"].append(plan)
        cols["label"].append(round(10.0 + 5.0 * num + bonus[plan] + rng.gauss(0, 1.5), 3))
    table = pa.table(cols)
    return InMemoryDatasetHandle(
        {"train": table.slice(0, int(n * 0.8)), "test": table.slice(int(n * 0.8))},
        snapshot_id="sha256:spark-regression",
        label_column="label",
    )


def test_spark_regression_with_string_feature() -> None:
    """Spark now trains a GBTRegressor for `task: regression` (R2-18), emitting
    target-scale predictions (not probabilities), and a string feature indexes
    into the same pipeline - so the whole regression vertical reaches Spark."""
    from mbt_adapter_base import MetricSpec

    adapter = SparkMLTrainingAdapter({})
    data = _regression_dataset()
    spec = _spark_spec(task=TaskType.REGRESSION, metrics=("rmse", "r2"))
    model = adapter.train(spec, data, _spark_ctx())

    preds = adapter.predict(model, data, "test")
    assert "prediction" in preds.column_names
    # target-scale (~10-30), well outside the [0, 1] probability range a
    # classifier would emit - proves the regressor path, not a misrouted classifier
    assert max(preds.column("prediction").to_pylist()) > 5.0
    result = adapter.evaluate(model, data, "test", [MetricSpec(name="rmse"), MetricSpec(name="r2")])
    assert result.metrics["r2"] > 0.5 and result.metrics["rmse"] >= 0.0
    importance = adapter.feature_importance(model)  # GBTRegressor also attributes
    assert set(importance) == {"f_num", "plan"} and importance["plan"] > 0


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
