"""Local adapter scoring-input builds + the prediction store (ADR-20/21)."""

from datetime import datetime, timedelta
from pathlib import Path
from typing import Any

import pyarrow as pa
import pyarrow.parquet as pq
import pytest

from mbt.adapters.local.data import LocalDataAdapter
from mbt.contracts import ManifestNode, ScoringInputSpec, SourceTable
from mbt.exceptions import AdapterError
from mbt.execute.runners import BuildContext
from mbt_adapter_base.compliance import PredictionStoreCompliance
from mbt_adapter_base.predictions import LocalPredictionStore, PredictionStoreError

ANCHOR = datetime(2026, 7, 1)


class _CapturingSink:
    def __init__(self) -> None:
        self.events: list[Any] = []

    def emit(self, event: object) -> None:
        self.events.append(event)


def _write_batch(root: Path, n: int = 30) -> None:
    table = pa.table(
        {
            "user_id": list(range(n)),
            "snapshot_date": [ANCHOR - timedelta(days=1 + i % 10) for i in range(n)],
            "is_active": [i % 5 != 0 for i in range(n)],
            "tenure_days": [30 + i for i in range(n)],
        }
    )
    out = root / "data" / "batch"
    out.mkdir(parents=True, exist_ok=True)
    pq.write_table(table, out / "part-000.parquet")


def _write_features(root: Path, n: int = 30) -> None:
    table = pa.table(
        {
            "user_id": list(range(n)),
            "monthly_usage": [round(i * 3.3, 2) for i in range(n)],
        }
    )
    out = root / "data" / "usage"
    out.mkdir(parents=True, exist_ok=True)
    pq.write_table(table, out / "part-000.parquet")


def _ctx(
    adapter: LocalDataAdapter,
    tables: dict[str, SourceTable],
    output_dir: Path,
    *,
    windows: dict[str, tuple[str, str]] | None = None,
    snapshot_override: str | None = None,
) -> tuple[BuildContext, _CapturingSink]:
    from mbt_adapter_base.materialization import combine_snapshots

    snapshot = snapshot_override or combine_snapshots(
        {uid: adapter.snapshot_id(table) for uid, table in tables.items()}
    )
    node = ManifestNode(
        unique_id="scoring.demo.batch_scoring",
        resource_type="scoring",
        name="batch_scoring",
        path="scoring/batch_scoring.yml",
        config={},
        snapshot_id=snapshot,
    )
    events = _CapturingSink()
    ctx = BuildContext(
        node=node,
        source=next(iter(tables.values())),
        source_tables=tables,
        resolved_windows=windows or {},
        sample_fraction=1.0,
        deep_snapshot=False,
        output_dir=output_dir,
        events=events,
    )
    return ctx, events


def test_single_source_build_with_window_and_filters(tmp_path: Path) -> None:
    _write_batch(tmp_path)
    adapter = LocalDataAdapter({"root": str(tmp_path)})
    tables = {"source.demo.lakehouse.batch": SourceTable(name="batch", path="data/batch/*.parquet")}
    spec = ScoringInputSpec.model_validate(
        {
            "source": "source.demo.lakehouse.batch",
            "filters": ["is_active = true"],
            "time_column": "snapshot_date",
            "window": "-5d:now",
        }
    )
    ctx, events = _ctx(
        adapter,
        tables,
        tmp_path / "target/scoring_inputs/batch_scoring/k1",
        windows={"score": ("2026-06-26T00:00:00Z", "2026-07-01T00:00:00Z")},
    )
    handle = adapter.build_scoring_input(spec, ctx)
    assert handle.splits() == {"score"}
    # the positive path reports its row count on the bus (not just 0-row warns)
    assert any("materialized" in getattr(e, "message", "") for e in events.events)
    table = handle.read("score")
    assert table.num_rows > 0
    dates = table.column("snapshot_date").to_pylist()
    assert all(d >= datetime(2026, 6, 26) for d in dates)
    actives = set(table.column("is_active").to_pylist())
    assert actives == {True}
    assert handle.label_column == ""
    profile = handle.profile()
    assert profile.label_balance is None
    assert profile.n_rows == {"score": table.num_rows}


def test_multi_table_spine_join(tmp_path: Path) -> None:
    _write_batch(tmp_path)
    _write_features(tmp_path)
    adapter = LocalDataAdapter({"root": str(tmp_path)})
    tables = {
        "source.demo.lakehouse.batch": SourceTable(name="batch", path="data/batch/*.parquet"),
        "source.demo.lakehouse.usage": SourceTable(name="usage", path="data/usage/*.parquet"),
    }
    spec = ScoringInputSpec.model_validate(
        {
            "inputs": {
                "spine": "source.demo.lakehouse.batch",
                "features": ["source.demo.lakehouse.usage"],
                "join_key": "user_id",
            }
        }
    )
    ctx, _ = _ctx(adapter, tables, tmp_path / "target/scoring_inputs/batch_scoring/k2")
    handle = adapter.build_scoring_input(spec, ctx)
    table = handle.read("score")
    assert table.num_rows == 30
    assert "monthly_usage" in table.column_names


def test_per_table_using_columns_on_scoring_features(tmp_path: Path) -> None:
    """Feature entries with their own join columns (ADR-22) at scoring time."""
    _write_batch(tmp_path)
    _write_features(tmp_path)
    adapter = LocalDataAdapter({"root": str(tmp_path)})
    tables = {
        "source.demo.lakehouse.batch": SourceTable(name="batch", path="data/batch/*.parquet"),
        "source.demo.lakehouse.usage": SourceTable(name="usage", path="data/usage/*.parquet"),
    }
    spec = ScoringInputSpec.model_validate(
        {
            "inputs": {
                "spine": "source.demo.lakehouse.batch",
                "features": [{"source": "source.demo.lakehouse.usage", "using": ["user_id"]}],
            }
        }
    )
    assert spec.inputs is not None
    assert spec.inputs.feature_sources == ["source.demo.lakehouse.usage"]
    ctx, _ = _ctx(adapter, tables, tmp_path / "target/scoring_inputs/batch_scoring/k5")
    handle = adapter.build_scoring_input(spec, ctx)
    table = handle.read("score")
    assert table.num_rows == 30
    assert "monthly_usage" in table.column_names


def test_zero_rows_warns_instead_of_failing(tmp_path: Path) -> None:
    _write_batch(tmp_path)
    adapter = LocalDataAdapter({"root": str(tmp_path)})
    tables = {"source.demo.lakehouse.batch": SourceTable(name="batch", path="data/batch/*.parquet")}
    spec = ScoringInputSpec.model_validate(
        {"source": "source.demo.lakehouse.batch", "filters": ["tenure_days > 100000"]}
    )
    ctx, events = _ctx(adapter, tables, tmp_path / "target/scoring_inputs/batch_scoring/k3")
    handle = adapter.build_scoring_input(spec, ctx)
    assert handle.read("score").num_rows == 0
    assert any("0 rows" in getattr(e, "message", "") for e in events.events)


def test_snapshot_drift_under_pinned_manifest_fails(tmp_path: Path) -> None:
    _write_batch(tmp_path)
    adapter = LocalDataAdapter({"root": str(tmp_path)})
    tables = {"source.demo.lakehouse.batch": SourceTable(name="batch", path="data/batch/*.parquet")}
    spec = ScoringInputSpec.model_validate({"source": "source.demo.lakehouse.batch"})
    ctx, _ = _ctx(
        adapter,
        tables,
        tmp_path / "target/scoring_inputs/batch_scoring/k4",
        snapshot_override="sha256:pinned-elsewhere",
    )
    with pytest.raises(AdapterError, match="source data changed under the pinned manifest"):
        adapter.build_scoring_input(spec, ctx)


def test_open_predictions_roots_under_data_root(tmp_path: Path) -> None:
    from mbt.contracts import ScoringOutputSpec

    adapter = LocalDataAdapter({"root": str(tmp_path)})
    store = adapter.open_predictions(
        ScoringOutputSpec(path="predictions/churn", columns=["user_id"])
    )
    assert isinstance(store, LocalPredictionStore)
    assert store.root == tmp_path / "predictions/churn"


def test_marker_on_incomplete_run_fails(tmp_path: Path) -> None:
    store = LocalPredictionStore(tmp_path / "predictions")
    with pytest.raises(PredictionStoreError, match="no complete prediction run"):
        store.write_marker("missing", "ground_truth", {})
    with pytest.raises(PredictionStoreError, match="no complete prediction run"):
        store.read("missing")


def test_list_runs_ignores_incomplete_writes(tmp_path: Path) -> None:
    """A run whose _SUCCESS marker is absent (a crash after the info sidecar
    but before completion) is invisible to list_runs, so `mbt monitor` never
    tries to evaluate a half-written prediction run."""
    from mbt_adapter_base.interchange import PredictionRunInfo
    from mbt_adapter_base.predictions import INFO_FILE

    store = LocalPredictionStore(tmp_path / "predictions")
    info = PredictionRunInfo(
        run_key="complete",
        uri="",
        scored_at="2026-07-01T00:00:00Z",
        run_id="r1",
        model_name="m",
        model_version="1",
        row_count=3,
    )
    store.write_run(pa.table({"user_id": [1, 2, 3], "prediction": [0.1, 0.2, 0.3]}), info)

    # Simulate a crash after the info sidecar was written but before _SUCCESS.
    half = store.root / "half_written"
    half.mkdir(parents=True)
    half_info = info.model_copy(update={"run_key": "half_written"})
    (half / INFO_FILE).write_text(half_info.model_dump_json())

    assert [r.run_key for r in store.list_runs()] == ["complete"]


class TestLocalPredictionStoreCompliance(PredictionStoreCompliance):
    def make_store(self, root: Path) -> LocalPredictionStore:
        return LocalPredictionStore(root / "predictions")
