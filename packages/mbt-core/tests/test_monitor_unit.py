"""Unit tests for delayed ground-truth monitoring edge cases (mbt/execute/monitor.py)."""

import json
from datetime import timedelta
from pathlib import Path

import pyarrow as pa
import pyarrow.parquet as pq
import pytest
from core_helpers import TEST_ANCHOR, write
from exec_unit_helpers import SCORING_UID, make_options, recording_bus
from mbt_testing.adapters import FakeTrackingAdapter
from test_execution import invoke
from test_monitor_ground_truth import GT_SCORING_YML, _write_outcomes, monitor
from test_scoring_execution import _build_and_promote, _prediction_runs, _write_batch

from mbt.adapters.registry import AdapterRegistry
from mbt.execute.orchestrator import prepare

GT_SOURCES_YML = """
sources:
  - name: lakehouse
    tables:
      - name: subscribers
        path: data/subscribers/*.parquet
      - name: scoring_batch
        path: data/scoring_batch/*.parquet
      - name: churn_outcomes
        path: data/churn_outcomes/*.parquet
"""


@pytest.fixture()
def gt_project(demo_project: Path) -> Path:
    write(demo_project / "sources.yml", GT_SOURCES_YML)
    write(demo_project / "scoring/churn_scoring.yml", GT_SCORING_YML)
    _write_batch(demo_project)
    _write_outcomes(demo_project)
    return demo_project


def _node(results, uid=SCORING_UID):
    return next(r for r in results.results if r.unique_id == uid)


def test_zero_row_prediction_runs_are_never_matured(
    gt_project: Path, fake_registry: AdapterRegistry
) -> None:
    _build_and_promote(gt_project, fake_registry)
    # a stale batch: every row falls outside the -7d:now scoring window
    base = TEST_ANCHOR.replace(tzinfo=None) - timedelta(days=100)
    n = 20
    stale = pa.table(
        {
            "user_id": list(range(n)),
            "snapshot_date": [base] * n,
            "is_active": [True] * n,
            "tenure_days": [100] * n,
            "monthly_usage": [50.0] * n,
            "plan_type": ["basic"] * n,
        }
    )
    pq.write_table(stale, gt_project / "data" / "scoring_batch" / "part-000.parquet")
    assert invoke(gt_project, fake_registry, "score").exit_code() == 0

    results = monitor(gt_project, fake_registry)
    assert results.exit_code() == 0
    node = _node(results)
    assert node.status == "success"
    assert node.message == "0 matured prediction runs to evaluate"


def test_monitor_manifest_tampering_is_a_node_error(
    gt_project: Path, fake_registry: AdapterRegistry
) -> None:
    _build_and_promote(gt_project, fake_registry)
    assert invoke(gt_project, fake_registry, "score").exit_code() == 0
    # recompile at a matured anchor: --manifest execution reads the STORED
    # anchor, and the run must count as matured under it (maturity is 14d)
    prepare(
        make_options(gt_project, command="monitor", anchor=TEST_ANCHOR + timedelta(days=20)),
        registry=fake_registry,
    )
    manifest_file = gt_project / "target" / "manifest.json"
    original = manifest_file.read_text()

    # 1) unknown + non-builtin ground-truth metrics
    payload = json.loads(original)
    ground_truth = payload["nodes"][SCORING_UID]["config"]["ground_truth"]
    ground_truth["metrics"] = ["mystery_metric", "hooky"]
    ground_truth["gates"] = []
    payload["metrics"]["hooky"] = {"name": "hooky", "kind": "hook"}
    manifest_file.write_text(json.dumps(payload))
    results = monitor(gt_project, fake_registry, manifest_path=str(manifest_file))
    assert results.exit_code() == 1
    node = _node(results)
    assert node.status == "error"
    assert "unknown metric" in (node.message or "")
    assert "must be a builtin" in (node.message or "")

    # 2) the label source is gone from the manifest
    payload = json.loads(original)
    label = payload["nodes"][SCORING_UID]["config"]["ground_truth"]["label"]
    label["source"] = "source.demo.lakehouse.nowhere"
    manifest_file.write_text(json.dumps(payload))
    results = monitor(gt_project, fake_registry, manifest_path=str(manifest_file))
    assert results.exit_code() == 1
    node = _node(results)
    assert node.status == "error"
    assert "label source" in (node.message or "")
    assert "missing from the manifest" in (node.message or "")


def test_label_table_missing_columns_is_a_node_error(
    gt_project: Path, fake_registry: AdapterRegistry
) -> None:
    _build_and_promote(gt_project, fake_registry)
    assert invoke(gt_project, fake_registry, "score").exit_code() == 0
    labels_only_keys = pa.table({"user_id": list(range(120))})
    pq.write_table(labels_only_keys, gt_project / "data" / "churn_outcomes" / "part-000.parquet")
    results = monitor(gt_project, fake_registry)
    assert results.exit_code() == 1
    node = _node(results)
    assert node.status == "error"
    assert "lacks column(s): churned" in (node.message or "")


def test_single_class_labels_are_retried_later(
    gt_project: Path, fake_registry: AdapterRegistry
) -> None:
    _build_and_promote(gt_project, fake_registry)
    assert invoke(gt_project, fake_registry, "score").exit_code() == 0
    single_class = pa.table({"user_id": list(range(120)), "churned": [0] * 120})
    pq.write_table(single_class, gt_project / "data" / "churn_outcomes" / "part-000.parquet")
    results = monitor(gt_project, fake_registry)
    assert results.exit_code() == 0
    node = _node(results)
    assert node.status == "success"
    assert "evaluated 0 of 1" in (node.message or "")
    # no ledger marker: the run stays eligible once two-class labels arrive
    assert not list(_prediction_runs(gt_project)[0].glob("*.marker.json"))


def test_unparseable_scored_at_is_skipped_not_fatal(
    gt_project: Path, fake_registry: AdapterRegistry
) -> None:
    """A sidecar with a bad scored_at (e.g. from an external store) must not
    crash the whole node with a bare ValueError; it is skipped with a warning
    (R2-19)."""
    _build_and_promote(gt_project, fake_registry)
    assert invoke(gt_project, fake_registry, "score").exit_code() == 0

    sidecar = _prediction_runs(gt_project)[0] / "predictions.json"
    info = json.loads(sidecar.read_text())
    info["scored_at"] = "not-a-timestamp"
    sidecar.write_text(json.dumps(info))

    with recording_bus() as sink:
        results = monitor(gt_project, fake_registry)

    # the bad run is skipped, not fatal: the node stays green and warns
    assert results.exit_code() == 0
    node = _node(results)
    assert node.status == "success"
    assert node.message == "0 matured prediction runs to evaluate"
    assert any("unparseable scored_at" in m and "not-a-timestamp" in m for m in sink.messages())


def test_monitor_tracking_failure_warns_but_evaluates(
    gt_project: Path, fake_registry: AdapterRegistry, monkeypatch: pytest.MonkeyPatch
) -> None:
    _build_and_promote(gt_project, fake_registry)
    assert invoke(gt_project, fake_registry, "score").exit_code() == 0

    def _boom(self, node, meta):
        raise RuntimeError("tracking down")

    monkeypatch.setattr(FakeTrackingAdapter, "start_run", _boom)
    with recording_bus() as sink:
        results = monitor(gt_project, fake_registry)
    assert results.exit_code() == 0
    node = _node(results)
    assert node.status == "success"
    assert "evaluated 1 of 1" in (node.message or "")
    # evaluation still landed in the ledger despite the tracking failure
    assert list(_prediction_runs(gt_project)[0].glob("*.marker.json"))
    assert any("could not log monitor metrics" in m for m in sink.messages())


def test_overlapping_monitor_does_not_re_evaluate_a_recorded_run(
    gt_project: Path, fake_registry: AdapterRegistry, monkeypatch: pytest.MonkeyPatch
) -> None:
    """Two overlapping monitor crons must not double-evaluate: even if the
    second reads the ledger as unevaluated, the atomic marker write loses the
    race and the run is skipped, not re-recorded or double-alerted (R2-11)."""
    from mbt_adapter_base.predictions import LocalPredictionStore

    _build_and_promote(gt_project, fake_registry)
    assert invoke(gt_project, fake_registry, "score").exit_code() == 0
    monitor(gt_project, fake_registry)  # records the ground_truth marker

    # the second cron reads the marker as absent (stale) while it is on disk
    monkeypatch.setattr(LocalPredictionStore, "read_marker", lambda self, rk, name: None)
    with recording_bus() as sink:
        results = monitor(gt_project, fake_registry)
    node = _node(results)
    assert node.status == "success"
    assert "evaluated 0 of 1" in (node.message or "")  # re-claim lost -> skipped
    assert any("already evaluated by a concurrent monitor run" in m for m in sink.messages())
