"""``mbt monitor``: maturity, label joins, gates, ledger idempotency (ADR-21)."""

import json
from datetime import timedelta
from pathlib import Path

import pyarrow as pa
import pyarrow.parquet as pq
import pytest
from core_helpers import TEST_ANCHOR, write
from test_execution import invoke
from test_scoring_execution import SCORING, _build_and_promote, _prediction_runs, _write_batch

from mbt.adapters.registry import AdapterRegistry
from mbt.execute.monitor import run_monitor
from mbt.execute.orchestrator import InvocationOptions

GT_SCORING_YML = """
scoring:
  - name: churn_scoring
    owner: lifecycle-eng@example.com
    model: ref('churn_model')
    tags: [daily]
    input:
      source: source('lakehouse', 'scoring_batch')
      time_column: snapshot_date
      window: "-7d:now"
    ground_truth:
      label:
        source: source('lakehouse', 'churn_outcomes')
        column: churned
      join_key: user_id
      maturity: "14d"
      metrics: [pr_auc, roc_auc]
      gates:
        - metric: pr_auc
          threshold: 0.15
    output:
      path: predictions/churn_scores
      columns: [user_id]
"""


def _write_outcomes(project_dir: Path, *, coverage: int = 120) -> None:
    table = pa.table(
        {
            "user_id": list(range(coverage)),
            "churned": [1 if i % 4 == 0 else 0 for i in range(coverage)],
        }
    )
    out = project_dir / "data" / "churn_outcomes"
    out.mkdir(parents=True, exist_ok=True)
    pq.write_table(table, out / "part-000.parquet")


@pytest.fixture()
def monitored_project(demo_project: Path) -> Path:
    write(
        demo_project / "sources.yml",
        """
        sources:
          - name: lakehouse
            tables:
              - name: subscribers
                path: data/subscribers/*.parquet
              - name: scoring_batch
                path: data/scoring_batch/*.parquet
              - name: churn_outcomes
                path: data/churn_outcomes/*.parquet
        """,
    )
    write(demo_project / "scoring/churn_scoring.yml", GT_SCORING_YML)
    _write_batch(demo_project)
    _write_outcomes(demo_project)
    return demo_project


def monitor(project_dir: Path, registry: AdapterRegistry, *, days_later: int = 20, **kwargs):
    opts = InvocationOptions(
        command="monitor",
        project_dir=project_dir,
        anchor=TEST_ANCHOR + timedelta(days=days_later),
        **kwargs,
    )
    return run_monitor(opts, registry=registry)


def _score(project_dir: Path, registry: AdapterRegistry) -> None:
    assert invoke(project_dir, registry, "score").exit_code() == 0


def test_matured_run_is_evaluated_once(
    monitored_project: Path, fake_registry: AdapterRegistry
) -> None:
    _build_and_promote(monitored_project, fake_registry)
    _score(monitored_project, fake_registry)

    results = monitor(monitored_project, fake_registry)
    assert results.exit_code() == 0
    node = next(r for r in results.results if r.unique_id == SCORING)
    assert node.status == "success"
    assert node.message and "evaluated 1 of 1" in node.message
    assert 0.0 <= node.metrics["roc_auc"] <= 1.0
    assert node.metrics["pr_auc"] >= 0.15
    gate = node.monitors[0]
    assert gate.monitor == "ground_truth" and gate.passed and gate.measure == "pr_auc"

    run_dir = _prediction_runs(monitored_project)[0]
    marker = json.loads((run_dir / "ground_truth.marker.json").read_text())
    assert marker["gates_passed"] is True
    assert marker["matched_rows"] == 120
    assert marker["coverage"] == 1.0
    assert marker["metrics"]["pr_auc"] == node.metrics["pr_auc"]

    # run_results written with command=monitor
    stored = json.loads((monitored_project / "target/run_results.json").read_text())
    assert stored["metadata"]["command"] == "monitor"

    # the ledger makes a second monitor run a no-op
    again = monitor(monitored_project, fake_registry)
    assert again.exit_code() == 0
    node = next(r for r in again.results if r.unique_id == SCORING)
    assert node.message == "0 matured prediction runs to evaluate"


def test_immature_runs_are_not_evaluated(
    monitored_project: Path, fake_registry: AdapterRegistry
) -> None:
    _build_and_promote(monitored_project, fake_registry)
    _score(monitored_project, fake_registry)
    results = monitor(monitored_project, fake_registry, days_later=5)  # maturity is 14d
    node = next(r for r in results.results if r.unique_id == SCORING)
    assert node.status == "success"
    assert node.message == "0 matured prediction runs to evaluate"
    assert not list(_prediction_runs(monitored_project)[0].glob("*.marker.json"))


def test_gate_breach_exits_2(monitored_project: Path, fake_registry: AdapterRegistry) -> None:
    write(
        monitored_project / "scoring/churn_scoring.yml",
        GT_SCORING_YML.replace("threshold: 0.15", "threshold: 0.9"),
    )
    _build_and_promote(monitored_project, fake_registry)
    _score(monitored_project, fake_registry)
    results = monitor(monitored_project, fake_registry)
    assert results.exit_code() == 2
    node = next(r for r in results.results if r.unique_id == SCORING)
    assert node.status == "monitor_failed"
    assert node.message and "gate breach" in node.message
    # evaluated is evaluated: the marker is written even on breach
    run_dir = _prediction_runs(monitored_project)[0]
    marker = json.loads((run_dir / "ground_truth.marker.json").read_text())
    assert marker["gates_passed"] is False


def test_partial_label_coverage_evaluates_matched_rows(
    monitored_project: Path, fake_registry: AdapterRegistry
) -> None:
    _write_outcomes(monitored_project, coverage=80)
    _build_and_promote(monitored_project, fake_registry)
    _score(monitored_project, fake_registry)
    results = monitor(monitored_project, fake_registry)
    assert results.exit_code() == 0
    run_dir = _prediction_runs(monitored_project)[0]
    marker = json.loads((run_dir / "ground_truth.marker.json").read_text())
    assert marker["matched_rows"] == 80
    assert marker["coverage"] == round(80 / 120, 4)


def test_no_labels_yet_retries_later(
    monitored_project: Path, fake_registry: AdapterRegistry
) -> None:
    # outcomes exist (source must resolve) but cover none of the scored rows
    table = pa.table({"user_id": [9999], "churned": [1]})
    pq.write_table(table, monitored_project / "data" / "churn_outcomes" / "part-000.parquet")
    _build_and_promote(monitored_project, fake_registry)
    _score(monitored_project, fake_registry)
    results = monitor(monitored_project, fake_registry)
    assert results.exit_code() == 0
    node = next(r for r in results.results if r.unique_id == SCORING)
    assert node.message and "evaluated 0 of 1" in node.message
    # no marker: the run stays eligible once labels arrive
    assert not list(_prediction_runs(monitored_project)[0].glob("*.marker.json"))
    _write_outcomes(monitored_project)  # labels arrive
    results = monitor(monitored_project, fake_registry)
    node = next(r for r in results.results if r.unique_id == SCORING)
    assert node.message and "evaluated 1 of 1" in node.message


def test_scoring_without_ground_truth_is_skipped(
    monitored_project: Path, fake_registry: AdapterRegistry
) -> None:
    yml = GT_SCORING_YML.replace("churn_scoring", "plain_scoring").replace(
        "path: predictions/churn_scores", "path: predictions/plain_scores"
    )
    start = yml.index("    ground_truth:")
    end = yml.index("    output:")
    write(
        monitored_project / "scoring/plain_scoring.yml",
        yml[:start] + yml[end:].replace("columns: [user_id]", "columns: [user_id]"),
    )
    _build_and_promote(monitored_project, fake_registry)
    results = monitor(monitored_project, fake_registry, select=["plain_scoring"])
    node = next(r for r in results.results if r.unique_id == "scoring.demo.plain_scoring")
    assert node.status == "skipped"
    assert node.message == "no ground_truth block declared"


def test_monitor_metrics_reach_tracking(
    monitored_project: Path, fake_registry: AdapterRegistry
) -> None:
    _build_and_promote(monitored_project, fake_registry)
    _score(monitored_project, fake_registry)
    monitor(monitored_project, fake_registry)
    tracking_dir = monitored_project / "target/fake_tracking"
    payloads = [json.loads(p.read_text()) for p in tracking_dir.glob("*.json")]
    monitor_runs = [p for p in payloads if p.get("tags", {}).get("mbt.monitor") == "ground_truth"]
    assert len(monitor_runs) == 1
    assert monitor_runs[0]["metrics"]["ground_truth.coverage"] == 1.0
    assert "pr_auc" in monitor_runs[0]["metrics"]
