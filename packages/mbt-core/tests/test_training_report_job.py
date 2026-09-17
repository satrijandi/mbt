"""The training job's report step (ADR-30), end to end through real jobs."""

import json
import math
from pathlib import Path
from typing import Any

import pyarrow as pa
import pyarrow.parquet as pq
import pytest
from core_helpers import TEST_ANCHOR, write
from exec_unit_helpers import (
    MODEL_UID,
    make_inline_runtime,
    make_training_job,
    minimal_model_spec,
    recording_bus,
)
from mbt_testing.adapters import FakeTrainingAdapter
from report_unit_helpers import (
    MODEL_FILE,
    REPORT_CONFIG,
    edit,
    model_evaluation,
    reporting_project,
    with_out_of_time,
)

from mbt.adapters.registry import AdapterRegistry
from mbt.contracts import (
    MetricSpec,
    PeriodCell,
    ReportSummary,
    RunHandle,
    StabilityCell,
)
from mbt.execute import training_report as report_step
from mbt.execute.job import _run_train, run_job
from mbt.execute.orchestrator import InvocationOptions, run_command
from mbt.reporting.builder import ReportData, ScoredSplit
from mbt.reporting.writer import ReportMeta


def _tracking_payload(project: Path) -> dict[str, Any]:
    payloads = [
        json.loads(p.read_text()) for p in (project / "target" / "fake_tracking").glob("*.json")
    ]
    assert len(payloads) == 1
    return payloads[0]


def test_a_build_puts_the_report_on_the_run_the_store_and_the_version(
    demo_project: Path, fake_registry: AdapterRegistry
) -> None:
    reporting_project(demo_project)
    results = run_command(
        InvocationOptions(command="build", project_dir=demo_project, anchor=TEST_ANCHOR),
        registry=fake_registry,
    )
    model = next(r for r in results.results if r.unique_id == MODEL_UID)
    assert model.status == "success", model.message

    after_gate = next(g for g in model.gates if g.period is not None)
    assert after_gate.metric == "oot_roc_auc" and after_gate.applicable
    assert after_gate.cell == "2026-06"
    assert model.monitors and all(m.passed for m in model.monitors)
    assert all(m.subject and m.subject.startswith("2026-06") for m in model.monitors)

    run = _tracking_payload(demo_project)
    params = run["params"]
    assert params["seed"] == "42" and params["max_depth"] == "4"  # history-compatible
    assert params["model.evaluation.protocol.split"] == "temporal"
    assert params["dataset.split.out_of_time"] == "-30d:now"
    assert params["dataset.windows.out_of_time.end"] == "2026-07-01T00:00:00Z"
    assert params["model.resolved.n_features"] == "5"
    metrics = run["metrics"]
    assert "oot.window.roc_auc" in metrics and "oot.month.2026-06.roc_auc" in metrics
    assert "stability.month.2026-06.score_psi" in metrics
    assert "train.roc_auc" in metrics
    assert run["tags"]["mbt.oot_check.passed"] == "true"
    directories = run["directories"]
    assert {"report", "config", "logs"} <= set(directories)
    assert "performance/by_period.csv" in directories["report"]
    assert "predictions/out_of_time.parquet" in directories["report"]

    run_dir = demo_project / "target" / "fake_tracking" / run["run_id"]
    log = (run_dir / "logs" / "train.log").read_text()
    assert "## dataset.demo.churn_training" in log and "## model.demo.churn_model" in log
    assert "feature(s):" in log and "fit took" in log  # debug detail reaches the file
    predictions = pq.read_table(run_dir / "report" / "predictions" / "test.parquet")
    assert predictions.num_rows == 20  # max_rows caps it
    assert {"user_id", "snapshot_date", "split", "label", "p0", "p1"} <= set(
        predictions.column_names
    )
    page = (run_dir / "report" / "report.html").read_text()
    assert "After the test window" in page and "Custom edges" in page

    versions = json.loads(
        (demo_project / "target" / "fake_registry" / "churn_model.json").read_text()
    )
    tags = versions[-1]["tags"]
    assert tags["mbt.oot_check.passed"] == "true"
    assert tags["mbt.oot_check.source"] == "build"
    assert tags["mbt.report_uri"].endswith("/report/report.html")
    assert Path(tags["mbt.report_uri"].removeprefix("file://")).is_file()

    config_path = Path(tags["mbt.inference_config_uri"].removeprefix("file://"))
    config = json.loads(config_path.read_text())
    assert config["dataset"]["windows"]["out_of_time"][1] == "2026-07-01T00:00:00Z"
    assert config["dataset"]["sample_key"] == ["user_id"]
    assert config["resolved"]["hyperparameters"]["max_depth"] == 4
    assert config["report"]["out_of_time_rows"] > 0


def _engine_project(project: Path, max_reports: int) -> None:
    """Two after-test months, and the fake report engine on top."""
    with_out_of_time(project, test="-75d:-45d", after="-45d:now")
    model_evaluation(project, REPORT_CONFIG)
    edit(
        project / MODEL_FILE,
        "        binning: all\n",
        "        binning: all\n"
        f"        stability: {{engine: fake, feature_top_n: 2, max_html_reports: {max_reports}}}\n",
    )


def _report_dir(result: Any) -> Path:
    return Path(result.report.report_uri.removeprefix("file://")).parent


def test_a_report_engine_adds_its_drift_reports_beside_the_tables(
    demo_project: Path, fake_registry: AdapterRegistry
) -> None:
    _engine_project(demo_project, max_reports=2)
    _, job = make_training_job(demo_project, fake_registry)
    with recording_bus() as sink:
        result = run_job(job)
    assert result.status == "success", result.error
    documents = set(result.report.documents)
    assert {
        "stability/fake/window.html",
        "stability/fake/2026-06.html",
        "stability/fake/drift_by_period.csv",
        "stability/fake/drift_by_column.csv",
    } <= documents
    assert "stability/fake/2026-05.html" not in documents  # the newest month wins the slot
    assert (
        "report: fake drift reports cover the newest 1 of 2 after-test months "
        "(max_html_reports: 2)" in sink.messages()
    )
    root = _report_dir(result)
    columns = (root / "stability/fake/drift_by_column.csv").read_text().splitlines()
    assert columns[0] == "cell,column,method,score,threshold,drifted"
    # the two most important features and the score, never the label
    compared = {line.split(",")[1] for line in columns[1:] if line.startswith("window,")}
    assert len(compared) == 3 and "prediction" in compared and "churned" not in compared
    page = (root / "report.html").read_text()
    assert "Fake drift report" in page and "stability/fake/window.html" in page
    assert (
        "<title>churn_model: window against test</title>"
        in (root / "stability/fake/window.html").read_text()
    )


def test_a_report_engine_that_fails_is_a_warning_not_a_failed_build(
    demo_project: Path, fake_registry: AdapterRegistry, monkeypatch: pytest.MonkeyPatch
) -> None:
    from mbt_testing.adapters import FakeReportingEngine

    from mbt.adapters.registry import AdapterRegistry as Registry

    _engine_project(demo_project, max_reports=1)

    def boom(self: Any, *args: Any, **kwargs: Any) -> Any:
        raise RuntimeError("cannot render")

    monkeypatch.setattr(FakeReportingEngine, "drift_report", boom)
    _, job = make_training_job(demo_project, fake_registry)
    result = run_job(job)
    assert result.status == "success", result.error
    assert "fake drift report for window failed: cannot render" in result.report.warnings
    assert "fake drift reports cover the newest 0 of 2 after-test months (max_html_reports: 1)" in (
        result.report.warnings
    )
    assert not any(d.startswith("stability/fake/") for d in result.report.documents)

    real_component = Registry.component

    def no_reporting(self: Any, kind: str, name: str, config: dict[str, Any]) -> Any:
        if kind == "reporting":
            raise RuntimeError("engine unavailable")
        return real_component(self, kind, name, config)

    monkeypatch.setattr(Registry, "component", no_reporting)
    _, job = make_training_job(demo_project, fake_registry)
    result = run_job(job)
    assert result.status == "success", result.error
    assert "fake drift report skipped: engine unavailable" in result.report.warnings


def _row_dropping_hooks(project: Path) -> None:
    write(
        project / "models" / "churn_model.py",
        """
        def transform_features(table, ctx):
            return table.slice(1)
        """,
    )


def test_hooks_that_drop_rows_fail_a_report_that_needs_alignment(
    demo_project: Path, fake_registry: AdapterRegistry
) -> None:
    with_out_of_time(demo_project)
    _row_dropping_hooks(demo_project)
    _, job = make_training_job(demo_project, fake_registry)
    result = run_job(job)
    assert result.status == "error"
    assert "hooks changed the train split's row count" in (result.error or "")


def test_hooks_see_the_after_test_split_by_name(
    demo_project: Path, fake_registry: AdapterRegistry
) -> None:
    reporting_project(demo_project)
    seen = demo_project / "splits_seen.txt"
    write(
        demo_project / "models" / "churn_model.py",
        f"""
        def transform_features(table, ctx):
            with open({str(seen)!r}, "a") as handle:
                handle.write(ctx.split + "\\n")
            return table
        """,
    )
    _, job = make_training_job(demo_project, fake_registry)
    result = run_job(job)
    assert result.status == "success", result.error
    assert "out_of_time" in seen.read_text().split()


def test_boolean_labels_count_as_zero_and_one() -> None:
    labels = report_step._labels(pa.table({"y": pa.array([True, False, None])}), "y")
    assert labels[:2].tolist() == [1.0, 0.0]
    assert math.isnan(labels[2])  # a missing label stays missing


def test_hooks_that_drop_rows_are_fine_when_nothing_needs_keys(
    demo_project: Path, fake_registry: AdapterRegistry
) -> None:
    _row_dropping_hooks(demo_project)
    _, job = make_training_job(demo_project, fake_registry)
    result = run_job(job)
    assert result.status == "success", result.error
    assert result.report is not None and not result.report.periods


def test_path_adapters_score_the_after_test_split_from_a_staged_file(
    demo_project: Path, fake_registry: AdapterRegistry, monkeypatch: pytest.MonkeyPatch
) -> None:
    reporting_project(demo_project)
    monkeypatch.setattr(FakeTrainingAdapter, "data_access", "path")
    _, job = make_training_job(demo_project, fake_registry)
    result = run_job(job)
    assert result.status == "success", result.error
    assert result.report is not None and result.report.out_of_time_rows > 0
    assert any(cell.period == "day_of_month" for cell in result.report.periods)


def test_an_adapter_without_importance_gets_permutation_importance(
    demo_project: Path, fake_registry: AdapterRegistry, monkeypatch: pytest.MonkeyPatch
) -> None:
    from mbt.execute import job as job_module

    monkeypatch.setattr(job_module, "_feature_importance", lambda runtime, model: {})
    _, job = make_training_job(demo_project, fake_registry)
    with recording_bus() as sink:
        result = run_job(job)
    assert result.status == "success", result.error
    # the fake scores ignore features, so every shuffle drops nothing
    assert result.feature_importance
    assert set(result.feature_importance.values()) == {0.0}
    assert any("feature importance by permutation" in m for m in sink.messages())


class _FeatureDriven(FakeTrainingAdapter):
    """Scores follow one feature, so shuffling it is measurable."""

    def predict(self, model: Any, data: Any, split: str) -> pa.Table:
        table = data.read(split)
        values = [float(v) for v in table.column("signal").to_pylist()]
        return table.append_column("prediction", pa.array(values, type=pa.float64()))


def _permutation_runtime(n_features: int = 2, labels: list[int] | None = None) -> Any:
    labels = labels if labels is not None else [i % 2 for i in range(40)]
    columns = {
        "signal": [float(v) for v in labels],
        **{f"noise_{i}": [float(j % 3) for j in range(40)] for i in range(n_features - 1)},
        "y": labels,
    }
    table = pa.table(columns)
    spec = minimal_model_spec()
    runtime = make_inline_runtime({"train": table, "test": table}, spec, adapter=_FeatureDriven())
    runtime.transformed.read("test")
    return runtime, table


def _scored(table: pa.Table, runtime: Any) -> ScoredSplit:
    import numpy as np

    return ScoredSplit(
        name="test",
        features=table,
        passthrough=pa.table({}),
        key_columns=[],
        labels=np.asarray(table.column("y").to_pylist(), dtype=float),
        scores=np.asarray(table.column("signal").to_pylist(), dtype=float),
        times=None,
    )


def test_permutation_importance_ranks_the_feature_the_model_uses() -> None:
    runtime, table = _permutation_runtime()
    with recording_bus():
        importance = report_step.permutation_importance(runtime, None, _scored(table, runtime))
    assert importance["signal"] == 1.0 and importance["noise_0"] == 0.0


def test_permutation_importance_skips_wide_models_and_single_classes() -> None:
    runtime, table = _permutation_runtime(n_features=report_step.PERMUTATION_MAX_FEATURES + 1)
    with recording_bus() as sink:
        assert report_step.permutation_importance(runtime, None, _scored(table, runtime)) == {}
    assert any("skipping the permutation fallback" in m for m in sink.messages())

    runtime, table = _permutation_runtime(labels=[1] * 40)
    assert report_step.permutation_importance(runtime, None, _scored(table, runtime)) == {}


def test_permutation_importance_samples_large_test_splits(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setattr(report_step, "PERMUTATION_ROWS", 10)
    runtime, table = _permutation_runtime()
    with recording_bus() as sink:
        report_step.permutation_importance(runtime, None, _scored(table, runtime))
    assert any("on 10 test rows" in m for m in sink.messages())


def test_permutation_importance_for_regression() -> None:
    runtime, table = _permutation_runtime()
    runtime.spec = minimal_model_spec(
        task="regression", evaluation={"protocol": {"split": "temporal"}, "metrics": ["rmse"]}
    )
    with recording_bus():
        importance = report_step.permutation_importance(runtime, None, _scored(table, runtime))
    assert importance["signal"] == 1.0


class _DocumentOnlyTracker:
    def __init__(self) -> None:
        self.documents: list[str] = []

    def log_document(self, run: Any, path: Path) -> None:
        self.documents.append(path.name)


class _BrokenTracker:
    def log_directory(self, run: Any, local_dir: Path, artifact_path: str) -> None:
        raise RuntimeError("tracker is down")


def _publishing_runtime(tmp_path: Path) -> Any:
    from mbt.storage import artifact_store_for

    runtime = make_inline_runtime({"train": pa.table({"y": [0]})}, minimal_model_spec())
    runtime.store = artifact_store_for(f"file://{tmp_path}/store", run_prefix="m/r")
    return runtime


def _tiny_report() -> tuple[ReportData, ReportMeta]:
    summary = ReportSummary(
        anchor="2026-07-01T00:00:00+00:00", reference_rows=1, out_of_time_rows=0
    )
    return ReportData(summary=summary), ReportMeta(model="m", run="r", anchor="a", binary=True)


def test_a_tracker_without_directories_gets_the_page_and_summary(tmp_path: Path) -> None:
    runtime = _publishing_runtime(tmp_path)
    data, meta = _tiny_report()
    tracker = _DocumentOnlyTracker()
    summary = report_step.publish_report(
        runtime, data, meta, tracking=tracker, run_handle=RunHandle(run_id="x"), prefix="report"
    )
    assert tracker.documents == ["report.html", "summary.json"]
    assert summary.report_uri and summary.report_uri.endswith("m/r/report/report.html")
    stored = json.loads((tmp_path / "store" / "m" / "r" / "report" / "summary.json").read_text())
    assert stored["report_uri"] == summary.report_uri


def test_a_tracker_outage_does_not_fail_publishing(tmp_path: Path) -> None:
    runtime = _publishing_runtime(tmp_path)
    data, meta = _tiny_report()
    with recording_bus() as sink:
        report_step.publish_report(
            runtime,
            data,
            meta,
            tracking=_BrokenTracker(),
            run_handle=RunHandle(run_id="x"),
            prefix="report",
        )
    assert any("could not upload the training report" in m for m in sink.messages())
    assert (tmp_path / "store" / "m" / "r" / "report" / "report.html").is_file()


def test_config_documents_need_a_directory_capable_tracker(tmp_path: Path) -> None:
    runtime = _publishing_runtime(tmp_path)
    tracker = _DocumentOnlyTracker()
    report_step.log_config_documents(runtime, tracker, RunHandle(run_id="x"), runtime.spec)
    assert tracker.documents == []


def test_report_metrics_keep_window_and_months_only() -> None:
    def cell(period: str, key: str) -> PeriodCell:
        return PeriodCell(
            period=period,  # type: ignore[arg-type]
            key=key,
            slot="all",
            start="s",
            end="e",
            n_rows=3,
            n_labelled=2,
            mature=True,
            metrics={"roc_auc": 0.7, "ks": float("nan")},
        )

    def stable(period: str, key: str) -> StabilityCell:
        return StabilityCell(
            period=period,  # type: ignore[arg-type]
            key=key,
            slot="all",
            n_rows=3,
            reference_rows=4,
            score_psi=0.01,
            features_over_fail=1,
        )

    summary = ReportSummary(
        anchor="a",
        reference_rows=4,
        out_of_time_rows=3,
        split_metrics={"train": {"roc_auc": 0.9}, "test": {"roc_auc": 0.8}},
        periods=[cell("window", "window"), cell("month", "2026-06"), cell("day_of_month", "x")],
        stability=[
            stable("window", "window"),
            stable("month", "2026-06"),
            stable("week_of_month", "w"),
        ],
    )
    metrics = report_step.report_metrics(summary)
    assert metrics == {
        "train.roc_auc": 0.9,
        "oot.window.roc_auc": 0.7,
        "oot.window.n_labelled": 2.0,
        "oot.month.2026-06.roc_auc": 0.7,
        "oot.month.2026-06.n_labelled": 2.0,
        "stability.window.score_psi": 0.01,
        "stability.window.features_over_fail": 1.0,
        "stability.month.2026-06.score_psi": 0.01,
        "stability.month.2026-06.features_over_fail": 1.0,
    }


def test_the_anchor_falls_back_to_now_and_records_need_a_dataset() -> None:
    runtime = make_inline_runtime({"train": pa.table({"y": [0]})}, minimal_model_spec())
    assert report_step.job_anchor(runtime).tzinfo is not None
    assert report_step.dataset_record(runtime) is None
    runtime.job.anchor = "2026-07-01T00:00:00Z"
    assert report_step.job_anchor(runtime).isoformat() == "2026-07-01T00:00:00+00:00"


def test_training_reads_its_own_detail_into_the_log(
    demo_project: Path, fake_registry: AdapterRegistry
) -> None:
    from report_unit_helpers import edit

    edit(
        demo_project / MODEL_FILE,
        "protocol: {split: temporal}",
        'protocol: {split: temporal, test_window: "-14d:now"}',
    )
    _, job = make_training_job(demo_project, fake_registry)
    with recording_bus() as sink:
        result = run_job(job)
    assert result.status == "success", result.error
    messages = sink.messages()
    assert any(m.startswith("test narrowed by test_window") for m in messages)
    assert any(m.startswith("window test:") for m in messages)


def test_oot_check_mode_needs_an_artifact(
    demo_project: Path, fake_registry: AdapterRegistry
) -> None:
    _, job = make_training_job(demo_project, fake_registry, mode="oot_check")
    result = run_job(job)
    assert result.status == "error"
    assert "oot_check mode requires" in (result.error or "")


def test_run_train_reports_builder_warnings(
    monkeypatch: pytest.MonkeyPatch, tmp_path: Path
) -> None:
    from mbt.storage import artifact_store_for

    table = pa.table({"x": [float(i) for i in range(20)], "y": [i % 2 for i in range(20)]})
    spec = minimal_model_spec(
        evaluation={
            "protocol": {"split": "temporal"},
            "metrics": ["pr_auc"],
            "report": {"binning": [{"strategy": "fixed_width", "width": 0.0001}]},
        }
    )
    runtime = make_inline_runtime(
        {"train": table, "test": table},
        spec,
        adapter=FakeTrainingAdapter(),
        builtin_specs=[MetricSpec(name="pr_auc")],
    )
    runtime.job.champion = None
    runtime.store = artifact_store_for(f"file://{tmp_path}", run_prefix="t")
    with recording_bus() as sink:
        result = _run_train(runtime, None, None)
    assert result.status == "success"
    assert any(m.startswith("report: fixed_width binning skipped") for m in sink.messages())
