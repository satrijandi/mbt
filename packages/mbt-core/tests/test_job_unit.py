"""In-process unit tests for the training-job entrypoint (mbt/execute/job.py).

Jobs normally run in subprocesses (which coverage does not trace); these
tests import and call the job functions directly with real inputs.
"""

import datetime as dt
import json
import runpy
import sys
from pathlib import Path
from types import SimpleNamespace

import pyarrow as pa
import pytest
from core_helpers import write
from exec_unit_helpers import (
    make_inline_runtime,
    make_scoring_job,
    make_training_job,
    minimal_model_spec,
    recording_bus,
)
from mbt_testing.adapters import FakeTrainingAdapter
from test_execution import invoke
from test_scoring_execution import SCORING_YML, _build_and_promote, _prediction_runs, _write_batch

from mbt.adapters.local.compute import result_path_for
from mbt.adapters.registry import AdapterRegistry, get_registry
from mbt.contracts import (
    AUTO,
    AdapterRef,
    JobResult,
    MetricResults,
    MetricSpec,
    ScoringOutputSpec,
    TuningResult,
    ValidationIssue,
)
from mbt.events import get_bus, set_bus
from mbt.exceptions import AdapterError, ConfigError
from mbt.execute.job import (
    _carve_validation,
    _champion_delta_bounds,
    _feature_importance,
    _prepare,
    _render_adapter_ref,
    _run_score,
    _run_train,
    _run_tuning,
    main,
    run_job,
)

SOURCES_WITH_BATCH = """
sources:
  - name: lakehouse
    tables:
      - name: subscribers
        path: data/subscribers/*.parquet
      - name: scoring_batch
        path: data/scoring_batch/*.parquet
"""

PREDICTION_SHIFT_YML = """
scoring:
  - name: churn_scoring
    owner: lifecycle-eng@example.com
    model: ref('churn_model')
    input:
      source: source('lakehouse', 'scoring_batch')
      time_column: snapshot_date
      window: "-7d:now"
    monitors:
      prediction_shift:
        threshold: 0.95
    output:
      path: predictions/churn_scores
      columns: [user_id]
"""


@pytest.fixture()
def score_project(demo_project: Path) -> Path:
    write(demo_project / "sources.yml", SOURCES_WITH_BATCH)
    write(demo_project / "scoring/churn_scoring.yml", SCORING_YML)
    _write_batch(demo_project)
    return demo_project


@pytest.fixture()
def restore_bus():
    previous = get_bus()
    yield
    set_bus(previous)


def _tables(n: int = 10) -> dict[str, pa.Table]:
    return {
        "train": pa.table({"x": [float(i) for i in range(n)], "y": [i % 2 for i in range(n)]}),
        "test": pa.table({"x": [1.0, 2.0], "y": [0, 1]}),
    }


class _RegistryWithoutTraining:
    """Delegates everything but returns plugins with no training adapter."""

    def __init__(self, real: AdapterRegistry) -> None:
        self._real = real

    def component(self, kind, name, config):
        return self._real.component(kind, name, config)

    def get(self, name):
        return SimpleNamespace(training=None)


# -- _render_adapter_ref (TSD §18) -------------------------------------------------


def test_render_adapter_ref_env_var_and_var(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setenv("MBT_UNIT_SECRET", "s3cret")
    monkeypatch.delenv("MBT_UNIT_MISSING", raising=False)
    ref = AdapterRef(
        adapter="fake",
        config={
            "token": "{{ env_var('MBT_UNIT_SECRET') }}",
            "fallback": "{{ env_var('MBT_UNIT_MISSING', 'dflt') }}",
            "threshold": "{{ var('threshold', 3) }}",
            "nested": {"paths": ["{{ var('root') }}/data"]},
            "plain": 7,
        },
    )
    rendered = _render_adapter_ref(ref, {"root": "/tmp/unit"})
    assert rendered.config["token"] == "s3cret"
    assert rendered.config["fallback"] == "dflt"
    assert rendered.config["threshold"] == "3"
    assert rendered.config["nested"]["paths"] == ["/tmp/unit/data"]
    assert rendered.config["plain"] == 7


def test_render_adapter_ref_missing_env_raises(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.delenv("MBT_UNIT_ABSENT", raising=False)
    ref = AdapterRef(adapter="fake", config={"k": "{{ env_var('MBT_UNIT_ABSENT') }}"})
    with pytest.raises(ConfigError, match="MBT_UNIT_ABSENT"):
        _render_adapter_ref(ref, {})


# -- run_job dispatch and error handling --------------------------------------------


def test_evaluate_mode_reevaluates_artifact_and_champion(
    demo_project: Path, fake_registry: AdapterRegistry
) -> None:
    _, job = make_training_job(demo_project, fake_registry)
    trained = run_job(job)
    assert trained.status == "success", trained.error
    eval_job = job.model_copy(
        update={
            "mode": "evaluate",
            "artifact": trained.artifact,
            "champion": trained.artifact,
            "tracking": None,
        }
    )
    result = run_job(eval_job)
    assert result.status == "success", result.error
    assert result.metrics is not None and result.metrics.metrics["pr_auc"] > 0.5
    assert result.champion_metrics is not None
    # demo gates are threshold gates: no champion-confidence gates, no bounds
    assert result.champion_delta_bounds == {}
    assert result.feature_importance == {"fake_signal": 0.75, "fake_noise": 0.25}
    assert result.artifact == trained.artifact


def test_evaluate_mode_without_artifact_is_an_mbt_error(
    demo_project: Path, fake_registry: AdapterRegistry
) -> None:
    _, job = make_training_job(demo_project, fake_registry, mode="evaluate", tracking=None)
    result = run_job(job)
    assert result.status == "error"
    assert "evaluate mode requires an artifact" in (result.error or "")


def test_score_mode_requires_scoring_fields(
    demo_project: Path, fake_registry: AdapterRegistry
) -> None:
    _, job = make_training_job(demo_project, fake_registry)
    missing_node = run_job(job.model_copy(update={"mode": "score"}))
    assert missing_node.status == "error"
    assert "score mode requires model_node" in (missing_node.error or "")

    missing_artifact = run_job(
        job.model_copy(
            update={
                "mode": "score",
                "model_node": job.node,
                "output": ScoringOutputSpec(path="predictions/unit"),
                "run_key": "unit-key",
            }
        )
    )
    assert missing_artifact.status == "error"
    assert "champion artifact" in (missing_artifact.error or "")


# -- _prepare ------------------------------------------------------------------------


def test_prepare_rejects_adapter_without_training(
    demo_project: Path, fake_registry: AdapterRegistry, monkeypatch: pytest.MonkeyPatch
) -> None:
    _, job = make_training_job(demo_project, fake_registry)
    real = get_registry()
    monkeypatch.setattr("mbt.execute.job.get_registry", lambda: _RegistryWithoutTraining(real))
    with pytest.raises(AdapterError, match="provides no training adapter"):
        _prepare(job)


def test_prepare_emits_dataset_validation_warnings(
    demo_project: Path, fake_registry: AdapterRegistry, monkeypatch: pytest.MonkeyPatch
) -> None:
    _, job = make_training_job(demo_project, fake_registry)
    issue = ValidationIssue(
        severity="warning",
        resource=job.node.unique_id,
        field_path="/target",
        message="unit imbalance warning",
    )

    class _Schema:
        def validate_dataset(self, spec, profile):
            return [issue]

    monkeypatch.setattr("mbt.config.tasks.get_task_schema", lambda task: _Schema())
    with recording_bus() as sink:
        result = run_job(job)
    assert result.status == "success", result.error
    assert any("unit imbalance warning" in m for m in sink.messages())


def test_prepare_fails_on_dataset_validation_errors(
    demo_project: Path, fake_registry: AdapterRegistry, monkeypatch: pytest.MonkeyPatch
) -> None:
    _, job = make_training_job(demo_project, fake_registry)
    issue = ValidationIssue(
        severity="error",
        resource=job.node.unique_id,
        field_path="/target",
        message="unit label error",
        hint="fix the label",
    )

    class _Schema:
        def validate_dataset(self, spec, profile):
            return [issue]

    monkeypatch.setattr("mbt.config.tasks.get_task_schema", lambda task: _Schema())
    result = run_job(job)
    assert result.status == "error"
    assert "unit label error" in (result.error or "")


def test_path_adapter_gets_materialized_splits(
    demo_project: Path, fake_registry: AdapterRegistry, monkeypatch: pytest.MonkeyPatch
) -> None:
    monkeypatch.setattr(FakeTrainingAdapter, "data_access", "path")
    _, job = make_training_job(demo_project, fake_registry)
    result = run_job(job)
    assert result.status == "success", result.error
    assert result.metrics is not None and result.metrics.metrics["pr_auc"] > 0.5


# -- hooks in the job (TSD §5.8) -----------------------------------------------------


def _use_hook_metric(demo_project: Path) -> None:
    model_yml = demo_project / "models/churn_model.yml"
    model_yml.write_text(
        model_yml.read_text().replace(
            "metrics: [pr_auc, roc_auc]", "metrics: [pr_auc, roc_auc, my_metric]"
        )
    )


def test_hook_transform_and_custom_metrics_run_in_job(
    demo_project: Path, fake_registry: AdapterRegistry
) -> None:
    write(
        demo_project / "models/churn_model.py",
        """
        import pyarrow as pa

        def transform_features(table, ctx):
            doubled = pa.array(
                [v.as_py() * 2 for v in table.column("tenure_days")], type=pa.int64()
            )
            return table.append_column("tenure_days_x2", doubled)

        def custom_metrics(predictions, ctx):
            return {"my_metric": 0.5}
        """,
    )
    _use_hook_metric(demo_project)
    _, job = make_training_job(demo_project, fake_registry)
    assert job.node.hooks_path == "models/churn_model.py"
    result = run_job(job)
    assert result.status == "success", result.error
    assert result.metrics is not None and result.metrics.metrics["my_metric"] == 0.5


def test_hook_metric_without_custom_metrics_errors(
    demo_project: Path, fake_registry: AdapterRegistry
) -> None:
    write(
        demo_project / "models/churn_model.py",
        """
        def transform_features(table, ctx):
            return table
        """,
    )
    _use_hook_metric(demo_project)
    _, job = make_training_job(demo_project, fake_registry)
    result = run_job(job)
    assert result.status == "error"
    assert "exposes no custom_metrics" in (result.error or "")


def test_hook_metrics_missing_declared_name_errors(
    demo_project: Path, fake_registry: AdapterRegistry
) -> None:
    write(
        demo_project / "models/churn_model.py",
        """
        def custom_metrics(predictions, ctx):
            return {"other_metric": 0.1}
        """,
    )
    _use_hook_metric(demo_project)
    _, job = make_training_job(demo_project, fake_registry)
    result = run_job(job)
    assert result.status == "error"
    assert "did not return declared metric(s): my_metric" in (result.error or "")


# -- small helpers -------------------------------------------------------------------


def test_feature_importance_absent_returns_empty() -> None:
    runtime = SimpleNamespace(adapter=SimpleNamespace())
    assert _feature_importance(runtime, object()) == {}


def test_champion_delta_bounds_without_confidence_gates() -> None:
    spec = minimal_model_spec(
        evaluation={
            "protocol": {"split": "temporal"},
            "metrics": ["pr_auc"],
            "gates": [{"metric": "pr_auc", "threshold": 0.5}],
        }
    )
    runtime = make_inline_runtime(_tables(), spec, builtin_specs=[MetricSpec(name="pr_auc")])
    assert _champion_delta_bounds(runtime, object(), object()) == {}


# -- implicit validation carve (TSD §13.5, ADR-8) ------------------------------------


def test_carve_returns_none_with_explicit_validation_split() -> None:
    tables = _tables()
    tables["validation"] = pa.table({"x": [9.0], "y": [1]})
    runtime = make_inline_runtime(tables, minimal_model_spec())
    assert _carve_validation(runtime) is None


def test_carve_temporal_with_date_column() -> None:
    n = 10
    dates = [dt.date(2026, 1, 1) + dt.timedelta(days=i * 9) for i in range(n)]
    tables = {
        "train": pa.table(
            {
                "t": pa.array(dates, type=pa.date32()),
                "x": [float(i) for i in range(n)],
                "y": [i % 2 for i in range(n)],
            }
        ),
        "test": pa.table(
            {
                "t": pa.array([dt.date(2026, 5, 1)], type=pa.date32()),
                "x": [1.0],
                "y": [1],
            }
        ),
    }
    job = SimpleNamespace(
        dataset_windows={"windows": {"train": ["2026-01-01T00:00:00Z", "2026-04-01T00:00:00Z"]}},
        node=SimpleNamespace(unique_id="model.demo.unit_model"),
    )
    runtime = make_inline_runtime(tables, minimal_model_spec(), time_column="t", job=job)
    carved = _carve_validation(runtime)
    assert carved is not None
    assert carved.read("train").num_rows == 8
    assert carved.read("validation").num_rows == 2


def test_carve_random_without_time_column() -> None:
    runtime = make_inline_runtime(_tables(), minimal_model_spec())
    carved = _carve_validation(runtime)
    assert carved is not None
    assert carved.read("train").num_rows == 8
    assert carved.read("validation").num_rows == 2


def test_carve_with_too_few_rows_errors() -> None:
    tables = {
        "train": pa.table({"x": [1.0], "y": [1]}),
        "test": pa.table({"x": [2.0], "y": [0]}),
    }
    runtime = make_inline_runtime(tables, minimal_model_spec())
    with pytest.raises(ConfigError, match="empty split"):
        _carve_validation(runtime)


def test_carve_applies_hooks_to_the_carved_handle() -> None:
    class _IdentityHooks:
        has_transform = True
        has_custom_metrics = False

        def transform_features(self, table, ctx):
            return table

    runtime = make_inline_runtime(_tables(), minimal_model_spec(), hooks=_IdentityHooks())
    carved = _carve_validation(runtime)
    assert carved is not None
    assert carved.read("train").num_rows == 8  # triggers the carved hook context


# -- tuning (ADR-8, FR-TUNE-01..04) --------------------------------------------------


def _tuning_spec(objective: str = "pr_auc", **extra):
    return {
        "engine": "fake",
        "n_trials": 2,
        "search_space": {"max_depth": {"type": "int", "low": 2, "high": 5}},
        "objective": {"metric": objective, "direction": "maximize"},
        **extra,
    }


def _tuning_job(engine: bool = True) -> SimpleNamespace:
    return SimpleNamespace(
        dataset_windows={},
        node=SimpleNamespace(unique_id="model.demo.unit_model"),
        tuning_engine=AdapterRef(adapter="fake") if engine else None,
        tuning_cap=None,
        vars={},
        metric_specs=[],
    )


def test_tuning_without_engine_errors() -> None:
    spec = minimal_model_spec(hyperparameters={"fake_metric_value": 0.6}, tuning=_tuning_spec())
    runtime = make_inline_runtime(
        _tables(),
        spec,
        adapter=FakeTrainingAdapter(),
        builtin_specs=[MetricSpec(name="pr_auc")],
        job=_tuning_job(engine=False),
    )
    with pytest.raises(ConfigError, match="no tuning engine"):
        _run_tuning(runtime, spec)


def test_tuning_objective_not_resolved_errors() -> None:
    spec = minimal_model_spec(
        hyperparameters={"fake_metric_value": 0.6},
        evaluation={"protocol": {"split": "temporal"}, "metrics": ["pr_auc", "roc_auc"]},
        tuning=_tuning_spec(objective="roc_auc"),
    )
    runtime = make_inline_runtime(
        _tables(),
        spec,
        adapter=FakeTrainingAdapter(),
        builtin_specs=[MetricSpec(name="pr_auc")],  # roc_auc deliberately absent
        job=_tuning_job(),
    )
    with pytest.raises(ConfigError, match="not a resolved metric"):
        _run_tuning(runtime, spec)


def test_tuning_pruner_without_progress_reports_warns() -> None:
    class _PlainAdapter:  # no train_with_report
        def train(self, spec, data, ctx):
            data.read("train")
            return SimpleNamespace(value=float(spec.hyperparameters.get("max_depth", 1)))

        def evaluate(self, model, data, split, metrics, slices=None):
            data.read(split)
            return MetricResults(metrics={m.name: model.value for m in metrics}, slices={})

    spec = minimal_model_spec(
        hyperparameters={"fake_metric_value": 0.6}, tuning=_tuning_spec(pruner="median")
    )
    runtime = make_inline_runtime(
        _tables(),
        spec,
        adapter=_PlainAdapter(),
        builtin_specs=[MetricSpec(name="pr_auc")],
        job=_tuning_job(),
    )
    with recording_bus() as sink:
        tuned, result = _run_tuning(runtime, spec)
    assert result is not None and result.n_trials == 2
    assert "max_depth" in tuned.hyperparameters
    assert any("does not report training progress" in m for m in sink.messages())


def test_tuning_engine_report_uses_train_with_report(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    class _ReportingEngine:
        def tune(self, spec, objective, n_trials, seed):
            value = objective({"max_depth": 3}, report=lambda step, val: None)
            return TuningResult(best_params={"max_depth": 3}, best_value=value, n_trials=1)

    real = get_registry()

    class _Proxy:
        def component(self, kind, name, config):
            if kind == "tuning":
                return _ReportingEngine()
            return real.component(kind, name, config)

        def get(self, name):
            return real.get(name)

    proxy = _Proxy()
    monkeypatch.setattr("mbt.execute.job.get_registry", lambda: proxy)
    spec = minimal_model_spec(hyperparameters={"fake_metric_value": 0.6}, tuning=_tuning_spec())
    runtime = make_inline_runtime(
        _tables(),
        spec,
        adapter=FakeTrainingAdapter(),
        builtin_specs=[MetricSpec(name="pr_auc")],
        job=_tuning_job(),
    )
    tuned, result = _run_tuning(runtime, spec)
    assert result is not None and result.best_params == {"max_depth": 3}
    assert tuned.hyperparameters["max_depth"] == 3


def test_tuning_logs_trials_when_tracking_supports_it() -> None:
    class _TrialTracker:
        def __init__(self) -> None:
            self.trials: list[tuple[int, dict, float]] = []

        def log_trial(self, run, index, params, value):
            self.trials.append((index, params, value))

    tracker = _TrialTracker()
    spec = minimal_model_spec(hyperparameters={"fake_metric_value": 0.6}, tuning=_tuning_spec())
    runtime = make_inline_runtime(
        _tables(),
        spec,
        adapter=FakeTrainingAdapter(),
        builtin_specs=[MetricSpec(name="pr_auc")],
        job=_tuning_job(),
    )
    _, result = _run_tuning(runtime, spec, tracker, SimpleNamespace(run_id="run-1"))
    assert result is not None and result.n_trials == 2
    assert [index for index, _, _ in tracker.trials] == [0, 1]


def test_tuning_materializes_carve_for_path_adapters() -> None:
    adapter = FakeTrainingAdapter()
    adapter.data_access = "path"  # instance-level override
    spec = minimal_model_spec(hyperparameters={"fake_metric_value": 0.6}, tuning=_tuning_spec())
    runtime = make_inline_runtime(
        _tables(),
        spec,
        adapter=adapter,
        builtin_specs=[MetricSpec(name="pr_auc")],
        job=_tuning_job(),
    )
    _, result = _run_tuning(runtime, spec)
    assert result is not None and result.n_trials == 2


# -- _run_train ----------------------------------------------------------------------


def test_leftover_auto_sentinels_are_an_adapter_error() -> None:
    class _KeepAutoAdapter:
        def resolve_auto(self, spec, profile):
            return spec

    spec = minimal_model_spec(hyperparameters={"scale_pos_weight": AUTO})
    runtime = make_inline_runtime(_tables(), spec, adapter=_KeepAutoAdapter())
    with pytest.raises(AdapterError, match="AUTO sentinels unresolved"):
        _run_train(runtime, None, None)


def test_failed_training_marks_tracking_run_failed(
    demo_project: Path, fake_registry: AdapterRegistry
) -> None:
    _, job = make_training_job(demo_project, fake_registry)
    job.node.config["hyperparameters"]["fail_training"] = True
    result = run_job(job)
    assert result.status == "error"
    assert "fake training failure" in (result.error or "")
    payloads = [
        json.loads(p.read_text())
        for p in (demo_project / "target" / "fake_tracking").glob("*.json")
    ]
    assert any(p["status"] == "FAILED" for p in payloads)


# -- score mode (ADR-20/21) ----------------------------------------------------------


def test_score_plugin_without_training_adapter_errors(
    score_project: Path, fake_registry: AdapterRegistry, monkeypatch: pytest.MonkeyPatch
) -> None:
    _build_and_promote(score_project, fake_registry)
    _, job = make_scoring_job(score_project, fake_registry)
    real = get_registry()
    monkeypatch.setattr("mbt.execute.job.get_registry", lambda: _RegistryWithoutTraining(real))
    with pytest.raises(AdapterError, match="provides no training adapter"):
        _run_score(job)


def test_score_applies_model_hooks(score_project: Path, fake_registry: AdapterRegistry) -> None:
    write(
        score_project / "models/churn_model.py",
        """
        import pyarrow as pa

        def transform_features(table, ctx):
            doubled = pa.array(
                [v.as_py() * 2 for v in table.column("tenure_days")], type=pa.int64()
            )
            return table.append_column("tenure_days_x2", doubled)
        """,
    )
    _build_and_promote(score_project, fake_registry)
    results = invoke(score_project, fake_registry, "score")
    assert results.exit_code() == 0
    assert results.results[0].metrics["rows_scored"] == 120.0


def test_score_hooks_changing_row_count_errors(
    score_project: Path, fake_registry: AdapterRegistry
) -> None:
    write(
        score_project / "models/churn_model.py",
        """
        def transform_features(table, ctx):
            if ctx.split == "score":
                return table.slice(0, table.num_rows - 5)
            return table
        """,
    )
    _build_and_promote(score_project, fake_registry)
    results = invoke(score_project, fake_registry, "score")
    assert results.exit_code() == 1
    node = results.results[0]
    assert node.status == "error"
    assert "row count" in (node.message or "")


def test_score_missing_passthrough_column_errors(
    score_project: Path, fake_registry: AdapterRegistry
) -> None:
    scoring_yml = score_project / "scoring/churn_scoring.yml"
    # only the output block (6-space indent), not the not_null check columns
    scoring_yml.write_text(
        scoring_yml.read_text().replace(
            "\n      columns: [user_id]", "\n      columns: [user_id, mystery_column]"
        )
    )
    _build_and_promote(score_project, fake_registry)
    results = invoke(score_project, fake_registry, "score")
    assert results.exit_code() == 1
    node = results.results[0]
    assert node.status == "error"
    assert "passthrough column(s) missing" in (node.message or "")
    assert "mystery_column" in (node.message or "")


def test_score_path_adapter_materializes_input(
    score_project: Path, fake_registry: AdapterRegistry, monkeypatch: pytest.MonkeyPatch
) -> None:
    _build_and_promote(score_project, fake_registry)
    monkeypatch.setattr(FakeTrainingAdapter, "data_access", "path")
    results = invoke(score_project, fake_registry, "score")
    assert results.exit_code() == 0
    assert results.results[0].metrics["rows_scored"] == 120.0


def test_score_empty_batch_writes_zero_row_run(
    score_project: Path, fake_registry: AdapterRegistry
) -> None:
    _build_and_promote(score_project, fake_registry)
    # every row falls outside the -7d:now scoring window
    from datetime import timedelta

    from core_helpers import TEST_ANCHOR

    base = TEST_ANCHOR.replace(tzinfo=None) - timedelta(days=100)
    table = pa.table(
        {
            "user_id": list(range(30)),
            "snapshot_date": [base] * 30,
            "is_active": [True] * 30,
            "tenure_days": [100] * 30,
            "monthly_usage": [50.0] * 30,
            "plan_type": ["basic"] * 30,
        }
    )
    import pyarrow.parquet as pq

    pq.write_table(table, score_project / "data" / "scoring_batch" / "part-000.parquet")
    results = invoke(score_project, fake_registry, "score")
    assert results.exit_code() == 0
    node = results.results[0]
    assert node.status == "success"
    assert node.metrics["rows_scored"] == 0.0
    runs = _prediction_runs(score_project)
    assert len(runs) == 1
    info = json.loads((runs[0] / "predictions.json").read_text())
    assert info["row_count"] == 0


def test_score_prediction_shift_monitor_reaches_tracking(
    demo_project: Path, fake_registry: AdapterRegistry
) -> None:
    write(demo_project / "sources.yml", SOURCES_WITH_BATCH)
    write(demo_project / "scoring/churn_scoring.yml", PREDICTION_SHIFT_YML)
    _write_batch(demo_project)
    _build_and_promote(demo_project, fake_registry)
    results = invoke(demo_project, fake_registry, "score")
    # the fake model scores unlabeled input with a constant, so the score
    # distribution deterministically breaches the test-split baseline
    assert results.exit_code() == 2
    node = results.results[0]
    assert node.status == "monitor_failed"
    assert node.monitors and {m.monitor for m in node.monitors} == {"prediction_shift"}
    assert not node.monitors[0].passed
    assert "score distribution" in (node.message or "")
    tracking_file = demo_project / "target/fake_tracking" / f"{node.tracking_run_id}.json"
    payload = json.loads(tracking_file.read_text())
    assert "monitor.prediction_shift" in payload["metrics"]


# -- main() / module entrypoint ------------------------------------------------------


def test_main_usage_error(capsys: pytest.CaptureFixture) -> None:
    assert main([]) == 2
    assert "usage:" in capsys.readouterr().err


def test_main_executes_job_file(
    demo_project: Path, fake_registry: AdapterRegistry, restore_bus
) -> None:
    _, job = make_training_job(demo_project, fake_registry)
    job_path = demo_project / "target" / "unit_job.json"
    job_path.write_text(job.model_dump_json())
    assert main([str(job_path)]) == 0
    result = JobResult.model_validate_json(result_path_for(job_path).read_text())
    assert result.status == "success"


def test_main_returns_3_on_job_failure(
    demo_project: Path, fake_registry: AdapterRegistry, restore_bus
) -> None:
    _, job = make_training_job(demo_project, fake_registry)
    job.node.config["hyperparameters"]["fail_training"] = True
    job_path = demo_project / "target" / "failing_job.json"
    job_path.write_text(job.model_dump_json())
    assert main([str(job_path)]) == 3
    result = JobResult.model_validate_json(result_path_for(job_path).read_text())
    assert result.status == "error"


@pytest.mark.filterwarnings("ignore::RuntimeWarning")
def test_module_entrypoint_raises_systemexit(
    demo_project: Path,
    fake_registry: AdapterRegistry,
    monkeypatch: pytest.MonkeyPatch,
    restore_bus,
) -> None:
    _, job = make_training_job(demo_project, fake_registry)
    job_path = demo_project / "target" / "entry_job.json"
    job_path.write_text(job.model_dump_json())
    monkeypatch.setattr(sys, "argv", ["mbt-job", str(job_path)])
    with pytest.raises(SystemExit) as excinfo:
        runpy.run_module("mbt.execute.job", run_name="__main__")
    assert excinfo.value.code == 0
