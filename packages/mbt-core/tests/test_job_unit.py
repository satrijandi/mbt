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
    _backtest_folds,
    _carve_calibration,
    _carve_validation,
    _champion_delta_bounds,
    _feature_importance,
    _partial_dependence,
    _prepare,
    _render_adapter_ref,
    _run_score,
    _run_train,
    _run_tuning,
    _walk_forward_backtest,
    main,
    run_job,
)
from mbt.secrets import clear_taints, redact

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
    ref = AdapterRef(adapter="fake", config={"k": "{{ env('MBT_UNIT_ABSENT') }}"})
    with pytest.raises(ConfigError, match="MBT_UNIT_ABSENT"):
        _render_adapter_ref(ref, {})


def test_render_adapter_ref_env_resolves_without_tainting(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """The job re-resolves both accessors, and must keep the split (A-1).

    A job that re-tainted a non-secret would corrupt its own result JSON on
    the way back to the coordinator, which is exactly the reported symptom.
    """
    monkeypatch.setenv("MBT_UNIT_PLAIN", "1")
    monkeypatch.delenv("MBT_UNIT_MISSING", raising=False)
    ref = AdapterRef(
        adapter="fake",
        config={
            "port": "{{ env('MBT_UNIT_PLAIN') }}",
            "fallback": "{{ env('MBT_UNIT_MISSING', 'dflt') }}",
        },
    )
    clear_taints()
    rendered = _render_adapter_ref(ref, {})
    assert rendered.config == {"port": "1", "fallback": "dflt"}
    assert redact('{"pr_auc":0.1234}') == '{"pr_auc":0.1234}'


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


def test_feature_importance_prefers_shap_over_gain_when_available() -> None:
    """The model card uses the data-grounded SHAP importance when the adapter
    exposes it (tree adapters), falling back to model-intrinsic gain otherwise."""
    shap_adapter = SimpleNamespace(
        shap_importance=lambda model, handle, split: {"a": 0.7, "b": 0.3},
        feature_importance=lambda model: {"a": 0.5, "b": 0.5},  # the gain fallback
    )
    runtime = SimpleNamespace(adapter=shap_adapter, handle=object())
    assert _feature_importance(runtime, object()) == {"a": 0.7, "b": 0.3}  # SHAP preferred

    gain_only = SimpleNamespace(feature_importance=lambda model: {"a": 0.5, "b": 0.5})
    assert _feature_importance(SimpleNamespace(adapter=gain_only), object()) == {"a": 0.5, "b": 0.5}


def test_partial_dependence_covers_top_numeric_features_only() -> None:
    """Partial dependence is computed for the top numeric features by importance;
    categorical and unknown-column features are skipped, and each curve is a list
    of [grid_value, avg_prediction] pairs (explainability)."""
    from mbt_testing.adapters import FakeModel

    from mbt_adapter_base.datasets import InMemoryDatasetHandle

    table = pa.table(
        {
            "num": [float(i) for i in range(20)],  # numeric -> PD computed
            "cat": ["a", "b"] * 10,  # categorical -> skipped
            "label": [i % 2 for i in range(20)],
        }
    )
    runtime = SimpleNamespace(
        handle=InMemoryDatasetHandle({"test": table}, label_column="label"),
        adapter=FakeTrainingAdapter({}),
    )
    importance = {"num": 0.7, "cat": 0.2, "missing": 0.1}
    curves = _partial_dependence(runtime, FakeModel(value=0.6, target="label"), importance)

    assert set(curves) == {"num"}  # categorical + unknown-column features skipped
    curve = curves["num"]
    assert len(curve) >= 2 and all(len(point) == 2 for point in curve)  # [grid, avg] pairs


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


# -- dedicated calibration carve (F17) + backtest composition (F5) -------------------


def test_calibration_carve_keeps_all_splits_and_shrinks_train() -> None:
    spec = minimal_model_spec(calibration="isotonic")
    runtime = make_inline_runtime(_tables(), spec)
    carved = _carve_calibration(runtime, spec, runtime.transformed)
    assert carved.splits() == {"train", "calibration", "test"}
    assert carved.read("train").num_rows == 8
    assert carved.read("calibration").num_rows == 2
    # test passes through untouched
    assert carved.read("test").num_rows == 2
    # train and calibration partition the original train rows exactly
    train_x = carved.read("train").column("x").to_pylist()
    cal_x = carved.read("calibration").column("x").to_pylist()
    assert not set(train_x) & set(cal_x)
    assert sorted(train_x + cal_x) == [float(i) for i in range(10)]


def test_calibration_carve_is_seeded_independently_of_the_validation_carve() -> None:
    """The calibration carve is seed+5, the implicit validation carve seed+2:
    a spec using both must not hand the calibrator the tuning-selection rows."""
    spec = minimal_model_spec(calibration="isotonic")
    runtime = make_inline_runtime(_tables(40), spec)
    validation = _carve_validation(runtime)
    assert validation is not None
    calibration = _carve_calibration(runtime, spec, runtime.transformed)
    val_rows = set(validation.read("validation").column("x").to_pylist())
    cal_rows = set(calibration.read("calibration").column("x").to_pylist())
    assert val_rows != cal_rows  # different seeds pick different held-out rows
    # and the carve itself is deterministic
    again = _carve_calibration(runtime, spec, runtime.transformed)
    assert set(again.read("calibration").column("x").to_pylist()) == cal_rows


def test_calibration_carve_temporal_takes_the_train_tail() -> None:
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
            {"t": pa.array([dt.date(2026, 5, 1)], type=pa.date32()), "x": [1.0], "y": [1]}
        ),
    }
    job = SimpleNamespace(
        dataset_windows={"windows": {"train": ["2026-01-01T00:00:00Z", "2026-04-01T00:00:00Z"]}},
        node=SimpleNamespace(unique_id="model.demo.unit_model"),
    )
    spec = minimal_model_spec(calibration="isotonic")
    runtime = make_inline_runtime(tables, spec, time_column="t", job=job)
    carved = _carve_calibration(runtime, spec, runtime.transformed)
    assert carved.read("train").num_rows == 8
    cal = carved.read("calibration")
    assert cal.num_rows == 2
    # the slice is the temporal TAIL of the train window (most recent rows)
    assert cal.column("x").to_pylist() == [8.0, 9.0]


def test_calibration_carve_with_too_few_rows_errors() -> None:
    tables = {
        "train": pa.table({"x": [1.0], "y": [1]}),
        "test": pa.table({"x": [2.0], "y": [0]}),
    }
    spec = minimal_model_spec(calibration="isotonic")
    runtime = make_inline_runtime(tables, spec)
    with pytest.raises(ConfigError, match="calibration carve produced an empty split") as excinfo:
        _carve_calibration(runtime, spec, runtime.transformed)
    assert "drop 'calibration'" in (excinfo.value.hint or "")


class _SplitRecordingAdapter(FakeTrainingAdapter):
    """Records the splits and calibration setting of every train() call."""

    def __init__(self, config=None):
        super().__init__(config)
        self.seen: list[tuple[frozenset, str | None]] = []

    def train(self, spec, data, ctx):
        self.seen.append((frozenset(data.splits()), spec.calibration))
        return super().train(spec, data, ctx)


def test_backtest_folds_carve_a_calibration_slice_when_the_spec_calibrates() -> None:
    """F5 (real fix): each fold's fit sees a carved 'calibration' split, so the
    fold models calibrate the way the production model does."""
    adapter = _SplitRecordingAdapter()
    spec = minimal_model_spec(
        calibration="isotonic",
        evaluation={
            "protocol": {"split": "random", "backtest_folds": 3},
            "metrics": ["pr_auc"],
        },
    )
    runtime = make_inline_runtime(
        _tables(30), spec, adapter=adapter, builtin_specs=[MetricSpec(name="pr_auc")]
    )
    means, _stds = _walk_forward_backtest(runtime, spec, 3)
    assert means  # folds actually ran
    assert len(adapter.seen) == 3
    for splits, calibration in adapter.seen:
        assert "calibration" in splits
        assert calibration == "isotonic"


def test_backtest_folds_do_not_carve_without_calibration() -> None:
    adapter = _SplitRecordingAdapter()
    spec = minimal_model_spec(
        evaluation={
            "protocol": {"split": "random", "backtest_folds": 3},
            "metrics": ["pr_auc"],
        },
    )
    runtime = make_inline_runtime(
        _tables(30), spec, adapter=adapter, builtin_specs=[MetricSpec(name="pr_auc")]
    )
    _walk_forward_backtest(runtime, spec, 3)
    assert adapter.seen and all("calibration" not in splits for splits, _ in adapter.seen)


def test_calibration_carve_applies_hooks_to_the_carved_handle() -> None:
    class _IdentityHooks:
        has_transform = True
        has_custom_metrics = False

        def transform_features(self, table, ctx):
            return table

    spec = minimal_model_spec(calibration="isotonic")
    runtime = make_inline_runtime(_tables(), spec, hooks=_IdentityHooks())
    carved = _carve_calibration(runtime, spec, runtime.transformed)
    assert carved.read("train").num_rows == 8  # triggers the carved hook context
    assert carved.read("calibration").num_rows == 2


def test_final_fit_trains_on_the_calibration_carved_handle(tmp_path: Path) -> None:
    """_run_train (F17): a calibrated spec's final fit sees train minus the
    dedicated 'calibration' slice, for arrow and path adapters alike."""
    from mbt.storage import artifact_store_for

    for data_access in ("arrow", "path"):
        adapter = _SplitRecordingAdapter()
        if data_access == "path":
            adapter.data_access = "path"  # instance-level override
        spec = minimal_model_spec(calibration="isotonic")
        runtime = make_inline_runtime(
            _tables(), spec, adapter=adapter, builtin_specs=[MetricSpec(name="pr_auc")]
        )
        runtime.job.champion = None
        runtime.store = artifact_store_for(f"file://{tmp_path}/{data_access}", run_prefix="t")
        result = _run_train(runtime, None, None)
        assert result.status == "success"
        assert len(adapter.seen) == 1
        splits, calibration = adapter.seen[0]
        assert "calibration" in splits and calibration == "isotonic"


def test_tuning_trials_never_calibrate() -> None:
    """Trials must not fit a calibrator on the split their objective is scored
    on - that would make a brier/ece objective circularly optimal (F17); the
    final fit calibrates on its own dedicated slice instead."""
    adapter = _SplitRecordingAdapter()
    spec = minimal_model_spec(
        calibration="isotonic",
        hyperparameters={"fake_metric_value": 0.6},
        tuning=_tuning_spec(),
    )
    runtime = make_inline_runtime(
        _tables(30),
        spec,
        adapter=adapter,
        builtin_specs=[MetricSpec(name="pr_auc")],
        job=_tuning_job(),
    )
    _run_tuning(runtime, spec)
    assert adapter.seen and all(calibration is None for _, calibration in adapter.seen)


# -- tuning (ADR-8, FR-TUNE-01..04) --------------------------------------------------


def _tuning_spec(objective: str = "pr_auc", robust: bool = False, **extra):
    return {
        "engine": "fake",
        "n_trials": 2,
        "search_space": {"max_depth": {"type": "int", "low": 2, "high": 5}},
        "objective": {"metric": objective, "direction": "maximize", "robust": robust},
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


def test_robust_tuning_selects_on_the_bootstrap_lower_bound() -> None:
    """A robust objective (R2-7) reports the bootstrap lower bound of the
    validation metric, which sits below the point estimate the plain objective
    uses - the same trial wins, but the selection is defended against luck."""

    def _tuned(robust: bool) -> TuningResult:
        spec = minimal_model_spec(
            hyperparameters={"fake_metric_value": 0.7},
            evaluation={"protocol": {"split": "temporal"}, "metrics": ["pr_auc"]},
            tuning=_tuning_spec(robust=robust),
        )
        runtime = make_inline_runtime(
            _tables(60),
            spec,
            adapter=FakeTrainingAdapter(),
            builtin_specs=[MetricSpec(name="pr_auc")],
            job=_tuning_job(),
        )
        _, result = _run_tuning(runtime, spec)
        assert result is not None
        return result

    plain, robust = _tuned(False), _tuned(True)
    assert robust.best_params == plain.best_params  # the same trial wins either way
    assert robust.best_value < plain.best_value  # ...reported as the pessimistic bound


def test_robust_tuning_rejects_a_hook_objective() -> None:
    spec = minimal_model_spec(
        hyperparameters={"fake_metric_value": 0.6},
        evaluation={"protocol": {"split": "temporal"}, "metrics": ["pr_auc", "custom_hook"]},
        tuning=_tuning_spec(objective="custom_hook", robust=True),
    )
    runtime = make_inline_runtime(
        _tables(),
        spec,
        adapter=FakeTrainingAdapter(),
        builtin_specs=[MetricSpec(name="pr_auc")],
        hook_specs=[MetricSpec(name="custom_hook", kind="hook")],
        job=_tuning_job(),
    )
    with pytest.raises(ConfigError, match="builtin metrics only"):
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


# -- walk-forward backtest (_walk_forward_backtest, R2-7) ---------------------------


class _RecordingBacktestAdapter:
    """Records the feature values it trains/evaluates on per fold, so the test
    can assert the walk-forward folds are time-ordered, expanding, and non-leaky."""

    data_access = "arrow"

    def __init__(self, fold_values: list[float] | None = None) -> None:
        self.train_x: list[list[float]] = []
        self.test_x: list[list[float]] = []
        # per-fold metric values (to exercise a non-zero backtest std); default is
        # a fixed 0.5 every fold (std 0).
        self._fold_values = fold_values

    def train(self, spec: object, handle: object, ctx: object) -> object:
        self.train_x.append(handle.read("train").column("x").to_pylist())  # type: ignore[attr-defined]
        return object()

    def evaluate(
        self, model: object, handle: object, split: str, metrics: list, slices: object = None
    ) -> MetricResults:
        self.test_x.append(handle.read(split).column("x").to_pylist())  # type: ignore[attr-defined]
        value = 0.5 if self._fold_values is None else self._fold_values[len(self.test_x) - 1]
        return MetricResults(metrics={m.name: value for m in metrics}, slices={})


class _IdentityHooks:
    """A transform hook that leaves the table unchanged - present so each fold's
    TransformedDatasetHandle exercises the hook-context path (faithful refit)."""

    has_transform = True

    def transform_features(self, table: object, ctx: object) -> object:
        return table


def test_walk_forward_backtest_refits_on_expanding_time_ordered_prefixes() -> None:
    # 20 rows written in DESCENDING time order (x tracks the time rank), so the
    # backtest must sort by the time column before cutting folds.
    n = 20
    anchor = dt.datetime(2026, 1, 1)
    table = pa.table(
        {
            "ts": [anchor + dt.timedelta(days=n - 1 - i) for i in range(n)],
            "x": [float(n - 1 - i) for i in range(n)],
            "y": [(n - 1 - i) % 2 for i in range(n)],
        }
    )
    adapter = _RecordingBacktestAdapter()
    runtime = make_inline_runtime(
        {"train": table},
        minimal_model_spec(),
        time_column="ts",
        adapter=adapter,
        hooks=_IdentityHooks(),  # exercise the per-fold hook-context path
        builtin_specs=[MetricSpec(name="pr_auc")],
    )
    result, std = _walk_forward_backtest(runtime, runtime.spec, 4)

    # 4 folds -> 3 walk-forward steps, each on a strictly larger prefix
    assert [len(t) for t in adapter.train_x] == [5, 10, 15]
    for train_x, test_x in zip(adapter.train_x, adapter.test_x, strict=True):
        assert max(train_x) < min(test_x)  # train entirely before test (no leakage)
    assert result == {"pr_auc": 0.5}  # mean of the per-fold metric
    assert std == {"pr_auc": 0.0}  # identical folds -> zero spread


def _embargo_job(embargo: str) -> SimpleNamespace:
    return SimpleNamespace(
        dataset_windows={"embargo": embargo},
        node=SimpleNamespace(unique_id="model.demo.unit_model"),
        tuning_engine=None,
        tuning_cap=None,
        vars={},
        metric_specs=[],
    )


def _daily_train(n: int) -> pa.Table:
    anchor = dt.datetime(2026, 1, 1)
    return pa.table(
        {
            "ts": [anchor + dt.timedelta(days=i) for i in range(n)],
            "x": [float(i) for i in range(n)],
            "y": [i % 2 for i in range(n)],
        }
    )


def test_backtest_folds_gap_each_boundary_by_the_embargo() -> None:
    """R2-7/F6: split.embargo must gap each walk-forward fold's train tail from
    its test window, exactly as it gaps the single train/test split - otherwise
    the backtest leaks at every fold boundary (train-tail labels are observed
    inside the next fold's evaluation window)."""
    base_train = _daily_train(20)
    runtime = make_inline_runtime(
        {"train": base_train}, minimal_model_spec(), time_column="ts", job=_embargo_job("3d")
    )
    folds = _backtest_folds(runtime, runtime.spec, base_train, 4)

    # 4 folds -> boundaries at rows 5/10/15; a 3-day embargo drops the 3-day tail
    # before each, shrinking the prefixes 5/10/15 -> 2/7/12.
    assert [train.num_rows for train, _ in folds] == [2, 7, 12]
    for train, test in folds:
        last_train = max(train.column("ts").to_pylist())
        first_test = min(test.column("ts").to_pylist())
        assert first_test - last_train > dt.timedelta(days=3)  # strictly beyond the embargo


def test_backtest_fold_dropped_when_the_embargo_consumes_its_prefix() -> None:
    """A fold whose entire (earliest, shortest) train prefix falls inside the
    embargo has no leakage-free history and is dropped rather than trained on
    nothing (F6)."""
    base_train = _daily_train(20)
    runtime = make_inline_runtime(
        {"train": base_train}, minimal_model_spec(), time_column="ts", job=_embargo_job("8d")
    )
    folds = _backtest_folds(runtime, runtime.spec, base_train, 4)
    # fold 1's 5-row prefix (days 0-4) lies entirely within 8 days of its day-5
    # boundary, so it is dropped; only folds 2 and 3 survive (2 and 7 train rows).
    assert [train.num_rows for train, _ in folds] == [2, 7]


def test_backtest_reports_the_fold_to_fold_std_beside_the_mean() -> None:
    """R2-7: the backtest reports the population std across folds, so an estimate
    whose folds disagree (an unstable model) is distinguishable from a stable one
    with the same mean."""
    import statistics

    n = 20
    anchor = dt.datetime(2026, 1, 1)
    table = pa.table(
        {
            "ts": [anchor + dt.timedelta(days=i) for i in range(n)],
            "x": [float(i) for i in range(n)],
            "y": [i % 2 for i in range(n)],
        }
    )
    # 4 folds -> 3 walk-forward steps, each returning a different metric value
    fold_values = [0.4, 0.6, 0.8]
    adapter = _RecordingBacktestAdapter(fold_values=fold_values)
    runtime = make_inline_runtime(
        {"train": table},
        minimal_model_spec(),
        time_column="ts",
        adapter=adapter,
        builtin_specs=[MetricSpec(name="pr_auc")],
    )
    means, stds = _walk_forward_backtest(runtime, runtime.spec, 4)

    assert means == {"pr_auc": round(statistics.fmean(fold_values), 6)}  # 0.6
    assert stds == {"pr_auc": round(statistics.pstdev(fold_values), 6)}  # ~0.163299
    assert stds["pr_auc"] > 0.0  # the folds genuinely disagree


def test_walk_forward_backtest_without_time_column_returns_empty() -> None:
    table = pa.table({"x": [1.0, 2.0, 3.0, 4.0], "y": [0, 1, 0, 1]})
    adapter = _RecordingBacktestAdapter()
    runtime = make_inline_runtime(
        {"train": table},
        minimal_model_spec(),
        adapter=adapter,
        builtin_specs=[MetricSpec(name="pr_auc")],
    )
    assert _walk_forward_backtest(runtime, runtime.spec, 3) == ({}, {})
    assert adapter.train_x == []  # no refit attempted without a time order


def test_walk_forward_backtest_skips_when_folds_are_degenerate() -> None:
    # One row cannot form a single train->test step across 4 folds -> {}.
    table = pa.table({"ts": [dt.datetime(2026, 1, 1)], "x": [1.0], "y": [0]})
    adapter = _RecordingBacktestAdapter()
    runtime = make_inline_runtime(
        {"train": table},
        minimal_model_spec(),
        time_column="ts",
        adapter=adapter,
        builtin_specs=[MetricSpec(name="pr_auc")],
    )
    assert _walk_forward_backtest(runtime, runtime.spec, 4) == ({}, {})
    assert adapter.train_x == []  # every fold degenerate -> skipped


def test_backtest_runs_k_fold_cross_validation_on_a_random_split() -> None:
    """R2-7: backtest_folds on a RANDOM split does k-fold CV - each row is the
    test set exactly once, and each fold trains on the complement (no time order)."""
    n = 20
    table = pa.table({"x": [float(i) for i in range(n)], "y": [i % 2 for i in range(n)]})
    adapter = _RecordingBacktestAdapter()
    spec = minimal_model_spec(evaluation={"protocol": {"split": "random"}, "metrics": ["pr_auc"]})
    runtime = make_inline_runtime(
        {"train": table},
        spec,
        adapter=adapter,
        builtin_specs=[MetricSpec(name="pr_auc")],
    )
    result, _ = _walk_forward_backtest(runtime, runtime.spec, 4)

    all_x = {float(i) for i in range(n)}
    assert len(adapter.test_x) == 4  # k folds
    tested = [x for fold in adapter.test_x for x in fold]
    assert sorted(tested) == sorted(all_x)  # each row tested exactly once (a partition)
    for train_x, test_x in zip(adapter.train_x, adapter.test_x, strict=True):
        assert set(train_x).isdisjoint(test_x)  # leave-one-fold-out: no overlap
        assert set(train_x) | set(test_x) == all_x  # train is the complement of the test fold
    assert result == {"pr_auc": 0.5}  # mean across the k folds


class _NestedRecordingAdapter:
    """Records the (split, rows) of every evaluate call, so the test can verify
    nested CV: outer folds partition the rows, and each fold's inner tuning
    (evaluated on 'validation') never touches that fold's outer-test rows."""

    data_access = "arrow"

    def __init__(self) -> None:
        self.evals: list[tuple[str, list[float]]] = []

    def train(self, spec: object, handle: object, ctx: object) -> object:
        return object()

    def evaluate(
        self, model: object, handle: object, split: str, metrics: list, slices: object = None
    ) -> MetricResults:
        xs = sorted(handle.read(split).column("x").to_pylist())  # type: ignore[attr-defined]
        self.evals.append((split, xs))
        return MetricResults(metrics={m.name: 0.5 for m in metrics}, slices={})


def test_nested_cv_re_tunes_per_fold_without_leaking_the_outer_test() -> None:
    """R2-7: nested CV re-tunes inside each outer fold, so the outer-test fold
    (evaluated on 'test') never appears in that fold's inner tuning (evaluated on
    'validation'), and the outer folds partition the rows."""
    n = 24
    table = pa.table({"x": [float(i) for i in range(n)], "y": [i % 2 for i in range(n)]})
    adapter = _NestedRecordingAdapter()
    spec = minimal_model_spec(
        evaluation={
            "protocol": {"split": "random", "backtest_folds": 3, "nested_cv": True},
            "metrics": ["pr_auc"],
        },
        tuning=_tuning_spec(),
    )
    runtime = make_inline_runtime(
        {"train": table},
        spec,
        adapter=adapter,
        builtin_specs=[MetricSpec(name="pr_auc")],
        job=_tuning_job(),
    )
    result, _ = _walk_forward_backtest(runtime, runtime.spec, 3, nested=True)
    assert result == {"pr_auc": 0.5}  # the k-fold nested estimate

    # group evals into folds: each is [validation evals..., one 'test' eval]
    folds: list[list[tuple[str, list[float]]]] = []
    current: list[tuple[str, list[float]]] = []
    for split, xs in adapter.evals:
        current.append((split, xs))
        if split == "test":
            folds.append(current)
            current = []
    assert len(folds) == 3  # one outer test per fold
    tested: list[float] = []
    for fold in folds:
        outer_test = set(fold[-1][1])
        tested += list(outer_test)
        for split, xs in fold[:-1]:
            assert split == "validation"  # inner tuning evaluates on validation, never test
            assert set(xs).isdisjoint(outer_test)  # the outer-test never leaks into tuning
    assert sorted(tested) == [float(i) for i in range(n)]  # outer folds partition all rows


def test_temporal_nested_cv_tunes_on_the_past_only() -> None:
    """R2-7: temporal nested CV re-tunes on each fold's PAST (the expanding
    prefix), so the inner validation is EARLIER in time than the outer-test fold
    it is scored against - no future leakage."""
    n = 30
    anchor = dt.datetime(2026, 1, 1)
    # x tracks the time rank; rows in DESCENDING time order so the sort is exercised
    table = pa.table(
        {
            "ts": [anchor + dt.timedelta(days=n - 1 - i) for i in range(n)],
            "x": [float(n - 1 - i) for i in range(n)],
            "y": [(n - 1 - i) % 2 for i in range(n)],
        }
    )
    adapter = _NestedRecordingAdapter()
    spec = minimal_model_spec(
        evaluation={
            "protocol": {"split": "temporal", "backtest_folds": 3, "nested_cv": True},
            "metrics": ["pr_auc"],
        },
        tuning=_tuning_spec(),
    )
    runtime = make_inline_runtime(
        {"train": table},
        spec,
        time_column="ts",
        adapter=adapter,
        builtin_specs=[MetricSpec(name="pr_auc")],
        job=_tuning_job(),
    )
    assert "pr_auc" in _walk_forward_backtest(runtime, runtime.spec, 3, nested=True)[0]

    folds: list[list[tuple[str, list[float]]]] = []
    current: list[tuple[str, list[float]]] = []
    for split, xs in adapter.evals:
        current.append((split, xs))
        if split == "test":
            folds.append(current)
            current = []
    assert len(folds) == 2  # walk-forward: 3 folds -> 2 outer steps
    for fold in folds:
        outer_test = fold[-1][1]
        for split, xs in fold[:-1]:
            assert split == "validation"
            # the inner validation is the temporal TAIL of the fold's prefix: it
            # ends immediately before the outer-test (a random carve would not).
            assert max(xs) == min(outer_test) - 1


def test_run_train_reports_walk_forward_backtest_when_configured(
    demo_project: Path, fake_registry: AdapterRegistry
) -> None:
    """End to end: a model whose protocol sets backtest_folds gets a populated
    backtest_metrics in its JobResult (the fake adapter evaluates each fold)."""
    from mbt.contracts import ModelSpec

    _, job = make_training_job(demo_project, fake_registry)
    spec = ModelSpec.model_validate(job.node.config)
    protocol = spec.evaluation.protocol.model_copy(update={"backtest_folds": 2})
    evaluation = spec.evaluation.model_copy(update={"protocol": protocol})
    new_spec = spec.model_copy(update={"evaluation": evaluation})
    job = job.model_copy(
        update={"node": job.node.model_copy(update={"config": new_spec.model_dump(mode="json")})}
    )

    result = run_job(job)
    assert result.status == "success", result.error
    assert result.backtest_metrics and "pr_auc" in result.backtest_metrics
    # the std is reported alongside the mean, on the same metric keys
    assert result.backtest_std.keys() == result.backtest_metrics.keys()
