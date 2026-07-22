"""mbt-h2o against the adapter compliance suite (FR-ADPT-05).

Heavy (starts a JVM-backed H2O cluster); runs under the e2e marker and
skips when no JVM is available.
"""

import shutil
from typing import ClassVar

import pytest
from mbt_h2o.adapter import H2OAutoMLAdapter

from mbt_adapter_base.compliance import TrainingAdapterCompliance

pytestmark = [
    pytest.mark.e2e,
    pytest.mark.skipif(shutil.which("java") is None, reason="H2O needs a JVM"),
]


class TestH2OAutoMLCompliance(TrainingAdapterCompliance):
    adapter_factory = H2OAutoMLAdapter
    plugin_module = "mbt_h2o.plugin"
    framework_modules = ("h2o",)
    # GLM-only, models-bounded: fast and repeatable for the determinism test
    valid_hyperparameters: ClassVar[dict] = {
        "max_models": 2,
        "include_algos": ["GLM"],
        "nfolds": 0,
    }
    auto_hyperparameter = None

    def test_load_restores_the_column_context(self) -> None:
        """A reloaded MOJO must recover its feature/target context from the
        columns sidecar, so feature_importance (and the champion card's
        importance table) survives a reload, not just a fresh train."""
        import tempfile
        from pathlib import Path

        from mbt_adapter_base import TaskType
        from mbt_adapter_base.compliance.suite import TempArtifactStore

        adapter = self.adapter()
        data = self.dataset()
        model = adapter.train(
            self.model_spec(TaskType.BINARY_CLASSIFICATION), data, self.run_context()
        )
        fresh = adapter.feature_importance(model)
        assert fresh, "GLM leader should attribute importance to features"
        with tempfile.TemporaryDirectory() as tmp:
            store = TempArtifactStore(Path(tmp))
            reloaded = adapter.load(adapter.export(model, "native", store), store)
        assert reloaded.features == model.features
        assert reloaded.target == model.target
        assert adapter.feature_importance(reloaded) == fresh  # importance survives reload

    def test_regression_predictions_are_target_scale(self) -> None:
        """H2O now trains a regressor for `task: regression` (R2-18): the target
        stays numeric (not asfactor'd), so AutoML picks regression and emits
        target-scale predictions - not [0, 1] probabilities - and the GLM leader
        still attributes feature importance."""
        from mbt_adapter_base import MetricSpec, TaskType
        from mbt_adapter_base.compliance.suite import tiny_regression_dataset

        adapter = self.adapter()
        data = tiny_regression_dataset()
        model = adapter.train(self.model_spec(TaskType.REGRESSION), data, self.run_context())

        preds = adapter.predict(model, data, "test")
        assert "prediction" in preds.column_names
        # the target spans well beyond [0, 1]; a misrouted classifier would emit
        # probabilities in [0, 1], so a prediction above 2 proves the regressor
        assert max(preds.column("prediction").to_pylist()) > 2.0
        result = adapter.evaluate(
            model, data, "test", [MetricSpec(name="rmse"), MetricSpec(name="r2")]
        )
        assert result.metrics["r2"] > 0.5 and result.metrics["rmse"] >= 0.0
        importance = adapter.feature_importance(model)
        assert importance and set(importance) <= set(model.features)


# -- post-hoc calibration (R2-8) ------------------------------------------------------


def _h2o_calibration_dataset(n: int = 240):  # type: ignore[no-untyped-def]
    from random import Random

    import pyarrow as pa

    from mbt_adapter_base.datasets import InMemoryDatasetHandle

    rng = Random(13)

    def tbl(m: int) -> "pa.Table":
        xs = [rng.gauss(0, 1) for _ in range(m)]
        ys = [1 if x + rng.gauss(0, 0.4) > 0.3 else 0 for x in xs]
        return pa.table({"x": xs, "label": ys})

    return InMemoryDatasetHandle(
        {"train": tbl(n), "validation": tbl(n // 2), "test": tbl(n // 2)}, label_column="label"
    )


def _h2o_spec(**overrides):  # type: ignore[no-untyped-def]
    from mbt_adapter_base import EvaluationProtocol, EvaluationSpec, ModelSpec, TaskType

    return ModelSpec(
        name="m",
        task=TaskType.BINARY_CLASSIFICATION,
        adapter="h2o_automl",
        owner="t@example.com",
        dataset="ref('d')",
        target="label",
        hyperparameters={"max_models": 1, "include_algos": ["GLM"], "nfolds": 0},
        evaluation=EvaluationSpec(protocol=EvaluationProtocol(), metrics=["roc_auc"]),
        seed=1,
        **overrides,
    )


def _h2o_ctx():  # type: ignore[no-untyped-def]
    from mbt_adapter_base import RunContext

    class _Null:
        def emit(self, event: object) -> None: ...

    return RunContext(
        run_id="t",
        unique_id="m",
        seed=1,
        target_name="dev",
        project_dir=".",
        vars={},
        events=_Null(),
    )


def test_h2o_calibration_applies_in_scores_and_survives_save_load(tmp_path) -> None:  # type: ignore[no-untyped-def]
    """R2-8: an H2O model can post-hoc calibrate. The calibrator is applied at
    the _scores chokepoint and rides through save/load in the MOJO sidecar."""
    import numpy as np

    from mbt_adapter_base.compliance.suite import TempArtifactStore

    adapter = H2OAutoMLAdapter({})
    data = _h2o_calibration_dataset()
    model = adapter.train(_h2o_spec(calibration="isotonic"), data, _h2o_ctx())
    assert model.calibrator is not None and model.calibrator.method == "isotonic"

    calibrated = adapter._scores(model, data, "test")
    model.calibrator, saved = None, model.calibrator
    raw = adapter._scores(model, data, "test")
    model.calibrator = saved
    # _scores routes raw scores through the calibrator (would return raw if not)
    np.testing.assert_allclose(calibrated, saved.transform(raw))
    assert np.all((calibrated >= 0.0) & (calibrated <= 1.0))  # valid probabilities

    store = TempArtifactStore(tmp_path)
    loaded = adapter.load(adapter.export(model, "native", store), store)
    assert loaded.calibrator is not None and loaded.calibrator.method == "isotonic"
    np.testing.assert_allclose(adapter._scores(loaded, data, "test"), calibrated)


def test_h2o_calibration_requires_a_validation_split() -> None:
    import pyarrow as pa

    from mbt_adapter_base.datasets import InMemoryDatasetHandle

    adapter = H2OAutoMLAdapter({})
    tbl = pa.table({"x": [0.1, 0.9, 0.2, 0.8, 0.3, 0.7], "label": [0, 1, 0, 1, 0, 1]})
    data = InMemoryDatasetHandle({"train": tbl, "test": tbl}, label_column="label")
    with pytest.raises(ValueError, match="validation"):
        adapter.train(_h2o_spec(calibration="isotonic"), data, _h2o_ctx())


def test_walk_forward_backtest_runs_on_h2o() -> None:
    """R2-7: the walk-forward backtest works on h2o (a path adapter) too - each
    fold stages to parquet and AutoML (GLM, models-bounded) refits it."""
    from datetime import datetime, timedelta
    from types import SimpleNamespace

    import pyarrow as pa

    from mbt.execute.job import _walk_forward_backtest
    from mbt_adapter_base import MetricSpec
    from mbt_adapter_base.datasets import InMemoryDatasetHandle

    n = 150
    anchor = datetime(2026, 1, 1)
    signal = [((i * 37) % 100) / 100.0 for i in range(n)]  # jumps around within any time slice
    table = pa.table(
        {
            "ts": [anchor + timedelta(days=n - 1 - i) for i in range(n)],  # descending -> must sort
            "x": [signal[n - 1 - i] for i in range(n)],
            "label": [1 if signal[n - 1 - i] > 0.5 else 0 for i in range(n)],
        }
    )
    base = InMemoryDatasetHandle({"train": table}, label_column="label", time_column="ts")
    runtime = SimpleNamespace(
        base_handle=base,
        spec=_h2o_spec(),
        adapter=H2OAutoMLAdapter({}),
        ctx=_h2o_ctx(),
        base_profile=base.profile(),
        hooks=None,
        builtin_specs=[MetricSpec(name="roc_auc")],
        # the fold logic reads the job's resolved windows (embargo, F6)
        job=SimpleNamespace(dataset_windows={}),
    )
    means, stds = _walk_forward_backtest(runtime, runtime.spec, 3)
    assert "roc_auc" in means and 0.0 <= means["roc_auc"] <= 1.0
    assert stds["roc_auc"] >= 0.0  # a std is reported alongside the mean


def test_tuning_block_is_rejected() -> None:
    from mbt_adapter_base import (
        EvaluationProtocol,
        EvaluationSpec,
        ModelSpec,
        TaskType,
        TuningObjective,
        TuningSpec,
    )

    spec = ModelSpec(
        name="m",
        task=TaskType.BINARY_CLASSIFICATION,
        adapter="h2o_automl",
        owner="t@example.com",
        dataset="ref('d')",
        target="label",
        evaluation=EvaluationSpec(protocol=EvaluationProtocol(), metrics=["roc_auc"]),
        tuning=TuningSpec(
            engine="optuna",
            n_trials=5,
            search_space={"max_models": {"type": "int", "low": 1, "high": 5}},
            objective=TuningObjective(metric="roc_auc", direction="maximize"),
        ),
        seed=1,
    )
    issues = H2OAutoMLAdapter({}).validate(spec)
    assert any(i.severity == "error" and "tune the tuner" in i.message for i in issues)


def test_wall_clock_budgets_warn() -> None:
    from mbt_adapter_base import EvaluationProtocol, EvaluationSpec, ModelSpec, TaskType

    spec = ModelSpec(
        name="m",
        task=TaskType.BINARY_CLASSIFICATION,
        adapter="h2o_automl",
        owner="t@example.com",
        dataset="ref('d')",
        target="label",
        hyperparameters={"max_models": 5, "max_runtime_secs": 60},
        evaluation=EvaluationSpec(protocol=EvaluationProtocol(), metrics=["roc_auc"]),
        seed=1,
    )
    adapter = H2OAutoMLAdapter({})
    warnings = adapter.nondeterminism_warnings(spec)
    assert warnings and "max_runtime_secs" in warnings[0]
    assert not adapter.nondeterminism_warnings(
        spec.model_copy(update={"hyperparameters": {"max_models": 5}})
    )


def test_sparkling_backend_without_extra_is_actionable() -> None:
    """h2o_backend=sparkling without mbt-h2o[sparkling] fails with the pip hint."""
    from mbt_adapter_base import RunContext

    class _Null:
        def emit(self, event: object) -> None: ...

    ctx = RunContext(
        run_id="t",
        unique_id="m",
        seed=1,
        target_name="dev",
        project_dir=".",
        vars={"h2o_backend": "sparkling"},
        events=_Null(),
    )
    adapter = H2OAutoMLAdapter({})
    try:
        import pysparkling  # noqa: F401

        pytest.skip("pysparkling installed; guard not reachable")
    except ImportError:
        pass
    with pytest.raises(RuntimeError, match=r"mbt-h2o\[sparkling\]"):
        adapter._h2o(ctx)
