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
