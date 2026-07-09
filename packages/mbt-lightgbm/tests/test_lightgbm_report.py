"""train_with_report: per-round validation progress for tuning pruners
(section 3.5), mirroring the xgboost contract via lightgbm callbacks."""

import pytest
from mbt_lightgbm.adapter import LightGBMTrainingAdapter

from mbt_adapter_base import EvaluationProtocol, EvaluationSpec, ModelSpec, RunContext, TaskType
from mbt_adapter_base.compliance import tiny_binary_dataset
from mbt_adapter_base.datasets import InMemoryDatasetHandle


def _spec(n_estimators: int) -> ModelSpec:
    return ModelSpec.model_validate(
        {
            "name": "m",
            "task": TaskType.BINARY_CLASSIFICATION,
            "adapter": "lightgbm",
            "owner": "t@example.com",
            "dataset": "ref('d')",
            "target": "label",
            "hyperparameters": {
                "n_estimators": n_estimators,
                "num_leaves": 15,
                "min_child_samples": 5,
            },
            "evaluation": EvaluationSpec(protocol=EvaluationProtocol(), metrics=["roc_auc"]),
            "seed": 5,
        }
    )


def _ctx() -> RunContext:
    class _Null:
        def emit(self, event: object) -> None: ...

    return RunContext(
        run_id="t",
        unique_id="m",
        seed=5,
        target_name="dev",
        project_dir=".",
        vars={},
        events=_Null(),
    )


def _tuning_handle() -> InMemoryDatasetHandle:
    base = tiny_binary_dataset()
    train, val = base.read("train"), base.read("test")
    return InMemoryDatasetHandle(
        {"train": train, "validation": val, "test": val}, label_column="label"
    )


def test_train_with_report_streams_validation_auc_per_round() -> None:
    seen: list[tuple[int, float]] = []
    LightGBMTrainingAdapter({}).train_with_report(
        _spec(15), _tuning_handle(), _ctx(), lambda step, value: seen.append((step, value))
    )
    assert [step for step, _ in seen] == list(range(15))
    assert all(0.0 <= value <= 1.0 for _, value in seen)


def test_raising_report_aborts_training_promptly() -> None:
    class Pruned(Exception):
        pass

    calls: list[int] = []

    def report(step: int, value: float) -> None:
        calls.append(step)
        if step == 2:
            raise Pruned()

    with pytest.raises(Pruned):
        LightGBMTrainingAdapter({}).train_with_report(_spec(50), _tuning_handle(), _ctx(), report)
    assert calls == [0, 1, 2]
