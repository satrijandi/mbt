"""Unit tests for the task-schema registry and the binary_classification schema."""

from typing import ClassVar

import pytest

import mbt.config.tasks as tasks_mod
from mbt.config.tasks import (
    BinaryClassificationSchema,
    get_task_schema,
    register_task_schema,
    supported_tasks,
)
from mbt.contracts import DatasetProfile, ModelSpec, TaskType
from mbt.exceptions import ConfigError


@pytest.fixture()
def scratch_registry(monkeypatch) -> None:
    """Mutations land in a copy of the global task-schema registry."""
    monkeypatch.setattr(tasks_mod, "_REGISTRY", dict(tasks_mod._REGISTRY))


class _RegressionSchema:
    task = TaskType.REGRESSION
    allowed_metrics: ClassVar[set[str]] = set()

    def is_allowed_metric(self, name: str) -> bool:
        return False

    def validate_spec(self, spec):
        return []

    def validate_dataset(self, spec, profile):
        return []


def test_get_task_schema_builtin_and_unsupported() -> None:
    assert isinstance(get_task_schema(TaskType.BINARY_CLASSIFICATION), BinaryClassificationSchema)
    with pytest.raises(ConfigError, match="'regression' has no registered task schema") as excinfo:
        get_task_schema(TaskType.REGRESSION)
    assert "binary_classification" in (excinfo.value.hint or "")


def test_register_task_schema_rejects_duplicates(scratch_registry) -> None:
    with pytest.raises(ConfigError, match="already registered") as excinfo:
        register_task_schema(BinaryClassificationSchema())
    assert "override=True" in (excinfo.value.hint or "")


def test_register_task_schema_new_and_override(scratch_registry) -> None:
    schema = _RegressionSchema()
    register_task_schema(schema)
    assert tasks_mod.get_task_schema(TaskType.REGRESSION) is schema
    assert TaskType.REGRESSION in tasks_mod.supported_tasks()

    replacement = BinaryClassificationSchema()
    register_task_schema(replacement, override=True)
    assert tasks_mod.get_task_schema(TaskType.BINARY_CLASSIFICATION) is replacement


def test_supported_tasks_reflects_registry() -> None:
    assert TaskType.BINARY_CLASSIFICATION in supported_tasks()


# -- BinaryClassificationSchema ---------------------------------------------------


def make_model_spec(slices: list[str] | None = None) -> ModelSpec:
    return ModelSpec.model_validate(
        {
            "name": "m",
            "task": "binary_classification",
            "adapter": "fake",
            "owner": "ds@example.com",
            "dataset": "ref('d')",
            "target": "churned",
            "evaluation": {
                "protocol": {"split": "temporal"},
                "metrics": ["pr_auc"],
                "slices": slices or [],
            },
            "seed": 1,
        }
    )


def make_profile(balance: dict[str, float] | None) -> DatasetProfile:
    return DatasetProfile(
        n_rows={"train": 100, "test": 20},
        columns={"churned": "int64", "plan_type": "string"},
        label_column="churned",
        label_balance=balance,
    )


def test_allowed_metrics_and_sugar() -> None:
    schema = BinaryClassificationSchema()
    assert "pr_auc" in schema.allowed_metrics
    assert schema.is_allowed_metric("recall_at_precision_0.9")
    assert not schema.is_allowed_metric("mean_squared_error")


def test_validate_spec_rejects_slicing_by_target() -> None:
    schema = BinaryClassificationSchema()
    issues = schema.validate_spec(make_model_spec(slices=["churned", "plan_type"]))
    assert len(issues) == 1
    assert issues[0].severity == "error"
    assert issues[0].field_path == "/evaluation/slices/0"
    assert "meaningless" in issues[0].message

    assert schema.validate_spec(make_model_spec(slices=["plan_type"])) == []


def test_validate_dataset_branches() -> None:
    schema = BinaryClassificationSchema()
    spec = make_model_spec()

    not_binary = schema.validate_dataset(spec, make_profile({"1": 1.0}))
    assert len(not_binary) == 1
    assert "requires a binary label" in not_binary[0].message

    bad_encoding = schema.validate_dataset(spec, make_profile({"yes": 0.5, "no": 0.5}))
    assert len(bad_encoding) == 1
    assert "must be encoded as 0/1 or bool" in bad_encoding[0].message

    imbalanced = schema.validate_dataset(spec, make_profile({"1": 0.0005, "0": 0.9995}))
    assert len(imbalanced) == 1
    assert imbalanced[0].severity == "warning"
    assert "extreme class imbalance" in imbalanced[0].message

    assert schema.validate_dataset(spec, make_profile({"1": 0.3, "0": 0.7})) == []
