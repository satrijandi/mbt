"""Unit tests for the task-schema registry and the binary_classification schema."""

from typing import ClassVar

import pytest

import mbt.config.tasks as tasks_mod
from mbt.config.tasks import (
    BinaryClassificationSchema,
    RegressionSchema,
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


class _StubTaskSchema:
    """A stand-in for a task a plugin might register (survival is not builtin)."""

    task = TaskType.SURVIVAL
    allowed_metrics: ClassVar[set[str]] = set()

    def validate_spec(self, spec):
        return []

    def validate_dataset(self, spec, profile):
        return []


def test_get_task_schema_builtin_and_unsupported() -> None:
    assert isinstance(get_task_schema(TaskType.BINARY_CLASSIFICATION), BinaryClassificationSchema)
    assert isinstance(get_task_schema(TaskType.REGRESSION), RegressionSchema)  # now a builtin
    with pytest.raises(ConfigError, match="'survival' has no registered task schema") as excinfo:
        get_task_schema(TaskType.SURVIVAL)
    assert "binary_classification" in (excinfo.value.hint or "")


def test_register_task_schema_rejects_duplicates(scratch_registry) -> None:
    with pytest.raises(ConfigError, match="already registered") as excinfo:
        register_task_schema(BinaryClassificationSchema())
    assert "override=True" in (excinfo.value.hint or "")


def test_register_task_schema_new_and_override(scratch_registry) -> None:
    schema = _StubTaskSchema()
    register_task_schema(schema)
    assert tasks_mod.get_task_schema(TaskType.SURVIVAL) is schema
    assert TaskType.SURVIVAL in tasks_mod.supported_tasks()

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


# -- RegressionSchema -------------------------------------------------------------


def make_regression_spec(slices: list[str] | None = None) -> ModelSpec:
    return ModelSpec.model_validate(
        {
            "name": "m",
            "task": "regression",
            "adapter": "xgboost",
            "owner": "ds@example.com",
            "dataset": "ref('d')",
            "target": "spend",
            "evaluation": {
                "protocol": {"split": "temporal"},
                "metrics": ["rmse"],
                "slices": slices or [],
            },
            "seed": 1,
        }
    )


def make_regression_profile(label_dtype: str) -> DatasetProfile:
    return DatasetProfile(
        n_rows={"train": 100, "test": 20},
        columns={"spend": label_dtype, "region": "string"},
        label_column="spend",
    )


def test_regression_allowed_metrics() -> None:
    schema = RegressionSchema()
    assert {"rmse", "mae", "r2"} <= schema.allowed_metrics
    assert "roc_auc" not in schema.allowed_metrics


def test_regression_validate_spec_rejects_slicing_by_target() -> None:
    schema = RegressionSchema()
    issues = schema.validate_spec(make_regression_spec(slices=["spend"]))
    assert len(issues) == 1 and "meaningless" in issues[0].message
    assert schema.validate_spec(make_regression_spec(slices=["region"])) == []


def test_regression_validate_dataset_requires_numeric_target() -> None:
    schema = RegressionSchema()
    spec = make_regression_spec()
    assert schema.validate_dataset(spec, make_regression_profile("double")) == []
    assert schema.validate_dataset(spec, make_regression_profile("int64")) == []

    non_numeric = schema.validate_dataset(spec, make_regression_profile("string"))
    assert len(non_numeric) == 1
    assert "requires a numeric target" in non_numeric[0].message

    # unknown label column -> empty dtype -> error names the column
    missing = schema.validate_dataset(
        spec,
        DatasetProfile(n_rows={"train": 1}, columns={"region": "string"}, label_column="spend"),
    )
    assert len(missing) == 1 and "requires a numeric target" in missing[0].message
