"""Shared helpers for the parsing/config/jinja/compile unit-test cluster."""

from typing import Any, ClassVar

from pydantic import BaseModel, ConfigDict

from mbt.adapters.registry import AdapterRegistry
from mbt.contracts import (
    CONTRACT_VERSION,
    AdapterPlugin,
    ModelSpec,
    TaskType,
    ValidationIssue,
)
from mbt.parsing import ParsedProject


class EmptyParams(BaseModel):
    """A hyperparameter model with no fields (strict)."""

    model_config = ConfigDict(extra="forbid")


class _UnitTrainingAdapter:
    """Minimal training-adapter surface for parse-time checks."""

    contract_version = CONTRACT_VERSION
    data_access = "arrow"
    supported_tasks: ClassVar[set[TaskType]] = {TaskType.BINARY_CLASSIFICATION}

    def __init__(self, config: dict[str, Any] | None = None) -> None:
        self.config = config or {}

    def param_model(self, task: TaskType) -> type[BaseModel]:
        return EmptyParams

    def validate(self, spec: ModelSpec) -> list[ValidationIssue]:
        return []


class SchemalessTaskAdapter(_UnitTrainingAdapter):
    """Supports a task that has no registered task schema (survival is a v1
    task; core registers no schema for it)."""

    name = "reggy"
    supported_tasks: ClassVar[set[TaskType]] = {TaskType.SURVIVAL}


class FussyAdapter(_UnitTrainingAdapter):
    """Emits one error and one warning from ``validate()``."""

    name = "fussy"

    def validate(self, spec: ModelSpec) -> list[ValidationIssue]:
        return [
            ValidationIssue(
                severity="error",
                resource=spec.name,
                field_path="/adapter",
                message="fussy adapter says no",
                hint="be less fussy",
            ),
            ValidationIssue(
                severity="warning",
                resource=spec.name,
                field_path="/adapter",
                message="fussy adapter is uneasy",
            ),
        ]


class PathAdapter(_UnitTrainingAdapter):
    """A path-access adapter (like the JVM adapters); used to check the
    arrow-only walk-forward backtest gate (R2-7)."""

    name = "pathy"
    data_access = "path"


def register_unit_plugins(registry: AdapterRegistry) -> None:
    """Register the notrain/reggy/fussy/pathy plugins used by parser unit tests."""
    registry.register(
        AdapterPlugin(name="notrain", contract_version=CONTRACT_VERSION, training=None)
    )
    registry.register(
        AdapterPlugin(
            name="reggy", contract_version=CONTRACT_VERSION, training=SchemalessTaskAdapter
        )
    )
    registry.register(
        AdapterPlugin(name="fussy", contract_version=CONTRACT_VERSION, training=FussyAdapter)
    )
    registry.register(
        AdapterPlugin(name="pathy", contract_version=CONTRACT_VERSION, training=PathAdapter)
    )


class ListSink:
    """Event sink that records every event it sees."""

    def __init__(self) -> None:
        self.events: list[Any] = []

    def write(self, event: Any) -> None:
        self.events.append(event)


def error_messages(parsed: ParsedProject) -> list[str]:
    return [issue.message for issue in parsed.report.errors]


def warning_messages(parsed: ParsedProject) -> list[str]:
    return [issue.message for issue in parsed.report.warnings]
