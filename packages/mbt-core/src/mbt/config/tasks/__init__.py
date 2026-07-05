"""Task schema registry (TSD §5.6, FR-RES-08).

Core registers builtin task schemas; adapter plugins may register more via
``AdapterPlugin.task_schemas`` (v1: survival, ranking) without core changes.
"""

from mbt.config.tasks.binary import BinaryClassificationSchema
from mbt.contracts import TaskSchema, TaskType
from mbt.exceptions import ConfigError

_REGISTRY: dict[TaskType, TaskSchema] = {
    TaskType.BINARY_CLASSIFICATION: BinaryClassificationSchema(),
}


def register_task_schema(schema: TaskSchema, *, override: bool = False) -> None:
    """Register a task schema (used by adapter plugins, TSD §12.3)."""
    if schema.task in _REGISTRY and not override:
        raise ConfigError(
            f"task schema for {schema.task} is already registered",
            hint="pass override=True to replace a builtin (not recommended)",
        )
    _REGISTRY[schema.task] = schema


def get_task_schema(task: TaskType) -> TaskSchema:
    """The schema for a task; unsupported tasks are a parse error."""
    schema = _REGISTRY.get(task)
    if schema is None:
        supported = ", ".join(sorted(t.value for t in _REGISTRY))
        raise ConfigError(
            f"task {task.value!r} has no registered task schema",
            hint=f"supported tasks: {supported}. v1 tasks arrive via adapter plugins.",
        )
    return schema


def supported_tasks() -> set[TaskType]:
    return set(_REGISTRY)


__all__ = [
    "BinaryClassificationSchema",
    "get_task_schema",
    "register_task_schema",
    "supported_tasks",
]
