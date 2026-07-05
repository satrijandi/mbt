"""Metric name resolution (TSD §5.7, FR-RES-04).

Resolution order: explicit ``metrics.yml`` entry > sugar-parsed builtin >
task/adapter builtin > hook metric; unknown names are a parse error listing
the candidates.
"""

from mbt_adapter_base.metrics import parse_metric_sugar

from mbt.contracts import MetricSpec, ModelSpec, TaskSchema

#: Builtin metrics where lower is better; everything else defaults to higher.
LOWER_IS_BETTER = frozenset({"logloss", "ece", "brier"})


def default_direction(base_name: str) -> bool:
    """greater_is_better default for a builtin metric."""
    return base_name not in LOWER_IS_BETTER


def resolve_metric(
    name: str,
    declared: dict[str, MetricSpec],
    task_schema: TaskSchema,
    has_hooks: bool,
) -> MetricSpec | str:
    """Resolve one metric name to a MetricSpec, or an error message."""
    if name in declared:
        spec = declared[name]
        if spec.kind == "builtin":
            base = spec.name if spec.name in task_schema.allowed_metrics else None
            if base is None:
                sugar = parse_metric_sugar(spec.name)
                if sugar is None or sugar[0] not in task_schema.allowed_metrics:
                    return (
                        f"metric {name!r} is declared builtin in metrics.yml but is not "
                        f"a builtin for task {task_schema.task.value!r}"
                    )
        return spec

    sugar = parse_metric_sugar(name)
    if sugar is not None and sugar[0] in task_schema.allowed_metrics:
        base, params = sugar
        return MetricSpec(
            name=name, kind="builtin", params=params, greater_is_better=default_direction(base)
        )

    if name in task_schema.allowed_metrics:
        return MetricSpec(name=name, kind="builtin", greater_is_better=default_direction(name))

    if has_hooks:
        # Existence of the metric is validated at run time against the dict
        # returned by hooks.custom_metrics (TSD §5.6).
        return MetricSpec(name=name, kind="hook", greater_is_better=True)

    candidates = sorted(task_schema.allowed_metrics | set(declared))
    return (
        f"unknown metric {name!r} for task {task_schema.task.value!r}; "
        f"candidates: {', '.join(candidates)}; parameterized forms like "
        "'recall_at_precision_0.9' are also accepted; hook metrics require a hooks.py"
    )


def resolve_model_metrics(
    spec: ModelSpec,
    declared: dict[str, MetricSpec],
    task_schema: TaskSchema,
    has_hooks: bool,
) -> tuple[list[MetricSpec], list[str]]:
    """Resolve all of a model's evaluation metrics; returns (specs, errors)."""
    resolved: list[MetricSpec] = []
    errors: list[str] = []
    for name in spec.evaluation.metrics:
        outcome = resolve_metric(name, declared, task_schema, has_hooks)
        if isinstance(outcome, str):
            errors.append(outcome)
        else:
            resolved.append(outcome)
    return resolved, errors


def metric_direction(name: str, metric_specs: list[MetricSpec]) -> bool:
    """greater_is_better for a resolved metric name (gates need this)."""
    for spec in metric_specs:
        if spec.name == name:
            return spec.greater_is_better
    sugar = parse_metric_sugar(name)
    base = sugar[0] if sugar else name
    return default_direction(base)
