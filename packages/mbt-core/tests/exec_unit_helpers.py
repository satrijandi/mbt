"""Shared helpers for the execute-layer unit tests (unique module name for pytest)."""

import contextlib
from collections.abc import Iterator
from pathlib import Path
from types import SimpleNamespace
from typing import Any

from core_helpers import TEST_ANCHOR

from mbt.adapters.registry import AdapterRegistry
from mbt.contracts import ModelSpec, ScoringSpec, TrainingJob
from mbt.events import EventBus, get_bus, set_bus
from mbt.execute.handles import TransformedDatasetHandle
from mbt.execute.job import _JobRuntime
from mbt.execute.orchestrator import InvocationOptions, prepare
from mbt.execute.runners import DatasetRunner, ExecutionContext, ModelRunner, ScoringRunner
from mbt_adapter_base.datasets import InMemoryDatasetHandle

DATASET_UID = "dataset.demo.churn_training"
MODEL_UID = "model.demo.churn_model"
SCORING_UID = "scoring.demo.churn_scoring"


class RecordingSink:
    """Collects emitted events so tests can assert on warnings."""

    def __init__(self) -> None:
        self.events: list[Any] = []

    def write(self, event: Any) -> None:
        self.events.append(event)

    def messages(self) -> list[str]:
        return [str(getattr(event, "message", "")) for event in self.events]


@contextlib.contextmanager
def recording_bus() -> Iterator[RecordingSink]:
    """Swap in a fresh event bus with a recording sink; always restores."""
    sink = RecordingSink()
    previous = get_bus()
    set_bus(EventBus(sinks=[sink]))
    try:
        yield sink
    finally:
        set_bus(previous)


def make_options(project_dir: Path, command: str = "run", **kwargs: Any) -> InvocationOptions:
    kwargs.setdefault("anchor", TEST_ANCHOR)
    return InvocationOptions(command=command, project_dir=project_dir, **kwargs)


def make_execution_context(
    project_dir: Path,
    registry: AdapterRegistry,
    command: str = "run",
    **opt_kwargs: Any,
) -> ExecutionContext:
    """Compile the project and build a coordinator ExecutionContext."""
    opts = make_options(project_dir, command=command, **opt_kwargs)
    prepared = prepare(opts, registry=registry)
    return ExecutionContext(
        manifest=prepared.manifest,
        profiles=prepared.profiles,
        registry=prepared.registry,
        project_dir=opts.project_dir.resolve(),
        run_id=prepared.run_id,
        command=command,
        cli_vars=opts.cli_vars,
        python_tests=prepared.parsed.python_tests if prepared.parsed else [],
        total_nodes=len(prepared.manifest.nodes),
    )


def make_training_job(
    project_dir: Path, registry: AdapterRegistry, **overrides: Any
) -> tuple[ExecutionContext, TrainingJob]:
    """Materialize the demo dataset and assemble a real fake-adapter job."""
    ctx = make_execution_context(project_dir, registry)
    dataset_result = DatasetRunner(ctx).run(DATASET_UID)
    assert dataset_result.status == "success", dataset_result.message
    node = ctx.manifest.nodes[MODEL_UID]
    spec = ModelSpec.model_validate(node.config)
    runner = ModelRunner(ctx)
    metric_specs = runner._metric_specs(spec, node)
    job = runner._assemble_job(node, spec, metric_specs, None)
    if overrides:
        job = job.model_copy(update=overrides)
    return ctx, job


def make_scoring_job(
    project_dir: Path, registry: AdapterRegistry, **overrides: Any
) -> tuple[ExecutionContext, TrainingJob]:
    """Assemble a real score-mode job (the project must be built + promoted)."""
    ctx = make_execution_context(project_dir, registry, command="score")
    node = ctx.manifest.nodes[SCORING_UID]
    spec = ScoringSpec.model_validate(node.config)
    model_node = ctx.manifest.nodes[MODEL_UID]
    model_spec = ModelSpec.model_validate(model_node.config)
    runner = ScoringRunner(ctx)
    champion = runner._champion(spec, model_spec, node.unique_id)
    baseline = runner._baseline_ref(champion)
    handle = runner._materialize_input(node, spec)
    job = runner._assemble_job(node, model_node, spec, champion, baseline, handle)
    if overrides:
        job = job.model_copy(update=overrides)
    return ctx, job


def minimal_model_spec(**overrides: Any) -> ModelSpec:
    """A tiny valid ModelSpec for direct unit calls (fake adapter, target y)."""
    payload: dict[str, Any] = {
        "name": "unit_model",
        "task": "binary_classification",
        "adapter": "fake",
        "owner": "unit@example.com",
        "dataset": "ref('unit_dataset')",
        "target": "y",
        "evaluation": {"protocol": {"split": "temporal"}, "metrics": ["pr_auc"]},
        "seed": 11,
    }
    payload.update(overrides)
    return ModelSpec.model_validate(payload)


def make_inline_runtime(
    tables: dict[str, Any],
    spec: ModelSpec,
    *,
    hooks: Any = None,
    time_column: str | None = None,
    job: Any = None,
    adapter: Any = None,
    builtin_specs: list[Any] | None = None,
    hook_specs: list[Any] | None = None,
) -> _JobRuntime:
    """A _JobRuntime over in-memory tables for direct job-function tests."""
    base = InMemoryDatasetHandle(tables, label_column=spec.target, time_column=time_column)
    transformed = TransformedDatasetHandle(
        base, spec, hooks, lambda split: SimpleNamespace(split=split), time_column
    )
    if job is None:
        job = SimpleNamespace(
            dataset_windows={},
            node=SimpleNamespace(unique_id="model.demo.unit_model"),
            tuning_engine=None,
            tuning_cap=None,
            vars={},
            metric_specs=builtin_specs or [],
        )
    return _JobRuntime(
        job=job,
        spec=spec,
        adapter=adapter,
        handle=transformed,
        transformed=transformed,
        base_handle=base,
        base_profile=base.profile(),
        hooks=hooks,
        builtin_specs=builtin_specs or [],
        hook_specs=hook_specs or [],
        ctx=None,
        store=None,
    )
