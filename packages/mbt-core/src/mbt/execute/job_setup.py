"""Job setup: the job's own environment, and the ``JobRuntime`` built from it (TSD §10.3, §18).

Secrets never travel in the serialized job, so every adapter config is rendered
again here from this process's environment, then the runtime (adapter, hooks,
store, base dataset handle) is assembled once for whichever step runs next.
"""

import os
from pathlib import Path
from typing import Any

from mbt.adapters.registry import get_registry
from mbt.events import get_bus
from mbt.events.bus import HookEventSink
from mbt.events.models import LogMessage
from mbt.exceptions import AdapterError, ConfigError
from mbt.execute.handles import TrainingSplitView, TransformedDatasetHandle
from mbt.execute.job_runtime import JobRuntime
from mbt.execute.seeds import (
    train_seed,
)
from mbt.jinja.sandbox import sandboxed_environment
from mbt.quality.hooks import ModelHooks, load_hooks
from mbt.runtime import normalized_adapter_config, tracking_adapter_config
from mbt.secrets import taint
from mbt.storage import artifact_store_for
from mbt_adapter_base import (
    AdapterRef,
    HookContext,
    ModelSpec,
    RunContext,
    TrainingJob,
)


def _render_adapter_ref(ref: AdapterRef, job_vars: dict[str, Any]) -> AdapterRef:
    """Re-render env_var()/env()/var() in an unrendered adapter config (TSD §18)."""
    jinja_env = sandboxed_environment()

    def lookup_env(name: str, default: str | None, *, secret: bool) -> str:
        value = os.environ.get(name)
        if value is None:
            if default is None:
                raise ConfigError(
                    f"environment variable {name!r} required by the job is not set",
                    hint="the job re-resolves secrets from its own environment (TSD §18)",
                )
            return default
        return taint(value) if secret else value

    def env_var(name: str, default: str | None = None) -> str:
        return lookup_env(name, default, secret=True)

    def env(name: str, default: str | None = None) -> str:
        """Non-secret environment value: same lookup, no taint."""
        return lookup_env(name, default, secret=False)

    def var(name: str, default: Any = None) -> Any:
        return job_vars.get(name, default)

    def render(value: Any) -> Any:
        if isinstance(value, str) and ("{{" in value or "{%" in value):
            return jinja_env.from_string(value).render(env_var=env_var, env=env, var=var)
        if isinstance(value, dict):
            return {k: render(v) for k, v in value.items()}
        if isinstance(value, list):
            return [render(v) for v in value]
        return value

    return AdapterRef(adapter=ref.adapter, config=render(ref.config))


def _tracking_adapter(job: TrainingJob) -> Any:
    assert job.tracking is not None
    rendered = _render_adapter_ref(job.tracking, _job_vars(job))
    config = tracking_adapter_config(rendered, Path(job.project_dir), job.project)
    return get_registry().component("tracking", rendered.adapter, config)


def _job_vars(job: TrainingJob) -> dict[str, Any]:
    return dict(job.vars)


def _materialize_for_path_adapter(handle: Any, spec: ModelSpec) -> Any:
    """Write a handle's transformed splits to parquet for adapters that
    declare ``data_access == "path"`` (JVM/cluster frameworks ingest files
    natively). Hooks and feature selection are already applied by the
    transformed handle, so path adapters see exactly what arrow adapters see.
    """
    import atexit
    import shutil
    import tempfile

    import pyarrow.parquet as pq

    from mbt_adapter_base.materialization import (
        MaterializedDatasetHandle,
        write_materialization_metadata,
    )

    directory = Path(tempfile.mkdtemp(prefix="mbt-path-data-"))
    # The handle reads these files for the rest of the job; the job process
    # is short-lived, so process exit is the natural cleanup point.
    atexit.register(shutil.rmtree, directory, ignore_errors=True)
    counts: dict[str, int] = {}
    for split in sorted(handle.splits()):
        table = handle.read(split)
        pq.write_table(table, directory / f"{split}.parquet")
        counts[split] = table.num_rows
    write_materialization_metadata(
        directory,
        snapshot_id=handle.snapshot_id,
        dataset=spec.name,
        label_column=spec.target,
        time_column=None,  # split time columns never reach adapters
        windows={},
        sample_fraction=1.0,
        row_counts=counts,
    )
    return MaterializedDatasetHandle(directory)


def _prepare(job: TrainingJob) -> JobRuntime:
    registry = get_registry()
    checking = job.mode == "oot_check"
    # The pre-deploy check runs the version's own spec, like scoring (ADR-28).
    spec = ModelSpec.model_validate((job.champion_spec if checking else None) or job.node.config)

    data_ref = _render_adapter_ref(job.data, _job_vars(job))
    data_adapter = registry.component(
        "data", data_ref.adapter, normalized_adapter_config(data_ref, Path(job.project_dir))
    )
    materialization = data_adapter.from_locator(job.dataset)
    test_window = job.node.resolved.get("test_window")
    base_handle = TrainingSplitView(
        materialization,
        time_column=getattr(materialization, "time_column", None),
        test_window=(str(test_window[0]), str(test_window[1])) if test_window else None,
    )
    base_profile = base_handle.profile()

    plugin = registry.get(spec.adapter)
    if plugin.training is None:
        raise AdapterError(f"adapter {spec.adapter!r} provides no training adapter")
    adapter = plugin.training({})

    # Run-time task validation now that the dataset profile exists (TSD §5.6).
    from mbt.config.tasks import get_task_schema

    # The check's materialization holds no train split, so there is no label
    # balance to validate; the version was validated when it trained.
    issues = [] if checking else get_task_schema(spec.task).validate_dataset(spec, base_profile)
    errors = [i for i in issues if i.severity == "error"]
    for issue in issues:
        if issue.severity == "warning":
            get_bus().emit(
                LogMessage(level="warn", unique_id=job.node.unique_id, message=issue.message)
            )
    if errors:
        raise ConfigError(
            "; ".join(i.message for i in errors),
            resource=job.node.unique_id,
            hint=errors[0].hint,
        )

    hooks: ModelHooks | None = None
    if job.node.hooks_path is not None:
        hooks = load_hooks(Path(job.project_dir), job.node.hooks_path)

    ctx = RunContext(
        run_id=job.run_id,
        unique_id=job.node.unique_id,
        seed=train_seed(spec.seed),
        target_name=job.target_name,
        project_dir=job.project_dir,
        vars=_job_vars(job),
        events=get_bus(),
    )

    def hook_ctx(split: str) -> HookContext:
        return HookContext(
            spec=spec, profile=base_profile, split=split, logger=HookEventSink(get_bus())
        )

    time_column = getattr(base_handle, "time_column", None)
    transformed = TransformedDatasetHandle(
        base_handle,
        spec,
        hooks,
        hook_ctx,
        time_column,
        pinned_features=job.champion_feature_columns if checking else None,
    )
    handle: Any = transformed
    if getattr(adapter, "data_access", "arrow") == "path":
        handle = _materialize_for_path_adapter(transformed, spec)

    # The champion gate re-scores the champion on the challenger's test rows
    # (ADR-9), but through the CHAMPION's own spec and the exact columns it was
    # fit on (ADR-28), as scoring does. Through the challenger's handle, a
    # champion whose features the challenger dropped was scored with those
    # columns missing - H2O reads them as all-NA - so a worse challenger beat
    # a crippled champion and was staged.
    champion_handle: Any = handle
    if job.champion is not None and job.champion_spec is not None and not checking:
        champion_spec = ModelSpec.model_validate(job.champion_spec)

        def champion_hook_ctx(split: str) -> HookContext:
            return HookContext(
                spec=champion_spec,
                profile=base_profile,
                split=split,
                logger=HookEventSink(get_bus()),
            )

        champion_view = TransformedDatasetHandle(
            base_handle,
            champion_spec,
            hooks,
            champion_hook_ctx,
            time_column,
            pinned_features=job.champion_feature_columns,
        )
        champion_handle = champion_view
        if getattr(adapter, "data_access", "arrow") == "path":
            champion_handle = _materialize_for_path_adapter(champion_view, champion_spec)

    return JobRuntime(
        job=job,
        spec=spec,
        adapter=adapter,
        handle=handle,
        transformed=transformed,
        base_handle=base_handle,
        materialization=materialization,
        base_profile=base_profile,
        hooks=hooks,
        builtin_specs=[m for m in job.metric_specs if m.kind == "builtin"],
        hook_specs=[m for m in job.metric_specs if m.kind == "hook"],
        ctx=ctx,
        store=artifact_store_for(job.artifact_store, run_prefix=_store_prefix(job)),
        champion_handle=champion_handle,
    )


def _store_prefix(job: TrainingJob) -> str:
    return f"{job.node.name}/{job.run_id}"
