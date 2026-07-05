"""Command orchestration: run / build / test / evaluate (TSD §10).

This is what the CLI calls after flag parsing: parse -> compile (or read a
stored manifest verbatim, FR-RUN-11) -> plan -> schedule -> run_results.
"""

import time
import uuid
from dataclasses import dataclass, field
from datetime import UTC, datetime
from pathlib import Path
from typing import Any

import mbt
from mbt.adapters.registry import AdapterRegistry, get_registry
from mbt.artifacts.manifest import Manifest, read_manifest
from mbt.artifacts.run_results import (
    NodeResult,
    RunResults,
    RunResultsMetadata,
)
from mbt.compile.compiler import CompileOptions, compile_project
from mbt.config.profiles import LoadedProfiles, load_profiles
from mbt.contracts import ModelSpec, Stage
from mbt.dag.selector import StateIndex
from mbt.events import get_bus
from mbt.events.models import LogMessage, RunFinished, RunStarted
from mbt.exceptions import ConfigError, MbtError, StateError
from mbt.execute.planner import ExecutionPlan, plan_execution
from mbt.execute.runners import DatasetRunner, ExecutionContext, ModelRunner, ModelTestRunner
from mbt.execute.scheduler import execute_plan
from mbt.parsing import ParsedProject, parse_project
from mbt.state.diff import ManifestStateIndex, load_state


@dataclass(frozen=True)
class InvocationOptions:
    """Uniform cross-command flags (FR-CLI-04)."""

    command: str  # run | build | test | evaluate
    project_dir: Path
    profiles_dir: Path | None = None
    target: str | None = None
    cli_vars: dict[str, Any] = field(default_factory=dict)
    select: list[str] | None = None
    exclude: list[str] | None = None
    threads: int | None = None
    fail_fast: bool = False
    state: str | None = None
    state_include_env: bool = False
    manifest_path: str | None = None
    anchor: datetime | None = None
    deep_snapshot: bool = False


@dataclass
class PreparedInvocation:
    manifest: Manifest
    profiles: LoadedProfiles
    parsed: ParsedProject | None
    registry: AdapterRegistry
    run_id: str


def _new_run_id() -> str:
    return datetime.now(tz=UTC).strftime("%Y%m%dT%H%M%SZ") + "-" + uuid.uuid4().hex[:8]


def prepare(
    opts: InvocationOptions, *, registry: AdapterRegistry | None = None
) -> PreparedInvocation:
    """parse + profiles + manifest (fresh compile, or --manifest verbatim)."""
    registry = registry or get_registry()

    if opts.manifest_path is not None:
        manifest = read_manifest(Path(opts.manifest_path), source="--manifest")
        parsed = _try_parse(opts, registry)
        profiles = load_profiles(
            manifest.metadata.project_name,
            opts.project_dir,
            profiles_dir=opts.profiles_dir,
            target_override=opts.target or manifest.metadata.target,
            cli_vars=opts.cli_vars,
            project_vars=parsed.project.vars if parsed else {},
        )
        if parsed is not None:
            _warn_on_drift(parsed, profiles, manifest, registry, opts)
        return PreparedInvocation(
            manifest=manifest,
            profiles=profiles,
            parsed=parsed,
            registry=registry,
            run_id=_new_run_id(),
        )

    parsed = parse_project(opts.project_dir, registry=registry, cli_vars=opts.cli_vars)
    profiles = load_profiles(
        parsed.project.name,
        opts.project_dir,
        profiles_dir=opts.profiles_dir,
        target_override=opts.target,
        cli_vars=opts.cli_vars,
        project_vars=parsed.project.vars,
    )
    manifest = compile_project(
        parsed,
        profiles,
        registry=registry,
        options=CompileOptions(anchor=opts.anchor, deep_snapshot=opts.deep_snapshot),
        cli_vars=opts.cli_vars,
    )
    manifest.write(opts.project_dir / "target" / "manifest.json")
    return PreparedInvocation(
        manifest=manifest,
        profiles=profiles,
        parsed=parsed,
        registry=registry,
        run_id=_new_run_id(),
    )


def _try_parse(opts: InvocationOptions, registry: AdapterRegistry) -> ParsedProject | None:
    try:
        return parse_project(opts.project_dir, registry=registry, cli_vars=opts.cli_vars)
    except MbtError as exc:
        get_bus().emit(
            LogMessage(
                level="warn",
                message=(
                    "project files no longer parse cleanly; executing the stored "
                    f"manifest verbatim anyway (FR-RUN-11). Parse said: {exc.message}"
                ),
            )
        )
        return None


def _warn_on_drift(
    parsed: ParsedProject,
    profiles: LoadedProfiles,
    manifest: Manifest,
    registry: AdapterRegistry,
    opts: InvocationOptions,
) -> None:
    """--manifest execution warns when project files disagree (TSD §10.6)."""
    try:
        anchor = datetime.fromisoformat(manifest.metadata.anchor.replace("Z", "+00:00"))
        fresh = compile_project(
            parsed,
            profiles,
            registry=registry,
            options=CompileOptions(
                anchor=anchor, deep_snapshot=manifest.metadata.deep_snapshot
            ),
            cli_vars=opts.cli_vars,
        )
    except MbtError:
        get_bus().emit(
            LogMessage(
                level="warn",
                message="could not re-render project files to check manifest freshness",
            )
        )
        return
    drifted = [
        uid
        for uid, node in manifest.nodes.items()
        if uid in fresh.nodes and fresh.nodes[uid].config_hash != node.config_hash
    ]
    drifted += [uid for uid in fresh.nodes if uid not in manifest.nodes]
    if drifted:
        get_bus().emit(
            LogMessage(
                level="warn",
                message=(
                    "project files disagree with the stored manifest for: "
                    + ", ".join(sorted(drifted))
                    + " - executing the manifest verbatim (FR-RUN-11)"
                ),
            )
        )


def build_state_index(
    opts: InvocationOptions, manifest: Manifest
) -> StateIndex | None:
    if opts.state is None:
        return None
    reference = load_state(opts.state)
    index = ManifestStateIndex(
        manifest, reference, include_env=opts.state_include_env
    )
    if index.env_changed:
        get_bus().emit(
            LogMessage(
                level="warn",
                message=(
                    "environment digest differs from the reference manifest; nodes are "
                    "NOT marked modified by this alone (pass --state-include-env to opt in, ADR-7)"
                ),
            )
        )
    return index


def run_command(
    opts: InvocationOptions, *, registry: AdapterRegistry | None = None
) -> RunResults:
    """Execute run/build/test and write run_results.json (FR-RUN-04)."""
    started_monotonic = time.monotonic()
    started_at = datetime.now(tz=UTC).isoformat()
    prepared = prepare(opts, registry=registry)
    manifest = prepared.manifest
    bus = get_bus()
    bus.run_id = prepared.run_id

    state_index = build_state_index(opts, manifest)
    plan = plan_execution(manifest, opts.select, opts.exclude, state_index)
    bus.emit(
        RunStarted(
            command=opts.command,
            target=manifest.metadata.target,
            selected=len(plan.selected),
        )
    )
    if plan.auto_materialized:
        bus.emit(
            LogMessage(
                message=(
                    "auto-materializing required upstream dataset(s): "
                    + ", ".join(sorted(plan.auto_materialized))
                    + " (FR-RUN-12)"
                )
            )
        )

    results = _execute(opts, prepared, plan)

    ordered = [results[uid] for uid in plan.order if uid in results]
    run_results = RunResults(
        metadata=RunResultsMetadata(
            run_id=prepared.run_id,
            mbt_version=mbt.__version__,
            target=manifest.metadata.target,
            manifest_hash=manifest.manifest_hash(),
            anchor=manifest.metadata.anchor,
            started_at=started_at,
            elapsed_s=round(time.monotonic() - started_monotonic, 3),
            command=opts.command,
            selector=" ".join(opts.select) if opts.select else None,
        ),
        results=ordered,
    )
    run_results.write(opts.project_dir / "target" / "run_results.json")

    statuses = [r.status for r in ordered]
    bus.emit(
        RunFinished(
            command=opts.command,
            status={0: "success", 1: "error", 2: "quality_failure"}[run_results.exit_code()],
            succeeded=statuses.count("success"),
            failed=sum(statuses.count(s) for s in ("error", "gate_failed", "test_failed")),
            skipped=statuses.count("skipped"),
            elapsed_s=run_results.metadata.elapsed_s,
        )
    )
    return run_results


def _execute(
    opts: InvocationOptions,
    prepared: PreparedInvocation,
    plan: ExecutionPlan,
) -> dict[str, NodeResult]:
    manifest = prepared.manifest
    ctx = ExecutionContext(
        manifest=manifest,
        profiles=prepared.profiles,
        registry=prepared.registry,
        project_dir=opts.project_dir.resolve(),
        run_id=prepared.run_id,
        command=opts.command,
        cli_vars=opts.cli_vars,
        python_tests=prepared.parsed.python_tests if prepared.parsed else [],
        total_nodes=len(plan.execution_set),
    )
    dataset_runner = DatasetRunner(ctx)
    model_runner = ModelRunner(ctx) if opts.command in ("run", "build") else ModelTestRunner(ctx)

    def run_node(uid: str) -> NodeResult:
        node = manifest.nodes[uid]
        if node.resource_type == "dataset":
            return dataset_runner.run(uid)
        return model_runner.run(uid)

    threads = opts.threads if opts.threads is not None else prepared.profiles.target.threads
    return execute_plan(
        plan,
        manifest.graph(),
        run_node,
        threads=threads,
        fail_fast=opts.fail_fast,
    )


# -- mbt evaluate (FR-RUN-07, TSD §10.6) -------------------------------------------


def run_evaluate(
    opts: InvocationOptions,
    *,
    model_name: str,
    version: str | None = None,
    stage: str | None = None,
    apply_gates: bool = False,
    registry: AdapterRegistry | None = None,
) -> RunResults:
    """Re-evaluate a registered artifact on freshly built data, no retraining."""
    started_monotonic = time.monotonic()
    started_at = datetime.now(tz=UTC).isoformat()
    prepared = prepare(opts, registry=registry)
    manifest = prepared.manifest
    bus = get_bus()
    bus.run_id = prepared.run_id

    model_uid = next(
        (
            uid
            for uid, node in manifest.nodes.items()
            if node.resource_type == "model" and node.name == model_name
        ),
        None,
    )
    if model_uid is None:
        raise ConfigError(
            f"unknown model {model_name!r}",
            hint="run 'mbt ls --select resource_type:model' to list models",
        )
    node = manifest.nodes[model_uid]
    spec = ModelSpec.model_validate(node.config)

    ctx = ExecutionContext(
        manifest=manifest,
        profiles=prepared.profiles,
        registry=prepared.registry,
        project_dir=opts.project_dir.resolve(),
        run_id=prepared.run_id,
        command="evaluate",
        cli_vars=opts.cli_vars,
        python_tests=[],
        total_nodes=len(node.depends_on) + 1,
    )

    # Build (or reuse) the model's dataset first.
    dataset_runner = DatasetRunner(ctx)
    results: list[NodeResult] = []
    for dep in node.depends_on:
        if manifest.nodes.get(dep) is not None and manifest.nodes[dep].resource_type == "dataset":
            results.append(dataset_runner.run(dep))
    if any(r.status == "error" for r in results):
        pass  # fall through: the model result will carry the error context
    else:
        model_runner = ModelRunner(ctx)
        registry_name = spec.registration.name if spec.registration else spec.name
        resolved_version = None
        registry_adapter = ctx.registry_adapter()
        if version is not None:
            resolved_version = registry_adapter.get_version(registry_name, version)
        else:
            stage_token = Stage(stage) if stage else (
                spec.registration.stage_on_pass if spec.registration else Stage.STAGING
            )
            resolved_version = registry_adapter.get_champion(registry_name, stage_token)
        if resolved_version is None or resolved_version.artifact is None:
            raise StateError(
                f"no registered version of {registry_name!r} to evaluate",
                hint="pass --version N or --stage <stage>, or train the model first",
            )
        metric_specs = model_runner._metric_specs(spec, node)  # noqa: SLF001
        champion, _ = (
            model_runner._champion(spec, node) if apply_gates else (None, None)  # noqa: SLF001
        )
        job = model_runner._assemble_job(  # noqa: SLF001
            node, spec, metric_specs, champion, mode="evaluate",
            artifact=resolved_version.artifact,
        )
        job_result = ctx.compute.wait(ctx.compute.submit(job))
        if job_result.status == "error" or job_result.metrics is None:
            results.append(
                NodeResult(
                    unique_id=model_uid,
                    status="error",
                    message=job_result.error or "no metrics",
                )
            )
        else:
            gates = []
            status = "success"
            if apply_gates and spec.evaluation.gates:
                gates = model_runner._gate_results(  # noqa: SLF001
                    spec, node, job_result, champion, metric_specs
                )
                from mbt.quality.gates import all_gates_passed

                status = "success" if all_gates_passed(gates) else "gate_failed"
            results.append(
                NodeResult(
                    unique_id=model_uid,
                    status=status,  # type: ignore[arg-type]
                    metrics=dict(job_result.metrics.metrics),
                    slices=dict(job_result.metrics.slices),
                    gates=gates,
                )
            )

    run_results = RunResults(
        metadata=RunResultsMetadata(
            run_id=prepared.run_id,
            mbt_version=mbt.__version__,
            target=manifest.metadata.target,
            manifest_hash=manifest.manifest_hash(),
            anchor=manifest.metadata.anchor,
            started_at=started_at,
            elapsed_s=round(time.monotonic() - started_monotonic, 3),
            command="evaluate",
            selector=model_name,
        ),
        results=results,
    )
    run_results.write(opts.project_dir / "target" / "run_results.json")
    return run_results
