"""Node runners: datasets in-process, models via ComputeAdapter (TSD §10.4/§10.5)."""

import contextlib
import hashlib
import threading
import time
from collections.abc import Callable
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, cast

import pyarrow as pa

from mbt.adapters.registry import AdapterRegistry
from mbt.artifacts.manifest import Manifest
from mbt.artifacts.run_results import (
    GateResult,
    NodeResult,
    RegistrationResult,
    TestResultEntry,
)
from mbt.config.profiles import LoadedProfiles
from mbt.contracts import (
    AdapterRef,
    ArtifactRef,
    DatasetLocator,
    DatasetSpec,
    JobResult,
    ManifestNode,
    MetricSpec,
    ModelSpec,
    ModelVersion,
    ScoringSpec,
    SourceTable,
    Stage,
    TrainingJob,
)
from mbt.dag.selector import SelectableNode, evaluate_selector
from mbt.events import get_bus
from mbt.events.models import ArtifactRegistered, LogMessage, NodeFinished, NodeStarted
from mbt.exceptions import AdapterError, ConfigError, MbtError, StateError
from mbt.quality.checks import run_checks, run_scoring_checks
from mbt.quality.gates import all_gates_passed, evaluate_gates
from mbt.quality.metrics import resolve_model_metrics
from mbt.quality.monitors import all_monitors_passed, evaluate_monitors
from mbt.quality.python_tests import PythonTestFile, run_python_tests
from mbt.runtime import (
    data_adapter as build_data_adapter,
)
from mbt.runtime import (
    registry_adapter as build_registry_adapter,
)
from mbt.runtime import (
    resolve_artifact_store_uri,
)
from mbt.runtime import (
    tracking_adapter as build_tracking_adapter,
)
from mbt.secrets import Secret
from mbt.utils import canonical_json


def _gate_failure_summary(gates: list[GateResult]) -> str:
    """A specific, reviewer-facing summary of which gate(s) failed - feeds the
    node message shown in the results table, the JSON run_results, and the
    GitOps PR comment, consistent with the monitor path's ``gate breach: ...``.
    """
    parts: list[str] = []
    for gate in gates:
        if gate.passed:
            continue
        where = f" [{gate.slice}]" if gate.slice else ""
        if gate.kind == "champion" and gate.delta_lower is not None:
            parts.append(
                f"{gate.metric}{where}: challenger delta lower bound "
                f"{gate.delta_lower:.4f} < required {gate.min_delta}"
            )
        elif gate.actual is not None:
            parts.append(f"{gate.metric}{where}={gate.actual:.4f} failed threshold {gate.expected}")
        else:
            parts.append(f"{gate.metric}{where}")
    return "gate breach: " + "; ".join(parts) if parts else "one or more gates failed"


@dataclass
class ExecutionContext:
    """Everything runners need for one invocation."""

    manifest: Manifest
    profiles: LoadedProfiles
    registry: AdapterRegistry
    project_dir: Path
    run_id: str
    command: str  # run | build | test | evaluate
    cli_vars: dict[str, Any] = field(default_factory=dict)
    python_tests: list[PythonTestFile] = field(default_factory=list)
    total_nodes: int = 0
    _dataset_handles: dict[str, Any] = field(default_factory=dict)
    _counter: list[int] = field(default_factory=lambda: [0])
    _active_job_handles: list[Any] = field(default_factory=list)
    _job_handles_lock: Any = field(default_factory=threading.Lock)

    def __post_init__(self) -> None:
        self.data_adapter = build_data_adapter(self.profiles, self.project_dir, self.registry)
        self.compute = self.registry.component(
            "compute",
            self.profiles.target.compute.adapter,
            dict(self.profiles.target.compute.config),
        )
        self._graph = self.manifest.graph()
        self._selectable = self.manifest.selectable_nodes()
        # Warm backends that need one-time setup (e.g. MLflow sqlite
        # migrations) before parallel jobs hit them concurrently.
        tracking = self.tracking()
        if hasattr(tracking, "prepare"):
            tracking.prepare()

    @property
    def merged_vars(self) -> dict[str, Any]:
        merged: dict[str, Any] = {}
        merged.update(self.manifest.metadata.vars)
        merged.update(self.cli_vars)
        return merged

    def job_safe_vars(self) -> dict[str, Any]:
        """Vars minus tainted values: secrets never serialize into job files."""
        return {k: v for k, v in self.merged_vars.items() if not isinstance(v, Secret)}

    def tracking(self) -> Any:
        return build_tracking_adapter(self.profiles, self.project_dir, self.registry)

    def registry_adapter(self) -> Any:
        return build_registry_adapter(self.profiles, self.project_dir, self.registry)

    def dataset_handle(self, uid: str) -> Any:
        return self._dataset_handles[uid]

    def store_dataset_handle(self, uid: str, handle: Any) -> None:
        self._dataset_handles[uid] = handle

    def next_index(self) -> int:
        self._counter[0] += 1
        return self._counter[0]

    def run_job(self, job: TrainingJob) -> JobResult:
        """Submit + wait, tracking the handle so --fail-fast can reclaim it."""
        handle = self.compute.submit(job)
        with self._job_handles_lock:
            self._active_job_handles.append(handle)
        try:
            return cast(JobResult, self.compute.wait(handle))
        finally:
            with self._job_handles_lock:
                self._active_job_handles.remove(handle)

    def cancel_active_jobs(self) -> None:
        """Terminate in-flight job subprocesses (--fail-fast); best-effort."""
        if not hasattr(self.compute, "terminate"):
            return  # older/remote compute adapters without a kill seam
        with self._job_handles_lock:
            handles = list(self._active_job_handles)
        for handle in handles:
            with contextlib.suppress(Exception):  # the job may have just exited
                self.compute.terminate(handle, "cancelled by --fail-fast")

    def raw_adapter_ref(self, kind: str) -> AdapterRef:
        raw = self.manifest.metadata.target_config.get(kind)
        if isinstance(raw, dict) and "adapter" in raw:
            return AdapterRef(adapter=str(raw["adapter"]), config=dict(raw.get("config", {})))
        # Fallback: rendered ref (still redacted on serialization paths).
        rendered = getattr(self.profiles.target, kind)
        return AdapterRef(adapter=rendered.adapter, config=dict(rendered.config))

    def graph(self) -> Any:
        return self._graph

    def selectable(self) -> dict[str, SelectableNode]:
        return self._selectable


def materialization_key(node: ManifestNode, sample_fraction: float = 1.0) -> str:
    """sha256(input_hash + canonical resolved windows) - two anchors may slice
    the same snapshot differently (TSD §10.4).

    A non-default ``sample_fraction`` partitions its own key: sampling is
    pushed into the source query, so a sampled materialization holds
    different rows and must never satisfy a full build's cache probe (or
    vice versa). The default contributes nothing, keeping every existing
    fraction-1.0 key stable."""
    windows = node.resolved.get("windows", {})
    digest = hashlib.sha256()
    digest.update(node.input_hash.encode())
    digest.update(canonical_json(windows).encode())
    if sample_fraction != 1.0:
        digest.update(f"sample_fraction={sample_fraction!r}".encode())
    return digest.hexdigest()[:16]


@dataclass(frozen=True)
class BuildContext:
    """DataBuildContext implementation handed to DataAdapters."""

    node: ManifestNode
    source: SourceTable  # the spine (single source, or the label table)
    source_tables: dict[str, SourceTable]  # every source dep by unique_id
    resolved_windows: dict[str, tuple[str, str]]
    sample_fraction: float
    deep_snapshot: bool
    output_dir: Path
    events: Any


def run_with_lifecycle(
    ctx: "ExecutionContext",
    uid: str,
    resource_type: str,
    inner: Callable[[], NodeResult],
) -> NodeResult:
    """Wrap a node's work in the shared node-lifecycle event protocol.

    Emits NodeStarted/NodeFinished with a monotonic index, times the body, and
    turns an MbtError into an ``error`` NodeResult. Every runner (and the
    monitor) shares this so the event contract lives in exactly one place.
    """
    bus = get_bus()
    index = ctx.next_index()
    bus.emit(
        NodeStarted(unique_id=uid, resource_type=resource_type, index=index, total=ctx.total_nodes)
    )
    started = time.monotonic()
    try:
        result = inner()
    except MbtError as exc:
        result = NodeResult(unique_id=uid, status="error", message=str(exc))
    result.execution_time_s = time.monotonic() - started
    bus.emit(
        NodeFinished(
            unique_id=uid,
            resource_type=resource_type,
            status=result.status,
            execution_time_s=result.execution_time_s,
            index=index,
            total=ctx.total_nodes,
            message=result.message,
        )
    )
    return result


class DatasetRunner:
    """Materialize (or reuse) a dataset, then run its checks and data tests."""

    def __init__(self, ctx: ExecutionContext) -> None:
        self.ctx = ctx

    def run(self, uid: str) -> NodeResult:
        node = self.ctx.manifest.nodes[uid]
        return run_with_lifecycle(self.ctx, uid, "dataset", lambda: self._run_inner(uid, node))

    def _run_inner(self, uid: str, node: ManifestNode) -> NodeResult:
        spec = DatasetSpec.model_validate(node.config)
        handle = self._materialize(node, spec)
        self.ctx.store_dataset_handle(uid, handle)
        tests = self._run_quality(uid, node, spec, handle)
        failed = [t for t in tests if not t.passed]
        status = "test_failed" if failed else "success"
        message = (
            f"{len(failed)} check/test failure(s): "
            + "; ".join(f"{t.name}: {t.message}" for t in failed)
            if failed
            else None
        )
        return NodeResult(
            unique_id=uid,
            status=status,  # type: ignore[arg-type]
            tests=[TestResultEntry(**t.model_dump()) for t in tests],
            message=message,
        )

    def _materialize(self, node: ManifestNode, spec: DatasetSpec) -> Any:
        ctx = self.ctx
        sample_fraction = float(ctx.merged_vars.get("sample_fraction", 1.0))
        key = materialization_key(node, sample_fraction)
        output_dir = ctx.project_dir / "target" / "datasets" / node.name / key
        if (output_dir / "_SUCCESS").is_file():
            get_bus().emit(LogMessage(unique_id=node.unique_id, message=f"cache hit ({key})"))
            return ctx.data_adapter.from_locator(
                DatasetLocator(
                    adapter=ctx.profiles.target.data.adapter,
                    uri=f"file://{output_dir.resolve()}",
                    snapshot_id=node.snapshot_id or "",
                )
            )
        source_tables = {
            dep: ctx.manifest.sources[dep].config
            for dep in node.depends_on
            if dep.startswith("source.") and dep in ctx.manifest.sources
        }
        if not source_tables:
            raise ConfigError(
                f"dataset {node.name!r} has no source in the manifest",
                resource=node.unique_id,
            )
        # The spine: population or label table for multi-table inputs (ADR-22),
        # else the single source.
        spine_uid = spec.inputs.spine if spec.inputs is not None else spec.source
        if spine_uid not in source_tables:
            raise ConfigError(
                f"dataset {node.name!r} spine source {spine_uid!r} missing from the manifest",
                resource=node.unique_id,
                hint="recompile: the manifest and spec disagree",
            )
        windows = {
            split: (bounds[0], bounds[1])
            for split, bounds in node.resolved.get("windows", {}).items()
        }
        build_ctx = BuildContext(
            node=node,
            source=source_tables[spine_uid],
            source_tables=source_tables,
            resolved_windows=windows,
            sample_fraction=sample_fraction,
            deep_snapshot=ctx.manifest.metadata.deep_snapshot,
            output_dir=output_dir,
            events=get_bus(),
        )
        return ctx.data_adapter.build_dataset(spec, build_ctx)

    def _run_quality(
        self, uid: str, node: ManifestNode, spec: DatasetSpec, handle: Any
    ) -> list[Any]:
        results = list(run_checks(spec, handle, node.resolved, resource=uid))
        if self.ctx.command in ("build", "test"):
            results.extend(self._run_python_tests(uid, spec, handle))
        return results

    def _run_python_tests(self, uid: str, spec: DatasetSpec, handle: Any) -> list[Any]:
        bound = [tf for tf in self.ctx.python_tests if self._binds(tf, uid, spec)]
        if not bound:
            return []
        table = pa.concat_tables([handle.read(split) for split in sorted(handle.splits())])
        results = []
        for test_file in bound:
            only = set(spec.tests) & set(test_file.test_names) if spec.tests else None
            results.extend(run_python_tests(test_file, table, spec, only=only))
        return results

    def _binds(self, test_file: PythonTestFile, uid: str, spec: DatasetSpec) -> bool:
        if spec.tests and set(spec.tests) & set(test_file.test_names):
            return True
        if test_file.selector is None:
            return not spec.tests
        matched = evaluate_selector(test_file.selector, self.ctx.graph(), self.ctx.selectable())
        return uid in matched


class ModelRunner:
    """Coordinator side of a model build (TSD §10.5): jobs compute, core compares."""

    def __init__(self, ctx: ExecutionContext) -> None:
        self.ctx = ctx

    def run(self, uid: str) -> NodeResult:
        node = self.ctx.manifest.nodes[uid]
        return run_with_lifecycle(self.ctx, uid, "model", lambda: self._run_inner(uid, node))

    # -- helpers -------------------------------------------------------------

    def _metric_specs(self, spec: ModelSpec, node: ManifestNode) -> list[MetricSpec]:
        from mbt.config.tasks import get_task_schema

        declared = {
            name: MetricSpec.model_validate(payload)
            for name, payload in self.ctx.manifest.metrics.items()
        }
        resolved, errors = resolve_model_metrics(
            spec,
            declared,
            get_task_schema(spec.task),
            has_hooks=node.hooks_path is not None,
        )
        if errors:
            raise ConfigError("; ".join(errors), resource=node.unique_id)
        return resolved

    def _champion(
        self, spec: ModelSpec, node: ManifestNode
    ) -> tuple[ModelVersion | None, Stage | None]:
        stages = {g.compare_to for g in spec.evaluation.gates if g.compare_to is not None}
        if not stages:
            return None, None
        stage = next(iter(stages))
        registry_name = spec.registration.name if spec.registration else spec.name
        champion = self.ctx.registry_adapter().get_champion(registry_name, stage)
        if champion is not None and champion.artifact is None:
            raise AdapterError(
                f"champion {registry_name} v{champion.version} in {stage.value!r} has no "
                "loadable artifact reference",
                resource=node.unique_id,
                hint="a champion that exists but cannot load is an error (ADR-10)",
            )
        return champion, stage

    def _assemble_job(
        self,
        node: ManifestNode,
        spec: ModelSpec,
        metric_specs: list[MetricSpec],
        champion: ModelVersion | None,
        *,
        mode: str = "train",
        artifact: ArtifactRef | None = None,
    ) -> TrainingJob:
        ctx = self.ctx
        dataset_uid = next(d for d in node.depends_on if d.startswith("dataset."))
        dataset_node = ctx.manifest.nodes[dataset_uid]
        handle = ctx.dataset_handle(dataset_uid)
        tuning_cap = ctx.merged_vars.get("max_tuning_trials")
        meta = ctx.manifest.metadata
        return TrainingJob(
            mode=mode,  # type: ignore[arg-type]
            run_id=ctx.run_id,
            project_dir=str(ctx.project_dir),
            target_name=meta.target,
            node=node,
            dataset=handle.locator(),
            dataset_windows=dict(dataset_node.resolved),
            data=ctx.raw_adapter_ref("data"),
            tracking=ctx.raw_adapter_ref("tracking"),
            metric_specs=metric_specs,
            champion=champion.artifact if champion else None,
            artifact=artifact,
            tuning_engine=(
                AdapterRef(adapter=spec.tuning.engine) if spec.tuning is not None else None
            ),
            tuning_cap=int(tuning_cap) if tuning_cap is not None else None,
            artifact_store=resolve_artifact_store_uri(
                ctx.profiles.target.artifact_store, ctx.project_dir
            ),
            required_env=list(ctx.profiles.required_env),
            tracking_meta={
                "mbt.run_id": ctx.run_id,
                "mbt.config_hash": node.config_hash,
                "mbt.input_hash": node.input_hash,
                "mbt.manifest_hash": ctx.manifest.manifest_hash(),
                "mbt.snapshot_id": dataset_node.snapshot_id or "",
                "mbt.git_commit": meta.git.commit or "",
            },
            vars=ctx.job_safe_vars(),
        )

    def _gate_results(
        self,
        spec: ModelSpec,
        node: ManifestNode,
        job_result: Any,
        champion: ModelVersion | None,
        metric_specs: list[MetricSpec],
    ) -> list[GateResult]:
        adapter = self.ctx.registry.training(spec.adapter)
        return evaluate_gates(
            spec.evaluation.gates,
            resource=node.unique_id,
            challenger=job_result.metrics,
            champion=job_result.champion_metrics,
            champion_version=champion.version if champion else None,
            metric_specs=metric_specs,
            determinism=adapter.determinism,
            champion_delta_bounds=job_result.champion_delta_bounds,
        )

    def _register(
        self,
        spec: ModelSpec,
        node: ManifestNode,
        job_result: Any,
        gates: list[GateResult],
    ) -> RegistrationResult | None:
        if spec.registration is None or job_result.artifact is None:
            return None
        ctx = self.ctx
        registry_adapter = ctx.registry_adapter()
        dataset_uid = next(d for d in node.depends_on if d.startswith("dataset."))
        metadata = {
            "mbt.config_hash": node.config_hash,
            "mbt.input_hash": node.input_hash,
            "mbt.manifest_hash": ctx.manifest.manifest_hash(),
            "mbt.snapshot_id": ctx.manifest.nodes[dataset_uid].snapshot_id or "",
            "mbt.git_commit": ctx.manifest.metadata.git.commit or "",
            "mbt.tracking_run_id": job_result.tracking_run_id or "",
            "mbt.gates_passed": "true",
            "mbt.artifact_uri": job_result.artifact.uri,
            "mbt.artifact_format": job_result.artifact.format,
            "mbt.artifact_content_hash": job_result.artifact.content_hash,
            "mbt.artifact_size_bytes": str(job_result.artifact.size_bytes),
            # Scoring-time feature-transform parity check (ADR-20).
            "mbt.hooks_hash": node.hooks_hash or "",
        }
        if job_result.baseline is not None:
            # The monitoring baseline scoring runs compare against (ADR-21).
            metadata["mbt.baseline_uri"] = job_result.baseline.uri
            metadata["mbt.baseline_format"] = job_result.baseline.format
            metadata["mbt.baseline_content_hash"] = job_result.baseline.content_hash
            metadata["mbt.baseline_size_bytes"] = str(job_result.baseline.size_bytes)
        version = registry_adapter.register(job_result.artifact, spec.registration.name, metadata)
        registry_adapter.transition(version, spec.registration.stage_on_pass)
        get_bus().emit(
            ArtifactRegistered(
                unique_id=node.unique_id,
                registry=ctx.profiles.target.registry.adapter,
                name=spec.registration.name,
                version=version.version,
                stage=spec.registration.stage_on_pass.value,
            )
        )
        return RegistrationResult(
            registry=ctx.profiles.target.registry.adapter,
            name=spec.registration.name,
            version=version.version,
            stage=spec.registration.stage_on_pass.value,
        )

    def _attach_tracking_tags(
        self,
        job_result: Any,
        gates: list[GateResult],
        registration: RegistrationResult | None,
    ) -> None:
        if job_result.tracking_run_id is None:
            return
        tags = {
            "mbt.gates_passed": str(all_gates_passed(gates)).lower(),
            "mbt.gates": canonical_json([g.model_dump(mode="json") for g in gates]),
        }
        if registration is not None:
            tags["mbt.registered_version"] = registration.version
            tags["mbt.registered_stage"] = registration.stage
        try:
            tracking = self.ctx.tracking()
            run = tracking.resume(job_result.tracking_run_id)
            tracking.log(run, tags=tags)
        except Exception as exc:
            get_bus().emit(
                LogMessage(level="warn", message=f"could not attach tracking tags: {exc}")
            )

    # -- main path -------------------------------------------------------------

    def _run_inner(self, uid: str, node: ManifestNode) -> NodeResult:
        spec = ModelSpec.model_validate(node.config)
        metric_specs = self._metric_specs(spec, node)
        champion, _stage = self._champion(spec, node)

        job = self._assemble_job(node, spec, metric_specs, champion)
        job_result = self.ctx.run_job(job)

        if job_result.status == "error" or job_result.metrics is None:
            return NodeResult(
                unique_id=uid,
                status="error",
                message=job_result.error or "job returned no metrics",
            )

        gates = self._gate_results(spec, node, job_result, champion, metric_specs)
        passed = all_gates_passed(gates)
        registration = None
        if passed:
            registration = self._register(spec, node, job_result, gates)
        self._attach_tracking_tags(job_result, gates, registration)

        return NodeResult(
            unique_id=uid,
            status="success" if passed else "gate_failed",
            metrics=dict(job_result.metrics.metrics),
            slices=dict(job_result.metrics.slices),
            gates=gates,
            artifact=job_result.artifact,
            registration=registration,
            tracking_run_id=job_result.tracking_run_id,
            resolved_auto=dict(job_result.resolved_auto),
            feature_importance=dict(job_result.feature_importance),
            message=None if passed else _gate_failure_summary(gates),
        )

    # -- re-evaluation of registered artifacts (FR-RUN-07, TSD §11.3) -----------

    def evaluate_artifact(
        self,
        node: ManifestNode,
        spec: ModelSpec,
        artifact: ArtifactRef,
        *,
        apply_gates: bool = True,
    ) -> tuple[JobResult, list[GateResult]]:
        """Run an evaluate-mode job for a registered artifact; never trains.

        The one flow behind both ``mbt test`` on models and ``mbt evaluate``,
        so the two commands cannot drift: fresh data, the stored artifact,
        gate logic against the current champion when requested.
        """
        metric_specs = self._metric_specs(spec, node)
        champion, _stage = self._champion(spec, node) if apply_gates else (None, None)
        job = self._assemble_job(
            node, spec, metric_specs, champion, mode="evaluate", artifact=artifact
        )
        job_result = self.ctx.run_job(job)
        gates: list[GateResult] = []
        if (
            apply_gates
            and job_result.status != "error"
            and job_result.metrics is not None
            and spec.evaluation.gates
        ):
            gates = self._gate_results(spec, node, job_result, champion, metric_specs)
        return job_result, gates


def scoring_run_key(node: ManifestNode, model_version: str) -> str:
    """Prediction-run idempotency key (ADR-21): same manifest + same champion
    re-scores overwrite cleanly; new data, window, or champion partitions
    fresh. The scoring analog of ``materialization_key``."""
    digest = hashlib.sha256()
    digest.update(node.input_hash.encode())
    digest.update(canonical_json(node.resolved.get("windows", {})).encode())
    digest.update(model_version.encode())
    return digest.hexdigest()[:16]


class ScoringRunner:
    """Coordinator side of a scoring run (ADR-20): jobs compute, core compares.

    The champion is resolved from the registry at RUN time by stage alias -
    promotions are registry state, deliberately outside node identity
    (ADR-5), so scheduled scoring picks up a new champion on its next run
    without a spec edit. The resolved version lands in run_results, the
    tracking tags, and the prediction sidecar.
    """

    def __init__(self, ctx: ExecutionContext) -> None:
        self.ctx = ctx

    def run(self, uid: str) -> NodeResult:
        node = self.ctx.manifest.nodes[uid]
        return run_with_lifecycle(self.ctx, uid, "scoring", lambda: self._run_inner(uid, node))

    # -- helpers -------------------------------------------------------------

    def _champion(self, spec: ScoringSpec, model_spec: ModelSpec, uid: str) -> ModelVersion:
        registry_name = model_spec.registration.name if model_spec.registration else model_spec.name
        champion: ModelVersion | None = self.ctx.registry_adapter().get_champion(
            registry_name, spec.stage
        )
        if champion is None:
            # Unlike a missing gate comparator (ADR-10 WARN), nothing to
            # score WITH is an operational failure.
            raise StateError(
                f"no champion of {registry_name!r} in stage {spec.stage.value!r} to score with",
                resource=uid,
                hint="train and promote the model first (mbt build, then mbt promote)",
            )
        if champion.artifact is None:
            raise AdapterError(
                f"champion {registry_name} v{champion.version} in {spec.stage.value!r} has no "
                "loadable artifact reference",
                resource=uid,
                hint="a champion that exists but cannot load is an error (ADR-10)",
            )
        return champion

    def _check_hooks_parity(
        self, champion: ModelVersion, model_node: ManifestNode, uid: str
    ) -> None:
        """The champion must have been trained with the hooks the scoring run
        will apply; silent feature skew is worse than a hard stop (ADR-20)."""
        registered = champion.tags.get("mbt.hooks_hash")
        if registered is None:
            get_bus().emit(
                LogMessage(
                    level="warn",
                    unique_id=uid,
                    message=(
                        "champion predates hooks-parity registration (no "
                        "mbt.hooks_hash tag); cannot verify that scoring applies "
                        "the hooks the champion was trained with"
                    ),
                )
            )
            return
        if registered != (model_node.hooks_hash or ""):
            raise StateError(
                "the champion was trained with a different hooks.py than the "
                "current project's (mbt.hooks_hash mismatch)",
                resource=uid,
                hint="retrain and promote, or check out the commit the champion was built from",
            )

    def _baseline_ref(self, champion: ModelVersion) -> ArtifactRef | None:
        uri = champion.tags.get("mbt.baseline_uri")
        if not uri:
            return None
        return ArtifactRef(
            uri=uri,
            format=champion.tags.get("mbt.baseline_format", "json"),
            content_hash=champion.tags.get("mbt.baseline_content_hash", ""),
            size_bytes=int(champion.tags.get("mbt.baseline_size_bytes", "0")),
        )

    def _materialize_input(self, node: ManifestNode, spec: ScoringSpec) -> Any:
        ctx = self.ctx
        sample_fraction = float(ctx.merged_vars.get("sample_fraction", 1.0))
        key = materialization_key(node, sample_fraction)
        output_dir = ctx.project_dir / "target" / "scoring_inputs" / node.name / key
        if (output_dir / "_SUCCESS").is_file():
            get_bus().emit(LogMessage(unique_id=node.unique_id, message=f"cache hit ({key})"))
            return ctx.data_adapter.from_locator(
                DatasetLocator(
                    adapter=ctx.profiles.target.data.adapter,
                    uri=f"file://{output_dir.resolve()}",
                    snapshot_id=node.snapshot_id or "",
                )
            )
        if spec.input.source is not None:
            spine_uid = spec.input.source
            input_uids = [spine_uid]
        else:
            assert spec.input.inputs is not None
            spine_uid = spec.input.inputs.spine
            input_uids = [spine_uid, *spec.input.inputs.feature_sources]
        source_tables = {
            uid: ctx.manifest.sources[uid].config
            for uid in input_uids
            if uid in ctx.manifest.sources
        }
        if set(source_tables) != set(input_uids):
            missing = sorted(set(input_uids) - set(source_tables))
            raise ConfigError(
                f"scoring input source(s) missing from the manifest: {', '.join(missing)}",
                resource=node.unique_id,
                hint="recompile: the manifest and spec disagree",
            )
        windows = {
            split: (bounds[0], bounds[1])
            for split, bounds in node.resolved.get("windows", {}).items()
        }
        build_ctx = BuildContext(
            node=node,
            source=source_tables[spine_uid],
            source_tables=source_tables,
            resolved_windows=windows,
            sample_fraction=sample_fraction,
            deep_snapshot=ctx.manifest.metadata.deep_snapshot,
            output_dir=output_dir,
            events=get_bus(),
        )
        return ctx.data_adapter.build_scoring_input(spec.input, build_ctx)

    def _assemble_job(
        self,
        node: ManifestNode,
        model_node: ManifestNode,
        spec: ScoringSpec,
        champion: ModelVersion,
        baseline: ArtifactRef | None,
        handle: Any,
    ) -> TrainingJob:
        ctx = self.ctx
        meta = ctx.manifest.metadata
        return TrainingJob(
            mode="score",
            run_id=ctx.run_id,
            project_dir=str(ctx.project_dir),
            target_name=meta.target,
            node=node,
            model_node=model_node,
            dataset=handle.locator(),
            data=ctx.raw_adapter_ref("data"),
            tracking=ctx.raw_adapter_ref("tracking"),
            artifact=champion.artifact,
            baseline=baseline,
            output=spec.output,
            model_version=champion.version,
            run_key=scoring_run_key(node, champion.version),
            artifact_store=resolve_artifact_store_uri(
                ctx.profiles.target.artifact_store, ctx.project_dir
            ),
            required_env=list(ctx.profiles.required_env),
            tracking_meta={
                "mbt.run_id": ctx.run_id,
                "mbt.config_hash": node.config_hash,
                "mbt.input_hash": node.input_hash,
                "mbt.manifest_hash": ctx.manifest.manifest_hash(),
                "mbt.snapshot_id": node.snapshot_id or "",
                "mbt.git_commit": meta.git.commit or "",
                "mbt.anchor": meta.anchor,
                "mbt.model_version": champion.version,
            },
            vars=ctx.job_safe_vars(),
        )

    # -- main path -------------------------------------------------------------

    def _run_inner(self, uid: str, node: ManifestNode) -> NodeResult:
        ctx = self.ctx
        spec = ScoringSpec.model_validate(node.config)
        model_uid = next(d for d in node.depends_on if d.startswith("model."))
        model_node = ctx.manifest.nodes.get(model_uid)
        if model_node is None:
            raise ConfigError(
                f"scoring model {model_uid!r} missing from the manifest",
                resource=uid,
                hint="recompile: the manifest and spec disagree",
            )
        model_spec = ModelSpec.model_validate(model_node.config)

        champion = self._champion(spec, model_spec, uid)
        self._check_hooks_parity(champion, model_node, uid)
        baseline = self._baseline_ref(champion)

        handle = self._materialize_input(node, spec)

        checks = run_scoring_checks(spec, handle, node.resolved, resource=uid)
        failed = [t for t in checks if not t.passed]
        if failed:
            # Never score on bad input: the job is skipped entirely.
            return NodeResult(
                unique_id=uid,
                status="test_failed",
                tests=[TestResultEntry(**t.model_dump()) for t in checks],
                message=(
                    f"{len(failed)} input check failure(s), scoring skipped: "
                    + "; ".join(f"{t.name}: {t.message}" for t in failed)
                ),
            )

        job = self._assemble_job(node, model_node, spec, champion, baseline, handle)
        job_result = ctx.run_job(job)
        if job_result.status == "error":
            return NodeResult(
                unique_id=uid, status="error", message=job_result.error or "scoring job failed"
            )

        monitors = evaluate_monitors(spec.monitors, job_result.monitor_stats, resource=uid)
        passed = all_monitors_passed(monitors)
        rows = float(job_result.predictions.row_count) if job_result.predictions else 0.0
        failures = [m.message for m in monitors if not m.passed and m.message]
        return NodeResult(
            unique_id=uid,
            status="success" if passed else "monitor_failed",
            metrics={"rows_scored": rows},
            monitors=monitors,
            tests=[TestResultEntry(**t.model_dump()) for t in checks],
            tracking_run_id=job_result.tracking_run_id,
            message=None if passed else "monitor breach: " + "; ".join(failures),
        )


class ModelTestRunner:
    """``mbt test`` for models: re-evaluate the latest registered version;
    training is never a side effect (TSD §11.3)."""

    def __init__(self, ctx: ExecutionContext) -> None:
        self.ctx = ctx
        self._model_runner = ModelRunner(ctx)

    def run(self, uid: str) -> NodeResult:
        node = self.ctx.manifest.nodes[uid]
        return run_with_lifecycle(self.ctx, uid, "model", lambda: self._run_inner(uid, node))

    def _run_inner(self, uid: str, node: ManifestNode) -> NodeResult:
        ctx = self.ctx
        spec = ModelSpec.model_validate(node.config)
        if not spec.evaluation.gates:
            return NodeResult(unique_id=uid, status="skipped", message="model declares no gates")
        registry_name = spec.registration.name if spec.registration else spec.name
        stage = spec.registration.stage_on_pass if spec.registration else Stage.STAGING
        version = ctx.registry_adapter().get_champion(registry_name, stage)
        if version is None or version.artifact is None:
            get_bus().emit(
                LogMessage(
                    level="warn",
                    unique_id=uid,
                    message=(
                        f"no registered version of {registry_name!r} in {stage.value!r}; "
                        "skipping model tests (mbt test never trains)"
                    ),
                )
            )
            return NodeResult(unique_id=uid, status="skipped", message="no registered version")

        job_result, gates = self._model_runner.evaluate_artifact(node, spec, version.artifact)
        if job_result.status == "error" or job_result.metrics is None:
            return NodeResult(
                unique_id=uid, status="error", message=job_result.error or "no metrics"
            )
        passed = all_gates_passed(gates)
        return NodeResult(
            unique_id=uid,
            status="success" if passed else "test_failed",
            metrics=dict(job_result.metrics.metrics),
            slices=dict(job_result.metrics.slices),
            gates=gates,
            feature_importance=dict(job_result.feature_importance),
            message=None if passed else _gate_failure_summary(gates),
        )
