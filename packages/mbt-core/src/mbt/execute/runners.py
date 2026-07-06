"""Node runners: datasets in-process, models via ComputeAdapter (TSD §10.4/§10.5)."""

import hashlib
import time
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any

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
    ManifestNode,
    MetricSpec,
    ModelSpec,
    ModelVersion,
    SourceTable,
    Stage,
    TrainingJob,
)
from mbt.dag.selector import SelectableNode, evaluate_selector
from mbt.events import get_bus
from mbt.events.models import ArtifactRegistered, LogMessage, NodeFinished, NodeStarted
from mbt.exceptions import AdapterError, ConfigError, MbtError
from mbt.quality.checks import run_checks
from mbt.quality.gates import all_gates_passed, evaluate_gates
from mbt.quality.metrics import resolve_model_metrics
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


def materialization_key(node: ManifestNode) -> str:
    """sha256(input_hash + canonical resolved windows) - two anchors may slice
    the same snapshot differently (TSD §10.4)."""
    windows = node.resolved.get("windows", {})
    digest = hashlib.sha256()
    digest.update(node.input_hash.encode())
    digest.update(canonical_json(windows).encode())
    return digest.hexdigest()[:16]


@dataclass(frozen=True)
class _BuildContext:
    """DataBuildContext implementation handed to DataAdapters."""

    node: ManifestNode
    source: SourceTable  # the spine (single source, or the label table)
    source_tables: dict[str, SourceTable]  # every source dep by unique_id
    resolved_windows: dict[str, tuple[str, str]]
    sample_fraction: float
    deep_snapshot: bool
    output_dir: Path
    events: Any


class DatasetRunner:
    """Materialize (or reuse) a dataset, then run its checks and data tests."""

    def __init__(self, ctx: ExecutionContext) -> None:
        self.ctx = ctx

    def run(self, uid: str) -> NodeResult:
        ctx = self.ctx
        node = ctx.manifest.nodes[uid]
        bus = get_bus()
        index = ctx.next_index()
        bus.emit(
            NodeStarted(unique_id=uid, resource_type="dataset", index=index, total=ctx.total_nodes)
        )
        started = time.monotonic()
        try:
            spec = DatasetSpec.model_validate(node.config)
            handle = self._materialize(node, spec)
            ctx.store_dataset_handle(uid, handle)
            tests = self._run_quality(uid, node, spec, handle)
            failed = [t for t in tests if not t.passed]
            status = "test_failed" if failed else "success"
            message = (
                f"{len(failed)} check/test failure(s): "
                + "; ".join(f"{t.name}: {t.message}" for t in failed)
                if failed
                else None
            )
            result = NodeResult(
                unique_id=uid,
                status=status,  # type: ignore[arg-type]
                execution_time_s=time.monotonic() - started,
                tests=[TestResultEntry(**t.model_dump()) for t in tests],
                message=message,
            )
        except MbtError as exc:
            result = NodeResult(
                unique_id=uid,
                status="error",
                execution_time_s=time.monotonic() - started,
                message=str(exc),
            )
        bus.emit(
            NodeFinished(
                unique_id=uid,
                resource_type="dataset",
                status=result.status,
                execution_time_s=result.execution_time_s,
                index=index,
                total=ctx.total_nodes,
                message=result.message,
            )
        )
        return result

    def _materialize(self, node: ManifestNode, spec: DatasetSpec) -> Any:
        ctx = self.ctx
        key = materialization_key(node)
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
        # The spine: the label table for multi-table inputs, else the source.
        spine_uid = spec.inputs.label if spec.inputs is not None else spec.source
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
        sample_fraction = float(ctx.merged_vars.get("sample_fraction", 1.0))
        build_ctx = _BuildContext(
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
        ctx = self.ctx
        node = ctx.manifest.nodes[uid]
        bus = get_bus()
        index = ctx.next_index()
        bus.emit(
            NodeStarted(unique_id=uid, resource_type="model", index=index, total=ctx.total_nodes)
        )
        started = time.monotonic()
        try:
            result = self._run_inner(uid, node)
        except MbtError as exc:
            result = NodeResult(unique_id=uid, status="error", message=str(exc))
        result.execution_time_s = time.monotonic() - started
        bus.emit(
            NodeFinished(
                unique_id=uid,
                resource_type="model",
                status=result.status,
                execution_time_s=result.execution_time_s,
                index=index,
                total=ctx.total_nodes,
                message=result.message,
            )
        )
        return result

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
        }
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
        handle = self.ctx.compute.submit(job)
        job_result = self.ctx.compute.wait(handle)

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
            message=None if passed else "one or more gates failed",
        )


class ModelTestRunner:
    """``mbt test`` for models: re-evaluate the latest registered version;
    training is never a side effect (TSD §11.3)."""

    def __init__(self, ctx: ExecutionContext) -> None:
        self.ctx = ctx
        self._model_runner = ModelRunner(ctx)

    def run(self, uid: str) -> NodeResult:
        ctx = self.ctx
        node = ctx.manifest.nodes[uid]
        started = time.monotonic()
        index = ctx.next_index()
        bus = get_bus()
        bus.emit(
            NodeStarted(unique_id=uid, resource_type="model", index=index, total=ctx.total_nodes)
        )
        try:
            result = self._run_inner(uid, node)
        except MbtError as exc:
            result = NodeResult(unique_id=uid, status="error", message=str(exc))
        result.execution_time_s = time.monotonic() - started
        bus.emit(
            NodeFinished(
                unique_id=uid,
                resource_type="model",
                status=result.status,
                execution_time_s=result.execution_time_s,
                index=index,
                total=ctx.total_nodes,
                message=result.message,
            )
        )
        return result

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

        metric_specs = self._model_runner._metric_specs(spec, node)
        champion, _ = self._model_runner._champion(spec, node)
        job = self._model_runner._assemble_job(
            node, spec, metric_specs, champion, mode="evaluate", artifact=version.artifact
        )
        job_result = ctx.compute.wait(ctx.compute.submit(job))
        if job_result.status == "error" or job_result.metrics is None:
            return NodeResult(
                unique_id=uid, status="error", message=job_result.error or "no metrics"
            )
        gates = self._model_runner._gate_results(spec, node, job_result, champion, metric_specs)
        passed = all_gates_passed(gates)
        return NodeResult(
            unique_id=uid,
            status="success" if passed else "test_failed",
            metrics=dict(job_result.metrics.metrics),
            slices=dict(job_result.metrics.slices),
            gates=gates,
            message=None if passed else "one or more gates failed",
        )
