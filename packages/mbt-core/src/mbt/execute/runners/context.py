"""The state every node runner shares, and the node lifecycle (TSD §10.4/§10.5).

``ExecutionContext`` is what a run knows (manifest, profiles, adapters, the lock
and layout); ``run_with_lifecycle`` is the one place the NodeStarted/NodeFinished
protocol lives; ``evaluation_node_result`` is the one place ``mbt test`` and
``mbt evaluate`` assemble a node's result.
"""

import contextlib
import hashlib
import threading
import time
from collections.abc import Callable
from concurrent.futures import ThreadPoolExecutor
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, cast

from mbt.adapters.registry import AdapterRegistry
from mbt.artifacts.manifest import Manifest
from mbt.artifacts.run_results import (
    GateResult,
    NodeResult,
)
from mbt.config.profiles import LoadedProfiles
from mbt.dag.selector import SelectableNode
from mbt.events import get_bus
from mbt.events.models import NodeFinished, NodeStarted
from mbt.events.node_log import NodeLogSink, node_scope
from mbt.exceptions import JobInterrupted, MbtError, StateError
from mbt.quality.checks import SourceAccess
from mbt.quality.gates import all_gates_passed
from mbt.quality.judgement import gate_failure_summary
from mbt.quality.python_tests import PythonTestFile
from mbt.runtime import (
    data_adapter as build_data_adapter,
)
from mbt.runtime import (
    registry_adapter as build_registry_adapter,
)
from mbt.runtime import (
    tracking_adapter as build_tracking_adapter,
)
from mbt.secrets import Secret
from mbt.state.layout import TargetLayout
from mbt.utils import canonical_json
from mbt_adapter_base import (
    AdapterRef,
    DatasetLocator,
    JobResult,
    ManifestNode,
    SourceTable,
    TrainingJob,
)
from mbt_adapter_base.capabilities import Capability, supports


def evaluation_node_result(
    uid: str,
    job_result: JobResult,
    gates: list[GateResult],
    *,
    pass_status: str,
    fail_status: str,
    gated: bool = True,
    include_feature_importance: bool = False,
    include_message: bool = False,
) -> NodeResult:
    """Assemble a NodeResult from an ``evaluate_artifact`` outcome.

    Shared by ``mbt test`` (ModelTestRunner) and ``mbt evaluate`` (run_evaluate)
    so the two cannot drift on error handling or the metrics/slices/gates shape.
    Callers own only what genuinely differs: the pass/fail status names, whether
    gates apply (evaluate can skip them), and whether feature importance and a
    gate-failure message are surfaced (test surfaces both; evaluate neither).
    """
    if job_result.status == "error" or job_result.metrics is None:
        return NodeResult(unique_id=uid, status="error", message=job_result.error or "no metrics")
    passed = all_gates_passed(gates) if gated else True
    return NodeResult(
        unique_id=uid,
        status=(pass_status if passed else fail_status),  # type: ignore[arg-type]
        metrics=dict(job_result.metrics.metrics),
        slices=dict(job_result.metrics.slices),
        gates=gates,
        feature_importance=(
            dict(job_result.feature_importance) if include_feature_importance else {}
        ),
        partial_dependence=(
            dict(job_result.partial_dependence) if include_feature_importance else {}
        ),
        backtest_metrics=dict(job_result.backtest_metrics),
        backtest_std=dict(job_result.backtest_std),
        message=gate_failure_summary(gates) if include_message and not passed else None,
    )


@dataclass
class ExecutionContext:
    """Everything runners need for one invocation."""

    manifest: Manifest
    profiles: LoadedProfiles
    registry: AdapterRegistry
    project_dir: Path
    #: Where this invocation writes and where datasets are cached (ADR-32).
    layout: TargetLayout
    run_id: str
    command: str  # run | build | test | evaluate
    cli_vars: dict[str, Any] = field(default_factory=dict)
    python_tests: list[PythonTestFile] = field(default_factory=list)
    total_nodes: int = 0
    #: The DAG execution pool size (--threads), i.e. the max number of builds
    #: that can run concurrently. In-process DuckDB builds divide cores/RAM by it
    #: so N parallel builds do not each grab all cores and 80% of RAM (F22).
    threads: int = 1
    _dataset_handles: dict[str, Any] = field(default_factory=dict)
    _counter: list[int] = field(default_factory=lambda: [0])
    _active_job_handles: list[Any] = field(default_factory=list)
    _job_handles_lock: Any = field(default_factory=threading.Lock)
    #: Guards the coordinator-side shared state touched by pool threads - the
    #: job-index counter and the dataset-handle map. Their mutations are safe
    #: under the GIL but genuinely racy under a free-threaded (3.13t/3.14t)
    #: build, so serialize them here (P3).
    _state_lock: Any = field(default_factory=threading.Lock)
    #: Per-node run logs for this invocation (ADR-30); None when the caller
    #: did not install them.
    node_logs: NodeLogSink | None = None
    #: ``--phase`` (ADR-35): the one step of each node this invocation runs, or None
    #: for the whole node.
    phase: str | None = None

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
        # migrations) before parallel jobs hit them concurrently. Only training
        # opens tracking runs (ADR-28), so a scoring or monitoring invocation
        # does not touch the tracker at all - and does not create its
        # experiment on a store that has never trained.
        if any(node.resource_type == "model" for node in self.manifest.nodes.values()):
            tracking = self.tracking()
            if supports(tracking, Capability.PREPARE):
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
        return build_tracking_adapter(
            self.profiles, self.project_dir, self.registry, self.manifest.metadata.project_name
        )

    def registry_adapter(self) -> Any:
        return build_registry_adapter(self.profiles, self.project_dir, self.registry)

    def dataset_handle(self, uid: str) -> Any:
        """The handle for a dataset this model reads.

        Normally the one this invocation's own dataset node stored. Under
        ``--no-auto-materialize`` the dataset was built by some other
        invocation, so it comes from the dataset cache - or is an error: an
        exact-node run never builds what it was not asked to build.
        """
        with self._state_lock:
            handle = self._dataset_handles.get(uid)
        if handle is not None:
            return handle
        node = self.manifest.nodes[uid]
        handle = self.cached_dataset(node)
        if handle is None:
            key = materialization_key(node, self.sample_fraction())
            raise StateError(
                f"dataset {node.name!r} is not materialized in {self.layout.datasets} "
                f"(key {key[:12]})",
                resource=uid,
                hint=(
                    f"build it first with 'mbt run --phase materialize --select {node.name}'"
                    if self.phase == "checks"
                    else f"build it first with 'mbt build --select {node.name}'"
                )
                + " against the same --manifest, --vars and --datasets-path"
                + ("" if self.phase else ", or drop --no-auto-materialize"),
            )
        self.store_dataset_handle(uid, handle)
        return handle

    def sample_fraction(self) -> float:
        return float(self.merged_vars.get("sample_fraction", 1.0))

    def dataset_dir(self, node: ManifestNode) -> Path:
        """Where ``node``'s materialization lives (or would): ``<datasets>/<name>/<key>``."""
        return self.layout.dataset_dir(node.name, materialization_key(node, self.sample_fraction()))

    def cached_dataset(self, node: ManifestNode) -> Any:
        """The handle of an already-materialized dataset, or None if the cache has none.

        A materialization counts only once its ``_SUCCESS`` marker exists; the
        marker is the last thing a build writes.
        """
        output_dir = self.dataset_dir(node)
        if not (output_dir / "_SUCCESS").is_file():
            return None
        return self.data_adapter.from_locator(
            DatasetLocator(
                adapter=self.profiles.target.data.adapter,
                uri=f"file://{output_dir.resolve()}",
                snapshot_id=node.snapshot_id or "",
            )
        )

    def store_dataset_handle(self, uid: str, handle: Any) -> None:
        with self._state_lock:
            self._dataset_handles[uid] = handle

    def next_index(self) -> int:
        with self._state_lock:
            self._counter[0] += 1
            return self._counter[0]

    def run_job(self, job: TrainingJob) -> JobResult:
        """Submit + wait, tracking the handle so --fail-fast can reclaim it."""
        handle = self.compute.submit(job)
        with self._job_handles_lock:
            self._active_job_handles.append(handle)
        try:
            result = cast(JobResult, self.compute.wait(handle))
        finally:
            with self._job_handles_lock:
                self._active_job_handles.remove(handle)
        if result.interrupted:
            raise JobInterrupted(result.error or "interrupted")
        return result

    def cancel_active_jobs(self) -> None:
        """Terminate in-flight job subprocesses (--fail-fast); best-effort.

        Terminations run in parallel: each ``terminate()`` blocks up to its
        SIGTERM grace period, so a serial sweep of N in-flight jobs would take
        N * grace (~80s for a full --threads=16 pool). One shared pool lets the
        grace periods overlap, so reclaiming the pool takes ~one grace period.
        """
        if not supports(self.compute, Capability.TERMINATE):
            return  # older/remote compute adapters without a kill seam
        with self._job_handles_lock:
            handles = list(self._active_job_handles)

        def _terminate(handle: object) -> None:
            with contextlib.suppress(Exception):  # the job may have just exited
                self.compute.terminate(handle, "cancelled by --fail-fast")

        with ThreadPoolExecutor(max_workers=max(1, len(handles))) as pool:
            list(pool.map(_terminate, handles))

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


def source_access(ctx: "ExecutionContext", node: ManifestNode) -> SourceAccess:
    """Pre-join source reach for the check layer (F2/F21): the node's source
    tables keyed ``group.name`` plus the resolved data adapter, so
    source-level checks (``unique`` with ``source:``, ``relationships``) can
    read the raw tables before any join fans them out."""
    tables = {
        f"{entry.group}.{entry.name}": entry.config
        for uid, entry in ctx.manifest.sources.items()
        if uid in node.depends_on
    }
    return SourceAccess(tables=tables, adapter=ctx.data_adapter)


@dataclass(frozen=True)
class BuildContext:
    """DataBuildContext implementation handed to DataAdapters."""

    node: ManifestNode
    source: SourceTable  # the one relation the node reads (ADR-29)
    source_tables: dict[str, SourceTable]  # every source dep by unique_id
    resolved_windows: dict[str, tuple[str, str]]
    sample_fraction: float
    deep_snapshot: bool
    output_dir: Path
    events: Any
    #: The DAG pool size (F22): a data adapter divides its in-process compute
    #: budget (DuckDB cores/RAM) by this so concurrent builds do not oversubscribe.
    build_parallelism: int = 1
    #: The projection hint (contract 1.3): the columns the caller reads, or
    #: None for all of them.
    columns: tuple[str, ...] | None = None


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
        with node_scope(uid):
            result = inner()
    except MbtError as exc:
        result = NodeResult(unique_id=uid, status="error", message=str(exc))
    except JobInterrupted as exc:
        bus.emit(
            NodeFinished(
                unique_id=uid,
                resource_type=resource_type,
                status="interrupted",
                execution_time_s=time.monotonic() - started,
                index=index,
                total=ctx.total_nodes,
                message=str(exc),
            )
        )
        raise
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
