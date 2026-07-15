"""Adapter Protocols: what plugins implement (TSD §12.2).

Construction convention: every adapter class is constructed with a single
positional argument, the adapter-specific config dict from the target's
``AdapterRef.config`` (an empty dict for adapters that need none). The
compliance suite enforces this.

Import hygiene rule (ADR-14): importing a plugin module - and constructing
its adapter classes - must not import the ML framework. Frameworks load
lazily inside ``train``/``evaluate``/``load``/``export``.
"""

from dataclasses import dataclass, field
from pathlib import Path
from typing import TYPE_CHECKING, Any, Protocol, runtime_checkable

import pyarrow as pa
from pydantic import BaseModel

from mbt_adapter_base.interchange import (
    ArtifactRef,
    DatasetLocator,
    DatasetProfile,
    DeterminismTier,
    JobResult,
    ManifestNode,
    MetricResults,
    ModelVersion,
    PredictionRunInfo,
    RunContext,
    RunHandle,
    TestResult,
    TrainingJob,
    TuningObjectiveFn,
    TuningResult,
    ValidationIssue,
)
from mbt_adapter_base.types import Stage, TaskType

if TYPE_CHECKING:
    from mbt_adapter_base.specs import (
        DatasetSpec,
        MetricSpec,
        ModelSpec,
        ScoringInputSpec,
        ScoringOutputSpec,
        TuningSpec,
    )


class EventSink(Protocol):
    """Minimal event outlet available to adapters and hooks."""

    def emit(self, event: object) -> None:
        """Emit a typed event or a plain message object."""
        ...


@runtime_checkable
class TrainedModel(Protocol):
    """Deliberately opaque: only the owning adapter's methods touch it."""


class JobHandle(Protocol):
    """A submitted compute job."""

    @property
    def job_id(self) -> str: ...


class ArtifactStore(Protocol):
    """Where exported model artifacts live (file:// in v0, s3:// later)."""

    @property
    def uri(self) -> str: ...

    def put_file(self, local_path: Path, name: str, format: str) -> ArtifactRef:
        """Store a file under this run's artifact prefix and return its ref."""
        ...

    def fetch(self, ref: ArtifactRef) -> Path:
        """Materialize an artifact locally and return its path."""
        ...


class DatasetHandle(Protocol):
    """A materialized dataset an adapter can read splits from (TSD §12.2)."""

    @property
    def snapshot_id(self) -> str: ...

    def splits(self) -> set[str]:
        """Available split names, e.g. {"train", "test"} (+ "validation")."""
        ...

    def read(self, split: str, columns: list[str] | None = None) -> pa.Table: ...

    def profile(self) -> DatasetProfile: ...

    def locator(self) -> DatasetLocator:
        """Serializable pointer for job payloads."""
        ...


class TaskSchema(Protocol):
    """Task-specific validation selected by a model's ``task`` field (FR-RES-08)."""

    @property
    def task(self) -> TaskType: ...

    @property
    def allowed_metrics(self) -> set[str]:
        """Builtin metric names valid for this task (hook metrics exempt)."""
        ...

    def validate_spec(self, spec: "ModelSpec") -> list[ValidationIssue]:
        """Parse-time validation (metric names, protocol sanity)."""
        ...

    def validate_dataset(self, spec: "ModelSpec", profile: DatasetProfile) -> list[ValidationIssue]:
        """Run-time validation once the dataset profile exists."""
        ...


class TrainingAdapter(Protocol):
    """Executes training for the tasks it supports (TSD §12.2).

    ``data_access`` declares how the adapter reads splits:

    - ``"arrow"`` (default): via ``DatasetHandle.read()`` as Arrow tables.
    - ``"path"``: via the handle's on-disk parquet files - for JVM/cluster
      frameworks (H2O, Spark) that ingest files natively. The training job
      guarantees such adapters a ``MaterializedDatasetHandle`` whose
      ``split_path(split)`` files already have hooks and feature selection
      applied.
    """

    @property
    def name(self) -> str: ...

    @property
    def data_access(self) -> str:
        """ "arrow" | "path" (see class docstring)."""
        ...

    @property
    def contract_version(self) -> str: ...

    @property
    def supported_tasks(self) -> set[TaskType]: ...

    @property
    def determinism(self) -> DeterminismTier: ...

    def param_model(self, task: TaskType) -> type[BaseModel]:
        """The Pydantic model validating static hyperparameters for a task."""
        ...

    def validate(self, spec: "ModelSpec") -> list[ValidationIssue]: ...

    def resolve_auto(self, spec: "ModelSpec", profile: DatasetProfile) -> "ModelSpec":
        """Replace AUTO sentinels with values derived from the profile."""
        ...

    def train(self, spec: "ModelSpec", data: DatasetHandle, ctx: RunContext) -> TrainedModel: ...

    def evaluate(
        self,
        model: TrainedModel,
        data: DatasetHandle,
        split: str,
        metrics: list["MetricSpec"],
        slices: list[str] | None = None,
    ) -> MetricResults: ...

    def predict(self, model: TrainedModel, data: DatasetHandle, split: str) -> pa.Table:
        """The split's table plus a ``prediction`` column (hook metrics, v1 scoring)."""
        ...

    def load(self, ref: ArtifactRef, store: ArtifactStore) -> TrainedModel: ...

    def export(self, model: TrainedModel, format: str, store: ArtifactStore) -> ArtifactRef: ...

    def nondeterminism_warnings(self, spec: "ModelSpec") -> list[str]:
        """Known nondeterminism sources in this spec (FR-RUN-06)."""
        ...


@runtime_checkable
class SupportsTrainWithReport(Protocol):
    """OPTIONAL TrainingAdapter capability: per-round tuning progress.

    Core probes for the method with ``hasattr``; this protocol pins the
    signature so adapters cannot drift apart. ``report(step, value)`` takes a
    HIGHER-IS-BETTER validation value each round (the tuning engine flips the
    sign for minimize objectives) and may raise to abort the trial (pruning);
    let that exception propagate out of the framework's training loop.
    Adapters without the method fall back to full trials with a warning.
    """

    def train_with_report(
        self,
        spec: "ModelSpec",
        data: DatasetHandle,
        ctx: RunContext,
        report: Any,
    ) -> TrainedModel: ...


@runtime_checkable
class SupportsFeatureImportance(Protocol):
    """OPTIONAL TrainingAdapter capability: per-feature importance.

    Probed with ``hasattr`` after training (FR-DOCS-02). Return importances
    normalized to fractions keyed by feature name; absence simply leaves
    model cards without an importance table. Adapters whose winning model
    cannot attribute importance (e.g. an AutoML ensemble leader) return {}.
    """

    def feature_importance(self, model: TrainedModel) -> dict[str, float]: ...


class PredictionStore(Protocol):
    """One scoring pipeline's prediction sink (contract 1.1, ADR-21).

    Owns writing, scanning, and the ground-truth evaluation ledger, so core
    never hard-codes a storage layout. ``write_run`` is idempotent by
    ``run_key``: an existing run with the same key is replaced atomically.
    """

    def write_run(self, table: pa.Table, info: PredictionRunInfo) -> PredictionRunInfo:
        """Write one prediction run; returns the info as persisted."""
        ...

    def list_runs(self) -> list[PredictionRunInfo]:
        """All complete runs, ordered by ``scored_at`` ascending."""
        ...

    def read(self, run_key: str, columns: list[str] | None = None) -> pa.Table: ...

    def read_marker(self, run_key: str, name: str) -> dict[str, Any] | None:
        """A named ledger marker for a run, or None if absent."""
        ...

    def write_marker(self, run_key: str, name: str, payload: dict[str, Any]) -> None: ...


class DataAdapter(Protocol):
    """Builds and reopens datasets (TSD §12.2).

    ``build_scoring_input`` and ``open_predictions`` are contract 1.1
    additions (ADR-20/21); core probes for them with ``hasattr`` and fails
    with a clear error before any job runs when an adapter predates them.
    """

    @property
    def name(self) -> str: ...

    def snapshot_id(self, source: "SourceTableLike", deep: bool = False) -> str: ...

    def build_dataset(self, spec: "DatasetSpec", ctx: "DataBuildContext") -> DatasetHandle: ...

    def from_locator(self, locator: DatasetLocator) -> DatasetHandle: ...

    def build_scoring_input(
        self, spec: "ScoringInputSpec", ctx: "DataBuildContext"
    ) -> DatasetHandle:
        """Materialize one unlabeled batch as a single ``score`` split."""
        ...

    def open_predictions(self, output: "ScoringOutputSpec") -> PredictionStore: ...


class TrackingAdapter(Protocol):
    """Experiment tracking (TSD §12.2)."""

    def start_run(self, node: ManifestNode, meta: dict[str, str]) -> RunHandle: ...

    def log(
        self,
        run: RunHandle,
        *,
        params: dict[str, Any] | None = None,
        metrics: dict[str, float] | None = None,
        tags: dict[str, str] | None = None,
        artifacts: list[ArtifactRef] | None = None,
    ) -> None: ...

    def end_run(self, run: RunHandle, status: str) -> None: ...

    def resume(self, run_id: str) -> RunHandle: ...


class RegistryAdapter(Protocol):
    """Model registry (TSD §12.2)."""

    def register(
        self, artifact: ArtifactRef, name: str, metadata: dict[str, str]
    ) -> ModelVersion: ...

    def get_champion(self, name: str, stage: Stage) -> ModelVersion | None: ...

    def get_version(self, name: str, version: str) -> ModelVersion | None: ...

    def transition(self, version: ModelVersion, stage: Stage) -> None: ...


class ComputeAdapter(Protocol):
    """Where training jobs run (local subprocess in v0; K8s/Ray in v1)."""

    def submit(self, job: TrainingJob) -> JobHandle: ...

    def wait(self, handle: JobHandle) -> JobResult: ...


class TuningEngine(Protocol):
    """Proposes hyperparameters; the trial loop runs inside the job (TSD §13.5)."""

    @property
    def name(self) -> str: ...

    def tune(
        self,
        spec: "TuningSpec",
        objective: TuningObjectiveFn,
        n_trials: int,
        seed: int,
    ) -> TuningResult: ...


class SourceTableLike(Protocol):
    """Structural view of a source table as DataAdapters need it."""

    @property
    def name(self) -> str: ...

    @property
    def path(self) -> str | None: ...

    @property
    def identifier(self) -> str | None: ...

    @property
    def format(self) -> str: ...


class DataBuildContext(Protocol):
    """What a DataAdapter gets to build one dataset materialization."""

    @property
    def node(self) -> ManifestNode: ...

    @property
    def source(self) -> SourceTableLike:
        """The spine table: the single source, or the label table for
        multi-table ``inputs`` datasets."""
        ...

    @property
    def source_tables(self) -> dict[str, SourceTableLike]:
        """Every source table by unique_id (spine + feature tables)."""
        ...

    @property
    def resolved_windows(self) -> dict[str, tuple[str, str]]: ...

    @property
    def sample_fraction(self) -> float: ...

    @property
    def deep_snapshot(self) -> bool: ...

    @property
    def output_dir(self) -> Path: ...

    @property
    def events(self) -> EventSink: ...


@dataclass(frozen=True)
class AdapterPlugin:
    """The entry-point descriptor exposed by every adapter package (TSD §12.3).

    Registered under the ``mbt.adapters`` entry-point group; the module
    holding it must be cheap to import (no ML framework imports, ADR-14).
    """

    name: str
    contract_version: str
    training: type[Any] | None = None
    data: type[Any] | None = None
    tracking: type[Any] | None = None
    registry: type[Any] | None = None
    compute: type[Any] | None = None
    tuning: type[Any] | None = None
    task_schemas: dict[TaskType, type[Any]] = field(default_factory=dict)
    fingerprint_packages: list[str] = field(default_factory=list)


#: Callable signature of Python data tests in ``tests/`` (TSD §5.7).
PythonDataTest = Any  # def test_*(dataset: pa.Table, spec: DatasetSpec) -> TestResult

__all__ = [
    "AdapterPlugin",
    "ArtifactStore",
    "ComputeAdapter",
    "DataAdapter",
    "DataBuildContext",
    "DatasetHandle",
    "EventSink",
    "JobHandle",
    "PredictionStore",
    "PythonDataTest",
    "RegistryAdapter",
    "SourceTableLike",
    "TaskSchema",
    "TestResult",
    "TrackingAdapter",
    "TrainedModel",
    "TrainingAdapter",
    "TuningEngine",
]
