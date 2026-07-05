"""Interchange types crossing the adapter contract boundary (TSD §12.1).

Everything here is a plain Pydantic model (or a frozen dataclass built from
them): serializable, framework-free, and stable under the contract version.
"""

from collections.abc import Callable
from dataclasses import dataclass
from typing import TYPE_CHECKING, Any, Literal

from pydantic import BaseModel, ConfigDict, Field

from mbt_adapter_base.specs import AdapterRef, MetricSpec
from mbt_adapter_base.types import Stage

if TYPE_CHECKING:
    from mbt_adapter_base.protocols import DatasetHandle, EventSink
    from mbt_adapter_base.specs import ModelSpec


class _InterchangeModel(BaseModel):
    model_config = ConfigDict(extra="forbid")


class ValidationIssue(_InterchangeModel):
    """One validation finding, reported with enough context to act on."""

    severity: Literal["error", "warning"]
    resource: str  # unique_id
    field_path: str  # JSON pointer, e.g. "/hyperparameters/max_depth"
    message: str
    hint: str | None = None


class DeterminismTier(_InterchangeModel):
    """An adapter's documented reproducibility guarantee (FR-ADPT-07)."""

    kind: Literal["exact", "tolerance"]
    tolerances: dict[str, float] = Field(default_factory=dict)  # metric -> abs tolerance

    def tolerance_for(self, metric: str) -> float:
        """Absolute tolerance for a metric; 0.0 under the exact tier."""
        if self.kind == "exact":
            return 0.0
        return self.tolerances.get(metric, self.tolerances.get("*", 0.0))


class ArtifactRef(_InterchangeModel):
    """A pointer to an exported model artifact in the artifact store."""

    uri: str
    format: str  # e.g. "xgboost_ubj", "lightgbm_txt", "onnx"
    content_hash: str  # "sha256:..."
    size_bytes: int


class MetricResults(_InterchangeModel):
    """Metrics computed by an adapter; core only ever compares these."""

    metrics: dict[str, float]
    slices: dict[str, dict[str, float]] = Field(default_factory=dict)


class DatasetProfile(_InterchangeModel):
    """Cheap dataset statistics used for validation and AUTO resolution."""

    n_rows: dict[str, int]  # per split
    columns: dict[str, str]  # name -> arrow dtype string
    label_column: str
    label_balance: dict[str, float] | None = None  # classification only
    time_range: tuple[str, str] | None = None


class DatasetLocator(_InterchangeModel):
    """Serializable pointer to a materialized dataset (job payloads)."""

    adapter: str
    uri: str  # e.g. "file:///.../target/datasets/churn/<key>"
    snapshot_id: str


class ModelVersion(_InterchangeModel):
    """A registered model version as seen through a RegistryAdapter."""

    name: str
    version: str
    stage: Stage | None = None
    artifact: ArtifactRef | None = None
    tags: dict[str, str] = Field(default_factory=dict)


class RunHandle(_InterchangeModel):
    """A tracking run reference."""

    run_id: str
    url: str | None = None


class ManifestNode(_InterchangeModel):
    """One compiled DAG node (dataset or model) as pinned in the manifest.

    ``config`` is the fully rendered spec (window *expressions* intact,
    AUTO sentinels intact); ``resolved`` holds anchor-dependent values that
    are deliberately excluded from hashing (TSD §8.2, ADR-12).
    """

    unique_id: str
    resource_type: Literal["dataset", "model"]
    name: str
    path: str  # spec file, relative to the project root
    depends_on: list[str] = Field(default_factory=list)
    config: dict[str, Any]
    resolved: dict[str, Any] = Field(default_factory=dict)
    snapshot_id: str | None = None  # datasets only
    adapter: str | None = None  # models only
    task: str | None = None  # models only
    seed: int | None = None  # models only
    hooks_path: str | None = None  # models only; relative to project root
    hooks_hash: str | None = None  # null if no hooks.py
    config_hash: str = ""
    input_hash: str = ""

    @property
    def tags(self) -> list[str]:
        tags = self.config.get("tags", [])
        return list(tags) if isinstance(tags, list) else []


class TrainingJob(_InterchangeModel):
    """The serialized coordinator -> job payload (TSD §10.3, ADR-3).

    Carries the unrendered data/tracking adapter refs so the job process can
    re-resolve ``env_var()`` secrets from its own environment (TSD §18), and
    the resolved metric specs so adapters compute exactly what core compares.
    """

    mode: Literal["train", "evaluate"] = "train"
    run_id: str
    project_dir: str
    target_name: str
    node: ManifestNode
    dataset: DatasetLocator
    data: AdapterRef
    tracking: AdapterRef | None = None
    metric_specs: list[MetricSpec] = Field(default_factory=list)
    champion: ArtifactRef | None = None
    artifact: ArtifactRef | None = None  # evaluate mode: the artifact under evaluation
    tuning_engine: AdapterRef | None = None
    tuning_cap: int | None = None
    artifact_store: str = ""
    required_env: list[str] = Field(default_factory=list)  # names only, never values
    tracking_meta: dict[str, str] = Field(default_factory=dict)  # git/manifest metadata tags


class TuningResult(_InterchangeModel):
    """Outcome of a tuning loop."""

    best_params: dict[str, Any]
    best_value: float
    n_trials: int


class JobResult(_InterchangeModel):
    """What a training job returns to the coordinator (via a result file)."""

    status: Literal["success", "error"]
    metrics: MetricResults | None = None
    champion_metrics: MetricResults | None = None
    resolved_auto: dict[str, Any] = Field(default_factory=dict)
    tuning: TuningResult | None = None
    artifact: ArtifactRef | None = None
    tracking_run_id: str | None = None
    error: str | None = None


class TestResult(_InterchangeModel):
    """Outcome of one data test or check."""

    name: str
    passed: bool
    message: str = ""


@dataclass(frozen=True)
class RunContext:
    """Execution context handed to adapters inside the training job."""

    run_id: str
    unique_id: str
    seed: int
    target_name: str
    project_dir: str
    vars: dict[str, Any]
    events: "EventSink"


@dataclass(frozen=True)
class HookContext:
    """Context handed to ``hooks.py`` functions (TSD §5.8, §12.1)."""

    spec: "ModelSpec"
    profile: DatasetProfile
    split: str
    logger: "EventSink"


#: Signature of the per-trial objective a TuningEngine drives.
TuningObjectiveFn = Callable[[dict[str, Any]], float]
