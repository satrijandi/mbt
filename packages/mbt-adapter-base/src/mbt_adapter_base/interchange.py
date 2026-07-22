"""Interchange types crossing the adapter contract boundary (TSD §12.1).

Everything here is a plain Pydantic model (or a frozen dataclass built from
them): serializable, framework-free, and stable under the contract version.
"""

from collections.abc import Callable
from dataclasses import dataclass
from typing import TYPE_CHECKING, Any, Literal

from pydantic import BaseModel, ConfigDict, Field

from mbt_adapter_base.specs import AdapterRef, MetricSpec, ScoringOutputSpec
from mbt_adapter_base.types import Stage

if TYPE_CHECKING:
    from mbt_adapter_base.protocols import EventSink
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


class BootstrapDelta(_InterchangeModel):
    """Paired-bootstrap uncertainty for one champion-gate delta (ADR-18).

    ``lower`` is the one-sided lower confidence bound of the
    challenger-champion delta on the pinned test split; the gate criterion
    is ``lower >= min_delta``. ``n_resamples`` counts the valid
    (non-degenerate) resamples; 0 means ``lower`` fell back to ``point``.
    """

    point: float
    lower: float
    confidence: float
    n_resamples: int


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


class PredictionRunInfo(_InterchangeModel):
    """Sidecar metadata for one prediction run in a prediction store (ADR-21).

    ``run_key`` is the idempotency key: re-running the same manifest against
    the same champion overwrites the same run; new data, a new window, or a
    new champion version partitions fresh.
    """

    run_key: str
    uri: str
    scored_at: str  # ISO; the scoring run's manifest anchor
    run_id: str
    model_name: str
    model_version: str
    row_count: int
    meta: dict[str, str] = Field(default_factory=dict)  # config_hash, input_hash, ...


class ShiftStat(_InterchangeModel):
    """One computed distribution-shift statistic (ADR-20)."""

    method: Literal["psi", "ks"]
    value: float
    n_current: int
    n_baseline: int
    #: The kind of feature the stat was computed on. The n-aware ``significance``
    #: bar is kind-matched (F15): numeric KS stats get the two-sample Kolmogorov
    #: critical value; a categorical stat computed under significance is a
    #: Pearson chi-square (``df`` set) judged against the chi-square critical
    #: value. Defaults to ``numeric`` (scores, and older stats).
    kind: Literal["numeric", "categorical"] = "numeric"
    #: Chi-square degrees of freedom, set only on a categorical stat computed
    #: under ``significance`` (F15). None (older stats, threshold-path stats)
    #: falls back to the fixed ``threshold`` with a warning.
    df: int | None = None


class MonitorStats(_InterchangeModel):
    """Shift statistics computed by a scoring job; core applies thresholds.

    ``baseline_missing`` is set when the champion carries no baseline
    artifact (registered before baselines existed); monitors then pass with
    a loud warning (ADR-10 spirit).
    """

    feature_shift: dict[str, ShiftStat] = Field(default_factory=dict)
    prediction_shift: ShiftStat | None = None
    baseline_missing: bool = False
    skipped_features: list[str] = Field(default_factory=list)


class ManifestNode(_InterchangeModel):
    """One compiled DAG node (dataset, model, or scoring) as pinned in the manifest.

    ``config`` is the fully rendered spec (window *expressions* intact,
    AUTO sentinels intact); ``resolved`` holds anchor-dependent values that
    are deliberately excluded from hashing (TSD §8.2, ADR-12).
    """

    unique_id: str
    resource_type: Literal["dataset", "model", "scoring"]
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

    mode: Literal["train", "evaluate", "score"] = "train"
    run_id: str
    project_dir: str
    target_name: str
    node: ManifestNode
    #: Score mode: the referenced model's manifest node (hooks path, ModelSpec).
    model_node: ManifestNode | None = None
    dataset: DatasetLocator
    #: The dataset node's resolved windows (implicit validation carve, TSD §13.5).
    dataset_windows: dict[str, Any] = Field(default_factory=dict)
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
    #: Resolved non-secret vars (tainted values are never serialized into jobs).
    vars: dict[str, Any] = Field(default_factory=dict)
    #: Score mode only (ADR-20/21): prediction sink, champion baseline,
    #: resolved champion version, and the prediction-run idempotency key.
    output: ScoringOutputSpec | None = None
    baseline: ArtifactRef | None = None
    model_version: str | None = None
    run_key: str | None = None


class TuningResult(_InterchangeModel):
    """Outcome of a tuning loop."""

    best_params: dict[str, Any]
    best_value: float
    n_trials: int
    #: Trials stopped early by a pruner (subset of n_trials); 0 without one.
    n_pruned: int = 0


class JobResult(_InterchangeModel):
    """What a training job returns to the coordinator (via a result file)."""

    status: Literal["success", "error"]
    metrics: MetricResults | None = None
    champion_metrics: MetricResults | None = None
    #: Paired-bootstrap delta bounds per champion-gate metric (ADR-18).
    champion_delta_bounds: dict[str, BootstrapDelta] = Field(default_factory=dict)
    #: Normalized per-feature importance from the adapter, when it exposes
    #: ``feature_importance`` (FR-DOCS-02); empty otherwise.
    feature_importance: dict[str, float] = Field(default_factory=dict)
    #: Partial dependence for the top numeric features (explainability): feature
    #: -> ``[[grid_value, avg_prediction], ...]``, how the average prediction
    #: moves as the feature sweeps its range. Empty when unavailable.
    partial_dependence: dict[str, list[list[float]]] = Field(default_factory=dict)
    #: Walk-forward backtest (R2-7): builtin metric -> mean value across the
    #: time-ordered folds. Empty unless ``evaluation.protocol.backtest_folds`` is set.
    backtest_metrics: dict[str, float] = Field(default_factory=dict)
    #: The population std of each backtest metric across the folds (R2-7): the
    #: CV stability signal that accompanies ``backtest_metrics``' mean. Same keys.
    backtest_std: dict[str, float] = Field(default_factory=dict)
    resolved_auto: dict[str, Any] = Field(default_factory=dict)
    tuning: TuningResult | None = None
    artifact: ArtifactRef | None = None
    #: Train mode: the monitoring baseline exported next to the artifact (ADR-21).
    baseline: ArtifactRef | None = None
    #: Score mode: computed shift statistics and the written prediction run.
    monitor_stats: MonitorStats | None = None
    predictions: PredictionRunInfo | None = None
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


#: Per-iteration progress report during a tuning trial: ``report(step, value)``
#: with a HIGHER-IS-BETTER validation value (engines flip the sign for
#: minimize objectives). The callback may raise to abort the trial (pruning);
#: adapters must let that exception propagate out of their training loop.
TrialReportFn = Callable[[int, float], None]

#: Signature of the per-trial objective a TuningEngine drives. When the
#: tuning spec declares a pruner, engines call ``objective(params,
#: report=...)``; objectives accept the keyword and forward it to training
#: adapters that expose ``train_with_report`` (optional, hasattr-based).
TuningObjectiveFn = Callable[[dict[str, Any]], float]
