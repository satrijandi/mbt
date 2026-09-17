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


#: The after-test grains a report cell can be cut at (ADR-30).
CellPeriod = Literal["window", "month", "week_of_month", "day_of_month"]


class PeriodCell(_InterchangeModel):
    """One after-test performance cell of the training report (ADR-30).

    ``metrics`` is empty when the cell has no mature labelled rows or, for a
    classifier, only one class - the metrics are undefined there, and an
    after-test gate treats the cell as not judgeable.
    """

    period: CellPeriod
    #: ``window``, ``2026-06``, ``2026-06/W1`` or ``2026-06-05``.
    key: str
    #: The reference slot it is compared with: ``all``, ``W1`` or ``D05``.
    slot: str
    start: str  # ISO bounds of the cell (the window's are its first/last row)
    end: str
    n_rows: int
    #: Rows whose outcome is known by the anchor (``label.horizon`` rule).
    n_labelled: int
    #: Every row in the cell is mature, so its metrics are final.
    mature: bool
    metrics: dict[str, float] = Field(default_factory=dict)
    reference_rows: int = 0
    reference_metrics: dict[str, float] = Field(default_factory=dict)
    note: str = ""


class StabilityCell(_InterchangeModel):
    """Score and feature stability of one after-test cell vs the test split (ADR-30)."""

    period: CellPeriod
    key: str
    slot: str
    n_rows: int
    reference_rows: int
    score_psi: float | None = None
    score_ks: float | None = None
    max_feature_psi: float | None = None
    max_feature: str | None = None
    #: Features whose PSI passes the conventional 0.1 / 0.25 bands.
    features_over_warn: int = 0
    features_over_fail: int = 0
    #: The declared ``evaluation.stability`` statistics, on the cells that
    #: stability gate judges; None everywhere else.
    gate: MonitorStats | None = None


class DriftColumn(_InterchangeModel):
    """One column of a report engine's drift table (ADR-30, display only)."""

    column: str
    #: The engine's own test or distance, e.g. ``K-S p_value``.
    method: str
    score: float
    threshold: float
    drifted: bool


class DriftReport(_InterchangeModel):
    """What a report engine found comparing one after-test cell with test."""

    #: Share of compared columns the engine calls drifted.
    drifted_share: float
    columns: list[DriftColumn] = Field(default_factory=list)


class ReportSummary(_InterchangeModel):
    """What the training report hands back to the coordinator (ADR-30).

    The per-feature and per-bin detail lives in the report's own files; this
    carries what core gates on and what the run records as metrics.
    """

    anchor: str
    reference_rows: int
    out_of_time_rows: int
    #: Metrics per split over its labelled rows (the after-test split: mature ones).
    split_metrics: dict[str, dict[str, float]] = Field(default_factory=dict)
    #: The top features by importance, in rank order.
    importance: dict[str, float] = Field(default_factory=dict)
    periods: list[PeriodCell] = Field(default_factory=list)
    stability: list[StabilityCell] = Field(default_factory=list)
    #: Grains the data cannot support, with the reason.
    skipped_periods: dict[str, str] = Field(default_factory=dict)
    #: Report files, relative to the report directory.
    documents: list[str] = Field(default_factory=list)
    #: Where ``report.html`` landed in the artifact store.
    report_uri: str | None = None
    warnings: list[str] = Field(default_factory=list)


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

    mode: Literal["train", "evaluate", "score", "oot_check"] = "train"
    run_id: str
    project_dir: str
    target_name: str
    #: The mbt project's name; half of the tracking experiment name (ADR-28).
    project: str = ""
    #: The manifest-wide anchor (ADR-12). Score mode stamps it on the
    #: prediction run as ``scored_at``, which is what ``mbt monitor`` measures
    #: maturity from - so it is job payload, not tracking metadata.
    anchor: str = ""
    node: ManifestNode
    #: Score mode: the referenced model's manifest node (hooks path, ModelSpec).
    model_node: ManifestNode | None = None
    #: Score mode: the champion's own exported model spec, read back from the
    #: artifact it was registered with (ADR-28). Authoritative over
    #: ``model_node.config``, which is whatever the working tree says today.
    #: None only for a champion registered before mbt exported one.
    champion_spec: dict[str, Any] | None = None
    #: Score mode: the exact feature columns, in order, the champion was fit on
    #: (ADR-28's ``resolved.feature_columns``). The manifest cannot answer this
    #: at all, because ``features.include: ["*"]`` does not say what it matched.
    #: Authoritative over re-evaluating the globs against whatever the batch
    #: happens to hold (ADR-29). None alongside a None ``champion_spec``.
    champion_feature_columns: list[str] | None = None
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
    #: The model's dataset node (ADR-30): its spec is logged as flat params and
    #: recorded in the inference config, and the report reads its sample key,
    #: time column and label horizon.
    dataset_node: ManifestNode | None = None
    #: An existing tracking run to append to instead of opening one - the
    #: pre-deploy check writes to the training run of the version it checks
    #: (ADR-30).
    tracking_run_id: str | None = None


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
    #: Train mode: the inference config exported next to the artifact (ADR-28),
    #: which is what a later scoring run reads the model's spec from.
    inference_config: ArtifactRef | None = None
    #: Score mode: computed shift statistics and the written prediction run.
    monitor_stats: MonitorStats | None = None
    predictions: PredictionRunInfo | None = None
    #: Train and oot_check modes: the training report (ADR-30).
    report: ReportSummary | None = None
    #: Set on failed training jobs too, so the coordinator can still attach
    #: the log to the run that recorded the failure.
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
