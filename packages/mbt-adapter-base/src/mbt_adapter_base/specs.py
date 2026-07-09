"""Resource spec schemas (TSD §5.4-§5.8).

These are the Pydantic models behind the YAML resources users write.
They live in mbt-adapter-base because adapters consume them across the
contract boundary (``TrainingAdapter.validate(spec)``, ``DataAdapter.build_dataset(spec)``).

All schemas reject unknown fields (``extra="forbid"``, FR-PARSE-04); the
parser layer turns those rejections into did-you-mean suggestions.
"""

from typing import Any, Literal

from pydantic import BaseModel, ConfigDict, Field, model_validator

from mbt_adapter_base.types import Materialization, SplitStrategy, Stage, TaskType

#: Resource names: lowercase snake_case, so unique_ids stay unambiguous.
NAME_PATTERN = r"^[a-z][a-z0-9_]*$"


class _SpecModel(BaseModel):
    """Base for all spec schemas: strict fields, validate on assignment."""

    model_config = ConfigDict(extra="forbid", validate_assignment=True)


class AdapterRef(_SpecModel):
    """A named adapter plus its adapter-specific config (TSD §5.3)."""

    adapter: str
    config: dict[str, Any] = Field(default_factory=dict)


class SourceTable(_SpecModel):
    """One external input table/path within a source group (TSD §5.4)."""

    name: str = Field(pattern=NAME_PATTERN)
    path: str | None = None  # for path-based sources (parquet/iceberg)
    identifier: str | None = None  # for warehouse/feature-store sources (v1)
    format: str = "parquet"  # parquet | iceberg
    description: str = ""

    @model_validator(mode="after")
    def _path_or_identifier(self) -> "SourceTable":
        if self.path is None and self.identifier is None:
            raise ValueError("a source table needs either 'path' or 'identifier'")
        return self


class SourceGroup(_SpecModel):
    """A named group of source tables, e.g. ``lakehouse`` (TSD §5.4)."""

    name: str = Field(pattern=NAME_PATTERN)
    description: str = ""
    tables: list[SourceTable]


class LabelSpec(_SpecModel):
    """The label column and its human definition (shown on model cards)."""

    column: str
    definition: str = ""


class SplitSpec(_SpecModel):
    """Split policy for a dataset (TSD §5.5).

    ``train``/``test``/``validation`` are window expressions for the temporal
    strategy (``"-180d:-28d"``) and fractions (``"0.8"``) for the random one.
    """

    strategy: SplitStrategy = SplitStrategy.TEMPORAL
    time_column: str | None = None  # required if temporal
    train: str
    test: str
    validation: str | None = None  # else carved from train when tuning needs it
    stratify_by: str | None = None  # random strategy only
    seed: int | None = None  # random strategy only; required then

    @model_validator(mode="after")
    def _strategy_requirements(self) -> "SplitSpec":
        if self.strategy is SplitStrategy.TEMPORAL:
            if self.time_column is None:
                raise ValueError("temporal split requires 'time_column'")
            if self.stratify_by is not None:
                raise ValueError("'stratify_by' applies to the random strategy only")
            if self.seed is not None:
                raise ValueError(
                    "'seed' applies to the random strategy only; "
                    "temporal splits are deterministic by time"
                )
        else:  # RANDOM
            if self.seed is None:
                raise ValueError(
                    "random split requires an explicit 'seed' (reproducibility, FR-RES-09)"
                )
            for field in ("train", "test", "validation"):
                value = getattr(self, field)
                if value is None:
                    continue
                try:
                    fraction = float(value)
                except ValueError:
                    raise ValueError(
                        f"random split '{field}' must be a fraction like '0.8', got {value!r}"
                    ) from None
                if not 0.0 < fraction < 1.0:
                    raise ValueError(f"random split '{field}' fraction must be in (0, 1)")
        return self


#: A dataset check: a bare name (``"label_leakage_scan"``) or a one-key map of
#: check name to parameters (``{not_null: {columns: [...]}}``), TSD §5.5.
CheckSpec = str | dict[str, dict[str, Any]]


class DatasetInputs(_SpecModel):
    """Multi-table dataset construction: feature table(s) + a label table.

    The label table is the spine - it defines which examples exist. Each
    feature table joins onto it via ``join_key`` (``left`` join by default,
    so examples with missing features arrive with NULLs; tree adapters
    handle those natively). Column names must be unique across tables apart
    from the join key(s).
    """

    features: list[str]  # source() refs to feature tables
    label: str  # source() ref to the label table (the spine)
    join_key: str | list[str]
    join: Literal["left", "inner"] = "left"

    @property
    def join_columns(self) -> list[str]:
        return [self.join_key] if isinstance(self.join_key, str) else list(self.join_key)

    @model_validator(mode="after")
    def _shape(self) -> "DatasetInputs":
        if not self.features:
            raise ValueError("inputs.features must list at least one feature table")
        if not self.join_columns or any(not c for c in self.join_columns):
            raise ValueError("inputs.join_key must name at least one non-empty column")
        return self


class DatasetSpec(_SpecModel):
    """Declarative training-set construction (TSD §5.5, FR-RES-02).

    Data comes from exactly one of:

    - ``source``: a single table holding features and the label, or
    - ``inputs``: feature table(s) joined onto a label table by a join key.
    """

    name: str = Field(pattern=NAME_PATTERN)
    description: str = ""
    source: str | None = None  # "source('lakehouse', 'gold_subscribers')"
    inputs: DatasetInputs | None = None  # multi-table form
    label: LabelSpec
    filters: list[str] = Field(default_factory=list)  # SQL WHERE fragments, ANDed
    split: SplitSpec
    #: Stable row-identity column(s) used for deterministic hash sampling and
    #: seeded random splits. Strongly recommended for wide tables: sampling
    #: hashes only these columns instead of every column, and warehouse
    #: adapters push the predicate down into the source query.
    sample_key: str | list[str] | None = None
    checks: list[CheckSpec] = Field(default_factory=list)
    tests: list[str] = Field(default_factory=list)  # names of Python data tests that apply
    snapshot: str | None = None  # explicit pin; normally pinned at compile
    tags: list[str] = Field(default_factory=list)

    @property
    def sample_key_columns(self) -> list[str]:
        """Sampling identity: explicit sample_key, else the join key, else []."""
        if self.sample_key is not None:
            return [self.sample_key] if isinstance(self.sample_key, str) else list(self.sample_key)
        if self.inputs is not None:
            return self.inputs.join_columns
        return []

    @model_validator(mode="after")
    def _source_xor_inputs(self) -> "DatasetSpec":
        if (self.source is None) == (self.inputs is None):
            raise ValueError(
                "a dataset needs exactly one of 'source' (single table) or "
                "'inputs' (feature tables + label table with a join key)"
            )
        return self


class FeatureSelection(_SpecModel):
    """Feature include/exclude globs against the post-hook column set."""

    include: list[str] = Field(default_factory=lambda: ["*"])
    exclude: list[str] = Field(default_factory=list)


class GateSpec(_SpecModel):
    """A promotion-blocking metric condition (TSD §5.6, FR-TEST-02)."""

    metric: str
    threshold: float | None = None  # absolute gate
    compare_to: Stage | None = None  # champion gate vs registry stage
    min_delta: float = 0.0  # only meaningful with compare_to
    slice: str | None = None  # per-slice gate, "column=value" (FR-TEST-04)
    #: Champion gates: one-sided confidence for the paired-bootstrap lower
    #: bound of the delta (ADR-18); ``null`` opts back into point estimates.
    confidence: float | None = 0.95
    bootstrap_resamples: int = 1000

    @model_validator(mode="after")
    def _exactly_one_kind(self) -> "GateSpec":
        if (self.threshold is None) == (self.compare_to is None):
            raise ValueError("a gate must set exactly one of 'threshold' or 'compare_to'")
        if self.min_delta != 0.0 and self.compare_to is None:
            raise ValueError("'min_delta' is only meaningful with 'compare_to'")
        if self.compare_to is None:
            # Value-based (not fields_set) so dump/re-parse roundtrips, same
            # as the min_delta check above.
            if self.confidence != 0.95:
                raise ValueError("'confidence' is only meaningful with 'compare_to'")
            if self.bootstrap_resamples != 1000:
                raise ValueError("'bootstrap_resamples' is only meaningful with 'compare_to'")
        if self.confidence is not None and not 0.5 < self.confidence < 1.0:
            raise ValueError("'confidence' must be in (0.5, 1.0), e.g. 0.95")
        if self.bootstrap_resamples < 100:
            raise ValueError("'bootstrap_resamples' must be at least 100")
        return self


class EvaluationProtocol(_SpecModel):
    """How the model is evaluated; must match the dataset split (FR-RES-09)."""

    split: SplitStrategy = SplitStrategy.TEMPORAL
    test_window: str | None = None  # narrows the dataset test window


class EvaluationSpec(_SpecModel):
    """Metrics, gates, and slices for a model (TSD §5.6)."""

    protocol: EvaluationProtocol
    metrics: list[str]
    gates: list[GateSpec] = Field(default_factory=list)
    slices: list[str] = Field(default_factory=list)

    @model_validator(mode="after")
    def _metrics_nonempty(self) -> "EvaluationSpec":
        if not self.metrics:
            raise ValueError("evaluation.metrics must list at least one metric")
        return self


class SearchDimension(_SpecModel):
    """One hyperparameter search dimension (TSD §5.6)."""

    type: Literal["int", "uniform", "loguniform", "categorical"]
    low: float | None = None
    high: float | None = None
    choices: list[Any] | None = None

    @model_validator(mode="after")
    def _shape_for_type(self) -> "SearchDimension":
        if self.type == "categorical":
            if not self.choices:
                raise ValueError("categorical dimension requires non-empty 'choices'")
            if self.low is not None or self.high is not None:
                raise ValueError("categorical dimension takes 'choices', not 'low'/'high'")
        else:
            if self.low is None or self.high is None:
                raise ValueError(f"{self.type} dimension requires 'low' and 'high'")
            if self.choices is not None:
                raise ValueError(f"{self.type} dimension takes 'low'/'high', not 'choices'")
            if self.low >= self.high:
                raise ValueError("'low' must be strictly less than 'high'")
            if self.type == "loguniform" and self.low <= 0:
                raise ValueError("loguniform dimension requires 'low' > 0")
        return self


class TuningObjective(_SpecModel):
    """The metric a tuning run optimizes; must appear in evaluation.metrics."""

    metric: str
    direction: Literal["maximize", "minimize"]


class TuningSpec(_SpecModel):
    """Optional hyperparameter tuning block (FR-TUNE-01)."""

    engine: str = "optuna"
    n_trials: int = Field(gt=0)
    search_space: dict[str, SearchDimension]
    objective: TuningObjective
    #: Optional early stopping of unpromising trials: "median" prunes when a
    #: trial's intermediate validation value falls below the median of prior
    #: trials at the same step. Needs a training adapter that reports
    #: progress; otherwise trials run to completion (with a warning).
    pruner: Literal["median"] | None = None

    @model_validator(mode="after")
    def _space_nonempty(self) -> "TuningSpec":
        if not self.search_space:
            raise ValueError("tuning.search_space must not be empty")
        return self


class RegistrationSpec(_SpecModel):
    """Where a passing model registers (TSD §5.6, FR-REG-02)."""

    registry: str | None = None  # defaults to the target's registry adapter
    name: str
    stage_on_pass: Stage = Stage.STAGING


class ModelSpec(_SpecModel):
    """The model resource: the heart of mbt (TSD §5.6, FR-RES-03)."""

    name: str = Field(pattern=NAME_PATTERN)
    description: str = ""
    task: TaskType
    adapter: str
    owner: str  # email; required, shown on model cards
    tags: list[str] = Field(default_factory=list)
    dataset: str  # "ref('churn_training_set')"
    target: str  # must equal the dataset's label.column
    features: FeatureSelection = Field(default_factory=FeatureSelection)
    hyperparameters: dict[str, Any] = Field(default_factory=dict)
    tuning: TuningSpec | None = None
    evaluation: EvaluationSpec
    registration: RegistrationSpec | None = None
    materialization: Materialization = Materialization.MODEL_ARTIFACT
    seed: int  # mandatory, no default (FR-RES-03)
    hooks: str | None = None  # path to hooks.py; sibling <name>.py auto-detected

    @model_validator(mode="after")
    def _gate_and_objective_metrics_declared(self) -> "ModelSpec":
        declared = set(self.evaluation.metrics)
        declared_slices = set(self.evaluation.slices)
        for gate in self.evaluation.gates:
            if gate.metric not in declared:
                raise ValueError(f"gate metric '{gate.metric}' must appear in evaluation.metrics")
            if gate.slice is not None:
                column, _, value = gate.slice.partition("=")
                if not column or not value:
                    raise ValueError(
                        f"gate slice '{gate.slice}' must be 'column=value', "
                        "e.g. 'plan_type=premium'"
                    )
                if column not in declared_slices:
                    raise ValueError(
                        f"gate slice column '{column}' must appear in evaluation.slices"
                    )
        stages = {g.compare_to for g in self.evaluation.gates if g.compare_to is not None}
        if len(stages) > 1:
            raise ValueError("all champion gates of one model must compare_to the same stage in v0")
        if self.tuning is not None and self.tuning.objective.metric not in declared:
            raise ValueError(
                f"tuning objective metric '{self.tuning.objective.metric}' "
                "must appear in evaluation.metrics"
            )
        return self


class MetricSpec(_SpecModel):
    """A reusable metric definition from ``metrics.yml`` (TSD §5.7, FR-RES-04)."""

    name: str
    kind: Literal["builtin", "hook"] = "builtin"
    params: dict[str, Any] = Field(default_factory=dict)
    greater_is_better: bool = True


class ExposureSpec(_SpecModel):
    """A downstream consumer, for lineage and impact analysis (FR-RES-06)."""

    name: str = Field(pattern=NAME_PATTERN)
    type: Literal["endpoint", "batch_job", "dashboard", "other"]
    depends_on: list[str]  # ref() strings
    owner: str
    url: str | None = None
    description: str = ""


class ScoringInputs(_SpecModel):
    """Multi-table scoring input: a spine table plus feature tables.

    The spine defines which rows are scored; feature tables join onto it
    exactly like ``DatasetInputs`` feature tables join onto the label table.
    There is no label anywhere - scoring inputs are unlabeled by design.
    """

    spine: str  # source() ref that defines which rows are scored
    features: list[str]  # source() refs joined onto the spine
    join_key: str | list[str]
    join: Literal["left", "inner"] = "left"

    @property
    def join_columns(self) -> list[str]:
        return [self.join_key] if isinstance(self.join_key, str) else list(self.join_key)

    @model_validator(mode="after")
    def _shape(self) -> "ScoringInputs":
        if not self.features:
            raise ValueError("inputs.features must list at least one feature table")
        if not self.join_columns or any(not c for c in self.join_columns):
            raise ValueError("inputs.join_key must name at least one non-empty column")
        return self


class ScoringInputSpec(_SpecModel):
    """The unlabeled, unsplit batch a scoring pipeline reads (ADR-20).

    Data comes from exactly one of ``source`` (single table) or ``inputs``
    (spine + feature tables). The optional ``window`` is a window expression
    over ``time_column``, resolved against the manifest anchor like dataset
    split windows (ADR-12), so re-scoring is snapshot-driven, never
    clock-driven.
    """

    source: str | None = None  # "source('lakehouse', 'scoring_batch')"
    inputs: ScoringInputs | None = None
    filters: list[str] = Field(default_factory=list)  # SQL WHERE fragments, ANDed
    time_column: str | None = None
    window: str | None = None  # window expression, e.g. "-7d:now"
    sample_key: str | list[str] | None = None

    @property
    def sample_key_columns(self) -> list[str]:
        if self.sample_key is not None:
            return [self.sample_key] if isinstance(self.sample_key, str) else list(self.sample_key)
        if self.inputs is not None:
            return self.inputs.join_columns
        return []

    @model_validator(mode="after")
    def _shape(self) -> "ScoringInputSpec":
        if (self.source is None) == (self.inputs is None):
            raise ValueError(
                "a scoring input needs exactly one of 'source' (single table) or "
                "'inputs' (spine + feature tables with a join key)"
            )
        if self.window is not None and self.time_column is None:
            raise ValueError("'window' requires 'time_column'")
        return self


class FeatureShiftSpec(_SpecModel):
    """Feature distribution-shift monitor vs the training baseline (ADR-20)."""

    method: Literal["psi", "ks"] = "psi"
    threshold: float = Field(gt=0)  # per-feature; e.g. 0.2 for psi, 0.15 for ks
    include: list[str] = Field(default_factory=lambda: ["*"])
    exclude: list[str] = Field(default_factory=list)


class PredictionShiftSpec(_SpecModel):
    """Score distribution-shift monitor vs the test-split baseline (ADR-20)."""

    method: Literal["psi", "ks"] = "psi"
    threshold: float = Field(gt=0)


class MonitorsSpec(_SpecModel):
    """Distribution-shift monitors evaluated on every scoring run."""

    feature_shift: FeatureShiftSpec | None = None
    prediction_shift: PredictionShiftSpec | None = None


class GroundTruthLabelSpec(_SpecModel):
    """Where matured labels arrive and what they mean."""

    source: str  # source() ref to the matured-label table
    column: str
    definition: str = ""


class MonitorGateSpec(_SpecModel):
    """A realized-metric threshold for ground-truth evaluation.

    Threshold-only by design: a champion comparison is meaningless here
    because the champion IS the model that produced the predictions.
    """

    metric: str
    threshold: float


class GroundTruthSpec(_SpecModel):
    """Delayed ground-truth evaluation config, run by ``mbt monitor`` (ADR-21).

    ``maturity`` is a bare duration (``"14d"``): a prediction run is evaluated
    once ``scored_at + maturity`` lies at or before the monitor run's anchor.
    Metrics must be builtin - hook metrics need a training job.
    """

    label: GroundTruthLabelSpec
    join_key: str | list[str]
    maturity: str  # bare duration, e.g. "14d"
    metrics: list[str]
    gates: list[MonitorGateSpec] = Field(default_factory=list)

    @property
    def join_columns(self) -> list[str]:
        return [self.join_key] if isinstance(self.join_key, str) else list(self.join_key)

    @model_validator(mode="after")
    def _shape(self) -> "GroundTruthSpec":
        if not self.metrics:
            raise ValueError("ground_truth.metrics must list at least one metric")
        if not self.join_columns or any(not c for c in self.join_columns):
            raise ValueError("ground_truth.join_key must name at least one non-empty column")
        declared = set(self.metrics)
        for gate in self.gates:
            if gate.metric not in declared:
                raise ValueError(
                    f"ground_truth gate metric '{gate.metric}' must appear in ground_truth.metrics"
                )
        return self


class ScoringOutputSpec(_SpecModel):
    """Where predictions land (ADR-21). ``path`` is adapter-interpreted."""

    format: Literal["parquet"] = "parquet"
    path: str
    #: Extra passthrough columns copied from the RAW input into the output
    #: (identity/audit columns; ground-truth join keys are always included).
    columns: list[str] = Field(default_factory=list)


class ScoringSpec(_SpecModel):
    """One batch scoring (serving) pipeline: champion + input + sink (ADR-20).

    The referenced model's registered champion for ``stage`` is resolved at
    run time, so promotions take effect on the next scheduled run without a
    spec edit. Monitors compare the batch against the champion's
    training-time baseline; ``ground_truth`` adds delayed realized-metric
    evaluation via ``mbt monitor``.
    """

    name: str = Field(pattern=NAME_PATTERN)
    description: str = ""
    owner: str  # email; required, like models
    tags: list[str] = Field(default_factory=list)
    model: str  # "ref('churn_classifier')" - the DAG edge
    stage: Stage = Stage.PRODUCTION  # which champion alias to load
    input: ScoringInputSpec
    checks: list[CheckSpec] = Field(default_factory=list)
    monitors: MonitorsSpec | None = None
    ground_truth: GroundTruthSpec | None = None
    output: ScoringOutputSpec

    @property
    def passthrough_columns(self) -> list[str]:
        """Identity columns copied from the raw input into the output.

        Union of ``output.columns``, the ground-truth join key(s), and
        ``input.time_column``, in that order, deduplicated.
        """
        columns = list(self.output.columns)
        if self.ground_truth is not None:
            columns.extend(self.ground_truth.join_columns)
        if self.input.time_column is not None:
            columns.append(self.input.time_column)
        deduped: dict[str, None] = dict.fromkeys(columns)
        return list(deduped)

    @model_validator(mode="after")
    def _passthrough_nonempty(self) -> "ScoringSpec":
        if not self.passthrough_columns:
            raise ValueError(
                "predictions need at least one identity column: set output.columns, "
                "a ground_truth.join_key, or input.time_column"
            )
        return self
