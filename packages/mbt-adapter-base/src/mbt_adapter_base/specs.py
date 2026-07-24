"""Resource spec schemas (TSD §5.4-§5.8).

These are the Pydantic models behind the YAML resources users write.
They live in mbt-adapter-base because adapters consume them across the
contract boundary (``TrainingAdapter.validate(spec)``, ``DataAdapter.build_dataset(spec)``).

All schemas reject unknown fields (``extra="forbid"``, FR-PARSE-04); the
parser layer turns those rejections into did-you-mean suggestions.
"""

import re
from typing import Any, Literal, NamedTuple

from pydantic import BaseModel, ConfigDict, Field, model_validator

from mbt_adapter_base.types import Materialization, SplitStrategy, Stage, TaskType

#: Resource names: lowercase snake_case, so unique_ids stay unambiguous.
NAME_PATTERN = r"^[a-z][a-z0-9_]*$"

#: Label-join time offsets: a signed count plus a unit, where ``mo`` is a
#: calendar month (rendered as engine-native interval arithmetic) and
#: ``d``/``w``/``h`` are fixed durations (ADR-22).
_TIME_OFFSET_RE = re.compile(r"^(?P<sign>[+-])?(?P<value>\d+)(?P<unit>mo|d|w|h)$")


def parse_time_offset(offset: str) -> tuple[int, str]:
    """``"1mo"`` -> ``(1, "mo")``; raises ValueError on bad grammar."""
    match = _TIME_OFFSET_RE.match(offset.strip())
    if match is None:
        raise ValueError(
            f"invalid time_offset {offset!r}: expected '<count><unit>' with a "
            "unit of mo (calendar months), d, w, or h - e.g. '1mo' or '-28d'"
        )
    value = int(match.group("value"))
    if match.group("sign") == "-":
        value = -value
    return value, match.group("unit")


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
    path: str | None = None  # for path-based sources (parquet, delta)
    identifier: str | None = None  # for warehouse/feature-store sources (v1)
    #: parquet reads on every adapter; delta is spark-only. iceberg is roadmap,
    #: not implemented, so it is rejected here rather than silently mis-read (F23).
    format: Literal["parquet", "delta"] = "parquet"
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
    #: Temporal only (R2-7): drop this much of the train window's tail (a
    #: positive duration like "7d"/"1mo"), embargoing the boundary so a training
    #: row whose label horizon reaches the evaluation window cannot leak.
    embargo: str | None = None

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
            if self.embargo is not None:
                count, _ = parse_time_offset(self.embargo)
                if count <= 0:
                    raise ValueError(f"embargo must be a positive duration, got {self.embargo!r}")
        else:  # RANDOM
            if self.embargo is not None:
                raise ValueError("'embargo' applies to the temporal strategy only")
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


class FeatureEntry(NamedTuple):
    """One normalized feature-table join: source, join columns, projection.

    ``columns``/``exclude`` mirror :class:`FeatureInput`; at most one is set.
    """

    source: str
    using: list[str]
    columns: list[str] | None = None  # keep-list; join columns always kept
    exclude: list[str] | None = None  # drop-list

    @property
    def keep_columns(self) -> list[str] | None:
        """The full projected column list for a keep-list entry (join columns
        first, then payload, deduplicated), or None when unprojected."""
        if self.columns is None:
            return None
        return list(dict.fromkeys([*self.using, *self.columns]))


def _normalize_columns(value: str | list[str] | None) -> list[str] | None:
    if value is None:
        return None
    return [value] if isinstance(value, str) else list(value)


def _validate_projection(where: str, entry: FeatureEntry) -> None:
    """A drop-list must not remove the entry's own join columns (the join
    needs them); a keep-list may name them redundantly (deduplicated)."""
    if entry.exclude is None:
        return
    clash = sorted(set(entry.exclude) & set(entry.using))
    if clash:
        raise ValueError(
            f"{where} excludes its own join column(s) {clash}: a join column cannot be dropped"
        )


class FeatureInput(_SpecModel):
    """One feature table with its own USING-style join columns (ADR-22) and
    an optional per-table column projection (ADR-25).

    ``columns`` keeps ONLY the named payload columns (join columns are always
    kept); ``exclude`` drops the named columns and keeps the rest. At most one
    of the two may be set. The projection is pushed into the warehouse/lake
    query itself, so pruned columns of a huge gold table are never scanned
    into a training set or scoring batch - this is source-side workload
    reduction, distinct from the model's ``features.include/exclude`` which
    selects AFTER materialization.

    The join field is named ``using`` (not ``on``) deliberately: bare ``on``
    is a YAML 1.1 boolean, so PyYAML would hand pydantic a ``True`` key.
    """

    source: str  # source() ref
    using: str | list[str] | None = None  # join columns; default: the dataset join_key
    columns: str | list[str] | None = None  # keep-list of payload columns
    exclude: str | list[str] | None = None  # drop-list of columns

    @property
    def using_columns(self) -> list[str] | None:
        if self.using is None:
            return None
        return [self.using] if isinstance(self.using, str) else list(self.using)

    @property
    def columns_list(self) -> list[str] | None:
        return _normalize_columns(self.columns)

    @property
    def exclude_list(self) -> list[str] | None:
        return _normalize_columns(self.exclude)

    @model_validator(mode="after")
    def _shape(self) -> "FeatureInput":
        if self.using_columns is not None and (
            not self.using_columns or any(not c for c in self.using_columns)
        ):
            raise ValueError("'using' must name at least one non-empty column")
        if self.columns is not None and self.exclude is not None:
            raise ValueError(
                "'columns' (keep-list) and 'exclude' (drop-list) are mutually "
                "exclusive on a feature table - set at most one"
            )
        for field_name, values in (("columns", self.columns_list), ("exclude", self.exclude_list)):
            if values is not None and (not values or any(not c for c in values)):
                raise ValueError(f"'{field_name}' must name at least one non-empty column")
        return self


class LabelInput(_SpecModel):
    """The label table joined onto a population spine (ADR-22).

    ``time_offset`` shifts the spine's ``split.time_column`` when matching
    the label's same-named column (``label.ts = spine.ts + offset``), so an
    outcome observed one month after the prediction snapshot is declared as
    ``time_offset: "1mo"`` instead of pre-aligned upstream. The join-column
    field is ``using`` for the same YAML 1.1 reason as ``FeatureInput``.
    """

    source: str  # source() ref
    using: str | list[str] | None = None  # join columns; default: the dataset join_key
    time_offset: str | None = None  # e.g. "1mo"; calendar-aware (ADR-22)

    @property
    def using_columns(self) -> list[str] | None:
        if self.using is None:
            return None
        return [self.using] if isinstance(self.using, str) else list(self.using)

    @model_validator(mode="after")
    def _shape(self) -> "LabelInput":
        if self.using_columns is not None and (
            not self.using_columns or any(not c for c in self.using_columns)
        ):
            raise ValueError("'using' must name at least one non-empty column")
        if self.time_offset is not None:
            try:
                parse_time_offset(self.time_offset)
            except ValueError as exc:
                raise ValueError(str(exc)) from None
        return self


class DatasetInputs(_SpecModel):
    """Multi-table dataset construction (ADR-16, ADR-22).

    Without ``population``, the label table is the spine - it defines which
    examples exist - and feature tables join onto it. With ``population``,
    the population table is the spine and the label joins like a feature
    table (always ``inner``: an example without an observed outcome is not
    a training example), optionally shifted by ``time_offset``.

    Feature tables join in declaration order onto the accumulated relation
    (``left`` by default, so missing features arrive as NULLs; tree
    adapters handle those natively), each by its own ``on`` columns or the
    dataset-level ``join_key``. Column names must be unique across tables
    apart from each table's join columns.
    """

    features: list[str | FeatureInput]  # source() refs, or {source, on} mappings
    label: str | LabelInput  # source() ref; mapping form requires 'population'
    population: str | None = None  # source() ref to the spine table (ADR-22)
    join_key: str | list[str] | None = None  # default join columns
    join: Literal["left", "inner"] = "left"  # feature joins; the label join is inner

    @property
    def join_columns(self) -> list[str]:
        if self.join_key is None:
            return []
        return [self.join_key] if isinstance(self.join_key, str) else list(self.join_key)

    @property
    def label_source(self) -> str:
        return self.label if isinstance(self.label, str) else self.label.source

    @property
    def label_join_columns(self) -> list[str]:
        """The label's effective join columns (its ``using``, else ``join_key``)."""
        if isinstance(self.label, LabelInput) and self.label.using_columns is not None:
            return self.label.using_columns
        return self.join_columns

    @property
    def label_time_offset(self) -> str | None:
        return self.label.time_offset if isinstance(self.label, LabelInput) else None

    @property
    def spine(self) -> str:
        """The table that defines which examples exist."""
        return self.population if self.population is not None else self.label_source

    @property
    def feature_entries(self) -> list[FeatureEntry]:
        """Normalized feature-table joins in declaration order."""
        entries: list[FeatureEntry] = []
        for feature in self.features:
            if isinstance(feature, str):
                entries.append(FeatureEntry(feature, self.join_columns))
            else:
                entries.append(
                    FeatureEntry(
                        feature.source,
                        feature.using_columns or self.join_columns,
                        feature.columns_list,
                        feature.exclude_list,
                    )
                )
        return entries

    @property
    def feature_sources(self) -> list[str]:
        return [entry.source for entry in self.feature_entries]

    @model_validator(mode="after")
    def _shape(self) -> "DatasetInputs":
        if not self.features:
            raise ValueError("inputs.features must list at least one feature table")
        if self.join_key is not None and (
            not self.join_columns or any(not c for c in self.join_columns)
        ):
            raise ValueError("inputs.join_key must name at least one non-empty column")
        for i, entry in enumerate(self.feature_entries):
            if not entry.using:
                raise ValueError(
                    f"inputs.features[{i}] ({entry.source}) has no join columns: "
                    "give it 'using' or set a dataset-level 'join_key'"
                )
            _validate_projection(f"inputs.features[{i}] ({entry.source})", entry)
        if self.population is None:
            if isinstance(self.label, LabelInput):
                raise ValueError(
                    "the label mapping form ('using'/'time_offset') requires "
                    "a 'population' spine; without one the label table is the "
                    "spine and joins nothing"
                )
        elif not self.label_join_columns:
            raise ValueError(
                "with a 'population' spine the label needs join columns: "
                "give label 'using' or set a dataset-level 'join_key'"
            )
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
        """Sampling identity: explicit sample_key, else join_key, else the
        label's join columns, else []."""
        if self.sample_key is not None:
            return [self.sample_key] if isinstance(self.sample_key, str) else list(self.sample_key)
        if self.inputs is not None:
            return self.inputs.join_columns or self.inputs.label_join_columns
        return []

    @model_validator(mode="after")
    def _source_xor_inputs(self) -> "DatasetSpec":
        if (self.source is None) == (self.inputs is None):
            raise ValueError(
                "a dataset needs exactly one of 'source' (single table) or "
                "'inputs' (feature tables + label table with a join key)"
            )
        offset = self.inputs.label_time_offset if self.inputs is not None else None
        if offset is not None:
            assert self.inputs is not None
            if self.split.time_column is None:
                raise ValueError(
                    "label time_offset shifts the split's 'time_column'; "
                    "this dataset's split declares none"
                )
            if self.split.time_column not in self.inputs.label_join_columns:
                raise ValueError(
                    f"label time_offset shifts the split time_column "
                    f"{self.split.time_column!r}, so it must be one of the "
                    f"label's join columns {self.inputs.label_join_columns!r}"
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
    across: str | None = None  # disparity gate: slice COLUMN to measure parity across
    min_delta: float = 0.0  # only meaningful with compare_to
    #: Disparity gates: the minimum acceptable worst/best slice ratio (min/max
    #: of the metric across the ``across`` column's values), in (0, 1] where
    #: 1.0 is perfect parity. Only meaningful with ``across``.
    min_ratio: float = 0.8
    slice: str | None = None  # per-slice gate, "column=value" (FR-TEST-04)
    #: Champion gates: one-sided confidence for the paired-bootstrap lower
    #: bound of the delta (ADR-18); ``null`` opts back into point estimates.
    confidence: float | None = 0.95
    bootstrap_resamples: int = 1000
    #: Metric source (R2-7): ``test`` gates the single held-out test window;
    #: ``backtest`` gates the walk-forward mean (needs ``protocol.backtest_folds``).
    #: NOT named ``on`` - that is a YAML 1.1 boolean (see FeatureInput.using).
    source: Literal["test", "backtest"] = "test"

    @model_validator(mode="after")
    def _exactly_one_kind(self) -> "GateSpec":
        kinds = (self.threshold is not None, self.compare_to is not None, self.across is not None)
        if sum(kinds) != 1:
            raise ValueError(
                "a gate must set exactly one of 'threshold', 'compare_to', or 'across'"
            )
        if self.source == "backtest" and (self.threshold is None or self.slice is not None):
            raise ValueError(
                "a backtest gate (source: backtest) must be a whole-split threshold gate: "
                "the walk-forward backtest reports only mean metrics, not champion deltas or slices"
            )
        if self.across is not None and self.slice is not None:
            raise ValueError("a disparity gate ('across') measures a whole column, not a 'slice'")
        if self.min_delta != 0.0 and self.compare_to is None:
            raise ValueError("'min_delta' is only meaningful with 'compare_to'")
        if self.min_ratio != 0.8 and self.across is None:
            raise ValueError("'min_ratio' is only meaningful with 'across'")
        if self.across is not None and not 0.0 < self.min_ratio <= 1.0:
            raise ValueError("'min_ratio' must be in (0, 1], e.g. 0.8")
        if self.across is not None and self.metric == "r2":
            # r2 is the one builtin metric that can be negative, so the disparity
            # gate's worst/best RATIO is ill-defined: two negative slices invert
            # it (-0.9 / -0.1 = 9.0 reads as parity) and a mixed-sign pair makes
            # it negative. Reject at parse rather than gate on a wrong number.
            raise ValueError(
                "a disparity gate ('across') on 'r2' is not supported: r2 can be "
                "negative, so the worst/best ratio is ill-defined; gate a "
                "non-negative regression metric like 'rmse' or 'mae' across the "
                "column instead"
            )
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
    #: Optional cross-validated backtest (R2-7): the training window is split
    #: into N folds and the model is refit and evaluated on each - time-ordered
    #: walk-forward for a temporal split, random k-fold for a random split - so a
    #: single lucky split cannot flatter the reported generalization.
    backtest_folds: int | None = Field(default=None, ge=2)
    #: Nested cross-validation (R2-7): re-tune within each backtest fold, so the
    #: reported fold mean is an UNBIASED estimate of the TUNED model - the tuning
    #: never sees the fold it is evaluated on (temporal walk-forward or random
    #: k-fold, per the split). Needs backtest_folds and (on the model) a tuning block.
    nested_cv: bool = False

    @model_validator(mode="after")
    def _nested_cv_requirements(self) -> "EvaluationProtocol":
        if self.nested_cv and self.backtest_folds is None:
            raise ValueError("nested_cv needs backtest_folds (the outer fold count)")
        return self


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
    #: Select on the bootstrap lower bound of the validation metric, not the
    #: point estimate (R2-7): defends the tuning selection against
    #: validation-window luck, the same idea ADR-18 applies to the champion gate.
    #: Builtin metric only. Off by default (unchanged single-split selection).
    robust: bool = False


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
    #: Post-hoc probability calibration (R2-8); binary classification only.
    #: Fit on a dedicated slice core carves from train (seed+5, F17), so it
    #: composes with tuning, early stopping, and the walk-forward backtest
    #: (each fold carves its own slice, F5). Adapter support is probed at parse.
    calibration: Literal["isotonic", "sigmoid"] | None = None

    @model_validator(mode="after")
    def _nested_cv_needs_tuning(self) -> "ModelSpec":
        if self.evaluation.protocol.nested_cv and self.tuning is None:
            raise ValueError("nested_cv re-tunes within each fold, so it needs a 'tuning' block")
        return self

    @model_validator(mode="after")
    def _calibration_is_binary_only(self) -> "ModelSpec":
        if self.calibration is not None and self.task != TaskType.BINARY_CLASSIFICATION:
            raise ValueError(
                "calibration applies to binary_classification only "
                "(it recalibrates predicted probabilities)"
            )
        return self

    @model_validator(mode="after")
    def _gate_and_objective_metrics_declared(self) -> "ModelSpec":
        declared = set(self.evaluation.metrics)
        declared_slices = set(self.evaluation.slices)
        for gate in self.evaluation.gates:
            if gate.metric not in declared:
                raise ValueError(f"gate metric '{gate.metric}' must appear in evaluation.metrics")
            if gate.source == "backtest" and self.evaluation.protocol.backtest_folds is None:
                raise ValueError(
                    f"gate on '{gate.metric}' uses source: backtest but "
                    "evaluation.protocol.backtest_folds is not set"
                )
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

    The spine defines which rows are scored - for a population-spine
    dataset (ADR-22) it is the same population table, minus the label.
    Feature tables join onto the accumulated relation in declaration order,
    each by its own ``on`` columns or the shared ``join_key``, exactly like
    ``DatasetInputs`` feature tables. There is no label anywhere - scoring
    inputs are unlabeled by design.
    """

    spine: str  # source() ref that defines which rows are scored
    features: list[str | FeatureInput]  # source() refs, or {source, on} mappings
    join_key: str | list[str] | None = None  # default join columns
    join: Literal["left", "inner"] = "left"

    @property
    def join_columns(self) -> list[str]:
        if self.join_key is None:
            return []
        return [self.join_key] if isinstance(self.join_key, str) else list(self.join_key)

    @property
    def feature_entries(self) -> list[FeatureEntry]:
        """Normalized feature-table joins in declaration order."""
        entries: list[FeatureEntry] = []
        for feature in self.features:
            if isinstance(feature, str):
                entries.append(FeatureEntry(feature, self.join_columns))
            else:
                entries.append(
                    FeatureEntry(
                        feature.source,
                        feature.using_columns or self.join_columns,
                        feature.columns_list,
                        feature.exclude_list,
                    )
                )
        return entries

    @property
    def feature_sources(self) -> list[str]:
        return [entry.source for entry in self.feature_entries]

    @model_validator(mode="after")
    def _shape(self) -> "ScoringInputs":
        if not self.features:
            raise ValueError("inputs.features must list at least one feature table")
        if self.join_key is not None and (
            not self.join_columns or any(not c for c in self.join_columns)
        ):
            raise ValueError("inputs.join_key must name at least one non-empty column")
        for i, entry in enumerate(self.feature_entries):
            if not entry.using:
                raise ValueError(
                    f"inputs.features[{i}] ({entry.source}) has no join columns: "
                    "give it 'on' or set an inputs-level 'join_key'"
                )
            _validate_projection(f"inputs.features[{i}] ({entry.source})", entry)
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


def _validate_shift_significance(
    significance: float | None, method: str, warn_threshold: float | None
) -> None:
    """Shared rule for the shift monitors' n-aware significance (R2-6): it
    rides on ``method: ks`` and is a principled bar that does not combine with
    an absolute warn band. The bar is kind-matched at evaluation time (F15):
    numeric features get the two-sample KS critical value, categorical
    features a two-sample (contingency) chi-square statistic judged at the
    chi-square critical value."""
    if significance is None:
        return
    if method != "ks":
        raise ValueError("shift significance requires 'method: ks' (it is a KS critical value)")
    if warn_threshold is not None:
        raise ValueError("shift significance and warn_threshold are mutually exclusive")


class FeatureShiftSpec(_SpecModel):
    """Feature distribution-shift monitor vs the training baseline (ADR-20)."""

    method: Literal["psi", "ks"] = "psi"
    threshold: float = Field(gt=0)  # per-feature fail bar; e.g. 0.2 psi, 0.15 ks
    #: Optional warn band: a shift in ``(warn_threshold, threshold]`` logs a
    #: warning without failing the run - a two-tier bar like label_leakage_scan.
    warn_threshold: float | None = Field(default=None, gt=0)
    #: Optional n-aware significance (R2-6): with ``method: ks``, the fail bar
    #: becomes a critical value at this p-value instead of the fixed
    #: ``threshold``, so it tightens on large nightly batches and loosens on
    #: small ones. Kind-matched (F15): numeric features use the two-sample KS
    #: critical value (sup over the merged baseline-quantile + current points);
    #: categorical features a two-sample contingency chi-square judged at the
    #: chi-square critical value. Excludes warn_threshold.
    significance: float | None = Field(default=None, gt=0.0, lt=1.0)
    include: list[str] = Field(default_factory=lambda: ["*"])
    exclude: list[str] = Field(default_factory=list)

    @model_validator(mode="after")
    def _warn_below_fail(self) -> "FeatureShiftSpec":
        if self.warn_threshold is not None and self.warn_threshold >= self.threshold:
            raise ValueError("feature_shift warn_threshold must be below threshold (the fail bar)")
        _validate_shift_significance(self.significance, self.method, self.warn_threshold)
        return self


class PredictionShiftSpec(_SpecModel):
    """Score distribution-shift monitor vs the test-split baseline (ADR-20)."""

    method: Literal["psi", "ks"] = "psi"
    threshold: float = Field(gt=0)
    #: Optional warn band, as in FeatureShiftSpec.
    warn_threshold: float | None = Field(default=None, gt=0)
    #: Optional n-aware KS significance (R2-6), as in FeatureShiftSpec.
    significance: float | None = Field(default=None, gt=0.0, lt=1.0)

    @model_validator(mode="after")
    def _warn_below_fail(self) -> "PredictionShiftSpec":
        if self.warn_threshold is not None and self.warn_threshold >= self.threshold:
            raise ValueError(
                "prediction_shift warn_threshold must be below threshold (the fail bar)"
            )
        _validate_shift_significance(self.significance, self.method, self.warn_threshold)
        return self


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
    #: The deployable operating point (R2-5): when set, scoring emits a 0/1
    #: ``decision`` column (``prediction >= decision_threshold``) alongside the
    #: probability, and records the cutoff in the run info, so consumers get a
    #: decision rule instead of re-deriving one out of band. A float is used
    #: verbatim; a string names one of the champion's operating-point metrics
    #: (``threshold_at_precision_<p>`` / ``threshold_at_recall_<r>``), resolved
    #: from the registered champion at score time so the cutoff tracks the model.
    decision_threshold: float | str | None = None
    #: Per-prediction local explanation (explainability): when set, scoring emits
    #: an ``explanation`` column naming the top-N features by |SHAP| for each row
    #: (a JSON ``[[feature, contribution], ...]``), so a consumer can answer "why
    #: did THIS row score this way". Requires an adapter that supports SHAP
    #: explanations (the tree adapters); others fail with an actionable error.
    explain_top_k: int | None = Field(default=None, gt=0)

    @model_validator(mode="after")
    def _validate_decision_threshold(self) -> "ScoringOutputSpec":
        value = self.decision_threshold
        if isinstance(value, float) and not 0.0 <= value <= 1.0:
            raise ValueError("a numeric decision_threshold must be in [0, 1]")
        if isinstance(value, str) and not value.startswith(
            ("threshold_at_precision_", "threshold_at_recall_")
        ):
            raise ValueError(
                "a string decision_threshold must name a champion operating-point metric "
                "(threshold_at_precision_<p> or threshold_at_recall_<r>)"
            )
        return self


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
