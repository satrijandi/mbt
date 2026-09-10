"""Resource spec schemas (TSD §5.4-§5.8).

These are the Pydantic models behind the YAML resources users write.
They live in mbt-adapter-base because adapters consume them across the
contract boundary (``TrainingAdapter.validate(spec)``, ``DataAdapter.build_dataset(spec)``).

All schemas reject unknown fields (``extra="forbid"``, FR-PARSE-04); the
parser layer turns those rejections into did-you-mean suggestions.
"""

import re
from typing import Any, Literal

from pydantic import BaseModel, ConfigDict, Field, field_validator, model_validator

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
    #: DECLARATIVE ONLY: this joins nothing and shifts nothing. It states how
    #: long after the prediction date the outcome is observed, so mbt can check
    #: that ``split.embargo`` and ``ground_truth.maturity`` agree with it and
    #: put the number on the model card (ADR-29).
    #:
    #: ADR-22 argued that pre-aligning label dates upstream "hides the outcome
    #: window, the exact thing a training-set definition should state". ADR-29
    #: moved the join to the gold layer, so the alignment is upstream now; this
    #: field keeps the statement without the mechanism. Bare duration ("1mo",
    #: "-28d", "2w", "12h"), same grammar as ``time_offset``.
    horizon: str | None = None

    @model_validator(mode="after")
    def _valid_horizon(self) -> "LabelSpec":
        if self.horizon is not None:
            parse_time_offset(self.horizon)
        return self


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


class DatasetSpec(_SpecModel):
    """Declarative training-set construction (TSD §5.5, FR-RES-02).

    ``source`` names ONE relation holding the features and the label: a table,
    a view, or a warehouse dynamic table. Whatever assembles it - joining a
    population to its labels and feature histories - is upstream and mbt does
    not model it (ADR-29). What mbt owns is everything below: which rows, which
    split, which checks, and the statement of what the training set is.
    """

    name: str = Field(pattern=NAME_PATTERN)
    description: str = ""
    source: str  # "source('lakehouse', 'ml_churn_panel')"
    #: The panel contract (ADR-29): the exact column set this dataset expects
    #: from its relation. An undeclared column fails the build, and so does a
    #: declared column that is absent.
    #:
    #: It asserts; it never selects. When the join lives upstream in dbt, this
    #: list is the reviewed record of what the training set IS - without it, a
    #: feature arriving upstream is indistinguishable from a routine data
    #: refresh, and adding one would not be a diff anywhere a reviewer looks.
    #: Distinct from the model's ``features.include``, which selects what one
    #: model uses out of the panel this declares.
    columns: list[str] | None = None
    label: LabelSpec
    filters: list[str] = Field(default_factory=list)  # SQL WHERE fragments, ANDed
    split: SplitSpec
    #: Stable row-identity column(s) used for deterministic hash sampling and
    #: seeded random splits, pushed down into the source query. Required: with
    #: no declared identity the digest hashes every column, so the column list
    #: becomes the hash preimage and one column arriving upstream re-buckets
    #: every row (ADR-29). Use the ENTITY id on a panel whose rows repeat per
    #: entity - hashing it keeps all of an entity's rows on one side of the
    #: split, which is what makes a sampled dev run comparable to a full one.
    sample_key: str | list[str]
    checks: list[CheckSpec] = Field(default_factory=list)
    tests: list[str] = Field(default_factory=list)  # names of Python data tests that apply
    snapshot: str | None = None  # explicit pin; normally pinned at compile
    tags: list[str] = Field(default_factory=list)

    @property
    def sample_key_columns(self) -> list[str]:
        """The sampling identity, always as a list."""
        return [self.sample_key] if isinstance(self.sample_key, str) else list(self.sample_key)

    @model_validator(mode="after")
    def _panel_contract(self) -> "DatasetSpec":
        if not self.sample_key_columns or not all(self.sample_key_columns):
            raise ValueError("'sample_key' must name at least one non-empty column")
        if self.columns is not None:
            if not self.columns:
                raise ValueError(
                    "'columns' declares the panel's expected column set, so an "
                    "empty list asserts an empty panel; omit it instead"
                )
            duplicates = sorted({c for c in self.columns if self.columns.count(c) > 1})
            if duplicates:
                raise ValueError(f"'columns' repeats: {', '.join(duplicates)}")
            if self.label.column not in self.columns:
                raise ValueError(
                    f"'columns' must include the label column "
                    f"{self.label.column!r}; it is part of the panel"
                )
            if self.split.time_column is not None and self.split.time_column not in self.columns:
                raise ValueError(
                    f"'columns' must include the split time column "
                    f"{self.split.time_column!r}; the split reads it"
                )
            missing_keys = [c for c in self.sample_key_columns if c not in self.columns]
            if missing_keys:
                raise ValueError(
                    f"'columns' must include the sample_key column(s) "
                    f"{', '.join(missing_keys)}; sampling hashes them"
                )
        return self


#: The pooled level a rare or undeclared categorical value maps to (ADR-27).
OTHER_LEVEL = "__other__"

#: The explicit level NULLs map to under ``null_as_level`` (ADR-27).
MISSING_LEVEL = "__missing__"

#: Monotone constraint directions (ADR-27); words, not the frameworks' +-1.
MonotonicDirection = Literal["increasing", "decreasing"]


class CapSpec(_SpecModel):
    """A two-sided plateau cap (ADR-27); the scalar form ``cap: 365`` is an
    upper plateau and normalizes to ``max``."""

    min: float | None = None
    max: float | None = None

    @model_validator(mode="after")
    def _shape(self) -> "CapSpec":
        if self.min is None and self.max is None:
            raise ValueError("'cap' must set 'min', 'max', or both")
        if self.min is not None and self.max is not None and self.min >= self.max:
            raise ValueError(f"cap min ({self.min}) must be below cap max ({self.max})")
        return self


class CategoricalPolicy(_SpecModel):
    """How one declared categorical column is levelled (ADR-27).

    Every field is optional; ``categorical: [plan_type]`` is sugar for
    ``categorical: {plan_type: {}}``, which declares the column categorical
    and takes the default treatment (levels learned from the train split,
    NULLs left to the framework's missing branch).
    """

    #: Pin the level set in the spec instead of learning it from train, so a
    #: new production value is a declared, hash-visible change rather than a
    #: silent unseen-level-to-missing. Values are compared as strings.
    levels: list[str | int | bool] | None = None
    #: Pool train levels below this share into ``__other__``, so a long tail
    #: of one-off values cannot each become a split. Needs the train-fitted
    #: level map, so it is supported only by the adapters that share
    #: ``mbt_adapter_base.encoding`` (probed at parse).
    min_frequency: float | None = Field(default=None, gt=0.0, lt=1.0)
    #: Fail when the column has more distinct train levels than this - the
    #: guard that catches a user_id declared categorical by accident.
    max_levels: int | None = Field(default=None, ge=2)
    #: Map NULL to an explicit ``__missing__`` level, so "we don't know" is a
    #: category the model can learn from rather than a missing-branch fallthrough.
    null_as_level: bool = False

    @property
    def level_strings(self) -> list[str] | None:
        """``levels`` as the strings the encoders compare against."""
        if self.levels is None:
            return None
        return [str(v) for v in self.levels]

    @model_validator(mode="after")
    def _shape(self) -> "CategoricalPolicy":
        if self.levels is not None:
            if not self.levels:
                raise ValueError("'levels' must name at least one level, or be omitted")
            seen = [str(v) for v in self.levels]
            duplicates = sorted({v for v in seen if seen.count(v) > 1})
            if duplicates:
                raise ValueError(f"'levels' repeats {duplicates}: levels compare as strings")
            reserved = sorted(set(seen) & {OTHER_LEVEL, MISSING_LEVEL})
            if reserved:
                raise ValueError(
                    f"'levels' may not name the reserved level(s) {reserved}: mbt "
                    "adds them itself for pooled and missing values"
                )
            if self.min_frequency is not None:
                raise ValueError(
                    "'levels' and 'min_frequency' both decide the level set - set at "
                    "most one ('levels' pins it in the spec, 'min_frequency' learns it "
                    "from the train split)"
                )
            if self.max_levels is not None and len(self.levels) > self.max_levels:
                raise ValueError(
                    f"'levels' names {len(self.levels)} levels, above max_levels "
                    f"({self.max_levels})"
                )
        return self


class FeatureTransform(_SpecModel):
    """Drift-mitigating treatment of one numeric feature (ADR-27).

    Applied in core, to the same table at train and at score time, in the
    fixed order ``cap`` -> ``log`` -> ``percentile``. Every step is stateless
    by construction: no reference distribution is fitted or persisted, so the
    two sides agree without a side-car artifact.
    """

    #: Plateau cap: ``365`` clamps above, ``{min: 0, max: 365}`` clamps both
    #: ends. A declared constant, never a fitted quantile - a train-fitted
    #: ``p99`` would need persisted state (see ADR-27's rejected alternatives).
    #: The scalar shorthand normalizes to ``{max: N}`` before validation, so a
    #: malformed cap reports CapSpec's error rather than a union branch the
    #: user never wrote.
    cap: CapSpec | None = None
    #: ``log1p`` compression of the tail, so a doubling of the raw value moves
    #: the feature by a constant instead of proportionally.
    log: bool = False
    #: Rank the value within the split or batch being read, into [0, 1]. This
    #: is the self-normalizing lever: a uniform population shift cancels,
    #: because every batch is ranked against itself. ``batch`` is the only
    #: member today; ``train`` is the reserved name for the fitted variant.
    percentile: Literal["batch"] | None = None
    #: Constrain the model's response to this feature to be monotone. Not a
    #: data rewrite - it reaches the adapter as a booster constraint, and is
    #: equivalent to naming the column in ``features.monotonic``.
    monotonic: MonotonicDirection | None = None

    @field_validator("cap", mode="before")
    @classmethod
    def _scalar_cap_is_an_upper_plateau(cls, value: Any) -> Any:
        if isinstance(value, bool) or not isinstance(value, int | float):
            return value
        return {"max": float(value)}

    @property
    def rewrites_values(self) -> bool:
        """Whether this entry changes the data (as opposed to only constraining
        the model), which is what decides if the column must be numeric."""
        return self.cap is not None or self.log or self.percentile is not None

    @model_validator(mode="after")
    def _shape(self) -> "FeatureTransform":
        if self.log and self.percentile is not None:
            raise ValueError(
                "'log' has no effect alongside 'percentile': a rank is invariant "
                "under any monotone transform, so drop 'log'"
            )
        if not self.rewrites_values and self.monotonic is None:
            raise ValueError(
                "a transform entry must set at least one of 'cap', 'log', "
                "'percentile', or 'monotonic'"
            )
        return self


class FeatureSelection(_SpecModel):
    """Which columns a model consumes and how each is treated.

    ``include``/``exclude`` are globs against the post-hook column set and
    decide membership; ``categorical``, ``transforms`` and ``monotonic`` are
    per-column treatment of the columns that survive (ADR-27). The three
    treatment blocks are kept apart because they act at three different
    seams: ``categorical`` retypes a column, ``transforms`` rewrites its
    values, and ``monotonic`` constrains the model rather than the data.
    """

    include: list[str] = Field(default_factory=lambda: ["*"])
    exclude: list[str] = Field(default_factory=list)
    #: DS-declared categoricals. **Absent** keeps dtype inference (string
    #: columns are categorical, everything else numeric). **Present** is
    #: authoritative: listed columns are categorical even when int-coded, and
    #: an undeclared string feature is an error rather than a silent guess.
    #: ``[]`` is therefore a meaningful assertion that the model has none.
    #: The bare-list shorthand normalizes to ``{name: {}}`` before validation
    #: (same reason as ``FeatureTransform.cap``).
    categorical: dict[str, CategoricalPolicy] | None = None
    #: Per-column numeric treatment, keyed by column name.
    transforms: dict[str, FeatureTransform] = Field(default_factory=dict)
    #: Monotone constraints, keyed by column name. The same constraint may be
    #: written inline under ``transforms``; both spellings resolve to one map.
    monotonic: dict[str, MonotonicDirection] = Field(default_factory=dict)

    @property
    def declares_categorical(self) -> bool:
        """Whether the spec takes over from dtype inference (``[]`` counts)."""
        return self.categorical is not None

    @property
    def categorical_policies(self) -> dict[str, CategoricalPolicy]:
        """The declared categoricals as column -> policy; empty when the block
        is absent (which means "infer", not "none" - see ``declares_categorical``)."""
        return {} if self.categorical is None else dict(self.categorical)

    @property
    def monotonic_constraints(self) -> dict[str, MonotonicDirection]:
        """The canonical constraint map, merging the inline ``transforms``
        spelling with the standalone ``monotonic`` block."""
        merged: dict[str, MonotonicDirection] = {
            name: entry.monotonic
            for name, entry in self.transforms.items()
            if entry.monotonic is not None
        }
        merged.update(self.monotonic)
        return merged

    @property
    def treated_columns(self) -> list[str]:
        """Every column named by a treatment block, in a stable order."""
        names = [
            *self.categorical_policies,
            *self.transforms,
            *self.monotonic,
        ]
        return list(dict.fromkeys(names))

    @field_validator("categorical", mode="before")
    @classmethod
    def _bare_list_takes_default_policies(cls, value: Any) -> Any:
        if not isinstance(value, list):
            return value
        names = [v for v in value if isinstance(v, str)]
        if len(names) != len(value):
            return value  # let the dict-shaped error report the real problem
        duplicates = sorted({n for n in names if names.count(n) > 1})
        if duplicates:
            raise ValueError(f"'categorical' repeats {duplicates}")
        return {name: {} for name in names}

    @model_validator(mode="after")
    def _treatments_are_coherent(self) -> "FeatureSelection":
        for field_name, names in (
            ("categorical", list(self.categorical_policies)),
            ("transforms", list(self.transforms)),
            ("monotonic", list(self.monotonic)),
        ):
            if any(not name for name in names):
                raise ValueError(f"'{field_name}' must name non-empty columns")
        categorical = set(self.categorical_policies)
        rewritten = {name for name, entry in self.transforms.items() if entry.rewrites_values}
        clash = sorted(categorical & rewritten)
        if clash:
            raise ValueError(
                f"column(s) {clash} are declared categorical and also given a numeric "
                "transform (cap/log/percentile) - a category has no magnitude to cap, "
                "scale, or rank"
            )
        constrained = sorted(categorical & set(self.monotonic_constraints))
        if constrained:
            raise ValueError(
                f"monotone constraint on categorical column(s) {constrained}: the "
                "levels are unordered, so 'increasing'/'decreasing' has no meaning"
            )
        for name, direction in self.monotonic.items():
            inline = self.transforms.get(name)
            if inline is not None and inline.monotonic not in (None, direction):
                raise ValueError(
                    f"column '{name}' declares monotonic '{inline.monotonic}' under "
                    f"'transforms' and '{direction}' under 'monotonic' - they must agree"
                )
        # A treated column that the model never sees is always a mistake, and
        # 'exclude' is the one membership rule knowable without the data.
        excluded = sorted(set(self.treated_columns) & set(self.exclude))
        if excluded:
            raise ValueError(
                f"column(s) {excluded} are treated under 'categorical'/'transforms'/"
                "'monotonic' but also listed in 'exclude', so the model never sees them"
            )
        return self


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
    #: NOT named ``on`` - that is a YAML 1.1 boolean, so PyYAML would hand
    #: pydantic a ``True`` key.
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


class ScoringInputSpec(_SpecModel):
    """The unlabeled, unsplit batch a scoring pipeline reads (ADR-20).

    ``source`` names ONE relation, the serving twin of the training panel:
    identical features, no label (ADR-29). The two are separate upstream
    objects that must stay in lockstep, which is why the champion's recorded
    feature columns are enforced here at score time.

    The optional ``window`` is a window expression over ``time_column``,
    resolved against the manifest anchor like dataset split windows (ADR-12),
    so re-scoring is snapshot-driven, never clock-driven. Its END bound also
    says how old the freshest scored row can be, which is what relates
    ``ground_truth.maturity`` to the dataset's ``label.horizon``.
    """

    source: str  # "source('lakehouse', 'ml_churn_panel_scoring')"
    filters: list[str] = Field(default_factory=list)  # SQL WHERE fragments, ANDed
    time_column: str | None = None
    window: str | None = None  # window expression, e.g. "-7d:now"
    sample_key: str | list[str] | None = None

    @property
    def sample_key_columns(self) -> list[str]:
        if self.sample_key is None:
            return []
        return [self.sample_key] if isinstance(self.sample_key, str) else list(self.sample_key)

    @model_validator(mode="after")
    def _shape(self) -> "ScoringInputSpec":
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
