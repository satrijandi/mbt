"""The project's cross-resource invariants, as an addressable list (A-3).

Nineteen `_check_*` / `_validate_*` functions used to live in
``project_parser.py``, wired into ``_link_and_check`` - which was not a
dispatcher but a transcript: a fixed sequence of calls that nothing else could
reach. That shape drove the parser's churn (it is the repo's most-edited source
file): every ADR adding an invariant edited a new function, a call line inside
the transcript, and often ``_validate_dataset_windows`` too.

It also left a hole. ``compile/compiler.py`` discards ``res.spec`` and
re-renders ``res.raw`` with TARGET vars before re-validating from scratch, which
is deliberate and necessary - the capture phase cannot see target scope, so a
``var()`` used inside a spec can genuinely differ per target (ADR-5). But of the
nineteen rules, compile re-ran exactly ONE (``validate_hyperparameters``). So a
target var that moved ``spec.target``, ``evaluation.protocol.test_window`` or a
gate's ``source`` passed ``mbt parse`` and reached execution unchecked.

So the rules are a list here, and ``run_rules`` runs it twice: once at parse
over the captured specs, once at compile over the RESOLVED ones. The hole
closes by construction rather than by remembering to re-run each rule.

**Adding an invariant** is one function plus one ``RULES`` entry. It runs in
both phases automatically, and ``test_parse_rules_unit.py`` asserts that every
entry is reachable and that the two phases run the same set.
"""

import re
from dataclasses import dataclass
from typing import Any, Literal, Protocol

from mbt.adapters.registry import AdapterRegistry
from mbt.compile.windows import VALIDATION_ANCHOR, is_subrange, parse_window
from mbt.config.tasks import get_task_schema
from mbt.exceptions import ConfigError
from mbt.parsing.errors import ParseReport
from mbt.parsing.loader import validate_resource
from mbt.quality.check_names import BUILTIN_CHECK_NAMES, SCORING_CHECK_NAMES
from mbt.utils import did_you_mean
from mbt_adapter_base import (
    AUTO,
    OUT_OF_TIME_SPLIT,
    DatasetSpec,
    ModelSpec,
    ScoringSpec,
    SplitStrategy,
    TaskType,
)
from mbt_adapter_base.capabilities import Capability, capabilities_of
from mbt_adapter_base.specs import parse_time_offset

_SOURCE_RE = re.compile(r"^\s*source\(\s*['\"][^'\"]+['\"]\s*,\s*['\"][^'\"]+['\"]\s*\)\s*$")
_REF_RE = re.compile(r"^\s*ref\(\s*['\"](?P<name>[^'\"]+)['\"]\s*\)\s*$")

#: Built-in checks a user may declare, from the single authoritative source
#: (mbt.quality.check_names). Importing the NAMES - not the implementations -
#: keeps duckdb/pyarrow off the parse path (ADR-14).
_BUILTIN_CHECKS = BUILTIN_CHECK_NAMES

#: Checks valid on a scoring input: no label exists, so label-dependent
#: checks are rejected (ADR-20).
_SCORING_CHECKS = SCORING_CHECK_NAMES

#: Nominal days per duration unit, for comparing two declared durations only.
#: Calendar months are not 30 days, but this never resolves a window - it only
#: answers "is this embargo shorter than that horizon", where being approximate
#: at the boundary is fine and being absent is not.
_NOMINAL_DAYS = {"h": 1 / 24, "d": 1.0, "w": 7.0, "mo": 30.0}


def _duration_days(duration: str) -> float:
    value, unit = parse_time_offset(duration)
    return abs(value) * _NOMINAL_DAYS[unit]


def _newest_row_age_days(window: str | None) -> float:
    """How old the freshest row in a scoring batch is, at minimum, when scored.

    Read off the window's END bound: ``"-31d:-28d"`` cannot contain anything
    newer than 28 days, while an end of ``now`` (or no window at all) admits a
    row scored on its own inference date. An absolute end is anchor-relative
    and unknowable here, so it counts as zero - the conservative direction.
    """
    if window is None:
        return 0.0
    try:
        end = parse_window(window).end
    except ConfigError:
        return 0.0  # reported by the scoring.windows rule in the same pass
    if end.kind != "duration" or end.delta is None:
        return 0.0
    return max(0.0, -(end.delta.total_seconds() / 86400.0) - end.months * _NOMINAL_DAYS["mo"])


#: Which phase the rules are running in. The rules themselves are phase-blind;
#: the field exists so a message can say where a problem was found.
Phase = Literal["parse", "compile"]

ResourceKind = Literal["dataset", "model", "scoring"]


class RuleTarget(Protocol):
    """One resource as a rule sees it.

    ``ParsedResource`` satisfies this structurally, and so does
    ``ResolvedTarget`` - which is what lets the same rule run over a captured
    spec at parse time and a target-rendered one at compile time.
    """

    @property
    def unique_id(self) -> str: ...

    @property
    def name(self) -> str: ...

    @property
    def path(self) -> str: ...

    @property
    def spec(self) -> Any: ...


@dataclass(frozen=True)
class ResolvedTarget:
    """A compile-phase target: the spec as it renders against THIS target."""

    unique_id: str
    name: str
    path: str
    spec: Any


@dataclass(frozen=True)
class RuleContext:
    """The whole project as a rule sees it.

    Rules read neighbours through ``dataset_of``/``model_of`` rather than
    receiving them as arguments, so one signature serves every rule and the
    registry can invoke them uniformly.
    """

    project_name: str
    datasets: dict[str, RuleTarget]
    models: dict[str, RuleTarget]
    scoring: dict[str, RuleTarget]
    #: uid -> the uids it depends on, as the linker resolved them.
    depends_on: dict[str, list[str]]
    registry: AdapterRegistry
    phase: Phase = "parse"

    def targets(self, kind: ResourceKind) -> dict[str, RuleTarget]:
        return {"dataset": self.datasets, "model": self.models, "scoring": self.scoring}[kind]

    def dataset_of(self, model: RuleTarget) -> RuleTarget | None:
        """The dataset a model reads, or None when the edge did not resolve."""
        for uid in self.depends_on.get(model.unique_id, ()):
            if uid in self.datasets:
                return self.datasets[uid]
        return None

    def model_of(self, scoring: RuleTarget) -> RuleTarget | None:
        """The model a scoring pipeline scores with, or None."""
        for uid in self.depends_on.get(scoring.unique_id, ()):
            if uid in self.models:
                return self.models[uid]
        return None

    def dataset_by_name(self) -> dict[str, RuleTarget]:
        return {target.name: target for target in self.datasets.values()}


#: Both phases, which is the default and what closes the compile-phase hole.
BOTH_PHASES: tuple[Phase, ...] = ("parse", "compile")


@dataclass(frozen=True)
class Rule:
    """One invariant, which resources it is asked about, and when it can run.

    ``phases`` is ``BOTH_PHASES`` for everything that is a property of a
    resolved spec. The exceptions are rules about the SYNTAX of an unresolved
    reference: the resolve phase rewrites ``source('lake', 'rows')`` into the
    source's unique_id, so a rule asserting the call form can only hold before
    that. Marking them is how the registry stays "run everything, always"
    without quietly reporting a false error at compile.
    """

    name: str
    applies_to: ResourceKind
    check: Any  # Callable[[RuleTarget, RuleContext, ParseReport], None]
    phases: tuple[Phase, ...] = BOTH_PHASES


def run_rules(ctx: RuleContext, report: ParseReport) -> None:
    """Run every rule that applies in this phase, over every resource it covers.

    Errors accumulate on ``report`` rather than raising, so one parse reports
    every problem instead of the first.
    """
    for rule in RULES:
        if ctx.phase not in rule.phases:
            continue
        for target in ctx.targets(rule.applies_to).values():
            rule.check(target, ctx, report)


def _validate_dataset_windows(spec: DatasetSpec, rel: str, uid: str, report: ParseReport) -> None:
    if spec.split.strategy is not SplitStrategy.TEMPORAL:
        return
    parsed: dict[str, Any] = {}
    for split_field in ("train", "test", "validation", OUT_OF_TIME_SPLIT):
        expression = getattr(spec.split, split_field)
        if expression is None:
            continue
        try:
            parsed[split_field] = parse_window(expression)
        except ConfigError as exc:
            report.error(
                exc.message,
                file=rel,
                resource=uid,
                field_path=f"/split/{split_field}",
                hint=exc.hint,
            )
    test, after = parsed.get("test"), parsed.get(OUT_OF_TIME_SPLIT)
    if test is None or after is None:
        return
    # Anchor-independent only when both bounds are the same kind; a mixed
    # relative/absolute pair is ordered at compile time instead (ADR-30).
    kinds = {test.end.kind, after.start.kind}
    if not (kinds <= {"duration", "now"} or kinds == {"absolute"}):
        return
    if after.start.resolve(VALIDATION_ANCHOR) < test.end.resolve(VALIDATION_ANCHOR):
        report.error(
            f"split.out_of_time {spec.split.out_of_time!r} starts before the test "
            f"window {spec.split.test!r} ends",
            file=rel,
            resource=uid,
            field_path="/split/out_of_time",
            hint="the after-test window must start where the test window ends, so no "
            "row the model was evaluated on is scored again as new",
        )


def _validate_split_protocol(spec: DatasetSpec, rel: str, uid: str, report: ParseReport) -> None:
    """Warn on split configurations that invite leakage (FR-RES-09).

    Warnings, not errors: a random split over truly exchangeable rows is
    legitimate, and these flag the configurations that usually are not.
    """
    # Temporal split + a label horizon but no embargo (R2-7): rows near the
    # train boundary have their labels observed inside the evaluation window and
    # leak. The embargo mechanism exists; guide the user to actually set it.
    # The alignment itself happens upstream (ADR-29), so `label.horizon` is the
    # only place the project states the outcome window - which is exactly why
    # it exists: without it, moving the alignment out of mbt would have
    # silently switched this warning off.
    horizon = spec.label.horizon
    if (
        spec.split.strategy is SplitStrategy.TEMPORAL
        and horizon is not None
        and spec.split.embargo is None
    ):
        report.warning(
            "temporal split with a declared label horizon but no "
            "'split.embargo': training rows near the boundary have labels "
            "observed inside the evaluation window and can leak into it",
            file=rel,
            resource=uid,
            field_path="/split/embargo",
            hint=f"set split.embargo to at least the label horizon ({horizon})",
        )
    if (
        spec.split.strategy is SplitStrategy.TEMPORAL
        and horizon is not None
        and spec.split.embargo is not None
        and _duration_days(spec.split.embargo) < _duration_days(horizon)
    ):
        report.warning(
            f"split.embargo ({spec.split.embargo}) is shorter than the label "
            f"horizon ({horizon}), so the gap does not cover the window in "
            "which the label is observed",
            file=rel,
            resource=uid,
            field_path="/split/embargo",
            hint=f"set split.embargo to at least {horizon}",
        )
    if spec.split.strategy is not SplitStrategy.RANDOM:
        return
    if spec.split.time_column is not None:
        report.warning(
            "random split on a dataset with a time column invites temporal "
            "leakage: rows from after the test period can train the model",
            file=rel,
            resource=uid,
            field_path="/split/strategy",
            hint="use 'strategy: temporal', or drop 'time_column' if it is not event time",
        )


def _validate_checks(spec: DatasetSpec, rel: str, uid: str, report: ParseReport) -> None:
    for i, check in enumerate(spec.checks):
        check_name = check if isinstance(check, str) else next(iter(check), "")
        if check_name not in _BUILTIN_CHECKS:
            suggestion = did_you_mean(str(check_name), sorted(_BUILTIN_CHECKS))
            report.error(
                f"unknown dataset check {check_name!r}",
                file=rel,
                resource=uid,
                field_path=f"/checks/{i}",
                hint=f"did you mean {suggestion!r}?"
                if suggestion
                else f"built-in checks: {', '.join(sorted(_BUILTIN_CHECKS))}",
            )


def _check_adapter(
    spec: ModelSpec, uid: str, rel: str, registry: AdapterRegistry, report: ParseReport
) -> None:
    """Adapter installed, task supported, static hyperparameters valid (TSD §7)."""
    try:
        plugin = registry.get(spec.adapter)
    except ConfigError as exc:
        report.error(exc.message, file=rel, resource=uid, field_path="/adapter", hint=exc.hint)
        return
    if plugin.training is None:
        report.error(
            f"adapter {spec.adapter!r} provides no training adapter",
            file=rel,
            resource=uid,
            field_path="/adapter",
        )
        return
    adapter = plugin.training({})
    if spec.task not in adapter.supported_tasks:
        supported = ", ".join(sorted(t.value for t in adapter.supported_tasks))
        report.error(
            f"adapter {spec.adapter!r} does not support task {spec.task.value!r}",
            file=rel,
            resource=uid,
            field_path="/task",
            hint=f"supported tasks: {supported}",
        )
        return
    can = capabilities_of(adapter, spec)
    if spec.calibration is not None and Capability.CALIBRATION not in can:
        report.error(
            f"adapter {spec.adapter!r} does not support calibration",
            file=rel,
            resource=uid,
            field_path="/calibration",
            hint="drop 'calibration', or use a built-in adapter (all support it)",
        )
        return
    if not _check_feature_capabilities(spec, adapter, uid, rel, report):
        return
    validate_hyperparameters(
        adapter,
        spec.task,
        spec.hyperparameters,
        resource=uid,
        rel=rel,
        report=report,
    )
    for issue in adapter.validate(spec):
        add = report.error if issue.severity == "error" else report.warning
        add(issue.message, file=rel, resource=uid, field_path=issue.field_path, hint=issue.hint)

    try:
        task_schema = get_task_schema(spec.task)
    except ConfigError as exc:
        report.error(exc.message, file=rel, resource=uid, field_path="/task", hint=exc.hint)
        return
    for issue in task_schema.validate_spec(spec):
        add = report.error if issue.severity == "error" else report.warning
        add(issue.message, file=rel, resource=uid, field_path=issue.field_path, hint=issue.hint)


#: Feature-treatment capabilities probed on the adapter class (ADR-27), as
#: (what the spec asks for, the ClassVar, the field, what to say). A
#: declaration the adapter cannot honour fails at parse rather than being
#: dropped at train time: a constraint the DS believes is protecting them but
#: that silently does not exist is worse than no constraint at all.
_FEATURE_CAPABILITIES = (
    (
        "monotonic_constraints",
        Capability.MONOTONIC_CONSTRAINTS,
        "/features/monotonic",
        "enforce monotone constraints",
        "use the xgboost or lightgbm adapter, or drop features.monotonic",
    ),
    (
        "pooled_categoricals",
        Capability.CATEGORICAL_POOLING,
        "/features/categorical",
        "pool rare categorical levels (min_frequency)",
        "pin the level set with 'levels' instead, which needs no fitted level map, "
        "or use the xgboost, lightgbm, or sklearn adapter",
    ),
)


def _requested_feature_capability(spec: ModelSpec, name: str) -> bool:
    if name == "monotonic_constraints":
        return bool(spec.features.monotonic_constraints)
    return any(p.min_frequency is not None for p in spec.features.categorical_policies.values())


def _check_feature_capabilities(
    spec: ModelSpec, adapter: Any, uid: str, rel: str, report: ParseReport
) -> bool:
    """False when a declaration the adapter cannot honour was reported."""
    can = capabilities_of(adapter, spec)
    for name, needed, field_path, capability, hint in _FEATURE_CAPABILITIES:
        if not _requested_feature_capability(spec, name):
            continue
        if needed in can:
            continue
        report.error(
            f"adapter {spec.adapter!r} cannot {capability}",
            file=rel,
            resource=uid,
            field_path=field_path,
            hint=hint,
        )
        return False
    return True


def _is_deferred_value(value: Any) -> bool:
    """Values not statically checkable: AUTO sentinels or unresolved Jinja."""
    if value is None:
        return True
    return isinstance(value, str) and (value == AUTO or "{{" in value or "{%" in value)


def validate_hyperparameters(
    adapter: Any,
    task: TaskType,
    hyperparameters: dict[str, Any],
    *,
    resource: str,
    rel: str,
    report: ParseReport,
) -> None:
    """Two-step param validation: unknown keys always; values when static.

    At parse time, values still holding Jinja or AUTO sentinels are skipped;
    at compile time only AUTO survives (resolved later by the adapter).
    """
    param_model = adapter.param_model(task)
    known = set(param_model.model_fields)
    static: dict[str, Any] = {}
    for key, value in hyperparameters.items():
        if key not in known:
            suggestion = did_you_mean(key, sorted(known))
            report.error(
                f"unknown hyperparameter {key!r} for adapter "
                f"{adapter.name!r} / task {task.value!r}",
                file=rel,
                resource=resource,
                field_path=f"/hyperparameters/{key}",
                hint=f"did you mean {suggestion!r}?"
                if suggestion
                else f"valid: {', '.join(sorted(known))}",
            )
        elif not _is_deferred_value(value):
            static[key] = value
    if not static:
        return
    validate_resource(
        param_model,
        static,
        rel=rel,
        resource_name=resource,
        base_pointer="/hyperparameters",
        report=report,
    )


def _validate_scoring_windows(spec: ScoringSpec, rel: str, uid: str, report: ParseReport) -> None:
    if spec.input.window is not None:
        try:
            parse_window(spec.input.window)
        except ConfigError as exc:
            report.error(
                exc.message, file=rel, resource=uid, field_path="/input/window", hint=exc.hint
            )
    if spec.ground_truth is None:
        return
    maturity = spec.ground_truth.maturity
    problem: str | None = None
    if ":" in maturity:
        problem = f"ground_truth.maturity must be a bare duration, got {maturity!r}"
    else:
        try:
            window = parse_window(maturity)
            if window.start.kind != "duration" or window.start.delta is None:
                problem = (  # pragma: no cover - bare parse_window output is always a duration
                    f"ground_truth.maturity must be a duration, got {maturity!r}"
                )
        except ConfigError as exc:
            problem = exc.message
    if problem is not None:
        report.error(
            problem,
            file=rel,
            resource=uid,
            field_path="/ground_truth/maturity",
            hint="examples: 14d, 2w, 72h",
        )


def _validate_scoring_checks(spec: ScoringSpec, rel: str, uid: str, report: ParseReport) -> None:
    """Scoring inputs are unlabeled: only label-free checks apply (ADR-20)."""
    for i, check in enumerate(spec.checks):
        check_name = check if isinstance(check, str) else next(iter(check), "")
        if check_name not in _SCORING_CHECKS:
            suggestion = did_you_mean(str(check_name), sorted(_SCORING_CHECKS))
            report.error(
                f"check {check_name!r} is not available on scoring inputs",
                file=rel,
                resource=uid,
                field_path=f"/checks/{i}",
                hint=f"did you mean {suggestion!r}?"
                if suggestion
                else f"scoring checks: {', '.join(sorted(_SCORING_CHECKS))}",
            )
            continue
        if check_name == "not_null":
            params = check.get("not_null", {}) if isinstance(check, dict) else {}
            if not params.get("columns"):
                report.error(
                    "not_null on a scoring input requires explicit 'columns' "
                    "(there is no label column to default to)",
                    file=rel,
                    resource=uid,
                    field_path=f"/checks/{i}",
                    hint="e.g. not_null: {columns: [user_id]}",
                )


def _check_dataset_source_syntax(dataset: RuleTarget, report: ParseReport) -> None:
    """Dataset table references must be source() calls, not bare names."""
    spec = dataset.spec
    assert isinstance(spec, DatasetSpec)
    entries: list[tuple[str, str]] = [("/source", spec.source)]
    for field_path, value in entries:
        if not _SOURCE_RE.match(value):
            report.error(
                f"expected a source() reference, got {value!r}",
                file=dataset.path,
                resource=dataset.unique_id,
                field_path=field_path,
                hint="e.g. source('lakehouse', 'subscribers')",
            )


def _check_model_vs_dataset(
    spec: ModelSpec, model: RuleTarget, dataset_res: RuleTarget, report: ParseReport
) -> None:
    ds_spec = dataset_res.spec
    assert isinstance(ds_spec, DatasetSpec)
    if spec.target != ds_spec.label.column:
        report.error(
            f"model target {spec.target!r} must equal the dataset's label column "
            f"{ds_spec.label.column!r}",
            file=model.path,
            resource=model.unique_id,
            field_path="/target",
            hint="mismatches are an error, not a silent override (TSD §5.6)",
        )
    if spec.evaluation.protocol.split is not ds_spec.split.strategy:
        report.error(
            f"evaluation.protocol.split ({spec.evaluation.protocol.split.value}) must match "
            f"the dataset's split.strategy ({ds_spec.split.strategy.value}) (FR-RES-09)",
            file=model.path,
            resource=model.unique_id,
            field_path="/evaluation/protocol/split",
            hint="the redundancy is deliberate: it keeps the model spec self-describing",
        )
        return
    test_window = spec.evaluation.protocol.test_window
    if test_window is None:
        return
    if ds_spec.split.strategy is not SplitStrategy.TEMPORAL:
        report.error(
            "evaluation.protocol.test_window requires a temporal split",
            file=model.path,
            resource=model.unique_id,
            field_path="/evaluation/protocol/test_window",
        )
        return
    try:
        inner = parse_window(test_window)
        outer = parse_window(ds_spec.split.test)
    except ConfigError as exc:
        report.error(
            exc.message,
            file=model.path,
            resource=model.unique_id,
            field_path="/evaluation/protocol/test_window",
            hint=exc.hint,
        )
        return
    if not is_subrange(inner, outer, VALIDATION_ANCHOR):
        report.error(
            f"test_window {test_window!r} must resolve to a sub-range of the dataset's "
            f"test window {ds_spec.split.test!r}",
            file=model.path,
            resource=model.unique_id,
            field_path="/evaluation/protocol/test_window",
        )


def _check_after_test_needs_window(
    spec: ModelSpec, model: RuleTarget, ds_spec: DatasetSpec, report: ParseReport
) -> None:
    """After-test gates judge rows the dataset must actually produce (ADR-30).

    Without ``split.out_of_time`` there is no after-test split, so such a gate
    could never see a cell - it would pass as not-applicable on every build,
    a control that looks enforced and never is.
    """
    if ds_spec.split.out_of_time is not None:
        return
    for i, gate in enumerate(spec.evaluation.gates):
        if gate.source == "out_of_time":
            report.error(
                f"gate on {gate.metric!r} uses source: out_of_time, but dataset "
                f"{ds_spec.name!r} declares no split.out_of_time window",
                file=model.path,
                resource=model.unique_id,
                field_path=f"/evaluation/gates/{i}/source",
                hint='add split.out_of_time to the dataset, e.g. out_of_time: "-3mo:now"',
            )
    if spec.evaluation.stability is not None:
        report.error(
            f"evaluation.stability judges the after-test window, but dataset "
            f"{ds_spec.name!r} declares no split.out_of_time window",
            file=model.path,
            resource=model.unique_id,
            field_path="/evaluation/stability",
            hint='add split.out_of_time to the dataset, e.g. out_of_time: "-3mo:now"',
        )


def _check_maturity_vs_horizon(
    spec: ScoringSpec,
    scoring: RuleTarget,
    dataset_res: RuleTarget | None,
    report: ParseReport,
) -> None:
    """`ground_truth.maturity` must reach the label horizon (ADR-29).

    A prediction run is evaluated once ``scored_at + maturity`` has passed; the
    outcome of a scored row is observed at ``inference_date + horizon``. Those
    are anchored to different instants, and the scoring window is what bridges
    them: a batch selected with ``window: "-31d:-28d"`` holds rows that are
    already at least 28 days old when they are scored, so it needs 28 days less
    maturity than one whose window ends at ``now``.

    So the bar is ``maturity + (minimum age of the newest scored row) >=
    horizon``, and the newest row's age comes from the window's END bound. Below
    it, the monitor grades predictions against outcomes that have not been
    observed, and the realized metrics are quietly wrong rather than missing.
    """
    if spec.ground_truth is None:
        return
    if dataset_res is None:
        return
    ds_spec = dataset_res.spec
    assert isinstance(ds_spec, DatasetSpec)
    horizon = ds_spec.label.horizon
    if horizon is None:
        return
    try:
        reach = _duration_days(spec.ground_truth.maturity) + _newest_row_age_days(spec.input.window)
        too_short = reach < _duration_days(horizon)
    except ValueError:
        # A malformed maturity is already reported by _validate_scoring_windows,
        # and parsing collects every error in one pass rather than stopping, so
        # this runs anyway. Units cannot KeyError: parse_time_offset accepts
        # only the four _NOMINAL_DAYS knows.
        return
    if too_short:
        report.warning(
            f"ground_truth.maturity ({spec.ground_truth.maturity}) does not "
            f"reach the label horizon ({horizon}) declared by dataset "
            f"{ds_spec.name!r}: predictions would be evaluated against outcomes "
            "that have not been observed yet",
            file=scoring.path,
            resource=scoring.unique_id,
            field_path="/ground_truth/maturity",
            hint="raise ground_truth.maturity, or end the input window earlier "
            "than 'now' so the batch is already partly matured when it is scored",
        )


def _check_scoring_source_syntax(sc: RuleTarget, report: ParseReport) -> None:
    """Scoring table references must be source() calls, not bare names."""
    spec = sc.spec
    assert isinstance(spec, ScoringSpec)
    entries: list[tuple[str, str]] = [("/input/source", spec.input.source)]
    if spec.ground_truth is not None:
        entries.append(("/ground_truth/label/source", spec.ground_truth.label.source))
    for field_path, value in entries:
        if not _SOURCE_RE.match(value):
            report.error(
                f"expected a source() reference, got {value!r}",
                file=sc.path,
                resource=sc.unique_id,
                field_path=field_path,
                hint="e.g. source('lakehouse', 'scoring_batch')",
            )


def _check_tuning_engine(
    spec: ModelSpec, model: RuleTarget, registry: AdapterRegistry, report: ParseReport
) -> None:
    if spec.tuning is None:
        return
    try:
        plugin = registry.get(spec.tuning.engine)
    except ConfigError as exc:
        report.error(
            exc.message,
            file=model.path,
            resource=model.unique_id,
            field_path="/tuning/engine",
            hint=exc.hint,
        )
        return
    if plugin.tuning is None:
        report.error(
            f"adapter {spec.tuning.engine!r} provides no tuning engine",
            file=model.path,
            resource=model.unique_id,
            field_path="/tuning/engine",
        )


def _check_report_engine(
    spec: ModelSpec, model: RuleTarget, registry: AdapterRegistry, report: ParseReport
) -> None:
    """A non-native report engine is a plugin, probed like the tuning engine."""
    report_spec = spec.evaluation.report
    if report_spec is None or report_spec.stability.engine == "native":
        return
    engine = report_spec.stability.engine
    field_path = "/evaluation/report/stability/engine"
    try:
        plugin = registry.get(engine)
    except ConfigError as exc:
        report.error(
            exc.message,
            file=model.path,
            resource=model.unique_id,
            field_path=field_path,
            hint=f"install mbt-{engine}, or use engine: native",
        )
        return
    if getattr(plugin, "reporting", None) is None:
        report.error(
            f"adapter {engine!r} provides no report engine",
            file=model.path,
            resource=model.unique_id,
            field_path=field_path,
        )


def _check_rebalancing_vs_calibration(
    spec: ModelSpec, model: RuleTarget, report: ParseReport
) -> None:
    """Auto-rebalancing plus a calibration metric with no calibrator (D-4).

    ``scale_pos_weight: '{{ auto }}'`` resolves to ``(1 - p) / p`` and
    deliberately destroys probability calibration - it is a ranking aid. Asking
    for ``brier`` or ``ece`` on the result, with no ``calibration:`` set,
    reports a calibration number for scores that were miscalibrated on purpose.

    R2-8 was this exact combination and was closed twice: by BUILDING
    calibration, and then by fixing the demo fixture. Both fixed the instance;
    neither added a guard, so a user writing the same three lines today got the
    same silently meaningless number the demo used to report.

    A warning, not an error: the combination is legal and a user who knows what
    they are doing may want the ranking metrics anyway.
    """
    if spec.calibration is not None:
        return
    weight = spec.hyperparameters.get("scale_pos_weight")
    if weight is None:
        return
    rebalanced = _is_auto(weight) or _is_large_weight(weight)
    if not rebalanced:
        return
    # evaluation.metrics is a list of NAMES on the spec; a project-level metric
    # may alias a builtin, so match on the declared name.
    declared = sorted({m for m in spec.evaluation.metrics if m in _CALIBRATION_METRICS})
    if not declared:
        return
    report.warning(
        f"scale_pos_weight={weight!r} rebalances the positive class, which shifts "
        f"predicted probabilities away from the base rate, but "
        f"{', '.join(declared)} measures calibration and no 'calibration:' is set - "
        "the reported value describes scores that were miscalibrated on purpose",
        file=model.path,
        resource=model.unique_id,
        field_path="/evaluation/metrics",
        hint="add 'calibration: isotonic' (or 'platt') to fit a calibrator on a "
        f"held-out slice, or drop {declared[0]!r} and gate on ranking metrics",
    )


#: Metrics that measure how well a score matches an observed frequency.
_CALIBRATION_METRICS = frozenset({"brier", "ece"})

#: Above this, a static scale_pos_weight is a deliberate rebalance rather than
#: a mild nudge. The auto value for a 20% positive rate is 4.0.
_REBALANCE_THRESHOLD = 2.0


def _is_auto(value: Any) -> bool:
    return isinstance(value, str) and value.strip() == AUTO


def _is_large_weight(value: Any) -> bool:
    try:
        return float(value) >= _REBALANCE_THRESHOLD
    except (TypeError, ValueError):
        return False  # a Jinja expression: not statically checkable


# -- rules needing a neighbour ----------------------------------------------


def _rule_model_vs_dataset(t: RuleTarget, ctx: RuleContext, report: ParseReport) -> None:
    dataset = ctx.dataset_of(t)
    if dataset is not None:
        _check_model_vs_dataset(t.spec, t, dataset, report)


def _rule_after_test_needs_window(t: RuleTarget, ctx: RuleContext, report: ParseReport) -> None:
    dataset = ctx.dataset_of(t)
    if dataset is not None:
        _check_after_test_needs_window(t.spec, t, dataset.spec, report)


def _rule_maturity_vs_horizon(t: RuleTarget, ctx: RuleContext, report: ParseReport) -> None:
    """The scoring pipeline's model's dataset, through the LINKED graph.

    This used to re-parse ``ref('name')`` off the model spec and look the name
    up - which silently found nothing at compile time, because the resolve
    phase has already rewritten that ref into a unique_id.
    """
    model = ctx.model_of(t)
    _check_maturity_vs_horizon(t.spec, t, ctx.dataset_of(model) if model else None, report)


# -- the registry -----------------------------------------------------------
#
# One entry per invariant. Order is the order a reader meets them, not a
# dependency: every rule is a pure function of its target and the context, so
# they may run in any order and all of them always run.

RULES: tuple[Rule, ...] = (
    Rule(
        "dataset.source_syntax",
        "dataset",
        lambda t, ctx, report: _check_dataset_source_syntax(t, report),
        phases=("parse",),  # the resolve phase rewrites source() into a uid
    ),
    Rule(
        "dataset.windows",
        "dataset",
        lambda t, ctx, report: _validate_dataset_windows(t.spec, t.path, t.unique_id, report),
    ),
    Rule(
        "dataset.split_protocol",
        "dataset",
        lambda t, ctx, report: _validate_split_protocol(t.spec, t.path, t.unique_id, report),
    ),
    Rule(
        "dataset.checks",
        "dataset",
        lambda t, ctx, report: _validate_checks(t.spec, t.path, t.unique_id, report),
    ),
    Rule(
        "model.adapter",
        "model",
        lambda t, ctx, report: _check_adapter(t.spec, t.unique_id, t.path, ctx.registry, report),
    ),
    Rule(
        "model.tuning_engine",
        "model",
        lambda t, ctx, report: _check_tuning_engine(t.spec, t, ctx.registry, report),
    ),
    Rule(
        "model.report_engine",
        "model",
        lambda t, ctx, report: _check_report_engine(t.spec, t, ctx.registry, report),
    ),
    Rule("model.vs_dataset", "model", _rule_model_vs_dataset),
    Rule("model.after_test_needs_window", "model", _rule_after_test_needs_window),
    Rule(
        "model.rebalancing_vs_calibration_metrics",
        "model",
        lambda t, ctx, report: _check_rebalancing_vs_calibration(t.spec, t, report),
    ),
    Rule(
        "scoring.source_syntax",
        "scoring",
        lambda t, ctx, report: _check_scoring_source_syntax(t, report),
        phases=("parse",),  # the resolve phase rewrites source() into a uid
    ),
    Rule(
        "scoring.windows",
        "scoring",
        lambda t, ctx, report: _validate_scoring_windows(t.spec, t.path, t.unique_id, report),
    ),
    Rule(
        "scoring.checks",
        "scoring",
        lambda t, ctx, report: _validate_scoring_checks(t.spec, t.path, t.unique_id, report),
    ),
    Rule("scoring.maturity_vs_horizon", "scoring", _rule_maturity_vs_horizon),
)


__all__ = [
    "RULES",
    "Phase",
    "ResolvedTarget",
    "Rule",
    "RuleContext",
    "RuleTarget",
    "run_rules",
    "validate_hyperparameters",
]
