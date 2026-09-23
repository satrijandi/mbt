"""Gate evaluation: pure comparisons over MetricResults (TSD §11.2).

Adapters compute metrics; core compares them. This module has zero ML
dependencies and is fully unit-testable (FR-TEST-02/03/06).
"""

from typing import Literal

from mbt.artifacts.run_results import GateResult
from mbt.events import EventBus, get_bus
from mbt.events.models import GateEvaluated, LogMessage
from mbt.exceptions import MbtError
from mbt.quality.metrics import metric_direction
from mbt_adapter_base import (
    BootstrapDelta,
    DeterminismTier,
    GateSpec,
    MetricResults,
    MetricSpec,
    ReportSummary,
)


def _backtest_value(
    backtest_metrics: dict[str, float] | None, gate: GateSpec, *, resource: str
) -> float:
    """The walk-forward backtest mean of a gate's metric (R2-7, source: backtest)."""
    if not backtest_metrics or gate.metric not in backtest_metrics:
        raise MbtError(
            f"gate on {gate.metric!r} uses source: backtest but no backtest metric was computed",
            resource=resource,
            hint="set evaluation.protocol.backtest_folds and give the training window "
            "enough rows to form the folds",
        )
    return backtest_metrics[gate.metric]


def _metric_value(results: MetricResults, gate: GateSpec, *, who: str, resource: str) -> float:
    pool = results.metrics
    if gate.slice is not None:
        slice_pool = results.slices.get(gate.slice)
        if slice_pool is None:
            raise MbtError(
                f"gate on slice {gate.slice!r} has no slice metrics for {who}",
                resource=resource,
                hint=(
                    "the slice value must occur in the test split with both classes "
                    "present; degenerate slices are dropped from metrics"
                ),
            )
        pool = slice_pool
    if gate.metric not in pool:
        raise MbtError(
            f"metric {gate.metric!r} was not computed for {who}",
            resource=resource,
            hint="the adapter did not return a metric core resolved; this is an adapter bug",
        )
    return pool[gate.metric]


def _disparity_result(
    gate: GateSpec, challenger: MetricResults, greater: bool, resource: str
) -> GateResult:
    """Fairness/disparity gate: the metric's worst slice must stay within
    ``min_ratio`` of its best slice across the ``across`` column's values (R2-9).

    The ratio is min/max of the metric over the column's slices - in (0, 1]
    with 1.0 = perfect parity - which is direction-agnostic: disparity is about
    the *gap* between the extreme slices, so min/max captures it for both
    higher-is-better (roc_auc) and lower-is-better (rmse) metrics. Only which
    extreme is labelled 'worst' depends on the metric direction. The gated
    metric is non-negative here: r2 (the one signed builtin, whose worst/best
    ratio would be ill-defined) is rejected on an ``across`` gate at parse
    (GateSpec validation, F16), so every metric that reaches this ratio is >= 0.
    """
    prefix = f"{gate.across}="
    values = {
        label: pool[gate.metric]
        for label, pool in challenger.slices.items()
        if label.startswith(prefix) and gate.metric in pool
    }
    if len(values) < 2:
        raise MbtError(
            f"disparity gate on {gate.across!r} needs at least two non-degenerate slices "
            f"with metric {gate.metric!r}, found {len(values)}",
            resource=resource,
            hint=(
                f"list {gate.across!r} under evaluation.slices and ensure at least two of its "
                "values survive the test split (degenerate slices are dropped from metrics)"
            ),
        )
    lo_label, lo = min(values.items(), key=lambda kv: kv[1])
    hi_label, hi = max(values.items(), key=lambda kv: kv[1])
    ratio = 1.0 if hi == 0.0 else lo / hi
    (worst_label, worst), (best_label, best) = (
        ((lo_label, lo), (hi_label, hi)) if greater else ((hi_label, hi), (lo_label, lo))
    )
    return GateResult(
        metric=gate.metric,
        kind="disparity",
        across=gate.across,
        passed=ratio >= gate.min_ratio,
        expected=gate.min_ratio,
        actual=round(ratio, 12),
        worst_slice=worst_label,
        best_slice=best_label,
        message=(
            f"{gate.metric} parity across {gate.across}: worst/best ratio {ratio:.4f} "
            f"(worst {worst_label}={worst:.4f}, best {best_label}={best:.4f}); "
            f"min_ratio {gate.min_ratio}"
        ),
    )


def _within(actual: float, threshold: float, greater: bool, tolerance: float) -> bool:
    """Threshold comparison; tolerance widens it in the model's favour only."""
    if greater:
        return actual >= threshold - tolerance
    return actual <= threshold + tolerance


def _out_of_time_result(
    gate: GateSpec,
    report: ReportSummary | None,
    greater: bool,
    tolerance: float,
    resource: str,
) -> GateResult:
    """An after-test gate (ADR-30): the worst mature cell at the gate's grain.

    A cell is judged only when every row in it has matured and at least
    ``min_rows`` of them carry a label; anything less measures the calendar,
    not the model. With nothing to judge the gate passes as not applicable,
    loudly - the ADR-10 stance - so a model trained before its after-test
    labels mature is not blocked by time alone.
    """
    assert gate.threshold is not None  # validator: after-test gates are threshold gates
    period = gate.effective_period
    label = f"oot_{gate.metric}"
    cells = [
        cell
        for cell in (report.periods if report is not None else [])
        if cell.period == period
        and cell.mature
        and cell.n_labelled >= gate.effective_min_rows
        and gate.metric in cell.metrics
    ]
    if not cells:
        message = (
            f"no mature after-test {period} cell with at least {gate.effective_min_rows} "
            "labelled rows yet; gate not applicable"
        )
        get_bus().emit(
            LogMessage(level="warn", unique_id=resource, message=f"gate {label}: {message}")
        )
        return GateResult(
            metric=label,
            kind="threshold",
            passed=True,
            expected=gate.threshold,
            period=period,
            applicable=False,
            message=message,
        )
    values = [cell.metrics[gate.metric] for cell in cells]
    index = values.index(min(values) if greater else max(values))
    worst, actual = cells[index], values[index]
    return GateResult(
        metric=label,
        kind="threshold",
        passed=_within(actual, gate.threshold, greater, tolerance),
        expected=gate.threshold,
        actual=actual,
        period=period,
        cell=worst.key,
        message=f"worst {period} cell {worst.key} of {len(cells)} judged",
    )


def evaluate_gates(
    gates: list[GateSpec],
    *,
    resource: str,
    challenger: MetricResults,
    champion: MetricResults | None,
    champion_version: str | None,
    metric_specs: list[MetricSpec],
    determinism: DeterminismTier | None = None,
    champion_delta_bounds: dict[str, BootstrapDelta] | None = None,
    backtest_metrics: dict[str, float] | None = None,
    evaluated_is_champion: bool = False,
    report: ReportSummary | None = None,
    out_of_time: Literal["include", "only", "skip"] = "include",
) -> list[GateResult]:
    """Evaluate all gates for one model; emits GateEvaluated events.

    ``evaluated_is_champion`` marks a re-evaluation of the registered champion
    itself (``mbt evaluate --stage production --gates``, the decay check).
    There is no challenger then, so champion gates are not applicable: the
    artifact would be compared with its own predictions, a delta of exactly 0
    that fails every positive ``min_delta``, which turned the documented decay
    check into a deterministic exit 2. Threshold and disparity gates still run.

    ``report`` carries the after-test cells ``source: out_of_time`` gates judge
    (ADR-30). ``out_of_time="only"`` evaluates just those gates - what the
    pre-deploy check re-runs - and ``"skip"`` leaves them out, for the
    re-evaluations (``mbt test``, a plain ``mbt evaluate``) that build no
    after-test report.
    """
    bus = get_bus()
    results: list[GateResult] = []

    def record(gate: GateSpec, result: GateResult) -> GateResult:
        """Stamp the gate's own ``source`` onto its result, then emit it.

        Done here rather than at each of the eight construction sites so a new
        gate kind cannot forget it - and so ``after_test_verdict`` can ask the
        question it means ("was this an after-test gate?") instead of inferring
        it from ``period is not None`` (C-1).
        """
        stamped = result.model_copy(update={"source": gate.source})
        results.append(stamped)
        _emit_gate(bus, resource, stamped)
        return stamped

    for gate in gates:
        is_after_test = gate.source == "out_of_time"
        if (out_of_time == "only" and not is_after_test) or (
            out_of_time == "skip" and is_after_test
        ):
            continue
        greater = metric_direction(gate.metric, metric_specs)

        if gate.source == "out_of_time":
            tolerance = determinism.tolerance_for(gate.metric) if determinism else 0.0
            record(gate, _out_of_time_result(gate, report, greater, tolerance, resource))
            continue

        if gate.across is not None:
            record(gate, _disparity_result(gate, challenger, greater, resource))
            continue

        if gate.source == "backtest":
            # A backtest gate (validator-guaranteed threshold + whole-split) checks
            # the walk-forward mean, labelled backtest_<metric> so the card/PR
            # comment distinguishes it from the single-split gate (R2-7).
            actual = _backtest_value(backtest_metrics, gate, resource=resource)
            label = f"backtest_{gate.metric}"
        else:
            actual = _metric_value(challenger, gate, who="the challenger", resource=resource)
            label = gate.metric

        if gate.threshold is not None:
            # Tolerance widens threshold comparisons in the model's favor only
            # (FR-ADPT-07); champion deltas never get widened.
            tolerance = determinism.tolerance_for(gate.metric) if determinism else 0.0
            passed = _within(actual, gate.threshold, greater, tolerance)
            result = GateResult(
                metric=label,
                kind="threshold",
                slice=gate.slice,
                passed=passed,
                expected=gate.threshold,
                actual=actual,
            )
        elif evaluated_is_champion:
            result = GateResult(
                metric=gate.metric,
                kind="champion",
                slice=gate.slice,
                passed=True,
                actual=actual,
                min_delta=gate.min_delta,
                message=(
                    f"the evaluated version is the '{gate.compare_to}' champion itself - "
                    "no challenger to compare, gate not applicable"
                ),
            )
        elif champion is None:
            # Bootstrap: no champion exists yet -> pass with a loud WARN
            # (ADR-10). An unloadable champion never reaches this code -
            # the job fails hard instead.
            result = GateResult(
                metric=gate.metric,
                kind="champion",
                slice=gate.slice,
                passed=True,
                actual=actual,
                champion_version=None,
                min_delta=gate.min_delta,
                message="no champion registered yet; gate passes (bootstrap)",
            )
            bus.emit(
                LogMessage(
                    level="warn",
                    unique_id=resource,
                    message=(
                        f"champion gate on {gate.metric!r}: no champion in "
                        f"'{gate.compare_to}' yet - passing with a warning (FR-TEST-06)"
                    ),
                )
            )
        else:
            champion_value = _metric_value(champion, gate, who="the champion", resource=resource)
            delta = (actual - champion_value) if greater else (champion_value - actual)
            bound = (champion_delta_bounds or {}).get(gate.metric)
            if bound is not None and gate.confidence is not None and gate.slice is None:
                # Paired-bootstrap criterion (ADR-18): the delta must clear
                # min_delta at the gate's one-sided confidence, not merely on
                # the point estimate, so noise cannot promote a challenger.
                if bound.n_resamples > 0:
                    message = (
                        f"paired bootstrap ({bound.n_resamples} resamples): delta lower "
                        f"bound {bound.lower:.6f} at {bound.confidence:.0%} confidence"
                    )
                else:
                    message = "bootstrap degenerate (no valid resamples); point delta used"
                result = GateResult(
                    metric=gate.metric,
                    kind="champion",
                    slice=gate.slice,
                    passed=bound.lower >= gate.min_delta,
                    actual=actual,
                    champion_version=champion_version,
                    champion_value=champion_value,
                    min_delta=gate.min_delta,
                    actual_delta=round(delta, 12),
                    delta_lower=round(bound.lower, 12),
                    confidence=bound.confidence,
                    message=message,
                )
            else:
                # Point-estimate fallback: slice gates, hook metrics, or an
                # explicit ``confidence: null`` opt-out.
                result = GateResult(
                    metric=gate.metric,
                    kind="champion",
                    slice=gate.slice,
                    passed=delta >= gate.min_delta,
                    actual=actual,
                    champion_version=champion_version,
                    champion_value=champion_value,
                    min_delta=gate.min_delta,
                    actual_delta=round(delta, 12),
                )
        record(gate, result)
    return results


def _emit_gate(bus: EventBus, resource: str, result: GateResult) -> None:
    bus.emit(
        GateEvaluated(
            unique_id=resource,
            metric=result.metric,
            kind=result.kind,
            passed=result.passed,
            expected=result.expected,
            actual=result.actual,
            champion_version=result.champion_version,
            message=result.message or "",
        )
    )


def all_gates_passed(results: list[GateResult]) -> bool:
    return all(result.passed for result in results)
