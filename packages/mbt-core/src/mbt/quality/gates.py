"""Gate evaluation: pure comparisons over MetricResults (TSD §11.2).

Adapters compute metrics; core compares them. This module has zero ML
dependencies and is fully unit-testable (FR-TEST-02/03/06).
"""

from mbt.artifacts.run_results import GateResult
from mbt.contracts import BootstrapDelta, DeterminismTier, GateSpec, MetricResults, MetricSpec
from mbt.events import get_bus
from mbt.events.models import GateEvaluated, LogMessage
from mbt.exceptions import MbtError
from mbt.quality.metrics import metric_direction


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
) -> list[GateResult]:
    """Evaluate all gates for one model; emits GateEvaluated events."""
    bus = get_bus()
    results: list[GateResult] = []
    for gate in gates:
        greater = metric_direction(gate.metric, metric_specs)
        actual = _metric_value(challenger, gate, who="the challenger", resource=resource)

        if gate.threshold is not None:
            # Tolerance widens threshold comparisons in the model's favor only
            # (FR-ADPT-07); champion deltas never get widened.
            tolerance = determinism.tolerance_for(gate.metric) if determinism else 0.0
            if greater:
                passed = actual >= gate.threshold - tolerance
            else:
                passed = actual <= gate.threshold + tolerance
            result = GateResult(
                metric=gate.metric,
                kind="threshold",
                slice=gate.slice,
                passed=passed,
                expected=gate.threshold,
                actual=actual,
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
        results.append(result)
        bus.emit(
            GateEvaluated(
                unique_id=resource,
                metric=gate.metric,
                kind=result.kind,
                passed=result.passed,
                expected=result.expected,
                actual=result.actual,
                champion_version=result.champion_version,
                message=result.message or "",
            )
        )
    return results


def all_gates_passed(results: list[GateResult]) -> bool:
    return all(result.passed for result in results)
