"""Gate evaluation: pure comparisons over MetricResults (TSD §11.2).

Adapters compute metrics; core compares them. This module has zero ML
dependencies and is fully unit-testable (FR-TEST-02/03/06).
"""

from mbt.artifacts.run_results import GateResult
from mbt.contracts import DeterminismTier, GateSpec, MetricResults, MetricSpec
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
                hint="declare the slice column under evaluation.slices",
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
            result = GateResult(
                metric=gate.metric,
                kind="champion",
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
