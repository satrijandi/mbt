"""Monitor threshold evaluation (ADR-20/21).

Pure comparisons, zero ML dependencies: scoring jobs compute ``ShiftStat``s
and ``mbt monitor`` computes realized metrics; this module only applies the
declared thresholds ("jobs compute, core compares", ADR-3). A breach sets
node status ``monitor_failed`` (exit code 2).
"""

from mbt.artifacts.run_results import MonitorResult
from mbt.contracts import MetricSpec, MonitorsSpec, MonitorStats
from mbt.events import get_bus
from mbt.events.models import LogMessage
from mbt.quality.metrics import metric_direction
from mbt_adapter_base.specs import MonitorGateSpec


def evaluate_monitors(
    monitors: MonitorsSpec | None,
    stats: MonitorStats | None,
    *,
    resource: str,
) -> list[MonitorResult]:
    """Apply shift thresholds to the statistics a scoring job computed."""
    if monitors is None:
        return []
    if stats is None or stats.baseline_missing:
        # ADR-10 spirit: a champion registered before baselines existed
        # cannot be monitored; pass loudly rather than block scoring.
        get_bus().emit(
            LogMessage(
                level="warn",
                unique_id=resource,
                message=(
                    "champion has no monitoring baseline (registered by an older "
                    "mbt); shift monitors pass without comparison - retrain to "
                    "capture a baseline (ADR-21)"
                ),
            )
        )
        return _baseline_missing_results(monitors)

    results: list[MonitorResult] = []
    if monitors.feature_shift is not None:
        spec = monitors.feature_shift
        for feature, stat in sorted(stats.feature_shift.items()):
            passed = stat.value <= spec.threshold
            results.append(
                MonitorResult(
                    monitor="feature_shift",
                    subject=feature,
                    measure=stat.method,
                    value=stat.value,
                    threshold=spec.threshold,
                    passed=passed,
                    message=None
                    if passed
                    else (
                        f"{feature}: {stat.method}={stat.value:.4f} exceeds "
                        f"{spec.threshold} (n={stat.n_current} vs baseline "
                        f"n={stat.n_baseline})"
                    ),
                )
            )
        for feature in stats.skipped_features:
            get_bus().emit(
                LogMessage(
                    level="warn",
                    unique_id=resource,
                    message=(
                        f"feature_shift skipped {feature!r}: not comparable against "
                        "the baseline (missing column, type change, or all-null batch)"
                    ),
                )
            )
    if monitors.prediction_shift is not None and stats.prediction_shift is not None:
        shift_spec = monitors.prediction_shift
        stat = stats.prediction_shift
        passed = stat.value <= shift_spec.threshold
        results.append(
            MonitorResult(
                monitor="prediction_shift",
                subject=None,
                measure=stat.method,
                value=stat.value,
                threshold=shift_spec.threshold,
                passed=passed,
                message=None
                if passed
                else (
                    f"score distribution {stat.method}={stat.value:.4f} exceeds "
                    f"{shift_spec.threshold} vs the champion's test-split baseline"
                ),
            )
        )
    return results


def _baseline_missing_results(monitors: MonitorsSpec) -> list[MonitorResult]:
    results: list[MonitorResult] = []
    message = "baseline missing; passed without comparison"
    if monitors.feature_shift is not None:
        results.append(
            MonitorResult(
                monitor="feature_shift",
                measure=monitors.feature_shift.method,
                threshold=monitors.feature_shift.threshold,
                passed=True,
                message=message,
            )
        )
    if monitors.prediction_shift is not None:
        results.append(
            MonitorResult(
                monitor="prediction_shift",
                measure=monitors.prediction_shift.method,
                threshold=monitors.prediction_shift.threshold,
                passed=True,
                message=message,
            )
        )
    return results


def evaluate_ground_truth_gates(
    gates: list[MonitorGateSpec],
    metrics: dict[str, float],
    metric_specs: list[MetricSpec],
    *,
    run_key: str,
) -> list[MonitorResult]:
    """Apply realized-metric thresholds to one evaluated prediction run."""
    results: list[MonitorResult] = []
    for gate in gates:
        value = metrics.get(gate.metric)
        if value is None:
            results.append(
                MonitorResult(
                    monitor="ground_truth",
                    subject=run_key,
                    measure=gate.metric,
                    threshold=gate.threshold,
                    passed=False,
                    message=f"metric {gate.metric!r} was not computed",
                )
            )
            continue
        greater_is_better = metric_direction(gate.metric, metric_specs)
        passed = value >= gate.threshold if greater_is_better else value <= gate.threshold
        comparator = ">=" if greater_is_better else "<="
        results.append(
            MonitorResult(
                monitor="ground_truth",
                subject=run_key,
                measure=gate.metric,
                value=value,
                threshold=gate.threshold,
                passed=passed,
                message=None
                if passed
                else (
                    f"run {run_key}: realized {gate.metric}={value:.4f} failed "
                    f"{comparator} {gate.threshold}"
                ),
            )
        )
    return results


def all_monitors_passed(results: list[MonitorResult]) -> bool:
    return all(r.passed for r in results)
