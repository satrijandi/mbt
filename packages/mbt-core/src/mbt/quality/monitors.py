"""Monitor threshold evaluation (ADR-20/21).

Pure comparisons, zero ML dependencies: scoring jobs compute ``ShiftStat``s
and ``mbt monitor`` computes realized metrics; this module only applies the
declared thresholds ("jobs compute, core compares", ADR-3). A breach sets
node status ``monitor_failed`` (exit code 2).
"""

from typing import Literal, NamedTuple

from mbt.artifacts.run_results import MonitorResult
from mbt.contracts import MetricSpec, MonitorsSpec, MonitorStats, ShiftStat
from mbt.events import get_bus
from mbt.events.models import LogMessage
from mbt.quality.metrics import metric_direction
from mbt_adapter_base.specs import FeatureShiftSpec, MonitorGateSpec, PredictionShiftSpec


def ks_critical_value(significance: float, n_baseline: int, n_current: int) -> float:
    """Two-sample KS critical value at ``significance``: the D above which the
    two samples differ significantly (R2-6).

    ``c(a) * sqrt((n1 + n2) / (n1 * n2))`` with ``c(a) = sqrt(-ln(a/2) / 2)``.
    Sample-size-aware, so the bar tightens on large batches and loosens on small
    ones - unlike a fixed threshold, which over-fires on big nightly batches and
    under-fires on small ones.
    """
    import math

    coefficient = math.sqrt(-math.log(significance / 2.0) / 2.0)
    return coefficient * math.sqrt((n_baseline + n_current) / (n_baseline * n_current))


def _lower_regularized_gamma(a: float, x: float) -> float:
    """The regularized lower incomplete gamma ``P(a, x)``.

    Pure stdlib (mbt-core carries no scipy, matching ``_cramers_v``): series for
    ``x < a + 1``, continued fraction (Lentz) for the upper tail above it -
    Numerical Recipes ``gser``/``gcf``. ``x`` is always positive here (a
    bisection midpoint, or half a chi-square statistic on the p-value path), so
    no zero guard is needed; both branches are well-defined for x > 0.
    """
    import math

    if x < a + 1.0:  # series converges fast here
        term = 1.0 / a
        total = term
        n = a
        for _ in range(500):
            n += 1.0
            term *= x / n
            total += term
            if abs(term) < abs(total) * 1e-15:
                break
        return total * math.exp(-x + a * math.log(x) - math.lgamma(a))
    # continued fraction for the upper tail (Lentz's method)
    tiny = 1e-300
    b = x + 1.0 - a
    c = 1.0 / tiny
    d = 1.0 / b
    h = d
    for i in range(1, 500):
        an = -i * (i - a)
        b += 2.0
        d = an * d + b
        d = tiny if abs(d) < tiny else d
        c = b + an / c
        c = tiny if abs(c) < tiny else c
        d = 1.0 / d
        delta = d * c
        h *= delta
        if abs(delta - 1.0) < 1e-15:
            break
    upper = math.exp(-x + a * math.log(x) - math.lgamma(a)) * h
    return 1.0 - upper


def chi2_critical_value(significance: float, df: int) -> float:
    """Upper-tail chi-square quantile at ``significance`` with ``df`` degrees
    of freedom - the n-aware bar for a categorical ``significance`` monitor
    (F15), sibling to :func:`ks_critical_value` for numeric features.

    Bisection on :func:`_lower_regularized_gamma`, solving ``P = 1 -
    significance`` to ~1e-10.
    """
    target = 1.0 - significance
    lo, hi = 0.0, float(df)
    while _lower_regularized_gamma(df / 2.0, hi / 2.0) < target:
        hi *= 2.0
    for _ in range(200):
        mid = (lo + hi) / 2.0
        if _lower_regularized_gamma(df / 2.0, mid / 2.0) < target:
            lo = mid
        else:
            hi = mid
    return (lo + hi) / 2.0


def ks_p_value(value: float, n_baseline: int, n_current: int) -> float:
    """The two-sample KS p-value for statistic ``value``, exactly inverting
    :func:`ks_critical_value`.

    ``2 * exp(-2 * D^2 * n1*n2/(n1+n2))`` is the first term of the Kolmogorov
    series, which is the same approximation ``c(a) = sqrt(-ln(a/2)/2)`` inverts.
    Using the same one in both directions is what makes "p <= alpha" and
    "D > critical value at alpha" the same decision, so a family of one behaves
    identically before and after the correction below.
    """
    import math

    effective_n = (n_baseline * n_current) / (n_baseline + n_current)
    return min(1.0, 2.0 * math.exp(-2.0 * value * value * effective_n))


def chi2_p_value(value: float, df: int) -> float:
    """The upper-tail chi-square p-value for statistic ``value``, the exact
    inverse of :func:`chi2_critical_value`."""
    return 1.0 - _lower_regularized_gamma(df / 2.0, value / 2.0)


def _uses_ks_critical_value(spec: FeatureShiftSpec | PredictionShiftSpec, stat: ShiftStat) -> bool:
    """Whether the n-aware KS ``significance`` bar applies to this stat: a
    *numeric* ``ks`` stat under a ``significance`` monitor (the categorical
    stat follows the chi-square null instead, F15)."""
    return spec.significance is not None and stat.method == "ks" and stat.kind == "numeric"


def _uses_chi2_critical_value(
    spec: FeatureShiftSpec | PredictionShiftSpec, stat: ShiftStat
) -> bool:
    """Whether the chi-square ``significance`` bar applies: a categorical ``ks``
    stat computed WITH its df (the job computed a Pearson chi-square because
    significance was set, F15). A categorical stat without df - an older mbt's
    job, or a threshold-path stat - falls back to the fixed threshold."""
    return (
        spec.significance is not None
        and stat.method == "ks"
        and stat.kind == "categorical"
        and stat.df is not None
    )


def _p_value(spec: FeatureShiftSpec | PredictionShiftSpec, stat: ShiftStat) -> float | None:
    """The p-value behind one stat, or None when this stat is on the fixed-
    threshold path and is therefore not a hypothesis test at all."""
    if _uses_ks_critical_value(spec, stat):
        return ks_p_value(stat.value, stat.n_baseline, stat.n_current)
    if _uses_chi2_critical_value(spec, stat):
        assert stat.df is not None  # narrowed by _uses_chi2_critical_value
        return chi2_p_value(stat.value, stat.df)
    return None


def benjamini_hochberg_significance(p_values: list[float], significance: float) -> float:
    """The per-test alpha that reproduces the Benjamini-Hochberg decision on
    ``p_values`` at false-discovery rate ``significance``.

    BH rejects the ``k`` smallest p-values for the largest ``k`` with
    ``p_(k) <= k/m * alpha``. Returning ``k/m * alpha`` as a single per-test bar
    is equivalent, because ``p_(k+1) > (k+1)/m * alpha`` by construction: every
    rejected p-value is at or below it and every accepted one is strictly above.
    With no rejections it returns ``alpha/m``, which every p-value clears for
    the same reason. That equivalence is what lets the caller keep comparing
    statistics against a critical value - the units an operator can read - while
    the decision is BH's.

    A family of one returns ``alpha`` unchanged, so ``prediction_shift`` and a
    single-feature model behave exactly as before.
    """
    family_size = len(p_values)
    if family_size <= 1:
        return significance
    rejected = 0
    for rank, p_value in enumerate(sorted(p_values), start=1):
        if p_value <= rank / family_size * significance:
            rejected = rank
    return max(rejected, 1) / family_size * significance


class _Correction(NamedTuple):
    """The multiple-comparison correction in force for one monitor's family.

    ``significance`` is the per-test alpha after correction (None on the fixed-
    threshold path); ``family_size`` is how many hypothesis tests it covers.
    """

    significance: float | None
    family_size: int


def _feature_shift_correction(
    spec: FeatureShiftSpec, feature_shift: dict[str, ShiftStat]
) -> _Correction:
    """Benjamini-Hochberg across the feature set (D-1).

    Every feature under a ``significance`` monitor is an independent test at
    that alpha, and every breach is exit 2. On a 40-feature model at
    ``significance: 0.05`` with no real drift, the expected number of breaches
    is 2 - so the monitor fires on most clean nightly runs and the operational
    response converges on ignoring it, which is worse than the fixed threshold
    it replaced, because a fixed threshold never claimed to be a test.

    BH is the right default for a screening monitor: it holds the false-
    DISCOVERY rate rather than the family-wise error rate, so it stays
    sensitive as the feature set grows instead of going numb the way
    Bonferroni's alpha/m does.
    """
    if spec.significance is None:
        return _Correction(None, 0)
    p_values = [
        p_value
        for p_value in (_p_value(spec, stat) for stat in feature_shift.values())
        if p_value is not None
    ]
    if not p_values:
        return _Correction(spec.significance, 0)
    return _Correction(benjamini_hochberg_significance(p_values, spec.significance), len(p_values))


def _fail_bar(
    spec: FeatureShiftSpec | PredictionShiftSpec,
    stat: ShiftStat,
    correction: _Correction,
) -> float:
    """The fail threshold for one stat: the kind-matched n-aware critical value
    at the CORRECTED significance when the monitor sets ``significance`` on a
    ``ks`` stat (Kolmogorov for numeric, chi-square for categorical), else the
    fixed ``threshold``."""
    if _uses_ks_critical_value(spec, stat):
        assert correction.significance is not None  # narrowed by _uses_ks_critical_value
        return ks_critical_value(correction.significance, stat.n_baseline, stat.n_current)
    if _uses_chi2_critical_value(spec, stat):
        assert correction.significance is not None and stat.df is not None
        return chi2_critical_value(correction.significance, stat.df)
    return spec.threshold


def _bar_label(
    spec: FeatureShiftSpec | PredictionShiftSpec,
    stat: ShiftStat,
    bar: float,
    correction: _Correction,
) -> str:
    family = (
        f", Benjamini-Hochberg over {correction.family_size} features"
        if correction.family_size > 1
        else ""
    )
    if _uses_ks_critical_value(spec, stat):
        return f"the KS critical value {bar:.4f} (significance={spec.significance}{family})"
    if _uses_chi2_critical_value(spec, stat):
        return (
            f"the chi-square critical value {bar:.4f} "
            f"(significance={spec.significance}{family}, df={stat.df})"
        )
    return f"{bar:.4f}"


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
        correction = _feature_shift_correction(spec, stats.feature_shift)
        if correction.family_size > 1:
            get_bus().emit(
                LogMessage(
                    unique_id=resource,
                    message=(
                        f"feature_shift significance: Benjamini-Hochberg over "
                        f"{correction.family_size} features holds the false-discovery "
                        f"rate at {spec.significance}; the per-feature bar this run is "
                        f"alpha={correction.significance:.6g}"
                    ),
                )
            )
        for feature, stat in sorted(stats.feature_shift.items()):
            if (
                spec.significance is not None
                and stat.method == "ks"
                and stat.kind == "categorical"
                and stat.df is None
            ):
                # A categorical stat under significance normally carries a
                # chi-square df (F15); one without it was computed by an older
                # mbt job, so it falls back to the fixed threshold - say so
                # rather than silently under-firing.
                get_bus().emit(
                    LogMessage(
                        level="warn",
                        unique_id=resource,
                        message=(
                            f"feature_shift significance: categorical feature "
                            f"{feature!r} has no chi-square df (stat computed by an "
                            f"older mbt job); falling back to the fixed threshold "
                            f"{spec.threshold} (F15)"
                        ),
                    )
                )
            bar = _fail_bar(spec, stat, correction)
            results.append(
                _shift_result(
                    "feature_shift",
                    subject=feature,
                    stat=stat,
                    threshold=bar,
                    warn_threshold=spec.warn_threshold,
                    breach=(
                        f"{feature}: {stat.method}={stat.value:.4f} exceeds "
                        f"{_bar_label(spec, stat, bar, correction)} "
                        f"(n={stat.n_current} vs baseline n={stat.n_baseline})"
                    ),
                    warn=f"{feature}: {stat.method}={stat.value:.4f} in the shift warn band",
                    resource=resource,
                )
            )
        if stats.feature_shift:
            _emit_top_shifts(stats.feature_shift, resource=resource)
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
        # A family of one: the score distribution is a single hypothesis, so
        # there is nothing to correct for and the bar is the declared alpha.
        pred_correction = _Correction(shift_spec.significance, 1)
        bar = _fail_bar(shift_spec, stat, pred_correction)
        results.append(
            _shift_result(
                "prediction_shift",
                subject=None,
                stat=stat,
                threshold=bar,
                warn_threshold=shift_spec.warn_threshold,
                breach=(
                    f"score distribution {stat.method}={stat.value:.4f} exceeds "
                    f"{_bar_label(shift_spec, stat, bar, pred_correction)} "
                    "vs the champion's test-split baseline"
                ),
                warn=f"score distribution {stat.method}={stat.value:.4f} in the shift warn band",
                resource=resource,
            )
        )
    return results


#: How many of the most-shifted features the per-run summary names (R2-6).
_TOP_SHIFTS = 3


def _emit_top_shifts(feature_shift: dict[str, ShiftStat], *, resource: str) -> None:
    """Log the most-shifted features this run, ranked by shift value, so drift
    is visible before it crosses a threshold (R2-6). Ties break by name."""
    ranked = sorted(feature_shift.items(), key=lambda kv: (-kv[1].value, kv[0]))
    top = ranked[:_TOP_SHIFTS]
    summary = ", ".join(f"{name}={stat.value:.4f}" for name, stat in top)
    suffix = f" (top {len(top)} of {len(feature_shift)})" if len(feature_shift) > len(top) else ""
    get_bus().emit(
        LogMessage(
            level="info",
            unique_id=resource,
            message=f"feature_shift most shifted: {summary}{suffix}",
        )
    )


def _shift_result(
    monitor: Literal["feature_shift", "prediction_shift"],
    *,
    subject: str | None,
    stat: ShiftStat,
    threshold: float,
    warn_threshold: float | None,
    breach: str,
    warn: str,
    resource: str,
) -> MonitorResult:
    """One shift comparison with a two-tier bar (ADR-20, R2-6).

    ``value > threshold`` fails (exit 2); ``warn_threshold < value <= threshold``
    passes but logs a warning and records the reason; otherwise a clean pass.
    """
    passed = stat.value <= threshold
    warned = passed and warn_threshold is not None and stat.value > warn_threshold
    if not passed:
        message: str | None = breach
    elif warned:
        message = f"{warn} [{warn_threshold}, {threshold}]"
        get_bus().emit(
            LogMessage(level="warn", unique_id=resource, message=f"{monitor} warn: {message}")
        )
    else:
        message = None
    return MonitorResult(
        monitor=monitor,
        subject=subject,
        measure=stat.method,
        value=stat.value,
        threshold=threshold,
        passed=passed,
        message=message,
    )


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
