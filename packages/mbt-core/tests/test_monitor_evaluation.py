"""Monitor evaluation unit tests: pure comparisons, zero ML deps (ADR-20/21).

Sibling of ``test_gates.py``/``test_checks.py`` for the "core compares" half
of monitoring (ADR-3): ``evaluate_monitors`` applies shift thresholds and
``evaluate_ground_truth_gates`` applies realized-metric thresholds. These
were previously only exercised through integration/e2e; here they are pinned
directly, including the misconfig and baseline-missing branches.
"""

import pytest
from exec_unit_helpers import recording_bus

from mbt.contracts import (
    FeatureShiftSpec,
    MetricSpec,
    MonitorsSpec,
    MonitorStats,
    PredictionShiftSpec,
    ShiftStat,
)
from mbt.quality.monitors import (
    all_monitors_passed,
    evaluate_ground_truth_gates,
    evaluate_monitors,
)
from mbt_adapter_base.specs import MonitorGateSpec

SPECS = [
    MetricSpec(name="pr_auc", kind="builtin", greater_is_better=True),
    MetricSpec(name="logloss", kind="builtin", greater_is_better=False),
]


def _stat(value: float) -> ShiftStat:
    return ShiftStat(method="psi", value=value, n_current=100, n_baseline=200)


# -- evaluate_monitors ------------------------------------------------------------


def test_no_monitors_yields_no_results() -> None:
    assert evaluate_monitors(None, None, resource="scoring.p.s") == []


def test_feature_shift_threshold_is_directionful() -> None:
    monitors = MonitorsSpec(feature_shift=FeatureShiftSpec(method="psi", threshold=0.2))
    stats = MonitorStats(feature_shift={"age": _stat(0.05), "tenure": _stat(0.5)})
    results = {r.subject: r for r in evaluate_monitors(monitors, stats, resource="scoring.p.s")}
    assert results["age"].passed  # 0.05 <= 0.2
    assert not results["tenure"].passed  # 0.5 > 0.2
    assert "exceeds" in (results["tenure"].message or "")


def test_prediction_shift_breach_is_flagged() -> None:
    monitors = MonitorsSpec(prediction_shift=PredictionShiftSpec(method="ks", threshold=0.15))
    breached = MonitorStats(
        prediction_shift=ShiftStat(method="ks", value=0.4, n_current=1, n_baseline=1)
    )
    (result,) = evaluate_monitors(monitors, breached, resource="scoring.p.s")
    assert result.monitor == "prediction_shift"
    assert not result.passed
    assert result.subject is None

    calm = MonitorStats(
        prediction_shift=ShiftStat(method="ks", value=0.05, n_current=1, n_baseline=1)
    )
    assert evaluate_monitors(monitors, calm, resource="scoring.p.s")[0].passed


def test_feature_shift_warn_band_passes_but_warns() -> None:
    monitors = MonitorsSpec(
        feature_shift=FeatureShiftSpec(method="psi", threshold=0.2, warn_threshold=0.1)
    )
    stats = MonitorStats(
        feature_shift={
            "calm": _stat(0.05),  # <= warn_threshold: clean pass
            "elevated": _stat(0.15),  # (warn, threshold]: warn band
            "breached": _stat(0.5),  # > threshold: fail
        }
    )
    with recording_bus() as sink:
        results = {r.subject: r for r in evaluate_monitors(monitors, stats, resource="scoring.p.s")}
    assert results["calm"].passed and results["calm"].message is None
    assert results["elevated"].passed  # a warn does not fail the run ...
    assert "warn band" in (results["elevated"].message or "")  # ... but is recorded
    assert not results["breached"].passed
    # only the warn-band feature reached the event stream
    warns = [m for m in sink.messages() if "warn band" in m]
    assert len(warns) == 1 and "elevated" in warns[0]

    # a warn alone keeps the node green (only breaches flip all_monitors_passed)
    warn_only = evaluate_monitors(
        monitors,
        MonitorStats(feature_shift={"elevated": _stat(0.15)}),
        resource="scoring.p.s",
    )
    assert all_monitors_passed(warn_only)


def _ks_stat(value: float, n: int) -> ShiftStat:
    return ShiftStat(method="ks", value=value, n_current=n, n_baseline=n)


def test_feature_shift_significance_scales_the_fail_bar_with_batch_size() -> None:
    """An n-aware KS significance tightens the fail bar on large batches and
    loosens it on small ones (R2-6): the SAME KS statistic fails a big nightly
    batch but passes a small one, where a fixed threshold would over/under-fire."""
    aware = MonitorsSpec(
        feature_shift=FeatureShiftSpec(method="ks", threshold=0.15, significance=0.05)
    )
    for n, expect_pass in ((5000, False), (40, True)):
        stats = MonitorStats(feature_shift={"age": _ks_stat(0.10, n)})
        result = evaluate_monitors(aware, stats, resource="scoring.p.s")[0]
        assert result.passed is expect_pass, (n, result.threshold)
        assert result.threshold != 0.15  # the recorded bar is the per-batch critical value
        if not expect_pass:
            assert "KS critical value" in (result.message or "")

    # the same 0.10 always clears a FIXED 0.15 threshold, at either batch size
    fixed = MonitorsSpec(feature_shift=FeatureShiftSpec(method="ks", threshold=0.15))
    for n in (5000, 40):
        stats = MonitorStats(feature_shift={"age": _ks_stat(0.10, n)})
        assert evaluate_monitors(fixed, stats, resource="scoring.p.s")[0].passed


def test_feature_shift_significance_is_kind_matched() -> None:
    """F15: significance applies a KIND-MATCHED n-aware bar - the two-sample KS
    critical value for a numeric stat, the chi-square critical value (df from
    the stat) for a categorical Pearson chi-square stat - and only a legacy
    categorical stat WITHOUT df falls back to the fixed threshold, loudly."""
    from mbt.quality.monitors import chi2_critical_value, ks_critical_value

    aware = MonitorsSpec(
        feature_shift=FeatureShiftSpec(method="ks", threshold=0.15, significance=0.05)
    )
    num = ShiftStat(method="ks", value=0.05, n_current=5000, n_baseline=5000, kind="numeric")
    chi2 = ShiftStat(
        method="ks", value=9.2, n_current=5000, n_baseline=5000, kind="categorical", df=3
    )
    legacy = ShiftStat(method="ks", value=0.05, n_current=5000, n_baseline=5000, kind="categorical")

    # categorical WITH df: judged against the chi-square critical value
    (chi2_result,) = evaluate_monitors(
        aware, MonitorStats(feature_shift={"plan": chi2}), resource="scoring.p.s"
    )
    assert chi2_result.threshold == pytest.approx(chi2_critical_value(0.05, 3))
    assert not chi2_result.passed  # 9.2 exceeds the 7.81 bar at df=3
    assert "chi-square critical value" in (chi2_result.message or "")

    # legacy categorical stat without df: fixed threshold + a loud fallback
    with recording_bus() as sink:
        (legacy_result,) = evaluate_monitors(
            aware, MonitorStats(feature_shift={"plan": legacy}), resource="scoring.p.s"
        )
    assert legacy_result.threshold == 0.15
    assert legacy_result.passed  # 0.05 clears the fixed 0.15
    assert any("no chi-square df" in m for m in sink.messages())

    # numeric: the tighter n-aware KS critical value still applies (not the fixed bar)
    (num_result,) = evaluate_monitors(
        aware, MonitorStats(feature_shift={"age": num}), resource="scoring.p.s"
    )
    assert num_result.threshold == pytest.approx(ks_critical_value(0.05, 5000, 5000))
    assert num_result.threshold != 0.15
    assert not num_result.passed  # 0.05 exceeds the ~0.027 critical value at n=5000


def test_significance_delivers_close_to_its_nominal_level_under_h0() -> None:
    """F15's measured claim, now guarded by Monte-Carlo: with NO true shift at
    alpha=0.05, both significance bars fire close to 5% - the numeric
    merged-points KS (the old grid-only sup was measured conservative) and the
    categorical contingency chi-square (the old fixed-threshold path was
    measured ~7x conservative, and a goodness-of-fit-against-estimated-shares
    variant ~2x anti-conservative). Seeded, so the measured rates are exact."""
    import numpy as np

    from mbt.quality.monitors import chi2_critical_value, ks_critical_value
    from mbt_adapter_base.monitoring import (
        _GRID_POINTS,
        FeatureBaseline,
        _categorical_chi2,
        _numeric_ks,
    )

    rng = np.random.default_rng(20260722)
    reps = 300
    grid = np.linspace(0.0, 1.0, _GRID_POINTS)

    ks_bar = ks_critical_value(0.05, 2000, 500)
    numeric_hits = 0
    for _ in range(reps):
        quantiles = list(np.quantile(rng.normal(size=2000), grid))
        numeric_hits += _numeric_ks(quantiles, rng.normal(size=500)) > ks_bar
    numeric_fpr = numeric_hits / reps
    assert 0.02 <= numeric_fpr <= 0.09, numeric_fpr

    categorical_hits = 0
    for _ in range(reps):
        base_draw = rng.choice(["a", "b", "c"], size=2000, p=[0.5, 0.3, 0.2])
        shares = [float(np.mean(base_draw == c)) for c in ("a", "b", "c")]
        baseline = FeatureBaseline(
            kind="categorical",
            categories=["a", "b", "c", "__other__"],
            proportions=[*shares, 0.0],
            n=2000,
        )
        current = list(rng.choice(["a", "b", "c"], size=500, p=[0.5, 0.3, 0.2]))
        stat, df = _categorical_chi2(baseline, current)
        categorical_hits += stat > chi2_critical_value(0.05, df)
    categorical_fpr = categorical_hits / reps
    assert 0.02 <= categorical_fpr <= 0.09, categorical_fpr


def test_chi2_critical_value_matches_the_reference_quantiles() -> None:
    """The stdlib-only chi-square quantile matches the standard table values
    (scipy.stats.chi2.isf reference) to 1e-5 at alpha 0.05 and 0.01."""
    from mbt.quality.monitors import chi2_critical_value

    references = {
        (0.05, 1): 3.841459,
        (0.05, 2): 5.991465,
        (0.05, 3): 7.814728,
        (0.05, 5): 11.070498,
        (0.05, 10): 18.307038,
        (0.01, 3): 11.344867,
    }
    for (alpha, df), expected in references.items():
        assert chi2_critical_value(alpha, df) == pytest.approx(expected, abs=1e-5)


def test_ks_critical_value_is_sample_size_and_significance_aware() -> None:
    import math

    from mbt.quality.monitors import ks_critical_value

    # tighter (smaller D) on larger samples...
    assert ks_critical_value(0.05, 5000, 5000) < ks_critical_value(0.05, 50, 50)
    # ...and a laxer significance needs a smaller D than a strict one
    assert ks_critical_value(0.10, 500, 500) < ks_critical_value(0.01, 500, 500)
    # concrete: c(a) * sqrt((n1 + n2) / (n1 * n2)), c(a) = sqrt(-ln(a/2)/2)
    coefficient = math.sqrt(-math.log(0.05 / 2) / 2)
    assert ks_critical_value(0.05, 100, 300) == pytest.approx(coefficient * math.sqrt(400 / 30000))


def test_prediction_shift_warn_band_passes_but_warns() -> None:
    monitors = MonitorsSpec(
        prediction_shift=PredictionShiftSpec(method="ks", threshold=0.2, warn_threshold=0.1)
    )
    stats = MonitorStats(
        prediction_shift=ShiftStat(method="ks", value=0.15, n_current=1, n_baseline=1)
    )
    with recording_bus() as sink:
        (result,) = evaluate_monitors(monitors, stats, resource="scoring.p.s")
    assert result.passed and "warn band" in (result.message or "")
    assert any("prediction_shift warn" in m for m in sink.messages())


def test_feature_shift_emits_a_most_shifted_summary() -> None:
    monitors = MonitorsSpec(feature_shift=FeatureShiftSpec(method="psi", threshold=0.9))
    stats = MonitorStats(
        feature_shift={"a": _stat(0.10), "b": _stat(0.40), "c": _stat(0.20), "d": _stat(0.30)}
    )
    with recording_bus() as sink:
        evaluate_monitors(monitors, stats, resource="scoring.p.s")
    summary = next(m for m in sink.messages() if "most shifted" in m)
    # top 3 by shift value, descending; the 4th (a=0.10) is omitted
    assert summary == "feature_shift most shifted: b=0.4000, d=0.3000, c=0.2000 (top 3 of 4)"


def test_most_shifted_summary_lists_all_when_few_and_skips_when_none() -> None:
    monitors = MonitorsSpec(feature_shift=FeatureShiftSpec(method="psi", threshold=0.9))
    # <= _TOP_SHIFTS features: no "(top k of n)" suffix
    with recording_bus() as sink:
        evaluate_monitors(
            monitors,
            MonitorStats(feature_shift={"x": _stat(0.2), "y": _stat(0.1)}),
            resource="scoring.p.s",
        )
    assert next(m for m in sink.messages() if "most shifted" in m) == (
        "feature_shift most shifted: x=0.2000, y=0.1000"
    )
    # all features skipped -> no summary line at all
    with recording_bus() as sink2:
        evaluate_monitors(
            monitors,
            MonitorStats(feature_shift={}, skipped_features=["z"]),
            resource="scoring.p.s",
        )
    assert not any("most shifted" in m for m in sink2.messages())


def test_shift_warn_threshold_must_be_below_the_fail_threshold() -> None:
    with pytest.raises(ValueError, match="warn_threshold must be below threshold"):
        FeatureShiftSpec(method="psi", threshold=0.2, warn_threshold=0.3)
    with pytest.raises(ValueError, match="warn_threshold must be below threshold"):
        PredictionShiftSpec(method="ks", threshold=0.2, warn_threshold=0.2)  # equal is not below


def test_skipped_features_do_not_break_evaluation() -> None:
    monitors = MonitorsSpec(feature_shift=FeatureShiftSpec(method="psi", threshold=0.2))
    stats = MonitorStats(feature_shift={"age": _stat(0.05)}, skipped_features=["mystery"])
    results = evaluate_monitors(monitors, stats, resource="scoring.p.s")
    # The skipped feature is warned about, not turned into a (passing) result.
    assert [r.subject for r in results] == ["age"]


def test_baseline_missing_passes_loudly() -> None:
    monitors = MonitorsSpec(
        feature_shift=FeatureShiftSpec(method="psi", threshold=0.2),
        prediction_shift=PredictionShiftSpec(method="psi", threshold=0.2),
    )
    results = evaluate_monitors(
        monitors, MonitorStats(baseline_missing=True), resource="scoring.p.s"
    )
    assert len(results) == 2
    assert all(r.passed for r in results)
    assert all("baseline missing" in (r.message or "") for r in results)


# -- evaluate_ground_truth_gates --------------------------------------------------


def test_ground_truth_gate_direction_aware() -> None:
    up = evaluate_ground_truth_gates(
        [MonitorGateSpec(metric="pr_auc", threshold=0.3)], {"pr_auc": 0.45}, SPECS, run_key="rk"
    )
    assert up[0].passed and up[0].monitor == "ground_truth" and up[0].subject == "rk"
    assert not evaluate_ground_truth_gates(
        [MonitorGateSpec(metric="pr_auc", threshold=0.3)], {"pr_auc": 0.2}, SPECS, run_key="rk"
    )[0].passed

    lower = evaluate_ground_truth_gates(
        [MonitorGateSpec(metric="logloss", threshold=0.5)], {"logloss": 0.4}, SPECS, run_key="rk"
    )
    assert lower[0].passed  # lower is better: 0.4 <= 0.5


def test_ground_truth_gate_on_uncomputed_metric_fails_loudly() -> None:
    # A gate that names a metric the run never computed must fail (exit 2),
    # never silently pass.
    (result,) = evaluate_ground_truth_gates(
        [MonitorGateSpec(metric="pr_auc", threshold=0.3)], {}, SPECS, run_key="rk"
    )
    assert not result.passed
    assert result.value is None
    assert "was not computed" in (result.message or "")


def test_all_monitors_passed() -> None:
    monitors = MonitorsSpec(feature_shift=FeatureShiftSpec(method="psi", threshold=0.2))
    passing = evaluate_monitors(
        monitors, MonitorStats(feature_shift={"age": _stat(0.05)}), resource="s"
    )
    failing = evaluate_monitors(
        monitors, MonitorStats(feature_shift={"age": _stat(0.9)}), resource="s"
    )
    assert all_monitors_passed(passing)
    assert not all_monitors_passed(failing)
    assert all_monitors_passed([])  # vacuously true
