"""Monitor evaluation unit tests: pure comparisons, zero ML deps (ADR-20/21).

Sibling of ``test_gates.py``/``test_checks.py`` for the "core compares" half
of monitoring (ADR-3): ``evaluate_monitors`` applies shift thresholds and
``evaluate_ground_truth_gates`` applies realized-metric thresholds. These
were previously only exercised through integration/e2e; here they are pinned
directly, including the misconfig and baseline-missing branches.
"""

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
