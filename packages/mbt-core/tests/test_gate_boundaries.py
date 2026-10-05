"""Every comparison that decides whether a model ships or a batch alarms, at
exactly its bar (FEEDBACK v6 C-1).

100% line coverage proved each of these lines ran, not that a test would notice
one flipping: of 20 hand-made mutants over gates, verdicts and monitors, 8 of
the 10 that survived the whole fast suite were ``>=`` turned into ``>`` (or the
mirror), because no test sat ON a threshold. Each test here does, with values
that are exact in binary floating point so the boundary is the boundary.

The direction of each bar is the documented one: a gate passes AT its
threshold, a monitor passes AT its threshold and fails strictly above it, a
warn band starts strictly above ``warn_threshold``, and a cell is judged once
it holds ``min_rows``.
"""

from typing import Any

from exec_unit_helpers import recording_bus

from mbt.artifacts.run_results import GateResult
from mbt.contracts import (
    BootstrapDelta,
    DeterminismTier,
    FeatureShiftSpec,
    GateSpec,
    MetricResults,
    MetricSpec,
    ModelSpec,
    MonitorsSpec,
    MonitorStats,
    PeriodCell,
    ReportSummary,
    ShiftStat,
    StabilityCell,
    StabilitySpec,
)
from mbt.quality.gates import evaluate_gates
from mbt.quality.judgement import judge
from mbt.quality.monitors import (
    benjamini_hochberg_significance,
    evaluate_ground_truth_gates,
    evaluate_monitors,
    evaluate_stability,
)
from mbt_adapter_base.specs import MonitorGateSpec

SPECS = [
    MetricSpec(name="pr_auc", kind="builtin", greater_is_better=True),
    MetricSpec(name="logloss", kind="builtin", greater_is_better=False),
]


def _gate(gate: GateSpec, challenger: dict[str, float], **kwargs: Any) -> GateResult:
    champion = kwargs.pop("champion", None)
    (result,) = evaluate_gates(
        [gate],
        resource="model.t.m",
        challenger=MetricResults(metrics=challenger, slices=kwargs.pop("slices", {})),
        champion=MetricResults(metrics=champion) if champion else None,
        champion_version="7" if champion else None,
        metric_specs=SPECS,
        **kwargs,
    )
    return result


# -- threshold gates ------------------------------------------------------------------


def test_a_threshold_gate_passes_at_its_threshold_in_both_directions() -> None:
    assert _gate(GateSpec(metric="pr_auc", threshold=0.5), {"pr_auc": 0.5}).passed
    assert _gate(GateSpec(metric="logloss", threshold=0.5), {"logloss": 0.5}).passed


def test_tolerance_widens_a_lower_is_better_threshold_upward() -> None:
    """Mutant G3. logloss is in the scaffold's own metric list, and every
    regression metric since ADR-24 is lower-is-better: the tolerance must widen
    toward HIGHER values for them, and never past its own width."""
    tier = DeterminismTier(kind="tolerance", tolerances={"logloss": 0.25})
    gate = GateSpec(metric="logloss", threshold=0.5)
    assert _gate(gate, {"logloss": 0.625}, determinism=tier).passed
    assert _gate(gate, {"logloss": 0.75}, determinism=tier).passed  # exactly the widened bar
    assert not _gate(gate, {"logloss": 0.875}, determinism=tier).passed


def test_tolerance_widens_a_higher_is_better_threshold_to_exactly_its_width() -> None:
    tier = DeterminismTier(kind="tolerance", tolerances={"pr_auc": 0.25})
    gate = GateSpec(metric="pr_auc", threshold=0.5)
    assert _gate(gate, {"pr_auc": 0.25}, determinism=tier).passed
    assert not _gate(gate, {"pr_auc": 0.125}, determinism=tier).passed


# -- disparity and champion gates -----------------------------------------------------


def test_a_disparity_gate_passes_at_exactly_min_ratio() -> None:
    slices = {"plan=basic": {"pr_auc": 1.0}, "plan=pro": {"pr_auc": 0.5}}
    gate = GateSpec(metric="pr_auc", across="plan", min_ratio=0.5)
    assert _gate(gate, {"pr_auc": 0.75}, slices=slices).passed


def test_a_champion_gate_passes_when_the_bootstrap_bound_equals_min_delta() -> None:
    bound = BootstrapDelta(point=0.25, lower=0.125, confidence=0.95, n_resamples=1000)
    gate = GateSpec(metric="pr_auc", compare_to="production", min_delta=0.125)
    result = _gate(
        gate, {"pr_auc": 0.75}, champion={"pr_auc": 0.5}, champion_delta_bounds={"pr_auc": bound}
    )
    assert result.delta_lower == 0.125 and result.passed


def test_a_champion_gate_passes_when_the_point_delta_equals_min_delta() -> None:
    gate = GateSpec(metric="pr_auc", compare_to="production", min_delta=0.25, confidence=None)
    result = _gate(gate, {"pr_auc": 0.75}, champion={"pr_auc": 0.5})
    assert result.actual_delta == 0.25 and result.passed
    lower = GateSpec(metric="logloss", compare_to="production", min_delta=0.25, confidence=None)
    assert _gate(lower, {"logloss": 0.25}, champion={"logloss": 0.5}).passed


# -- after-test gates and stability ---------------------------------------------------


def _period_cell(n_labelled: int) -> PeriodCell:
    return PeriodCell(
        period="month",
        key="2026-06",
        slot="all",
        start="s",
        end="e",
        n_rows=n_labelled,
        n_labelled=n_labelled,
        mature=True,
        metrics={"pr_auc": 0.25},
    )


def test_an_after_test_cell_with_exactly_min_rows_is_judged() -> None:
    report = ReportSummary(
        anchor="a", reference_rows=1, out_of_time_rows=10, periods=[_period_cell(10)]
    )
    gate = GateSpec(metric="pr_auc", threshold=0.5, source="out_of_time", min_rows=10)
    result = _gate(gate, {"pr_auc": 0.75}, report=report)
    assert result.applicable and not result.passed  # judged, and 0.25 < 0.5


def _stability_report(value: float, n_rows: int) -> ReportSummary:
    stat = ShiftStat(method="psi", value=value, n_current=n_rows, n_baseline=100)
    cell = StabilityCell(
        period="month",
        key="2026-06",
        slot="all",
        n_rows=n_rows,
        reference_rows=100,
        gate=MonitorStats(prediction_shift=stat),
    )
    return ReportSummary(anchor="a", reference_rows=100, out_of_time_rows=n_rows, stability=[cell])


def test_a_stability_cell_with_exactly_min_rows_is_judged() -> None:
    spec = StabilitySpec.model_validate(
        {"min_rows": 50, "prediction_shift": {"method": "psi", "threshold": 0.25}}
    )
    outcome = evaluate_stability(spec, _stability_report(0.5, n_rows=50), resource="m")
    assert outcome.judged and not outcome.results[0].passed


def _model_spec(**evaluation: Any) -> ModelSpec:
    return ModelSpec.model_validate(
        {
            "name": "m",
            "task": "binary_classification",
            "adapter": "fake",
            "owner": "o@example.com",
            "dataset": "ref('d')",
            "target": "y",
            "seed": 1,
            "evaluation": {"protocol": {"split": "temporal"}, "metrics": ["roc_auc"], **evaluation},
        }
    )


def test_judge_fails_a_stability_breach_when_every_gate_passed() -> None:
    """Mutant J1 at the seam; test_execution.py pins the registration end of it."""
    spec = _model_spec(stability={"prediction_shift": {"method": "psi", "threshold": 0.25}})
    gate = GateResult(metric="roc_auc", kind="threshold", passed=True)
    with recording_bus():
        outcome = judge(spec, [gate], _stability_report(0.5, n_rows=500), resource="m")
    assert not outcome.passed
    assert outcome.failure_summary and "stability [2026-06]" in outcome.failure_summary


# -- scoring monitors -----------------------------------------------------------------


def _stat(value: float) -> ShiftStat:
    return ShiftStat(method="psi", value=value, n_current=100, n_baseline=200)


def test_a_shift_monitor_passes_at_its_threshold_and_fails_just_above() -> None:
    monitors = MonitorsSpec(feature_shift=FeatureShiftSpec(method="psi", threshold=0.25))
    stats = MonitorStats(feature_shift={"at": _stat(0.25), "above": _stat(0.25 + 2**-20)})
    results = {r.subject: r.passed for r in evaluate_monitors(monitors, stats, resource="s")}
    assert results == {"at": True, "above": False}


def test_the_warn_band_starts_strictly_above_warn_threshold() -> None:
    monitors = MonitorsSpec(
        feature_shift=FeatureShiftSpec(method="psi", threshold=0.5, warn_threshold=0.25)
    )
    stats = MonitorStats(feature_shift={"at": _stat(0.25), "inside": _stat(0.375)})
    with recording_bus() as sink:
        results = {r.subject: r for r in evaluate_monitors(monitors, stats, resource="s")}
    assert results["at"].passed and results["at"].message is None
    assert results["inside"].passed and results["inside"].message
    assert sum("feature_shift warn" in m for m in sink.messages()) == 1


def test_benjamini_hochberg_rejects_a_p_value_exactly_at_its_rank_bar() -> None:
    """Mutant M2: with m = 2 at alpha 0.05 the rank-2 bar is exactly 0.05, so
    both tests are rejected and the per-test alpha is 0.05, not 0.025."""
    assert benjamini_hochberg_significance([0.01, 0.05], 0.05) == 0.05
    assert benjamini_hochberg_significance([0.01, 0.0500001], 0.05) == 0.025


def test_a_ground_truth_gate_passes_at_its_threshold_in_both_directions() -> None:
    gates = [
        MonitorGateSpec(metric="pr_auc", threshold=0.5),
        MonitorGateSpec(metric="logloss", threshold=0.5),
    ]
    results = evaluate_ground_truth_gates(
        gates, {"pr_auc": 0.5, "logloss": 0.5}, SPECS, run_key="r"
    )
    assert [r.passed for r in results] == [True, True]
