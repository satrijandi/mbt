"""Gate engine unit tests: pure comparisons, zero ML deps (S5-01)."""

import pytest

from mbt.contracts import BootstrapDelta, DeterminismTier, GateSpec, MetricResults, MetricSpec
from mbt.exceptions import MbtError
from mbt.quality.gates import all_gates_passed, evaluate_gates

SPECS = [
    MetricSpec(name="pr_auc", kind="builtin", greater_is_better=True),
    MetricSpec(name="logloss", kind="builtin", greater_is_better=False),
]


def _eval(
    gates,
    challenger,
    champion=None,
    version=None,
    determinism=None,
    bounds=None,
    challenger_slices=None,
    champion_slices=None,
):
    return evaluate_gates(
        gates,
        resource="model.t.m",
        challenger=MetricResults(metrics=challenger, slices=challenger_slices or {}),
        champion=(
            MetricResults(metrics=champion, slices=champion_slices or {}) if champion else None
        ),
        champion_version=version,
        metric_specs=SPECS,
        determinism=determinism,
        champion_delta_bounds=bounds,
    )


def test_threshold_gate_direction_aware() -> None:
    gates = [GateSpec(metric="pr_auc", threshold=0.4)]
    assert _eval(gates, {"pr_auc": 0.45})[0].passed
    assert not _eval(gates, {"pr_auc": 0.35})[0].passed

    lower_better = [GateSpec(metric="logloss", threshold=0.5)]
    assert _eval(lower_better, {"logloss": 0.45})[0].passed
    assert not _eval(lower_better, {"logloss": 0.55})[0].passed


def test_tolerance_widens_thresholds_in_models_favor_only() -> None:
    tier = DeterminismTier(kind="tolerance", tolerances={"pr_auc": 0.01})
    gates = [GateSpec(metric="pr_auc", threshold=0.4)]
    assert _eval(gates, {"pr_auc": 0.395}, determinism=tier)[0].passed  # widened
    assert not _eval(gates, {"pr_auc": 0.38}, determinism=tier)[0].passed


def test_champion_gate_delta_and_direction() -> None:
    gates = [GateSpec(metric="pr_auc", compare_to="production", min_delta=0.005)]
    result = _eval(gates, {"pr_auc": 0.45}, champion={"pr_auc": 0.44}, version="7")[0]
    assert result.passed and result.champion_version == "7"
    assert result.actual_delta == pytest.approx(0.01)

    worse = _eval(gates, {"pr_auc": 0.441}, champion={"pr_auc": 0.44}, version="7")[0]
    assert not worse.passed  # delta 0.001 < min_delta 0.005

    # lower-is-better metric: improvement means challenger < champion
    ll_gates = [GateSpec(metric="logloss", compare_to="production", min_delta=0.0)]
    assert _eval(ll_gates, {"logloss": 0.40}, champion={"logloss": 0.45}, version="7")[0].passed
    assert not _eval(ll_gates, {"logloss": 0.50}, champion={"logloss": 0.45}, version="7")[0].passed


def test_champion_delta_never_widened_by_tolerance() -> None:
    tier = DeterminismTier(kind="tolerance", tolerances={"pr_auc": 0.05})
    gates = [GateSpec(metric="pr_auc", compare_to="production", min_delta=0.005)]
    result = _eval(
        gates, {"pr_auc": 0.441}, champion={"pr_auc": 0.44}, version="3", determinism=tier
    )[0]
    assert not result.passed  # tolerance must not rescue the delta (FR-ADPT-07)


def test_noisy_point_improvement_is_blocked_by_the_bootstrap_bound() -> None:
    """A challenger ahead on the point estimate but not at confidence must
    not be promoted (ADR-18); this is the failure mode min_delta alone
    cannot catch."""
    gates = [GateSpec(metric="pr_auc", compare_to="production", min_delta=0.0)]
    bound = BootstrapDelta(point=0.01, lower=-0.004, confidence=0.95, n_resamples=1000)
    result = _eval(
        gates,
        {"pr_auc": 0.45},
        champion={"pr_auc": 0.44},
        version="7",
        bounds={"pr_auc": bound},
    )[0]
    assert not result.passed
    assert result.actual_delta == pytest.approx(0.01)
    assert result.delta_lower == pytest.approx(-0.004)
    assert result.confidence == 0.95
    assert "paired bootstrap" in (result.message or "")


def test_significant_improvement_passes_the_bootstrap_bound() -> None:
    gates = [GateSpec(metric="pr_auc", compare_to="production", min_delta=0.005)]
    bound = BootstrapDelta(point=0.03, lower=0.011, confidence=0.95, n_resamples=1000)
    result = _eval(
        gates,
        {"pr_auc": 0.47},
        champion={"pr_auc": 0.44},
        version="7",
        bounds={"pr_auc": bound},
    )[0]
    assert result.passed
    assert result.delta_lower == pytest.approx(0.011)


def test_point_estimate_fallbacks() -> None:
    bound = BootstrapDelta(point=0.01, lower=-0.004, confidence=0.95, n_resamples=1000)

    # no bound computed for the metric -> point comparison
    gates = [GateSpec(metric="pr_auc", compare_to="production", min_delta=0.0)]
    result = _eval(gates, {"pr_auc": 0.45}, champion={"pr_auc": 0.44}, version="7")[0]
    assert result.passed and result.delta_lower is None

    # explicit confidence: null opts out even when a bound is present
    opt_out = [GateSpec(metric="pr_auc", compare_to="production", confidence=None)]
    result = _eval(
        opt_out,
        {"pr_auc": 0.45},
        champion={"pr_auc": 0.44},
        version="7",
        bounds={"pr_auc": bound},
    )[0]
    assert result.passed and result.delta_lower is None

    # degenerate bootstrap: lower fell back to the point delta, message says so
    degenerate = BootstrapDelta(point=0.01, lower=0.01, confidence=0.95, n_resamples=0)
    result = _eval(
        gates,
        {"pr_auc": 0.45},
        champion={"pr_auc": 0.44},
        version="7",
        bounds={"pr_auc": degenerate},
    )[0]
    assert result.passed
    assert "degenerate" in (result.message or "")


def test_missing_champion_passes_with_null_version() -> None:
    gates = [GateSpec(metric="pr_auc", compare_to="production", min_delta=0.005)]
    result = _eval(gates, {"pr_auc": 0.3})[0]
    assert result.passed
    assert result.champion_version is None
    assert "bootstrap" in (result.message or "")


def test_slice_threshold_gate_evaluates_the_slice_pool() -> None:
    gates = [GateSpec(metric="pr_auc", threshold=0.4, slice="plan_type=pro")]
    ok = _eval(gates, {"pr_auc": 0.9}, challenger_slices={"plan_type=pro": {"pr_auc": 0.45}})[0]
    assert ok.passed and ok.slice == "plan_type=pro"

    # a strong whole-split value cannot rescue a failing slice
    bad = _eval(gates, {"pr_auc": 0.9}, challenger_slices={"plan_type=pro": {"pr_auc": 0.35}})[0]
    assert not bad.passed


def test_slice_champion_gate_uses_point_delta_not_the_whole_split_bound() -> None:
    """Slice gates compare point deltas; the ADR-18 bootstrap bound is
    whole-split only and must never leak into a slice decision."""
    gates = [
        GateSpec(metric="pr_auc", compare_to="production", min_delta=0.005, slice="plan_type=pro")
    ]
    poison = BootstrapDelta(point=0.01, lower=-0.5, confidence=0.95, n_resamples=1000)
    result = _eval(
        gates,
        {"pr_auc": 0.9},
        champion={"pr_auc": 0.9},
        version="7",
        bounds={"pr_auc": poison},  # whole-split bound for the same metric
        challenger_slices={"plan_type=pro": {"pr_auc": 0.46}},
        champion_slices={"plan_type=pro": {"pr_auc": 0.44}},
    )[0]
    assert result.passed  # slice delta 0.02 >= 0.005 on the point criterion
    assert result.delta_lower is None
    assert result.actual_delta == pytest.approx(0.02)


def test_slice_gate_missing_slice_metrics_is_hard_error() -> None:
    gates = [GateSpec(metric="pr_auc", threshold=0.4, slice="plan_type=vip")]
    with pytest.raises(MbtError, match="no slice metrics"):
        _eval(gates, {"pr_auc": 0.9}, challenger_slices={"plan_type=pro": {"pr_auc": 0.5}})


def test_missing_metric_is_hard_error_not_quality_failure() -> None:
    gates = [GateSpec(metric="pr_auc", threshold=0.4)]
    with pytest.raises(MbtError, match="was not computed"):
        _eval(gates, {"roc_auc": 0.9})


def test_all_gates_passed() -> None:
    gates = [
        GateSpec(metric="pr_auc", threshold=0.4),
        GateSpec(metric="logloss", threshold=0.5),
    ]
    results = _eval(gates, {"pr_auc": 0.45, "logloss": 0.6})
    assert not all_gates_passed(results)
    assert [r.passed for r in results] == [True, False]
