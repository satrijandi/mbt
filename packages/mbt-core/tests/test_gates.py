"""Gate engine unit tests: pure comparisons, zero ML deps (S5-01)."""

import pytest

from mbt.contracts import DeterminismTier, GateSpec, MetricResults, MetricSpec
from mbt.exceptions import MbtError
from mbt.quality.gates import all_gates_passed, evaluate_gates

SPECS = [
    MetricSpec(name="pr_auc", kind="builtin", greater_is_better=True),
    MetricSpec(name="logloss", kind="builtin", greater_is_better=False),
]


def _eval(gates, challenger, champion=None, version=None, determinism=None):
    return evaluate_gates(
        gates,
        resource="model.t.m",
        challenger=MetricResults(metrics=challenger),
        champion=MetricResults(metrics=champion) if champion else None,
        champion_version=version,
        metric_specs=SPECS,
        determinism=determinism,
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


def test_missing_champion_passes_with_null_version() -> None:
    gates = [GateSpec(metric="pr_auc", compare_to="production", min_delta=0.005)]
    result = _eval(gates, {"pr_auc": 0.3})[0]
    assert result.passed
    assert result.champion_version is None
    assert "bootstrap" in (result.message or "")


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
