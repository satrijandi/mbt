"""PSI/KS shift statistics and baseline behavior (ADR-20/21)."""

import numpy as np
import pyarrow as pa
from hypothesis import given, settings
from hypothesis import strategies as st
from hypothesis.extra.numpy import arrays as np_arrays

from mbt_adapter_base.monitoring import (
    MonitoringBaseline,
    build_baseline,
    compute_monitor_stats,
    read_baseline,
    write_baseline,
)
from mbt_adapter_base.specs import FeatureShiftSpec, MonitorsSpec, PredictionShiftSpec


def _rng() -> np.random.Generator:
    return np.random.default_rng(42)


def _feature_table(n: int, *, loc: float = 0.0, plan: str = "basic") -> pa.Table:
    rng = _rng()
    return pa.table(
        {
            "tenure": rng.normal(loc, 1.0, n),
            "spend": rng.uniform(0.0, 100.0, n),
            "plan_type": [plan if i % 3 else "premium" for i in range(n)],
        }
    )


def _baseline(n: int = 4000) -> MonitoringBaseline:
    scores = _rng().beta(2.0, 5.0, 1000)
    return build_baseline(
        _feature_table(n),
        ["tenure", "spend", "plan_type"],
        scores,
        model_name="churn_classifier",
    )


def _monitors(method: str = "psi") -> MonitorsSpec:
    return MonitorsSpec(
        feature_shift=FeatureShiftSpec(method=method, threshold=0.2),  # type: ignore[arg-type]
        prediction_shift=PredictionShiftSpec(method=method, threshold=0.2),  # type: ignore[arg-type]
    )


def test_baseline_roundtrip(tmp_path) -> None:
    baseline = _baseline()
    path = tmp_path / "baseline.json"
    write_baseline(baseline, path)
    assert read_baseline(path) == baseline
    assert set(baseline.features) == {"tenure", "spend", "plan_type"}
    assert baseline.features["plan_type"].kind == "categorical"
    assert len(baseline.score.quantiles) == 101


def test_identical_distribution_has_near_zero_shift() -> None:
    baseline = _baseline()
    scores = _rng().beta(2.0, 5.0, 800)
    for method in ("psi", "ks"):
        stats = compute_monitor_stats(baseline, _feature_table(800), scores, _monitors(method))
        for name, stat in stats.feature_shift.items():
            assert stat.value < 0.1, f"{method} on unshifted {name} = {stat.value}"
        assert stats.prediction_shift is not None
        assert stats.prediction_shift.value < 0.1
        assert stats.skipped_features == []


def test_shifted_distribution_is_flagged() -> None:
    baseline = _baseline()
    shifted = _feature_table(800, loc=2.5)  # tenure shifted by 2.5 sigma
    scores = _rng().beta(5.0, 2.0, 800)  # score distribution flipped
    for method, floor in (("psi", 0.5), ("ks", 0.3)):
        stats = compute_monitor_stats(baseline, shifted, scores, _monitors(method))
        assert stats.feature_shift["tenure"].value > floor
        assert stats.prediction_shift is not None
        assert stats.prediction_shift.value > floor


def test_shift_statistics_are_deterministic() -> None:
    baseline = _baseline()
    table = _feature_table(500, loc=1.0)
    scores = _rng().beta(2.0, 5.0, 500)
    first = compute_monitor_stats(baseline, table, scores, _monitors())
    second = compute_monitor_stats(baseline, table, scores, _monitors())
    assert first == second


def test_unseen_categories_pool_into_other() -> None:
    baseline = _baseline()
    novel = _feature_table(600, plan="brand_new_plan")
    stats = compute_monitor_stats(baseline, novel, _rng().beta(2.0, 5.0, 600), _monitors())
    assert stats.feature_shift["plan_type"].value > 0.2


def test_significance_switches_categorical_stats_to_chi_square() -> None:
    """F15: a ``method: ks`` monitor WITH significance computes a categorical
    feature's Pearson chi-square (df set, judged against the chi-square
    critical value downstream); without significance the fixed-threshold
    total-variation stat is unchanged (df absent)."""
    baseline = _baseline()
    table = _feature_table(800)
    scores = _rng().beta(2.0, 5.0, 800)
    aware = MonitorsSpec(
        feature_shift=FeatureShiftSpec(method="ks", threshold=0.15, significance=0.05)
    )
    stats = compute_monitor_stats(baseline, table, scores, aware)
    plan = stats.feature_shift["plan_type"]
    assert plan.kind == "categorical" and plan.df is not None and plan.df >= 1
    assert plan.value >= 0.0  # a chi-square statistic, not a [0,1] proportion gap
    # numeric features under the same spec keep the KS stat (df stays absent)
    assert stats.feature_shift["tenure"].df is None
    # and the threshold path is untouched
    fixed = compute_monitor_stats(baseline, table, scores, _monitors("ks"))
    assert fixed.feature_shift["plan_type"].df is None
    assert 0.0 <= fixed.feature_shift["plan_type"].value <= 1.0


def test_categorical_chi2_is_the_contingency_form() -> None:
    """The chi-square is the TWO-SAMPLE (2xk contingency) statistic - the
    baseline shares are estimates from a finite train sample, and judging
    current counts against them as exact truth was measured ~2x
    anti-conservative (F15). Hand-checked on a tiny table, df counts the
    populated categories, and a novel category inflates the statistic."""
    from mbt_adapter_base.monitoring import FeatureBaseline, _categorical_chi2

    baseline = FeatureBaseline(
        kind="categorical",
        categories=["a", "b", "__other__"],
        proportions=[0.75, 0.25, 0.0],
        n=100,
    )
    # current: 50 a, 50 b -> baseline counts 75/25; totals 125/75; N=200
    stat, df = _categorical_chi2(baseline, ["a"] * 50 + ["b"] * 50)
    expected = 0.0
    for base_count, observed in ((75.0, 50.0), (25.0, 50.0)):
        total = base_count + observed
        expected += (base_count - total * 100 / 200) ** 2 / (total * 100 / 200)
        expected += (observed - total * 100 / 200) ** 2 / (total * 100 / 200)
    assert stat == expected and df == 1  # __other__ is unpopulated on both sides
    # a category the training data never saw still lands sharply
    novel_stat, novel_df = _categorical_chi2(baseline, ["zzz"] * 50 + ["a"] * 50)
    assert novel_stat > stat and novel_df == 2  # __other__ now populated


def test_include_exclude_globs() -> None:
    baseline = _baseline()
    monitors = MonitorsSpec(
        feature_shift=FeatureShiftSpec(threshold=0.2, include=["t*"], exclude=["tenure_id"])
    )
    stats = compute_monitor_stats(
        baseline, _feature_table(300), _rng().beta(2.0, 5.0, 300), monitors
    )
    assert set(stats.feature_shift) == {"tenure"}


def test_missing_column_reported_as_skipped() -> None:
    baseline = _baseline()
    table = _feature_table(300).drop_columns(["spend"])
    stats = compute_monitor_stats(baseline, table, _rng().beta(2.0, 5.0, 300), _monitors())
    assert "spend" in stats.skipped_features
    assert "spend" not in stats.feature_shift


def test_empty_batch_yields_no_statistics() -> None:
    baseline = _baseline()
    stats = compute_monitor_stats(
        baseline, _feature_table(0), np.asarray([], dtype=np.float64), _monitors()
    )
    assert stats.feature_shift == {}
    assert stats.prediction_shift is None


def test_constant_column_survives_duplicate_bin_edges() -> None:
    rng = _rng()
    table = pa.table({"constant": [1.0] * 1000, "noise": rng.normal(0, 1, 1000)})
    baseline = build_baseline(table, ["constant", "noise"], rng.beta(2.0, 5.0, 500), model_name="m")
    monitors = MonitorsSpec(feature_shift=FeatureShiftSpec(threshold=0.2))
    same = pa.table({"constant": [1.0] * 400, "noise": rng.normal(0, 1, 400)})
    stats = compute_monitor_stats(baseline, same, rng.beta(2.0, 5.0, 400), monitors)
    assert stats.feature_shift["constant"].value < 0.1
    moved = pa.table({"constant": [9.0] * 400, "noise": rng.normal(0, 1, 400)})
    stats = compute_monitor_stats(baseline, moved, rng.beta(2.0, 5.0, 400), monitors)
    assert stats.feature_shift["constant"].value > 0.5


# -- property-based invariants of the shift statistics ----------------------------

_FINITE = st.floats(min_value=-1e6, max_value=1e6, allow_nan=False, allow_infinity=False)


@settings(max_examples=60, deadline=None)
@given(
    baseline_values=np_arrays(np.float64, st.integers(50, 400), elements=_FINITE),
    current_values=np_arrays(np.float64, st.integers(1, 400), elements=_FINITE),
)
def test_psi_is_non_negative_and_ks_is_bounded(baseline_values, current_values) -> None:
    """For ANY input, PSI is a divergence (>= 0) and KS is a max abs ECDF gap
    (in [0, 1]) - exercised through the real quantile-grid binning, so a bin
    indexing or proportion-normalization bug would surface as a negative PSI
    or an out-of-range KS."""
    baseline = build_baseline(
        pa.table({"x": baseline_values}), ["x"], baseline_values, model_name="m"
    )
    current = pa.table({"x": current_values})
    for method in ("psi", "ks"):
        stats = compute_monitor_stats(baseline, current, current_values, _monitors(method))
        for stat in (stats.feature_shift.get("x"), stats.prediction_shift):
            assert stat is not None
            assert stat.value >= -1e-9  # float slack around the true bound of 0
            if method == "ks":
                assert stat.value <= 1.0 + 1e-9
