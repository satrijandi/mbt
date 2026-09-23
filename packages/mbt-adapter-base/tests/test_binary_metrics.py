"""Builtin binary-metric dispatch and helpers (TSD §5.7)."""

import numpy as np
import pytest

from mbt_adapter_base.metrics import (
    _slice_groups,
    compute_binary_metric,
    compute_results,
    is_builtin_binary_metric,
)
from mbt_adapter_base.specs import MetricSpec

#: A tiny separable set: positives score high, negatives score low.
_Y_TRUE = np.array([0, 0, 0, 0, 1, 1, 1, 1])
_Y_SCORE = np.array([0.05, 0.1, 0.2, 0.3, 0.7, 0.8, 0.9, 0.95])


def test_is_builtin_binary_metric() -> None:
    assert is_builtin_binary_metric("roc_auc")
    assert is_builtin_binary_metric("recall_at_precision_0.9")
    assert not is_builtin_binary_metric("business_value")


def test_unknown_metric_rejected() -> None:
    with pytest.raises(ValueError, match="unknown builtin binary metric"):
        compute_binary_metric(MetricSpec(name="business_value"), _Y_TRUE, _Y_SCORE)


def test_accuracy_uses_threshold_param() -> None:
    default = compute_binary_metric(MetricSpec(name="accuracy"), _Y_TRUE, _Y_SCORE)
    assert default == 1.0
    strict = compute_binary_metric(
        MetricSpec(name="accuracy", params={"threshold": 0.85}), _Y_TRUE, _Y_SCORE
    )
    assert strict == pytest.approx(6 / 8)


def test_ece_zero_for_perfectly_calibrated_scores() -> None:
    y_true = np.array([0, 1, 0, 1])
    y_score = np.array([0.0, 1.0, 0.0, 1.0])
    assert compute_binary_metric(MetricSpec(name="ece"), y_true, y_score) == 0.0


def test_ece_reflects_miscalibration() -> None:
    value = compute_binary_metric(MetricSpec(name="ece", params={"n_bins": 4}), _Y_TRUE, _Y_SCORE)
    assert 0.0 < value < 1.0


def test_ece_uses_equal_frequency_bins() -> None:
    # Skewed scores: equal-frequency bins put [0.1, 0.2] (both negative) in one
    # bin and [0.3, 0.9] (both positive) in the other, giving
    # 0.5*|0.15-0| + 0.5*|0.6-1| = 0.275. Fixed-width bins split at 0.5 instead
    # ([0.1, 0.2, 0.3] vs [0.9]) and would report 0.125, so this pins the switch.
    y_true = np.array([0, 0, 1, 1])
    y_score = np.array([0.1, 0.2, 0.3, 0.9])
    value = compute_binary_metric(MetricSpec(name="ece", params={"n_bins": 2}), y_true, y_score)
    assert value == pytest.approx(0.275)


def test_recall_at_precision_on_separable_scores() -> None:
    spec = MetricSpec(name="recall_at_precision", params={"precision": 1.0})
    assert compute_binary_metric(spec, _Y_TRUE, _Y_SCORE) == 1.0


def test_precision_at_recall_on_separable_scores() -> None:
    spec = MetricSpec(name="precision_at_recall", params={"recall": 1.0})
    assert compute_binary_metric(spec, _Y_TRUE, _Y_SCORE) == 1.0


def test_high_cardinality_numeric_slice_is_quantile_binned() -> None:
    # 51 distinct ages must not explode into 51 one-row slices (R2-9)
    age = np.array(list(range(20, 71)))
    assert list(_slice_groups(age)) == ["[20, 32.5)", "[32.5, 45)", "[45, 57.5)", "[57.5, 70]"]


def test_low_cardinality_and_categorical_slices_stay_per_value() -> None:
    assert sorted(_slice_groups(np.array(["a", "b", "a"]))) == ["a", "b"]
    assert sorted(_slice_groups(np.array([1, 2, 3, 4, 5]))) == ["1", "2", "3", "4", "5"]


def test_concentrated_numeric_slice_falls_back_to_per_value() -> None:
    # 14 distinct values but 75% are 0, so the quantile edges collapse below the
    # 3 needed to bin; fall back to per-value rather than a single useless bin
    skewed = np.concatenate([np.zeros(40), np.arange(1, 14)])
    groups = _slice_groups(skewed)
    assert "0.0" in groups and not any(k.startswith("[") for k in groups)


def test_compute_results_reports_binned_slice_metrics() -> None:
    n = 40
    age = np.array(list(range(n)))  # 40 distinct -> quartile ranges, not per-value
    y_true = np.array([i % 2 for i in range(n)])  # both labels in every bin
    y_score = np.linspace(0.1, 0.9, n)
    results = compute_results([MetricSpec(name="roc_auc")], y_true, y_score, {"age": age})
    assert any(k.startswith("age=[") for k in results.slices)  # range labels
    assert "age=0" not in results.slices  # not one slice per distinct age


def test_single_class_slice_is_skipped() -> None:
    y_true = np.array([0, 1, 0, 1, 1, 1])
    y_score = np.array([0.2, 0.8, 0.3, 0.7, 0.9, 0.6])
    slices = {"plan": np.array(["basic", "basic", "basic", "basic", "premium", "premium"])}
    results = compute_results([MetricSpec(name="roc_auc")], y_true, y_score, slices)
    assert "plan=basic" in results.slices
    assert "plan=premium" not in results.slices  # single-class: metrics undefined


# -- D-1: an operating point is a fitted parameter, so count its support -----


def test_operating_point_support_counts_the_rows_at_or_above_the_cutoff() -> None:
    import numpy as np

    from mbt_adapter_base.metrics import operating_point_support

    scores = np.array([0.1, 0.5, 0.5, 0.9])
    assert operating_point_support(scores, 0.5) == 3  # inclusive, as the rule is
    assert operating_point_support(scores, 0.95) == 0
    assert operating_point_support(scores, 0.0) == 4


def test_a_high_precision_target_on_rare_positives_is_thinly_supported() -> None:
    """D-1's danger zone, as a regression test.

    The cutoff is selected by scanning the test split's own PR curve and the
    precision is then reported from the same rows, so a rare-positive,
    high-precision target produces a cutoff resting on a handful of rows - and
    the realized precision on fresh rows is materially lower than the number
    shown. ``churn_demo`` sits comfortably outside this corner, which is why
    no test caught it before v5.
    """
    import numpy as np

    from mbt_adapter_base.metrics import (
        MIN_OPERATING_POINT_SUPPORT,
        compute_metric,
        operating_point_support,
    )
    from mbt_adapter_base.specs import MetricSpec

    rng = np.random.default_rng(11)
    n, base = 5000, 0.02
    labels = (rng.random(n) < base).astype(float)
    scores = 1 / (1 + np.exp(-(rng.normal(0, 1, n) + 2.2 * labels)))

    cutoff = compute_metric(MetricSpec(name="threshold_at_precision_0.7"), labels, scores)
    support = operating_point_support(scores, cutoff)
    assert support < MIN_OPERATING_POINT_SUPPORT  # the corner the warning names

    # the reported precision meets the target ON THESE ROWS ...
    selected = scores >= cutoff
    assert float(labels[selected].mean()) >= 0.7
    # ... and falls short of it on an independent draw from the same process
    fresh_labels = (rng.random(n) < base).astype(float)
    fresh_scores = 1 / (1 + np.exp(-(rng.normal(0, 1, n) + 2.2 * fresh_labels)))
    fresh = fresh_scores >= cutoff
    assert float(fresh_labels[fresh].mean()) < 0.7


def test_a_well_supported_cutoff_is_not_flagged() -> None:
    """The other half of the table: the bias is negligible once the cutoff
    rests on roughly a hundred rows, so the warning must stay quiet there."""
    import numpy as np

    from mbt_adapter_base.metrics import (
        MIN_OPERATING_POINT_SUPPORT,
        compute_metric,
        operating_point_support,
    )
    from mbt_adapter_base.specs import MetricSpec

    rng = np.random.default_rng(7)
    n, base = 20000, 0.2  # the churn_demo shape
    labels = (rng.random(n) < base).astype(float)
    scores = 1 / (1 + np.exp(-(rng.normal(0, 1, n) + 2.0 * labels)))

    cutoff = compute_metric(MetricSpec(name="threshold_at_precision_0.35"), labels, scores)
    assert operating_point_support(scores, cutoff) >= MIN_OPERATING_POINT_SUPPORT
