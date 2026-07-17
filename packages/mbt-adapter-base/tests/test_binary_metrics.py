"""Builtin binary-metric dispatch and helpers (TSD §5.7)."""

import numpy as np
import pytest

from mbt_adapter_base.metrics import (
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


def test_recall_at_precision_on_separable_scores() -> None:
    spec = MetricSpec(name="recall_at_precision", params={"precision": 1.0})
    assert compute_binary_metric(spec, _Y_TRUE, _Y_SCORE) == 1.0


def test_precision_at_recall_on_separable_scores() -> None:
    spec = MetricSpec(name="precision_at_recall", params={"recall": 1.0})
    assert compute_binary_metric(spec, _Y_TRUE, _Y_SCORE) == 1.0


def test_single_class_slice_is_skipped() -> None:
    y_true = np.array([0, 1, 0, 1, 1, 1])
    y_score = np.array([0.2, 0.8, 0.3, 0.7, 0.9, 0.6])
    slices = {"plan": np.array(["basic", "basic", "basic", "basic", "premium", "premium"])}
    results = compute_results([MetricSpec(name="roc_auc")], y_true, y_score, slices)
    assert "plan=basic" in results.slices
    assert "plan=premium" not in results.slices  # single-class: metrics undefined
