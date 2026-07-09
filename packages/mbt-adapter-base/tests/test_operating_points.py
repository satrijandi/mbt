"""Operating-point builtin metrics: threshold_at_precision / threshold_at_recall
(section 3.3/3.8: probability models need a deployable decision rule, not a
fixed 0.5)."""

import numpy as np
import pytest

from mbt_adapter_base.metrics import compute_binary_metric, parse_metric_sugar
from mbt_adapter_base.specs import MetricSpec

Y = np.array([1, 1, 0, 1, 0, 0], dtype=float)
SCORES = np.array([0.9, 0.8, 0.7, 0.6, 0.5, 0.4])


def _spec(name: str) -> MetricSpec:
    return MetricSpec(name=name, kind="builtin")


def test_sugar_parses_thresholds() -> None:
    assert parse_metric_sugar("threshold_at_precision_0.8") == (
        "threshold_at_precision",
        {"precision": 0.8},
    )
    assert parse_metric_sugar("threshold_at_recall_0.66") == (
        "threshold_at_recall",
        {"recall": 0.66},
    )
    assert parse_metric_sugar("threshold_at_precision_high") is None
    assert parse_metric_sugar("threshold_at_recall_1.5") is None  # out of range


def test_threshold_at_precision_hand_computed() -> None:
    # thresholds sweep: t=0.6 -> precision 0.75; t=0.7 -> 0.667; t=0.8 -> 1.0
    assert compute_binary_metric(_spec("threshold_at_precision_0.7"), Y, SCORES) == pytest.approx(
        0.6
    )
    # perfect precision is first reached at t=0.8 (top-2 are both positive)
    assert compute_binary_metric(_spec("threshold_at_precision_1.0"), Y, SCORES) == pytest.approx(
        0.8
    )


def test_threshold_at_recall_hand_computed() -> None:
    # full recall needs every positive: the largest such threshold is 0.6
    assert compute_binary_metric(_spec("threshold_at_recall_1.0"), Y, SCORES) == pytest.approx(0.6)
    # recall 2/3 is still met at t=0.8, where precision is perfect
    assert compute_binary_metric(_spec("threshold_at_recall_0.66"), Y, SCORES) == pytest.approx(0.8)


def test_returned_threshold_delivers_the_target() -> None:
    """The semantic guarantee interventions rely on: applying the returned
    threshold to the same scores meets the requested target."""
    rng = np.random.default_rng(7)
    y = (rng.random(2000) < 0.2).astype(float)
    scores = np.clip(rng.random(2000) * 0.5 + y * rng.random(2000) * 0.5, 0, 1)

    t = compute_binary_metric(_spec("threshold_at_precision_0.5"), y, scores)
    predicted = scores >= t
    assert predicted.any()
    assert float(y[predicted].mean()) >= 0.5

    t = compute_binary_metric(_spec("threshold_at_recall_0.9"), y, scores)
    captured = float(y[scores >= t].sum() / y.sum())
    assert captured >= 0.9


def test_unattainable_and_degenerate_sentinels() -> None:
    # only negatives score high: precision 0.9 is unreachable -> 1.0 sentinel
    # ("predict nothing" is the only rule honoring the target)
    y = np.array([0.0, 1.0])
    scores = np.array([0.9, 0.1])
    assert compute_binary_metric(_spec("threshold_at_precision_0.9"), y, scores) == 1.0
    # degenerate labels (no positives): sentinels, never an exception
    zeros = np.zeros(4)
    constant = np.full(4, 0.5)
    assert compute_binary_metric(_spec("threshold_at_precision_0.5"), zeros, constant) == 1.0
    assert compute_binary_metric(_spec("threshold_at_recall_0.5"), zeros, constant) == 0.0
