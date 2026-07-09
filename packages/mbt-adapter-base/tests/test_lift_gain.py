"""lift/gain builtin metric tests: first-class lift tables (TSD §5.7)."""

import numpy as np
import pytest

from mbt_adapter_base.metrics import compute_binary_metric, parse_metric_sugar
from mbt_adapter_base.specs import MetricSpec


def _spec(name: str) -> MetricSpec:
    return MetricSpec(name=name, kind="builtin")


def test_sugar_parses_lift_and_gain() -> None:
    assert parse_metric_sugar("lift_at_0.1") == ("lift", {"fraction": 0.1})
    assert parse_metric_sugar("gain_at_0.25") == ("gain", {"fraction": 0.25})
    assert parse_metric_sugar("lift_at_decile") is None  # not a fraction
    assert parse_metric_sugar("lift_at_1.5") is None  # out of range


def test_lift_and_gain_hand_computed() -> None:
    # 10 rows, 3 positives; the top-20% by score (2 rows) are both positive
    y = np.array([1, 1, 0, 0, 0, 1, 0, 0, 0, 0], dtype=float)
    scores = np.array([0.9, 0.8, 0.7, 0.6, 0.5, 0.4, 0.3, 0.2, 0.1, 0.0])
    assert compute_binary_metric(_spec("lift_at_0.2"), y, scores) == pytest.approx(1.0 / 0.3)
    assert compute_binary_metric(_spec("gain_at_0.2"), y, scores) == pytest.approx(2 / 3)
    # full coverage captures every positive by construction
    assert compute_binary_metric(_spec("gain_at_1.0"), y, scores) == pytest.approx(1.0)


def test_random_scores_have_no_lift_and_bare_name_defaults_to_decile() -> None:
    rng = np.random.default_rng(11)
    y = (rng.random(4000) < 0.2).astype(float)
    scores = rng.random(4000)
    lift = compute_binary_metric(_spec("lift_at_0.5"), y, scores)
    assert lift == pytest.approx(1.0, abs=0.15)  # random ranking has no lift
    decile = compute_binary_metric(_spec("lift"), y, scores)
    assert decile == compute_binary_metric(_spec("lift_at_0.1"), y, scores)


def test_ties_break_deterministically_and_degenerate_labels_guard() -> None:
    y = np.array([1.0, 0.0, 0.0, 0.0])
    constant = np.array([0.5, 0.5, 0.5, 0.5])
    first = compute_binary_metric(_spec("lift_at_0.25"), y, constant)
    assert first == compute_binary_metric(_spec("lift_at_0.25"), y, constant)  # stable ties
    zeros = np.zeros(4)
    assert compute_binary_metric(_spec("lift_at_0.5"), zeros, constant) == 0.0
    assert compute_binary_metric(_spec("gain_at_0.5"), zeros, constant) == 0.0
