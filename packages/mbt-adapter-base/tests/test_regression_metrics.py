"""Builtin regression-metric computation and name-based dispatch."""

import numpy as np
import pytest

from mbt_adapter_base.metrics import (
    compute_metric,
    compute_regression_metric,
    compute_results,
    is_builtin_regression_metric,
    paired_bootstrap_delta,
)
from mbt_adapter_base.specs import MetricSpec

_Y_TRUE = np.array([1.0, 2.0, 3.0, 4.0])
_Y_PRED = np.array([1.5, 2.5, 2.5, 3.5])  # constant +/-0.5 error


def test_is_builtin_regression_metric() -> None:
    assert is_builtin_regression_metric("rmse")
    assert is_builtin_regression_metric("r2")
    assert not is_builtin_regression_metric("roc_auc")
    assert not is_builtin_regression_metric("business_value")


def test_regression_metric_values() -> None:
    def m(name: str) -> float:
        return compute_regression_metric(MetricSpec(name=name), _Y_TRUE, _Y_PRED)

    assert m("rmse") == pytest.approx(0.5)
    assert m("mae") == pytest.approx(0.5)
    assert m("r2") == pytest.approx(0.8)  # 1 - SS_res/SS_tot = 1 - 1.0/5.0
    assert m("mape") == pytest.approx(np.mean([0.5, 0.25, 0.5 / 3, 0.125]))


def test_unknown_regression_metric_rejected() -> None:
    with pytest.raises(ValueError, match="unknown builtin regression metric"):
        compute_regression_metric(MetricSpec(name="smape"), _Y_TRUE, _Y_PRED)


def test_compute_metric_dispatches_by_name() -> None:
    # a regression name routes to the regression engine
    assert compute_metric(MetricSpec(name="rmse"), _Y_TRUE, _Y_PRED) == pytest.approx(0.5)
    # a binary name still routes to the binary engine
    y_true = np.array([0, 0, 1, 1])
    y_score = np.array([0.1, 0.2, 0.8, 0.9])
    assert compute_metric(MetricSpec(name="roc_auc"), y_true, y_score) == pytest.approx(1.0)


def test_compute_results_handles_regression_metrics_and_slices() -> None:
    specs = [
        MetricSpec(name="rmse", greater_is_better=False),
        MetricSpec(name="r2"),
    ]
    region = np.array(["a", "a", "b", "b"])
    results = compute_results(specs, _Y_TRUE, _Y_PRED, {"region": region})
    assert results.metrics["rmse"] == pytest.approx(0.5)
    assert results.metrics["r2"] == pytest.approx(0.8)
    assert set(results.slices) == {"region=a", "region=b"}
    assert results.slices["region=a"]["rmse"] == pytest.approx(0.5)


def test_paired_bootstrap_respects_regression_direction() -> None:
    # challenger predicts closer to the truth -> lower RMSE -> better
    y_true = np.linspace(0.0, 10.0, 200)
    challenger = y_true + 0.1
    champion = y_true + 1.0
    delta = paired_bootstrap_delta(
        MetricSpec(name="rmse"),
        y_true,
        challenger,
        champion,
        greater_is_better=False,  # rmse: lower is better
        confidence=0.9,
        n_resamples=200,
        seed=7,
    )
    # lower-is-better delta = champion_rmse - challenger_rmse > 0 when challenger wins
    assert delta.point > 0
    assert delta.lower > 0
