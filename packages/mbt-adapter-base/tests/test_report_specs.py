"""Spec surface of the training report (ADR-30): windows, gates, stability, report."""

from typing import Any

import pytest
from pydantic import ValidationError

from mbt_adapter_base.specs import (
    DEFAULT_CUSTOM_EDGES,
    DEFAULT_GATE_MIN_ROWS,
    DEFAULT_TOP_PERCENT_CUTOFFS,
    CustomBinning,
    FixedWidthBinning,
    GateSpec,
    ModelSpec,
    QuantileBinning,
    ReportSpec,
    SplitSpec,
    StabilityReport,
    StabilitySpec,
    TopPercentBinning,
)
from mbt_adapter_base.types import SplitStrategy


def _model(**evaluation: Any) -> dict[str, Any]:
    return {
        "name": "m",
        "task": "binary_classification",
        "adapter": "fake",
        "owner": "ds@example.com",
        "dataset": "ref('d')",
        "target": "y",
        "seed": 1,
        "evaluation": {
            "protocol": {"split": "temporal"},
            "metrics": ["roc_auc"],
            **evaluation,
        },
    }


def test_out_of_time_is_a_temporal_window() -> None:
    split = SplitSpec(
        time_column="ts", train="-12mo:-4mo", test="-4mo:-3mo", out_of_time="-3mo:now"
    )
    assert split.out_of_time == "-3mo:now"
    with pytest.raises(ValidationError, match="temporal strategy only"):
        SplitSpec(
            strategy=SplitStrategy.RANDOM, train="0.8", test="0.2", seed=1, out_of_time="-3mo:now"
        )


def test_after_test_gate_defaults_and_shape() -> None:
    gate = GateSpec(metric="roc_auc", threshold=0.7, source="out_of_time")
    assert gate.effective_period == "month"
    assert gate.effective_min_rows == DEFAULT_GATE_MIN_ROWS
    tuned = GateSpec(
        metric="roc_auc", threshold=0.7, source="out_of_time", period="day_of_month", min_rows=5
    )
    assert (tuned.effective_period, tuned.effective_min_rows) == ("day_of_month", 5)


@pytest.mark.parametrize(
    "payload",
    [
        {"metric": "roc_auc", "compare_to": "production", "source": "out_of_time"},
        {"metric": "roc_auc", "threshold": 0.7, "slice": "plan=pro", "source": "out_of_time"},
        {"metric": "roc_auc", "across": "plan", "source": "out_of_time"},
    ],
)
def test_after_test_gate_is_a_whole_split_threshold(payload: dict[str, Any]) -> None:
    with pytest.raises(ValidationError, match=r"whole-split threshold|disparity gate|exactly one"):
        GateSpec.model_validate(payload)


@pytest.mark.parametrize("field", [{"period": "month"}, {"min_rows": 10}])
def test_period_and_min_rows_need_the_after_test_source(field: dict[str, Any]) -> None:
    with pytest.raises(ValidationError, match="only meaningful with source: out_of_time"):
        GateSpec.model_validate({"metric": "roc_auc", "threshold": 0.7, **field})


def test_stability_needs_a_monitor_and_exposes_them() -> None:
    with pytest.raises(ValidationError, match="must declare"):
        StabilitySpec()
    spec = StabilitySpec.model_validate({"prediction_shift": {"threshold": 0.25}})
    assert spec.period == "month"
    assert spec.min_rows == DEFAULT_GATE_MIN_ROWS
    assert spec.monitors.prediction_shift is not None
    assert spec.monitors.feature_shift is None


def test_report_defaults_are_the_cheap_ones() -> None:
    report = ReportSpec()
    assert report.predictions.enabled is False
    assert report.binning_strategies == [QuantileBinning(bins=10)]
    assert report.periods == ["month", "week_of_month", "day_of_month"]
    assert report.importance.top_n == 20
    assert report.stability.engine == "native"


def test_binning_all_expands_to_every_strategy() -> None:
    report = ReportSpec.model_validate({"binning": "all"})
    strategies = report.binning_strategies
    assert [b.strategy for b in strategies] == ["quantile", "fixed_width", "custom", "top_percent"]
    custom = strategies[2]
    assert isinstance(custom, CustomBinning) and custom.edges is None
    top = strategies[3]
    assert isinstance(top, TopPercentBinning)
    assert tuple(top.cutoffs) == DEFAULT_TOP_PERCENT_CUTOFFS
    assert DEFAULT_CUSTOM_EDGES[0] == 0.0 and DEFAULT_CUSTOM_EDGES[-1] == 1.0


def test_binning_list_is_discriminated_on_strategy() -> None:
    report = ReportSpec.model_validate(
        {
            "binning": [
                {"strategy": "fixed_width", "width": 0.1},
                {"strategy": "custom", "edges": [0, 0.2, 1]},
            ]
        }
    )
    assert report.binning_strategies == [
        FixedWidthBinning(width=0.1),
        CustomBinning(edges=[0.0, 0.2, 1.0]),
    ]
    with pytest.raises(ValidationError):
        ReportSpec.model_validate({"binning": [{"strategy": "jenks"}]})


@pytest.mark.parametrize(
    ("payload", "message"),
    [
        ({"binning": []}, "at least one strategy"),
        ({"periods": ["month", "month"]}, "repeats a period"),
        ({"binning": [{"strategy": "custom", "edges": [0.5]}]}, "at least two edges"),
        ({"binning": [{"strategy": "custom", "edges": [0, 0.5, 0.5]}]}, "strictly increasing"),
        ({"binning": [{"strategy": "top_percent", "cutoffs": []}]}, "at least one cutoff"),
        ({"binning": [{"strategy": "top_percent", "cutoffs": [0, 10]}]}, r"in \(0, 100\]"),
        ({"binning": [{"strategy": "top_percent", "cutoffs": [10, 5]}]}, "strictly increasing"),
        ({"binning": [{"strategy": "quantile", "bins": 1}]}, "greater than or equal to 2"),
        ({"binning": [{"strategy": "fixed_width", "width": 0}]}, "greater than 0"),
    ],
)
def test_report_rejects_malformed_blocks(payload: dict[str, Any], message: str) -> None:
    with pytest.raises(ValidationError, match=message):
        ReportSpec.model_validate(payload)


def test_after_test_evaluation_needs_a_temporal_protocol() -> None:
    payload = _model(stability={"prediction_shift": {"threshold": 0.2}})
    payload["evaluation"]["protocol"] = {"split": "random"}
    with pytest.raises(ValidationError, match="need a temporal split"):
        ModelSpec.model_validate(payload)
    gated = _model(gates=[{"metric": "roc_auc", "threshold": 0.6, "source": "out_of_time"}])
    gated["evaluation"]["protocol"] = {"split": "random"}
    with pytest.raises(ValidationError, match="need a temporal split"):
        ModelSpec.model_validate(gated)
    assert ModelSpec.model_validate(_model(stability={"prediction_shift": {"threshold": 0.2}}))


def test_report_is_limited_to_binary_and_regression() -> None:
    payload = _model(report={})
    payload["task"] = "multiclass_classification"
    with pytest.raises(ValidationError, match="supports binary_classification and regression"):
        ModelSpec.model_validate(payload)
    regression = _model(report={"binning": "all"})
    regression["task"] = "regression"
    regression["evaluation"]["metrics"] = ["rmse"]
    assert ModelSpec.model_validate(regression).evaluation.effective_report.binning == "all"


def test_regression_custom_binning_needs_edges() -> None:
    payload = _model(report={"binning": [{"strategy": "custom"}]})
    payload["task"] = "regression"
    payload["evaluation"]["metrics"] = ["rmse"]
    with pytest.raises(ValidationError, match="needs explicit 'edges'"):
        ModelSpec.model_validate(payload)
    payload["evaluation"]["report"] = {"binning": [{"strategy": "custom", "edges": [0, 10, 100]}]}
    assert ModelSpec.model_validate(payload)


def test_effective_report_falls_back_to_defaults() -> None:
    spec = ModelSpec.model_validate(_model())
    assert spec.evaluation.report is None
    assert spec.evaluation.effective_report == ReportSpec()


def test_a_report_engine_is_named_like_an_adapter() -> None:
    assert StabilityReport.model_validate({"engine": "evidently"}).engine == "evidently"
    with pytest.raises(ValidationError):
        StabilityReport.model_validate({"engine": "Evidently AI"})
