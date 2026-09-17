"""The report builder, renderer and writer over synthetic splits (ADR-30)."""

from datetime import UTC, datetime
from pathlib import Path
from typing import Any

import numpy as np
import pyarrow as pa
import pytest

from mbt.contracts import MetricSpec, ReportSpec, StabilitySpec
from mbt.reporting import render
from mbt.reporting.builder import ReportInputs, ScoredSplit, build_report
from mbt.reporting.writer import ReportMeta, write_report

ANCHOR = datetime(2026, 7, 1, tzinfo=UTC)


def _days(start: str, count: int, step: int = 1) -> np.ndarray:
    base = np.datetime64(start, "D")
    return np.array([base + np.timedelta64(i * step, "D") for i in range(count)]).astype(
        "datetime64[us]"
    )


def _split(
    name: str,
    times: np.ndarray | None,
    *,
    seed: int,
    labels: np.ndarray | None = None,
    shift: float = 0.0,
    keys: bool = True,
) -> ScoredSplit:
    n = len(times) if times is not None else 60
    rng = np.random.default_rng(seed)
    y = labels if labels is not None else (rng.random(n) < 0.3).astype(float)
    scores = np.clip(0.2 + 0.5 * np.nan_to_num(y) + rng.normal(0, 0.15, n) + shift, 0.001, 0.999)
    features = pa.table({"x": rng.normal(shift, 1, n), "y": pa.array(y, from_pandas=True)})
    passthrough = (
        pa.table({"user_id": list(range(n)), **({"t": times} if times is not None else {})})
        if keys
        else pa.table({})
    )
    return ScoredSplit(
        name=name,
        features=features,
        passthrough=passthrough,
        key_columns=["user_id"] if keys else [],
        labels=y,
        scores=scores,
        times=times,
    )


def _inputs(**overrides: Any) -> ReportInputs:
    payload: dict[str, Any] = {
        "model_name": "m",
        "binary": True,
        "report": ReportSpec.model_validate({"binning": "all", "min_rows": 3}),
        "feature_columns": ["x"],
        "importance": {"x": 1.0},
        "metric_specs": [MetricSpec(name="pr_auc"), MetricSpec(name="roc_auc")],
        "anchor": ANCHOR,
        "horizon": None,
    }
    payload.update(overrides)
    return ReportInputs(**payload)


def _splits(after_labels: np.ndarray | None = None) -> dict[str, ScoredSplit]:
    after_times = _days("2026-05-01", 61)
    return {
        "train": _split("train", _days("2026-01-01", 60), seed=1),
        "test": _split("test", _days("2026-04-01", 30), seed=2),
        "out_of_time": _split("out_of_time", after_times, seed=3, labels=after_labels, shift=0.05),
    }


def test_a_full_binary_report() -> None:
    data = build_report(
        _inputs(stability=StabilitySpec.model_validate({"prediction_shift": {"threshold": 1}})),
        _splits(),
    )
    summary = data.summary
    assert summary.reference_rows == 30 and summary.out_of_time_rows == 61
    assert set(summary.split_metrics) == {"train", "test", "out_of_time"}
    periods = {c.period for c in summary.periods}
    assert periods == {"window", "month", "week_of_month", "day_of_month"}
    window = next(c for c in summary.periods if c.period == "window")
    assert window.start == "2026-05-01T00:00:00" and window.end == "2026-06-30T00:00:00"
    months = [c.key for c in summary.periods if c.period == "month"]
    assert months == ["2026-05", "2026-06"]
    # stability: window + months carry the gate stats, slots do not
    gated = [c.key for c in summary.stability if c.gate is not None]
    assert gated == ["2026-05", "2026-06"]
    # slot 31 exists after the test window (May 31) but not in April: no reference
    assert not any(c.key == "2026-05-31" for c in summary.stability)
    assert "binning_top_percent" in data.tables and "stability_features" in data.tables
    reference = [r for r in data.tables["performance_by_period"] if r["role"] == "reference"]
    assert reference[0]["cell"] == "test window"
    assert not any(r["cell"] == "test all" for r in reference)


def test_the_window_cell_takes_the_resolved_bounds() -> None:
    data = build_report(
        _inputs(after_window=("2026-05-01T00:00:00Z", "2026-07-01T00:00:00Z")), _splits()
    )
    window = next(c for c in data.summary.periods if c.period == "window")
    assert window.end == "2026-07-01T00:00:00"


def test_immature_single_class_and_small_cells_say_why() -> None:
    inputs = _inputs(
        report=ReportSpec.model_validate({"periods": ["month"], "min_rows": 40}),
        horizon=None,
    )
    may = [float(i % 2) for i in range(31)]
    data = build_report(inputs, _splits(after_labels=np.array(may + [np.nan] * 30)))
    notes = {c.key: c.note for c in data.summary.periods}
    assert notes["2026-06"] == "no mature labels yet"
    assert notes["window"] == "30 row(s) not mature yet"
    assert notes["2026-05"] == "fewer than 40 labelled rows"

    positives = build_report(inputs, _splits(after_labels=np.ones(61)))
    assert {c.note for c in positives.summary.periods} == {
        "a single class: ranking metrics are undefined"
    }


def test_a_horizon_holds_back_recent_rows() -> None:
    data = build_report(_inputs(horizon="30d"), _splits())
    june = next(c for c in data.summary.periods if c.key == "2026-06")
    assert not june.mature and june.n_labelled < june.n_rows


def test_an_empty_after_test_split_is_a_warning() -> None:
    splits = _splits()
    empty = splits["out_of_time"]
    splits["out_of_time"] = ScoredSplit(
        name="out_of_time",
        features=empty.features.slice(0, 0),
        passthrough=empty.passthrough.slice(0, 0),
        key_columns=["user_id"],
        labels=np.array([]),
        scores=np.array([]),
        times=np.array([], dtype="datetime64[us]"),
    )
    data = build_report(_inputs(), splits)
    assert data.summary.periods == []
    assert "the after-test window holds no rows; nothing to compare yet" in data.summary.warnings
    assert "out_of_time" not in data.summary.split_metrics


def test_a_monthly_panel_skips_the_finer_grains() -> None:
    splits = {
        "test": _split("test", np.array(["2026-03-01"] * 40, dtype="datetime64[us]"), seed=1),
        "out_of_time": _split(
            "out_of_time",
            np.array(["2026-04-01"] * 20 + ["2026-05-01"] * 20, dtype="datetime64[us]"),
            seed=2,
        ),
    }
    data = build_report(_inputs(), splits)
    assert set(data.summary.skipped_periods) == {"week_of_month", "day_of_month"}
    assert any(w.startswith("week_of_month skipped") for w in data.summary.warnings)


def test_regression_reports_means_and_skips_default_custom_edges() -> None:
    splits = _splits()
    for split in splits.values():
        split.labels = split.scores * 100 + 3
        split.scores = split.scores * 100
    inputs = _inputs(
        binary=False,
        metric_specs=[MetricSpec(name="rmse"), MetricSpec(name="mape")],
        report=ReportSpec.model_validate({"binning": "all", "periods": ["month"]}),
    )
    data = build_report(inputs, splits)
    assert "label_mean" in data.summary.split_metrics["test"]
    assert "rmse" in data.summary.split_metrics["test"]
    assert "binning_custom" not in data.tables
    assert any(w.startswith("custom binning skipped") for w in data.summary.warnings)
    assert "score_histogram" not in data.tables
    assert "binning_fixed_width" in data.tables


def test_a_strategy_that_cannot_fit_is_skipped_with_a_warning() -> None:
    inputs = _inputs(
        report=ReportSpec.model_validate({"binning": [{"strategy": "fixed_width", "width": 1e-6}]})
    )
    data = build_report(inputs, _splits())
    assert data.bins == []
    assert any("widen it" in w for w in data.summary.warnings)


def test_metrics_that_are_undefined_on_a_cell_are_left_out(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    from mbt_adapter_base import metrics

    real = metrics.compute_metric

    def flaky(spec: MetricSpec, y: Any, s: Any) -> float:
        if spec.name == "pr_auc":
            raise ValueError("undefined here")
        if spec.name == "brier":
            return float("inf")
        return float(real(spec, y, s))

    monkeypatch.setattr(metrics, "compute_metric", flaky)
    inputs = _inputs(metric_specs=[MetricSpec(name="pr_auc"), MetricSpec(name="brier")])
    data = build_report(inputs, _splits())
    test = data.summary.split_metrics["test"]
    assert "pr_auc" not in test and "brier" not in test and "roc_auc" in test


def test_predictions_are_capped_by_key_or_by_position() -> None:
    report = ReportSpec.model_validate({"predictions": {"enabled": True, "max_rows": 10}})
    data = build_report(_inputs(report=report), _splits())
    assert {name: t.num_rows for name, t in data.predictions.items()} == {
        "train": 10,
        "test": 10,
        "out_of_time": 10,
    }
    assert data.predictions["test"].column_names == ["user_id", "t", "split", "label", "p0", "p1"]
    assert data.predictions["test"].schema.field("label").type == pa.int64()
    unlabelled = build_report(_inputs(report=report), _splits(after_labels=np.full(61, np.nan)))
    assert unlabelled.predictions["out_of_time"].column("label").null_count == 10
    keyless = {"test": _split("test", None, seed=4, keys=False)}
    data = build_report(_inputs(report=report), keyless)
    assert data.predictions["test"].num_rows == 10
    regression = build_report(_inputs(report=report, binary=False), keyless)
    assert "prediction" in regression.predictions["test"].column_names
    assert regression.predictions["test"].schema.field("label").type == pa.float64()


def test_a_split_with_no_rows_is_left_out_of_the_bin_tables() -> None:
    splits = _splits()
    splits["train"] = ScoredSplit(
        name="train",
        features=pa.table({"x": pa.array([], pa.float64()), "y": pa.array([], pa.float64())}),
        passthrough=pa.table({}),
        key_columns=[],
        labels=np.array([]),
        scores=np.array([]),
        times=None,
    )
    data = build_report(_inputs(), splits)
    assert all(r["split"] != "train" for r in data.tables["binning_quantile_10"])


def test_the_report_writes_every_file_and_a_page(tmp_path: Path) -> None:
    report = ReportSpec.model_validate({"binning": "all", "predictions": {"enabled": True}})
    data = build_report(
        _inputs(
            report=report,
            stability=StabilitySpec.model_validate({"prediction_shift": {"threshold": 1}}),
        ),
        _splits(),
    )
    meta = ReportMeta(
        model="m",
        run="20260701-m",
        anchor="2026-07-01T00:00:00Z",
        binary=True,
        dataset="d",
        windows={"test": ["2026-04-01", "2026-05-01"], "validation": ["x", "y"]},
        row_counts={"test": 30},
    )
    written = write_report(data, meta, tmp_path)
    assert "performance/by_period.csv" in written and "predictions/test.parquet" in written
    assert "evaluation/binning/fixed_width_0.05.csv" in written
    page = (tmp_path / "report.html").read_text()
    for heading in ("Training report: m", "Quantile bins (10)", "Top-percent cutoffs"):
        assert heading in page
    assert "Week of month" in page and "Day of month" in page
    csv = (tmp_path / "performance" / "by_period.csv").read_text().splitlines()
    assert csv[0].startswith("role,period,cell,slot,n_rows")


def test_a_check_page_without_importance_or_train(tmp_path: Path) -> None:
    splits = _splits()
    del splits["train"]
    data = build_report(
        _inputs(importance={}, binary=False, report=ReportSpec.model_validate({"periods": []})),
        splits,
    )
    meta = ReportMeta(model="m", run="r", anchor="a", binary=False, kind="pre-deploy check")
    write_report(data, meta, tmp_path)
    page = (tmp_path / "report.html").read_text()
    assert "Pre-deploy check: m" in page
    assert "reports no importance" in page
    assert "Week of month" not in page
    # nothing on the page promises a train split the check never read
    assert "against train" not in page and "PSI vs train" not in page
    assert "<summary>the after-test window and each after-test month</summary>" in page


def test_bin_charts_leave_out_thin_bins_and_tables_keep_them(tmp_path: Path) -> None:
    report = ReportSpec.model_validate(
        {"binning": [{"strategy": "top_percent", "cutoffs": [1, 50]}], "min_rows": 20}
    )
    data = build_report(_inputs(report=report), _splits())
    write_report(data, ReportMeta(model="m", run="r", anchor="a", binary=True), tmp_path)
    page = (tmp_path / "report.html").read_text()
    start = page.index("Top-percent cutoffs")
    chart = page[start : page.index("<table", start)]
    # the top half of 30 test rows is 15 rows: in the table, not on the line
    assert "<title>test - top 50%" not in chart
    assert "<title>train (in-sample) - top 50%: " in chart  # 20 rows
    assert "<title>after test - top 50%: " in chart  # 32 rows
    assert "bins under 20 labelled rows are left off the chart" in chart
    assert "<summary>train, the after-test window and each after-test month</summary>" in page


def test_declared_metrics_come_before_the_report_s_own(tmp_path: Path) -> None:
    data = build_report(_inputs(), _splits())
    meta = ReportMeta(model="m", run="r", anchor="a", binary=True)
    write_report(data, meta, tmp_path)
    page = (tmp_path / "report.html").read_text()
    header = page[page.index("Metrics by split") :]
    header = header[: header.index("</tr>")]
    # the headline metric first, the other declared ones, then KS and the base rate
    expected = ["roc_auc", "pr_auc", "ks", "label_rate"]
    assert sorted(expected, key=header.index) == expected


def test_warnings_and_skipped_grains_reach_the_page(tmp_path: Path) -> None:
    splits = {
        "test": _split("test", np.array(["2026-03-01"] * 40, dtype="datetime64[us]"), seed=1),
        "out_of_time": _split(
            "out_of_time", np.array(["2026-04-01"] * 40, dtype="datetime64[us]"), seed=2
        ),
    }
    data = build_report(_inputs(), splits)
    write_report(data, ReportMeta(model="m", run="r", anchor="a", binary=True), tmp_path)
    page = (tmp_path / "report.html").read_text()
    assert "day_of_month not shown" in page
    assert "class='warnings" in page


# -- render --------------------------------------------------------------------------------


def test_formatters() -> None:
    assert render.fmt_number(None) == "-" and render.fmt_number(float("nan")) == "-"
    assert render.fmt_number(True) == "yes" and render.fmt_number(False) == "no"
    assert render.fmt_number(1234) == "1,234" and render.fmt_number("w") == "w"
    assert render.fmt_share(None) == "-" and render.fmt_share(float("nan")) == "-"
    assert render.fmt_share(0.5) == "50.0%"
    assert render.fmt_text(None) == "" and render.fmt_text(3) == "3"


def test_empty_charts_and_tables_render_nothing_or_a_note() -> None:
    assert render.table([], [render.Column("a", "a")]) == "<p class='muted'>nothing to show</p>"
    assert render.line_chart(render.LineChart("c", [], [])) == ""
    assert render.line_chart(render.LineChart("c", ["a"], [render.Series("s", [None])])) == ""
    assert render.hbar_chart("c", [], render.fmt_number) == ""
    assert render.column_chart("c", ["a"], [None], render.fmt_number) == ""


def test_colliding_end_labels_are_dropped_for_the_legend() -> None:
    chart = render.LineChart(
        "c",
        ["a", "b"],
        [render.Series("one", [0.5, 0.5], slot=0), render.Series("two", [0.5, 0.5], slot=1)],
        fmt=render.fmt_share,
    )
    svg = render.line_chart(chart)
    assert "one 50.0%" not in svg and "class='legend'" in svg
    apart = render.line_chart(
        render.LineChart(
            "c", ["a", "b"], [render.Series("low", [0.1, 0.1]), render.Series("high", [0.9, 0.9])]
        )
    )
    assert "low 0.1000" in apart and "high 0.9000" in apart


def test_rate_axes_stop_at_their_ceiling_and_lines_break_at_gaps() -> None:
    svg = render.line_chart(
        render.LineChart(
            "c",
            ["a", "b", "c", "d"],
            [render.Series("s", [0.2, None, 0.98, 0.99])],
            fmt=render.fmt_share,
            zero_based=True,
            ceiling=1.0,
        )
    )
    assert "100.0%" in svg and "125.0%" not in svg
    assert svg.count("<polyline") == 1  # the lone first point starts no line


def test_flat_series_and_many_categories() -> None:
    svg = render.line_chart(
        render.LineChart("c", [str(i) for i in range(40)], [render.Series("s", [0.0] * 40)])
    )
    assert svg.count("text-anchor='middle'") == 20  # every other label: they would touch
    short = render.column_chart("c", [str(i) for i in range(30)], [0.01] * 30, render.fmt_number)
    assert short.count("text-anchor='middle'") == 30  # short labels all fit
    days = [f"2026-04-{d:02d}" for d in range(1, 31)]
    long = render.column_chart("c", days, [0.01] * 30, render.fmt_number)
    assert long.count("text-anchor='middle'") == 10  # 75 units of label, 26 per slot: every 3rd


def test_long_feature_names_are_shortened_on_the_chart_only() -> None:
    name = "a_very_long_feature_name_that_keeps_going"
    svg = render.hbar_chart("c", [(name, 1.0)], render.fmt_share)
    assert f"<title>{name}: 100.0%</title>" in svg and "..." in svg


def test_all_zero_columns_and_gaps_still_draw_an_axis() -> None:
    # a bin table whose every rate is zero, and a bin with no labelled rows
    svg = render.column_chart("c", ["a", "b", "c"], [0.0, None, 0.0], render.fmt_share)
    assert "<title>a: 0.0%</title>" in svg and "<title>b:" not in svg
    assert "100.0%" in svg  # a flat zero series gets a unit axis, not a divide by zero
