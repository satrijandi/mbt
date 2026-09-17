"""Report numerics (ADR-30): frozen bins, period cells, maturity, KS, sampling."""

from datetime import datetime

import numpy as np
import pyarrow as pa
import pytest

from mbt_adapter_base.reporting import (
    MAX_BINS,
    FittedBins,
    assign_bins,
    bin_rows,
    capped_indices,
    cell_bounds,
    cell_label,
    fit_bins,
    histogram_rows,
    horizon_delta,
    key_hash_buckets,
    ks_statistic,
    maturity_mask,
    period_codes,
    score_summary,
    slot_label,
    time_values,
    unsupported_periods,
    week_of_month,
)
from mbt_adapter_base.specs import (
    DEFAULT_CUSTOM_EDGES,
    CustomBinning,
    FixedWidthBinning,
    QuantileBinning,
    TopPercentBinning,
)


def _scores(n: int = 1000, seed: int = 0) -> tuple[np.ndarray, np.ndarray]:
    rng = np.random.default_rng(seed)
    scores = rng.beta(2, 5, n)
    labels = (rng.random(n) < scores).astype(float)
    return scores, labels


def _dates(*values: str) -> np.ndarray:
    return np.array(values, dtype="datetime64[us]")


# -- fitting -------------------------------------------------------------------------------


def test_quantile_bins_are_equal_frequency_on_the_reference() -> None:
    scores, labels = _scores()
    bins = fit_bins(QuantileBinning(bins=10), scores, probability=True)
    assert bins is not None and bins.name == "quantile_10" and bins.n_bins == 10
    assert bins.edges[0] == 0.0 and bins.edges[-1] == 1.0
    rows = bin_rows(bins, scores, labels, np.ones(scores.size, bool), binary=True)
    assert [row["n"] for row in rows] == [100] * 10
    assert [row["bin"] for row in rows] == list(range(1, 11))
    # rank 1 is the top band, and the positive rate falls towards the bottom
    assert rows[0]["label_rate"] > rows[-1]["label_rate"]
    assert rows[-1]["cum_share"] == 1.0 and rows[-1]["capture"] == 1.0


def test_frozen_edges_carry_over_to_another_split() -> None:
    reference, _ = _scores(seed=0)
    later, later_labels = _scores(seed=1)
    bins = fit_bins(QuantileBinning(bins=4), reference, probability=True)
    assert bins is not None
    rows = bin_rows(bins, later + 0.2, later_labels, np.ones(later.size, bool), binary=True)
    # the same score ranges, but a shifted population fills them unevenly
    assert [(r["lower"], r["upper"]) for r in rows] == [
        (bins.edges[i], bins.edges[i + 1]) for i in range(3, -1, -1)
    ]
    assert rows[0]["share"] > 0.5


def test_tied_quantiles_merge_rather_than_split_a_tie() -> None:
    scores = np.array([0.1] * 90 + [0.9] * 10)
    bins = fit_bins(QuantileBinning(bins=10), scores, probability=True)
    assert bins is not None and bins.edges == (0.0, 0.1, 1.0)


def test_fixed_width_defaults_and_regression_bands() -> None:
    scores, _ = _scores()
    bins = fit_bins(FixedWidthBinning(), scores, probability=True)
    assert bins is not None and bins.name == "fixed_width_0.05" and bins.n_bins == 20
    assert bins.edges[3] == 0.15  # rounded, not 0.15000000000000002
    predictions = np.linspace(10.0, 30.0, 50)
    regression = fit_bins(FixedWidthBinning(), predictions, probability=False)
    assert regression is not None and regression.name == "fixed_width"
    assert regression.n_bins == 20 and regression.edges[0] == 10.0 and regression.edges[-1] == 30.0
    flat = fit_bins(FixedWidthBinning(), np.full(5, 2.0), probability=False)
    assert flat is not None and flat.n_bins == 1
    with pytest.raises(ValueError, match="widen it"):
        fit_bins(FixedWidthBinning(width=1e-6), scores, probability=True)
    assert MAX_BINS == 1000


def test_custom_edges_default_on_probabilities_only() -> None:
    scores, _ = _scores()
    default = fit_bins(CustomBinning(), scores, probability=True)
    assert default is not None and default.edges == DEFAULT_CUSTOM_EDGES
    assert fit_bins(CustomBinning(), scores, probability=False) is None
    declared = fit_bins(CustomBinning(edges=[0.2, 0.5]), scores, probability=True)
    assert declared is not None and declared.n_bins == 1
    # outside the declared edges still lands in the outermost band
    assert assign_bins(declared, np.array([0.0, 0.9])).tolist() == [0, 0]


def test_top_percent_thresholds_come_from_the_reference() -> None:
    scores, labels = _scores()
    bins = fit_bins(TopPercentBinning(cutoffs=[10, 50]), scores, probability=True)
    assert bins is not None and bins.n_bins == 2
    rows = bin_rows(bins, scores, labels, np.ones(scores.size, bool), binary=True)
    assert [row["top_percent"] for row in rows] == [10, 50]
    assert rows[0]["share"] == pytest.approx(0.10, abs=0.002)
    assert rows[0]["precision"] > rows[1]["precision"]
    assert rows[1]["capture"] > rows[0]["capture"]
    regression = bin_rows(bins, scores, scores, np.ones(scores.size, bool), binary=False)
    assert "label_mean" in regression[0] and "capture" not in regression[0]


def test_fitting_needs_reference_rows() -> None:
    with pytest.raises(ValueError, match="empty reference"):
        fit_bins(QuantileBinning(), np.array([np.nan]), probability=True)


def test_unlabelled_rows_count_as_scored_not_observed() -> None:
    bins = FittedBins(name="custom", strategy="custom", edges=(0.0, 0.5, 1.0))
    scores = np.array([0.9, 0.8, 0.1, 0.2])
    labels = np.array([1.0, np.nan, 0.0, np.nan])
    labelled = np.array([True, False, True, False])
    top, bottom = bin_rows(bins, scores, labels, labelled, binary=True)
    assert (top["n"], top["n_labelled"], top["positives"], top["label_rate"]) == (2, 1, 1, 1.0)
    assert bottom["label_rate"] == 0.0 and bottom["ks"] == 0.0
    regression = bin_rows(bins, scores, np.array([3.0, 1.0, 1.0, 1.0]), labelled, binary=False)
    assert regression[0]["label_mean"] == 3.0 and "positives" not in regression[0]


def test_empty_splits_produce_empty_bands() -> None:
    bins = FittedBins(name="custom", strategy="custom", edges=(0.0, 0.5, 1.0))
    rows = bin_rows(bins, np.array([]), np.array([]), np.array([], bool), binary=True)
    assert [row["n"] for row in rows] == [0, 0]
    assert rows[0]["share"] is None and rows[0]["ks"] is None


def test_histogram_splits_each_band_by_class() -> None:
    bins = FittedBins(name="custom", strategy="custom", edges=(0.0, 0.5, 1.0))
    rows = histogram_rows(
        np.array([0.1, 0.2, 0.7, 0.9]),
        np.array([0.0, 1.0, 1.0, np.nan]),
        np.array([True, True, True, False]),
        bins=bins,
    )
    assert rows[0] == {
        "lower": 0.0,
        "upper": 0.5,
        "n": 2,
        "share": 0.5,
        "share_positive": 0.5,
        "share_negative": 1.0,
    }
    assert rows[1]["share_positive"] == 0.5 and rows[1]["share_negative"] == 0.0
    empty = histogram_rows(np.array([]), np.array([]), np.array([], bool), bins=bins)
    assert empty[0]["share"] is None


# -- periods --------------------------------------------------------------------------------


def test_week_of_month_boundaries() -> None:
    assert [week_of_month(d) for d in (1, 7, 8, 14, 15, 28, 29, 31)] == [1, 1, 2, 2, 3, 4, 5, 5]


def _labels(times: np.ndarray, period: str) -> tuple[list[str], list[str]]:
    codes = period_codes(times, period)  # type: ignore[arg-type]
    cells = [
        cell_label(int(c), period) if ok else ""  # type: ignore[arg-type]
        for c, ok in zip(codes.cells, codes.valid, strict=True)
    ]
    slots = [
        slot_label(int(s), period) if ok else ""  # type: ignore[arg-type]
        for s, ok in zip(codes.slots, codes.valid, strict=True)
    ]
    return cells, slots


def test_period_codes_and_labels() -> None:
    times = _dates("2026-06-01", "2026-06-09T13:00", "2026-07-31", "NaT", "1969-12-31")
    assert _labels(times, "month") == (
        ["2026-06", "2026-06", "2026-07", "", "1969-12"],
        ["all", "all", "all", "", "all"],
    )
    assert _labels(times, "week_of_month") == (
        ["2026-06/W1", "2026-06/W2", "2026-07/W5", "", "1969-12/W5"],
        ["W1", "W2", "W5", "", "W5"],
    )
    assert _labels(times, "day_of_month") == (
        ["2026-06-01", "2026-06-09", "2026-07-31", "", "1969-12-31"],
        ["D01", "D09", "D31", "", "D31"],
    )
    # the same slot in two months shares a slot code but not a cell code
    codes = period_codes(_dates("2026-06-03", "2026-07-03"), "day_of_month")
    assert codes.slots.tolist() == [3, 3]
    assert codes.cells[0] != codes.cells[1]


def test_cell_bounds_follow_the_calendar() -> None:
    assert cell_bounds("2026-02", "month") == (datetime(2026, 2, 1), datetime(2026, 3, 1))
    assert cell_bounds("2026-06/W1", "week_of_month") == (
        datetime(2026, 6, 1),
        datetime(2026, 6, 8),
    )
    # W5 stops at the end of the month
    assert cell_bounds("2026-02/W4", "week_of_month")[1] == datetime(2026, 3, 1)
    assert cell_bounds("2026-06/W5", "week_of_month") == (
        datetime(2026, 6, 29),
        datetime(2026, 7, 1),
    )
    assert cell_bounds("2026-12-31", "day_of_month") == (
        datetime(2026, 12, 31),
        datetime(2027, 1, 1),
    )


def test_unsupported_periods_follow_the_data_resolution() -> None:
    monthly = _dates("2026-01-01", "2026-02-01", "2026-03-01")
    assert set(unsupported_periods(monthly, ["week_of_month", "day_of_month"])) == {
        "week_of_month",
        "day_of_month",
    }
    weekly = _dates(*[f"2026-01-{d:02d}" for d in (1, 8, 15, 22, 29)])
    assert set(unsupported_periods(weekly, ["week_of_month", "day_of_month"])) == {"day_of_month"}
    daily = np.arange("2026-01-01", "2026-03-01", dtype="datetime64[D]").astype("datetime64[us]")
    assert unsupported_periods(daily, ["week_of_month", "day_of_month"]) == {}
    assert unsupported_periods(_dates("NaT"), ["day_of_month"]) == {}


def test_time_values_reads_chunked_columns() -> None:
    column = pa.chunked_array([["2026-06-01"], [None]])
    values = time_values(column)
    assert str(values[0]) == "2026-06-01T00:00:00.000000" and np.isnat(values[1])


# -- maturity --------------------------------------------------------------------------------


def test_horizon_delta_units() -> None:
    assert horizon_delta("2mo") == (2, horizon_delta("0d")[1])
    assert horizon_delta("30d")[1].days == 30
    assert horizon_delta("2w")[1].days == 14
    assert horizon_delta("-12h")[1].total_seconds() == 12 * 3600


def test_maturity_needs_the_horizon_to_have_passed_and_a_label() -> None:
    times = _dates("2026-05-01", "2026-06-10", "2026-05-01", "NaT")
    labelled = np.array([True, True, False, True])
    anchor = datetime(2026, 6, 30)
    # a row with no time cannot be placed in a cell, so it is never mature
    assert maturity_mask(times, labelled, horizon=None, anchor=anchor).tolist() == [
        True,
        True,
        False,
        False,
    ]
    assert maturity_mask(times, labelled, horizon="30d", anchor=anchor).tolist() == [
        True,
        False,
        False,
        False,
    ]


@pytest.mark.parametrize(
    ("time", "anchor", "mature"),
    [
        # calendar months clamp the day: May 31 + 1mo is June 30
        ("2026-05-31", datetime(2026, 6, 30), True),
        ("2026-05-31T01:00", datetime(2026, 6, 30), False),  # the time of day survives
        ("2026-01-31", datetime(2026, 2, 28), True),
        ("2026-12-15", datetime(2027, 1, 15), True),  # across a year boundary
        ("2026-12-16", datetime(2027, 1, 15), False),
    ],
)
def test_calendar_month_horizons_clamp_like_windows(
    time: str, anchor: datetime, mature: bool
) -> None:
    result = maturity_mask(_dates(time), np.array([True]), horizon="1mo", anchor=anchor)
    assert result.tolist() == [mature]


def test_hour_horizons() -> None:
    times = _dates("2026-06-29T12:00", "2026-06-29T12:01")
    result = maturity_mask(times, np.ones(2, bool), horizon="12h", anchor=datetime(2026, 6, 30))
    assert result.tolist() == [True, False]


# -- KS and summaries ---------------------------------------------------------------------------


def test_ks_matches_the_roc_definition() -> None:
    from sklearn.metrics import roc_curve

    scores, labels = _scores()
    fpr, tpr, _ = roc_curve(labels, scores)
    assert ks_statistic(labels, scores) == pytest.approx(float(np.max(tpr - fpr)))
    assert ks_statistic(np.array([1.0, 0.0]), np.array([0.9, 0.1])) == 1.0
    # tied scores cannot be separated
    assert ks_statistic(np.array([1.0, 0.0]), np.array([0.5, 0.5])) == 0.0
    assert np.isnan(ks_statistic(np.array([1.0, 1.0]), np.array([0.2, 0.3])))


def test_score_summary() -> None:
    assert score_summary(np.array([np.nan])) == {"n": 0}
    summary = score_summary(np.array([0.0, 0.5, 1.0]))
    assert summary["n"] == 3 and summary["p50"] == 0.5 and summary["p100"] == 1.0


# -- sampling ----------------------------------------------------------------------------------


def test_key_hash_matches_the_canonical_reference() -> None:
    import hashlib

    table = pa.table({"user_id": [7, None], "day": ["2026-06-01", "2026-06-02"]})
    expected = [
        int(hashlib.md5(b"7|2026-06-01").hexdigest()[16:32], 16),
        int(hashlib.md5(b"|2026-06-02").hexdigest()[16:32], 16),
    ]
    assert key_hash_buckets(table, ["user_id", "day"]) == expected


def test_capped_sample_is_deterministic_and_nested() -> None:
    table = pa.table({"user_id": list(range(100))})
    assert capped_indices(table, ["user_id"], None) == list(range(100))
    assert capped_indices(table, ["user_id"], 500) == list(range(100))
    ten = capped_indices(table, ["user_id"], 10)
    twenty = capped_indices(table, ["user_id"], 20)
    assert len(ten) == 10 and ten == sorted(ten)
    assert set(ten) <= set(twenty)
    assert capped_indices(table, ["user_id"], 10) == ten
