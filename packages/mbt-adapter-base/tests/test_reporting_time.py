"""Time-axis helpers behind the after-test report and the test_window view (ADR-30)."""

from datetime import date, datetime, timedelta, timezone

import pyarrow as pa
import pytest

from mbt_adapter_base.reporting import naive_timestamps, parse_window_bound, window_mask

START = "2026-06-01T00:00:00Z"
END = "2026-06-30T00:00:00Z"


@pytest.mark.parametrize(
    "iso",
    ["2026-06-30T00:00:00Z", "2026-06-30T07:00:00+07:00", "2026-06-30", "2026-06-30T00:00:00"],
)
def test_window_bounds_read_as_naive_utc(iso: str) -> None:
    assert parse_window_bound(iso) == datetime(2026, 6, 30)


@pytest.mark.parametrize(
    "column",
    [
        pa.chunked_array([[date(2026, 5, 31), date(2026, 6, 1), date(2026, 6, 30), None]]),
        pa.array(
            [datetime(2026, 5, 31), datetime(2026, 6, 1), datetime(2026, 6, 30), None],
            type=pa.timestamp("ns"),
        ),
        pa.array(["2026-05-31", "2026-06-01", "2026-06-30", None]),
        pa.array(["2026-05-31", "2026-06-01", "2026-06-30", None]).dictionary_encode(),
    ],
)
def test_window_mask_is_half_open_and_drops_nulls(column: pa.Array) -> None:
    mask = window_mask(column, START, END)
    assert mask.to_pylist() == [False, True, False, False]


def test_tz_aware_times_compare_in_utc() -> None:
    jakarta = timezone(timedelta(hours=7))
    column = pa.array(
        [
            datetime(2026, 6, 1, 6, 59, tzinfo=jakarta),  # 2026-05-31T23:59Z: before
            datetime(2026, 6, 1, 7, 0, tzinfo=jakarta),  # 2026-06-01T00:00Z: inside
        ],
        type=pa.timestamp("us", tz="Asia/Jakarta"),
    )
    assert naive_timestamps(column).to_pylist() == [
        datetime(2026, 5, 31, 23, 59),
        datetime(2026, 6, 1),
    ]
    assert window_mask(column, START, END).to_pylist() == [False, True]
