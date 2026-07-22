"""Window grammar and anchoring tests (S2-04)."""

from datetime import UTC, datetime, timedelta

import pytest

from mbt.compile.windows import format_ts, is_subrange, parse_window
from mbt.exceptions import ConfigError

ANCHOR = datetime(2026, 7, 6, 12, 0, 0, tzinfo=UTC)


def test_relative_window_resolves_against_anchor() -> None:
    start, end = parse_window("-28d:now").resolve(ANCHOR)
    assert end == ANCHOR
    assert start == ANCHOR - timedelta(days=28)


def test_bare_duration_sugar() -> None:
    assert parse_window("28d").resolve(ANCHOR) == parse_window("-28d:now").resolve(ANCHOR)


@pytest.mark.parametrize(
    ("expression", "delta"),
    [("-2w:now", timedelta(weeks=2)), ("-12h:now", timedelta(hours=12))],
)
def test_week_and_hour_units(expression: str, delta: timedelta) -> None:
    start, end = parse_window(expression).resolve(ANCHOR)
    assert end - start == delta


def test_calendar_month_unit_shifts_real_months() -> None:
    # -3mo from 2026-07-06 is 2026-04-06 (a real calendar shift, not 90 days)
    start, end = parse_window("-3mo:now").resolve(ANCHOR)
    assert end == ANCHOR
    assert start == datetime(2026, 4, 6, 12, 0, tzinfo=UTC)


def test_calendar_year_unit_is_twelve_months() -> None:
    start, _ = parse_window("-1y:now").resolve(ANCHOR)
    assert start == datetime(2025, 7, 6, 12, 0, tzinfo=UTC)


def test_bare_month_duration_sugar() -> None:
    assert parse_window("3mo").resolve(ANCHOR) == parse_window("-3mo:now").resolve(ANCHOR)


def test_calendar_shift_clamps_the_day() -> None:
    # Mar 31 - 1mo clamps to the last valid day of February (leap year 2028).
    leap_end = datetime(2028, 3, 31, tzinfo=UTC)
    start, _ = parse_window("-1mo:now").resolve(leap_end)
    assert start == datetime(2028, 2, 29, tzinfo=UTC)


def test_calendar_months_span_a_full_window() -> None:
    start, end = parse_window("-6mo:-3mo").resolve(ANCHOR)
    assert start == datetime(2026, 1, 6, 12, 0, tzinfo=UTC)
    assert end == datetime(2026, 4, 6, 12, 0, tzinfo=UTC)


@pytest.mark.parametrize("expression", ["1.5mo", "-2.5y:now", "0.5mo:now"])
def test_fractional_calendar_units_are_rejected(expression: str) -> None:
    # Calendar units must be whole (int(1.5) would silently truncate to 1 month).
    with pytest.raises(ConfigError):
        parse_window(expression)


def test_absolute_bounds() -> None:
    start, end = parse_window("2026-01-01:2026-02-01").resolve(ANCHOR)
    assert start == datetime(2026, 1, 1, tzinfo=UTC)
    assert end == datetime(2026, 2, 1, tzinfo=UTC)


def test_iso_timestamp_bounds_with_colons() -> None:
    start, end = parse_window("2026-01-01T06:30:00:now").resolve(ANCHOR)
    assert start == datetime(2026, 1, 1, 6, 30, tzinfo=UTC)
    assert end == ANCHOR


def test_mixed_bounds() -> None:
    start, end = parse_window("-180d:-28d").resolve(ANCHOR)
    assert end - start == timedelta(days=152)


@pytest.mark.parametrize("expression", ["", "notawindow", "28x", "now:-28d", "-1d:-2d", ":", "a:b"])
def test_invalid_windows_raise(expression: str) -> None:
    with pytest.raises(ConfigError):
        parse_window(expression)


def test_subrange() -> None:
    outer = parse_window("-28d:now")
    assert is_subrange(parse_window("-7d:now"), outer, ANCHOR)
    assert not is_subrange(parse_window("-29d:now"), outer, ANCHOR)


def test_format_ts_is_z_suffixed() -> None:
    assert format_ts(ANCHOR) == "2026-07-06T12:00:00Z"


def test_subtract_duration_days_weeks_and_calendar_months() -> None:
    """The split embargo shift (R2-7): fixed durations for d/w/h, calendar-aware
    for months."""
    from mbt.compile.windows import subtract_duration

    base = datetime(2026, 3, 15, tzinfo=UTC)
    assert subtract_duration(base, "7d") == base - timedelta(days=7)
    assert subtract_duration(base, "2w") == base - timedelta(weeks=2)
    assert subtract_duration(base, "6h") == base - timedelta(hours=6)
    assert subtract_duration(base, "1mo") == datetime(2026, 2, 15, tzinfo=UTC)  # calendar, not 30d
