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
