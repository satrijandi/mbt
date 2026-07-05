"""Window expressions and time anchoring (TSD §5.5, §8.2, ADR-12).

A window is ``"<start>:<end>"`` where each bound is a signed duration
relative to the manifest anchor (``-180d``, ``-28d``, ``now``) or an ISO
date/timestamp; a bare duration ``"28d"`` is sugar for ``"-28d:now"``.
Durations support ``d``, ``w``, ``h``.

Windows are stored and hashed as *expressions*; they resolve to concrete
``[start_ts, end_ts)`` UTC ranges against the manifest anchor at compile
time, stored outside the hashed config.
"""

import re
from dataclasses import dataclass
from datetime import UTC, datetime, timedelta

from mbt.exceptions import ConfigError

_DURATION_RE = re.compile(r"^(?P<sign>[+-])?(?P<value>\d+(?:\.\d+)?)(?P<unit>[dwh])$")
_UNIT_SECONDS = {"d": 86_400.0, "w": 7 * 86_400.0, "h": 3_600.0}

#: A fixed reference anchor for parse-time validation (never persisted).
VALIDATION_ANCHOR = datetime(2000, 1, 1, tzinfo=UTC)


@dataclass(frozen=True)
class WindowBound:
    """One bound of a window: 'now', a signed duration, or an absolute time."""

    kind: str  # "now" | "duration" | "absolute"
    delta: timedelta | None = None
    ts: datetime | None = None

    def resolve(self, anchor: datetime) -> datetime:
        if self.kind == "now":
            return anchor
        if self.kind == "duration":
            assert self.delta is not None
            return anchor + self.delta
        assert self.ts is not None
        return self.ts


@dataclass(frozen=True)
class Window:
    """A parsed window expression."""

    expression: str
    start: WindowBound
    end: WindowBound

    def resolve(self, anchor: datetime) -> tuple[datetime, datetime]:
        """Concrete ``[start, end)`` UTC range against an anchor."""
        start, end = self.start.resolve(anchor), self.end.resolve(anchor)
        if start >= end:
            raise ConfigError(
                f"window {self.expression!r} resolves to an empty range "
                f"[{start.isoformat()}, {end.isoformat()})",
                hint="the start bound must be strictly before the end bound",
            )
        return start, end


def _parse_bound(text: str, expression: str) -> WindowBound:
    text = text.strip()
    if text == "now":
        return WindowBound(kind="now")
    match = _DURATION_RE.match(text)
    if match:
        seconds = float(match.group("value")) * _UNIT_SECONDS[match.group("unit")]
        if match.group("sign") == "-":
            seconds = -seconds
        return WindowBound(kind="duration", delta=timedelta(seconds=seconds))
    try:
        ts = datetime.fromisoformat(text)
    except ValueError:
        raise ConfigError(
            f"invalid window bound {text!r} in {expression!r}",
            hint=(
                "use a signed duration (-28d, -12h, 2w), 'now', or an ISO "
                "date/timestamp (2026-01-01, 2026-01-01T00:00:00Z)"
            ),
        ) from None
    if ts.tzinfo is None:
        ts = ts.replace(tzinfo=UTC)
    return WindowBound(kind="absolute", ts=ts.astimezone(UTC))


def parse_window(expression: str) -> Window:
    """Parse a window expression, including bare-duration sugar."""
    text = expression.strip()
    if ":" not in text or _looks_like_timestamp(text):
        # bare duration sugar: "28d" == "-28d:now"
        match = _DURATION_RE.match(text)
        if not match:
            raise ConfigError(
                f"invalid window expression {expression!r}",
                hint="expected '<start>:<end>' or a bare duration like '28d'",
            )
        value = match.group("value") + match.group("unit")
        return parse_window(f"-{value}:now")

    start_text, end_text = _split_window(text, expression)
    window = Window(
        expression=expression,
        start=_parse_bound(start_text, expression),
        end=_parse_bound(end_text, expression),
    )
    # Ordering is anchor-independent only when both bounds are relative or
    # both absolute; mixed windows are checked at compile time instead.
    kinds = {window.start.kind, window.end.kind}
    if kinds <= {"duration", "now"} or kinds == {"absolute"}:
        window.resolve(VALIDATION_ANCHOR)
    return window


def _looks_like_timestamp(text: str) -> bool:
    """True for bare ISO timestamps containing ':' (e.g. '2026-01-01T00:00:00')."""
    if not text.count(":") or "T" not in text:
        return False
    try:
        datetime.fromisoformat(text)
    except ValueError:
        return False
    return True


def _split_window(text: str, expression: str) -> tuple[str, str]:
    """Split on the separator ':', tolerating ISO timestamps in the bounds."""
    for i, char in enumerate(text):
        if char != ":":
            continue
        start, end = text[:i], text[i + 1 :]
        try:
            _parse_bound(start, expression)
            _parse_bound(end, expression)
        except ConfigError:
            continue
        return start, end
    raise ConfigError(
        f"invalid window expression {expression!r}",
        hint="expected '<start>:<end>' with duration, 'now', or ISO bounds",
    )


def is_subrange(inner: Window, outer: Window, anchor: datetime) -> bool:
    """True when ``inner`` resolves inside ``outer`` at the given anchor."""
    inner_start, inner_end = inner.resolve(anchor)
    outer_start, outer_end = outer.resolve(anchor)
    return inner_start >= outer_start and inner_end <= outer_end


def format_ts(ts: datetime) -> str:
    """Canonical manifest timestamp format: UTC, second precision, Z suffix."""
    return ts.astimezone(UTC).replace(microsecond=0).isoformat().replace("+00:00", "Z")
