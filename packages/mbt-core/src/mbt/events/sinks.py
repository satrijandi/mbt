"""Event sinks: Rich console for humans, JSON lines for machines (TSD §16)."""

import sys
from typing import Protocol, TextIO

from rich.console import Console
from rich.markup import escape

from mbt.events.models import Event
from mbt.secrets import redact

_LEVEL_STYLES = {
    "debug": "dim",
    "info": "",
    "warn": "yellow",
    "error": "bold red",
}


class Sink(Protocol):
    def write(self, event: Event) -> None: ...


class NullSink:
    """Swallows everything (``--quiet``)."""

    def write(self, event: Event) -> None:
        pass


#: Width of the "HH:MM:SS  " gutter, so a wrapped message lines up under itself
#: instead of restarting in the timestamp column.
_GUTTER = len("00:00:00") + 2

#: Below this, wrapping does more harm than the overflow it prevents.
_MIN_WRAP_WIDTH = 20


class ConsoleSink:
    """Human-readable one-line-per-event rendering.

    Defaults to **stderr**: "events go to stderr, stdout is command data" is a
    load-bearing invariant (docs/architecture.md), and it should hold by
    construction rather than only because every caller remembers to pass an
    stderr console.
    """

    def __init__(self, console: Console | None = None, verbose: bool = False) -> None:
        self.console = console or Console(stderr=True, highlight=False)
        self.verbose = verbose

    def write(self, event: Event) -> None:
        if event.level == "debug" and not self.verbose:
            return
        style = _LEVEL_STYLES.get(event.level, "")
        # Local time, not UTC. event.ts is UTC-aware and strftime would print
        # the UTC wall clock with no marker, interleaved with third-party log
        # lines in local time - two clocks hours apart in one stream, neither
        # labelled (FEEDBACK v3 E-3). Machines read the ISO timestamp from the
        # JSON sink; humans reading a console want their own clock.
        stamp = event.ts.astimezone().strftime("%H:%M:%S")
        prefix = {"warn": "WARN ", "error": "ERROR "}.get(event.level, "")
        message = prefix + redact(event.human())
        first, *rest = _wrap(message, self.console.width - _GUTTER)
        self.console.print(f"[dim]{stamp}[/dim]  {escape(first)}", style=style, markup=True)
        for line in rest:
            # Hanging indent: a wrapped continuation used to restart in column
            # zero, so it read as a new event with a missing timestamp.
            self.console.print(" " * _GUTTER + escape(line), style=style, markup=True)


def _wrap(message: str, width: int) -> list[str]:
    """Split ``message`` into lines of at most ``width``, preferring word ends.

    Rich's own wrapping breaks long paths mid-token, which makes the one thing
    a user most often needs to copy out of a log - an absolute path - unusable
    (FEEDBACK v3 E-5). Words longer than the line get their own line and are
    only hard-split when they exceed the width alone, so a path stays whole
    whenever the terminal is wide enough to hold it.

    Embedded newlines are honoured as hard breaks. No built-in event renders
    one, but ``LogMessage`` carries whatever a hook or an adapter error hands
    it, and treating "a\\nb" as a single unbreakable word would mangle the
    width accounting for the whole line.
    """
    if width < _MIN_WRAP_WIDTH:  # a very narrow terminal: do not fight it
        return message.splitlines() or [message]
    lines: list[str] = []
    for paragraph in message.split("\n"):
        current = ""
        for word in paragraph.split(" "):
            candidate = f"{current} {word}" if current else word
            if len(candidate) <= width:
                current = candidate
                continue
            if current:
                lines.append(current)
            remainder = word
            while len(remainder) > width:
                lines.append(remainder[:width])
                remainder = remainder[width:]
            current = remainder
        lines.append(current)
    return lines


class JsonLinesSink:
    """One redacted JSON object per line (``--log-format json``)."""

    def __init__(self, stream: TextIO | None = None) -> None:
        self.stream = stream or sys.stdout

    def write(self, event: Event) -> None:
        line = event.model_dump_json(exclude_none=True)
        self.stream.write(redact(line) + "\n")
        self.stream.flush()
