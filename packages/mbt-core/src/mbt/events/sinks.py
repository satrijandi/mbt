"""Event sinks: Rich console for humans, JSON lines for machines (TSD §16)."""

import sys
from typing import Protocol, TextIO

from rich.console import Console

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


class ConsoleSink:
    """Human-readable one-line-per-event rendering."""

    def __init__(self, console: Console | None = None, verbose: bool = False) -> None:
        self.console = console or Console(stderr=False, highlight=False)
        self.verbose = verbose

    def write(self, event: Event) -> None:
        if event.level == "debug" and not self.verbose:
            return
        style = _LEVEL_STYLES.get(event.level, "")
        stamp = event.ts.strftime("%H:%M:%S")
        text = redact(event.human())
        prefix = {"warn": "WARN ", "error": "ERROR "}.get(event.level, "")
        self.console.print(f"[dim]{stamp}[/dim]  {prefix}{text}", style=style, markup=True)


class JsonLinesSink:
    """One redacted JSON object per line (``--log-format json``)."""

    def __init__(self, stream: TextIO | None = None) -> None:
        self.stream = stream or sys.stdout

    def write(self, event: Event) -> None:
        line = event.model_dump_json(exclude_none=True)
        self.stream.write(redact(line) + "\n")
        self.stream.flush()
