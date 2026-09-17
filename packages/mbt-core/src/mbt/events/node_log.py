"""Per-node run logs (ADR-30): every event a node's work produced, in one file.

The console shows what an operator needs while a run is live and forgets it.
A training run is reviewed later, by someone else, from the tracking UI - so
each node's events are also written, at every level and whatever the console
verbosity, to ``target/run_logs/<run_id>/<unique_id>.log``, and a model's log
is uploaded to its tracking run.

Attribution needs no cooperation from the code that emits: a runner enters
:func:`node_scope` for the node it works on, and an event that names no node
is filed under the node whose scope was active in the emitting thread. That
covers the lines a job subprocess forwards (framework output carries no node)
and the plain strings data adapters emit, without changing what the console
renders.
"""

import threading
from collections.abc import Iterator
from contextlib import contextmanager
from contextvars import ContextVar
from datetime import UTC
from pathlib import Path
from typing import TextIO

from mbt.events.models import Event
from mbt.secrets import redact

#: Events that name no node and were emitted outside any node's scope:
#: compile, run start and finish.
INVOCATION_LOG = "invocation.log"

_current_node: ContextVar[str | None] = ContextVar("mbt_current_node", default=None)


@contextmanager
def node_scope(unique_id: str) -> Iterator[None]:
    """File events without a ``unique_id`` under this node while active."""
    token = _current_node.set(unique_id)
    try:
        yield
    finally:
        _current_node.reset(token)


def current_node() -> str | None:
    return _current_node.get()


def format_event(event: Event) -> str:
    """``2026-06-30T00:00:01.250Z INFO  message``, redacted."""
    stamp = event.ts.astimezone(UTC).isoformat(timespec="milliseconds").replace("+00:00", "Z")
    return f"{stamp} {event.level.upper():<5} {redact(event.human())}"


class NodeLogSink:
    """Appends each event to its node's log file under one run directory."""

    def __init__(self, directory: Path) -> None:
        self.directory = directory
        self._lock = threading.Lock()
        self._files: dict[str, TextIO] = {}

    def path_for(self, unique_id: str | None) -> Path:
        return self.directory / (f"{unique_id}.log" if unique_id else INVOCATION_LOG)

    def write(self, event: Event) -> None:
        unique_id = event.unique_id or current_node()
        line = format_event(event) + "\n"
        with self._lock:
            handle = self._files.get(unique_id or "")
            if handle is None:
                path = self.path_for(unique_id)
                path.parent.mkdir(parents=True, exist_ok=True)
                handle = path.open("a", encoding="utf-8")
                self._files[unique_id or ""] = handle
            handle.write(line)
            handle.flush()

    def read(self, unique_id: str) -> str:
        """Everything logged for ``unique_id`` so far ('' when nothing was)."""
        path = self.path_for(unique_id)
        with self._lock:
            return path.read_text(encoding="utf-8") if path.is_file() else ""

    def close(self) -> None:
        with self._lock:
            for handle in self._files.values():
                handle.close()
            self._files.clear()


def combined_log(sink: NodeLogSink, unique_ids: list[str], *, header: list[str]) -> str:
    """One readable document: a header, then one section per node, in order.

    A model's uploaded log leads with its dataset's section, so the reader
    sees the rows the model trained on before the training itself.
    """
    parts = [*(f"# {line}" for line in header), ""]
    for unique_id in unique_ids:
        body = sink.read(unique_id)
        if not body:
            continue
        parts.extend([f"## {unique_id}", body.rstrip("\n"), ""])
    return "\n".join(parts).rstrip("\n") + "\n"
