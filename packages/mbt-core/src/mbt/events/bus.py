"""The process-wide event bus.

The coordinator installs sinks according to CLI flags; training jobs install
a JSON sink on stdout so the coordinator can forward their events (TSD §16).
"""

import threading

from mbt.events.models import Event
from mbt.events.sinks import Sink
from mbt_adapter_base.events import as_event


class EventBus:
    """Fans events out to sinks; also satisfies the EventSink contract."""

    def __init__(self, sinks: list[Sink] | None = None, run_id: str | None = None) -> None:
        self._sinks: list[Sink] = list(sinks or [])
        self._lock = threading.Lock()
        self.run_id = run_id

    def add_sink(self, sink: Sink) -> None:
        with self._lock:
            self._sinks.append(sink)

    def remove_sink(self, sink: Sink) -> None:
        """Detach a sink added for one invocation (a missing one is a no-op)."""
        with self._lock:
            if sink in self._sinks:
                self._sinks.remove(sink)

    def emit(self, event: Event) -> None:
        if event.run_id is None and self.run_id is not None:
            event.run_id = self.run_id
        with self._lock:
            sinks = list(self._sinks)
        for sink in sinks:
            sink.write(event)


class HookEventSink:
    """The bus as a user's ``hooks.py`` sees it: the one tolerant boundary.

    The event seam is typed (B-4), but a hook is user code and may reasonably
    emit a bare string. Coercing there rather than in ``EventBus.emit`` keeps
    the guess where foreign objects genuinely arrive, instead of applying it to
    every adapter and every core call site - which is how an adapter's warning
    silently became an info line (v5 live defect 1).
    """

    def __init__(self, bus: EventBus) -> None:
        self._bus = bus

    def emit(self, event: object) -> None:
        self._bus.emit(as_event(event))


_bus = EventBus()


def get_bus() -> EventBus:
    return _bus


def set_bus(bus: EventBus) -> None:
    global _bus  # noqa: PLW0603 - one bus per process by design
    _bus = bus
