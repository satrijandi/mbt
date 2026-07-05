"""The process-wide event bus.

The coordinator installs sinks according to CLI flags; training jobs install
a JSON sink on stdout so the coordinator can forward their events (TSD §16).
"""

import threading

from mbt.events.models import Event
from mbt.events.sinks import Sink


class EventBus:
    """Fans events out to sinks; also satisfies the EventSink contract."""

    def __init__(self, sinks: list[Sink] | None = None, run_id: str | None = None) -> None:
        self._sinks: list[Sink] = list(sinks or [])
        self._lock = threading.Lock()
        self.run_id = run_id

    def add_sink(self, sink: Sink) -> None:
        with self._lock:
            self._sinks.append(sink)

    def emit(self, event: object) -> None:
        if not isinstance(event, Event):  # tolerate foreign objects from hooks
            from mbt.events.models import LogMessage

            event = LogMessage(message=str(event))
        if event.run_id is None and self.run_id is not None:
            event.run_id = self.run_id
        with self._lock:
            sinks = list(self._sinks)
        for sink in sinks:
            sink.write(event)


_bus = EventBus()


def get_bus() -> EventBus:
    return _bus


def set_bus(bus: EventBus) -> None:
    global _bus
    _bus = bus
