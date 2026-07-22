"""Optional OpenTelemetry span bridge (R2-14, NFR-06).

Opt-in via ``MBT_OTEL``: when set, the coordinator's lifecycle START/FINISH
event pairs become OpenTelemetry spans, so a scheduled ``mbt run`` shows up as
one trace - a root span per command with a child span per node - in whatever
collector the operator has wired through the standard ``OTEL_*`` environment.

mbt only *emits* spans against the globally-configured tracer; standing up a
provider and an OTLP exporter is the operator's job (``opentelemetry-instrument``
or their own bootstrap), so mbt stays unopinionated about where traces go and
adds zero overhead when no provider is installed (the API returns a no-op
tracer). ``import opentelemetry`` is lazy (ADR-14) and lives behind the ``otel``
extra; bare ``mbt-core`` without it raises a loud, actionable error at setup
rather than dropping telemetry silently (the same stance as ``MBT_LOG_FILE``).
"""

import threading
from typing import TYPE_CHECKING, Any

from mbt.events.models import (
    Event,
    NodeFinished,
    NodeStarted,
    RunFinished,
    RunStarted,
)

if TYPE_CHECKING:
    from opentelemetry.trace import Span, Tracer


class OTelSpanSink:
    """Translates coordinator lifecycle events into OpenTelemetry spans.

    ``RunStarted`` opens the root span and captures its context; each
    ``NodeStarted`` opens a child span parented to that context (passed
    explicitly, so the parent link survives the pool-thread hop that emits node
    events); the matching ``*Finished`` event ends the span with a status and
    timing attributes. Any other event is ignored - the console/JSON sinks
    already carry the full stream. The span bookkeeping is lock-guarded because
    the bus fans out to sinks without holding its own lock, so ``write`` can run
    on several node-execution threads at once.
    """

    def __init__(self, tracer: "Tracer") -> None:
        self._tracer = tracer
        self._lock = threading.Lock()
        self._run_span: Span | None = None
        self._run_context: Any = None
        self._node_spans: dict[str, Span] = {}

    def write(self, event: Event) -> None:
        if isinstance(event, RunStarted):
            self._start_run(event)
        elif isinstance(event, NodeStarted):
            self._start_node(event)
        elif isinstance(event, NodeFinished):
            self._finish_node(event)
        elif isinstance(event, RunFinished):
            self._finish_run(event)

    def _start_run(self, event: RunStarted) -> None:
        from opentelemetry import trace

        span = self._tracer.start_span(f"mbt.{event.command or 'run'}")
        span.set_attribute("mbt.command", event.command)
        span.set_attribute("mbt.target", event.target)
        span.set_attribute("mbt.selected", event.selected)
        with self._lock:
            self._run_span = span
            self._run_context = trace.set_span_in_context(span)

    def _start_node(self, event: NodeStarted) -> None:
        with self._lock:
            context = self._run_context
        span = self._tracer.start_span(f"mbt.node.{event.resource_type or 'node'}", context=context)
        span.set_attribute("mbt.node.id", event.unique_id or "")
        span.set_attribute("mbt.node.resource_type", event.resource_type)
        span.set_attribute("mbt.node.index", event.index)
        with self._lock:
            self._node_spans[event.unique_id or ""] = span

    def _finish_node(self, event: NodeFinished) -> None:
        from opentelemetry.trace import Status, StatusCode

        with self._lock:
            span = self._node_spans.pop(event.unique_id or "", None)
        if span is None:  # a Finished with no matching Started (out-of-order)
            return
        span.set_attribute("mbt.node.status", event.status)
        span.set_attribute("mbt.node.execution_time_s", event.execution_time_s)
        if event.status != "success":
            span.set_status(Status(StatusCode.ERROR, event.message or event.status))
        span.end()

    def _finish_run(self, event: RunFinished) -> None:
        from opentelemetry.trace import Status, StatusCode

        with self._lock:
            span, self._run_span, self._run_context = self._run_span, None, None
        if span is None:  # RunFinished with no RunStarted (defensive)
            return
        span.set_attribute("mbt.status", event.status)
        span.set_attribute("mbt.succeeded", event.succeeded)
        span.set_attribute("mbt.failed", event.failed)
        span.set_attribute("mbt.skipped", event.skipped)
        if event.status != "success":
            span.set_status(Status(StatusCode.ERROR, event.status))
        span.end()


def make_otel_sink() -> OTelSpanSink:
    """Build a span sink against the process's globally-configured tracer.

    Raises ``ImportError`` when the ``otel`` extra is not installed; the caller
    turns that into an actionable ``ConfigError`` (bare ``mbt-core`` has no
    opentelemetry).
    """
    from opentelemetry import trace

    return OTelSpanSink(trace.get_tracer("mbt"))
