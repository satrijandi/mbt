"""OpenTelemetry span bridge (mbt/events/otel.py, R2-14/NFR-06).

The sink turns coordinator lifecycle events into a trace. Tests drive it
against an in-memory exporter (no provider is set globally, so they stay
isolated) and assert the run/node span tree, attributes, and error status.
"""

from opentelemetry.sdk.trace import TracerProvider
from opentelemetry.sdk.trace.export import SimpleSpanProcessor
from opentelemetry.sdk.trace.export.in_memory_span_exporter import InMemorySpanExporter
from opentelemetry.trace import StatusCode

from mbt.events.models import (
    LogMessage,
    NodeFinished,
    NodeStarted,
    RunFinished,
    RunStarted,
)
from mbt.events.otel import OTelSpanSink


def _sink() -> tuple[OTelSpanSink, InMemorySpanExporter]:
    exporter = InMemorySpanExporter()
    provider = TracerProvider()
    provider.add_span_processor(SimpleSpanProcessor(exporter))
    return OTelSpanSink(provider.get_tracer("mbt-test")), exporter


def test_run_and_nodes_become_a_parented_span_tree() -> None:
    sink, exporter = _sink()
    sink.write(RunStarted(command="run", target="prod", selected=1))
    sink.write(NodeStarted(unique_id="model.p.m", resource_type="model", index=1, total=1))
    # A non-lifecycle event is ignored - the console/JSON sinks carry it instead.
    sink.write(LogMessage(message="mid-run chatter"))
    sink.write(
        NodeFinished(
            unique_id="model.p.m", resource_type="model", status="success", execution_time_s=1.5
        )
    )
    sink.write(RunFinished(command="run", status="success", succeeded=1))

    by_name = {span.name: span for span in exporter.get_finished_spans()}
    assert set(by_name) == {"mbt.run", "mbt.node.model"}

    root, node = by_name["mbt.run"], by_name["mbt.node.model"]
    assert node.parent is not None and node.parent.span_id == root.context.span_id
    assert node.context.trace_id == root.context.trace_id  # one trace

    assert root.attributes["mbt.command"] == "run"
    assert root.attributes["mbt.target"] == "prod"
    assert root.attributes["mbt.status"] == "success"
    assert root.attributes["mbt.succeeded"] == 1
    assert node.attributes["mbt.node.id"] == "model.p.m"
    assert node.attributes["mbt.node.status"] == "success"
    assert node.attributes["mbt.node.execution_time_s"] == 1.5
    # A clean run leaves both spans unmarked (no error status).
    assert root.status.status_code is StatusCode.UNSET
    assert node.status.status_code is StatusCode.UNSET


def test_failed_node_and_run_carry_error_status() -> None:
    sink, exporter = _sink()
    sink.write(RunStarted(command="run", target="dev", selected=1))
    sink.write(NodeStarted(unique_id="model.p.m", resource_type="model", index=1, total=1))
    sink.write(
        NodeFinished(
            unique_id="model.p.m",
            resource_type="model",
            status="error",
            message="training blew up",
            execution_time_s=0.2,
        )
    )
    sink.write(RunFinished(command="run", status="quality_failure", succeeded=0, failed=1))

    by_name = {span.name: span for span in exporter.get_finished_spans()}
    node, root = by_name["mbt.node.model"], by_name["mbt.run"]
    assert node.status.status_code is StatusCode.ERROR
    assert node.status.description == "training blew up"
    assert root.status.status_code is StatusCode.ERROR
    assert root.attributes["mbt.failed"] == 1


def test_orphan_finish_events_are_ignored() -> None:
    # Out-of-order or duplicate Finished events (no matching Started) must not
    # raise or emit stray spans - the sink pops defensively.
    sink, exporter = _sink()
    sink.write(NodeFinished(unique_id="ghost", resource_type="model", status="success"))
    sink.write(RunFinished(command="run", status="success"))
    assert exporter.get_finished_spans() == ()
