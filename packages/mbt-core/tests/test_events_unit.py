"""Event models, bus fan-out, and sinks (TSD §16)."""

import io
import json

from misc_unit_helpers import RecordingSink
from rich.console import Console

from mbt.events.bus import EventBus, get_bus, set_bus
from mbt.events.models import (
    AdapterWarning,
    ArtifactRegistered,
    AutoResolved,
    CheckEvaluated,
    CompileCompleted,
    CompileStarted,
    Event,
    GateEvaluated,
    JobLine,
    LogMessage,
    NodeFinished,
    NodeStarted,
    ParseCompleted,
    ParseStarted,
    PromotionApplied,
    RunFinished,
    RunStarted,
    StageTransitioned,
    StateDiffed,
)
from mbt.events.sinks import ConsoleSink, JsonLinesSink, NullSink
from mbt.secrets import taint

# -- models ------------------------------------------------------------------


def test_base_event_renders_its_name_and_stamps_it() -> None:
    event = Event()
    assert event.event == "Event"
    assert event.human() == "Event"


def test_log_message_renders_message() -> None:
    assert LogMessage(message="hello").human() == "hello"


def test_parse_events_render() -> None:
    assert ParseStarted(project="demo").human() == "Parsing project 'demo'"
    ok = ParseCompleted(resources=5, errors=0, elapsed_s=1.234)
    assert ok.human() == "Parsed 5 resources in 1.23s [OK]"
    bad = ParseCompleted(resources=5, errors=2, elapsed_s=0.5)
    assert bad.human() == "Parsed 5 resources in 0.50s [2 error(s)]"


def test_compile_events_render() -> None:
    assert CompileStarted(target="dev").human() == "Compiling against target 'dev'"
    with_path = CompileCompleted(
        nodes=3, anchor="2026-07-01T00:00:00Z", manifest_path="target/manifest.json", elapsed_s=0.5
    )
    assert with_path.human() == (
        "Compiled 3 nodes in 0.50s (anchor 2026-07-01T00:00:00Z) -> target/manifest.json"
    )
    without_path = CompileCompleted(nodes=3, anchor="A", elapsed_s=0.5)
    assert without_path.human() == "Compiled 3 nodes in 0.50s (anchor A)"


def test_run_and_node_events_render() -> None:
    started = RunStarted(command="run", target="dev", selected=2)
    assert started.human() == "run: 2 node(s) selected on target 'dev'"
    node_started = NodeStarted(
        resource_type="model", index=1, total=2, unique_id="model.demo.churn"
    )
    assert node_started.human() == "[1/2] START model model.demo.churn"
    finished = NodeFinished(
        resource_type="model",
        status="success",
        execution_time_s=1.5,
        index=1,
        total=2,
        unique_id="model.demo.churn",
        message="registered v3",
    )
    assert finished.human() == ("[1/2] SUCCESS model model.demo.churn in 1.50s - registered v3")
    silent = NodeFinished(
        resource_type="dataset", status="error", index=2, total=2, unique_id="dataset.demo.d"
    )
    assert silent.human() == "[2/2] ERROR dataset dataset.demo.d in 0.00s"


def test_quality_events_render() -> None:
    # imported here: a module-level "Test*" name trips pytest class collection
    from mbt.events.models import TestEvaluated as TestEvaluatedEvent

    assert CheckEvaluated(check="class_balance").human() == "check class_balance: PASS"
    assert (
        CheckEvaluated(check="not_null", passed=False, message="3 nulls").human()
        == "check not_null: FAIL - 3 nulls"
    )
    assert (
        TestEvaluatedEvent(test="no_leakage", passed=False, message="boom").human()
        == "test no_leakage: FAIL - boom"
    )
    assert TestEvaluatedEvent(test="no_leakage").human() == "test no_leakage: PASS"
    gate = GateEvaluated(metric="pr_auc", kind="threshold", passed=False, expected=0.4, actual=0.3)
    assert gate.human() == "gate pr_auc (threshold): FAIL - expected 0.4, got 0.3"
    champion = GateEvaluated(metric="pr_auc", kind="champion", passed=True, message="beats v2")
    assert champion.human() == "gate pr_auc (champion): PASS - beats v2"


def test_registry_and_promotion_events_render() -> None:
    assert AutoResolved(param="scale_pos_weight", value="3.5").human() == (
        "auto-resolved scale_pos_weight = 3.5"
    )
    registered = ArtifactRegistered(registry="fake", name="churn", version="3", stage="staging")
    assert registered.human() == "registered churn v3 -> staging (fake)"
    transitioned = StageTransitioned(name="churn", version="3", stage="production")
    assert transitioned.human() == "transitioned churn v3 -> production"
    promoted = PromotionApplied(name="churn", version="3", to_stage="production", forced=True)
    assert promoted.human() == "promoted churn v3 -> production (FORCED)"
    unforced = PromotionApplied(name="churn", version="4", to_stage="staging")
    assert unforced.human() == "promoted churn v4 -> staging"


def test_adapter_warning_defaults_to_warn_level() -> None:
    warning = AdapterWarning(adapter="xgboost", message="deprecated param")
    assert warning.level == "warn"
    assert warning.human() == "[xgboost] deprecated param"


def test_state_and_run_summary_events_render() -> None:
    diffed = StateDiffed(added=1, removed=2, modified=3, env_changed=True)
    assert diffed.human() == "state diff: 1 added, 2 removed, 3 modified; env digest CHANGED"
    clean = StateDiffed()
    assert clean.human() == "state diff: 0 added, 0 removed, 0 modified"
    finished = RunFinished(
        command="build", status="success", succeeded=1, failed=0, skipped=2, elapsed_s=3.5
    )
    assert finished.human() == "build finished [success]: 1 ok, 0 failed, 2 skipped in 3.5s"


def test_job_line_is_debug_level_raw_text() -> None:
    line = JobLine(line="epoch 1: loss 0.4")
    assert line.level == "debug"
    assert line.human() == "epoch 1: loss 0.4"


# -- bus ----------------------------------------------------------------------


def test_bus_fans_out_and_stamps_run_id() -> None:
    sink = RecordingSink()
    bus = EventBus([sink], run_id="run-1")
    bus.emit(LogMessage(message="hi"))
    assert len(sink.events) == 1
    assert sink.events[0].run_id == "run-1"


def test_bus_preserves_existing_run_id() -> None:
    sink = RecordingSink()
    bus = EventBus([sink], run_id="run-1")
    bus.emit(LogMessage(message="hi", run_id="job-7"))
    assert sink.events[0].run_id == "job-7"


def test_bus_wraps_foreign_objects_in_log_messages() -> None:
    sink = RecordingSink()
    bus = EventBus([sink])
    bus.emit("plain string from a hook")
    assert isinstance(sink.events[0], LogMessage)
    assert sink.events[0].message == "plain string from a hook"


def test_add_sink_receives_subsequent_events() -> None:
    bus = EventBus()
    sink = RecordingSink()
    bus.add_sink(sink)
    bus.emit(LogMessage(message="after"))
    assert [e.message for e in sink.events] == ["after"]


def test_set_bus_swaps_the_process_bus() -> None:
    original = get_bus()
    replacement = EventBus()
    try:
        set_bus(replacement)
        assert get_bus() is replacement
    finally:
        set_bus(original)
    assert get_bus() is original


# -- sinks --------------------------------------------------------------------


def _console_sink(verbose: bool = False) -> tuple[ConsoleSink, io.StringIO]:
    buffer = io.StringIO()
    console = Console(file=buffer, force_terminal=False, highlight=False, width=200)
    return ConsoleSink(console=console, verbose=verbose), buffer


def test_null_sink_swallows_events() -> None:
    NullSink().write(LogMessage(message="dropped"))


def test_console_sink_skips_debug_unless_verbose() -> None:
    sink, buffer = _console_sink()
    sink.write(JobLine(line="hidden debug"))
    assert buffer.getvalue() == ""
    verbose_sink, verbose_buffer = _console_sink(verbose=True)
    verbose_sink.write(JobLine(line="shown debug"))
    assert "shown debug" in verbose_buffer.getvalue()


def test_console_sink_prefixes_warn_and_error() -> None:
    sink, buffer = _console_sink()
    sink.write(LogMessage(level="warn", message="careful"))
    sink.write(LogMessage(level="error", message="broken"))
    sink.write(LogMessage(message="plain"))
    output = buffer.getvalue()
    assert "WARN careful" in output
    assert "ERROR broken" in output
    assert "plain" in output


def test_console_sink_redacts_tainted_values() -> None:
    secret = taint("s3kr3t-console")
    sink, buffer = _console_sink()
    sink.write(LogMessage(message=f"uri is {secret}"))
    output = buffer.getvalue()
    assert "s3kr3t-console" not in output
    assert "***" in output


def test_console_sink_defaults_to_stderr() -> None:
    """ "Events go to stderr, stdout is command data" is a load-bearing
    invariant, and it should hold by construction rather than only because
    every caller remembers to pass an stderr console (FEEDBACK v3 E-2)."""
    assert ConsoleSink().console.stderr is True


def test_console_sink_hangs_wrapped_lines_under_the_message() -> None:
    """A wrapped continuation used to restart in column zero, so it read as a
    new event with a missing timestamp (FEEDBACK v3 E-5)."""
    buffer = io.StringIO()
    console = Console(file=buffer, force_terminal=False, highlight=False, width=40)
    sink = ConsoleSink(console=console)
    sink.write(LogMessage(message="alpha beta gamma delta epsilon zeta eta theta"))
    first, *rest = [line for line in buffer.getvalue().splitlines() if line.strip()]
    assert first.startswith(" ") is False
    assert rest, "expected the message to wrap at width 40"
    assert all(line.startswith(" " * 10) for line in rest), rest


def test_console_sink_keeps_long_paths_whole_when_they_fit() -> None:
    """The one thing a user most often copies out of a log is an absolute
    path; Rich's default wrapping split them mid-token."""
    buffer = io.StringIO()
    console = Console(file=buffer, force_terminal=False, highlight=False, width=80)
    path = "/very/long/directory/name/target/manifest.json"
    ConsoleSink(console=console).write(LogMessage(message=f"compiled 3 nodes -> {path}"))
    assert path in buffer.getvalue().replace("\n", "").replace(" " * 10, "")


def test_a_path_longer_than_the_line_is_hard_split_rather_than_dropped() -> None:
    """The fallback when even a whole line cannot hold the token.

    Preferring word boundaries is the point, but a 200-character path on an
    80-column terminal has no boundary to prefer - it has to break somewhere,
    and every character must survive the break.
    """
    buffer = io.StringIO()
    console = Console(file=buffer, force_terminal=False, highlight=False, width=50)
    path = "/" + "seg/" * 40 + "manifest.json"
    ConsoleSink(console=console).write(LogMessage(message=f"wrote {path}"))
    rendered = buffer.getvalue()
    assert path in rendered.replace("\n", "").replace(" ", "")


def test_console_sink_does_not_fight_a_very_narrow_terminal() -> None:
    """Below a usable width, wrapping shreds the message into fragments; emit
    it on one line and let the terminal do whatever it does."""
    buffer = io.StringIO()
    console = Console(file=buffer, force_terminal=False, highlight=False, width=12)
    ConsoleSink(console=console).write(LogMessage(message="alpha beta gamma"))
    assert "alpha beta gamma" in buffer.getvalue().replace("\n", "")


def test_console_sink_renders_local_time_not_bare_utc() -> None:
    """event.ts is UTC-aware; strftime would print the UTC wall clock with no
    marker, next to third-party lines in local time (FEEDBACK v3 E-3)."""
    import datetime as dt

    sink, buffer = _console_sink()
    event = LogMessage(message="x", ts=dt.datetime(2026, 1, 1, 12, 0, 0, tzinfo=dt.UTC))
    sink.write(event)
    expected = event.ts.astimezone().strftime("%H:%M:%S")
    assert expected in buffer.getvalue()


def test_json_lines_sink_writes_one_redacted_object_per_line() -> None:
    default_sink = JsonLinesSink()  # defaults to stdout; never written to here
    assert default_sink.stream is not None
    secret = taint("s3kr3t-json")
    buffer = io.StringIO()
    sink = JsonLinesSink(stream=buffer)
    sink.write(LogMessage(message=f"uri is {secret}"))
    lines = buffer.getvalue().splitlines()
    assert len(lines) == 1
    payload = json.loads(lines[0])
    assert payload["event"] == "LogMessage"
    assert payload["message"] == "uri is ***"


def test_console_sink_treats_embedded_newlines_as_hard_breaks() -> None:
    """No built-in event renders a newline, but LogMessage carries whatever a
    hook or adapter error hands it; a "a\\nb" word would otherwise wreck the
    width accounting for the whole line."""
    buffer = io.StringIO()
    console = Console(file=buffer, force_terminal=False, highlight=False, width=60)
    ConsoleSink(console=console).write(LogMessage(message="first line\nsecond line"))
    body = [line for line in buffer.getvalue().splitlines() if line.strip()]
    assert body[0].endswith("first line")
    assert body[1] == " " * 10 + "second line"


def test_narrow_terminal_still_splits_on_embedded_newlines() -> None:
    """Below the wrap floor mbt stops laying out and lets the terminal cope -
    but a hard newline is content, not layout, so it still separates lines.

    The assertion is deliberately about content rather than columns: at width
    12 the 10-column gutter alone overflows, so Rich re-wraps the indented
    continuation. That is the "do not fight it" case working as intended.
    """
    buffer = io.StringIO()
    console = Console(file=buffer, force_terminal=False, highlight=False, width=12)
    ConsoleSink(console=console).write(LogMessage(message="alpha\nbeta"))
    rendered = buffer.getvalue()
    assert "alpha" in rendered
    assert "beta" in rendered.replace("\n", "").replace(" ", "")
    # two logical lines, not one run-on
    assert rendered.count("\n") >= 2
