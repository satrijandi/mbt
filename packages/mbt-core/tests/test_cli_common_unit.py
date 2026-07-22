"""Unit tests for the shared CLI plumbing (mbt/cli/common.py)."""

import json
from datetime import UTC, datetime
from pathlib import Path
from types import SimpleNamespace

import pytest
from cli_unit_helpers import (  # noqa: F401 - autouse fixture import
    cli_process_state,
    install_recording_bus,
    make_results,
)

from mbt.artifacts.run_results import NodeResult, RegistrationResult
from mbt.cli.common import (
    CLIContext,
    err_console,
    fail,
    out_console,
    parse_anchor,
    parse_vars,
    print_warnings,
    render_results_table,
    setup_bus,
)
from mbt.events import get_bus
from mbt.events.otel import OTelSpanSink
from mbt.events.sinks import ConsoleSink, JsonLinesSink, NullSink
from mbt.exceptions import ConfigError

# -- CLIContext ---------------------------------------------------------------------


def test_resolve_cli_path_semantics(tmp_path: Path) -> None:
    ctx = CLIContext(invocation_cwd=tmp_path)
    assert ctx.resolve_cli_path(None) is None
    assert ctx.resolve_cli_path("s3://bucket/key") == "s3://bucket/key"  # URIs untouched
    # relative paths absolutize against where the user ran mbt, not the project
    assert ctx.resolve_cli_path("sub/manifest.json") == str(tmp_path / "sub" / "manifest.json")


def test_invocation_carries_context_flags(tmp_path: Path) -> None:
    ctx = CLIContext(project_dir=tmp_path, target="prod", cli_vars={"a": 1})
    opts = ctx.invocation("run", select=["churn_model"], fail_fast=True)
    assert opts.command == "run"
    assert opts.project_dir == tmp_path
    assert opts.target == "prod"
    assert opts.cli_vars == {"a": 1}
    assert opts.select == ["churn_model"]
    assert opts.fail_fast is True


def test_profiles_loads_selected_target(demo_project: Path) -> None:
    from mbt.parsing import parse_project

    parsed = parse_project(demo_project)
    profiles = CLIContext(project_dir=demo_project, target="prod").profiles(parsed)
    assert profiles.target_name == "prod"
    assert profiles.target.threads == 4


# -- parse_vars / parse_anchor ------------------------------------------------------


def test_parse_vars_accepts_yaml_and_json() -> None:
    assert parse_vars(None) == {}
    assert parse_vars("") == {}
    assert parse_vars("sample_fraction: 0.1") == {"sample_fraction": 0.1}
    assert parse_vars('{"a": 1}') == {"a": 1}


def test_parse_vars_rejects_invalid_yaml() -> None:
    with pytest.raises(ConfigError, match="not valid YAML/JSON"):
        parse_vars("{unclosed")


def test_parse_vars_rejects_non_mappings() -> None:
    with pytest.raises(ConfigError, match="must be a mapping"):
        parse_vars("[1, 2]")


def test_parse_anchor_handles_z_naive_and_none() -> None:
    assert parse_anchor(None) is None
    aware = parse_anchor("2026-07-06T12:00:00Z")
    assert aware == datetime(2026, 7, 6, 12, tzinfo=UTC)
    naive = parse_anchor("2026-07-06T12:00:00")
    assert naive is not None and naive.tzinfo is UTC  # naive input pinned to UTC


def test_parse_anchor_rejects_garbage() -> None:
    with pytest.raises(ConfigError, match="not an ISO timestamp"):
        parse_anchor("last tuesday")


# -- setup_bus ----------------------------------------------------------------------


def test_setup_bus_selects_sink_per_flags() -> None:
    setup_bus(CLIContext(quiet=True))
    assert isinstance(get_bus()._sinks[0], NullSink)

    setup_bus(CLIContext(log_format="json"))
    assert isinstance(get_bus()._sinks[0], JsonLinesSink)

    setup_bus(CLIContext())
    assert isinstance(get_bus()._sinks[0], ConsoleSink)


def test_setup_bus_appends_durable_json_log_when_env_set(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    # A nested path proves parent dirs are created; relative to invocation cwd.
    monkeypatch.setenv("MBT_LOG_FILE", "logs/events.jsonl")
    setup_bus(CLIContext(invocation_cwd=tmp_path))

    bus = get_bus()
    # Additive: the human console sink is kept, the JSON file sink added on top.
    assert isinstance(bus._sinks[0], ConsoleSink)
    assert isinstance(bus._sinks[1], JsonLinesSink)

    bus.emit("hello timeline")
    lines = (tmp_path / "logs" / "events.jsonl").read_text().splitlines()
    assert len(lines) == 1
    assert json.loads(lines[0])["message"] == "hello timeline"


def test_setup_bus_log_file_captures_even_when_console_is_quiet(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    # The scheduled-job shape: silent console, full durable timeline in the file.
    log = tmp_path / "events.jsonl"
    monkeypatch.setenv("MBT_LOG_FILE", str(log))
    setup_bus(CLIContext(quiet=True, invocation_cwd=tmp_path))

    bus = get_bus()
    assert isinstance(bus._sinks[0], NullSink)  # console suppressed...
    assert isinstance(bus._sinks[1], JsonLinesSink)  # ...file still captures
    bus.emit("still logged")
    assert "still logged" in log.read_text()


def test_setup_bus_log_file_appends_across_runs(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    # Append, not truncate: a nightly job accumulates its timeline (run_id
    # demultiplexes), rather than each run erasing the last.
    log = tmp_path / "events.jsonl"
    monkeypatch.setenv("MBT_LOG_FILE", str(log))
    for msg in ("run one", "run two"):
        setup_bus(CLIContext(invocation_cwd=tmp_path))
        get_bus().emit(msg)
    body = log.read_text()
    assert "run one" in body and "run two" in body


def test_setup_bus_log_file_bad_path_fails_loudly(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    # A silently dropped durable log is worse than an actionable error: pointing
    # at a directory (unopenable as a file) must raise, not swallow.
    monkeypatch.setenv("MBT_LOG_FILE", str(tmp_path))
    with pytest.raises(ConfigError, match="MBT_LOG_FILE"):
        setup_bus(CLIContext(invocation_cwd=tmp_path))


def test_setup_bus_appends_otel_span_sink_when_enabled(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    # Additive, like MBT_LOG_FILE: the console sink stays and the span bridge is
    # added on top so `mbt run` emits a trace alongside the human output.
    monkeypatch.setenv("MBT_OTEL", "1")
    setup_bus(CLIContext())

    sinks = get_bus()._sinks
    assert isinstance(sinks[0], ConsoleSink)
    assert isinstance(sinks[1], OTelSpanSink)


def test_setup_bus_otel_without_extra_fails_loudly(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    # Explicitly asking for telemetry and then dropping it silently is worse than
    # an actionable error: a missing `otel` extra must raise, not swallow.
    def _no_opentelemetry() -> OTelSpanSink:
        raise ImportError("No module named 'opentelemetry'")

    monkeypatch.setenv("MBT_OTEL", "1")
    monkeypatch.setattr("mbt.events.otel.make_otel_sink", _no_opentelemetry)
    with pytest.raises(ConfigError, match="opentelemetry is not installed"):
        setup_bus(CLIContext())


# -- fail ---------------------------------------------------------------------------


def test_fail_prints_all_error_context() -> None:
    exc = ConfigError(
        "bad config",
        resource="model.demo.churn_model",
        path="models/churn_model.yml",
        hint="fix the spec",
    )
    with err_console.capture() as capture:
        exit_exc = fail(exc)
    out = capture.get()
    assert "Error: bad config" in out
    assert "resource: model.demo.churn_model" in out
    assert "file: models/churn_model.yml" in out
    assert "hint:" in out and "fix the spec" in out
    assert exit_exc.exit_code == 1


# -- print_warnings -----------------------------------------------------------------


def test_print_warnings_emits_warn_events() -> None:
    sink = install_recording_bus()
    issue = SimpleNamespace(format=lambda: "dataset churn_training has no checks")
    parsed = SimpleNamespace(report=SimpleNamespace(warnings=[issue]))
    print_warnings(parsed)  # type: ignore[arg-type]
    assert [event.level for event in sink.events] == ["warn"]
    assert "no checks" in sink.events[0].message  # type: ignore[attr-defined]


# -- render_results_table -----------------------------------------------------------


def test_render_results_table_suppressed_when_quiet_or_json_or_empty() -> None:
    results = make_results("run", NodeResult(unique_id="model.demo.m", status="success"))
    with out_console.capture() as capture:
        render_results_table(results, SimpleNamespace(quiet=True, log_format="text"))  # type: ignore[arg-type]
        render_results_table(results, SimpleNamespace(quiet=False, log_format="json"))  # type: ignore[arg-type]
        render_results_table(make_results("run"), SimpleNamespace(quiet=False, log_format="text"))  # type: ignore[arg-type]
    assert capture.get() == ""


def test_render_results_table_shows_top_metrics_and_registration() -> None:
    result = NodeResult(
        unique_id="model.demo.m",
        status="success",
        execution_time_s=1.234,
        metrics={"pr_auc": 0.6123, "roc_auc": 0.75, "logloss": 0.5, "rows": 400.0},
        registration=RegistrationResult(registry="fake", name="m", version="3", stage="staging"),
    )
    with out_console.capture() as capture:
        render_results_table(
            make_results("build", result),
            SimpleNamespace(quiet=False, log_format="text"),  # type: ignore[arg-type]
        )
    out = capture.get()
    # top-3 metric keys, sorted; counts render as integers elsewhere (see
    # test_cli_rendering) so only fractional formatting is asserted here
    assert "logloss=0.5000" in out
    assert "pr_auc=0.6123" in out
    assert "roc_auc=0.7500" in out
    assert "rows" not in out  # only the first three metrics are shown
    assert "v3" in out  # registration detail
