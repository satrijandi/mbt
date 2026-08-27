"""In-process unit tests for the mbt Typer app (mbt/cli/main.py).

Commands run through typer.testing.CliRunner so the coordinator-side lines
are traced; job-facing orchestration is exercised for real on the fake
adapters where cheap and monkeypatched where only the CLI wiring matters.
"""

import importlib
import json
import os
import runpy
import sys
import time
from pathlib import Path

import click
import pytest
from cli_unit_helpers import (  # noqa: F401 - autouse fixture import
    ANCHOR,
    cli_process_state,
    debug,
    invoke,
    make_results,
)
from core_helpers import write

from mbt.artifacts.run_results import NodeResult

# -- version / context plumbing ----------------------------------------------------


def test_version_flag_prints_and_exits_zero() -> None:
    result = invoke(["--version"])
    assert result.exit_code == 0, debug(result)
    assert result.output.startswith("mbt ")


def test_missing_project_dir_is_hard_error(tmp_path: Path) -> None:
    result = invoke(["parse", "--project-dir", str(tmp_path / "nope")])
    assert result.exit_code == 1, debug(result)
    assert "is not a directory" in result.stderr


def test_import_falls_back_to_real_click_exceptions(monkeypatch: pytest.MonkeyPatch) -> None:
    """Older typer has no vendored click; the module must fall back to the
    real click exception types (the dual except tuples in main)."""
    from mbt.cli import main as cli_main

    original = cli_main.typer_click_exc
    monkeypatch.setitem(sys.modules, "typer._click", None)
    try:
        reloaded = importlib.reload(cli_main)
        assert reloaded.typer_click_exc is click.exceptions
    finally:
        monkeypatch.undo()
        importlib.reload(cli_main)
    # Restored to whatever THIS typer provides: the vendored module on >= 0.20,
    # the real click below it. Asserting "vendored" outright would encode the
    # newer typer into a test whose whole subject is that main() supports both,
    # and would red the floors job against the declared typer>=0.16.
    assert cli_main.typer_click_exc is original


# -- init ---------------------------------------------------------------------------


def test_init_scaffolds_project(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    home = tmp_path / "home"
    home.mkdir()
    monkeypatch.setenv("HOME", str(home))  # never touch the real ~/.mbt
    result = invoke(["init", "myproj", "--project-dir", str(tmp_path)])
    assert result.exit_code == 0, debug(result)
    assert (tmp_path / "myproj" / "mbt_project.yml").is_file()
    assert (home / ".mbt" / "profiles.yml").is_file()
    assert "Next steps" in result.output


def test_init_rejects_invalid_name(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setenv("HOME", str(tmp_path / "home"))
    result = invoke(["init", "Bad-Name", "--project-dir", str(tmp_path)])
    assert result.exit_code == 1, debug(result)
    assert "invalid project name" in result.stderr


# -- deps ---------------------------------------------------------------------------


def test_deps_dry_run_lists_packages(tmp_path: Path) -> None:
    write(tmp_path / "packages.yml", "packages: [{package: mbt-core, version: '>=0.0'}]")
    result = invoke(["deps", "--project-dir", str(tmp_path), "--dry-run"])
    assert result.exit_code == 0, debug(result)
    assert "would install: mbt-core>=0.0" in result.output


def test_deps_installs_nothing_for_empty_pins(tmp_path: Path) -> None:
    write(tmp_path / "packages.yml", "packages: []")
    result = invoke(["deps", "--project-dir", str(tmp_path)])
    assert result.exit_code == 0, debug(result)
    assert "installed: (nothing)" in result.output


# -- clean --------------------------------------------------------------------------


def test_clean_removes_target_directory(tmp_path: Path) -> None:
    (tmp_path / "target").mkdir()
    (tmp_path / "target" / "run_results.json").write_text("{}")
    result = invoke(["clean", "--project-dir", str(tmp_path)])
    assert result.exit_code == 0, debug(result)
    assert "removed" in result.output
    assert not (tmp_path / "target").exists()

    again = invoke(["clean", "--project-dir", str(tmp_path)])
    assert again.exit_code == 0, debug(again)
    assert "nothing to clean" in again.output


def test_clean_ages_out_stale_job_payloads(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    from mbt.adapters.local import compute

    # the sweep found stale payload dirs: clean reports how many it aged out
    monkeypatch.setattr(
        compute, "sweep_stale_job_payloads", lambda cutoff: [Path("/tmp/mbt-job-old")]
    )
    out = invoke(["clean", "--project-dir", str(tmp_path)])
    assert out.exit_code == 0, debug(out)
    assert "aged out 1 stale job payload dir" in out.output

    # nothing stale: no age-out line at all
    monkeypatch.setattr(compute, "sweep_stale_job_payloads", lambda cutoff: [])
    quiet = invoke(["clean", "--project-dir", str(tmp_path)])
    assert quiet.exit_code == 0, debug(quiet)
    assert "aged out" not in quiet.output


def test_clean_artifact_gc_rejects_non_duration(tmp_path: Path) -> None:
    absolute_window = "2026-01-01T00:00:00:2026-02-01T00:00:00"
    result = invoke(
        ["clean", "--project-dir", str(tmp_path), "--artifacts-older-than", absolute_window]
    )
    assert result.exit_code == 1, debug(result)
    assert "expects a duration" in result.stderr


def test_clean_artifact_gc_prunes_old_run_prefixes(demo_project: Path) -> None:
    old_run = demo_project / "target" / "artifacts" / "20200101-000000-deadbeef"
    old_run.mkdir(parents=True)
    artifact = old_run / "model.bin"
    artifact.write_bytes(b"x" * 10)
    stale = time.time() - 90 * 86400
    os.utime(artifact, (stale, stale))
    base = ["clean", "--project-dir", str(demo_project), "--artifacts-older-than", "30d"]

    dry = invoke([*base, "--dry-run"])
    assert dry.exit_code == 0, debug(dry)
    assert "would delete" in dry.output
    assert old_run.is_dir()  # dry run deletes nothing

    real = invoke(base)
    assert real.exit_code == 0, debug(real)
    assert not old_run.exists()
    assert "deleted 1 run prefix(es)" in real.output


# -- parse / compile ----------------------------------------------------------------


def test_parse_reports_counts(demo_project: Path) -> None:
    result = invoke(["parse", "--project-dir", str(demo_project)])
    assert result.exit_code == 0, debug(result)
    assert "Parsed" in result.output


def test_parse_emits_parse_started_and_completed(demo_project: Path) -> None:
    """parse brackets its work with ParseStarted/ParseCompleted on the bus
    (rendered to stderr), symmetric with compile's Compile* events - the
    command was previously silent on the event stream."""
    result = invoke(["parse", "--project-dir", str(demo_project)])
    assert result.exit_code == 0, debug(result)
    assert "Parsing project" in result.stderr
    assert "resources in" in result.stderr


def test_profiles_dir_override_is_honored(demo_project: Path) -> None:
    # profiles.yml only exists in --profiles-dir; compile succeeds only if the
    # override is resolved and used
    profiles_dir = demo_project / "profiles_elsewhere"
    profiles_dir.mkdir()
    (demo_project / "profiles.yml").rename(profiles_dir / "profiles.yml")
    result = invoke(
        [
            "compile",
            "--project-dir",
            str(demo_project),
            "--profiles-dir",
            str(profiles_dir),
            "--anchor",
            ANCHOR,
        ]
    )
    assert result.exit_code == 0, debug(result)
    assert (demo_project / "target" / "manifest.json").is_file()


def test_parse_write_json_schema_publishes_editor_schemas(demo_project: Path) -> None:
    result = invoke(["parse", "--project-dir", str(demo_project), "--write-json-schema"])
    assert result.exit_code == 0, debug(result)
    assert (demo_project / "target" / "json-schemas" / "models.schema.json").is_file()
    assert "wrote 8 JSON Schemas" in result.output


def test_compile_writes_manifest(demo_project: Path) -> None:
    result = invoke(["compile", "--project-dir", str(demo_project), "--anchor", ANCHOR])
    assert result.exit_code == 0, debug(result)
    assert (demo_project / "target" / "manifest.json").is_file()
    assert "wrote" in result.output


# -- run / build / evaluate / docs (fake adapters) ----------------------------------


def test_build_evaluate_docs_happy_path(demo_project: Path) -> None:
    build = invoke(["build", "--project-dir", str(demo_project), "--anchor", ANCHOR])
    assert build.exit_code == 0, debug(build)
    assert "churn_model" in build.output  # results table on stdout

    evaluate = invoke(
        [
            "evaluate",
            "--model",
            "churn_model",
            "--project-dir",
            str(demo_project),
            "--anchor",
            ANCHOR,
        ]
    )
    assert evaluate.exit_code == 0, debug(evaluate)

    # --manifest branch, with run_results.json present from the build
    manifest = demo_project / "target" / "manifest.json"
    docs = invoke(
        ["docs", "generate", "--project-dir", str(demo_project), "--manifest", str(manifest)]
    )
    assert docs.exit_code == 0, debug(docs)
    assert (demo_project / "target" / "docs" / "index.html").is_file()


def test_docs_generate_compiles_when_no_manifest_given(demo_project: Path) -> None:
    result = invoke(["docs", "generate", "--project-dir", str(demo_project)])
    assert result.exit_code == 0, debug(result)
    assert (demo_project / "target" / "docs" / "index.html").is_file()


def test_execution_command_exits_two_on_quality_failure(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    from mbt.execute import orchestrator

    def fake_run_command(opts):
        assert opts.command == "run"
        return make_results(
            "run",
            NodeResult(
                unique_id="model.demo.m",
                status="gate_failed",
                message="gate pr_auc failed\n  hint: lower the threshold",
            ),
        )

    monkeypatch.setattr(orchestrator, "run_command", fake_run_command)
    result = invoke(["run", "--project-dir", str(tmp_path)])
    assert result.exit_code == 2, debug(result)
    assert "gate_failed" in result.output


def test_evaluate_exits_two_on_gate_failure(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    from mbt.execute import orchestrator

    def fake_run_evaluate(opts, *, model_name, version=None, stage=None, apply_gates=False):
        assert model_name == "m" and version == "2" and stage == "staging" and apply_gates
        return make_results("evaluate", NodeResult(unique_id="model.demo.m", status="gate_failed"))

    monkeypatch.setattr(orchestrator, "run_evaluate", fake_run_evaluate)
    result = invoke(
        [
            "evaluate",
            "--model",
            "m",
            "--version",
            "2",
            "--stage",
            "staging",
            "--gates",
            "--project-dir",
            str(tmp_path),
        ]
    )
    assert result.exit_code == 2, debug(result)


def test_monitor_renders_results_and_exit_codes(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    import mbt.execute.monitor as monitor_mod

    healthy = make_results(
        "monitor",
        NodeResult(unique_id="scoring.demo.s", status="success", metrics={"psi": 0.01}),
    )
    monkeypatch.setattr(monitor_mod, "run_monitor", lambda opts: healthy)
    ok = invoke(["monitor", "--project-dir", str(tmp_path)])
    assert ok.exit_code == 0, debug(ok)

    failing = make_results(
        "monitor", NodeResult(unique_id="scoring.demo.s", status="monitor_failed")
    )
    monkeypatch.setattr(monitor_mod, "run_monitor", lambda opts: failing)
    bad = invoke(["monitor", "--project-dir", str(tmp_path)])
    assert bad.exit_code == 2, debug(bad)


# -- ls / show ----------------------------------------------------------------------


def test_ls_output_modes(demo_project: Path) -> None:
    base = ["ls", "--project-dir", str(demo_project)]

    names = invoke([*base, "--output", "name"])
    assert names.exit_code == 0, debug(names)
    listed = names.stdout.splitlines()
    assert {"churn_model", "churn_training", "subscribers"} <= set(listed)

    paths = invoke([*base, "--output", "path"])
    assert any(line.endswith("churn_model.yml") for line in paths.stdout.splitlines())

    as_json = invoke([*base, "--output", "json"])
    payload = json.loads(as_json.stdout)
    assert {entry["resource_type"] for entry in payload} == {"model", "dataset", "source"}

    table = invoke([*base, "--select", "churn_model"])
    assert table.exit_code == 0, debug(table)
    assert "churn_model" in table.output
    assert "churn_training" not in table.output  # selector applied


def test_show_renders_yaml_and_json(demo_project: Path) -> None:
    yaml_out = invoke(["show", "churn_model", "--project-dir", str(demo_project)])
    assert yaml_out.exit_code == 0, debug(yaml_out)
    assert "name: churn_model" in yaml_out.stdout

    json_out = invoke(
        ["show", "churn_model", "--project-dir", str(demo_project), "--output", "json"]
    )
    assert json_out.exit_code == 0, debug(json_out)
    assert json.loads(json_out.stdout)["name"] == "churn_model"


def test_show_unknown_resource_suggests_a_name(demo_project: Path) -> None:
    result = invoke(["show", "churn_modell", "--project-dir", str(demo_project)])
    assert result.exit_code == 1, debug(result)
    assert "unknown resource" in result.stderr
    assert "churn_model" in result.stderr  # did-you-mean hint


# -- state diff ---------------------------------------------------------------------


def test_state_diff_outputs_and_modified_detection(demo_project: Path) -> None:
    compiled = invoke(["compile", "--project-dir", str(demo_project), "--anchor", ANCHOR])
    assert compiled.exit_code == 0, debug(compiled)
    reference = demo_project / "target" / "manifest.json"
    base = [
        "state",
        "diff",
        "--state",
        str(reference),
        "--project-dir",
        str(demo_project),
        "--anchor",
        ANCHOR,
    ]

    clean = invoke(base)
    assert clean.exit_code == 0, debug(clean)
    assert "no node changes" in clean.output

    as_json = invoke([*base, "--output", "json"])
    payload = json.loads(as_json.stdout)
    assert payload["modified"] == [] and payload["env"]["changed"] is False

    # a CLI var flips the gate threshold -> the model's config hash changes
    modified = invoke([*base, "--vars", "default_threshold: 0.9"])
    assert modified.exit_code == 0, debug(modified)
    assert "churn_model" in modified.output
    assert "no node changes" not in modified.output


def test_state_diff_emits_state_diffed_event(demo_project: Path) -> None:
    """state-diff puts a StateDiffed on the bus (rendered to stderr) so the
    command is not silent on the event stream - the counts track the diff."""
    compiled = invoke(["compile", "--project-dir", str(demo_project), "--anchor", ANCHOR])
    assert compiled.exit_code == 0, debug(compiled)
    reference = demo_project / "target" / "manifest.json"
    base = ["state", "diff", "--state", str(reference), "--project-dir", str(demo_project)]

    clean = invoke([*base, "--anchor", ANCHOR])
    assert clean.exit_code == 0, debug(clean)
    assert "state diff: 0 added, 0 removed, 0 modified" in clean.stderr

    # a CLI var flips the gate threshold -> the model's config hash changes
    modified = invoke([*base, "--anchor", ANCHOR, "--vars", "default_threshold: 0.9"])
    assert modified.exit_code == 0, debug(modified)
    assert "1 modified" in modified.stderr


def test_state_diff_manifest_input_and_env_change(demo_project: Path) -> None:
    compiled = invoke(["compile", "--project-dir", str(demo_project), "--anchor", ANCHOR])
    assert compiled.exit_code == 0, debug(compiled)
    reference = demo_project / "target" / "manifest.json"

    same = invoke(
        [
            "state",
            "diff",
            "--state",
            str(reference),
            "--manifest",
            str(reference),
            "--project-dir",
            str(demo_project),
        ]
    )
    assert same.exit_code == 0, debug(same)
    assert "no node changes" in same.output

    payload = json.loads(reference.read_text())
    payload["metadata"]["env_digest"] = "sha256:" + "0" * 64
    tampered = demo_project / "target" / "reference_env.json"
    tampered.write_text(json.dumps(payload))
    env = invoke(
        [
            "state",
            "diff",
            "--state",
            str(tampered),
            "--manifest",
            str(reference),
            "--project-dir",
            str(demo_project),
        ]
    )
    assert env.exit_code == 0, debug(env)
    assert "environment digest CHANGED" in env.output


# -- docs serve ---------------------------------------------------------------------


def test_docs_serve_requires_generated_docs(tmp_path: Path) -> None:
    result = invoke(["docs", "serve", "--project-dir", str(tmp_path)])
    assert result.exit_code == 1, debug(result)
    assert "no generated docs" in result.stderr


def test_docs_serve_starts_http_server(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    import http.server

    docs_dir = tmp_path / "target" / "docs"
    docs_dir.mkdir(parents=True)
    (docs_dir / "index.html").write_text("<html></html>")
    served: dict[str, object] = {}

    class FakeServer:
        def __init__(self, addr: tuple[str, int], handler: object) -> None:
            served["addr"] = addr

        def serve_forever(self) -> None:
            served["running"] = True

    monkeypatch.setattr(http.server, "ThreadingHTTPServer", FakeServer)
    result = invoke(["docs", "serve", "--project-dir", str(tmp_path), "--port", "8123"])
    assert result.exit_code == 0, debug(result)
    assert served == {"addr": ("127.0.0.1", 8123), "running": True}
    assert "serving" in result.output


# -- run-operation ------------------------------------------------------------------


def test_run_operation_renders_macro(demo_project: Path) -> None:
    write(
        demo_project / "macros" / "helpers.jinja",
        "{% macro greet(name) %}hello {{ name }}{% endmacro %}",
    )
    result = invoke(
        ["run-operation", "greet", "--args", "name: world", "--project-dir", str(demo_project)]
    )
    assert result.exit_code == 0, debug(result)
    assert result.stdout.strip() == "hello world"


def test_run_operation_unknown_macro(demo_project: Path) -> None:
    result = invoke(["run-operation", "nope", "--project-dir", str(demo_project)])
    assert result.exit_code == 1, debug(result)
    assert "unknown macro" in result.stderr


def test_verbose_flag_wires_debug_sink() -> None:
    from mbt.cli.common import CLIContext, setup_bus
    from mbt.events import ConsoleSink, get_bus

    setup_bus(CLIContext(verbose=True))
    sink = get_bus()._sinks[0]
    assert isinstance(sink, ConsoleSink) and sink.verbose is True

    setup_bus(CLIContext(verbose=False))
    sink = get_bus()._sinks[0]
    assert isinstance(sink, ConsoleSink) and sink.verbose is False


def test_command_accepts_verbose_flag(demo_project: Path) -> None:
    # -v threads through make_ctx on a real command (FR-CLI-04 parity).
    result = invoke(["parse", "-v", "--project-dir", str(demo_project)])
    assert result.exit_code == 0, debug(result)


# -- main() entry point (TSD §17 exit-code semantics) -------------------------------


def test_main_exits_zero_on_success(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    from mbt.cli import main as cli_main

    monkeypatch.setattr(sys, "argv", ["mbt", "clean", "--project-dir", str(tmp_path)])
    with pytest.raises(SystemExit) as excinfo:
        cli_main.main()
    assert excinfo.value.code == 0


def test_main_remaps_usage_errors_to_one(
    monkeypatch: pytest.MonkeyPatch, capsys: pytest.CaptureFixture[str]
) -> None:
    from mbt.cli import main as cli_main

    monkeypatch.setattr(sys, "argv", ["mbt", "no-such-command"])
    with pytest.raises(SystemExit) as excinfo:
        cli_main.main()
    assert excinfo.value.code == 1  # NOT click's default 2: that means quality failure
    assert "No such command" in capsys.readouterr().err


def test_main_propagates_exit_exception_codes(monkeypatch: pytest.MonkeyPatch) -> None:
    from mbt.cli import main as cli_main

    def raise_exit(standalone_mode: bool = True) -> None:
        raise click.exceptions.Exit(3)

    monkeypatch.setattr(cli_main, "app", raise_exit)
    monkeypatch.setattr(sys, "argv", ["mbt"])
    with pytest.raises(SystemExit) as excinfo:
        cli_main.main()
    assert excinfo.value.code == 3


def test_main_shows_click_exceptions_as_hard_errors(
    monkeypatch: pytest.MonkeyPatch, capsys: pytest.CaptureFixture[str]
) -> None:
    from mbt.cli import main as cli_main

    def raise_click_exception(standalone_mode: bool = True) -> None:
        raise click.ClickException("boom")

    monkeypatch.setattr(cli_main, "app", raise_click_exception)
    monkeypatch.setattr(sys, "argv", ["mbt"])
    with pytest.raises(SystemExit) as excinfo:
        cli_main.main()
    assert excinfo.value.code == 1
    assert "boom" in capsys.readouterr().err


def test_main_handles_abort(
    monkeypatch: pytest.MonkeyPatch, capsys: pytest.CaptureFixture[str]
) -> None:
    from mbt.cli import main as cli_main

    def raise_abort(standalone_mode: bool = True) -> None:
        raise click.Abort

    monkeypatch.setattr(cli_main, "app", raise_abort)
    monkeypatch.setattr(sys, "argv", ["mbt"])
    with pytest.raises(SystemExit) as excinfo:
        cli_main.main()
    assert excinfo.value.code == 1
    assert "aborted" in capsys.readouterr().err


def test_main_wraps_unexpected_errors_as_internal_error(
    monkeypatch: pytest.MonkeyPatch, capsys: pytest.CaptureFixture[str]
) -> None:
    from mbt.cli import main as cli_main

    def raise_unexpected(standalone_mode: bool = True) -> None:
        raise ValueError("kaboom")  # a non-MbtError from the coordinator

    monkeypatch.delenv("MBT_DEBUG", raising=False)
    monkeypatch.setattr(cli_main, "app", raise_unexpected)
    monkeypatch.setattr(sys, "argv", ["mbt"])
    with pytest.raises(SystemExit) as excinfo:
        cli_main.main()
    assert excinfo.value.code == 1  # hard error, not a raw traceback
    err = capsys.readouterr().err
    assert "Internal error" in err
    assert "ValueError: kaboom" in err
    assert "MBT_DEBUG=1" in err


def test_main_debug_env_reraises_unexpected_errors(monkeypatch: pytest.MonkeyPatch) -> None:
    from mbt.cli import main as cli_main

    def raise_unexpected(standalone_mode: bool = True) -> None:
        raise ValueError("kaboom")

    monkeypatch.setenv("MBT_DEBUG", "1")
    monkeypatch.setattr(cli_main, "app", raise_unexpected)
    monkeypatch.setattr(sys, "argv", ["mbt"])
    with pytest.raises(ValueError, match="kaboom"):  # re-raised for the traceback
        cli_main.main()


# re-running an already-imported module is the point here, so runpy's
# "found in sys.modules" warning is expected noise
@pytest.mark.filterwarnings("ignore::RuntimeWarning")
def test_module_runs_main_when_executed(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setattr(sys, "argv", ["mbt", "--version"])
    with pytest.raises(SystemExit) as excinfo:
        runpy.run_module("mbt.cli.main", run_name="__main__")
    assert excinfo.value.code == 0
