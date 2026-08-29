"""The ``mbt`` Typer application: all commands (TSD §3).

Every command is non-interactive-safe (FR-CLI-01); exit codes follow TSD §17
(0 success, 1 hard error, 2 quality failure);
``--target/--vars/--select/--exclude/--threads/--state/--manifest`` behave
identically wherever they appear (FR-CLI-04). Common flags are per-command,
dbt-style: ``mbt build --target prod --vars '...'``.
"""

import functools
import json
import os
import sys
from collections.abc import Callable
from pathlib import Path
from typing import Annotated, Any

import click
import typer
import yaml
from rich.table import Table

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
from mbt.exceptions import ConfigError, MbtError


def _control_flow_exceptions(name: str) -> tuple[type[BaseException], ...]:
    """Every distinct class called ``name`` that a typer/click command might
    raise, across the places different versions keep them.

    typer >= 0.20 vendors click, and its commands raise the VENDORED exception
    types, which are not subclasses of the real click's - so both have to be
    caught or ``mbt`` exits on its own control flow. But which module holds
    which name is not stable either: typer 0.27 moved ``Exit`` and ``Abort``
    out of ``typer._click.exceptions`` into ``typer.exceptions`` as plain
    RuntimeErrors, leaving ``UsageError``/``ClickException`` where they were.

    The previous shape guarded the IMPORT (`except ImportError`), which that
    move sails straight through: the module still imports, it has simply lost
    two attributes - so every single `mbt` invocation died with
    `AttributeError: module 'typer._click.exceptions' has no attribute 'Exit'`
    while evaluating the except clause. Not a test failure: the CLI, gone, for
    anyone installing unpinned. The nightly upstream-resolution tier caught it
    before a release did.

    So: probe by name, take whatever is there, and stay indifferent to which
    module upstream keeps it in. A name found nowhere yields an empty tuple,
    which `except ()` simply never matches - degrading to "we stop special-
    casing this control-flow exception" rather than to a crash on the way to
    reporting some other error. tests/test_cli_exception_sources.py asserts
    none of them are actually empty, so the degradation cannot pass unnoticed.
    """
    sources: list[Any] = [click.exceptions, click, typer]
    try:  # absent on typer < 0.20, which drives the real click directly
        from typer._click import exceptions as vendored

        sources.append(vendored)
    except ImportError:  # typer < 0.20 drives the real click directly
        pass

    found: list[type[BaseException]] = []
    for source in sources:
        candidate = getattr(source, name, None)
        if (
            isinstance(candidate, type)
            and issubclass(candidate, BaseException)
            and candidate not in found
        ):
            found.append(candidate)
    return tuple(found)


#: Resolved once at import; see _control_flow_exceptions for why by name.
EXIT_EXCEPTIONS = _control_flow_exceptions("Exit")
USAGE_ERROR_EXCEPTIONS = _control_flow_exceptions("UsageError")
CLICK_EXCEPTIONS = _control_flow_exceptions("ClickException")
ABORT_EXCEPTIONS = _control_flow_exceptions("Abort")

app = typer.Typer(
    name="mbt",
    help="mbt: a declarative build tool for machine learning models.",
    no_args_is_help=True,
    add_completion=True,
    pretty_exceptions_enable=False,
)
docs_app = typer.Typer(help="Generate or serve the model cards + lineage site.")
state_app = typer.Typer(help="Compare manifests (state:modified mechanics).")
predictions_app = typer.Typer(help="Inspect the prediction store (runs + ground-truth ledger).")
app.add_typer(docs_app, name="docs")
app.add_typer(state_app, name="state")
app.add_typer(predictions_app, name="predictions")


def _version_callback(show: bool) -> None:
    if show:
        import mbt

        typer.echo(f"mbt {mbt.__version__}")
        raise typer.Exit()


@app.callback()
def _root(
    version: Annotated[
        bool,
        typer.Option(
            "--version",
            callback=_version_callback,
            is_eager=True,
            help="Show the mbt version and exit.",
        ),
    ] = False,
) -> None:
    """mbt: a declarative build tool for machine learning models."""


# -- common option aliases (FR-CLI-04) ------------------------------------------

ProjectDirOpt = Annotated[Path, typer.Option("--project-dir", help="Project root (default: cwd).")]
ProfilesDirOpt = Annotated[
    Path | None, typer.Option("--profiles-dir", help="Directory holding profiles.yml.")
]
TargetOpt = Annotated[
    str | None, typer.Option("--target", "-t", help="Profile target (dev/prod/...).")
]
VarsOpt = Annotated[str | None, typer.Option("--vars", help="YAML/JSON dict overriding vars.")]
LogFormatOpt = Annotated[str, typer.Option("--log-format", help="text | json (events, on stderr).")]
QuietOpt = Annotated[bool, typer.Option("--quiet", "-q", help="Suppress event output.")]
VerboseOpt = Annotated[
    bool, typer.Option("--verbose", "-v", help="Show debug-level events (text mode).")
]
SelectOpt = Annotated[
    list[str] | None, typer.Option("--select", "-s", help="Node selector(s); space = union.")
]
ExcludeOpt = Annotated[list[str] | None, typer.Option("--exclude", help="Selector(s) to subtract.")]
ThreadsOpt = Annotated[
    int | None, typer.Option("--threads", help="Parallel DAG branches (default: target).")
]
StateOpt = Annotated[
    str | None,
    typer.Option("--state", help="Reference manifest path/URI for state: selectors."),
]
StateIncludeEnvOpt = Annotated[
    bool,
    typer.Option(
        "--state-include-env",
        help="Treat env_digest changes as modifying every node (ADR-7).",
    ),
]
ManifestOpt = Annotated[
    str | None,
    typer.Option("--manifest", help="Execute a stored manifest verbatim (FR-RUN-11)."),
]
AllowEnvMismatchOpt = Annotated[
    bool,
    typer.Option(
        "--allow-env-mismatch",
        help="Downgrade the --manifest env_digest check from error to warning (ADR-19).",
    ),
]
AnchorOpt = Annotated[
    str | None, typer.Option("--anchor", help="Pin the time anchor (ISO timestamp).")
]
DeepSnapshotOpt = Annotated[
    bool, typer.Option("--deep-snapshot", help="Content-hash snapshots (slow, exact).")
]
OutputOpt = Annotated[str, typer.Option("--output", "-o", help="Output format.")]


def make_ctx(
    project_dir: Path,
    profiles_dir: Path | None,
    target: str | None,
    vars_: str | None,
    log_format: str,
    quiet: bool,
    verbose: bool = False,
    chdir: bool = True,
) -> CLIContext:
    """Build the per-command context and enter the project directory.

    The coordinator chdirs to the project dir so config-relative paths
    (file:// artifact stores, sqlite URIs, adapter roots) resolve against
    the project no matter where mbt was invoked - job subprocesses already
    run with cwd=project_dir, this makes the coordinator match. Paths the
    user typed on the command line are absolutized against the invocation
    cwd via ctx.resolve_cli_path BEFORE they are used.
    """
    invocation_cwd = Path.cwd()
    project_dir = (invocation_cwd / project_dir).resolve()
    if profiles_dir is not None:
        profiles_dir = (invocation_cwd / profiles_dir).resolve()
    if chdir:
        if not project_dir.is_dir():
            raise ConfigError(
                f"--project-dir {project_dir} is not a directory",
                hint="run mbt from a project or point --project-dir at one",
            )
        os.chdir(project_dir)
    ctx = CLIContext(
        project_dir=project_dir,
        invocation_cwd=invocation_cwd,
        profiles_dir=profiles_dir,
        target=target,
        cli_vars=parse_vars(vars_),
        log_format=log_format,
        quiet=quiet,
        verbose=verbose,
    )
    setup_bus(ctx)
    return ctx


def guard(fn: Callable[..., Any]) -> Callable[..., Any]:
    """Uniform MbtError -> message + exit code handling (TSD §17)."""

    @functools.wraps(fn)
    def wrapper(*args: Any, **kwargs: Any) -> Any:
        try:
            return fn(*args, **kwargs)
        except MbtError as exc:
            raise fail(exc) from exc

    return wrapper


# -- project lifecycle -------------------------------------------------------------


@app.command()
@guard
def init(
    name: Annotated[str, typer.Argument(help="Project name (lowercase snake_case).")],
    project_dir: ProjectDirOpt = Path("."),
    log_format: LogFormatOpt = "text",
    quiet: QuietOpt = False,
    verbose: VerboseOpt = False,
) -> None:
    """Scaffold a golden-path project (FR-PROJ-01)."""
    from mbt.cli.scaffold import scaffold_project

    # chdir=False: project_dir is the parent to scaffold into, not a project
    cli = make_ctx(project_dir, None, None, None, log_format, quiet, verbose, chdir=False)
    destination = scaffold_project(name, cli.project_dir)
    out_console.print(f"Created [bold]{destination}[/bold]")
    out_console.print(
        f"Next steps:\n  cd {name}\n  python scripts/generate_sample_data.py\n  mbt build"
    )


@app.command()
@guard
def deps(
    project_dir: ProjectDirOpt = Path("."),
    dry_run: Annotated[bool, typer.Option("--dry-run", help="Print, do not install.")] = False,
    log_format: LogFormatOpt = "text",
    quiet: QuietOpt = False,
    verbose: VerboseOpt = False,
) -> None:
    """Install adapter packages pinned in packages.yml (FR-PROJ-04)."""
    from mbt.deps import install_packages, load_packages

    cli = make_ctx(project_dir, None, None, None, log_format, quiet, verbose)
    pinned = cli.project_dir / "requirements.txt"
    requirements = install_packages(
        load_packages(cli.project_dir),
        dry_run=dry_run,
        requirements_file=pinned if pinned.is_file() else None,
    )
    verb = "would install" if dry_run else "installed"
    out_console.print(f"{verb}: " + (", ".join(requirements) or "(nothing)"))


@app.command()
@guard
def clean(
    project_dir: ProjectDirOpt = Path("."),
    profiles_dir: ProfilesDirOpt = None,
    target: TargetOpt = None,
    vars_: VarsOpt = None,
    artifacts_older_than: Annotated[
        str | None,
        typer.Option(
            "--artifacts-older-than",
            help="Prune artifact-store run prefixes older than a duration (30d, 12h); "
            "stage champions and the latest run's artifacts always survive.",
        ),
    ] = None,
    dry_run: Annotated[
        bool, typer.Option("--dry-run", help="List what artifact GC would delete.")
    ] = False,
) -> None:
    """Delete target/ (default), or prune the artifact store (--artifacts-older-than)."""
    import shutil

    if artifacts_older_than is None:
        target_dir = project_dir / "target"
        if target_dir.is_dir():
            shutil.rmtree(target_dir)
            out_console.print(f"removed {target_dir}")
        else:
            out_console.print(f"nothing to clean at {target_dir}")
        # Age out leaked error-payload dirs (kept for debugging, never
        # self-cleaned); recent ones survive for an in-progress reproduction.
        from datetime import UTC, datetime, timedelta

        from mbt.adapters.local.compute import sweep_stale_job_payloads

        swept = sweep_stale_job_payloads(datetime.now(tz=UTC) - timedelta(days=7))
        if swept:
            out_console.print(f"aged out {len(swept)} stale job payload dir(s) (>7d old)")
        return

    from datetime import UTC, datetime

    from mbt.adapters.registry import get_registry
    from mbt.compile.windows import parse_window
    from mbt.exceptions import ConfigError
    from mbt.gc import (
        apply_gc_plan,
        artifact_gc_plan,
        champion_artifact_uris,
        run_results_artifact_uris,
    )
    from mbt.parsing import parse_project
    from mbt.runtime import registry_adapter as build_registry_adapter
    from mbt.runtime import resolve_artifact_store_uri

    window = parse_window(artifacts_older_than)
    if window.start.kind != "duration" or window.start.delta is None:
        raise ConfigError(
            f"--artifacts-older-than expects a duration, got {artifacts_older_than!r}",
            hint="examples: 30d, 2w, 12h, 3mo, 1y",
        )
    cutoff = window.start.resolve(datetime.now(tz=UTC))  # a past duration is negative

    cli = make_ctx(project_dir, profiles_dir, target, vars_, "text", False)
    parsed = parse_project(cli.project_dir, cli_vars=cli.cli_vars)
    profiles = cli.profiles(parsed)
    store_uri = resolve_artifact_store_uri(
        profiles.target.artifact_store, cli.project_dir.resolve()
    )
    keep_uris = run_results_artifact_uris(cli.project_dir)
    registry_adapter = build_registry_adapter(profiles, cli.project_dir.resolve(), get_registry())
    keep_uris |= champion_artifact_uris(parsed, registry_adapter)

    plan = artifact_gc_plan(store_uri, cutoff=cutoff, keep_uris=keep_uris)
    verb = "would delete" if dry_run else "deleted"
    for path in plan.delete:
        out_console.print(f"{verb} {path}")
    if not dry_run:
        apply_gc_plan(plan)
    out_console.print(
        f"{verb} {len(plan.delete)} run prefix(es) ({plan.freed_bytes} bytes); "
        f"kept {len(plan.keep)}"
    )


# -- parse / compile ------------------------------------------------------------------


@app.command()
@guard
def parse(
    project_dir: ProjectDirOpt = Path("."),
    profiles_dir: ProfilesDirOpt = None,
    target: TargetOpt = None,
    vars_: VarsOpt = None,
    write_json_schema: Annotated[
        bool, typer.Option("--write-json-schema", help="Publish JSON Schemas for editors.")
    ] = False,
    log_format: LogFormatOpt = "text",
    quiet: QuietOpt = False,
    verbose: VerboseOpt = False,
) -> None:
    """Validate all configs and build the DAG; no execution (FR-PARSE-01)."""
    from mbt.events import get_bus
    from mbt.events.models import ParseCompleted, ParseStarted
    from mbt.parsing import parse_project

    cli = make_ctx(project_dir, profiles_dir, target, vars_, log_format, quiet, verbose)
    bus = get_bus()
    bus.emit(ParseStarted(project=cli.project_dir.name))
    parsed = parse_project(cli.project_dir, cli_vars=cli.cli_vars)
    bus.emit(
        ParseCompleted(
            resources=len(parsed.nodes) + len(parsed.sources) + len(parsed.exposures),
            errors=len(parsed.report.errors),
            elapsed_s=parsed.elapsed_s,
        )
    )
    print_warnings(parsed)
    out_console.print(
        f"Parsed [bold]{len(parsed.nodes)}[/bold] nodes, "
        f"{len(parsed.sources)} sources, {len(parsed.exposures)} exposures "
        f"in {parsed.elapsed_s:.2f}s"
    )
    if write_json_schema:
        from mbt.cli.schema_export import write_json_schemas

        written = write_json_schemas(cli.project_dir / "target" / "json-schemas")
        out_console.print(f"wrote {len(written)} JSON Schemas to target/json-schemas/")


@app.command()
@guard
def compile(
    project_dir: ProjectDirOpt = Path("."),
    profiles_dir: ProfilesDirOpt = None,
    target: TargetOpt = None,
    vars_: VarsOpt = None,
    anchor: AnchorOpt = None,
    deep_snapshot: DeepSnapshotOpt = False,
    log_format: LogFormatOpt = "text",
    quiet: QuietOpt = False,
    verbose: VerboseOpt = False,
) -> None:
    """Resolve Jinja + profiles + snapshots into target/manifest.json (FR-COMP-01)."""
    from mbt.compile.compiler import CompileOptions, compile_project
    from mbt.parsing import parse_project

    cli = make_ctx(project_dir, profiles_dir, target, vars_, log_format, quiet, verbose)
    parsed = parse_project(cli.project_dir, cli_vars=cli.cli_vars)
    print_warnings(parsed)
    profiles = cli.profiles(parsed)
    path = cli.project_dir / "target" / "manifest.json"
    manifest = compile_project(
        parsed,
        profiles,
        options=CompileOptions(
            anchor=parse_anchor(anchor), deep_snapshot=deep_snapshot, manifest_path=path
        ),
        cli_vars=cli.cli_vars,
    )
    manifest.write(path)
    out_console.print(f"wrote {path}")


# -- run / build / test -----------------------------------------------------------------


def _register_execution_command(command: str, help_text: str) -> None:
    @app.command(name=command, help=help_text)
    @guard
    def _cmd(
        project_dir: ProjectDirOpt = Path("."),
        profiles_dir: ProfilesDirOpt = None,
        target: TargetOpt = None,
        vars_: VarsOpt = None,
        select: SelectOpt = None,
        exclude: ExcludeOpt = None,
        threads: ThreadsOpt = None,
        fail_fast: Annotated[
            bool, typer.Option("--fail-fast", help="Stop everything on first failure.")
        ] = False,
        state: StateOpt = None,
        state_include_env: StateIncludeEnvOpt = False,
        manifest: ManifestOpt = None,
        allow_env_mismatch: AllowEnvMismatchOpt = False,
        anchor: AnchorOpt = None,
        deep_snapshot: DeepSnapshotOpt = False,
        log_format: LogFormatOpt = "text",
        quiet: QuietOpt = False,
        verbose: VerboseOpt = False,
    ) -> None:
        from mbt.execute.orchestrator import run_command

        cli = make_ctx(project_dir, profiles_dir, target, vars_, log_format, quiet, verbose)
        results = run_command(
            cli.invocation(
                command,
                select=select,
                exclude=exclude,
                threads=threads,
                fail_fast=fail_fast,
                state=cli.resolve_cli_path(state),
                state_include_env=state_include_env,
                manifest_path=cli.resolve_cli_path(manifest),
                allow_env_mismatch=allow_env_mismatch,
                anchor=parse_anchor(anchor),
                deep_snapshot=deep_snapshot,
            )
        )
        render_results_table(results, cli)
        code = results.exit_code()
        if code:
            raise typer.Exit(code)


_register_execution_command("run", "Build datasets and train models in DAG order (FR-RUN-01).")
_register_execution_command(
    "build", "run + test interleaved in DAG order - the CI workhorse (FR-RUN-01)."
)
_register_execution_command("test", "Data tests + model quality gates; never trains (FR-TEST-01).")
_register_execution_command(
    "score", "Batch-score fresh data with registered champions + shift monitors (ADR-20)."
)


@app.command()
@guard
def evaluate(
    model: Annotated[str, typer.Option("--model", help="Model resource name.")],
    project_dir: ProjectDirOpt = Path("."),
    profiles_dir: ProfilesDirOpt = None,
    target: TargetOpt = None,
    vars_: VarsOpt = None,
    version: Annotated[
        str | None, typer.Option("--version", help="Registry version (default: latest).")
    ] = None,
    stage: Annotated[
        str | None, typer.Option("--stage", help="Stage to pull the version from.")
    ] = None,
    gates: Annotated[
        bool, typer.Option("--gates", help="Apply gate logic to the fresh metrics.")
    ] = False,
    manifest: ManifestOpt = None,
    allow_env_mismatch: AllowEnvMismatchOpt = False,
    anchor: AnchorOpt = None,
    log_format: LogFormatOpt = "text",
    quiet: QuietOpt = False,
    verbose: VerboseOpt = False,
) -> None:
    """Re-evaluate a registered artifact on freshly built data (FR-RUN-07)."""
    from mbt.execute.orchestrator import run_evaluate

    cli = make_ctx(project_dir, profiles_dir, target, vars_, log_format, quiet, verbose)
    results = run_evaluate(
        cli.invocation(
            "evaluate",
            manifest_path=cli.resolve_cli_path(manifest),
            allow_env_mismatch=allow_env_mismatch,
            anchor=parse_anchor(anchor),
        ),
        model_name=model,
        version=version,
        stage=stage,
        apply_gates=gates,
    )
    render_results_table(results, cli)
    code = results.exit_code()
    if code:
        raise typer.Exit(code)


@app.command()
@guard
def monitor(
    project_dir: ProjectDirOpt = Path("."),
    profiles_dir: ProfilesDirOpt = None,
    target: TargetOpt = None,
    vars_: VarsOpt = None,
    select: SelectOpt = None,
    exclude: ExcludeOpt = None,
    threads: ThreadsOpt = None,
    fail_fast: Annotated[
        bool, typer.Option("--fail-fast", help="Stop everything on first failure.")
    ] = False,
    manifest: ManifestOpt = None,
    allow_env_mismatch: AllowEnvMismatchOpt = False,
    anchor: AnchorOpt = None,
    deep_snapshot: DeepSnapshotOpt = False,
    log_format: LogFormatOpt = "text",
    quiet: QuietOpt = False,
    verbose: VerboseOpt = False,
) -> None:
    """Evaluate matured predictions against arrived labels; never trains (ADR-21)."""
    from mbt.execute.monitor import run_monitor

    cli = make_ctx(project_dir, profiles_dir, target, vars_, log_format, quiet, verbose)
    results = run_monitor(
        cli.invocation(
            "monitor",
            select=select,
            exclude=exclude,
            threads=threads,
            fail_fast=fail_fast,
            manifest_path=cli.resolve_cli_path(manifest),
            allow_env_mismatch=allow_env_mismatch,
            anchor=parse_anchor(anchor),
            deep_snapshot=deep_snapshot,
        )
    )
    render_results_table(results, cli)
    code = results.exit_code()
    if code:
        raise typer.Exit(code)


def _mark(value: bool | None) -> str:
    return "-" if value is None else ("yes" if value else "no")


@predictions_app.command("ls")
@guard
def predictions_ls(
    project_dir: ProjectDirOpt = Path("."),
    profiles_dir: ProfilesDirOpt = None,
    target: TargetOpt = None,
    vars_: VarsOpt = None,
    manifest: ManifestOpt = None,
    output: OutputOpt = "table",
    log_format: LogFormatOpt = "text",
    quiet: QuietOpt = False,
    verbose: VerboseOpt = False,
) -> None:
    """List prediction runs across scoring nodes (matured/evaluated state)."""
    from dataclasses import asdict

    from mbt.execute.predictions_view import list_prediction_runs

    cli = make_ctx(project_dir, profiles_dir, target, vars_, log_format, quiet, verbose)
    runs = list_prediction_runs(
        cli.invocation("predictions", manifest_path=cli.resolve_cli_path(manifest))
    )
    if output == "json":
        typer.echo(json.dumps([asdict(r) for r in runs], indent=2))
        return
    if not runs:
        out_console.print("no prediction runs found (score first, or check --target)")
        return
    table = Table()
    for column in ("scoring", "run_key", "scored_at", "model", "rows", "matured", "evaluated"):
        table.add_column(column)
    for run in runs:
        table.add_row(
            run.scoring,
            run.run_key,
            run.scored_at,
            f"{run.model_name} v{run.model_version}",
            str(run.row_count),
            _mark(run.matured),
            _mark(run.evaluated),
        )
    out_console.print(table)


@predictions_app.command("show")
@guard
def predictions_show(
    run_key: Annotated[str, typer.Argument(help="The prediction run_key to detail.")],
    project_dir: ProjectDirOpt = Path("."),
    profiles_dir: ProfilesDirOpt = None,
    target: TargetOpt = None,
    vars_: VarsOpt = None,
    manifest: ManifestOpt = None,
    output: OutputOpt = "table",
    log_format: LogFormatOpt = "text",
    quiet: QuietOpt = False,
    verbose: VerboseOpt = False,
) -> None:
    """Detail one prediction run: run info plus its ground-truth ledger marker."""
    from dataclasses import asdict

    from mbt.exceptions import StateError
    from mbt.execute.predictions_view import show_prediction_run

    cli = make_ctx(project_dir, profiles_dir, target, vars_, log_format, quiet, verbose)
    run = show_prediction_run(
        cli.invocation("predictions", manifest_path=cli.resolve_cli_path(manifest)), run_key
    )
    if run is None:
        raise StateError(
            f"no prediction run {run_key!r} found", hint="mbt predictions ls to list runs"
        )
    if output == "json":
        typer.echo(json.dumps(asdict(run), indent=2))
        return
    out_console.print(f"[bold]{run.run_key}[/bold]  ({run.scoring})")
    out_console.print(f"  scored_at   {run.scored_at}")
    out_console.print(f"  champion    {run.model_name} v{run.model_version}")
    out_console.print(f"  rows        {run.row_count}")
    out_console.print(f"  matured     {_mark(run.matured)}")
    out_console.print(f"  evaluated   {_mark(run.evaluated)}")
    if run.evaluated:
        realized = ", ".join(f"{k}={v:.4f}" for k, v in sorted(run.realized.items())) or "(none)"
        out_console.print(f"  coverage    {run.coverage}")
        out_console.print(f"  realized    {realized}")


@app.command()
@guard
def promote(
    project_dir: ProjectDirOpt = Path("."),
    profiles_dir: ProfilesDirOpt = None,
    target: TargetOpt = None,
    vars_: VarsOpt = None,
    model: Annotated[str | None, typer.Option("--model")] = None,
    to: Annotated[str | None, typer.Option("--to", help="Target stage.")] = None,
    version: Annotated[str | None, typer.Option("--version")] = None,
    from_file: Annotated[
        Path | None, typer.Option("--from-file", help="Reviewed promotions.yml (GitOps).")
    ] = None,
    force: Annotated[
        bool, typer.Option("--force", help="Promote even without recorded gate passes.")
    ] = False,
    log_format: LogFormatOpt = "text",
    quiet: QuietOpt = False,
    verbose: VerboseOpt = False,
) -> None:
    """Transition a registered version, verifying recorded gate passes (FR-REG-03)."""
    from mbt.adapters.registry import get_registry
    from mbt.contracts import Stage
    from mbt.exceptions import ConfigError
    from mbt.parsing import parse_project
    from mbt.promote import load_promotions_file, promote_model
    from mbt.runtime import registry_adapter as build_registry_adapter

    cli = make_ctx(project_dir, profiles_dir, target, vars_, log_format, quiet, verbose)
    parsed = parse_project(cli.project_dir, cli_vars=cli.cli_vars)
    profiles = cli.profiles(parsed)
    registry_adapter = build_registry_adapter(profiles, cli.project_dir.resolve(), get_registry())

    if from_file is not None:
        from_file = Path(cli.resolve_cli_path(str(from_file)) or from_file)
        entries = load_promotions_file(from_file)
        for entry in entries:
            promote_model(
                registry_adapter,
                name=entry.model,
                to_stage=entry.to,
                version=entry.version,
                force=force,
            )
        out_console.print(f"applied {len(entries)} promotion(s) from {from_file}")
        return
    if model is None or to is None:
        raise ConfigError(
            "promote needs --model and --to (or --from-file promotions.yml)",
            hint="e.g. mbt promote --model churn_classifier --to production",
        )
    try:
        stage_token = Stage(to)
    except ValueError as exc:
        raise ConfigError(
            f"unknown stage {to!r}", hint=f"stages: {', '.join(s.value for s in Stage)}"
        ) from exc
    outcome = promote_model(
        registry_adapter, name=model, to_stage=stage_token, version=version, force=force
    )
    out_console.print(
        f"promoted [bold]{outcome.name}[/bold] v{outcome.version} -> {outcome.to_stage.value}"
    )


@app.command()
@guard
def rollback(
    project_dir: ProjectDirOpt = Path("."),
    profiles_dir: ProfilesDirOpt = None,
    target: TargetOpt = None,
    vars_: VarsOpt = None,
    model: Annotated[str | None, typer.Option("--model", help="Model resource name.")] = None,
    to_version: Annotated[
        str | None,
        typer.Option(
            "--to-version",
            help="Version to revert to (default: last gated version below the current champion).",
        ),
    ] = None,
    force: Annotated[
        bool, typer.Option("--force", help="Roll back even to a version without recorded gates.")
    ] = False,
    log_format: LogFormatOpt = "text",
    quiet: QuietOpt = False,
    verbose: VerboseOpt = False,
) -> None:
    """Revert the production champion to a prior version (incident rollback)."""
    from mbt.adapters.registry import get_registry
    from mbt.exceptions import ConfigError
    from mbt.parsing import parse_project
    from mbt.promote import rollback_model
    from mbt.runtime import registry_adapter as build_registry_adapter

    if model is None:
        raise ConfigError(
            "rollback needs --model",
            hint="e.g. mbt rollback --model churn_classifier",
        )
    cli = make_ctx(project_dir, profiles_dir, target, vars_, log_format, quiet, verbose)
    parsed = parse_project(cli.project_dir, cli_vars=cli.cli_vars)
    profiles = cli.profiles(parsed)
    registry_adapter = build_registry_adapter(profiles, cli.project_dir.resolve(), get_registry())
    outcome = rollback_model(registry_adapter, name=model, to_version=to_version, force=force)
    out_console.print(
        f"rolled back [bold]{outcome.name}[/bold] to v{outcome.version} in {outcome.to_stage.value}"
    )


# -- inspection --------------------------------------------------------------------------


@app.command()
@guard
def ls(
    project_dir: ProjectDirOpt = Path("."),
    profiles_dir: ProfilesDirOpt = None,
    target: TargetOpt = None,
    vars_: VarsOpt = None,
    select: SelectOpt = None,
    exclude: ExcludeOpt = None,
    output: OutputOpt = "table",
    log_format: LogFormatOpt = "text",
    quiet: QuietOpt = False,
    verbose: VerboseOpt = False,
) -> None:
    """List resources with selector support (FR-PARSE-05)."""
    from mbt.dag.selector import SelectableNode, select_nodes
    from mbt.parsing import parse_project

    cli = make_ctx(project_dir, profiles_dir, target, vars_, log_format, quiet, verbose)
    parsed = parse_project(cli.project_dir, cli_vars=cli.cli_vars)
    nodes: dict[str, SelectableNode] = {}
    paths: dict[str, str] = {}
    for pool in (parsed.datasets, parsed.models, parsed.scoring, parsed.exposures):
        for uid, res in pool.items():
            nodes[uid] = SelectableNode(
                unique_id=uid,
                name=res.name,
                resource_type=res.resource_type,
                tags=tuple(res.tags),
            )
            paths[uid] = res.path
    for uid, entry in parsed.sources.items():
        nodes[uid] = SelectableNode(
            unique_id=uid, name=entry.table.name, resource_type="source", tags=()
        )
        paths[uid] = entry.path

    selected = sorted(select_nodes(parsed.graph, nodes, select, exclude))
    if output == "name":
        for uid in selected:
            typer.echo(nodes[uid].name)
    elif output == "path":
        for uid in selected:
            typer.echo(paths[uid])
    elif output == "json":
        payload = [
            {
                "unique_id": uid,
                "name": nodes[uid].name,
                "resource_type": nodes[uid].resource_type,
                "tags": list(nodes[uid].tags),
                "path": paths[uid],
            }
            for uid in selected
        ]
        typer.echo(json.dumps(payload, indent=2))
    else:
        table = Table()
        table.add_column("unique_id")
        table.add_column("type")
        table.add_column("tags")
        table.add_column("path")
        for uid in selected:
            table.add_row(uid, nodes[uid].resource_type, ", ".join(nodes[uid].tags), paths[uid])
        out_console.print(table)


@app.command()
@guard
def show(
    name: Annotated[str, typer.Argument(help="Resource name or unique_id.")],
    project_dir: ProjectDirOpt = Path("."),
    profiles_dir: ProfilesDirOpt = None,
    target: TargetOpt = None,
    vars_: VarsOpt = None,
    output: OutputOpt = "yaml",
    log_format: LogFormatOpt = "text",
    quiet: QuietOpt = False,
    verbose: VerboseOpt = False,
) -> None:
    """Print one resource's compile-rendered config (FR-PARSE-05)."""
    from mbt.compile.compiler import compile_project
    from mbt.exceptions import ConfigError
    from mbt.parsing import parse_project
    from mbt.utils import did_you_mean

    cli = make_ctx(project_dir, profiles_dir, target, vars_, log_format, quiet, verbose)
    parsed = parse_project(cli.project_dir, cli_vars=cli.cli_vars)
    profiles = cli.profiles(parsed)
    manifest = compile_project(parsed, profiles, cli_vars=cli.cli_vars)

    found: dict[str, Any] | None = None
    pools: list[dict[str, Any]] = [manifest.nodes, manifest.sources, manifest.exposures]
    for pool in pools:
        for uid, resource in pool.items():
            if name in (uid, resource.name):
                found = resource.model_dump(mode="json")
                break
        if found:
            break
    if found is None:
        suggestion = did_you_mean(name, parsed.all_names())
        raise ConfigError(
            f"unknown resource {name!r}",
            hint=f"did you mean {suggestion!r}?" if suggestion else "run 'mbt ls'",
        )
    # Redact tainted secrets: a spec field may render an env_var() value
    # (jinja resolve context taints it), so this echoes rendered config the
    # same way the manifest file does - through redact (NFR-07).
    from mbt.secrets import redact

    if output == "json":
        typer.echo(redact(json.dumps(found, indent=2)))
    else:
        typer.echo(redact(yaml.safe_dump(found, sort_keys=False, default_flow_style=False)))


@state_app.command("diff")
@guard
def state_diff(
    state: Annotated[str, typer.Option("--state", help="Reference manifest path/URI (required).")],
    project_dir: ProjectDirOpt = Path("."),
    profiles_dir: ProfilesDirOpt = None,
    target: TargetOpt = None,
    vars_: VarsOpt = None,
    manifest: ManifestOpt = None,
    output: OutputOpt = "table",
    anchor: AnchorOpt = None,
    deep_snapshot: DeepSnapshotOpt = False,
    log_format: LogFormatOpt = "text",
    quiet: QuietOpt = False,
    verbose: VerboseOpt = False,
) -> None:
    """What changed vs a previous manifest, with components (FR-STATE-02)."""
    from mbt.artifacts.manifest import read_manifest
    from mbt.compile.compiler import CompileOptions, compile_project
    from mbt.events import get_bus
    from mbt.events.models import StateDiffed
    from mbt.parsing import parse_project
    from mbt.state.diff import diff_manifests, load_state

    cli = make_ctx(project_dir, profiles_dir, target, vars_, log_format, quiet, verbose)
    manifest = cli.resolve_cli_path(manifest)
    state = cli.resolve_cli_path(state) or state
    if manifest is not None:
        current = read_manifest(Path(manifest), source="--manifest")
    else:
        parsed = parse_project(cli.project_dir, cli_vars=cli.cli_vars)
        profiles = cli.profiles(parsed)
        current = compile_project(
            parsed,
            profiles,
            options=CompileOptions(anchor=parse_anchor(anchor), deep_snapshot=deep_snapshot),
            cli_vars=cli.cli_vars,
        )
    reference = load_state(state)
    diff = diff_manifests(current, reference)
    # Surface the diff on the event stream (independent of the --output data
    # format) so a machine watching --log-format json sees what changed.
    get_bus().emit(
        StateDiffed(
            added=len(diff.added),
            removed=len(diff.removed),
            modified=len(diff.modified),
            env_changed=diff.env_changed,
        )
    )

    if output == "json":
        typer.echo(json.dumps(diff.to_dict(), indent=2))
        return
    table = Table(title="mbt state diff")
    table.add_column("change")
    table.add_column("unique_id")
    table.add_column("components")
    for entry in (*diff.added, *diff.removed, *diff.modified):
        table.add_row(entry.change, entry.unique_id, ", ".join(entry.components))
    out_console.print(table)
    if diff.env_changed:
        out_console.print(
            "[yellow]environment digest CHANGED[/yellow] - nodes are not marked "
            "modified by this alone (ADR-7)"
        )
    if diff.is_empty:
        out_console.print("no node changes")


# -- docs -----------------------------------------------------------------------------------


@docs_app.command("generate")
@guard
def docs_generate(
    project_dir: ProjectDirOpt = Path("."),
    profiles_dir: ProfilesDirOpt = None,
    target: TargetOpt = None,
    vars_: VarsOpt = None,
    manifest: ManifestOpt = None,
    log_format: LogFormatOpt = "text",
    quiet: QuietOpt = False,
    verbose: VerboseOpt = False,
) -> None:
    """Render model cards + lineage into target/docs (FR-DOCS-01)."""
    from mbt.artifacts.manifest import read_manifest
    from mbt.artifacts.run_results import RunResults
    from mbt.compile.compiler import compile_project
    from mbt.docsgen import generate_docs
    from mbt.parsing import parse_project

    cli = make_ctx(project_dir, profiles_dir, target, vars_, log_format, quiet, verbose)
    manifest = cli.resolve_cli_path(manifest)
    if manifest is not None:
        current = read_manifest(Path(manifest), source="--manifest")
    else:
        parsed = parse_project(cli.project_dir, cli_vars=cli.cli_vars)
        current = compile_project(parsed, cli.profiles(parsed), cli_vars=cli.cli_vars)
    run_results = None
    results_path = cli.project_dir / "target" / "run_results.json"
    if results_path.is_file():
        run_results = RunResults.model_validate_json(results_path.read_text())
    index = generate_docs(current, run_results, cli.project_dir / "target" / "docs")
    out_console.print(f"wrote {index}")


@docs_app.command("serve")
@guard
def docs_serve(
    project_dir: ProjectDirOpt = Path("."),
    port: Annotated[int, typer.Option("--port", "-p")] = 8080,
) -> None:
    """Serve target/docs locally (FR-DOCS-02)."""
    import functools as ft
    import http.server

    from mbt.exceptions import ConfigError

    docs_dir = project_dir / "target" / "docs"
    if not (docs_dir / "index.html").is_file():
        raise ConfigError(f"no generated docs at {docs_dir}", hint="run 'mbt docs generate' first")
    handler = ft.partial(http.server.SimpleHTTPRequestHandler, directory=str(docs_dir))
    out_console.print(f"serving {docs_dir} at http://127.0.0.1:{port} (Ctrl+C to stop)")
    http.server.ThreadingHTTPServer(("127.0.0.1", port), handler).serve_forever()


# -- escape hatch ------------------------------------------------------------------------------


@app.command("run-operation")
@guard
def run_operation(
    macro: Annotated[str, typer.Argument(help="Macro name from macros/*.jinja.")],
    project_dir: ProjectDirOpt = Path("."),
    profiles_dir: ProfilesDirOpt = None,
    target: TargetOpt = None,
    vars_: VarsOpt = None,
    args: Annotated[
        str | None, typer.Option("--args", help="YAML/JSON dict of macro arguments.")
    ] = None,
    log_format: LogFormatOpt = "text",
    quiet: QuietOpt = False,
    verbose: VerboseOpt = False,
) -> None:
    """Render a macro with the full compile context (FR-RUN-08)."""
    from mbt.compile.compiler import build_resolve_context
    from mbt.exceptions import ConfigError
    from mbt.parsing import parse_project

    cli = make_ctx(project_dir, profiles_dir, target, vars_, log_format, quiet, verbose)
    parsed = parse_project(cli.project_dir, cli_vars=cli.cli_vars)
    profiles = cli.profiles(parsed)
    if macro not in parsed.renderer.macro_names:
        raise ConfigError(
            f"unknown macro {macro!r}",
            hint=f"available: {', '.join(parsed.renderer.macro_names) or '(none)'}. "
            "Adapter-invoking operations are out of scope in v0 (TSD §10.7).",
        )
    macro_args = parse_vars(args)
    resolve_ctx = build_resolve_context(parsed, profiles, cli.cli_vars)
    arg_list = ", ".join(f"{k}={json.dumps(v)}" for k, v in macro_args.items())
    rendered = parsed.renderer.resolve(
        {"result": f"{{{{ {macro}({arg_list}) }}}}"},
        resolve_ctx,
        resource=f"run-operation:{macro}",
        path=cli.project_dir,
    )
    typer.echo(str(rendered["result"]))


def main() -> None:
    """Entry point with mbt exit-code semantics (TSD §17).

    Click's usage errors default to exit code 2, which collides with mbt's
    "quality failure" code; remap them to 1 (hard error).
    """
    try:
        result = app(standalone_mode=False)
        # click returns the code from ctx.exit()/typer.Exit in this mode.
        sys.exit(result if isinstance(result, int) else 0)
    except EXIT_EXCEPTIONS as exc:
        sys.exit(getattr(exc, "exit_code", 0))
    except USAGE_ERROR_EXCEPTIONS as exc:
        exc.show(file=sys.stderr)  # type: ignore[attr-defined]
        sys.exit(1)
    except CLICK_EXCEPTIONS as exc:
        exc.show(file=sys.stderr)  # type: ignore[attr-defined]
        sys.exit(1)
    except ABORT_EXCEPTIONS:
        err_console.print("aborted")
        sys.exit(1)
    except Exception as exc:
        # Coordinator-side safety net: the job subprocess already wraps any
        # non-MbtError crash into a structured error row (execute/job.py), but
        # the coordinator half had no equivalent, so a stray assert/ValueError
        # in parse/compile surfaced as a raw traceback. Redact (the error path
        # is a serialization path too) and point at a bug report; MBT_DEBUG=1
        # re-raises so that report can capture the full traceback.
        if os.environ.get("MBT_DEBUG"):
            raise
        from mbt.secrets import redact

        err_console.print(
            f"[bold red]Internal error:[/bold red] {redact(f'{type(exc).__name__}: {exc}')}"
        )
        err_console.print(
            "  [yellow]hint:[/yellow] this is a bug in mbt; please report it with the "
            "command you ran. Set MBT_DEBUG=1 to see the full traceback."
        )
        sys.exit(1)


if __name__ == "__main__":
    main()
