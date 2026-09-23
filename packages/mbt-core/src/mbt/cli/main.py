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
)
from mbt.exceptions import MbtError


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
    typer.Option("--manifest", help="Execute a stored manifest verbatim (ADR-19)."),
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
    name: Annotated[
        str,
        typer.Argument(
            help="Project name: letters, digits, underscores; must start with a letter."
        ),
    ],
    project_dir: ProjectDirOpt = Path("."),
    log_format: LogFormatOpt = "text",
    quiet: QuietOpt = False,
    verbose: VerboseOpt = False,
) -> None:
    """Scaffold a golden-path project: specs, profiles, CI workflows, sample data."""
    from mbt.cli.scaffold import scaffold_project

    # chdir=False: project_dir is the parent to scaffold into, not a project
    cli = CLIContext.enter(
        project_dir, log_format=log_format, quiet=quiet, verbose=verbose, chdir=False
    )
    destination = scaffold_project(name, cli.project_dir)
    # soft_wrap: a path is the thing a user copies out of this line, and a
    # hard-inserted newline would split it (FEEDBACK v3 E-5, as in ConsoleSink)
    out_console.print(f"Created [bold]{destination}[/bold]", soft_wrap=True)
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
    """Install the adapter packages pinned in packages.yml."""
    from mbt.deps import install_packages, load_packages

    cli = CLIContext.enter(project_dir, log_format=log_format, quiet=quiet, verbose=verbose)
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

    # BOTH branches go through the composition root (A-2). The default branch
    # used to call shutil.rmtree on the raw, unresolved typer path with no
    # context and no event bus, so `mbt clean --project-dir ../other` deleted
    # relative to the invocation cwd, and a --project-dir that is not a
    # directory printed "nothing to clean" instead of the ConfigError every
    # other command raises. All four clean tests passed an absolute
    # --project-dir, so the divergence was untested.
    cli = CLIContext.enter(project_dir, profiles_dir, target, vars_, "text", False)

    if artifacts_older_than is None:
        from mbt.cli.inspect import clean_target_cmd

        for note in clean_target_cmd(cli):
            out_console.print(note)
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
    """Validate all configs and build the DAG; nothing executes."""
    from mbt.config.project import load_project
    from mbt.events import get_bus
    from mbt.events.models import ParseCompleted, ParseStarted
    from mbt.exceptions import ConfigError
    from mbt.parsing import parse_project

    cli = CLIContext.enter(project_dir, profiles_dir, target, vars_, log_format, quiet, verbose)
    bus = get_bus()
    # The banner names the PROJECT, read from mbt_project.yml, not the directory
    # it happens to sit in. profiles.yml is keyed by the project name, so when
    # the two differ - any CI checkout directory, any --project-dir at a
    # differently named path, any clone into a renamed folder - the banner used
    # to print one name while the very next error said "profiles.yml has no
    # entry for project 'other'", in exactly the moment someone is reading the
    # banner to find out which name to use.
    try:
        project_name = load_project(cli.project_dir).name
    except ConfigError:
        # An unreadable mbt_project.yml is parse_project's error to raise, one
        # line below and with the full diagnostic; the banner just steps aside.
        project_name = cli.project_dir.name
    bus.emit(ParseStarted(project=project_name))
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
    """Resolve Jinja + profiles + snapshots into target/manifest.json."""
    cli = CLIContext.enter(project_dir, profiles_dir, target, vars_, log_format, quiet, verbose)
    path = cli.project_dir / "target" / "manifest.json"
    cli.compile(anchor=anchor, deep_snapshot=deep_snapshot, write_to=path)
    out_console.print(f"wrote {path}", soft_wrap=True)


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

        cli = CLIContext.enter(project_dir, profiles_dir, target, vars_, log_format, quiet, verbose)
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


_register_execution_command("run", "Build datasets and train models in DAG order.")
_register_execution_command("build", "run + test interleaved in DAG order - the CI workhorse.")
_register_execution_command("test", "Data tests + model quality gates; never trains.")
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
        str | None,
        typer.Option(
            "--version",
            help="Registry version (default: the version in --stage, else in the model's "
            "stage_on_pass).",
        ),
    ] = None,
    stage: Annotated[
        str | None, typer.Option("--stage", help="Stage to pull the version from.")
    ] = None,
    gates: Annotated[
        bool, typer.Option("--gates", help="Apply gate logic to the fresh metrics.")
    ] = False,
    out_of_time: Annotated[
        bool,
        typer.Option(
            "--out-of-time",
            help="Pre-deploy check: the version's recorded test window against everything "
            "since, reported on its training run; with --gates, judged and recorded on "
            "the version.",
        ),
    ] = False,
    manifest: ManifestOpt = None,
    allow_env_mismatch: AllowEnvMismatchOpt = False,
    anchor: AnchorOpt = None,
    log_format: LogFormatOpt = "text",
    quiet: QuietOpt = False,
    verbose: VerboseOpt = False,
) -> None:
    """Re-evaluate a registered artifact on freshly built data; never trains."""
    from mbt.execute.orchestrator import run_evaluate

    cli = CLIContext.enter(project_dir, profiles_dir, target, vars_, log_format, quiet, verbose)
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
        out_of_time=out_of_time,
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

    cli = CLIContext.enter(project_dir, profiles_dir, target, vars_, log_format, quiet, verbose)
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

    cli = CLIContext.enter(project_dir, profiles_dir, target, vars_, log_format, quiet, verbose)
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

    cli = CLIContext.enter(project_dir, profiles_dir, target, vars_, log_format, quiet, verbose)
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
    model: Annotated[
        str | None, typer.Option("--model", help="Registered model name to promote.")
    ] = None,
    to: Annotated[str | None, typer.Option("--to", help="Target stage.")] = None,
    version: Annotated[
        str | None,
        typer.Option("--version", help="Registry version (default: the current staging version)."),
    ] = None,
    from_file: Annotated[
        Path | None, typer.Option("--from-file", help="Reviewed promotions.yml (GitOps).")
    ] = None,
    force: Annotated[
        bool, typer.Option("--force", help="Promote even without recorded gate passes.")
    ] = False,
    require_oot_check: Annotated[
        bool,
        typer.Option(
            "--require-oot-check",
            help="Refuse unless the version's latest after-test check passed "
            "(mbt evaluate --out-of-time --gates, or the build's own).",
        ),
    ] = False,
    log_format: LogFormatOpt = "text",
    quiet: QuietOpt = False,
    verbose: VerboseOpt = False,
) -> None:
    """Transition a registered version, verifying recorded gate passes."""
    from mbt.adapters.registry import get_registry
    from mbt.exceptions import ConfigError
    from mbt.parsing import parse_project
    from mbt.promote import load_promotions_file, promote_model
    from mbt.runtime import registry_adapter as build_registry_adapter
    from mbt_adapter_base import (
        Stage,
    )

    cli = CLIContext.enter(project_dir, profiles_dir, target, vars_, log_format, quiet, verbose)
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
                require_oot_check=require_oot_check or entry.require_oot_check,
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
        registry_adapter,
        name=model,
        to_stage=stage_token,
        version=version,
        force=force,
        require_oot_check=require_oot_check,
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
    cli = CLIContext.enter(project_dir, profiles_dir, target, vars_, log_format, quiet, verbose)
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
    state: StateOpt = None,
    state_include_env: StateIncludeEnvOpt = False,
    output: OutputOpt = "table",
    anchor: AnchorOpt = None,
    log_format: LogFormatOpt = "text",
    quiet: QuietOpt = False,
    verbose: VerboseOpt = False,
) -> None:
    """List resources, with the same selectors --select accepts.

    "The same selectors" was not true until v5: this hand-rolled its own node
    view off the PARSED project and never passed ``state`` to the selector, so
    ``mbt ls --select state:modified`` failed with "requires --state", and
    ``mbt ls --state foo.json`` failed with "No such option" - while its own
    docstring and ``docs/cli-reference.md`` both promised the full grammar
    (v5 live defect 3).
    """
    from mbt.cli.inspect import ls_cmd

    cli = CLIContext.enter(project_dir, profiles_dir, target, vars_, log_format, quiet, verbose)
    listed = ls_cmd(
        cli,
        select=select,
        exclude=exclude,
        state=state,
        state_include_env=state_include_env,
        anchor=anchor,
    )
    if output == "name":
        for row in listed:
            typer.echo(row.name)
    elif output == "path":
        for row in listed:
            typer.echo(row.path)
    elif output == "json":
        typer.echo(
            json.dumps(
                [
                    {
                        "unique_id": row.unique_id,
                        "name": row.name,
                        "resource_type": row.resource_type,
                        "tags": list(row.tags),
                        "path": row.path,
                    }
                    for row in listed
                ],
                indent=2,
            )
        )
    else:
        table = Table()
        table.add_column("unique_id")
        table.add_column("type")
        table.add_column("tags")
        table.add_column("path")
        for row in listed:
            table.add_row(row.unique_id, row.resource_type, ", ".join(row.tags), row.path)
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
    anchor: AnchorOpt = None,
    log_format: LogFormatOpt = "text",
    quiet: QuietOpt = False,
    verbose: VerboseOpt = False,
) -> None:
    """Print one resource's compile-rendered config.

    Takes ``--anchor`` for the same reason every compiling command does: without
    it this re-anchored to ``now()`` on each invocation, so the resolved windows
    it printed drifted from ``target/manifest.json`` with wall-clock time - the
    exact drift ADR-12 exists to pin (A-2).
    """
    from mbt.cli.inspect import show_cmd

    cli = CLIContext.enter(project_dir, profiles_dir, target, vars_, log_format, quiet, verbose)
    typer.echo(show_cmd(cli, name, output=output, anchor=anchor))


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
    """What changed vs a previous manifest, and which component changed."""
    from mbt.artifacts.manifest import read_manifest
    from mbt.events import get_bus
    from mbt.events.models import StateDiffed
    from mbt.state.diff import diff_manifests, load_state

    cli = CLIContext.enter(project_dir, profiles_dir, target, vars_, log_format, quiet, verbose)
    manifest = cli.resolve_cli_path(manifest)
    state = cli.resolve_cli_path(state) or state
    if manifest is not None:
        current = read_manifest(Path(manifest), source="--manifest")
    else:
        _, current = cli.compile(anchor=anchor, deep_snapshot=deep_snapshot)
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
    anchor: AnchorOpt = None,
    log_format: LogFormatOpt = "text",
    quiet: QuietOpt = False,
    verbose: VerboseOpt = False,
) -> None:
    """Render model cards + lineage into target/docs.

    Takes ``--anchor`` like every other compiling command (A-2); without it the
    compile re-anchored to ``now()``, so a card's resolved windows disagreed
    with the manifest the run actually used.
    """
    from mbt.artifacts.manifest import read_manifest
    from mbt.artifacts.run_results import read_latest_results
    from mbt.docsgen import generate_docs

    cli = CLIContext.enter(project_dir, profiles_dir, target, vars_, log_format, quiet, verbose)
    manifest = cli.resolve_cli_path(manifest)
    if manifest is not None:
        current = read_manifest(Path(manifest), source="--manifest")
    else:
        _, current = cli.compile(anchor=anchor)
    # Model cards want the metrics a TRAINING command produced, which is not
    # necessarily the last command that ran: `mbt score`/`mbt monitor` rewrite
    # the shared run_results.json with only their own nodes (A-2).
    run_results = read_latest_results(
        cli.project_dir / "target" / "run_results.json",
        commands=("build", "run", "evaluate"),
    )
    index = generate_docs(current, run_results, cli.project_dir / "target" / "docs")
    out_console.print(f"wrote {index}", soft_wrap=True)


@docs_app.command("serve")
@guard
def docs_serve(
    project_dir: ProjectDirOpt = Path("."),
    port: Annotated[int, typer.Option("--port", "-p", help="Local port to listen on.")] = 8080,
) -> None:
    """Serve target/docs over HTTP on localhost."""
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
    """Render a macro with the full compile context."""
    from mbt.compile.compiler import build_resolve_context
    from mbt.exceptions import ConfigError
    from mbt.parsing import parse_project

    cli = CLIContext.enter(project_dir, profiles_dir, target, vars_, log_format, quiet, verbose)
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
