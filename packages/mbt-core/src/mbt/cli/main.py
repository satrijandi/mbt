"""The ``mbt`` Typer application: all commands (TSD §3).

Every command is non-interactive-safe (FR-CLI-01); exit codes follow TSD §17
(0 success, 1 hard error, 2 quality failure);
``--target/--vars/--select/--exclude/--threads/--state/--manifest`` behave
identically wherever they appear (FR-CLI-04).
"""

import functools
import json
from collections.abc import Callable
from pathlib import Path
from typing import Annotated, Any

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
from mbt.exceptions import MbtError

app = typer.Typer(
    name="mbt",
    help="mbt: a declarative build tool for machine learning models.",
    no_args_is_help=True,
    add_completion=True,
    pretty_exceptions_enable=False,
)
docs_app = typer.Typer(help="Generate or serve the model cards + lineage site.")
state_app = typer.Typer(help="Compare manifests (state:modified mechanics).")
app.add_typer(docs_app, name="docs")
app.add_typer(state_app, name="state")

# -- global flags -----------------------------------------------------------------


@app.callback()
def _global_flags(
    ctx: typer.Context,
    project_dir: Annotated[
        Path, typer.Option("--project-dir", help="Project root (default: cwd).")
    ] = Path("."),
    profiles_dir: Annotated[
        Path | None, typer.Option("--profiles-dir", help="Directory holding profiles.yml.")
    ] = None,
    target: Annotated[
        str | None, typer.Option("--target", "-t", help="Profile target (dev/prod/...).")
    ] = None,
    vars_: Annotated[
        str | None, typer.Option("--vars", help="YAML/JSON dict overriding vars.")
    ] = None,
    log_format: Annotated[
        str, typer.Option("--log-format", help="text | json (one event per line).")
    ] = "text",
    quiet: Annotated[bool, typer.Option("--quiet", "-q", help="Suppress event output.")] = False,
) -> None:
    try:
        cli = CLIContext(
            project_dir=project_dir,
            profiles_dir=profiles_dir,
            target=target,
            cli_vars=parse_vars(vars_),
            log_format=log_format,
            quiet=quiet,
        )
    except MbtError as exc:
        raise fail(exc) from exc
    setup_bus(cli)
    ctx.obj = cli


def guard(fn: Callable[..., Any]) -> Callable[..., Any]:
    """Uniform MbtError -> message + exit code handling (TSD §17)."""

    @functools.wraps(fn)
    def wrapper(*args: Any, **kwargs: Any) -> Any:
        try:
            return fn(*args, **kwargs)
        except MbtError as exc:
            raise fail(exc) from exc

    return wrapper


# -- shared option aliases ----------------------------------------------------------

SelectOpt = Annotated[
    list[str] | None, typer.Option("--select", "-s", help="Node selector(s); space = union.")
]
ExcludeOpt = Annotated[
    list[str] | None, typer.Option("--exclude", help="Selector(s) to subtract.")
]
ThreadsOpt = Annotated[
    int | None, typer.Option("--threads", help="Parallel DAG branches (default: target).")
]
StateOpt = Annotated[
    str | None,
    typer.Option("--state", help="Reference manifest path/URI for state: selectors."),
]
ManifestOpt = Annotated[
    str | None,
    typer.Option("--manifest", help="Execute a stored manifest verbatim (FR-RUN-11)."),
]
AnchorOpt = Annotated[
    str | None, typer.Option("--anchor", help="Pin the time anchor (ISO timestamp).")
]


# -- project lifecycle ----------------------------------------------------------------


@app.command()
@guard
def init(
    ctx: typer.Context,
    name: Annotated[str, typer.Argument(help="Project name (lowercase snake_case).")],
) -> None:
    """Scaffold a golden-path project (FR-PROJ-01)."""
    from mbt.cli.scaffold import scaffold_project

    cli: CLIContext = ctx.obj
    destination = scaffold_project(name, cli.project_dir)
    out_console.print(f"Created [bold]{destination}[/bold]")
    out_console.print(
        "Next steps:\n"
        f"  cd {name}\n"
        "  python scripts/generate_sample_data.py\n"
        "  mbt build"
    )


@app.command()
@guard
def deps(
    ctx: typer.Context,
    dry_run: Annotated[bool, typer.Option("--dry-run", help="Print, do not install.")] = False,
) -> None:
    """Install adapter packages pinned in packages.yml (FR-PROJ-04)."""
    from mbt.deps import install_packages, load_packages

    cli: CLIContext = ctx.obj
    requirements = install_packages(load_packages(cli.project_dir), dry_run=dry_run)
    verb = "would install" if dry_run else "installed"
    out_console.print(f"{verb}: " + (", ".join(requirements) or "(nothing)"))


@app.command()
@guard
def clean(ctx: typer.Context) -> None:
    """Delete target/ including the dataset cache (FR-RUN-09)."""
    import shutil

    cli: CLIContext = ctx.obj
    target = cli.project_dir / "target"
    if target.is_dir():
        shutil.rmtree(target)
        out_console.print(f"removed {target}")
    else:
        out_console.print(f"nothing to clean at {target}")


# -- parse / compile -------------------------------------------------------------------


@app.command()
@guard
def parse(
    ctx: typer.Context,
    write_json_schema: Annotated[
        bool,
        typer.Option("--write-json-schema", help="Publish JSON Schemas for editors."),
    ] = False,
) -> None:
    """Validate all configs and build the DAG; no execution (FR-PARSE-01)."""
    from mbt.parsing import parse_project

    cli: CLIContext = ctx.obj
    parsed = parse_project(cli.project_dir, cli_vars=cli.cli_vars)
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
def compile(  # noqa: A001 - dbt-style command name
    ctx: typer.Context,
    anchor: AnchorOpt = None,
    deep_snapshot: Annotated[
        bool,
        typer.Option("--deep-snapshot", help="Content-hash snapshots (slow, exact)."),
    ] = False,
) -> None:
    """Resolve Jinja + profiles + snapshots into target/manifest.json (FR-COMP-01)."""
    from mbt.compile.compiler import CompileOptions, compile_project
    from mbt.parsing import parse_project

    cli: CLIContext = ctx.obj
    parsed = parse_project(cli.project_dir, cli_vars=cli.cli_vars)
    print_warnings(parsed)
    profiles = cli.profiles(parsed)
    manifest = compile_project(
        parsed,
        profiles,
        options=CompileOptions(anchor=parse_anchor(anchor), deep_snapshot=deep_snapshot),
        cli_vars=cli.cli_vars,
    )
    path = cli.project_dir / "target" / "manifest.json"
    manifest.write(path)
    out_console.print(f"wrote {path}")


# -- run / build / test ------------------------------------------------------------------


def _execute_command(
    ctx: typer.Context,
    command: str,
    select: list[str] | None,
    exclude: list[str] | None,
    threads: int | None,
    fail_fast: bool,
    state: str | None,
    state_include_env: bool,
    manifest: str | None,
    anchor: str | None,
    deep_snapshot: bool,
) -> None:
    from mbt.execute.orchestrator import run_command

    cli: CLIContext = ctx.obj
    results = run_command(
        cli.invocation(
            command,
            select=select,
            exclude=exclude,
            threads=threads,
            fail_fast=fail_fast,
            state=state,
            state_include_env=state_include_env,
            manifest_path=manifest,
            anchor=parse_anchor(anchor),
            deep_snapshot=deep_snapshot,
        )
    )
    render_results_table(results, cli)
    code = results.exit_code()
    if code:
        raise typer.Exit(code)


def _register_execution_command(command: str, help_text: str) -> None:
    @app.command(name=command, help=help_text)
    @guard
    def _cmd(
        ctx: typer.Context,
        select: SelectOpt = None,
        exclude: ExcludeOpt = None,
        threads: ThreadsOpt = None,
        fail_fast: Annotated[
            bool, typer.Option("--fail-fast", help="Stop everything on first failure.")
        ] = False,
        state: StateOpt = None,
        state_include_env: Annotated[
            bool,
            typer.Option(
                "--state-include-env",
                help="Treat env_digest changes as modifying every node (ADR-7).",
            ),
        ] = False,
        manifest: ManifestOpt = None,
        anchor: AnchorOpt = None,
        deep_snapshot: Annotated[bool, typer.Option("--deep-snapshot")] = False,
    ) -> None:
        _execute_command(
            ctx, command, select, exclude, threads, fail_fast, state,
            state_include_env, manifest, anchor, deep_snapshot,
        )


_register_execution_command(
    "run", "Build datasets and train models in DAG order (FR-RUN-01)."
)
_register_execution_command(
    "build", "run + test interleaved in DAG order - the CI workhorse (FR-RUN-01)."
)
_register_execution_command(
    "test", "Data tests + model quality gates; never trains (FR-TEST-01)."
)


@app.command()
@guard
def evaluate(
    ctx: typer.Context,
    model: Annotated[str, typer.Option("--model", help="Model resource name.")],
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
    anchor: AnchorOpt = None,
) -> None:
    """Re-evaluate a registered artifact on freshly built data (FR-RUN-07)."""
    from mbt.execute.orchestrator import run_evaluate

    cli: CLIContext = ctx.obj
    results = run_evaluate(
        cli.invocation("evaluate", manifest_path=manifest, anchor=parse_anchor(anchor)),
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
def promote(
    ctx: typer.Context,
    model: Annotated[str | None, typer.Option("--model")] = None,
    to: Annotated[str | None, typer.Option("--to", help="Target stage.")] = None,
    version: Annotated[str | None, typer.Option("--version")] = None,
    from_file: Annotated[
        Path | None, typer.Option("--from-file", help="Reviewed promotions.yml (GitOps).")
    ] = None,
    force: Annotated[
        bool, typer.Option("--force", help="Promote even without recorded gate passes.")
    ] = False,
) -> None:
    """Transition a registered version, verifying recorded gate passes (FR-REG-03)."""
    from mbt.contracts import Stage
    from mbt.exceptions import ConfigError
    from mbt.parsing import parse_project
    from mbt.promote import load_promotions_file, promote_model
    from mbt.runtime import registry_adapter as build_registry_adapter

    cli: CLIContext = ctx.obj
    parsed = parse_project(cli.project_dir, cli_vars=cli.cli_vars)
    profiles = cli.profiles(parsed)
    from mbt.adapters.registry import get_registry

    registry_adapter = build_registry_adapter(profiles, cli.project_dir.resolve(), get_registry())

    if from_file is not None:
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
        stage = Stage(to)
    except ValueError as exc:
        raise ConfigError(
            f"unknown stage {to!r}",
            hint=f"stages: {', '.join(s.value for s in Stage)}",
        ) from exc
    outcome = promote_model(
        registry_adapter, name=model, to_stage=stage, version=version, force=force
    )
    out_console.print(
        f"promoted [bold]{outcome.name}[/bold] v{outcome.version} -> {outcome.to_stage.value}"
    )


# -- inspection ---------------------------------------------------------------------------


@app.command()
@guard
def ls(
    ctx: typer.Context,
    select: SelectOpt = None,
    exclude: ExcludeOpt = None,
    output: Annotated[
        str, typer.Option("--output", "-o", help="table | name | path | json")
    ] = "table",
) -> None:
    """List resources with selector support (FR-PARSE-05)."""
    from mbt.dag.selector import SelectableNode, select_nodes
    from mbt.parsing import parse_project

    cli: CLIContext = ctx.obj
    parsed = parse_project(cli.project_dir, cli_vars=cli.cli_vars)
    nodes: dict[str, SelectableNode] = {}
    paths: dict[str, str] = {}
    for pool in (parsed.datasets, parsed.models, parsed.exposures):
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
            out_console.print(nodes[uid].name)
    elif output == "path":
        for uid in selected:
            out_console.print(paths[uid])
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
            table.add_row(
                uid, nodes[uid].resource_type, ", ".join(nodes[uid].tags), paths[uid]
            )
        out_console.print(table)


@app.command()
@guard
def show(
    ctx: typer.Context,
    name: Annotated[str, typer.Argument(help="Resource name or unique_id.")],
    output: Annotated[str, typer.Option("--output", "-o", help="yaml | json")] = "yaml",
) -> None:
    """Print one resource's compile-rendered config (FR-PARSE-05)."""
    from mbt.compile.compiler import compile_project
    from mbt.parsing import parse_project
    from mbt.utils import did_you_mean

    cli: CLIContext = ctx.obj
    parsed = parse_project(cli.project_dir, cli_vars=cli.cli_vars)
    profiles = cli.profiles(parsed)
    manifest = compile_project(parsed, profiles, cli_vars=cli.cli_vars)

    found = None
    for uid, node in {**manifest.nodes}.items():
        if uid == name or node.name == name:
            found = node.model_dump(mode="json")
            break
    if found is None:
        for uid, source in manifest.sources.items():
            if uid == name or source.name == name:
                found = source.model_dump(mode="json")
                break
    if found is None:
        for uid, exposure in manifest.exposures.items():
            if uid == name or exposure.name == name:
                found = exposure.model_dump(mode="json")
                break
    if found is None:
        from mbt.exceptions import ConfigError

        suggestion = did_you_mean(name, parsed.all_names())
        raise ConfigError(
            f"unknown resource {name!r}",
            hint=f"did you mean {suggestion!r}?" if suggestion else "run 'mbt ls'",
        )
    if output == "json":
        typer.echo(json.dumps(found, indent=2))
    else:
        out_console.print(yaml.safe_dump(found, sort_keys=False, default_flow_style=False))


@state_app.command("diff")
@guard
def state_diff(
    ctx: typer.Context,
    state: Annotated[
        str, typer.Option("--state", help="Reference manifest path/URI (required).")
    ],
    manifest: ManifestOpt = None,
    output: Annotated[str, typer.Option("--output", "-o", help="table | json")] = "table",
    anchor: AnchorOpt = None,
) -> None:
    """What changed vs a previous manifest, with components (FR-STATE-02)."""
    from mbt.artifacts.manifest import read_manifest
    from mbt.compile.compiler import CompileOptions, compile_project
    from mbt.parsing import parse_project
    from mbt.state.diff import diff_manifests, load_state

    cli: CLIContext = ctx.obj
    if manifest is not None:
        current = read_manifest(Path(manifest), source="--manifest")
    else:
        parsed = parse_project(cli.project_dir, cli_vars=cli.cli_vars)
        profiles = cli.profiles(parsed)
        current = compile_project(
            parsed,
            profiles,
            options=CompileOptions(anchor=parse_anchor(anchor)),
            cli_vars=cli.cli_vars,
        )
    reference = load_state(state)
    diff = diff_manifests(current, reference)

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
            "[yellow]environment digest CHANGED[/yellow] "
            f"(current {diff.env_digest_current[:20]}..., "
            f"reference {diff.env_digest_reference[:20]}...) - nodes are not marked "
            "modified by this alone (ADR-7)"
        )
    if diff.is_empty:
        out_console.print("no node changes")


# -- docs -----------------------------------------------------------------------------------


@docs_app.command("generate")
@guard
def docs_generate(ctx: typer.Context, manifest: ManifestOpt = None) -> None:
    """Render model cards + lineage into target/docs (FR-DOCS-01)."""
    from mbt.artifacts.manifest import read_manifest
    from mbt.artifacts.run_results import RunResults
    from mbt.compile.compiler import compile_project
    from mbt.docsgen import generate_docs
    from mbt.parsing import parse_project

    cli: CLIContext = ctx.obj
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
    ctx: typer.Context,
    port: Annotated[int, typer.Option("--port", "-p")] = 8080,
) -> None:
    """Serve target/docs locally (FR-DOCS-02)."""
    import functools as ft
    import http.server

    from mbt.exceptions import ConfigError

    cli: CLIContext = ctx.obj
    docs_dir = cli.project_dir / "target" / "docs"
    if not (docs_dir / "index.html").is_file():
        raise ConfigError(
            f"no generated docs at {docs_dir}", hint="run 'mbt docs generate' first"
        )
    handler = ft.partial(http.server.SimpleHTTPRequestHandler, directory=str(docs_dir))
    out_console.print(f"serving {docs_dir} at http://127.0.0.1:{port} (Ctrl+C to stop)")
    http.server.ThreadingHTTPServer(("127.0.0.1", port), handler).serve_forever()


# -- escape hatch -----------------------------------------------------------------------------


@app.command("run-operation")
@guard
def run_operation(
    ctx: typer.Context,
    macro: Annotated[str, typer.Argument(help="Macro name from macros/*.jinja.")],
    args: Annotated[
        str | None, typer.Option("--args", help="YAML/JSON dict of macro arguments.")
    ] = None,
) -> None:
    """Render a macro with the full compile context (FR-RUN-08)."""
    from mbt.compile.compiler import _build_resolve_context  # noqa: PLC2701
    from mbt.exceptions import ConfigError
    from mbt.parsing import parse_project

    cli: CLIContext = ctx.obj
    parsed = parse_project(cli.project_dir, cli_vars=cli.cli_vars)
    profiles = cli.profiles(parsed)
    if macro not in parsed.renderer.macro_names:
        raise ConfigError(
            f"unknown macro {macro!r}",
            hint=f"available: {', '.join(parsed.renderer.macro_names) or '(none)'}. "
            "Adapter-invoking operations are out of scope in v0 (TSD §10.7).",
        )
    macro_args = parse_vars(args)
    resolve_ctx = _build_resolve_context(parsed, profiles, cli.cli_vars)
    arg_list = ", ".join(f"{k}={json.dumps(v)}" for k, v in macro_args.items())
    rendered = parsed.renderer.resolve(
        {"result": f"{{{{ {macro}({arg_list}) }}}}"},
        resolve_ctx,
        resource=f"run-operation:{macro}",
        path=cli.project_dir,
    )
    out_console.print(str(rendered["result"]))


def main() -> None:
    app()


if __name__ == "__main__":
    main()
