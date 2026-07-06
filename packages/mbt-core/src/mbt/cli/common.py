"""Shared CLI plumbing: global flags, event sinks, error handling (TSD §3, §17)."""

from dataclasses import dataclass, field
from datetime import UTC, datetime
from pathlib import Path
from typing import Any

import typer
import yaml
from rich.console import Console
from rich.table import Table

from mbt.artifacts.run_results import RunResults
from mbt.config.profiles import LoadedProfiles, load_profiles
from mbt.events import ConsoleSink, EventBus, JsonLinesSink, NullSink, set_bus
from mbt.events.models import LogMessage
from mbt.exceptions import ConfigError, MbtError
from mbt.execute.orchestrator import InvocationOptions
from mbt.parsing import ParsedProject

err_console = Console(stderr=True, highlight=False)
out_console = Console(highlight=False)


@dataclass
class CLIContext:
    """Global flag state shared by all commands (FR-CLI-04)."""

    project_dir: Path = Path(".")
    profiles_dir: Path | None = None
    target: str | None = None
    cli_vars: dict[str, Any] = field(default_factory=dict)
    log_format: str = "text"
    quiet: bool = False

    def profiles(self, parsed: ParsedProject) -> LoadedProfiles:
        return load_profiles(
            parsed.project.name,
            self.project_dir,
            profiles_dir=self.profiles_dir,
            target_override=self.target,
            cli_vars=self.cli_vars,
            project_vars=parsed.project.vars,
        )

    def invocation(self, command: str, **kwargs: Any) -> InvocationOptions:
        return InvocationOptions(
            command=command,
            project_dir=self.project_dir,
            profiles_dir=self.profiles_dir,
            target=self.target,
            cli_vars=self.cli_vars,
            **kwargs,
        )


def parse_vars(raw: str | None) -> dict[str, Any]:
    if not raw:
        return {}
    try:
        value = yaml.safe_load(raw)
    except yaml.YAMLError as exc:
        raise ConfigError(f"--vars is not valid YAML/JSON: {exc}") from exc
    if not isinstance(value, dict):
        raise ConfigError(
            f"--vars must be a mapping, got {type(value).__name__}",
            hint="e.g. --vars 'sample_fraction: 0.1'",
        )
    return value


def parse_anchor(raw: str | None) -> datetime | None:
    if raw is None:
        return None
    try:
        anchor = datetime.fromisoformat(raw.replace("Z", "+00:00"))
    except ValueError as exc:
        raise ConfigError(
            f"--anchor is not an ISO timestamp: {raw!r}",
            hint="e.g. --anchor 2026-07-06T12:00:00Z",
        ) from exc
    return anchor if anchor.tzinfo else anchor.replace(tzinfo=UTC)


def setup_bus(ctx: CLIContext) -> None:
    """Events go to stderr; stdout is reserved for command data output."""
    import sys

    if ctx.quiet:
        sinks: list[Any] = [NullSink()]
    elif ctx.log_format == "json":
        sinks = [JsonLinesSink(stream=sys.stderr)]
    else:
        sinks = [ConsoleSink(console=err_console)]
    set_bus(EventBus(sinks=sinks))


def fail(exc: MbtError) -> "typer.Exit":
    err_console.print(f"[bold red]Error:[/bold red] {exc.message}")
    if exc.resource:
        err_console.print(f"  resource: {exc.resource}")
    if exc.path:
        err_console.print(f"  file: {exc.path}")
    if exc.hint:
        err_console.print(f"  [yellow]hint:[/yellow] {exc.hint}")
    return typer.Exit(exc.exit_code)


def print_warnings(parsed: ParsedProject) -> None:
    from mbt.events import get_bus

    for issue in parsed.report.warnings:
        get_bus().emit(LogMessage(level="warn", message=issue.format()))


def render_results_table(results: RunResults, ctx: CLIContext) -> None:
    if ctx.quiet or ctx.log_format == "json" or not results.results:
        return
    table = Table(title=f"mbt {results.metadata.command} results", show_lines=False)
    table.add_column("node")
    table.add_column("status")
    table.add_column("time", justify="right")
    table.add_column("detail")
    styles = {
        "success": "green",
        "error": "red",
        "gate_failed": "red",
        "test_failed": "red",
        "skipped": "yellow",
    }
    for result in results.results:
        detail = ""
        if result.metrics:
            top = sorted(result.metrics.items())[:3]
            detail = "  ".join(f"{k}={v:.4f}" for k, v in top)
        if result.registration:
            detail += f"  -> {result.registration.name} v{result.registration.version}"
        if result.message and result.status != "success":
            detail = result.message[:100]
        table.add_row(
            result.unique_id,
            f"[{styles.get(result.status, '')}]{result.status}[/]",
            f"{result.execution_time_s:.2f}s",
            detail,
        )
    out_console.print(table)
