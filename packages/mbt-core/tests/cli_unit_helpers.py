"""Shared helpers for in-process CLI unit tests (unique module name).

The CLI coordinator chdirs into ``--project-dir`` (make_ctx) and swaps the
process-wide event bus (setup_bus); tests importing the autouse fixture below
restore both after every test so the repo-root session guard and unrelated
tests keep working.
"""

import os
from collections.abc import Iterator
from typing import Any

import pytest
from typer.testing import CliRunner, Result

from mbt.artifacts.run_results import NodeResult, RunResults, RunResultsMetadata
from mbt.events import EventBus, set_bus
from mbt.events.models import Event

#: ISO form of core_helpers.TEST_ANCHOR, for --anchor flags.
ANCHOR = "2026-07-01T00:00:00Z"

runner = CliRunner()


@pytest.fixture(autouse=True)
def cli_process_state() -> Iterator[None]:
    cwd = os.getcwd()
    yield
    os.chdir(cwd)
    set_bus(EventBus())


def invoke(args: list[str], **kwargs: Any) -> Result:
    """Run the mbt Typer app in-process."""
    from mbt.cli import main as cli_main

    return runner.invoke(cli_main.app, args, **kwargs)


def debug(result: Result) -> str:
    """Assertion context: output plus any captured exception."""
    return f"output={result.output!r} stderr={result.stderr!r} exc={result.exception!r}"


def make_results(command: str = "run", *nodes: NodeResult) -> RunResults:
    return RunResults(
        metadata=RunResultsMetadata(
            run_id="r1",
            mbt_version="0",
            target="dev",
            manifest_hash="sha256:0",
            anchor=ANCHOR,
            started_at="2026-07-01T00:00:00+00:00",
            command=command,
        ),
        results=list(nodes),
    )


class RecordingSink:
    """Collects emitted events for assertions."""

    def __init__(self) -> None:
        self.events: list[Event] = []

    def write(self, event: Event) -> None:
        self.events.append(event)


def install_recording_bus() -> RecordingSink:
    sink = RecordingSink()
    set_bus(EventBus(sinks=[sink]))
    return sink
