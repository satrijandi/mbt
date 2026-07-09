"""Typed event models (TSD §16).

Every significant occurrence is a Pydantic event carrying ``run_id``,
``unique_id`` (where applicable), and a timestamp. Sinks render them for
humans (Rich) or machines (JSON lines).
"""

from datetime import UTC, datetime
from typing import Literal

from pydantic import BaseModel, ConfigDict, Field


def _now() -> datetime:
    return datetime.now(tz=UTC)


class Event(BaseModel):
    """Base event: name, level, timestamps, correlation ids."""

    model_config = ConfigDict(extra="forbid")

    event: str = ""
    level: Literal["debug", "info", "warn", "error"] = "info"
    ts: datetime = Field(default_factory=_now)
    run_id: str | None = None
    unique_id: str | None = None

    def model_post_init(self, __context: object) -> None:
        if not self.event:
            self.event = type(self).__name__

    def human(self) -> str:
        """One-line human rendering; sinks may add color."""
        return self.event


class LogMessage(Event):
    """Free-form informational message."""

    message: str = ""

    def human(self) -> str:
        return self.message


class ParseStarted(Event):
    project: str = ""

    def human(self) -> str:
        return f"Parsing project '{self.project}'"


class ParseCompleted(Event):
    resources: int = 0
    errors: int = 0
    elapsed_s: float = 0.0

    def human(self) -> str:
        status = "OK" if self.errors == 0 else f"{self.errors} error(s)"
        return f"Parsed {self.resources} resources in {self.elapsed_s:.2f}s [{status}]"


class CompileStarted(Event):
    target: str = ""

    def human(self) -> str:
        return f"Compiling against target '{self.target}'"


class CompileCompleted(Event):
    nodes: int = 0
    anchor: str = ""
    manifest_path: str = ""
    elapsed_s: float = 0.0

    def human(self) -> str:
        base = f"Compiled {self.nodes} nodes in {self.elapsed_s:.2f}s (anchor {self.anchor})"
        return f"{base} -> {self.manifest_path}" if self.manifest_path else base


class RunStarted(Event):
    command: str = ""
    target: str = ""
    selected: int = 0

    def human(self) -> str:
        return f"{self.command}: {self.selected} node(s) selected on target '{self.target}'"


class NodeStarted(Event):
    resource_type: str = ""
    index: int = 0
    total: int = 0

    def human(self) -> str:
        return f"[{self.index}/{self.total}] START {self.resource_type} {self.unique_id}"


class NodeFinished(Event):
    resource_type: str = ""
    status: str = ""
    execution_time_s: float = 0.0
    index: int = 0
    total: int = 0
    message: str | None = None

    def human(self) -> str:
        line = (
            f"[{self.index}/{self.total}] {self.status.upper()} "
            f"{self.resource_type} {self.unique_id} in {self.execution_time_s:.2f}s"
        )
        if self.message:
            line += f" - {self.message}"
        return line


class CheckEvaluated(Event):
    check: str = ""
    passed: bool = True
    message: str = ""

    def human(self) -> str:
        status = "PASS" if self.passed else "FAIL"
        suffix = f" - {self.message}" if self.message else ""
        return f"check {self.check}: {status}{suffix}"


class TestEvaluated(Event):
    test: str = ""
    passed: bool = True
    message: str = ""

    def human(self) -> str:
        status = "PASS" if self.passed else "FAIL"
        suffix = f" - {self.message}" if self.message else ""
        return f"test {self.test}: {status}{suffix}"


class GateEvaluated(Event):
    metric: str = ""
    kind: str = ""  # threshold | champion
    passed: bool = True
    expected: float | None = None
    actual: float | None = None
    champion_version: str | None = None
    message: str = ""

    def human(self) -> str:
        status = "PASS" if self.passed else "FAIL"
        detail = self.message or f"expected {self.expected}, got {self.actual}"
        return f"gate {self.metric} ({self.kind}): {status} - {detail}"


class AutoResolved(Event):
    param: str = ""
    value: str = ""

    def human(self) -> str:
        return f"auto-resolved {self.param} = {self.value}"


class ArtifactRegistered(Event):
    registry: str = ""
    name: str = ""
    version: str = ""
    stage: str = ""

    def human(self) -> str:
        return f"registered {self.name} v{self.version} -> {self.stage} ({self.registry})"


class StageTransitioned(Event):
    name: str = ""
    version: str = ""
    stage: str = ""

    def human(self) -> str:
        return f"transitioned {self.name} v{self.version} -> {self.stage}"


class PromotionApplied(Event):
    name: str = ""
    version: str = ""
    to_stage: str = ""
    forced: bool = False

    def human(self) -> str:
        forced = " (FORCED)" if self.forced else ""
        return f"promoted {self.name} v{self.version} -> {self.to_stage}{forced}"


class AdapterWarning(Event):
    adapter: str = ""
    message: str = ""
    level: Literal["debug", "info", "warn", "error"] = "warn"

    def human(self) -> str:
        return f"[{self.adapter}] {self.message}"


class StateDiffed(Event):
    added: int = 0
    removed: int = 0
    modified: int = 0
    env_changed: bool = False

    def human(self) -> str:
        env = "; env digest CHANGED" if self.env_changed else ""
        return (
            f"state diff: {self.added} added, {self.removed} removed, {self.modified} modified{env}"
        )


class RunFinished(Event):
    command: str = ""
    status: str = ""  # success | error | quality_failure
    succeeded: int = 0
    failed: int = 0
    skipped: int = 0
    elapsed_s: float = 0.0

    def human(self) -> str:
        return (
            f"{self.command} finished [{self.status}]: {self.succeeded} ok, "
            f"{self.failed} failed, {self.skipped} skipped in {self.elapsed_s:.1f}s"
        )


class JobLine(Event):
    """A raw line forwarded from a training-job subprocess."""

    line: str = ""
    level: Literal["debug", "info", "warn", "error"] = "debug"

    def human(self) -> str:
        return self.line
