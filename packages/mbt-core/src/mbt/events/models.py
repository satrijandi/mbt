"""Typed event models (TSD §16).

Every significant occurrence is a Pydantic event carrying ``run_id``,
``unique_id`` (where applicable), and a timestamp. Sinks render them for
humans (Rich) or machines (JSON lines).

``Event`` and ``LogMessage`` are DEFINED in ``mbt_adapter_base.events`` and
re-exported here, along with the events adapters emit. An adapter package
cannot import core, so keeping the base on the adapter side is what lets the
seam be typed at all (B-4) - and it makes ``isinstance(x, Event)`` mean the
same thing in an adapter, in the bus, and in a sink. This module holds the
events only core emits.
"""

from mbt_adapter_base.events import (
    AdapterMessage,
    DatasetMaterialized,
    EarlyStoppingWithoutValidation,
    EmptyAfterTestSplit,
    Event,
    Level,
    LogMessage,
    ScoringInputMaterialized,
)


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
        expected, actual = _comparable_pair(self.expected, self.actual)
        numbers = f"expected {expected}, got {actual}"
        if not self.message:
            detail = numbers
        elif self.kind == "threshold" and self.actual is not None:
            # an after-test gate names the cell that decided (ADR-30)
            detail = f"{numbers} ({self.message})"
        else:
            detail = self.message
        return f"gate {self.metric} ({self.kind}): {status} - {detail}"


def _comparable_pair(expected: float | None, actual: float | None) -> tuple[str, str]:
    """Render a gate's two numbers as short as they can be while still differing.

    The raw float (``got 0.45782833946947715``) is noise on a console line, but
    a fixed precision is worse: a FAIL at 0.29996 against a 0.3 floor would
    render ``expected 0.3, got 0.3``. So widen the precision only as far as it
    takes to keep two different values visibly different. The JSON event keeps
    the exact floats either way.
    """
    if expected is None or actual is None:
        return str(expected), str(actual)
    for digits in range(4, 17):
        shown = f"{expected:.{digits}g}", f"{actual:.{digits}g}"
        if expected == actual or shown[0] != shown[1]:
            return shown
    # 17 significant digits distinguish any two distinct doubles
    return f"{expected:.17g}", f"{actual:.17g}"


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
    """The registry transition itself.

    Emitted at debug level because ``mbt promote`` already prints the outcome
    to stdout as command data: at info level the terminal showed the identical
    sentence twice, once from the event stream on stderr and once from the
    command (FEEDBACK v3 E-4). The level only gates the *console* sink, so the
    JSON-lines stream and any machine consumer still receive it unconditionally
    - and ``-v`` brings it back for anyone debugging a promotion.

    A FORCED promotion is different: it bypasses the quality contract, so it
    stays visible, and ``promote_model`` emits its own warn-level message for
    that case before this one.
    """

    name: str = ""
    version: str = ""
    to_stage: str = ""
    forced: bool = False
    level: Level = "debug"

    def human(self) -> str:
        forced = " (FORCED)" if self.forced else ""
        return f"promoted {self.name} v{self.version} -> {self.to_stage}{forced}"


class AdapterWarning(Event):
    adapter: str = ""
    message: str = ""
    level: Level = "warn"

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
    level: Level = "debug"

    def human(self) -> str:
        return self.line


__all__ = [
    "AdapterMessage",
    "AdapterWarning",
    "ArtifactRegistered",
    "AutoResolved",
    "CheckEvaluated",
    "CompileCompleted",
    "CompileStarted",
    "DatasetMaterialized",
    "EarlyStoppingWithoutValidation",
    "EmptyAfterTestSplit",
    "Event",
    "GateEvaluated",
    "JobLine",
    "Level",
    "LogMessage",
    "NodeFinished",
    "NodeStarted",
    "ParseCompleted",
    "ParseStarted",
    "PromotionApplied",
    "RunFinished",
    "RunStarted",
    "ScoringInputMaterialized",
    "StageTransitioned",
    "StateDiffed",
    "TestEvaluated",
]
