"""``target/run_results.json`` (TSD §10.8, FR-RUN-04)."""

import json
from pathlib import Path
from typing import Any, Literal

from pydantic import BaseModel, ConfigDict, Field

from mbt.contracts import ArtifactRef
from mbt.secrets import redact

RUN_RESULTS_SCHEMA_VERSION = 1

NodeStatus = Literal["success", "error", "skipped", "gate_failed", "test_failed"]


class GateResult(BaseModel):
    model_config = ConfigDict(extra="forbid")

    metric: str
    kind: Literal["threshold", "champion"]
    passed: bool
    expected: float | None = None  # threshold gates
    actual: float | None = None
    champion_version: str | None = None  # champion gates
    champion_value: float | None = None
    min_delta: float | None = None
    actual_delta: float | None = None
    message: str | None = None


class RegistrationResult(BaseModel):
    model_config = ConfigDict(extra="forbid")

    registry: str
    name: str
    version: str
    stage: str


class TestResultEntry(BaseModel):
    model_config = ConfigDict(extra="forbid")

    name: str
    passed: bool
    message: str = ""


class NodeResult(BaseModel):
    model_config = ConfigDict(extra="forbid")

    unique_id: str
    status: NodeStatus
    execution_time_s: float = 0.0
    metrics: dict[str, float] = Field(default_factory=dict)
    slices: dict[str, dict[str, float]] = Field(default_factory=dict)
    gates: list[GateResult] = Field(default_factory=list)
    tests: list[TestResultEntry] = Field(default_factory=list)
    artifact: ArtifactRef | None = None
    registration: RegistrationResult | None = None
    tracking_run_id: str | None = None
    resolved_auto: dict[str, Any] = Field(default_factory=dict)
    message: str | None = None


class RunResultsMetadata(BaseModel):
    model_config = ConfigDict(extra="forbid")

    run_results_schema_version: int = RUN_RESULTS_SCHEMA_VERSION
    run_id: str
    mbt_version: str
    target: str
    manifest_hash: str
    anchor: str
    started_at: str
    elapsed_s: float = 0.0
    command: str
    selector: str | None = None


class RunResults(BaseModel):
    model_config = ConfigDict(extra="forbid")

    metadata: RunResultsMetadata
    results: list[NodeResult] = Field(default_factory=list)

    def exit_code(self) -> int:
        """Exit precedence: error -> 1, quality failure -> 2, else 0 (TSD §17)."""
        statuses = {r.status for r in self.results}
        if "error" in statuses:
            return 1
        if statuses & {"gate_failed", "test_failed"}:
            return 2
        return 0

    def to_json(self) -> str:
        return redact(json.dumps(self.model_dump(mode="json"), indent=2, sort_keys=True) + "\n")

    def write(self, path: Path) -> None:
        path.parent.mkdir(parents=True, exist_ok=True)
        path.write_text(self.to_json())
