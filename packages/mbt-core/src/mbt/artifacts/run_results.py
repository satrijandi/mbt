"""``target/run_results.json`` (TSD §10.8, FR-RUN-04)."""

import json
from pathlib import Path
from typing import Any, Literal

from pydantic import BaseModel, ConfigDict, Field

from mbt.artifacts.atomic import atomic_write_text
from mbt.contracts import ArtifactRef
from mbt.secrets import redact

RUN_RESULTS_SCHEMA_VERSION = 1

NodeStatus = Literal["success", "error", "skipped", "gate_failed", "test_failed", "monitor_failed"]


class GateResult(BaseModel):
    model_config = ConfigDict(extra="forbid")

    metric: str
    kind: Literal["threshold", "champion", "disparity"]
    slice: str | None = None  # "column=value" for per-slice gates
    passed: bool
    expected: float | None = None  # threshold gates; min_ratio for disparity gates
    actual: float | None = None  # worst/best ratio for disparity gates
    champion_version: str | None = None  # champion gates
    champion_value: float | None = None
    min_delta: float | None = None
    actual_delta: float | None = None
    delta_lower: float | None = None  # paired-bootstrap lower bound (ADR-18)
    confidence: float | None = None
    across: str | None = None  # disparity gates: the slice column measured
    worst_slice: str | None = None  # disparity gates: "column=value" of the worst slice
    best_slice: str | None = None  # disparity gates: "column=value" of the best slice
    message: str | None = None


class RegistrationResult(BaseModel):
    model_config = ConfigDict(extra="forbid")

    registry: str
    name: str
    version: str
    stage: str


class MonitorResult(BaseModel):
    """One monitor comparison on a scoring node (ADR-20/21)."""

    model_config = ConfigDict(extra="forbid")

    monitor: Literal["feature_shift", "prediction_shift", "ground_truth"]
    #: What was compared: a feature name, or a prediction run_key.
    subject: str | None = None
    #: The shift method (psi/ks) or the realized metric name.
    measure: str
    value: float | None = None
    threshold: float
    passed: bool
    message: str | None = None


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
    monitors: list[MonitorResult] = Field(default_factory=list)  # scoring nodes (ADR-20)
    artifact: ArtifactRef | None = None
    registration: RegistrationResult | None = None
    tracking_run_id: str | None = None
    resolved_auto: dict[str, Any] = Field(default_factory=dict)
    feature_importance: dict[str, float] = Field(default_factory=dict)  # FR-DOCS-02
    #: Partial dependence per top numeric feature (explainability): feature ->
    #: [[grid_value, avg_prediction], ...]; rendered as a sparkline in the card.
    partial_dependence: dict[str, list[list[float]]] = Field(default_factory=dict)
    #: Walk-forward backtest (R2-7): builtin metric -> mean across time-ordered
    #: folds; shown beside the single-split metrics on the card.
    backtest_metrics: dict[str, float] = Field(default_factory=dict)
    #: Population std of each backtest metric across folds (R2-7); rendered as
    #: ``mean ± std`` so the card exposes how stable the estimate is.
    backtest_std: dict[str, float] = Field(default_factory=dict)
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
        if statuses & {"gate_failed", "test_failed", "monitor_failed"}:
            return 2
        return 0

    def to_json(self) -> str:
        return redact(json.dumps(self.model_dump(mode="json"), indent=2, sort_keys=True) + "\n")

    def write(self, path: Path) -> None:
        atomic_write_text(path, self.to_json())
