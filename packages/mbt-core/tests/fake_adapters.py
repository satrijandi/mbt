"""Contract-conformant fake adapters for core tests (TSD §21).

No ML dependencies: the "model" is the label mean plus a controllable
metric table, so planner/scheduler/gate/skip logic is testable in-process.
"""

import json
import tempfile
from pathlib import Path
from typing import Any

from pydantic import BaseModel, ConfigDict

from mbt.contracts import (
    AdapterPlugin,
    ArtifactRef,
    ArtifactStore,
    DatasetHandle,
    DatasetProfile,
    DeterminismTier,
    MetricResults,
    MetricSpec,
    ModelSpec,
    RunContext,
    TaskType,
    ValidationIssue,
)
from mbt.contracts import AUTO, CONTRACT_VERSION


class FakeParams(BaseModel):
    model_config = ConfigDict(extra="forbid")

    max_depth: int = 3
    learning_rate: float = 0.1
    scale_pos_weight: float | str | None = None
    fake_metric_value: float = 0.5  # test control: every metric returns this


class FakeModel:
    def __init__(self, value: float) -> None:
        self.value = value


class FakeTrainingAdapter:
    """Deterministic, controllable stand-in for a real training adapter."""

    name = "fake"
    contract_version = CONTRACT_VERSION
    supported_tasks = {TaskType.BINARY_CLASSIFICATION}
    determinism = DeterminismTier(kind="exact")

    def __init__(self, config: dict[str, Any] | None = None) -> None:
        self.config = config or {}

    def param_model(self, task: TaskType) -> type[BaseModel]:
        return FakeParams

    def validate(self, spec: ModelSpec) -> list[ValidationIssue]:
        return []

    def resolve_auto(self, spec: ModelSpec, profile: DatasetProfile) -> ModelSpec:
        resolved = dict(spec.hyperparameters)
        for key, value in resolved.items():
            if value == AUTO:
                balance = profile.label_balance or {}
                positive = balance.get("1", balance.get("1.0", 0.5)) or 0.5
                resolved[key] = round((1 - positive) / positive, 4)
        return spec.model_copy(update={"hyperparameters": resolved})

    def train(self, spec: ModelSpec, data: DatasetHandle, ctx: RunContext) -> FakeModel:
        params = FakeParams.model_validate(
            {k: v for k, v in spec.hyperparameters.items() if v != AUTO}
        )
        return FakeModel(value=params.fake_metric_value)

    def evaluate(
        self,
        model: FakeModel,
        data: DatasetHandle,
        split: str,
        metrics: list[MetricSpec],
        slices: list[str] | None = None,
    ) -> MetricResults:
        values = {m.name: model.value for m in metrics if m.kind == "builtin"}
        slice_values: dict[str, dict[str, float]] = {}
        for column in slices or []:
            slice_values[f"{column}=all"] = dict(values)
        return MetricResults(metrics=values, slices=slice_values)

    def load(self, ref: ArtifactRef, store: ArtifactStore) -> FakeModel:
        payload = json.loads(store.fetch(ref).read_text())
        return FakeModel(value=payload["value"])

    def export(self, model: FakeModel, format: str, store: ArtifactStore) -> ArtifactRef:
        with tempfile.TemporaryDirectory() as tmp:
            path = Path(tmp) / "fake_model.json"
            path.write_text(json.dumps({"value": model.value}))
            return store.put_file(path, "fake_model.json", format="fake_json")

    def nondeterminism_warnings(self, spec: ModelSpec) -> list[str]:
        return []


FAKE_PLUGIN = AdapterPlugin(
    name="fake",
    contract_version=CONTRACT_VERSION,
    training=FakeTrainingAdapter,
    fingerprint_packages=[],
)
