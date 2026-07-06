"""Fake adapters: deterministic, file-backed, framework-free.

The training adapter's "model" predicts a constant controlled by the
``fake_metric_value`` hyperparameter, so gate scenarios are scriptable.
Tracking and registry persist as JSON under a configurable root, making
assertions work across the coordinator/job process boundary.
"""

import json
import random
import tempfile
import threading
import uuid
from pathlib import Path
from typing import Any, ClassVar

import pyarrow as pa
from pydantic import BaseModel, ConfigDict

from mbt_adapter_base import (
    AUTO,
    CONTRACT_VERSION,
    ArtifactRef,
    ArtifactStore,
    DatasetHandle,
    DatasetProfile,
    DeterminismTier,
    JobResult,
    MetricResults,
    MetricSpec,
    ModelSpec,
    ModelVersion,
    RunContext,
    RunHandle,
    Stage,
    TaskType,
    TrainingJob,
    TuningResult,
    TuningSpec,
    ValidationIssue,
)
from mbt_adapter_base.interchange import ManifestNode, TuningObjectiveFn

_ = CONTRACT_VERSION  # re-exported via plugin


class FakeParams(BaseModel):
    model_config = ConfigDict(extra="forbid")

    max_depth: int = 3
    learning_rate: float = 0.1
    scale_pos_weight: float | str | None = None
    fake_metric_value: float = 0.5  # test control: every metric returns this
    fail_training: bool = False  # test control: raise inside train()


class FakeModel:
    def __init__(self, value: float) -> None:
        self.value = value


class FakeTrainingAdapter:
    """Deterministic, controllable stand-in for a real training adapter."""

    name = "fake"
    contract_version = CONTRACT_VERSION
    supported_tasks: ClassVar[set[TaskType]] = {TaskType.BINARY_CLASSIFICATION}
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

    def _params(self, spec: ModelSpec) -> FakeParams:
        return FakeParams.model_validate(
            {k: v for k, v in spec.hyperparameters.items() if v != AUTO}
        )

    def train(self, spec: ModelSpec, data: DatasetHandle, ctx: RunContext) -> FakeModel:
        params = self._params(spec)
        if params.fail_training:
            raise RuntimeError("fake training failure (fail_training=true)")
        data.read("train")  # honor the contract: training reads the data
        # A tiny deterministic dependence on params so tuning has a landscape.
        value = params.fake_metric_value + params.max_depth * 1e-4
        return FakeModel(value=round(value, 6))

    def evaluate(
        self,
        model: FakeModel,
        data: DatasetHandle,
        split: str,
        metrics: list[MetricSpec],
        slices: list[str] | None = None,
    ) -> MetricResults:
        table = data.read(split)
        values = {m.name: model.value for m in metrics if m.kind == "builtin"}
        slice_results: dict[str, dict[str, float]] = {}
        for column in slices or []:
            if column not in table.column_names:
                continue
            for value in sorted({str(v) for v in table.column(column).to_pylist()}):
                slice_results[f"{column}={value}"] = dict(values)
        return MetricResults(metrics=values, slices=slice_results)

    def predict(self, model: FakeModel, data: DatasetHandle, split: str) -> pa.Table:
        table = data.read(split)
        scores = [model.value] * table.num_rows
        return table.append_column("prediction", pa.array(scores, type=pa.float64()))

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


class FakeTrackingAdapter:
    """File-backed tracking: one JSON file per run under config.root."""

    def __init__(self, config: dict[str, Any] | None = None) -> None:
        config = config or {}
        self.root = Path(config.get("root", "./target/fake_tracking"))
        self._lock = threading.Lock()

    def _path(self, run_id: str) -> Path:
        return self.root / f"{run_id}.json"

    def _update(self, run_id: str, mutate: Any) -> None:
        with self._lock:
            path = self._path(run_id)
            payload = json.loads(path.read_text()) if path.is_file() else {}
            mutate(payload)
            path.parent.mkdir(parents=True, exist_ok=True)
            path.write_text(json.dumps(payload, indent=2, sort_keys=True))

    def start_run(self, node: ManifestNode, meta: dict[str, str]) -> RunHandle:
        run_id = f"fake-{uuid.uuid4().hex[:12]}"

        def init(payload: dict[str, Any]) -> None:
            payload.update(
                {
                    "run_id": run_id,
                    "node": node.unique_id,
                    "tags": dict(meta),
                    "params": {},
                    "metrics": {},
                    "artifacts": [],
                    "status": "RUNNING",
                }
            )

        self._update(run_id, init)
        return RunHandle(run_id=run_id)

    def log(
        self,
        run: RunHandle,
        *,
        params: dict[str, Any] | None = None,
        metrics: dict[str, float] | None = None,
        tags: dict[str, str] | None = None,
        artifacts: list[ArtifactRef] | None = None,
    ) -> None:
        def apply(payload: dict[str, Any]) -> None:
            payload.setdefault("params", {}).update(params or {})
            payload.setdefault("metrics", {}).update(metrics or {})
            payload.setdefault("tags", {}).update(tags or {})
            payload.setdefault("artifacts", []).extend(
                a.model_dump(mode="json") for a in artifacts or []
            )

        self._update(run.run_id, apply)

    def end_run(self, run: RunHandle, status: str) -> None:
        self._update(run.run_id, lambda p: p.__setitem__("status", status))

    def resume(self, run_id: str) -> RunHandle:
        return RunHandle(run_id=run_id)


class FakeRegistryAdapter:
    """File-backed registry: one JSON file per model name under config.root."""

    def __init__(self, config: dict[str, Any] | None = None) -> None:
        config = config or {}
        self.root = Path(config.get("root", "./target/fake_registry"))
        self._lock = threading.Lock()

    def _path(self, name: str) -> Path:
        return self.root / f"{name}.json"

    def _versions(self, name: str) -> list[dict[str, Any]]:
        path = self._path(name)
        if not path.is_file():
            return []
        return list(json.loads(path.read_text()))

    def _write(self, name: str, versions: list[dict[str, Any]]) -> None:
        path = self._path(name)
        path.parent.mkdir(parents=True, exist_ok=True)
        path.write_text(json.dumps(versions, indent=2, sort_keys=True))

    def register(self, artifact: ArtifactRef, name: str, metadata: dict[str, str]) -> ModelVersion:
        with self._lock:
            versions = self._versions(name)
            version = str(len(versions) + 1)
            entry = {
                "version": version,
                "stage": None,
                "artifact": artifact.model_dump(mode="json"),
                "tags": dict(metadata),
            }
            versions.append(entry)
            self._write(name, versions)
        return ModelVersion(name=name, version=version, artifact=artifact, tags=dict(metadata))

    def _to_model_version(self, name: str, entry: dict[str, Any]) -> ModelVersion:
        return ModelVersion(
            name=name,
            version=str(entry["version"]),
            stage=Stage(entry["stage"]) if entry.get("stage") else None,
            artifact=(
                ArtifactRef.model_validate(entry["artifact"]) if entry.get("artifact") else None
            ),
            tags=dict(entry.get("tags", {})),
        )

    def get_champion(self, name: str, stage: Stage) -> ModelVersion | None:
        candidates = [e for e in self._versions(name) if e.get("stage") == stage.value]
        if not candidates:
            return None
        latest = max(candidates, key=lambda e: int(e["version"]))
        return self._to_model_version(name, latest)

    def get_version(self, name: str, version: str) -> ModelVersion | None:
        for entry in self._versions(name):
            if str(entry["version"]) == str(version):
                return self._to_model_version(name, entry)
        return None

    def transition(self, version: ModelVersion, stage: Stage) -> None:
        with self._lock:
            versions = self._versions(version.name)
            for entry in versions:
                if str(entry["version"]) == version.version:
                    entry["stage"] = stage.value
                    self._write(version.name, versions)
                    return
        raise LookupError(f"version {version.version} of {version.name!r} not found")


class FakeTuningEngine:
    """Seeded random search honoring n_trials (no optuna dependency)."""

    name = "fake"

    def __init__(self, config: dict[str, Any] | None = None) -> None:
        self.config = config or {}

    def tune(
        self,
        spec: TuningSpec,
        objective: TuningObjectiveFn,
        n_trials: int,
        seed: int,
    ) -> TuningResult:
        rng = random.Random(seed)
        maximize = spec.objective.direction == "maximize"
        best_params: dict[str, Any] = {}
        best_value = float("-inf") if maximize else float("inf")
        for _ in range(n_trials):
            params: dict[str, Any] = {}
            for name, dim in spec.search_space.items():
                if dim.type == "categorical":
                    assert dim.choices is not None
                    params[name] = rng.choice(dim.choices)
                elif dim.type == "int":
                    assert dim.low is not None and dim.high is not None
                    params[name] = rng.randint(int(dim.low), int(dim.high))
                elif dim.type == "loguniform":
                    import math

                    assert dim.low is not None and dim.high is not None
                    low, high = math.log(dim.low), math.log(dim.high)
                    params[name] = math.exp(rng.uniform(low, high))
                else:  # uniform
                    assert dim.low is not None and dim.high is not None
                    params[name] = rng.uniform(dim.low, dim.high)
            value = objective(params)
            if (maximize and value > best_value) or (not maximize and value < best_value):
                best_params, best_value = params, value
        return TuningResult(best_params=best_params, best_value=best_value, n_trials=n_trials)


class _InlineJobHandle:
    def __init__(self, job: TrainingJob) -> None:
        self.job = job
        self.job_id = f"inline-{uuid.uuid4().hex[:12]}"


class InlineComputeAdapter:
    """Runs jobs in-process: fast tests, easy debugging. Not for production."""

    name = "fake"

    def __init__(self, config: dict[str, Any] | None = None) -> None:
        self.config = config or {}

    def submit(self, job: TrainingJob) -> _InlineJobHandle:
        return _InlineJobHandle(job)

    def wait(self, handle: _InlineJobHandle) -> JobResult:
        from mbt.execute.job import run_job

        return run_job(handle.job)
