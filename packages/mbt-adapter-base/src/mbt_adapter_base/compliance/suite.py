"""Compliance test base classes (TSD §12.4).

The tiny datasets are generated deterministically (fixed seed, stdlib
random) rather than committed as binary fixtures - equivalent stability,
reviewable source.
"""

import hashlib
import json
import shutil
import subprocess
import sys
import tempfile

from pathlib import Path
from typing import Any, ClassVar

import pyarrow as pa

from mbt_adapter_base import (
    AUTO,
    ArtifactRef,
    EvaluationProtocol,
    EvaluationSpec,
    MetricSpec,
    ModelSpec,
    RunContext,
    TaskType,
)
from mbt_adapter_base.datasets import InMemoryDatasetHandle


class _NullSink:
    def emit(self, event: object) -> None:
        pass


class TempArtifactStore:
    """Minimal file:// ArtifactStore for compliance runs."""

    def __init__(self, root: Path) -> None:
        self._root = root
        self.uri = f"file://{root}"

    def put_file(self, local_path: Path, name: str, format: str) -> ArtifactRef:
        destination = self._root / name
        destination.parent.mkdir(parents=True, exist_ok=True)
        shutil.copyfile(local_path, destination)
        payload = destination.read_bytes()
        return ArtifactRef(
            uri=f"file://{destination}",
            format=format,
            content_hash="sha256:" + hashlib.sha256(payload).hexdigest(),
            size_bytes=len(payload),
        )

    def fetch(self, ref: ArtifactRef) -> Path:
        return Path(ref.uri.removeprefix("file://"))


def tiny_binary_dataset(n_rows: int = 1000, seed: int = 99) -> InMemoryDatasetHandle:
    """~1k deterministic rows: 4 numeric features, learnable binary label.

    The table matches what adapters see at run time: features + label only
    (core strips split time columns before the adapter reads a table).
    """
    from random import Random

    rng = Random(seed)
    columns: dict[str, list[Any]] = {
        "f_signal": [],
        "f_noise": [],
        "f_scale": [],
        "f_binary": [],
        "label": [],
    }
    for _ in range(n_rows):
        signal = rng.gauss(0, 1)
        columns["f_signal"].append(signal)
        columns["f_noise"].append(rng.gauss(0, 1))
        columns["f_scale"].append(rng.uniform(0, 100))
        columns["f_binary"].append(rng.random() > 0.5)
        columns["label"].append(1 if signal + rng.gauss(0, 0.5) > 0.3 else 0)
    table = pa.table(columns)
    split = int(n_rows * 0.8)
    return InMemoryDatasetHandle(
        {"train": table.slice(0, split), "test": table.slice(split)},
        snapshot_id="sha256:compliance-tiny-binary",
        label_column="label",
    )


_BINARY_METRICS = [
    MetricSpec(name="roc_auc", kind="builtin"),
    MetricSpec(name="pr_auc", kind="builtin"),
    MetricSpec(name="logloss", kind="builtin", greater_is_better=False),
]


class TrainingAdapterCompliance:
    """Subclass per adapter; pytest collects the test_ methods (FR-ADPT-05)."""

    #: The adapter class (constructed with an empty config dict).
    adapter_factory: ClassVar[Any]
    #: Dotted module path of the plugin descriptor (import hygiene check).
    plugin_module: ClassVar[str]
    #: Framework modules that must NOT load at plugin import time (ADR-14).
    framework_modules: ClassVar[tuple[str, ...]]
    #: Hyperparameters guaranteed valid for this adapter's param model.
    valid_hyperparameters: ClassVar[dict[str, Any]] = {}
    #: A hyperparameter that supports the AUTO sentinel, if any.
    auto_hyperparameter: ClassVar[str | None] = None

    # -- helpers -------------------------------------------------------------

    def adapter(self) -> Any:
        return self.adapter_factory({})

    def dataset(self) -> InMemoryDatasetHandle:
        return tiny_binary_dataset()

    def model_spec(self, task: TaskType, **overrides: Any) -> ModelSpec:
        hyperparameters = dict(self.valid_hyperparameters)
        hyperparameters.update(overrides.pop("hyperparameters", {}))
        seed = overrides.pop("seed", 1234)
        return ModelSpec(
            name="compliance_model",
            task=task,
            adapter=getattr(self.adapter_factory, "name", "adapter"),
            owner="compliance@mbt.dev",
            dataset="ref('compliance_dataset')",
            target="label",
            hyperparameters=hyperparameters,
            evaluation=EvaluationSpec(
                protocol=EvaluationProtocol(), metrics=[m.name for m in _BINARY_METRICS]
            ),
            seed=seed,
            **overrides,
        )

    def run_context(self, seed: int = 1234) -> RunContext:
        return RunContext(
            run_id="compliance",
            unique_id="model.compliance.compliance_model",
            seed=seed,
            target_name="compliance",
            project_dir=".",
            vars={},
            events=_NullSink(),
        )

    def _train_and_evaluate(self, seed: int = 1234) -> dict[str, float]:
        adapter = self.adapter()
        data = self.dataset()
        spec = self.model_spec(TaskType.BINARY_CLASSIFICATION, seed=seed)
        model = adapter.train(spec, data, self.run_context(seed))
        results = adapter.evaluate(model, data, "test", _BINARY_METRICS)
        return dict(results.metrics)

    # -- the suite ---------------------------------------------------------------

    def test_declares_contract_metadata(self) -> None:
        adapter = self.adapter()
        assert adapter.name, "adapter must declare a name"
        assert adapter.contract_version, "adapter must pin a contract_version"
        assert adapter.supported_tasks, "adapter must declare supported_tasks"
        assert adapter.determinism.kind in ("exact", "tolerance")

    def test_plugin_import_hygiene(self) -> None:
        """Importing the plugin module must not import the framework (ADR-14)."""
        probe = (
            "import json, sys\n"
            f"import {self.plugin_module}\n"
            f"loaded = [m for m in {json.dumps(list(self.framework_modules))} "
            "if m in sys.modules]\n"
            "print(json.dumps(loaded))\n"
        )
        proc = subprocess.run(  # noqa: S603
            [sys.executable, "-c", probe], capture_output=True, text=True, check=True
        )
        loaded = json.loads(proc.stdout.strip().splitlines()[-1])
        assert loaded == [], (
            f"importing {self.plugin_module} loaded framework module(s) {loaded}; "
            "frameworks must load lazily inside adapter methods (ADR-14)"
        )

    def test_param_model_rejects_unknown_params(self) -> None:
        import pytest

        adapter = self.adapter()
        for task in adapter.supported_tasks:
            param_model = adapter.param_model(task)
            with pytest.raises(Exception, match="(?i)extra|unknown|forbid|permitted"):
                param_model.model_validate(
                    {**self.valid_hyperparameters, "definitely_not_a_param": 1}
                )

    def test_seed_determinism_within_declared_tier(self) -> None:
        adapter = self.adapter()
        first = self._train_and_evaluate(seed=1234)
        second = self._train_and_evaluate(seed=1234)
        for metric, value in first.items():
            tolerance = adapter.determinism.tolerance_for(metric)
            assert abs(value - second[metric]) <= tolerance, (
                f"{metric}: {value} vs {second[metric]} exceeds the declared "
                f"determinism tier ({adapter.determinism.kind})"
            )

    def test_resolve_auto_idempotent_and_no_sentinels(self) -> None:
        adapter = self.adapter()
        data = self.dataset()
        profile = data.profile()
        if self.auto_hyperparameter is None:
            spec = self.model_spec(TaskType.BINARY_CLASSIFICATION)
        else:
            spec = self.model_spec(
                TaskType.BINARY_CLASSIFICATION,
                hyperparameters={self.auto_hyperparameter: AUTO},
            )
        once = adapter.resolve_auto(spec, profile)
        assert AUTO not in once.hyperparameters.values(), "AUTO sentinels left unresolved"
        twice = adapter.resolve_auto(once, profile)
        assert once.hyperparameters == twice.hyperparameters, "resolve_auto not idempotent"

    def test_train_export_load_evaluate_round_trip(self) -> None:
        adapter = self.adapter()
        data = self.dataset()
        spec = self.model_spec(TaskType.BINARY_CLASSIFICATION)
        model = adapter.train(spec, data, self.run_context())
        direct = adapter.evaluate(model, data, "test", _BINARY_METRICS)
        with tempfile.TemporaryDirectory() as tmp:
            store = TempArtifactStore(Path(tmp))
            ref = adapter.export(model, "native", store)
            assert ref.content_hash.startswith("sha256:")
            assert ref.size_bytes > 0
            loaded = adapter.load(ref, store)
            reloaded = adapter.evaluate(loaded, data, "test", _BINARY_METRICS)
        for metric, value in direct.metrics.items():
            tolerance = adapter.determinism.tolerance_for(metric)
            assert abs(value - reloaded.metrics[metric]) <= tolerance, (
                f"{metric} changed across export -> load: {value} vs "
                f"{reloaded.metrics[metric]}"
            )

    def test_predict_appends_prediction_column(self) -> None:
        adapter = self.adapter()
        data = self.dataset()
        spec = self.model_spec(TaskType.BINARY_CLASSIFICATION)
        model = adapter.train(spec, data, self.run_context())
        predictions = adapter.predict(model, data, "test")
        assert "prediction" in predictions.column_names
        assert predictions.num_rows == data.read("test").num_rows

    def test_model_actually_learns(self) -> None:
        """A signal-bearing dataset must beat coin-flip ROC AUC comfortably."""
        metrics = self._train_and_evaluate()
        assert metrics["roc_auc"] > 0.7, f"roc_auc {metrics['roc_auc']} suggests no learning"
