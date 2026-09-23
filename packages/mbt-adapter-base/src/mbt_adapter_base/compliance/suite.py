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
from collections.abc import Mapping
from dataclasses import dataclass
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
    PredictionRunInfo,
    RunContext,
    TaskType,
)
from mbt_adapter_base.capabilities import Capability, capabilities_of, supports
from mbt_adapter_base.datasets import InMemoryDatasetHandle
from mbt_adapter_base.types import OUT_OF_TIME_SPLIT


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


def tiny_regression_dataset(n_rows: int = 1000, seed: int = 99) -> InMemoryDatasetHandle:
    """~1k deterministic rows: 4 numeric features, learnable continuous label."""
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
        scale = rng.uniform(0, 100)
        columns["f_signal"].append(signal)
        columns["f_noise"].append(rng.gauss(0, 1))
        columns["f_scale"].append(scale)
        columns["f_binary"].append(rng.random() > 0.5)
        # continuous target: mostly linear in the signal, with a small scaled
        # term and gaussian noise, so a real regressor comfortably beats the mean
        columns["label"].append(3.0 * signal + 0.02 * scale + rng.gauss(0, 0.5))
    table = pa.table(columns)
    split = int(n_rows * 0.8)
    return InMemoryDatasetHandle(
        {"train": table.slice(0, split), "test": table.slice(split)},
        snapshot_id="sha256:compliance-tiny-regression",
        label_column="label",
    )


def tiny_mixed_dataset(n_rows: int = 1000, seed: int = 99) -> InMemoryDatasetHandle:
    """~1k deterministic rows where the binary label is driven by a CATEGORICAL
    (string) feature plus numeric noise.

    The other fixtures are all-numeric, so the ship bar never exercises a string
    feature - which is exactly where cross-adapter handling diverges (F24/F25). An
    adapter that cannot train on a categorical, or silently mishandles it, fails
    to learn on this dataset and is caught here.
    """
    from random import Random

    rng = Random(seed)
    positive_rate = {"north": 0.92, "south": 0.08, "east": 0.5}
    columns: dict[str, list[Any]] = {"region": [], "f_noise": [], "f_scale": [], "label": []}
    for _ in range(n_rows):
        region = rng.choice(list(positive_rate))
        columns["region"].append(region)
        columns["f_noise"].append(rng.gauss(0, 1))
        columns["f_scale"].append(rng.uniform(0, 100))
        columns["label"].append(1 if rng.random() < positive_rate[region] else 0)
    table = pa.table(columns)
    split = int(n_rows * 0.8)
    return InMemoryDatasetHandle(
        {"train": table.slice(0, split), "test": table.slice(split)},
        snapshot_id="sha256:compliance-tiny-mixed",
        label_column="label",
    )


_BINARY_METRICS = [
    MetricSpec(name="roc_auc", kind="builtin"),
    MetricSpec(name="pr_auc", kind="builtin"),
    MetricSpec(name="logloss", kind="builtin", greater_is_better=False),
]

_REGRESSION_METRICS = [
    MetricSpec(name="rmse", kind="builtin", greater_is_better=False),
    MetricSpec(name="mae", kind="builtin", greater_is_better=False),
    MetricSpec(name="r2", kind="builtin"),
]

#: Builtin metrics used by the compliance model spec for each task.
_TASK_METRICS = {
    TaskType.BINARY_CLASSIFICATION: _BINARY_METRICS,
    TaskType.REGRESSION: _REGRESSION_METRICS,
}


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
    #: Overrides for the REGRESSION task, when an adapter's valid binary
    #: hyperparameters are not also valid for regression. Most adapters take
    #: the same knobs for both and leave this None; an adapter that selects an
    #: estimator in the spec (mbt-sklearn: `estimator: logistic` is binary-only,
    #: `linear` is regression-only) cannot, and would otherwise be untestable
    #: for one of its two declared tasks.
    regression_hyperparameters: ClassVar[dict[str, Any] | None] = None
    #: A hyperparameter that supports the AUTO sentinel, if any.
    auto_hyperparameter: ClassVar[str | None] = None

    # -- helpers -------------------------------------------------------------

    def adapter(self) -> Any:
        return self.adapter_factory({})

    def dataset(self) -> InMemoryDatasetHandle:
        return tiny_binary_dataset()

    def dataset_with_validation(self) -> InMemoryDatasetHandle:
        """A 3-split (train/validation/test) binary dataset for the optional
        capabilities that read a held-out validation split - train_with_report
        (it reports a validation value per round) and calibration's documented
        validation fallback - which the 2-split ``dataset()`` lacks."""
        base = tiny_binary_dataset()
        train_full = base.read("train")
        cut = int(train_full.num_rows * 0.75)
        return InMemoryDatasetHandle(
            {
                "train": train_full.slice(0, cut),
                "validation": train_full.slice(cut),
                "test": base.read("test"),
            },
            snapshot_id="sha256:compliance-tiny-binary-validation",
            label_column="label",
        )

    def dataset_with_calibration(self) -> InMemoryDatasetHandle:
        """A 3-split (train/calibration/test) binary dataset mirroring what core
        hands a calibrated fit: a dedicated ``calibration`` slice carved from
        train (F17), with no validation split at all - so the probe fails on an
        adapter that still insists on ``validation``."""
        base = tiny_binary_dataset()
        train_full = base.read("train")
        cut = int(train_full.num_rows * 0.75)
        return InMemoryDatasetHandle(
            {
                "train": train_full.slice(0, cut),
                "calibration": train_full.slice(cut),
                "test": base.read("test"),
            },
            snapshot_id="sha256:compliance-tiny-binary-calibration",
            label_column="label",
        )

    def model_spec(self, task: TaskType, **overrides: Any) -> ModelSpec:
        if task == TaskType.REGRESSION and self.regression_hyperparameters is not None:
            hyperparameters = dict(self.regression_hyperparameters)
        else:
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
                protocol=EvaluationProtocol(),
                metrics=[m.name for m in _TASK_METRICS[task]],
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
        proc = subprocess.run(
            [sys.executable, "-c", probe], capture_output=True, text=True, check=True
        )
        loaded = json.loads(proc.stdout.strip().splitlines()[-1])
        assert loaded == [], (
            f"importing {self.plugin_module} loaded framework module(s) {loaded}; "
            "frameworks must load lazily inside adapter methods (ADR-14)"
        )

    def test_plugin_imports_no_mbt_core(self) -> None:
        """An adapter builds against mbt-adapter-base only (G4).

        Importing the plugin must not load any ``mbt.*`` module: core internals
        are not a contract, and an adapter that reaches into them breaks on the
        next core refactor without a contract version ever changing. mbt-lightgbm
        and mbt-sklearn carried this check privately; the ship bar is where it
        belongs, so every adapter that claims compliance is held to it.
        """
        probe = (
            "import json, sys\n"
            f"import {self.plugin_module}\n"
            "loaded = sorted(m for m in sys.modules if m == 'mbt' or m.startswith('mbt.'))\n"
            "print(json.dumps(loaded))\n"
        )
        proc = subprocess.run(
            [sys.executable, "-c", probe], capture_output=True, text=True, check=True
        )
        loaded = json.loads(proc.stdout.strip().splitlines()[-1])
        assert loaded == [], (
            f"importing {self.plugin_module} loaded mbt-core module(s) {loaded}; "
            "adapters import mbt_adapter_base, never mbt-core internals"
        )

    def test_param_model_rejects_unknown_params(self) -> None:
        import pytest

        adapter = self.adapter()
        for task in adapter.supported_tasks:
            param_model = adapter.param_model(task)
            with pytest.raises(Exception, match=r"(?i)extra|unknown|forbid|permitted"):
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
                f"{metric} changed across export -> load: {value} vs {reloaded.metrics[metric]}"
            )

    def test_predict_appends_prediction_column(self) -> None:
        adapter = self.adapter()
        data = self.dataset()
        spec = self.model_spec(TaskType.BINARY_CLASSIFICATION)
        model = adapter.train(spec, data, self.run_context())
        predictions = adapter.predict(model, data, "test")
        assert "prediction" in predictions.column_names
        assert predictions.num_rows == data.read("test").num_rows

    def test_predict_scores_unlabeled_split(self) -> None:
        """``predict`` must work without the target column (batch scoring, ADR-20)."""
        adapter = self.adapter()
        data = self.dataset()
        spec = self.model_spec(TaskType.BINARY_CLASSIFICATION)
        model = adapter.train(spec, data, self.run_context())
        unlabeled = data.read("test").drop_columns(["label"])
        scoring = InMemoryDatasetHandle({"score": unlabeled}, snapshot_id="sha256:compliance-score")
        predictions = adapter.predict(model, scoring, "score")
        assert "prediction" in predictions.column_names
        assert predictions.num_rows == unlabeled.num_rows

    def test_model_actually_learns(self) -> None:
        """A signal-bearing dataset must beat coin-flip ROC AUC comfortably."""
        metrics = self._train_and_evaluate()
        assert metrics["roc_auc"] > 0.7, f"roc_auc {metrics['roc_auc']} suggests no learning"

    def test_learns_from_a_categorical_feature(self) -> None:
        """The other fixtures are all-numeric; this one puts the signal in a
        STRING feature, so an adapter that cannot train on a categorical - or
        silently mishandles it - fails to clear coin-flip ROC AUC (F25). This is
        the string-feature path the ship bar otherwise never exercises."""
        adapter = self.adapter()
        data = tiny_mixed_dataset()
        spec = self.model_spec(TaskType.BINARY_CLASSIFICATION)
        model = adapter.train(spec, data, self.run_context())
        metrics = adapter.evaluate(model, data, "test", _BINARY_METRICS)
        assert metrics.metrics["roc_auc"] > 0.7, (
            f"roc_auc {metrics.metrics['roc_auc']} suggests the categorical was not learned"
        )

    def test_regression_train_predict_evaluate(self) -> None:
        """OPTIONAL (adapters that declare ``REGRESSION``): train a real
        regressor - predictions are target-scale and a signal-bearing set beats
        predicting the mean (R^2 well above 0)."""
        import pytest

        if TaskType.REGRESSION not in self.adapter().supported_tasks:
            pytest.skip("adapter does not support regression")
        adapter = self.adapter()
        data = tiny_regression_dataset()
        spec = self.model_spec(TaskType.REGRESSION)
        model = adapter.train(spec, data, self.run_context())
        predictions = adapter.predict(model, data, "test")
        assert "prediction" in predictions.column_names
        assert predictions.num_rows == data.read("test").num_rows
        results = adapter.evaluate(model, data, "test", _REGRESSION_METRICS)
        assert results.metrics["rmse"] >= 0.0
        assert results.metrics["r2"] > 0.5, f"r2 {results.metrics['r2']} suggests no learning"

    def test_feature_importance_is_normalized_when_supported(self) -> None:
        """OPTIONAL capability (``SupportsFeatureImportance``): when the method
        exists, importances are non-negative fractions summing to ~1 - or {}
        when the winning model cannot attribute (e.g. an ensemble leader)."""
        import pytest

        adapter = self.adapter()
        if not supports(adapter, Capability.FEATURE_IMPORTANCE):
            pytest.skip("adapter does not declare Capability.FEATURE_IMPORTANCE (optional)")
        data = self.dataset()
        model = adapter.train(
            self.model_spec(TaskType.BINARY_CLASSIFICATION), data, self.run_context()
        )
        importance = adapter.feature_importance(model)
        if not importance:
            return  # the documented escape hatch for unattributable models
        features = getattr(model, "features", None)
        if features is not None:
            assert set(importance) <= set(features)
        assert all(value >= 0 for value in importance.values())
        assert sum(importance.values()) == pytest.approx(1.0, abs=1e-2)

    def test_shap_importance_is_normalized_when_supported(self) -> None:
        """OPTIONAL capability (``SupportsShapImportance``): when the method
        exists, mean-|SHAP| importances over a split are non-negative fractions
        summing to ~1 (the model card prefers this over ``feature_importance``).
        Data-grounded, so it takes the split, and it is computed on the actual
        model - a broken SHAP path is caught here, at the ship bar (F25)."""
        import pytest

        adapter = self.adapter()
        if not supports(adapter, Capability.SHAP_IMPORTANCE):
            pytest.skip("adapter does not declare Capability.SHAP_IMPORTANCE (optional)")
        data = self.dataset()
        model = adapter.train(
            self.model_spec(TaskType.BINARY_CLASSIFICATION), data, self.run_context()
        )
        importance = adapter.shap_importance(model, data, "test")
        features = getattr(model, "features", None)
        if features is not None:
            assert set(importance) <= set(features)
        assert all(value >= 0 for value in importance.values())
        assert sum(importance.values()) == pytest.approx(1.0, abs=1e-2)

    def test_explain_returns_per_row_attribution_when_supported(self) -> None:
        """OPTIONAL capability (``SupportsExplain``): when the method exists it
        returns one JSON string per row of the split - that row's <= ``top_k``
        ``[feature, contribution]`` pairs ordered by descending |contribution|
        (this is what a scoring node's ``output.explain_top_k`` serializes). A
        drifted explain shape is caught here, at the ship bar (F25)."""
        import json

        import pytest

        adapter = self.adapter()
        if not supports(adapter, Capability.EXPLAIN):
            pytest.skip("adapter does not declare Capability.EXPLAIN (optional)")
        data = self.dataset()
        model = adapter.train(
            self.model_spec(TaskType.BINARY_CLASSIFICATION), data, self.run_context()
        )
        top_k = 2
        explanations = adapter.explain(model, data, "test", top_k)
        assert len(explanations) == data.read("test").num_rows  # exactly one per row
        for blob in explanations:
            pairs = json.loads(blob)
            assert len(pairs) <= top_k
            magnitudes = [abs(float(contribution)) for _, contribution in pairs]
            assert magnitudes == sorted(magnitudes, reverse=True)  # descending |contribution|

    def test_declared_capabilities_match_what_the_adapter_can_do(self) -> None:
        """Capability is a declaration now, so it can be checked (B-1).

        Two directions, and both used to be unenforceable:

        - every capability the adapter DECLARES must have a callable method, so
          a declared-but-missing capability fails here instead of at train time;
        - every optional method the adapter DEFINES must be declared, so adding
          ``explain`` without declaring it cannot leave the capability dark.

        The old shape could express neither: the compliance suite gated on
        ``getattr(adapter, "supports_calibration", False)``, so a typo in the
        class-variable name silently SKIPPED the test rather than failing it.
        """
        from mbt_adapter_base.capabilities import _CAPABILITY_METHODS

        adapter = self.adapter()
        declared = capabilities_of(adapter)
        for capability, method in _CAPABILITY_METHODS.items():
            present = callable(getattr(adapter, method, None))
            if capability in declared:
                assert present, (
                    f"{type(adapter).__name__} declares {capability} but has no callable {method!r}"
                )
            else:
                assert not present, (
                    f"{type(adapter).__name__} defines {method!r} but does not declare "
                    f"{capability}, so core will never call it"
                )

    def test_calibration_round_trips_through_export_when_supported(self) -> None:
        """OPTIONAL capability (``Capability.CALIBRATION``): a model trained with
        post-hoc calibration must carry its calibrator through export -> load, so
        the calibration-sensitive metrics are unchanged afterward. A calibrator
        dropped on export silently un-calibrates a promoted model while its
        rank-based metrics (roc_auc/pr_auc) still look fine - so this probes
        brier/ece, which move under calibration, not auc, which does not. The
        calibrator is fit on the dedicated ``calibration`` slice (the handle
        core passes has no validation split at all, F17). F25."""
        import pytest

        adapter = self.adapter()
        if not supports(adapter, Capability.CALIBRATION):
            pytest.skip("adapter does not declare Capability.CALIBRATION (optional)")
        metrics = [
            MetricSpec(name="brier", kind="builtin", greater_is_better=False),
            MetricSpec(name="ece", kind="builtin", greater_is_better=False),
        ]
        data = self.dataset_with_calibration()
        spec = self.model_spec(TaskType.BINARY_CLASSIFICATION, calibration="isotonic")
        model = adapter.train(spec, data, self.run_context())
        direct = adapter.evaluate(model, data, "test", metrics)
        with tempfile.TemporaryDirectory() as tmp:
            store = TempArtifactStore(Path(tmp))
            ref = adapter.export(model, "native", store)
            reloaded = adapter.evaluate(adapter.load(ref, store), data, "test", metrics)
        for metric, value in direct.metrics.items():
            tolerance = adapter.determinism.tolerance_for(metric)
            assert abs(value - reloaded.metrics[metric]) <= tolerance, (
                f"{metric} changed across export -> load with calibration "
                f"({value} vs {reloaded.metrics[metric]}): the calibrator did not round-trip"
            )

    def test_calibration_falls_back_to_the_validation_split_when_supported(self) -> None:
        """OPTIONAL capability (``Capability.CALIBRATION``): a direct caller (this
        suite, a notebook) that passes a handle with a held-out ``validation``
        split and no ``calibration`` slice still gets a calibrated model - the
        documented fallback (F17). Probes that training succeeds and predictions
        stay within [0, 1] after calibration."""
        import pytest

        adapter = self.adapter()
        if not supports(adapter, Capability.CALIBRATION):
            pytest.skip("adapter does not declare Capability.CALIBRATION (optional)")
        data = self.dataset_with_validation()
        spec = self.model_spec(TaskType.BINARY_CLASSIFICATION, calibration="isotonic")
        model = adapter.train(spec, data, self.run_context())
        scores = adapter.predict(model, data, "test").column("prediction").to_pylist()
        assert scores and all(0.0 <= s <= 1.0 for s in scores)

    def test_train_with_report_streams_validation_progress_when_supported(self) -> None:
        """OPTIONAL tuning contract (``Capability.TRAIN_WITH_REPORT``): reports a
        higher-is-better validation value per round to the callback the Optuna
        pruner consumes, and still returns a usable model. A silent or drifted
        report path breaks pruning without failing training - caught here. It
        reports off the held-out ``validation`` split. F25."""
        import pytest

        adapter = self.adapter()
        if not supports(adapter, Capability.TRAIN_WITH_REPORT):
            pytest.skip("adapter does not declare Capability.TRAIN_WITH_REPORT (optional)")
        data = self.dataset_with_validation()
        spec = self.model_spec(TaskType.BINARY_CLASSIFICATION)
        reports: list[tuple[Any, Any]] = []
        model = adapter.train_with_report(
            spec, data, self.run_context(), lambda step, value: reports.append((step, value))
        )
        assert reports, "train_with_report never invoked its report callback"
        # the returned model is real and usable, not a placeholder
        assert "prediction" in adapter.predict(model, data, "test").column_names


def _tiny_predictions(n_rows: int, offset: int = 0) -> pa.Table:
    return pa.table(
        {
            "user_id": list(range(offset, offset + n_rows)),
            "prediction": [(i % 10) / 10.0 for i in range(n_rows)],
        }
    )


def _run_info(run_key: str, scored_at: str) -> PredictionRunInfo:
    return PredictionRunInfo(
        run_key=run_key,
        uri="",
        scored_at=scored_at,
        run_id=f"compliance-{run_key}",
        model_name="compliance_model",
        model_version="1",
        row_count=0,
    )


class PredictionStoreCompliance:
    """Subclass per DataAdapter with prediction support (contract 1.1, ADR-21).

    Override ``make_store`` to hand back a fresh, empty ``PredictionStore``
    rooted under ``root``.
    """

    def make_store(self, root: Path) -> Any:
        raise NotImplementedError

    def test_write_run_is_idempotent_by_run_key(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            store = self.make_store(Path(tmp))
            store.write_run(_tiny_predictions(3), _run_info("k1", "2026-01-01T00:00:00Z"))
            info = store.write_run(_tiny_predictions(2), _run_info("k1", "2026-01-02T00:00:00Z"))
            assert info.row_count == 2
            assert info.uri
            runs = store.list_runs()
            assert [r.run_key for r in runs] == ["k1"]
            assert store.read("k1").num_rows == 2

    def test_list_runs_ordered_by_scored_at(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            store = self.make_store(Path(tmp))
            store.write_run(_tiny_predictions(1), _run_info("later", "2026-02-01T00:00:00Z"))
            store.write_run(_tiny_predictions(1), _run_info("earlier", "2026-01-01T00:00:00Z"))
            assert [r.run_key for r in store.list_runs()] == ["earlier", "later"]

    def test_read_projects_columns(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            store = self.make_store(Path(tmp))
            store.write_run(_tiny_predictions(4), _run_info("k1", "2026-01-01T00:00:00Z"))
            table = store.read("k1", columns=["prediction"])
            assert table.column_names == ["prediction"]
            assert table.num_rows == 4

    def test_marker_roundtrip(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            store = self.make_store(Path(tmp))
            store.write_run(_tiny_predictions(2), _run_info("k1", "2026-01-01T00:00:00Z"))
            assert store.read_marker("k1", "ground_truth") is None
            payload = {"evaluated_at": "2026-01-20T00:00:00Z", "metrics": {"roc_auc": 0.9}}
            store.write_marker("k1", "ground_truth", payload)
            assert store.read_marker("k1", "ground_truth") == payload
            # A rewrite of the run clears its markers (fresh run, fresh ledger).
            store.write_run(_tiny_predictions(2), _run_info("k1", "2026-01-03T00:00:00Z"))
            assert store.read_marker("k1", "ground_truth") is None


# -- data adapters (A-1) ----------------------------------------------------


@dataclass
class ComplianceBuildContext:
    """A ``DataBuildContext`` for compliance runs, with plain fields."""

    node: Any
    source: Any
    source_tables: dict[str, Any]
    resolved_windows: dict[str, tuple[str, str]]
    sample_fraction: float
    deep_snapshot: bool
    output_dir: Path
    events: Any
    build_parallelism: int = 1


class RecordingEvents:
    """Collects emitted events so a compliance case can assert on their type."""

    def __init__(self) -> None:
        self.events: list[Any] = []

    def emit(self, event: Any) -> None:
        self.events.append(event)

    def of_type(self, kind: type) -> list[Any]:
        return [e for e in self.events if isinstance(e, kind)]


#: The compliance dataset's key column and the split it is bucketed into.
COMPLIANCE_SAMPLE_KEY = "row_id"
COMPLIANCE_SPLIT_SEED = 7
COMPLIANCE_FRACTIONS = {"train": 0.6, "validation": 0.2, "test": 0.2}


def tiny_source_rows(n_rows: int = 200) -> pa.Table:
    """The one relation a data-adapter compliance run reads (ADR-29).

    ``row_id`` is the sample key, ``ts`` the split time column, ``label`` the
    target. Deterministic, so every engine buckets identical rows.
    """
    import datetime as _dt

    start = _dt.datetime(2026, 1, 1)
    return pa.table(
        {
            COMPLIANCE_SAMPLE_KEY: list(range(n_rows)),
            "ts": [start + _dt.timedelta(days=i % 60) for i in range(n_rows)],
            "feature": [float(i % 17) for i in range(n_rows)],
            "label": [i % 2 for i in range(n_rows)],
        }
    )


class DataAdapterCompliance:
    """Subclass per DataAdapter (A-1). The data seam's ship bar.

    The seam had no compliance suite at all: ``compliance/suite.py`` shipped
    base classes for training adapters and prediction stores only, and
    ``mbt-testing`` shipped fakes for six seams and none for this one - so
    every dataset test ran real DuckDB, and the three adapters agreed on the
    build recipe only by each copying it.

    The load-bearing case is ``test_random_split_membership_matches_the_reference``:
    bucket membership against ``materialization.reference_bucket``, which is the
    arithmetic that decides which rows train. It used to be re-derived in three
    separate test files, one per adapter.

    Override ``make_adapter``: persist ``rows`` where the adapter reads them and
    return ``(adapter, source_table)``.
    """

    #: Engines whose zero-row split cannot be provoked through a window (e.g. a
    #: stub that answers every query) may set this False.
    supports_empty_split_error: ClassVar[bool] = True

    def make_adapter(self, root: Path, rows: pa.Table) -> tuple[Any, Any]:
        """Return ``(adapter, source_table)`` reading ``rows`` as one relation."""
        raise NotImplementedError

    # -- helpers -----------------------------------------------------------

    def _build(
        self,
        root: Path,
        *,
        windows: dict[str, tuple[str, str]] | None = None,
        sample_fraction: float = 1.0,
        n_rows: int = 200,
        events: Any = None,
    ) -> tuple[Any, Any, RecordingEvents]:
        from mbt_adapter_base.specs import DatasetSpec

        adapter, source = self.make_adapter(root, tiny_source_rows(n_rows))
        uid = f"source.compliance.{source.name}"
        sink = events or RecordingEvents()
        temporal = windows is not None
        spec = DatasetSpec.model_validate(
            {
                "name": "compliance_ds",
                "source": uid,
                "sample_key": [COMPLIANCE_SAMPLE_KEY],
                "label": {"column": "label"},
                "split": (
                    {"strategy": "temporal", "time_column": "ts", **_window_exprs(windows or {})}
                    if temporal
                    else {
                        "strategy": "random",
                        "train": str(COMPLIANCE_FRACTIONS["train"]),
                        "validation": str(COMPLIANCE_FRACTIONS["validation"]),
                        "test": str(COMPLIANCE_FRACTIONS["test"]),
                        "seed": COMPLIANCE_SPLIT_SEED,
                    }
                ),
            }
        )
        ctx = ComplianceBuildContext(
            node=_compliance_node("dataset.compliance.compliance_ds", "dataset"),
            source=source,
            source_tables={uid: source},
            resolved_windows=dict(windows or {}),
            sample_fraction=sample_fraction,
            deep_snapshot=False,
            output_dir=root / "materialization",
            events=sink,
        )
        return adapter, (spec, ctx), sink

    # -- cases -------------------------------------------------------------

    def test_random_split_membership_matches_the_reference(self) -> None:
        """The arithmetic that decides which rows train, pinned to one source.

        Every engine expresses the bucket in its own dialect; all of them must
        reproduce ``reference_bucket`` exactly, or a model validated on one
        backend trains on different rows on another (F19).
        """
        from mbt_adapter_base.materialization import reference_split

        with tempfile.TemporaryDirectory() as tmp:
            adapter, (spec, ctx), _ = self._build(Path(tmp))
            handle = adapter.build_dataset(spec, ctx)
            assert handle.splits() == {"train", "validation", "test"}
            seen = 0
            for split in sorted(handle.splits()):
                for key in handle.read(split).column(COMPLIANCE_SAMPLE_KEY).to_pylist():
                    expected = reference_split(
                        [str(key)], COMPLIANCE_FRACTIONS, COMPLIANCE_SPLIT_SEED
                    )
                    assert expected == split, f"{COMPLIANCE_SAMPLE_KEY}={key}"
                    seen += 1
            assert seen == 200  # every row landed somewhere; no bucket gap

    def test_a_successful_build_reports_its_row_counts_once(self) -> None:
        from mbt_adapter_base.events import DatasetMaterialized

        with tempfile.TemporaryDirectory() as tmp:
            adapter, (spec, ctx), sink = self._build(Path(tmp))
            handle = adapter.build_dataset(spec, ctx)
            reported = sink.of_type(DatasetMaterialized)
            assert len(reported) == 1
            assert reported[0].level == "info"
            assert reported[0].row_counts == {
                split: handle.read(split).num_rows for split in handle.splits()
            }

    def test_an_empty_after_test_split_warns_and_is_kept(self) -> None:
        """ADR-30's one exempt empty split, at one severity for every engine.

        This is v5 live defect 1: the local adapter emitted a typed WARN and
        the two warehouse adapters emitted a bare string the bus logged at
        INFO, so the same condition had two severities depending only on which
        adapter ran. No per-adapter test could see it, because each asserted
        against its own adapter.
        """
        from mbt_adapter_base.events import EmptyAfterTestSplit
        from mbt_adapter_base.types import OUT_OF_TIME_SPLIT

        windows = {
            "train": ("2026-01-01T00:00:00Z", "2026-02-01T00:00:00Z"),
            "test": ("2026-02-01T00:00:00Z", "2026-03-05T00:00:00Z"),
            OUT_OF_TIME_SPLIT: ("2027-01-01T00:00:00Z", "2027-02-01T00:00:00Z"),
        }
        with tempfile.TemporaryDirectory() as tmp:
            adapter, (spec, ctx), sink = self._build(Path(tmp), windows=windows)
            handle = adapter.build_dataset(spec, ctx)
            assert handle.read(OUT_OF_TIME_SPLIT).num_rows == 0
            warned = sink.of_type(EmptyAfterTestSplit)
            assert [e.level for e in warned] == ["warn"]
            assert warned[0].window == windows[OUT_OF_TIME_SPLIT]

    def test_an_empty_ordinary_split_is_an_error(self) -> None:
        """Only ``out_of_time`` is exempt; anything else empty is a build failure."""
        if not self.supports_empty_split_error:  # pragma: no cover - opt-out
            return
        windows = {
            "train": ("2026-01-01T00:00:00Z", "2026-02-01T00:00:00Z"),
            "test": ("2027-01-01T00:00:00Z", "2027-02-01T00:00:00Z"),  # no rows
        }
        import pytest

        with tempfile.TemporaryDirectory() as tmp:
            adapter, (spec, ctx), _ = self._build(Path(tmp), windows=windows)
            # the engine's own error type, whatever it is
            with pytest.raises(Exception, match="materialized 0 rows"):
                adapter.build_dataset(spec, ctx)

    def test_sample_fraction_outside_the_unit_interval_is_rejected(self) -> None:
        import pytest

        with tempfile.TemporaryDirectory() as tmp:
            adapter, (spec, ctx), _ = self._build(Path(tmp), sample_fraction=1.5)
            with pytest.raises(Exception, match="sample_fraction"):
                adapter.build_dataset(spec, ctx)

    def test_sampling_keeps_a_stable_subset(self) -> None:
        """Smaller fractions are subsets of larger ones, on every backend (F19)."""
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            keys: dict[float, set[int]] = {}
            for index, fraction in enumerate((1.0, 0.5)):
                adapter, (spec, ctx), _ = self._build(root / f"f{index}", sample_fraction=fraction)
                handle = adapter.build_dataset(spec, ctx)
                keys[fraction] = {
                    key
                    for split in handle.splits()
                    for key in handle.read(split).column(COMPLIANCE_SAMPLE_KEY).to_pylist()
                }
            assert keys[0.5] < keys[1.0]
            assert keys[0.5]

    def test_source_level_checks_are_available(self) -> None:
        """``count_source_duplicates`` and ``read_source_distinct`` are part of
        the DataAdapter protocol, not undeclared methods core finds by
        ``getattr`` (A-1)."""
        with tempfile.TemporaryDirectory() as tmp:
            adapter, source = self.make_adapter(Path(tmp), tiny_source_rows(50))
            assert adapter.count_source_duplicates(source, [COMPLIANCE_SAMPLE_KEY]) == 0
            distinct = adapter.read_source_distinct(source, "label")
            assert distinct.column_names == ["value"]
            assert set(distinct.column("value").to_pylist()) == {0, 1}


def _window_exprs(windows: Mapping[str, tuple[str, str]]) -> dict[str, Any]:
    """Temporal split expressions; the resolved windows drive the actual build."""
    exprs = {split: "-1y:now" for split in windows if split in ("train", "test")}
    if OUT_OF_TIME_SPLIT in windows:
        exprs[OUT_OF_TIME_SPLIT] = "now:+1y"
    return exprs


def _compliance_node(unique_id: str, resource_type: str) -> Any:
    from mbt_adapter_base.interchange import ManifestNode

    return ManifestNode(
        unique_id=unique_id,
        resource_type=resource_type,  # type: ignore[arg-type]
        name=unique_id.rsplit(".", 1)[-1],
        path=f"{unique_id}.yml",
        config={},
    )


# -- native categorical support (A-4) ---------------------------------------


def categorical_dataset(levels: list[str] | None = None) -> InMemoryDatasetHandle:
    """A dataset whose signal lives almost entirely in a categorical column."""
    n = 400
    plan_levels = levels or ["basic", "pro", "enterprise"]
    plan = [plan_levels[i % len(plan_levels)] for i in range(n)]
    noise = [((i * 37) % 100) / 100.0 for i in range(n)]
    label = [1 if (p == "enterprise") != (noise[i] > 0.9) else 0 for i, p in enumerate(plan)]
    table = pa.table({"noise": noise, "plan": plan, "label": label})
    return InMemoryDatasetHandle({"train": table, "test": table}, label_column="label")


class CategoricalAdapterCompliance:
    """Native categorical handling, for adapters that have it (FR-ADPT-03).

    This suite existed twice, once per package: after normalising the framework
    name, ``test_xgboost_categorical.py`` and ``test_lightgbm_categorical.py``
    differed by SIX lines and their four test function names were identical
    (A-4). The shape of the duplication matched the adapters' own.

    Override ``adapter_factory`` and ``hyperparameters``; what genuinely differs
    per framework is the tuning of a tiny model, not what the test asserts.
    """

    adapter_factory: ClassVar[Any]
    #: Enough rounds to learn a 3-level signal on 400 rows, per framework.
    hyperparameters: ClassVar[dict[str, Any]] = {}
    #: Where the level map is persisted, for the export/load assertion message.
    categories_home: ClassVar[str] = "the artifact"

    def adapter(self) -> Any:
        return self.adapter_factory({})

    def spec(self) -> ModelSpec:
        return ModelSpec.model_validate(
            {
                "name": "m",
                "task": TaskType.BINARY_CLASSIFICATION,
                "adapter": self.adapter().name,
                "owner": "t@example.com",
                "dataset": "ref('d')",
                "target": "label",
                "hyperparameters": dict(self.hyperparameters),
                "evaluation": EvaluationSpec(
                    protocol=EvaluationProtocol(), metrics=["roc_auc", "pr_auc", "logloss"]
                ),
                "seed": 5,
            }
        )

    def run_ctx(self) -> RunContext:
        return RunContext(
            run_id="t",
            unique_id="m",
            seed=5,
            target_name="dev",
            project_dir=".",
            vars={},
            events=_NullSink(),
        )

    def _trained(self) -> tuple[Any, Any, Any]:
        adapter = self.adapter()
        data = categorical_dataset()
        return adapter, data, adapter.train(self.spec(), data, self.run_ctx())

    def test_categorical_feature_carries_the_signal(self) -> None:
        import pytest

        adapter, data, model = self._trained()
        assert model.categories == {"plan": ["basic", "enterprise", "pro"]}  # sorted levels
        results = adapter.evaluate(
            model, data, "test", [MetricSpec(name="roc_auc", kind="builtin")], slices=None
        )
        assert results.metrics["roc_auc"] > 0.85  # only the categorical explains this

        importance = adapter.feature_importance(model)
        assert set(importance) == {"noise", "plan"}
        assert sum(importance.values()) == pytest.approx(1.0, abs=1e-3)
        assert importance["plan"] > 0.5  # the categorical dominates, as constructed

    def test_shap_importance_is_normalized_and_signal_dominant(self) -> None:
        """The model card prefers mean-|SHAP| (additive, not cardinality-biased)
        over split-gain when eval data is available."""
        import pytest

        adapter, data, model = self._trained()
        shap = adapter.shap_importance(model, data, "test")
        assert set(shap) == {"noise", "plan"}
        assert all(value >= 0 for value in shap.values())  # mean |SHAP| is non-negative
        assert sum(shap.values()) == pytest.approx(1.0, abs=1e-3)  # normalized to fractions
        assert shap["plan"] > 0.5  # SHAP agrees the categorical carries the signal

    def test_explain_gives_per_row_top_k_contributors(self) -> None:
        """Local attribution: each row's top_k features by |SHAP|, ordered, as JSON."""
        import json

        adapter, data, model = self._trained()
        rows = adapter.explain(model, data, "test", top_k=2)
        assert len(rows) == data.read("test").num_rows
        top = json.loads(rows[0])
        assert len(top) == 2 and all(feature in model.features for feature, _ in top)
        assert abs(top[0][1]) >= abs(top[1][1])  # ordered by descending |contribution|

    def test_categories_survive_export_load_and_unseen_levels_predict(self) -> None:
        adapter, data, model = self._trained()
        scores = adapter.predict(model, data, "test").column("prediction").to_pylist()

        with tempfile.TemporaryDirectory() as tmp:
            store = TempArtifactStore(Path(tmp))
            ref = adapter.export(model, "native", store)
            loaded = adapter.load(ref, store)
        assert loaded.categories == model.categories, (
            f"the level map did not survive export -> load ({self.categories_home})"
        )
        reloaded = adapter.predict(loaded, data, "test").column("prediction").to_pylist()
        assert reloaded == scores  # champion path scores identically

        # a level unseen at train time maps to missing, never crashes
        unseen = categorical_dataset(["basic", "pro", "enterprise", "trial"])
        assert adapter.predict(loaded, unseen, "test").num_rows == 400


# -- registries (A-5) -------------------------------------------------------


class RegistryAdapterCompliance:
    """Subclass per RegistryAdapter. The champion contract's ship bar.

    There was no such suite: ``mbt-mlflow/tests/`` was the de facto contract,
    so a second registry adapter had nothing to build against - while 53
    ``mbt.*`` keys across 11 files in 2 packages conveyed which artifact, which
    hooks hash, whether gates passed and where the baseline lives, entirely by
    key spelling (A-5).

    The load-bearing case is ``test_champion_record_round_trips``: what
    ``promote.py``, ``oot_check.py`` and ``runners.py`` each separately assumed,
    asserted in one place.

    Override ``make_registry`` to return a fresh, empty registry rooted under
    ``root``.
    """

    def make_registry(self, root: Path) -> Any:
        raise NotImplementedError

    def _record(self) -> Any:
        from mbt_adapter_base.champion import AfterTestVerdict, ChampionRecord

        ref = ArtifactRef(
            uri="file:///tmp/model.json",
            format="native",
            content_hash="sha256:" + "ab" * 32,
            size_bytes=1234,
        )
        return ChampionRecord(
            artifact=ref,
            config_hash="sha256:cfg",
            input_hash="sha256:inp",
            manifest_hash="sha256:man",
            snapshot_id="sha256:snap",
            git_commit="deadbeef",
            tracking_run_id="run-1",
            hooks_hash="sha256:hooks",
            gates_passed=True,
            baseline=ArtifactRef(
                uri="file:///tmp/baseline.json", format="json", content_hash="", size_bytes=7
            ),
            inference_config=ArtifactRef(
                uri="file:///tmp/inference.json", format="json", content_hash="", size_bytes=9
            ),
            report_uri="file:///tmp/report/index.html",
            after_test=AfterTestVerdict(
                passed="true", anchor="2026-07-01T00:00:00Z", source="build"
            ),
            operating_points={"threshold_at_precision_0.35": "0.61"},
        )

    def test_champion_record_round_trips(self) -> None:
        """``get_version(register(record)).record == record``.

        Every load-bearing fact a promotion decision reads must survive the
        registry unchanged. A typo on the write side used to be caught only by
        whichever e2e run happened to read that key back.
        """
        from mbt_adapter_base.champion import ChampionRecord

        with tempfile.TemporaryDirectory() as tmp:
            registry = self.make_registry(Path(tmp))
            record = self._record()
            version = registry.register(record.artifact, "compliance_model", record.pack())
            fetched = registry.get_version("compliance_model", version.version)
            assert fetched is not None
            assert ChampionRecord.unpack(fetched.tags) == record

    def test_registered_version_carries_a_loadable_artifact(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            registry = self.make_registry(Path(tmp))
            record = self._record()
            version = registry.register(record.artifact, "compliance_model", record.pack())
            assert version.artifact is not None
            assert version.artifact.uri == record.artifact.uri
            assert version.artifact.size_bytes == record.artifact.size_bytes

    def test_transition_makes_the_version_the_stage_champion(self) -> None:
        from mbt_adapter_base.types import Stage

        with tempfile.TemporaryDirectory() as tmp:
            registry = self.make_registry(Path(tmp))
            record = self._record()
            version = registry.register(record.artifact, "compliance_model", record.pack())
            assert registry.get_champion("compliance_model", Stage.PRODUCTION) is None
            registry.transition(version, Stage.PRODUCTION)
            champion = registry.get_champion("compliance_model", Stage.PRODUCTION)
            assert champion is not None and champion.version == version.version

    def test_a_second_registration_does_not_disturb_the_first(self) -> None:
        """Versions are independent records, so promoting one never rewrites
        another's tags - which is what makes a rollback target intact."""
        from mbt_adapter_base.champion import ChampionRecord

        with tempfile.TemporaryDirectory() as tmp:
            registry = self.make_registry(Path(tmp))
            first = self._record()
            v1 = registry.register(first.artifact, "compliance_model", first.pack())
            from dataclasses import replace

            second = replace(first, gates_passed=False, git_commit="cafebabe")
            v2 = registry.register(second.artifact, "compliance_model", second.pack())
            assert v1.version != v2.version
            back = registry.get_version("compliance_model", v1.version)
            assert back is not None
            assert ChampionRecord.unpack(back.tags) == first

    def test_get_version_of_an_unknown_version_is_none(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            registry = self.make_registry(Path(tmp))
            assert registry.get_version("compliance_model", "999") is None
