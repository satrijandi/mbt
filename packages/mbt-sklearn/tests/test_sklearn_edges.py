"""Edge paths of the sklearn adapter the compliance suite does not reach.

The suite proves the happy path for each estimator; these cover the refusals,
the optional knobs, and the encodings that differ between the linear and tree
families - the places a second estimator would otherwise silently regress.
"""

from pathlib import Path
from typing import Any

import numpy as np
import pyarrow as pa
import pytest
from mbt_sklearn.adapter import ARTIFACT_FORMAT, SklearnModel, SklearnTrainingAdapter
from mbt_sklearn.params import LogisticParams, SklearnBinaryParams

from mbt_adapter_base import (
    ArtifactRef,
    EvaluationProtocol,
    EvaluationSpec,
    ModelSpec,
    RunContext,
    TaskType,
)
from mbt_adapter_base.compliance.suite import TempArtifactStore, _NullSink
from mbt_adapter_base.datasets import InMemoryDatasetHandle


def _ctx(seed: int = 1) -> RunContext:
    """The same shape the compliance harness hands an adapter."""
    return RunContext(
        run_id="edges",
        unique_id="model.p.m",
        seed=seed,
        target_name="dev",
        project_dir=".",
        vars={},
        events=_NullSink(),
    )


def _spec(task: TaskType = TaskType.BINARY_CLASSIFICATION, **hyper: Any) -> ModelSpec:
    return ModelSpec(
        name="m",
        task=task,
        adapter="sklearn",
        owner="t@example.com",
        dataset="ref('d')",
        target="label",
        hyperparameters=dict(hyper),
        evaluation=EvaluationSpec(protocol=EvaluationProtocol(), metrics=["roc_auc"]),
        seed=7,
    )


def _data(n: int = 200) -> InMemoryDatasetHandle:
    rng = np.random.default_rng(3)
    signal = rng.normal(size=n)
    table = pa.table(
        {
            "f1": signal,
            "region": ["north" if s > 0 else "south" for s in signal],
            "label": (signal > 0).astype("int64"),
        }
    )
    cut = int(n * 0.8)
    return InMemoryDatasetHandle(
        {"train": table.slice(0, cut), "test": table.slice(cut)},
        snapshot_id="sha256:edges",
        label_column="label",
    )


def test_invalid_hyperparameters_fail_with_a_readable_message() -> None:
    adapter = SklearnTrainingAdapter({})
    with pytest.raises(ValueError, match="invalid sklearn hyperparameters"):
        adapter.train(_spec(estimator="logistic", C=-1), _data(), _ctx())


def test_a_hyperparameter_from_another_estimator_names_both() -> None:
    """The union model accepts the key; the concrete estimator must reject it,
    and the message has to say which estimator and what IS legal."""
    with pytest.raises(ValueError, match="not valid for estimator 'random_forest'"):
        SklearnBinaryParams.model_validate({"estimator": "random_forest", "C": 0.5})


def test_optional_logistic_knobs_are_forwarded_only_when_set() -> None:
    """`penalty` is deprecated upstream in 1.8, so the default path must not
    forward it; the others are plain optional passthroughs."""
    default = LogisticParams().estimator_kwargs(seed=1)
    assert "penalty" not in default
    assert "l1_ratio" not in default
    assert "class_weight" not in default

    explicit = LogisticParams(
        penalty="elasticnet", l1_ratio=0.3, class_weight="balanced", solver="saga"
    ).estimator_kwargs(seed=1)
    assert explicit["penalty"] == "elasticnet"
    assert explicit["l1_ratio"] == 0.3
    assert explicit["class_weight"] == "balanced"
    assert explicit["n_jobs"] is None  # only liblinear takes it


def test_liblinear_receives_n_jobs() -> None:
    kwargs = LogisticParams(solver="liblinear", n_jobs=1).estimator_kwargs(seed=1)
    assert kwargs["n_jobs"] == 1


def test_hist_gradient_boosting_early_stopping_is_wired_by_the_adapter() -> None:
    adapter = SklearnTrainingAdapter({})
    spec = _spec(estimator="hist_gradient_boosting", max_iter=20, early_stopping_rounds=3)
    estimator = adapter._build_estimator(spec, _ctx())
    assert estimator.early_stopping is True
    assert estimator.n_iter_no_change == 3


def test_linear_estimators_one_hot_categoricals_and_trees_do_not() -> None:
    """The encoding is a property of the model family, not the data (a linear
    model reads an ordinal code as a magnitude). One column per level for the
    linear family; one column total for the trees."""
    adapter = SklearnTrainingAdapter({})
    data = _data()
    table = data.read("train")

    linear = SklearnModel(
        None,
        "logistic",
        ["f1", "region"],
        "label",
        TaskType.BINARY_CLASSIFICATION,
        {"region": ["north", "south"]},
    )
    matrix, owners = adapter._design_matrix(linear, table)
    assert matrix.shape[1] == 3  # f1 + 2 one-hot levels
    assert owners == ["f1", "region", "region"]

    tree = SklearnModel(
        None,
        "random_forest",
        ["f1", "region"],
        "label",
        TaskType.BINARY_CLASSIFICATION,
        {"region": ["north", "south"]},
    )
    matrix, owners = adapter._design_matrix(tree, table)
    assert matrix.shape[1] == 2
    assert owners == ["f1", "region"]


def test_a_featureless_model_yields_an_empty_matrix() -> None:
    adapter = SklearnTrainingAdapter({})
    model = SklearnModel(None, "logistic", [], "label", TaskType.BINARY_CLASSIFICATION, {})
    matrix, owners = adapter._design_matrix(model, _data().read("train"))
    assert matrix.shape == (160, 0)
    assert owners == []


def test_one_hot_importance_is_reported_per_feature_not_per_level() -> None:
    adapter = SklearnTrainingAdapter({})
    data = _data()
    model = adapter.train(_spec(estimator="logistic"), data, _ctx())
    importance = adapter.feature_importance(model)
    assert set(importance) == {"f1", "region"}
    assert abs(sum(importance.values()) - 1.0) < 1e-6


def test_hist_gradient_boosting_reports_no_importance_rather_than_zeros() -> None:
    """It exposes neither `coef_` nor `feature_importances_`; the contract's
    escape hatch is {}, not a row of zeros dressed up as a ranking."""
    adapter = SklearnTrainingAdapter({})
    data = _data()
    model = adapter.train(_spec(estimator="hist_gradient_boosting", max_iter=10), data, _ctx())
    assert adapter.feature_importance(model) == {}


def test_an_all_zero_importance_vector_degrades_to_zeros() -> None:
    """A model that learned nothing must not divide by zero."""
    adapter = SklearnTrainingAdapter({})
    model = SklearnModel(None, "logistic", ["a", "b"], "label", TaskType.BINARY_CLASSIFICATION, {})
    model.estimator = type("E", (), {"coef_": np.zeros((1, 2))})()
    model.column_owners = ["a", "b"]
    assert adapter.feature_importance(model) == {"a": 0.0, "b": 0.0}


def test_export_refuses_an_unknown_format(tmp_path: Path) -> None:
    adapter = SklearnTrainingAdapter({})
    model = adapter.train(_spec(estimator="logistic"), _data(), _ctx())
    store = TempArtifactStore(tmp_path)
    with pytest.raises(ValueError, match="unsupported export format"):
        adapter.export(model, "onnx", store)


def test_load_refuses_a_foreign_artifact_format(tmp_path: Path) -> None:
    adapter = SklearnTrainingAdapter({})
    store = TempArtifactStore(tmp_path)
    ref = ArtifactRef(
        uri="file:///nope", format="lightgbm_json", size_bytes=1, content_hash="sha256:x"
    )
    with pytest.raises(ValueError, match="cannot load artifact format"):
        adapter.load(ref, store)


def test_export_load_round_trips_the_one_hot_owners(tmp_path: Path) -> None:
    """column_owners must survive the artifact, or importance after a reload
    reports per encoded column instead of per feature."""
    adapter = SklearnTrainingAdapter({})
    data = _data()
    model = adapter.train(_spec(estimator="logistic"), data, _ctx())
    store = TempArtifactStore(tmp_path)
    ref = adapter.export(model, "native", store)
    assert ref.format == ARTIFACT_FORMAT

    restored = adapter.load(ref, store)
    assert restored.column_owners == model.column_owners
    assert adapter.feature_importance(restored) == adapter.feature_importance(model)
    original = adapter.predict(model, data, "test").column("prediction").to_pylist()
    assert adapter.predict(restored, data, "test").column("prediction").to_pylist() == original


def test_plugin_descriptor_wires_the_training_adapter() -> None:
    from mbt_sklearn.plugin import PLUGIN

    from mbt_adapter_base import CONTRACT_VERSION

    assert PLUGIN.name == "sklearn"
    assert PLUGIN.contract_version == CONTRACT_VERSION
    assert PLUGIN.training is SklearnTrainingAdapter
    # The fingerprint is the distribution name, not the import name: the
    # env_digest resolves it through importlib.metadata (ADR-19).
    assert PLUGIN.fingerprint_packages == ["scikit-learn"]
