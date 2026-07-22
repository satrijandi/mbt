"""Edge and error-path coverage for the lightgbm adapter (unit tier).

Real training stays in the compliance/report tests; here the focus is the
defensive branches: validate() wrapping, AUTO resolution failures, invalid
hyperparameters, featureless matrices, zero-gain importance, the artifact
format guards, params plumbing, and the plugin descriptor.
"""

from pathlib import Path
from typing import Any

import pytest
from mbt_lightgbm.adapter import LightGBMModel, LightGBMTrainingAdapter
from mbt_lightgbm.params import LightGBMBinaryParams

from mbt_adapter_base import (
    AUTO,
    CONTRACT_VERSION,
    ArtifactRef,
    DatasetProfile,
    EvaluationProtocol,
    EvaluationSpec,
    ModelSpec,
    RunContext,
    TaskType,
)
from mbt_adapter_base.compliance import tiny_binary_dataset
from mbt_adapter_base.compliance.suite import TempArtifactStore
from mbt_adapter_base.datasets import InMemoryDatasetHandle


def _ctx() -> RunContext:
    class _Null:
        def emit(self, event: object) -> None: ...

    return RunContext(
        run_id="t",
        unique_id="m",
        seed=5,
        target_name="dev",
        project_dir=".",
        vars={},
        events=_Null(),
    )


def _spec(**overrides: Any) -> ModelSpec:
    base: dict[str, Any] = {
        "name": "m",
        "task": TaskType.BINARY_CLASSIFICATION,
        "adapter": "lightgbm",
        "owner": "t@example.com",
        "dataset": "ref('d')",
        "target": "label",
        "evaluation": EvaluationSpec(protocol=EvaluationProtocol(), metrics=["roc_auc"]),
        "seed": 5,
    }
    base.update(overrides)
    return ModelSpec.model_validate(base)


def _profile(label_balance: dict[str, float] | None) -> DatasetProfile:
    return DatasetProfile(
        n_rows={"train": 10, "test": 5},
        columns={"f": "double", "label": "int64"},
        label_column="label",
        label_balance=label_balance,
    )


def test_validate_wraps_threading_warnings_as_issues() -> None:
    adapter = LightGBMTrainingAdapter({})
    issues = adapter.validate(_spec(hyperparameters={"num_threads": 4}))
    assert issues and issues[0].severity == "warning"
    assert issues[0].field_path == "/hyperparameters/num_threads"
    assert not adapter.validate(_spec(hyperparameters={"num_leaves": 7}))


def test_auto_scale_pos_weight_needs_a_positive_class_balance() -> None:
    adapter = LightGBMTrainingAdapter({})
    spec = _spec(hyperparameters={"scale_pos_weight": AUTO})
    with pytest.raises(ValueError, match="no positive-class balance"):
        adapter.resolve_auto(spec, _profile({"0": 1.0}))
    with pytest.raises(ValueError, match="no positive-class balance"):
        adapter.resolve_auto(spec, _profile(None))


def test_auto_rejects_unsupported_hyperparameters() -> None:
    adapter = LightGBMTrainingAdapter({})
    spec = _spec(hyperparameters={"num_leaves": AUTO})
    with pytest.raises(ValueError, match="only scale_pos_weight"):
        adapter.resolve_auto(spec, _profile({"1": 0.5, "0": 0.5}))


def test_invalid_hyperparameters_fail_with_actionable_error() -> None:
    adapter = LightGBMTrainingAdapter({})
    spec = _spec(hyperparameters={"not_a_real_knob": 1})
    with pytest.raises(ValueError, match="invalid lightgbm hyperparameters"):
        adapter.train(spec, tiny_binary_dataset(), None)  # type: ignore[arg-type]


def test_features_matrix_with_no_features_is_empty() -> None:
    import pyarrow as pa

    table = pa.table({"label": [0, 1, 1]})
    matrix = LightGBMTrainingAdapter({})._features_matrix(table, [], {})
    assert matrix.shape == (3, 0)


class _StubBooster:
    def __init__(self, gains: dict[str, float]) -> None:
        self._gains = gains

    def feature_importance(self, importance_type: str) -> list[float]:
        assert importance_type == "gain"
        return list(self._gains.values())

    def feature_name(self) -> list[str]:
        return list(self._gains)


def test_feature_importance_with_zero_total_gain_is_all_zeros() -> None:
    model = LightGBMModel(
        booster=_StubBooster({"a": 0.0, "b": 0.0}),  # type: ignore[arg-type]
        features=["a", "b"],
        target="label",
    )
    assert LightGBMTrainingAdapter({}).feature_importance(model) == {"a": 0.0, "b": 0.0}


def test_export_rejects_unknown_formats(tmp_path: Path) -> None:
    model = LightGBMModel(booster=object(), features=["a"], target="label")  # type: ignore[arg-type]
    with pytest.raises(ValueError, match="unsupported export format 'onnx'"):
        LightGBMTrainingAdapter({}).export(model, "onnx", TempArtifactStore(tmp_path))


def test_load_rejects_foreign_artifact_formats(tmp_path: Path) -> None:
    ref = ArtifactRef(uri="file:///x", format="xgboost_ubj", content_hash="sha256:0", size_bytes=1)
    with pytest.raises(ValueError, match="cannot load artifact format 'xgboost_ubj'"):
        LightGBMTrainingAdapter({}).load(ref, TempArtifactStore(tmp_path))


def test_booster_params_pass_scale_pos_weight_through() -> None:
    params = LightGBMBinaryParams(scale_pos_weight=3.5).booster_params(seed=7)
    assert params["scale_pos_weight"] == 3.5
    assert "scale_pos_weight" not in LightGBMBinaryParams().booster_params(seed=7)


def test_booster_params_enable_bagging_when_subsampling() -> None:
    # bagging_fraction is inert in LightGBM unless bagging_freq > 0.
    sampled = LightGBMBinaryParams(subsample=0.6).booster_params(seed=7)
    assert sampled["bagging_fraction"] == 0.6
    assert sampled["bagging_freq"] == 1
    assert "bagging_freq" not in LightGBMBinaryParams().booster_params(seed=7)


def test_early_stopping_stops_before_the_round_budget(tmp_path: Path) -> None:
    # Mirrors the xgboost contract: early_stopping_rounds needs a validation
    # split. LightGBM scores at the best iteration natively, and
    # model_to_string persists only the best iteration, so an export/load
    # round trip scores exactly like the in-memory model.
    base = tiny_binary_dataset()
    train, val = base.read("train"), base.read("test")
    data = InMemoryDatasetHandle(
        {"train": train, "validation": val, "test": val}, label_column="label"
    )
    adapter = LightGBMTrainingAdapter({})
    hp = {
        "n_estimators": 300,
        "num_leaves": 15,
        "min_child_samples": 5,
        "early_stopping_rounds": 5,
    }
    model = adapter.train(_spec(hyperparameters=hp), data, _ctx())
    best = model.booster.best_iteration
    assert 0 < best < 300  # early stopping actually fired
    x = adapter._features_matrix(val, model.features, model.categories)
    best_slice = model.booster.predict(x, num_iteration=best)
    scores = adapter.predict(model, data, "test").column("prediction").to_pylist()
    assert scores == pytest.approx(best_slice)  # predict honors best_iteration
    loaded = adapter.load(
        adapter.export(model, "native", TempArtifactStore(tmp_path)), TempArtifactStore(tmp_path)
    )
    reloaded_scores = adapter.predict(loaded, data, "test").column("prediction").to_pylist()
    assert reloaded_scores == pytest.approx(scores)


def test_subsample_changes_the_trained_model() -> None:
    adapter = LightGBMTrainingAdapter({})
    data = tiny_binary_dataset()
    hp = {"n_estimators": 20, "num_leaves": 15, "min_child_samples": 5}
    full = adapter.train(_spec(hyperparameters=hp), data, _ctx())
    sampled = adapter.train(_spec(hyperparameters={**hp, "subsample": 0.6}), data, _ctx())
    full_scores = adapter.predict(full, data, "test").column("prediction").to_pylist()
    sampled_scores = adapter.predict(sampled, data, "test").column("prediction").to_pylist()
    assert full_scores != sampled_scores


def test_plugin_descriptor_wires_the_training_adapter() -> None:
    from mbt_lightgbm.plugin import PLUGIN

    assert PLUGIN.name == "lightgbm"
    assert PLUGIN.contract_version == CONTRACT_VERSION
    assert PLUGIN.training is LightGBMTrainingAdapter
    assert PLUGIN.fingerprint_packages == ["lightgbm"]


# -- post-hoc calibration (R2-8) ------------------------------------------------------


def _synthetic(n: int, seed: int):
    """A binary problem with base rate ~0.25, so scale_pos_weight inflates the
    predicted probabilities and leaves the model measurably miscalibrated."""
    import numpy as np
    import pyarrow as pa

    rng = np.random.default_rng(seed)
    x = rng.normal(size=n)
    p = 1.0 / (1.0 + np.exp(-(1.6 * x - 1.2)))
    y = (rng.uniform(size=n) < p).astype("int64")
    return pa.table({"x": x.astype("float64"), "label": y})


def _calibration_data() -> InMemoryDatasetHandle:
    return InMemoryDatasetHandle(
        {"train": _synthetic(800, 1), "validation": _synthetic(400, 2), "test": _synthetic(400, 3)},
        label_column="label",
    )


@pytest.mark.parametrize("method", ["isotonic", "sigmoid"])
def test_calibration_reduces_ece_without_hurting_ranking(method: str) -> None:
    """scale_pos_weight inflates the probabilities; a calibrator fit on validation
    pulls ece down, and being a monotonic transform it preserves roc_auc (R2-8)."""
    from mbt_adapter_base import MetricSpec

    adapter = LightGBMTrainingAdapter({})
    data = _calibration_data()
    hp = {"n_estimators": 40, "max_depth": 3, "scale_pos_weight": 6}
    metrics = [MetricSpec(name="ece"), MetricSpec(name="roc_auc")]

    raw = adapter.evaluate(
        adapter.train(_spec(hyperparameters=hp), data, _ctx()), data, "test", metrics
    )
    cal_model = adapter.train(_spec(hyperparameters=hp, calibration=method), data, _ctx())
    cal = adapter.evaluate(cal_model, data, "test", metrics)

    assert cal_model.calibrator is not None and cal_model.calibrator.method == method
    assert cal.metrics["ece"] < raw.metrics["ece"]  # calibration fixed the inflation
    assert cal.metrics["roc_auc"] == pytest.approx(raw.metrics["roc_auc"], abs=0.02)


def test_calibrator_survives_save_load(tmp_path: Path) -> None:
    """The calibrator rides in the lightgbm artifact payload, so a reloaded
    champion produces identical calibrated scores (parity across save/load)."""
    import numpy as np

    adapter = LightGBMTrainingAdapter({})
    data = _calibration_data()
    model = adapter.train(
        _spec(hyperparameters={"n_estimators": 20}, calibration="sigmoid"), data, _ctx()
    )
    store = TempArtifactStore(tmp_path)
    loaded = adapter.load(adapter.export(model, "native", store), store)

    assert loaded.calibrator is not None and loaded.calibrator.method == "sigmoid"
    test = data.read("test")
    np.testing.assert_allclose(adapter._scores(loaded, test), adapter._scores(model, test))


def test_calibration_requires_a_holdout_split() -> None:
    # neither a carved 'calibration' slice nor a 'validation' fallback present
    adapter = LightGBMTrainingAdapter({})
    data = InMemoryDatasetHandle(
        {"train": _synthetic(200, 1), "test": _synthetic(100, 2)}, label_column="label"
    )
    with pytest.raises(ValueError, match="validation"):
        adapter.train(
            _spec(hyperparameters={"n_estimators": 10}, calibration="isotonic"), data, _ctx()
        )
