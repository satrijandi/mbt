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
    TaskType,
)
from mbt_adapter_base.compliance import tiny_binary_dataset
from mbt_adapter_base.compliance.suite import TempArtifactStore


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
    with pytest.raises(ValueError, match="without a positive-class balance"):
        adapter.resolve_auto(spec, _profile({"0": 1.0}))
    with pytest.raises(ValueError, match="without a positive-class balance"):
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


def test_plugin_descriptor_wires_the_training_adapter() -> None:
    from mbt_lightgbm.plugin import PLUGIN

    assert PLUGIN.name == "lightgbm"
    assert PLUGIN.contract_version == CONTRACT_VERSION
    assert PLUGIN.training is LightGBMTrainingAdapter
    assert PLUGIN.fingerprint_packages == ["lightgbm"]
