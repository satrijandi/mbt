"""Edge and error-path coverage for the xgboost adapter (unit tier).

Real training stays in the compliance/report tests; here the focus is the
defensive branches: AUTO resolution failures, invalid hyperparameters,
featureless matrices, eval_metric merging, zero-gain importance, and the
export/load format guards (ONNX via an injected fake onnxmltools).
"""

import sys
import types
from pathlib import Path
from typing import Any

import pytest
from mbt_xgboost.adapter import XGBoostModel, XGBoostTrainingAdapter
from mbt_xgboost.params import XGBoostBinaryParams

from mbt_adapter_base import (
    AUTO,
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


def _spec(**overrides: Any) -> ModelSpec:
    base: dict[str, Any] = {
        "name": "m",
        "task": TaskType.BINARY_CLASSIFICATION,
        "adapter": "xgboost",
        "owner": "t@example.com",
        "dataset": "ref('d')",
        "target": "label",
        "evaluation": EvaluationSpec(protocol=EvaluationProtocol(), metrics=["roc_auc"]),
        "seed": 5,
    }
    base.update(overrides)
    return ModelSpec.model_validate(base)


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


def _profile(label_balance: dict[str, float] | None) -> DatasetProfile:
    return DatasetProfile(
        n_rows={"train": 10, "test": 5},
        columns={"f": "double", "label": "int64"},
        label_column="label",
        label_balance=label_balance,
    )


def _tuning_handle() -> InMemoryDatasetHandle:
    base = tiny_binary_dataset()
    val = base.read("test")
    return InMemoryDatasetHandle(
        {"train": base.read("train"), "validation": val, "test": val}, label_column="label"
    )


# -- validation and AUTO resolution -------------------------------------------


def test_validate_surfaces_nondeterminism_warnings_as_issues() -> None:
    adapter = XGBoostTrainingAdapter({})
    issues = adapter.validate(_spec(hyperparameters={"tree_method": "approx"}))
    assert issues and all(i.severity == "warning" for i in issues)
    assert issues[0].field_path == "/hyperparameters"
    assert not adapter.validate(_spec(hyperparameters={"max_depth": 3}))


def test_auto_scale_pos_weight_needs_a_positive_class_balance() -> None:
    adapter = XGBoostTrainingAdapter({})
    spec = _spec(hyperparameters={"scale_pos_weight": AUTO})
    with pytest.raises(ValueError, match="no positive-class balance"):
        adapter.resolve_auto(spec, _profile({"0": 1.0}))  # no positive key at all
    with pytest.raises(ValueError, match="no positive-class balance"):
        adapter.resolve_auto(spec, _profile(None))


def test_auto_rejects_unsupported_hyperparameters() -> None:
    adapter = XGBoostTrainingAdapter({})
    spec = _spec(hyperparameters={"max_depth": AUTO})
    with pytest.raises(ValueError, match="only scale_pos_weight"):
        adapter.resolve_auto(spec, _profile({"1": 0.5, "0": 0.5}))


def test_invalid_hyperparameters_fail_with_actionable_error() -> None:
    adapter = XGBoostTrainingAdapter({})
    spec = _spec(hyperparameters={"not_a_real_knob": 1})
    with pytest.raises(ValueError, match="invalid xgboost hyperparameters"):
        adapter.train(spec, tiny_binary_dataset(), _ctx())


# -- data plumbing --------------------------------------------------------------


def test_matrix_with_no_features_builds_an_empty_dmatrix() -> None:
    import pyarrow as pa

    table = pa.table({"label": [0, 1, 1]})
    matrix, y = XGBoostTrainingAdapter({})._matrix(table, [], {}, None)
    assert matrix.num_row() == 3
    assert matrix.num_col() == 0
    assert y is None


# -- eval_metric merging in the report path --------------------------------------


def _train_reporting(monkeypatch: pytest.MonkeyPatch, mutate: Any) -> list[tuple[int, float]]:
    original = XGBoostBinaryParams.booster_params

    def patched(self: XGBoostBinaryParams, seed: int, positive_rate_default: float = 1.0) -> dict:
        params = original(self, seed, positive_rate_default)
        mutate(params)
        return params

    monkeypatch.setattr(XGBoostBinaryParams, "booster_params", patched)
    seen: list[tuple[int, float]] = []
    XGBoostTrainingAdapter({}).train_with_report(
        _spec(hyperparameters={"n_estimators": 5, "max_depth": 2}),
        _tuning_handle(),
        _ctx(),
        lambda step, value: seen.append((step, value)),
    )
    return seen


def test_report_appends_auc_to_an_eval_metric_list(monkeypatch: pytest.MonkeyPatch) -> None:
    seen = _train_reporting(monkeypatch, lambda p: p.__setitem__("eval_metric", ["logloss"]))
    assert [step for step, _ in seen] == list(range(5))
    assert all(0.0 <= value <= 1.0 for _, value in seen)


def test_report_adds_auc_when_no_eval_metric_is_configured(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    seen = _train_reporting(monkeypatch, lambda p: p.pop("eval_metric"))
    assert [step for step, _ in seen] == list(range(5))


# -- feature importance -----------------------------------------------------------


class _StubBooster:
    def __init__(self, gains: dict[str, Any]) -> None:
        self._gains = gains

    def get_score(self, importance_type: str) -> dict[str, Any]:
        assert importance_type == "gain"
        return self._gains


def test_feature_importance_with_zero_total_gain_is_all_zeros() -> None:
    model = XGBoostModel(booster=_StubBooster({}), features=["a", "b"], target="label")
    assert XGBoostTrainingAdapter({}).feature_importance(model) == {"a": 0.0, "b": 0.0}


def test_feature_importance_unwraps_per_class_lists() -> None:
    booster = _StubBooster({"a": [1.0], "b": [3.0]})
    model = XGBoostModel(booster=booster, features=["a", "b"], target="label")
    importance = XGBoostTrainingAdapter({}).feature_importance(model)
    assert importance == {"a": 0.25, "b": 0.75}


# -- export / load format guards ----------------------------------------------------


def _fake_onnxmltools(monkeypatch: pytest.MonkeyPatch, captured: dict[str, Any]) -> None:
    class FloatTensorType:
        def __init__(self, shape: list) -> None:
            self.shape = shape

    class _OnnxModel:
        def SerializeToString(self) -> bytes:  # onnx API casing
            return b"fake-onnx-bytes"

    def convert_xgboost(booster: Any, initial_types: list) -> _OnnxModel:
        captured["booster"] = booster
        captured["initial_types"] = initial_types
        return _OnnxModel()

    root = types.ModuleType("onnxmltools")
    convert = types.ModuleType("onnxmltools.convert")
    common = types.ModuleType("onnxmltools.convert.common")
    data_types = types.ModuleType("onnxmltools.convert.common.data_types")
    data_types.FloatTensorType = FloatTensorType  # type: ignore[attr-defined]
    root.convert_xgboost = convert_xgboost  # type: ignore[attr-defined]
    root.convert = convert  # type: ignore[attr-defined]
    convert.common = common  # type: ignore[attr-defined]
    common.data_types = data_types  # type: ignore[attr-defined]
    for name, module in {
        "onnxmltools": root,
        "onnxmltools.convert": convert,
        "onnxmltools.convert.common": common,
        "onnxmltools.convert.common.data_types": data_types,
    }.items():
        monkeypatch.setitem(sys.modules, name, module)


def test_export_onnx_converts_and_stores(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    captured: dict[str, Any] = {}
    _fake_onnxmltools(monkeypatch, captured)
    booster = object()
    model = XGBoostModel(booster=booster, features=["a", "b"], target="label")  # type: ignore[arg-type]
    ref = XGBoostTrainingAdapter({}).export(model, "onnx", TempArtifactStore(tmp_path))
    assert ref.format == "onnx"
    assert captured["booster"] is booster
    ((name, tensor),) = captured["initial_types"]
    assert name == "input" and tensor.shape == [None, 2]
    assert (tmp_path / "model.onnx").read_bytes() == b"fake-onnx-bytes"


def test_export_onnx_rejects_categorical_models(tmp_path: Path) -> None:
    model = XGBoostModel(
        booster=object(),  # type: ignore[arg-type]
        features=["plan"],
        target="label",
        categories={"plan": ["a", "b"]},
    )
    with pytest.raises(ValueError, match="does not support categorical"):
        XGBoostTrainingAdapter({}).export(model, "onnx", TempArtifactStore(tmp_path))


def test_export_rejects_unknown_formats(tmp_path: Path) -> None:
    model = XGBoostModel(booster=object(), features=["a"], target="label")  # type: ignore[arg-type]
    with pytest.raises(ValueError, match="unsupported export format 'coreml'"):
        XGBoostTrainingAdapter({}).export(model, "coreml", TempArtifactStore(tmp_path))


def test_load_rejects_foreign_artifact_formats(tmp_path: Path) -> None:
    ref = ArtifactRef(
        uri="file:///x", format="lightgbm_json", content_hash="sha256:0", size_bytes=1
    )
    with pytest.raises(ValueError, match="cannot load artifact format 'lightgbm_json'"):
        XGBoostTrainingAdapter({}).load(ref, TempArtifactStore(tmp_path))
