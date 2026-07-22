"""Native categorical support: train, persist levels, survive load (G4)."""

import tempfile
from pathlib import Path

import pyarrow as pa
import pytest
from mbt_lightgbm.adapter import LightGBMTrainingAdapter

from mbt_adapter_base import EvaluationProtocol, EvaluationSpec, ModelSpec, RunContext, TaskType
from mbt_adapter_base.compliance.suite import TempArtifactStore
from mbt_adapter_base.datasets import InMemoryDatasetHandle
from mbt_adapter_base.specs import MetricSpec


def _spec() -> ModelSpec:
    return ModelSpec.model_validate(
        {
            "name": "m",
            "task": TaskType.BINARY_CLASSIFICATION,
            "adapter": "lightgbm",
            "owner": "t@example.com",
            "dataset": "ref('d')",
            "target": "label",
            "hyperparameters": {"n_estimators": 40, "num_leaves": 15, "min_child_samples": 5},
            "evaluation": EvaluationSpec(
                protocol=EvaluationProtocol(), metrics=["roc_auc", "pr_auc", "logloss"]
            ),
            "seed": 5,
        }
    )


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


def _categorical_handle(plans: list[str] | None = None) -> InMemoryDatasetHandle:
    """The signal lives almost entirely in the categorical column."""
    n = 400
    levels = plans or ["basic", "pro", "enterprise"]
    plan = [levels[i % len(levels)] for i in range(n)]
    noise = [((i * 37) % 100) / 100.0 for i in range(n)]
    label = [1 if (p == "enterprise") != (noise[i] > 0.9) else 0 for i, p in enumerate(plan)]
    table = pa.table({"noise": noise, "plan": plan, "label": label})
    return InMemoryDatasetHandle({"train": table, "test": table}, label_column="label")


def test_categorical_feature_carries_the_signal() -> None:
    adapter = LightGBMTrainingAdapter({})
    data = _categorical_handle()
    model = adapter.train(_spec(), data, _ctx())
    assert model.categories == {"plan": ["basic", "enterprise", "pro"]}  # sorted levels
    results = adapter.evaluate(
        model, data, "test", [MetricSpec(name="roc_auc", kind="builtin")], slices=None
    )
    assert results.metrics["roc_auc"] > 0.85  # only the categorical explains this

    importance = adapter.feature_importance(model)
    assert set(importance) == {"noise", "plan"}
    assert sum(importance.values()) == pytest.approx(1.0, abs=1e-3)
    assert importance["plan"] > 0.5  # the categorical dominates, as constructed


def test_shap_importance_is_normalized_and_signal_dominant() -> None:
    """The model card prefers mean-|SHAP| importance (additive, not
    cardinality-biased) over split-gain when eval data is available."""
    adapter = LightGBMTrainingAdapter({})
    data = _categorical_handle()
    model = adapter.train(_spec(), data, _ctx())

    shap = adapter.shap_importance(model, data, "test")
    assert set(shap) == {"noise", "plan"}
    assert all(value >= 0 for value in shap.values())  # mean |SHAP| is non-negative
    assert sum(shap.values()) == pytest.approx(1.0, abs=1e-3)  # normalized to fractions
    assert shap["plan"] > 0.5  # SHAP agrees the categorical carries the signal


def test_explain_gives_per_row_top_k_contributors() -> None:
    """Local attribution: each row's top_k features by |SHAP|, ordered, as JSON."""
    import json

    adapter = LightGBMTrainingAdapter({})
    data = _categorical_handle()
    model = adapter.train(_spec(), data, _ctx())

    rows = adapter.explain(model, data, "test", top_k=2)
    assert len(rows) == data.read("test").num_rows
    top = json.loads(rows[0])
    assert len(top) == 2 and all(feature in model.features for feature, _ in top)
    assert abs(top[0][1]) >= abs(top[1][1])  # ordered by descending |contribution|


def test_categories_survive_export_load_and_unseen_levels_predict() -> None:
    adapter = LightGBMTrainingAdapter({})
    data = _categorical_handle()
    model = adapter.train(_spec(), data, _ctx())
    scores = adapter.predict(model, data, "test").column("prediction").to_pylist()

    with tempfile.TemporaryDirectory() as tmp:
        store = TempArtifactStore(Path(tmp))
        ref = adapter.export(model, "native", store)
        loaded = adapter.load(ref, store)
    assert loaded.categories == model.categories  # persisted in the artifact envelope
    reloaded_scores = adapter.predict(loaded, data, "test").column("prediction").to_pylist()
    assert reloaded_scores == scores  # champion path scores identically

    # a level unseen at train time maps to missing, never crashes
    unseen = _categorical_handle(plans=["basic", "pro", "enterprise", "trial"])
    predictions = adapter.predict(loaded, unseen, "test")
    assert predictions.num_rows == 400
