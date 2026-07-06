"""Adapter-local unit tests beyond the compliance suite (TSD §13.1)."""

import pytest
from mbt_xgboost.adapter import XGBoostTrainingAdapter
from mbt_xgboost.params import XGBoostBinaryParams

from mbt_adapter_base import AUTO, TaskType
from mbt_adapter_base.compliance import tiny_binary_dataset
from mbt_adapter_base.compliance.suite import _BINARY_METRICS


def _spec(**overrides):
    from mbt_adapter_base import EvaluationProtocol, EvaluationSpec, ModelSpec

    base = {
        "name": "m",
        "task": TaskType.BINARY_CLASSIFICATION,
        "adapter": "xgboost",
        "owner": "t@example.com",
        "dataset": "ref('d')",
        "target": "label",
        "evaluation": EvaluationSpec(
            protocol=EvaluationProtocol(), metrics=["roc_auc", "pr_auc", "logloss"]
        ),
        "seed": 5,
    }
    base.update(overrides)
    return ModelSpec.model_validate(base)


def test_auto_scale_pos_weight_from_profile() -> None:
    adapter = XGBoostTrainingAdapter({})
    data = tiny_binary_dataset()
    spec = _spec(hyperparameters={"scale_pos_weight": AUTO, "n_estimators": 10})
    resolved = adapter.resolve_auto(spec, data.profile())
    value = resolved.hyperparameters["scale_pos_weight"]
    balance = data.profile().label_balance
    assert balance is not None
    expected = (1 - balance["1"]) / balance["1"]
    assert value == pytest.approx(expected, rel=1e-4)


def test_non_numeric_feature_raises_actionable_error() -> None:
    import pyarrow as pa

    from mbt_adapter_base.datasets import InMemoryDatasetHandle

    table = pa.table({"f": [1.0, 2.0], "s": ["a", "b"], "label": [0, 1]})
    data = InMemoryDatasetHandle({"train": table, "test": table}, label_column="label")
    adapter = XGBoostTrainingAdapter({})
    with pytest.raises(ValueError, match=r"non-numeric.*s.*hooks"):
        adapter.train(_spec(hyperparameters={"n_estimators": 5}), data, _ctx())


def _ctx():
    from mbt_adapter_base import RunContext

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


def test_nondeterminism_warnings_for_non_hist_and_cuda() -> None:
    adapter = XGBoostTrainingAdapter({})
    assert adapter.nondeterminism_warnings(_spec(hyperparameters={"tree_method": "approx"})), (
        "non-hist tree_method must warn (FR-RUN-06)"
    )
    assert adapter.nondeterminism_warnings(_spec(hyperparameters={"device": "cuda"}))
    assert not adapter.nondeterminism_warnings(_spec(hyperparameters={"max_depth": 3}))


def test_slice_metrics_grouped_by_column() -> None:
    adapter = XGBoostTrainingAdapter({})
    data = tiny_binary_dataset()
    spec = _spec(
        hyperparameters={"n_estimators": 20},
        evaluation={
            "protocol": {"split": "temporal"},
            "metrics": ["roc_auc"],
            "slices": ["f_binary"],
        },
    )
    model = adapter.train(spec, data, _ctx())
    results = adapter.evaluate(model, data, "test", _BINARY_METRICS, slices=["f_binary"])
    assert any(key.startswith("f_binary=") for key in results.slices)


def test_param_defaults_are_deterministic_config() -> None:
    params = XGBoostBinaryParams()
    booster = params.booster_params(seed=7)
    assert booster["nthread"] == 1
    assert booster["tree_method"] == "hist"
    assert booster["seed"] == 7
