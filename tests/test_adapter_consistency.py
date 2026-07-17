"""Cross-adapter consistency invariants.

The compliance suite checks each adapter independently against absolute
thresholds; it never checks that adapters AGREE. The recurring, hand-caught
bug class is exactly disagreement - a LightGBM subsample no-op trained a
materially different model than XGBoost from the same spec, and a fake adapter
once rounded auto scale_pos_weight to 4dp while the real adapters used 6dp.

This pins the one auto-resolution every classification adapter shares:
``{{ auto }}`` scale_pos_weight must resolve to the identical value (to the last
decimal) across XGBoost, LightGBM, and the fake, and match the shared
``resolve_scale_pos_weight`` helper. A drift in any single adapter fails here in
the fast tier instead of surfacing as silently divergent models.
"""

from typing import Any

from mbt_lightgbm.adapter import LightGBMTrainingAdapter
from mbt_testing.adapters import FakeTrainingAdapter
from mbt_xgboost.adapter import XGBoostTrainingAdapter

from mbt_adapter_base import (
    AUTO,
    EvaluationProtocol,
    EvaluationSpec,
    ModelSpec,
    TaskType,
)
from mbt_adapter_base.compliance.suite import tiny_binary_dataset
from mbt_adapter_base.training_helpers import resolve_scale_pos_weight

_ADAPTERS = {
    "xgboost": XGBoostTrainingAdapter,
    "lightgbm": LightGBMTrainingAdapter,
    "fake": FakeTrainingAdapter,
}


def _auto_spec(adapter_name: str) -> ModelSpec:
    return ModelSpec(
        name="parity_model",
        task=TaskType.BINARY_CLASSIFICATION,
        adapter=adapter_name,
        owner="consistency@mbt.dev",
        dataset="ref('parity_dataset')",
        target="label",
        hyperparameters={"scale_pos_weight": AUTO},
        evaluation=EvaluationSpec(protocol=EvaluationProtocol(), metrics=["roc_auc"]),
        seed=1,
    )


def test_auto_scale_pos_weight_is_identical_across_adapters() -> None:
    profile = tiny_binary_dataset().profile()
    expected = resolve_scale_pos_weight(profile)

    resolved: dict[str, Any] = {}
    for name, adapter_cls in _ADAPTERS.items():
        out = adapter_cls({}).resolve_auto(_auto_spec(name), profile)
        resolved[name] = out.hyperparameters["scale_pos_weight"]

    # exact equality (not approx): the whole point is that every adapter rounds
    # the same way to the same precision.
    for name, value in resolved.items():
        assert value == expected, f"{name} resolved {value!r}, shared helper gives {expected!r}"
    assert len(set(resolved.values())) == 1, f"adapters disagree: {resolved}"
