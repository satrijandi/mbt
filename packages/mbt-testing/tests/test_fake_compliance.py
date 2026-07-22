"""The fake training adapter against the shared adapter compliance suite.

The fake adapter backs nearly the entire mbt-core fast suite, yet it was never
held to the same contract the real adapters must pass - so a drift between the
fake and production adapters (one such 4dp-vs-6dp scale_pos_weight drift already
shipped) would keep the core tests green and surface only in the opt-in JVM
tier. Running the fake through TrainingAdapterCompliance closes that gap:
export/load round-trip, predict-appends-column, predict-unlabeled,
resolve_auto idempotence (the scale_pos_weight precision guard), seed
determinism, and feature-importance normalization all apply as-is.
"""

from typing import ClassVar

import pytest
from mbt_testing.adapters import FakeTrainingAdapter

from mbt_adapter_base.compliance import TrainingAdapterCompliance


class TestFakeAdapterCompliance(TrainingAdapterCompliance):
    adapter_factory = FakeTrainingAdapter
    plugin_module = "mbt_testing.plugin"
    framework_modules = ()  # the fake pulls no ML framework
    valid_hyperparameters: ClassVar[dict] = {"max_depth": 3, "learning_rate": 0.2}
    auto_hyperparameter = "scale_pos_weight"

    def test_model_actually_learns(self) -> None:
        # The fake's evaluate() returns scripted metrics (fake_metric_value) so
        # gate scenarios are controllable; it does not learn from data, so the
        # "beats coin-flip" assertion does not apply. Its genuine signal-bearing
        # behavior lives in predict() (label separation) and is exercised by the
        # paired-bootstrap tests in mbt-core.
        pytest.skip("fake adapter reports scripted metrics, not learned ones")

    def test_learns_from_a_categorical_feature(self) -> None:
        # Same reason as test_model_actually_learns: the fake reports scripted
        # metrics, so "learns from the categorical" does not apply. The fake does
        # train on the string feature without error (it reads the table); the
        # learning assertion is exercised by the real tree adapters (F25).
        pytest.skip("fake adapter reports scripted metrics, not learned ones")
