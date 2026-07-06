"""mbt-xgboost against the adapter compliance suite (S7-07, FR-ADPT-05)."""

from typing import ClassVar

from mbt_xgboost.adapter import XGBoostTrainingAdapter

from mbt_adapter_base.compliance import TrainingAdapterCompliance


class TestXGBoostCompliance(TrainingAdapterCompliance):
    adapter_factory = XGBoostTrainingAdapter
    plugin_module = "mbt_xgboost.plugin"
    framework_modules = ("xgboost",)
    valid_hyperparameters: ClassVar[dict] = {
        "max_depth": 3,
        "n_estimators": 30,
        "learning_rate": 0.2,
    }
    auto_hyperparameter = "scale_pos_weight"
