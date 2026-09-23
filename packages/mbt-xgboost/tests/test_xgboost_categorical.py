"""Native categorical support for XGBoost (FR-ADPT-03).

The four cases live in ``CategoricalAdapterCompliance``: they were duplicated
here and in ``test_lightgbm_categorical.py``, which differed by six lines and
shared all four test function names (A-4).
"""

from typing import Any, ClassVar

from mbt_xgboost.adapter import XGBoostTrainingAdapter

from mbt_adapter_base.compliance import CategoricalAdapterCompliance


class TestXGBoostCategoricalCompliance(CategoricalAdapterCompliance):
    adapter_factory: ClassVar[Any] = XGBoostTrainingAdapter
    hyperparameters: ClassVar[dict[str, Any]] = {"n_estimators": 30, "max_depth": 3}
    categories_home = "the booster attrs"
