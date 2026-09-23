"""Native categorical support for LightGBM (G4).

The four cases live in ``CategoricalAdapterCompliance``: they were duplicated
here and in ``test_xgboost_categorical.py``, which differed by six lines and
shared all four test function names (A-4).
"""

from typing import Any, ClassVar

from mbt_lightgbm.adapter import LightGBMTrainingAdapter

from mbt_adapter_base.compliance import CategoricalAdapterCompliance


class TestLightGBMCategoricalCompliance(CategoricalAdapterCompliance):
    adapter_factory: ClassVar[Any] = LightGBMTrainingAdapter
    hyperparameters: ClassVar[dict[str, Any]] = {
        "n_estimators": 40,
        "num_leaves": 15,
        "min_child_samples": 5,
    }
    categories_home = "the artifact envelope"
