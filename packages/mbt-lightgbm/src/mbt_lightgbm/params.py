"""LightGBM hyperparameter models (import-light, ADR-14)."""

from pydantic import BaseModel, ConfigDict, Field


class _LightGBMCommonParams(BaseModel):
    """Hyperparameters shared by every LightGBM task (extra='forbid')."""

    model_config = ConfigDict(extra="forbid")

    n_estimators: int = Field(default=100, ge=1)
    num_leaves: int = Field(default=31, ge=2)
    max_depth: int = Field(default=-1)  # -1 = no limit (LightGBM convention)
    learning_rate: float = Field(default=0.1, gt=0)
    min_child_samples: int = Field(default=20, ge=1)
    subsample: float = Field(default=1.0, gt=0, le=1)
    colsample_bytree: float = Field(default=1.0, gt=0, le=1)
    reg_alpha: float = Field(default=0.0, ge=0)
    reg_lambda: float = Field(default=0.0, ge=0)
    num_threads: int = Field(default=1, ge=1)  # >1 breaks the exact tier
    # Needs a validation split; mirrors the xgboost contract. model_to_string
    # persists only the best iteration, so export/load stays consistent.
    early_stopping_rounds: int | None = Field(default=None, ge=1)

    def _common_booster_params(self, seed: int) -> dict[str, object]:
        # early_stopping_rounds is train-loop plumbing, not a booster param.
        params: dict[str, object] = {
            "num_leaves": self.num_leaves,
            "max_depth": self.max_depth,
            "learning_rate": self.learning_rate,
            "min_child_samples": self.min_child_samples,
            "bagging_fraction": self.subsample,
            "feature_fraction": self.colsample_bytree,
            "lambda_l1": self.reg_alpha,
            "lambda_l2": self.reg_lambda,
            "seed": seed,
            "num_threads": self.num_threads,
            "deterministic": True,
            "force_row_wise": True,
            "verbosity": -1,
        }
        if self.subsample < 1.0:
            # bagging_fraction is a no-op unless bagging_freq > 0.
            params["bagging_freq"] = 1
        return params


class LightGBMBinaryParams(_LightGBMCommonParams):
    """Static hyperparameters for binary classification."""

    scale_pos_weight: float | None = None  # '{{ auto }}' resolves from the profile

    def booster_params(self, seed: int) -> dict[str, object]:
        params: dict[str, object] = {
            "objective": "binary",
            "metric": "binary_logloss",
            **self._common_booster_params(seed),
        }
        if self.scale_pos_weight is not None:
            params["scale_pos_weight"] = self.scale_pos_weight
        return params


class LightGBMRegressionParams(_LightGBMCommonParams):
    """Static hyperparameters for regression (L2 objective, no class weight)."""

    def booster_params(self, seed: int) -> dict[str, object]:
        return {
            "objective": "regression",
            "metric": "rmse",
            **self._common_booster_params(seed),
        }
