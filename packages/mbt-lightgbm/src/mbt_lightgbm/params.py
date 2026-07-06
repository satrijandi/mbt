"""LightGBM hyperparameter models (import-light, ADR-14)."""

from pydantic import BaseModel, ConfigDict, Field


class LightGBMBinaryParams(BaseModel):
    """Static hyperparameters for binary classification (extra='forbid')."""

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
    scale_pos_weight: float | None = None  # '{{ auto }}' resolves from the profile
    num_threads: int = Field(default=1, ge=1)  # >1 breaks the exact tier

    def booster_params(self, seed: int) -> dict[str, object]:
        params: dict[str, object] = {
            "objective": "binary",
            "metric": "binary_logloss",
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
        if self.scale_pos_weight is not None:
            params["scale_pos_weight"] = self.scale_pos_weight
        return params
