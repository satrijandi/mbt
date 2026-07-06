"""XGBoost hyperparameter models (TSD §13.1).

Plain Pydantic, import-light (ADR-14): validating these at ``mbt parse``
time never imports xgboost itself.
"""

from typing import Literal

from pydantic import BaseModel, ConfigDict, Field


class XGBoostBinaryParams(BaseModel):
    """Static hyperparameters for binary classification (extra='forbid')."""

    model_config = ConfigDict(extra="forbid")

    n_estimators: int = Field(default=100, ge=1)  # num_boost_round
    max_depth: int = Field(default=6, ge=1)
    learning_rate: float = Field(default=0.3, gt=0)
    min_child_weight: float = Field(default=1.0, ge=0)
    subsample: float = Field(default=1.0, gt=0, le=1)
    colsample_bytree: float = Field(default=1.0, gt=0, le=1)
    gamma: float = Field(default=0.0, ge=0)
    reg_alpha: float = Field(default=0.0, ge=0)
    reg_lambda: float = Field(default=1.0, ge=0)
    scale_pos_weight: float | None = None  # '{{ auto }}' resolves from the profile
    tree_method: Literal["hist", "exact", "approx"] = "hist"
    device: Literal["cpu", "cuda"] = "cpu"
    early_stopping_rounds: int | None = Field(default=None, ge=1)

    def booster_params(self, seed: int, positive_rate_default: float = 1.0) -> dict[str, object]:
        """The xgb.train param dict (fixed nthread for determinism, TSD §13.1)."""
        return {
            "objective": "binary:logistic",
            "eval_metric": "logloss",
            "max_depth": self.max_depth,
            "eta": self.learning_rate,
            "min_child_weight": self.min_child_weight,
            "subsample": self.subsample,
            "colsample_bytree": self.colsample_bytree,
            "gamma": self.gamma,
            "alpha": self.reg_alpha,
            "lambda": self.reg_lambda,
            "scale_pos_weight": (
                self.scale_pos_weight
                if self.scale_pos_weight is not None
                else positive_rate_default
            ),
            "tree_method": self.tree_method,
            "device": self.device,
            "seed": seed,
            "nthread": 1,
        }
