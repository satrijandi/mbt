"""H2O AutoML hyperparameters (import-light, ADR-14).

The spec declares the *search*, not the final estimator: AutoML is the
tuner, so a model using this adapter must not also declare a ``tuning:``
block (the adapter rejects it at parse time).
"""

from typing import Literal

from pydantic import BaseModel, ConfigDict, Field

_ALGOS = Literal["GLM", "GBM", "XGBoost", "DRF", "DeepLearning", "StackedEnsemble"]


class H2OAutoMLParams(BaseModel):
    """Static AutoML parameters (extra='forbid')."""

    model_config = ConfigDict(extra="forbid")

    max_models: int = Field(default=10, ge=1)
    #: Wall-clock budgets make runs time-dependent and therefore
    #: irreproducible - setting either triggers a nondeterminism warning.
    max_runtime_secs: int | None = Field(default=None, ge=1)
    max_runtime_secs_per_model: int | None = Field(default=None, ge=1)
    include_algos: list[_ALGOS] | None = None
    exclude_algos: list[_ALGOS] | None = None
    sort_metric: Literal["AUTO", "auc", "aucpr", "logloss", "mean_per_class_error"] = "AUTO"
    nfolds: int = Field(default=0, ge=0)  # 0 = no CV (fast); >=2 enables stacking
    balance_classes: bool = False
    stopping_metric: str | None = None
    stopping_rounds: int | None = Field(default=None, ge=0)
    stopping_tolerance: float | None = Field(default=None, gt=0)

    def automl_kwargs(self, seed: int) -> dict[str, object]:
        kwargs: dict[str, object] = {
            "max_models": self.max_models,
            "seed": seed,
            "sort_metric": self.sort_metric,
            "nfolds": self.nfolds,
            "balance_classes": self.balance_classes,
            "verbosity": None,
        }
        for key in (
            "max_runtime_secs",
            "max_runtime_secs_per_model",
            "include_algos",
            "exclude_algos",
            "stopping_metric",
            "stopping_rounds",
            "stopping_tolerance",
        ):
            value = getattr(self, key)
            if value is not None:
                kwargs[key] = value
        # an explicit budget of 0 tells H2O "no time limit" (models-bounded)
        kwargs.setdefault("max_runtime_secs", 0)
        return kwargs
