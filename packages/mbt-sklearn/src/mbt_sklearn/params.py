"""scikit-learn hyperparameter models (import-light, ADR-14).

One spec field selects the estimator (``estimator: logistic``), and the rest
are that estimator's own hyperparameters. They are modelled per estimator with
``extra='forbid'`` rather than passed through as a free dict, so a typo is a
parse-time error with a field path instead of a ``TypeError`` inside a training
subprocess an hour into a build.
"""

from typing import ClassVar, Literal

from pydantic import BaseModel, ConfigDict, Field, model_validator

#: estimator name -> the tasks it can be used for. The parser reads this to
#: reject `estimator: logistic` on a regression model at parse time.
BINARY_ESTIMATORS = ("logistic", "random_forest", "hist_gradient_boosting")
REGRESSION_ESTIMATORS = ("linear", "random_forest", "hist_gradient_boosting")


class _CommonParams(BaseModel):
    """Shared by every estimator (extra='forbid')."""

    model_config = ConfigDict(extra="forbid")

    #: Single-threaded by default: sklearn's threaded paths reduce in
    #: nondeterministic order, which would cost the exact determinism tier.
    n_jobs: int = Field(default=1, ge=1)


class LogisticParams(_CommonParams):
    """`LogisticRegression`, the workhorse baseline for binary tasks."""

    estimator: Literal["logistic"] = "logistic"
    C: float = Field(default=1.0, gt=0)
    penalty: Literal["l1", "l2", "elasticnet"] | None = None
    l1_ratio: float | None = Field(default=None, ge=0, le=1)
    max_iter: int = Field(default=1000, ge=1)
    tol: float = Field(default=1e-4, gt=0)
    #: '{{ auto }}' resolves to "balanced" from the dataset profile.
    class_weight: str | None = None
    #: liblinear/saga are the deterministic-friendly solvers; lbfgs is the
    #: sklearn default and is deterministic for this problem class.
    solver: Literal["lbfgs", "liblinear", "saga", "newton-cholesky"] = "lbfgs"

    def estimator_kwargs(self, seed: int) -> dict[str, object]:
        kwargs: dict[str, object] = {
            "C": self.C,
            "max_iter": self.max_iter,
            "tol": self.tol,
            "solver": self.solver,
            "random_state": seed,
        }
        # `penalty` is deprecated in scikit-learn 1.8 and removed in 1.10 in
        # favour of `l1_ratio`. Forwarding it unconditionally raised a
        # FutureWarning on every fit, including the default case where the user
        # never asked for it - so it is passed only when explicitly set, which
        # keeps the knob working on 1.5-1.9 without warning anyone who is not
        # using it. Drop the field when the floor reaches 1.10.
        if self.penalty is not None:
            kwargs["penalty"] = self.penalty
        if self.l1_ratio is not None:
            kwargs["l1_ratio"] = self.l1_ratio
        if self.class_weight is not None:
            kwargs["class_weight"] = self.class_weight
        # LogisticRegression takes n_jobs only for the multinomial/ovr paths;
        # passing it is harmless and keeps the knob uniform across estimators.
        kwargs["n_jobs"] = self.n_jobs if self.solver == "liblinear" else None
        return kwargs


class LinearParams(_CommonParams):
    """`Ridge`, the regression counterpart of the logistic baseline.

    Ridge rather than plain ``LinearRegression`` because ``alpha=0`` recovers
    OLS exactly, so one estimator covers both and regularization stays a knob
    instead of a different spec.
    """

    estimator: Literal["linear"] = "linear"
    alpha: float = Field(default=1.0, ge=0)
    max_iter: int | None = Field(default=None, ge=1)
    tol: float = Field(default=1e-4, gt=0)

    def estimator_kwargs(self, seed: int) -> dict[str, object]:
        return {
            "alpha": self.alpha,
            "max_iter": self.max_iter,
            "tol": self.tol,
            "random_state": seed,
            # 'auto' picks a closed-form solver for dense input, which is
            # deterministic; the iterative solvers are not seeded identically
            # across BLAS builds.
            "solver": "auto",
        }


class RandomForestParams(_CommonParams):
    """`RandomForestClassifier` / `RandomForestRegressor`."""

    estimator: Literal["random_forest"] = "random_forest"
    n_estimators: int = Field(default=100, ge=1)
    max_depth: int | None = Field(default=None, ge=1)
    min_samples_split: int = Field(default=2, ge=2)
    min_samples_leaf: int = Field(default=1, ge=1)
    max_features: float | Literal["sqrt", "log2"] | None = "sqrt"
    class_weight: str | None = None  # classifier only; '{{ auto }}' -> "balanced"

    def estimator_kwargs(self, seed: int) -> dict[str, object]:
        return {
            "n_estimators": self.n_estimators,
            "max_depth": self.max_depth,
            "min_samples_split": self.min_samples_split,
            "min_samples_leaf": self.min_samples_leaf,
            "max_features": self.max_features,
            "random_state": seed,
            "n_jobs": self.n_jobs,
        }


class HistGradientBoostingParams(_CommonParams):
    """`HistGradientBoosting*`: sklearn's LightGBM-style histogram booster.

    The closest sklearn analogue to the xgboost/lightgbm adapters, and the one
    to reach for when a team wants gradient boosting without adding a
    dependency mbt does not already install.
    """

    estimator: Literal["hist_gradient_boosting"] = "hist_gradient_boosting"
    max_iter: int = Field(default=100, ge=1)
    learning_rate: float = Field(default=0.1, gt=0)
    max_depth: int | None = Field(default=None, ge=1)
    max_leaf_nodes: int | None = Field(default=31, ge=2)
    min_samples_leaf: int = Field(default=20, ge=1)
    l2_regularization: float = Field(default=0.0, ge=0)
    #: Needs a validation split; mirrors the xgboost/lightgbm contract.
    early_stopping_rounds: int | None = Field(default=None, ge=1)
    class_weight: str | None = None  # classifier only; '{{ auto }}' -> "balanced"

    def estimator_kwargs(self, seed: int) -> dict[str, object]:
        # early stopping is wired by the adapter (it owns the validation
        # split), so it is deliberately not forwarded here.
        return {
            "max_iter": self.max_iter,
            "learning_rate": self.learning_rate,
            "max_depth": self.max_depth,
            "max_leaf_nodes": self.max_leaf_nodes,
            "min_samples_leaf": self.min_samples_leaf,
            "l2_regularization": self.l2_regularization,
            "random_state": seed,
        }


#: Discriminated by the `estimator` field so pydantic reports the right
#: field paths; the adapter picks the model class before validating.
BINARY_PARAMS: dict[str, type[_CommonParams]] = {
    "logistic": LogisticParams,
    "random_forest": RandomForestParams,
    "hist_gradient_boosting": HistGradientBoostingParams,
}
REGRESSION_PARAMS: dict[str, type[_CommonParams]] = {
    "linear": LinearParams,
    "random_forest": RandomForestParams,
    "hist_gradient_boosting": HistGradientBoostingParams,
}


class _UnionParams(BaseModel):
    """Base for the per-task model the parser validates specs against.

    ``TrainingAdapter.param_model(task)`` returns exactly one model, and mbt
    uses it for two things: the set of legal hyperparameter names, and
    validation of the statically-known values (``validate_hyperparameters`` in
    mbt-core). A sklearn spec picks its estimator in that same block, so one
    model per estimator cannot work.

    The subclasses below are therefore the union of every field across a
    task's estimators, all optional, plus the ``estimator`` discriminator. The
    validator here re-checks the supplied keys against the *concrete*
    estimator, so both mistakes fail at parse time with a field path: a name
    no estimator has (caught by ``extra='forbid'``), and a name belonging to a
    different estimator than the one selected (caught here).

    Written out rather than generated from ``BINARY_PARAMS``: a dynamically
    built model is invisible to mypy and to anyone reading the file for the
    legal knobs, which is exactly what this file is for.
    """

    model_config = ConfigDict(extra="forbid")

    _choices: ClassVar[dict[str, type[_CommonParams]]] = {}

    n_jobs: int | None = Field(default=None, ge=1)

    @model_validator(mode="after")
    def _validate_for_estimator(self) -> "_UnionParams":
        estimator = str(self.__dict__["estimator"])
        concrete = self._choices[estimator]
        supplied = {k: v for k, v in self.__dict__.items() if k != "estimator" and v is not None}
        legal = set(concrete.model_fields) - {"estimator"}
        unknown = sorted(set(supplied) - legal)
        if unknown:
            raise ValueError(
                f"hyperparameter(s) {', '.join(unknown)} are not valid for "
                f"estimator {estimator!r}; valid: {', '.join(sorted(legal))}"
            )
        concrete.model_validate({**supplied, "estimator": estimator})
        return self

    def concrete(self) -> _CommonParams:
        """The estimator-specific params, with this spec's values applied."""
        estimator = str(self.__dict__["estimator"])
        supplied = {k: v for k, v in self.__dict__.items() if k != "estimator" and v is not None}
        return self._choices[estimator].model_validate({**supplied, "estimator": estimator})


class SklearnBinaryParams(_UnionParams):
    """Every hyperparameter any binary-classification estimator accepts."""

    _choices: ClassVar[dict[str, type[_CommonParams]]] = BINARY_PARAMS

    estimator: Literal["logistic", "random_forest", "hist_gradient_boosting"] = "logistic"
    # logistic
    C: float | None = Field(default=None, gt=0)
    penalty: Literal["l1", "l2", "elasticnet"] | None = None
    l1_ratio: float | None = Field(default=None, ge=0, le=1)
    tol: float | None = Field(default=None, gt=0)
    solver: Literal["lbfgs", "liblinear", "saga", "newton-cholesky"] | None = None
    # shared: '{{ auto }}' resolves to "balanced" from the dataset profile
    class_weight: str | None = None
    # random_forest
    n_estimators: int | None = Field(default=None, ge=1)
    max_depth: int | None = Field(default=None, ge=1)
    min_samples_split: int | None = Field(default=None, ge=2)
    min_samples_leaf: int | None = Field(default=None, ge=1)
    max_features: float | Literal["sqrt", "log2"] | None = None
    # hist_gradient_boosting (max_iter is also logistic's iteration cap)
    max_iter: int | None = Field(default=None, ge=1)
    learning_rate: float | None = Field(default=None, gt=0)
    max_leaf_nodes: int | None = Field(default=None, ge=2)
    l2_regularization: float | None = Field(default=None, ge=0)
    early_stopping_rounds: int | None = Field(default=None, ge=1)


class SklearnRegressionParams(_UnionParams):
    """Every hyperparameter any regression estimator accepts."""

    _choices: ClassVar[dict[str, type[_CommonParams]]] = REGRESSION_PARAMS

    estimator: Literal["linear", "random_forest", "hist_gradient_boosting"] = "linear"
    # linear (Ridge)
    alpha: float | None = Field(default=None, ge=0)
    tol: float | None = Field(default=None, gt=0)
    # random_forest
    n_estimators: int | None = Field(default=None, ge=1)
    max_depth: int | None = Field(default=None, ge=1)
    min_samples_split: int | None = Field(default=None, ge=2)
    min_samples_leaf: int | None = Field(default=None, ge=1)
    max_features: float | Literal["sqrt", "log2"] | None = None
    # hist_gradient_boosting (max_iter is also Ridge's iteration cap)
    max_iter: int | None = Field(default=None, ge=1)
    learning_rate: float | None = Field(default=None, gt=0)
    max_leaf_nodes: int | None = Field(default=None, ge=2)
    l2_regularization: float | None = Field(default=None, ge=0)
    early_stopping_rounds: int | None = Field(default=None, ge=1)
