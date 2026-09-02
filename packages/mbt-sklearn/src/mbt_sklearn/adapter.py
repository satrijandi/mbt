"""The scikit-learn training adapter.

Built against the public ``mbt-adapter-base`` contracts and the compliance
suite only - zero mbt-core imports, like ``mbt-lightgbm``. ``import sklearn``
happens lazily inside adapter methods (ADR-14).

Why this adapter exists: scikit-learn is the most common tabular modelling
stack there is, and mbt already depends on it - ``mbt-adapter-base[metrics]``
pulls it in to compute PR-AUC and friends, so every install that evaluates a
model already has it. Shipping XGBoost, LightGBM, SparkML and H2O AutoML while
leaving sklearn to "v1" meant the one framework a new team almost certainly
already uses was the one they could not declare (FEEDBACK v3 F-1).

Four estimators across two tasks, chosen to span the ground a team actually
covers before reaching for a boosted-tree package:

* ``logistic`` / ``linear`` - the interpretable baseline every model should
  beat before anything fancier is justified.
* ``random_forest`` - the strong, nearly-tuning-free default.
* ``hist_gradient_boosting`` - sklearn's histogram booster, the closest thing
  to the xgboost/lightgbm adapters without a new dependency.

String feature columns are encoded per estimator family, not uniformly: trees
take ordinal codes, the linear estimators take one-hot columns. See
``_design_matrix`` for why that is a correctness requirement rather than a
preference. Either way the train-time levels persist in the artifact and an
unseen level becomes missing at prediction time.

Determinism tier: exact. Every estimator is seeded from ``ctx.seed`` and
``n_jobs`` defaults to 1, because sklearn's threaded paths reduce in
nondeterministic order. Raising ``n_jobs`` trades that away and warns
(FR-RUN-06).
"""

import tempfile
from pathlib import Path
from typing import TYPE_CHECKING, Any, ClassVar

import pyarrow as pa
from pydantic import BaseModel, ValidationError

from mbt_adapter_base import (
    AUTO,
    CONTRACT_VERSION,
    ArtifactRef,
    ArtifactStore,
    DatasetHandle,
    DatasetProfile,
    DeterminismTier,
    MetricResults,
    MetricSpec,
    ModelSpec,
    RunContext,
    TaskType,
    ValidationIssue,
)
from mbt_adapter_base.encoding import categorical_codes, split_feature_columns, train_categories
from mbt_sklearn.params import (
    BINARY_PARAMS,
    REGRESSION_PARAMS,
    SklearnBinaryParams,
    SklearnRegressionParams,
)

if TYPE_CHECKING:
    import numpy as np

    from mbt_adapter_base import SupportsFeatureImportance
    from mbt_adapter_base.calibration import Calibrator

ARTIFACT_FORMAT = "sklearn_joblib"

#: Estimators whose fitted attributes expose a usable global importance.
#: `coef_` for the linear models, `feature_importances_` for the trees.
_COEF_ESTIMATORS = frozenset({"logistic", "linear"})


class SklearnModel:
    """Opaque trained-model wrapper: estimator + feature list + target + the
    train-time categorical level mapping."""

    def __init__(
        self,
        estimator: Any,
        estimator_name: str,
        features: list[str],
        target: str,
        task: TaskType,
        categories: dict[str, list[str]] | None = None,
        calibrator: "Calibrator | None" = None,
    ) -> None:
        self.estimator = estimator
        self.estimator_name = estimator_name
        #: Source feature owning each design-matrix column; one-hot expansion
        #: means this is longer than `features` for the linear estimators.
        self.column_owners: list[str] = []
        self.features = features
        self.target = target
        self.task = task
        self.categories = categories or {}
        #: Optional post-hoc probability calibrator (R2-8); applied in _scores.
        self.calibrator = calibrator


class SklearnTrainingAdapter:
    """TrainingAdapter for binary classification and regression over Arrow tables."""

    name = "sklearn"
    contract_version = CONTRACT_VERSION
    data_access = "arrow"
    supported_tasks: ClassVar[set[TaskType]] = {
        TaskType.BINARY_CLASSIFICATION,
        TaskType.REGRESSION,
    }
    #: Probed by the parser (R2-8): this adapter can post-hoc calibrate scores.
    supports_calibration: ClassVar[bool] = True
    determinism = DeterminismTier(kind="exact")

    def __init__(self, config: dict[str, Any] | None = None) -> None:
        self.config = config or {}

    # -- validation ------------------------------------------------------------

    def param_model(self, task: TaskType) -> type[BaseModel]:
        if task == TaskType.REGRESSION:
            return SklearnRegressionParams
        return SklearnBinaryParams

    def validate(self, spec: ModelSpec) -> list[ValidationIssue]:
        return [
            ValidationIssue(
                severity="warning",
                resource=spec.name,
                field_path="/hyperparameters/n_jobs",
                message=warning,
                hint="the exact determinism tier requires n_jobs=1",
            )
            for warning in self.nondeterminism_warnings(spec)
        ]

    def nondeterminism_warnings(self, spec: ModelSpec) -> list[str]:
        jobs = spec.hyperparameters.get("n_jobs", 1)
        if isinstance(jobs, int) and jobs > 1:
            return [f"n_jobs={jobs} makes sklearn training nondeterministic"]
        return []

    # -- AUTO resolution ----------------------------------------------------------

    def resolve_auto(self, spec: ModelSpec, profile: DatasetProfile) -> ModelSpec:
        """``class_weight: '{{ auto }}'`` -> sklearn's ``"balanced"``.

        sklearn spells the imbalance correction as a strategy name rather than
        as xgboost's numeric ``scale_pos_weight``, and "balanced" IS the
        n/(2*count) reweighting, so it resolves from the profile's own class
        balance without mbt computing a ratio.
        """
        resolved = dict(spec.hyperparameters)
        for key, value in list(resolved.items()):
            if value != AUTO:
                continue
            if key == "class_weight":
                resolved[key] = "balanced"
            else:
                raise ValueError(
                    f"sklearn cannot auto-resolve hyperparameter {key!r}; "
                    "only class_weight supports '{{ auto }}'"
                )
        return spec.model_copy(update={"hyperparameters": resolved})

    # -- data plumbing ---------------------------------------------------------------

    def _params(self, spec: ModelSpec) -> Any:
        model_cls = self.param_model(spec.task)
        try:
            union = model_cls.model_validate(spec.hyperparameters)
        except ValidationError as exc:
            raise ValueError(f"invalid sklearn hyperparameters: {exc}") from exc
        return union.concrete()  # type: ignore[attr-defined]

    def _estimator_name(self, spec: ModelSpec) -> str:
        default = "linear" if spec.task == TaskType.REGRESSION else "logistic"
        return str(spec.hyperparameters.get("estimator", default))

    def _build_estimator(self, spec: ModelSpec, ctx: RunContext) -> Any:
        from sklearn.ensemble import (
            HistGradientBoostingClassifier,
            HistGradientBoostingRegressor,
            RandomForestClassifier,
            RandomForestRegressor,
        )
        from sklearn.linear_model import LogisticRegression, Ridge

        params = self._params(spec)
        name = self._estimator_name(spec)
        kwargs = params.estimator_kwargs(ctx.seed)
        regression = spec.task == TaskType.REGRESSION

        if name == "logistic":
            return LogisticRegression(**kwargs)
        if name == "linear":
            return Ridge(**kwargs)
        if name == "random_forest":
            weight = getattr(params, "class_weight", None)
            if regression:
                return RandomForestRegressor(**kwargs)
            return RandomForestClassifier(**kwargs, class_weight=weight)
        # hist_gradient_boosting: early stopping is wired here because the
        # adapter owns the validation split, not the estimator.
        rounds = getattr(params, "early_stopping_rounds", None)
        extra: dict[str, Any] = {}
        if rounds is not None:
            extra = {"early_stopping": True, "n_iter_no_change": rounds}
        if regression:
            return HistGradientBoostingRegressor(**kwargs, **extra)
        weight = getattr(params, "class_weight", None)
        return HistGradientBoostingClassifier(**kwargs, **extra, class_weight=weight)

    def _design_matrix(
        self, model_or_spec: "SklearnModel", table: pa.Table
    ) -> tuple["np.ndarray", list[str]]:
        """Build the feature matrix and say which source feature owns each column.

        Categoricals are encoded differently per estimator family, because the
        right encoding is a property of the model, not of the data:

        * **trees** (`random_forest`, `hist_gradient_boosting`) take ordinal
          codes. A tree can split a code range arbitrarily, so the arbitrary
          ordering costs nothing.
        * **linear** (`logistic`, `linear`) take one-hot columns. A linear model
          reads an ordinal code as a *magnitude*, so it can only fit a
          categorical whose code order happens to track the label - which is a
          coin flip. The compliance suite's mixed fixture is exactly that case
          (levels sort to east/north/south with positive rates .5/.92/.08, not
          monotonic), so ordinal codes there produce a model that cannot learn.

        The owner list is what lets `feature_importance` report per *feature*
        rather than per one-hot column.
        """
        import numpy as np

        features, categories = model_or_spec.features, model_or_spec.categories
        one_hot = model_or_spec.estimator_name in _COEF_ESTIMATORS
        columns: list[np.ndarray] = []
        owners: list[str] = []
        for name in features:
            if name not in categories:
                columns.append(table.column(name).to_numpy(zero_copy_only=False))
                owners.append(name)
                continue
            codes = categorical_codes(table, name, categories[name])
            if not one_hot:
                columns.append(codes)
                owners.append(name)
                continue
            # One column per train-time level; an unseen level codes to -1 and
            # so lands in no column at all, which is the right "missing" answer.
            for index in range(len(categories[name])):
                columns.append((np.asarray(codes) == index).astype(np.float64))
                owners.append(name)
        if not columns:
            return np.empty((table.num_rows, 0)), []
        matrix = np.column_stack(columns).astype(np.float64)
        # Ordinal codes use -1 for "unseen level" and the linear estimators
        # reject NaN, so missing stays an explicit sentinel rather than NaN.
        return np.nan_to_num(matrix, nan=-1.0, posinf=0.0, neginf=0.0), owners

    # -- training -----------------------------------------------------------------------

    def train(self, spec: ModelSpec, data: DatasetHandle, ctx: RunContext) -> SklearnModel:
        import numpy as np

        table = data.read("train")
        features, categorical = split_feature_columns(
            table, target=spec.target, slices=spec.evaluation.slices, adapter="sklearn"
        )
        categories = train_categories(table, categorical)
        model = SklearnModel(
            estimator=None,
            estimator_name=self._estimator_name(spec),
            features=features,
            target=spec.target,
            task=spec.task,
            categories=categories,
        )
        x, owners = self._design_matrix(model, table)
        y = table.column(spec.target).to_numpy(zero_copy_only=False).astype(np.float64)

        model.estimator = self._build_estimator(spec, ctx)
        model.estimator.fit(x, y)
        model.column_owners = owners
        if spec.calibration is not None:
            self._fit_calibrator(model, spec, data)
        return model

    def _fit_calibrator(self, model: SklearnModel, spec: ModelSpec, data: DatasetHandle) -> None:
        """Fit a post-hoc probability calibrator on the held-out calibration
        slice (R2-8), the same mechanism and the same slice choice as the
        xgboost and lightgbm adapters (F17)."""
        from mbt_adapter_base.calibration import Calibrator
        from mbt_adapter_base.training_helpers import calibration_split

        assert spec.calibration is not None  # guarded by the caller
        val = data.read(calibration_split(data))
        raw = self._scores(model, val)  # no calibrator attached yet -> raw scores
        labels = val.column(spec.target).to_numpy(zero_copy_only=False)
        model.calibrator = Calibrator.fit(raw, labels, spec.calibration)

    # -- evaluation ----------------------------------------------------------------------

    def _scores(self, model: SklearnModel, table: pa.Table) -> "np.ndarray":
        import numpy as np

        x, _ = self._design_matrix(model, table)
        if model.task == TaskType.REGRESSION:
            raw = np.asarray(model.estimator.predict(x), dtype=np.float64)
        else:
            # Positive-class probability: mbt's binary metrics are all defined
            # on a score, never on a hard label.
            raw = np.asarray(model.estimator.predict_proba(x), dtype=np.float64)[:, 1]
        if model.calibrator is not None:
            return model.calibrator.transform(raw)
        return raw

    def evaluate(
        self,
        model: SklearnModel,
        data: DatasetHandle,
        split: str,
        metrics: list[MetricSpec],
        slices: list[str] | None = None,
    ) -> MetricResults:
        from mbt_adapter_base.training_helpers import evaluate_split

        table = data.read(split)
        return evaluate_split(table, model.target, self._scores(model, table), metrics, slices)

    def predict(self, model: SklearnModel, data: DatasetHandle, split: str) -> pa.Table:
        table = data.read(split)
        scores = self._scores(model, table)
        return table.append_column("prediction", pa.array(scores.astype("float64")))

    def feature_importance(self, model: SklearnModel) -> dict[str, float]:
        """Global importance, normalized to fractions (FR-DOCS-02).

        ``feature_importances_`` for the tree estimators; |coef| for the linear
        ones, which is only comparable across features on a common scale - the
        model card labels it importance, not effect size. A one-hot expanded
        categorical contributes the SUM of its levels' weights, so the report
        is per feature rather than per encoded column.

        ``HistGradientBoosting*`` exposes neither attribute, so it returns
        ``{}`` - the contract's documented escape hatch for a model that cannot
        attribute - rather than a row of zeros dressed up as a ranking. Use
        ``sklearn.inspection.permutation_importance`` offline if you need one.
        """
        import numpy as np

        if model.estimator_name in _COEF_ESTIMATORS:
            values = np.abs(np.asarray(model.estimator.coef_, dtype=np.float64)).ravel()
        else:
            raw = getattr(model.estimator, "feature_importances_", None)
            if raw is None:
                return {}
            values = np.asarray(raw, dtype=np.float64).ravel()

        owners = model.column_owners or model.features
        if len(owners) != len(values):  # pragma: no cover - shape guard
            return {}
        totals = dict.fromkeys(model.features, 0.0)
        for owner, value in zip(owners, values, strict=True):
            totals[owner] += float(value)
        total = sum(totals.values())
        if not total:
            return dict.fromkeys(model.features, 0.0)
        return {name: round(value / total, 6) for name, value in totals.items()}

    # -- artifacts -------------------------------------------------------------------------

    def export(self, model: SklearnModel, format: str, store: ArtifactStore) -> ArtifactRef:
        """joblib for the fitted estimator, JSON sidecar for mbt's envelope.

        joblib is sklearn's own documented persistence format. It is a pickle,
        so it is only loadable by a compatible environment - which is exactly
        what the manifest's ``env_digest`` already pins (ADR-19), and why
        ``load`` refuses a foreign format loudly rather than half-working.
        """
        import joblib

        if format not in ("native", ARTIFACT_FORMAT):
            raise ValueError(f"unsupported export format {format!r}")
        envelope = {
            "estimator_name": model.estimator_name,
            "features": model.features,
            "target": model.target,
            "task": model.task.value,
            "categories": model.categories,
            "column_owners": model.column_owners,
            "calibrator": model.calibrator.to_json() if model.calibrator is not None else None,
        }
        with tempfile.TemporaryDirectory() as tmp:
            path = Path(tmp) / "model.sklearn.joblib"
            joblib.dump({"estimator": model.estimator, "envelope": envelope}, path)
            return store.put_file(path, "model.sklearn.joblib", format=ARTIFACT_FORMAT)

    def load(self, ref: ArtifactRef, store: ArtifactStore) -> SklearnModel:
        import joblib

        if ref.format != ARTIFACT_FORMAT:
            raise ValueError(f"sklearn cannot load artifact format {ref.format!r}")
        payload = joblib.load(store.fetch(ref))
        envelope = payload["envelope"]
        calibrator_blob = envelope.get("calibrator")
        calibrator = None
        if calibrator_blob:
            from mbt_adapter_base.calibration import Calibrator

            calibrator = Calibrator.from_json(calibrator_blob)
        model = SklearnModel(
            estimator=payload["estimator"],
            estimator_name=envelope["estimator_name"],
            features=list(envelope["features"]),
            target=envelope["target"],
            task=TaskType(envelope["task"]),
            categories=dict(envelope.get("categories") or {}),
            calibrator=calibrator,
        )
        model.column_owners = list(envelope.get("column_owners") or [])
        return model


#: Exposed for tests and docs: which estimators each task accepts.
SUPPORTED_ESTIMATORS = {
    TaskType.BINARY_CLASSIFICATION: tuple(BINARY_PARAMS),
    TaskType.REGRESSION: tuple(REGRESSION_PARAMS),
}


if TYPE_CHECKING:
    # F27: strict mypy verifies this adapter conforms to each optional-capability
    # protocol it implements; @runtime_checkable alone only checks method names.
    def _capability_conformance(adapter: SklearnTrainingAdapter) -> None:
        _feature_importance: SupportsFeatureImportance = adapter
