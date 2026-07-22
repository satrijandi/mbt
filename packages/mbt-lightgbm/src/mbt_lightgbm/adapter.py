"""The LightGBM training adapter: the extensibility proof (FR-ADPT-06, G4).

Built against the public ``mbt-adapter-base`` contracts and the compliance
suite only - zero mbt-core imports. ``import lightgbm`` happens lazily
inside adapter methods (ADR-14).

String feature columns train as native categoricals (codes +
``categorical_feature``); train-time levels persist in the artifact
envelope and unseen levels become missing at prediction time.

Determinism tier: exact for CPU with ``num_threads=1``, ``deterministic``
and ``force_row_wise`` set (documented); more threads trade determinism for
speed and trigger a nondeterminism warning (FR-RUN-06).
"""

import json
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
from mbt_lightgbm.params import LightGBMBinaryParams, LightGBMRegressionParams

if TYPE_CHECKING:
    import lightgbm as lgb
    import numpy as np

    from mbt_adapter_base import (
        SupportsExplain,
        SupportsFeatureImportance,
        SupportsShapImportance,
    )
    from mbt_adapter_base.calibration import Calibrator

ARTIFACT_FORMAT = "lightgbm_json"


class LightGBMModel:
    """Opaque trained-model wrapper: booster + feature list + target + the
    train-time categorical level mapping."""

    def __init__(
        self,
        booster: "lgb.Booster",
        features: list[str],
        target: str,
        categories: dict[str, list[str]] | None = None,
        calibrator: "Calibrator | None" = None,
    ) -> None:
        self.booster = booster
        self.features = features
        self.target = target
        self.categories = categories or {}
        #: Optional post-hoc probability calibrator (R2-8); applied in _scores.
        self.calibrator = calibrator


class LightGBMTrainingAdapter:
    """TrainingAdapter for binary classification and regression over Arrow tables."""

    name = "lightgbm"
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
            return LightGBMRegressionParams
        return LightGBMBinaryParams

    def validate(self, spec: ModelSpec) -> list[ValidationIssue]:
        return [
            ValidationIssue(
                severity="warning",
                resource=spec.name,
                field_path="/hyperparameters/num_threads",
                message=warning,
                hint="the exact determinism tier requires num_threads=1",
            )
            for warning in self.nondeterminism_warnings(spec)
        ]

    def nondeterminism_warnings(self, spec: ModelSpec) -> list[str]:
        threads = spec.hyperparameters.get("num_threads", 1)
        if isinstance(threads, int) and threads > 1:
            return [f"num_threads={threads} makes LightGBM training nondeterministic"]
        return []

    # -- AUTO resolution ----------------------------------------------------------

    def resolve_auto(self, spec: ModelSpec, profile: DatasetProfile) -> ModelSpec:
        from mbt_adapter_base.training_helpers import resolve_scale_pos_weight

        resolved = dict(spec.hyperparameters)
        for key, value in list(resolved.items()):
            if value != AUTO:
                continue
            if key == "scale_pos_weight":
                resolved[key] = resolve_scale_pos_weight(profile)
            else:
                raise ValueError(
                    f"lightgbm cannot auto-resolve hyperparameter {key!r}; "
                    "only scale_pos_weight supports '{{ auto }}'"
                )
        return spec.model_copy(update={"hyperparameters": resolved})

    # -- data plumbing ---------------------------------------------------------------

    def _params(self, spec: ModelSpec) -> LightGBMBinaryParams | LightGBMRegressionParams:
        model_cls = (
            LightGBMRegressionParams if spec.task == TaskType.REGRESSION else LightGBMBinaryParams
        )
        try:
            return model_cls.model_validate(spec.hyperparameters)
        except ValidationError as exc:
            raise ValueError(f"invalid lightgbm hyperparameters: {exc}") from exc

    def _features_matrix(
        self, table: pa.Table, features: list[str], categories: dict[str, list[str]]
    ) -> "np.ndarray":
        import numpy as np

        columns = [
            categorical_codes(table, name, categories[name])
            if name in categories
            else table.column(name).to_numpy(zero_copy_only=False)
            for name in features
        ]
        if not columns:
            return np.empty((table.num_rows, 0))
        return np.column_stack(columns).astype(np.float64)

    # -- training -----------------------------------------------------------------------

    def train(self, spec: ModelSpec, data: DatasetHandle, ctx: RunContext) -> LightGBMModel:
        return self._train(spec, data, ctx, report=None)

    def train_with_report(
        self,
        spec: ModelSpec,
        data: DatasetHandle,
        ctx: RunContext,
        report: Any,
    ) -> LightGBMModel:
        """Optional tuning contract: report validation AUC (higher-is-better)
        per boosting round; the report callback may raise to abort the trial
        (pruning) and the exception propagates out of lgb.train."""
        return self._train(spec, data, ctx, report=report)

    def _train(
        self, spec: ModelSpec, data: DatasetHandle, ctx: RunContext, report: Any
    ) -> LightGBMModel:
        import lightgbm as lgb
        import numpy as np

        params = self._params(spec)
        table = data.read("train")
        features, categorical = split_feature_columns(
            table, target=spec.target, slices=spec.evaluation.slices, adapter="lightgbm"
        )
        categories = train_categories(table, categorical)
        x = self._features_matrix(table, features, categories)
        y = table.column(spec.target).to_numpy(zero_copy_only=False).astype(np.float64)
        train_set = lgb.Dataset(
            x,
            label=y,
            feature_name=features,
            categorical_feature=sorted(categories) if categories else "auto",
        )
        booster_params = params.booster_params(seed=ctx.seed)
        valid_sets = None
        callbacks: list[Any] = []
        want_eval = params.early_stopping_rounds is not None or report is not None
        if want_eval and "validation" in data.splits():
            val_table = data.read("validation")
            val_x = self._features_matrix(val_table, features, categories)
            val_y = val_table.column(spec.target).to_numpy(zero_copy_only=False)
            valid_sets = [lgb.Dataset(val_x, label=val_y.astype(np.float64), reference=train_set)]
            if params.early_stopping_rounds is not None:
                callbacks.append(lgb.early_stopping(params.early_stopping_rounds, verbose=False))
            if report is not None:
                # The report contract is higher-is-better per round: binary
                # reports validation AUC, regression reports -RMSE.
                is_regression = spec.task == TaskType.REGRESSION
                booster_params["metric"] = ["rmse"] if is_regression else ["auc"]

                def _report_progress(env: Any) -> None:
                    for _name, _metric, value, _bigger in env.evaluation_result_list or []:
                        report(env.iteration, -float(value) if is_regression else float(value))

                callbacks.append(_report_progress)

        booster = lgb.train(
            booster_params,
            train_set,
            num_boost_round=params.n_estimators,
            valid_sets=valid_sets,
            callbacks=callbacks or None,
        )
        model = LightGBMModel(
            booster=booster, features=features, target=spec.target, categories=categories
        )
        if spec.calibration is not None:
            self._fit_calibrator(model, spec, data)
        return model

    def _fit_calibrator(self, model: LightGBMModel, spec: ModelSpec, data: DatasetHandle) -> None:
        """Fit a post-hoc probability calibrator on the held-out calibration
        slice (R2-8); the same mechanism as the xgboost adapter, persisted in
        the lightgbm artifact payload instead of a booster attribute.

        Calibrated scores flow through ``_scores``, so ``evaluate`` (ece/brier),
        ``predict`` (scoring), and the paired champion delta all see calibrated
        probabilities - both models carry their own calibrator, so the gate stays
        apples-to-apples. Fits on the dedicated ``calibration`` slice core
        carves from train (falling back to ``validation`` for direct calls,
        F17); without either there is no honest calibration set (fails loudly)."""
        from mbt_adapter_base.calibration import Calibrator
        from mbt_adapter_base.training_helpers import calibration_split

        assert spec.calibration is not None  # guarded by the caller
        val = data.read(calibration_split(data))
        raw = self._scores(model, val)  # no calibrator attached yet -> raw scores
        labels = val.column(spec.target).to_numpy(zero_copy_only=False)
        model.calibrator = Calibrator.fit(raw, labels, spec.calibration)

    # -- evaluation ----------------------------------------------------------------------

    def _scores(self, model: LightGBMModel, table: pa.Table) -> "np.ndarray":
        import numpy as np

        x = self._features_matrix(table, model.features, model.categories)
        raw = np.asarray(model.booster.predict(x))
        # Post-hoc calibration is a monotonic transform on the raw score (R2-8),
        # so it recalibrates probabilities without changing rank ordering.
        if model.calibrator is not None:
            return model.calibrator.transform(raw)
        return raw

    def evaluate(
        self,
        model: LightGBMModel,
        data: DatasetHandle,
        split: str,
        metrics: list[MetricSpec],
        slices: list[str] | None = None,
    ) -> MetricResults:
        from mbt_adapter_base.training_helpers import evaluate_split

        table = data.read(split)
        return evaluate_split(table, model.target, self._scores(model, table), metrics, slices)

    def predict(self, model: LightGBMModel, data: DatasetHandle, split: str) -> pa.Table:
        table = data.read(split)
        scores = self._scores(model, table)
        return table.append_column("prediction", pa.array(scores.astype("float64")))

    def feature_importance(self, model: LightGBMModel) -> dict[str, float]:
        """Gain importance per feature, normalized to fractions (FR-DOCS-02)."""
        values = model.booster.feature_importance(importance_type="gain")
        by_name = dict(zip(model.booster.feature_name(), values, strict=True))
        total = float(sum(values))
        if not total:
            return dict.fromkeys(model.features, 0.0)
        return {name: round(float(by_name.get(name, 0.0)) / total, 6) for name in model.features}

    def _shap_values(self, model: LightGBMModel, table: "pa.Table") -> "np.ndarray":
        """Per-feature SHAP contributions ``[n_rows, n_features]`` (the trailing
        base-value column is dropped). Shared by the global importance and the
        per-prediction explanation."""
        import numpy as np

        x = self._features_matrix(table, model.features, model.categories)
        # pred_contrib: [n_rows, n_features + 1]; the trailing column is the base value
        contribs = np.asarray(model.booster.predict(x, pred_contrib=True))
        return contribs[:, :-1]

    def shap_importance(
        self, model: LightGBMModel, data: DatasetHandle, split: str
    ) -> dict[str, float]:
        """Global importance as mean |SHAP| over the split, normalized to
        fractions - additive and not cardinality-biased like split-gain, so the
        model card prefers it over gain when eval data is available (see the
        xgboost adapter for the rationale; explainability)."""
        import numpy as np

        mean_abs = np.abs(self._shap_values(model, data.read(split))).mean(axis=0)
        total = float(mean_abs.sum()) or 1.0  # a model that learned nothing -> all zeros
        return {
            name: round(float(value) / total, 6)
            for name, value in zip(model.features, mean_abs, strict=True)
        }

    def explain(
        self, model: LightGBMModel, data: DatasetHandle, split: str, top_k: int
    ) -> list[str]:
        """Per-prediction local attribution: top_k features by |SHAP| per row as
        JSON (see the xgboost adapter; explainability)."""
        from mbt_adapter_base.training_helpers import top_k_explanations

        return top_k_explanations(self._shap_values(model, data.read(split)), model.features, top_k)

    # -- artifacts -------------------------------------------------------------------------

    def export(self, model: LightGBMModel, format: str, store: ArtifactStore) -> ArtifactRef:
        if format not in ("native", ARTIFACT_FORMAT):
            raise ValueError(f"unsupported export format {format!r}")
        payload = {
            "model_str": model.booster.model_to_string(),
            "features": model.features,
            "target": model.target,
            "categories": model.categories,
            "calibrator": model.calibrator.to_json() if model.calibrator is not None else None,
        }
        with tempfile.TemporaryDirectory() as tmp:
            path = Path(tmp) / "model.lgb.json"
            path.write_text(json.dumps(payload))
            return store.put_file(path, "model.lgb.json", format=ARTIFACT_FORMAT)

    def load(self, ref: ArtifactRef, store: ArtifactStore) -> LightGBMModel:
        import lightgbm as lgb

        if ref.format != ARTIFACT_FORMAT:
            raise ValueError(f"lightgbm cannot load artifact format {ref.format!r}")
        payload = json.loads(store.fetch(ref).read_text())
        booster = lgb.Booster(model_str=payload["model_str"])
        calibrator_blob = payload.get("calibrator")
        calibrator = None
        if calibrator_blob:
            from mbt_adapter_base.calibration import Calibrator

            calibrator = Calibrator.from_json(calibrator_blob)
        return LightGBMModel(
            booster=booster,
            features=list(payload["features"]),
            target=payload["target"],
            categories=dict(payload.get("categories") or {}),
            calibrator=calibrator,
        )


if TYPE_CHECKING:
    # F27: strict mypy verifies this adapter conforms to each optional-capability
    # protocol it implements (feature_importance, shap_importance, explain);
    # ``@runtime_checkable`` alone only checks the method names. No runtime cost.
    def _capability_conformance(adapter: LightGBMTrainingAdapter) -> None:
        _feature_importance: SupportsFeatureImportance = adapter
        _shap_importance: SupportsShapImportance = adapter
        _explain: SupportsExplain = adapter
