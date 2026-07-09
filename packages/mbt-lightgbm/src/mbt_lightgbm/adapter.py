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
from mbt_adapter_base.metrics import compute_binary_results
from mbt_lightgbm.params import LightGBMBinaryParams

if TYPE_CHECKING:
    import lightgbm as lgb
    import numpy as np

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
    ) -> None:
        self.booster = booster
        self.features = features
        self.target = target
        self.categories = categories or {}


class LightGBMTrainingAdapter:
    """TrainingAdapter for binary classification over Arrow tables."""

    name = "lightgbm"
    contract_version = CONTRACT_VERSION
    data_access = "arrow"
    supported_tasks: ClassVar[set[TaskType]] = {TaskType.BINARY_CLASSIFICATION}
    determinism = DeterminismTier(kind="exact")

    def __init__(self, config: dict[str, Any] | None = None) -> None:
        self.config = config or {}

    # -- validation ------------------------------------------------------------

    def param_model(self, task: TaskType) -> type[BaseModel]:
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
        resolved = dict(spec.hyperparameters)
        for key, value in list(resolved.items()):
            if value != AUTO:
                continue
            if key == "scale_pos_weight":
                balance = profile.label_balance or {}
                positive = next(
                    (balance[k] for k in ("1", "1.0", "true", "True") if k in balance), None
                )
                if positive is None or positive <= 0:
                    raise ValueError(
                        "cannot auto-resolve scale_pos_weight without a positive-class balance"
                    )
                resolved[key] = round((1.0 - positive) / positive, 6)
            else:
                raise ValueError(
                    f"lightgbm cannot auto-resolve hyperparameter {key!r}; "
                    "only scale_pos_weight supports '{{ auto }}'"
                )
        return spec.model_copy(update={"hyperparameters": resolved})

    # -- data plumbing ---------------------------------------------------------------

    def _params(self, spec: ModelSpec) -> LightGBMBinaryParams:
        try:
            return LightGBMBinaryParams.model_validate(spec.hyperparameters)
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
        callbacks = None
        if report is not None and "validation" in data.splits():
            val_table = data.read("validation")
            val_x = self._features_matrix(val_table, features, categories)
            val_y = val_table.column(spec.target).to_numpy(zero_copy_only=False)
            valid_sets = [lgb.Dataset(val_x, label=val_y.astype(np.float64), reference=train_set)]
            booster_params["metric"] = ["auc"]  # higher-is-better report contract

            def _report_progress(env: Any) -> None:
                for _name, _metric, value, _bigger in env.evaluation_result_list or []:
                    report(env.iteration, float(value))

            callbacks = [_report_progress]

        booster = lgb.train(
            booster_params,
            train_set,
            num_boost_round=params.n_estimators,
            valid_sets=valid_sets,
            callbacks=callbacks,
        )
        return LightGBMModel(
            booster=booster, features=features, target=spec.target, categories=categories
        )

    # -- evaluation ----------------------------------------------------------------------

    def _scores(self, model: LightGBMModel, table: pa.Table) -> "np.ndarray":
        import numpy as np

        x = self._features_matrix(table, model.features, model.categories)
        return np.asarray(model.booster.predict(x))

    def evaluate(
        self,
        model: LightGBMModel,
        data: DatasetHandle,
        split: str,
        metrics: list[MetricSpec],
        slices: list[str] | None = None,
    ) -> MetricResults:
        import numpy as np

        table = data.read(split)
        y_true = table.column(model.target).to_numpy(zero_copy_only=False).astype(np.float64)
        y_score = self._scores(model, table).astype(np.float64)
        slice_columns = {
            name: table.column(name).to_numpy(zero_copy_only=False)
            for name in (slices or [])
            if name in table.column_names
        }
        return compute_binary_results(metrics, y_true, y_score, slice_columns)

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

    # -- artifacts -------------------------------------------------------------------------

    def export(self, model: LightGBMModel, format: str, store: ArtifactStore) -> ArtifactRef:
        if format not in ("native", ARTIFACT_FORMAT):
            raise ValueError(f"unsupported export format {format!r}")
        payload = {
            "model_str": model.booster.model_to_string(),
            "features": model.features,
            "target": model.target,
            "categories": model.categories,
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
        return LightGBMModel(
            booster=booster,
            features=list(payload["features"]),
            target=payload["target"],
            categories=dict(payload.get("categories") or {}),
        )
