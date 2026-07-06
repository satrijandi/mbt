"""The XGBoost training adapter (TSD §13.1, FR-ADPT-03).

``import xgboost`` happens only inside adapter methods (ADR-14). Determinism
tier: exact for CPU hist with a fixed seed and nthread=1 (documented).

Feature derivation contract: the table an adapter reads contains features +
target + declared slice columns; features are everything except the target
and the slice columns. Non-numeric features are an error - exclude them or
encode via a hooks.py transform.
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
from mbt_adapter_base.metrics import compute_binary_results
from mbt_xgboost.params import XGBoostBinaryParams

if TYPE_CHECKING:
    import numpy as np
    import xgboost as xgb

_NUMERIC_PREFIXES = ("int", "uint", "float", "double", "decimal", "bool")


class XGBoostModel:
    """Opaque trained-model wrapper: booster + the exact feature list."""

    def __init__(self, booster: "xgb.Booster", features: list[str], target: str) -> None:
        self.booster = booster
        self.features = features
        self.target = target


def _positive_rate(profile: DatasetProfile) -> float | None:
    balance = profile.label_balance or {}
    for key in ("1", "1.0", "true", "True"):
        if key in balance:
            return balance[key]
    return None


class XGBoostTrainingAdapter:
    """TrainingAdapter for binary classification over Arrow tables."""

    name = "xgboost"
    contract_version = CONTRACT_VERSION
    data_access = "arrow"
    supported_tasks: ClassVar[set[TaskType]] = {TaskType.BINARY_CLASSIFICATION}
    determinism = DeterminismTier(kind="exact")

    def __init__(self, config: dict[str, Any] | None = None) -> None:
        self.config = config or {}

    # -- validation ---------------------------------------------------------

    def param_model(self, task: TaskType) -> type[BaseModel]:
        return XGBoostBinaryParams

    def validate(self, spec: ModelSpec) -> list[ValidationIssue]:
        issues: list[ValidationIssue] = []
        for warning in self.nondeterminism_warnings(spec):
            issues.append(
                ValidationIssue(
                    severity="warning",
                    resource=spec.name,
                    field_path="/hyperparameters",
                    message=warning,
                    hint="the exact determinism tier only holds for CPU hist (FR-RUN-06)",
                )
            )
        return issues

    def nondeterminism_warnings(self, spec: ModelSpec) -> list[str]:
        warnings: list[str] = []
        tree_method = spec.hyperparameters.get("tree_method", "hist")
        if tree_method not in ("hist", AUTO):
            warnings.append(
                f"tree_method={tree_method!r} is not covered by the exact determinism tier"
            )
        if spec.hyperparameters.get("device") == "cuda":
            warnings.append("device='cuda' introduces floating-point nondeterminism")
        return warnings

    # -- AUTO resolution (FR-RES-10) -------------------------------------------

    def resolve_auto(self, spec: ModelSpec, profile: DatasetProfile) -> ModelSpec:
        resolved = dict(spec.hyperparameters)
        for key, value in list(resolved.items()):
            if value != AUTO:
                continue
            if key == "scale_pos_weight":
                positive = _positive_rate(profile)
                if positive is None or positive <= 0:
                    raise ValueError(
                        "cannot auto-resolve scale_pos_weight: the dataset profile has "
                        "no positive-class balance"
                    )
                resolved[key] = round((1.0 - positive) / positive, 6)
            else:
                raise ValueError(
                    f"xgboost cannot auto-resolve hyperparameter {key!r}; "
                    "only scale_pos_weight supports '{{ auto }}'"
                )
        return spec.model_copy(update={"hyperparameters": resolved})

    # -- data plumbing -----------------------------------------------------------

    def _params(self, spec: ModelSpec) -> XGBoostBinaryParams:
        try:
            return XGBoostBinaryParams.model_validate(spec.hyperparameters)
        except ValidationError as exc:
            raise ValueError(f"invalid xgboost hyperparameters: {exc}") from exc

    def _feature_columns(self, table: pa.Table, spec: ModelSpec) -> list[str]:
        candidates = [
            name
            for name in table.column_names
            if name != spec.target and name not in spec.evaluation.slices
        ]
        bad = [
            name
            for name in candidates
            if not str(table.schema.field(name).type).startswith(_NUMERIC_PREFIXES)
        ]
        if bad:
            raise ValueError(
                f"non-numeric feature column(s) for xgboost: {', '.join(bad)}. "
                "Exclude them under features.exclude or encode them in a hooks.py "
                "transform_features."
            )
        return candidates

    def _matrix(
        self, table: pa.Table, features: list[str], target: str | None
    ) -> tuple["xgb.DMatrix", "np.ndarray | None"]:
        import numpy as np
        import xgboost as xgb

        columns = [table.column(name).to_numpy(zero_copy_only=False) for name in features]
        if columns:
            x = np.column_stack(columns).astype(np.float32)
        else:
            x = np.empty((table.num_rows, 0), dtype=np.float32)
        y = None
        if target is not None:
            y = table.column(target).to_numpy(zero_copy_only=False).astype(np.float32)
        return xgb.DMatrix(x, label=y, feature_names=features), y

    # -- training ------------------------------------------------------------------

    def train(self, spec: ModelSpec, data: DatasetHandle, ctx: RunContext) -> XGBoostModel:
        import xgboost as xgb

        params = self._params(spec)
        table = data.read("train")
        features = self._feature_columns(table, spec)
        dtrain, _ = self._matrix(table, features, spec.target)

        evals = []
        if params.early_stopping_rounds is not None and "validation" in data.splits():
            dval, _ = self._matrix(data.read("validation"), features, spec.target)
            evals = [(dval, "validation")]

        booster = xgb.train(
            params.booster_params(seed=ctx.seed),
            dtrain,
            num_boost_round=params.n_estimators,
            evals=evals,
            early_stopping_rounds=params.early_stopping_rounds if evals else None,
            verbose_eval=False,
        )
        # Persisted with the model so load() can reconstruct the wrapper.
        booster.set_attr(mbt_target=spec.target)
        return XGBoostModel(booster=booster, features=features, target=spec.target)

    # -- evaluation ------------------------------------------------------------------

    def _scores(self, model: XGBoostModel, table: pa.Table) -> "np.ndarray":
        matrix, _ = self._matrix(table, model.features, None)
        return model.booster.predict(matrix)

    def evaluate(
        self,
        model: XGBoostModel,
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

    def predict(self, model: XGBoostModel, data: DatasetHandle, split: str) -> pa.Table:
        table = data.read(split)
        scores = self._scores(model, table)
        return table.append_column("prediction", pa.array(scores.astype("float64")))

    # -- artifacts -----------------------------------------------------------------------

    def export(self, model: XGBoostModel, format: str, store: ArtifactStore) -> ArtifactRef:
        if format in ("native", "xgboost_ubj"):
            with tempfile.TemporaryDirectory() as tmp:
                path = Path(tmp) / "model.ubj"
                model.booster.save_model(str(path))
                return store.put_file(path, "model.ubj", format="xgboost_ubj")
        if format == "onnx":
            return self._export_onnx(model, store)
        raise ValueError(
            f"unsupported export format {format!r} (supported: native/xgboost_ubj, onnx)"
        )

    def _export_onnx(self, model: XGBoostModel, store: ArtifactStore) -> ArtifactRef:
        try:
            from onnxmltools import convert_xgboost  # type: ignore[import-not-found]
            from onnxmltools.convert.common.data_types import (  # type: ignore[import-not-found]
                FloatTensorType,
            )
        except ImportError as exc:  # pragma: no cover - extra not installed
            raise ValueError(
                "ONNX export needs the onnx extra: pip install 'mbt-xgboost[onnx]'"
            ) from exc
        initial_types = [("input", FloatTensorType([None, len(model.features)]))]
        onnx_model = convert_xgboost(model.booster, initial_types=initial_types)
        with tempfile.TemporaryDirectory() as tmp:
            path = Path(tmp) / "model.onnx"
            path.write_bytes(onnx_model.SerializeToString())
            return store.put_file(path, "model.onnx", format="onnx")

    def load(self, ref: ArtifactRef, store: ArtifactStore) -> XGBoostModel:
        import xgboost as xgb

        if ref.format != "xgboost_ubj":
            raise ValueError(
                f"xgboost cannot load artifact format {ref.format!r} "
                "(champion evaluation needs an xgboost_ubj artifact)"
            )
        booster = xgb.Booster()
        booster.load_model(str(store.fetch(ref)))
        features = list(booster.feature_names or [])
        attributes = booster.attributes()
        return XGBoostModel(
            booster=booster,
            features=features,
            target=attributes.get("mbt_target") or "",
        )
