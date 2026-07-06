"""The LightGBM training adapter: the extensibility proof (FR-ADPT-06, G4).

Built against the public ``mbt-adapter-base`` contracts and the compliance
suite only - zero mbt-core imports. ``import lightgbm`` happens lazily
inside adapter methods (ADR-14).

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
from mbt_adapter_base.metrics import compute_binary_results
from mbt_lightgbm.params import LightGBMBinaryParams

if TYPE_CHECKING:
    import lightgbm as lgb
    import numpy as np

_NUMERIC_PREFIXES = ("int", "uint", "float", "double", "decimal", "bool")
ARTIFACT_FORMAT = "lightgbm_json"


class LightGBMModel:
    """Opaque trained-model wrapper: booster + feature list + target."""

    def __init__(self, booster: "lgb.Booster", features: list[str], target: str) -> None:
        self.booster = booster
        self.features = features
        self.target = target


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
                f"non-numeric feature column(s) for lightgbm: {', '.join(bad)}. "
                "Exclude them under features.exclude or encode them in hooks.py."
            )
        return candidates

    def _features_matrix(self, table: pa.Table, features: list[str]) -> "np.ndarray":
        import numpy as np

        columns = [table.column(name).to_numpy(zero_copy_only=False) for name in features]
        if not columns:
            return np.empty((table.num_rows, 0))
        return np.column_stack(columns).astype(np.float64)

    # -- training -----------------------------------------------------------------------

    def train(self, spec: ModelSpec, data: DatasetHandle, ctx: RunContext) -> LightGBMModel:
        import lightgbm as lgb
        import numpy as np

        params = self._params(spec)
        table = data.read("train")
        features = self._feature_columns(table, spec)
        x = self._features_matrix(table, features)
        y = table.column(spec.target).to_numpy(zero_copy_only=False).astype(np.float64)
        train_set = lgb.Dataset(x, label=y, feature_name=features)
        booster = lgb.train(
            params.booster_params(seed=ctx.seed),
            train_set,
            num_boost_round=params.n_estimators,
        )
        return LightGBMModel(booster=booster, features=features, target=spec.target)

    # -- evaluation ----------------------------------------------------------------------

    def _scores(self, model: LightGBMModel, table: pa.Table) -> "np.ndarray":
        import numpy as np

        x = self._features_matrix(table, model.features)
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

    # -- artifacts -------------------------------------------------------------------------

    def export(self, model: LightGBMModel, format: str, store: ArtifactStore) -> ArtifactRef:
        if format not in ("native", ARTIFACT_FORMAT):
            raise ValueError(f"unsupported export format {format!r}")
        payload = {
            "model_str": model.booster.model_to_string(),
            "features": model.features,
            "target": model.target,
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
            booster=booster, features=list(payload["features"]), target=payload["target"]
        )
