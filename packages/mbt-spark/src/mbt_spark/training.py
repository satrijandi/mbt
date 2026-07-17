"""SparkML training adapter: distributed gradient-boosted trees.

This is the distributed-training seam: data never becomes a single dense
in-memory matrix. The adapter declares ``data_access = "path"`` and reads
the materialized parquet splits straight into Spark DataFrames; a
VectorAssembler + GBTClassifier pipeline trains across executors.

Session configuration comes from target vars (``spark_master``,
``spark_conf``), so dev can run ``local[*]`` while prod points at a cluster
- same specs, different target, as everywhere else in mbt.

Determinism: tolerance tier (distributed floating-point reduction order).
"""

import atexit
import shutil
import tempfile
from pathlib import Path
from typing import TYPE_CHECKING, Any, ClassVar

import pyarrow as pa
from pydantic import BaseModel, ConfigDict, Field, ValidationError

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

if TYPE_CHECKING:
    import numpy as np
    from pyspark.ml import PipelineModel
    from pyspark.sql import DataFrame

ARTIFACT_FORMAT = "sparkml_zip"


class SparkGBTParams(BaseModel):
    """Static hyperparameters for the GBT pipeline (extra='forbid')."""

    model_config = ConfigDict(extra="forbid")

    max_iter: int = Field(default=50, ge=1)
    max_depth: int = Field(default=5, ge=1)
    step_size: float = Field(default=0.1, gt=0, le=1)
    subsampling_rate: float = Field(default=1.0, gt=0, le=1)
    min_instances_per_node: int = Field(default=1, ge=1)
    max_bins: int = Field(default=32, ge=2)


class SparkMLModel:
    """Opaque wrapper: fitted PipelineModel + column context."""

    def __init__(self, model: "PipelineModel", features: list[str], target: str) -> None:
        self.model = model
        self.features = features
        self.target = target


class SparkMLTrainingAdapter:
    """TrainingAdapter fitting a SparkML GBT pipeline over parquet splits."""

    name = "spark"
    contract_version = CONTRACT_VERSION
    data_access = "path"
    supported_tasks: ClassVar[set[TaskType]] = {TaskType.BINARY_CLASSIFICATION}
    determinism = DeterminismTier(kind="tolerance", tolerances={"*": 0.01})

    def __init__(self, config: dict[str, Any] | None = None) -> None:
        self.config = config or {}

    # -- session ------------------------------------------------------------------

    def _spark(self, ctx: RunContext | None = None) -> Any:
        from mbt_spark.session import get_session

        vars_ = ctx.vars if ctx is not None else {}
        return get_session(
            master=str(vars_.get("spark_master", "local[*]")),
            conf=dict(vars_.get("spark_conf", {})),
            app_name="mbt-sparkml",
        )

    # -- validation ----------------------------------------------------------------

    def param_model(self, task: TaskType) -> type[BaseModel]:
        return SparkGBTParams

    def validate(self, spec: ModelSpec) -> list[ValidationIssue]:
        return []

    def nondeterminism_warnings(self, spec: ModelSpec) -> list[str]:
        return []

    def resolve_auto(self, spec: ModelSpec, profile: DatasetProfile) -> ModelSpec:
        for key, value in spec.hyperparameters.items():
            if value == AUTO:
                raise ValueError(f"sparkml has no '{{{{ auto }}}}' resolution for {key!r}")
        return spec

    def _params(self, spec: ModelSpec) -> SparkGBTParams:
        try:
            return SparkGBTParams.model_validate(spec.hyperparameters)
        except ValidationError as exc:
            raise ValueError(f"invalid sparkml hyperparameters: {exc}") from exc

    # -- data plumbing ----------------------------------------------------------------

    def _split_frame(self, spark: Any, data: DatasetHandle, split: str) -> "DataFrame":
        # handles without on-disk backing (compliance fixtures) stage locally
        from mbt_adapter_base.training_helpers import staged_split_path

        frame: DataFrame = spark.read.parquet(
            str(staged_split_path(data, split, prefix="mbt-spark-stage-"))
        )
        return frame

    def _feature_columns(self, columns: list[str], spec: ModelSpec) -> list[str]:
        return [c for c in columns if c != spec.target and c not in spec.evaluation.slices]

    # -- training ------------------------------------------------------------------------

    def train(self, spec: ModelSpec, data: DatasetHandle, ctx: RunContext) -> SparkMLModel:
        from pyspark.ml import Pipeline
        from pyspark.ml.classification import GBTClassifier
        from pyspark.ml.feature import VectorAssembler
        from pyspark.sql import functions as F

        params = self._params(spec)
        spark = self._spark(ctx)
        frame = self._split_frame(spark, data, "train")
        features = self._feature_columns(frame.columns, spec)
        frame = frame.withColumn(spec.target, F.col(spec.target).cast("double"))

        pipeline = Pipeline(
            stages=[
                VectorAssembler(inputCols=features, outputCol="mbt_features", handleInvalid="keep"),
                GBTClassifier(
                    labelCol=spec.target,
                    featuresCol="mbt_features",
                    maxIter=params.max_iter,
                    maxDepth=params.max_depth,
                    stepSize=params.step_size,
                    subsamplingRate=params.subsampling_rate,
                    minInstancesPerNode=params.min_instances_per_node,
                    maxBins=params.max_bins,
                    seed=ctx.seed,
                ),
            ]
        )
        fitted = pipeline.fit(frame)
        return SparkMLModel(model=fitted, features=features, target=spec.target)

    # -- scoring --------------------------------------------------------------------------

    def _scores(self, model: SparkMLModel, data: DatasetHandle, split: str) -> "np.ndarray":
        import numpy as np
        from pyspark.ml.functions import vector_to_array
        from pyspark.sql import functions as F

        spark = self._spark()
        frame = self._split_frame(spark, data, split)
        scored = model.model.transform(frame).withColumn(
            "mbt_p1", vector_to_array(F.col("probability")).getItem(1)
        )
        rows = scored.select("mbt_p1").toPandas()
        return np.asarray(rows["mbt_p1"], dtype=np.float64)  # type: ignore[index]

    def evaluate(
        self,
        model: SparkMLModel,
        data: DatasetHandle,
        split: str,
        metrics: list[MetricSpec],
        slices: list[str] | None = None,
    ) -> MetricResults:
        from mbt_adapter_base.training_helpers import evaluate_split

        target = str(model.target or getattr(data, "label_column", ""))
        table = data.read(split)
        return evaluate_split(table, target, self._scores(model, data, split), metrics, slices)

    def feature_importance(self, model: SparkMLModel) -> dict[str, float]:
        """GBT featureImportances normalized to fractions (FR-DOCS-02).

        The importance vector is indexed by the VectorAssembler's inputCols,
        which is exactly ``model.features`` in order.
        """
        classifier = model.model.stages[-1]
        if not hasattr(classifier, "featureImportances"):
            return {}
        importances = classifier.featureImportances
        values = [float(importances[i]) for i in range(len(model.features))]
        total = sum(values)
        if not total:
            return dict.fromkeys(model.features, 0.0)
        return {
            name: round(value / total, 6)
            for name, value in zip(model.features, values, strict=True)
        }

    def predict(self, model: SparkMLModel, data: DatasetHandle, split: str) -> pa.Table:
        table = data.read(split)
        scores = self._scores(model, data, split)
        return pa.Table.from_arrays(
            [*table.columns, pa.array(scores)],
            names=[*table.column_names, "prediction"],
        )

    # -- artifacts (zipped PipelineModel directory) ------------------------------------------

    def export(self, model: SparkMLModel, format: str, store: ArtifactStore) -> ArtifactRef:
        if format not in ("native", ARTIFACT_FORMAT):
            raise ValueError(f"unsupported export format {format!r}")
        with tempfile.TemporaryDirectory() as tmp:
            model_dir = Path(tmp) / "model"
            model.model.write().overwrite().save(str(model_dir))
            (model_dir / "mbt_columns.txt").write_text("\n".join([model.target, *model.features]))
            archive = shutil.make_archive(str(Path(tmp) / "sparkml"), "zip", model_dir)
            return store.put_file(Path(archive), "model.sparkml.zip", format=ARTIFACT_FORMAT)

    def load(self, ref: ArtifactRef, store: ArtifactStore) -> SparkMLModel:
        from pyspark.ml import PipelineModel

        if ref.format != ARTIFACT_FORMAT:
            raise ValueError(f"sparkml cannot load artifact format {ref.format!r}")
        self._spark()  # loading needs an active session
        extract_dir = Path(tempfile.mkdtemp(prefix="mbt-sparkml-load-"))
        # The loaded model reads from this dir for the rest of the (short-
        # lived) job process; clean it up at exit like the staged splits.
        atexit.register(shutil.rmtree, extract_dir, ignore_errors=True)
        shutil.unpack_archive(store.fetch(ref), extract_dir, "zip")
        columns = (extract_dir / "mbt_columns.txt").read_text().splitlines()
        model = PipelineModel.load(str(extract_dir))
        return SparkMLModel(model=model, features=columns[1:], target=columns[0])
