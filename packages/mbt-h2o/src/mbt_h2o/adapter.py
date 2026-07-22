"""H2O AutoML training adapter (local backend + optional Sparkling Water).

- ``data_access = "path"``: H2O ingests the materialized parquet splits
  directly (``h2o.import_file``) - no Arrow round-trip through Python.
- Artifacts are **MOJOs**: single self-contained zips, reloaded with
  ``h2o.import_mojo`` for champion evaluation and ``mbt evaluate``.
- Metrics are computed by mbt's shared binary-metric helpers over MOJO/leader
  probabilities, so champion/challenger deltas stay comparable across
  adapters.
- Determinism: tolerance tier. With ``max_models`` + ``seed`` and no time
  budget, AutoML is *mostly* repeatable; wall-clock budgets
  (``max_runtime_secs*``) are flagged as nondeterminism sources (FR-RUN-06).
- Backends: ``local`` (in-process H2O cluster) or ``sparkling`` (H2O on
  Spark executors via PySparkling, ``mbt-h2o[sparkling]``), selected with
  the ``h2o_backend`` target var.

``import h2o`` happens lazily inside methods (ADR-14).
"""

import atexit
import shutil
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
from mbt_h2o.params import H2OAutoMLParams

if TYPE_CHECKING:
    import numpy as np

    from mbt_adapter_base.calibration import Calibrator

_shutdown_registered = False


class H2OModel:
    """Opaque wrapper: the leader (or an imported MOJO) + column context."""

    def __init__(
        self,
        model: Any,
        features: list[str],
        target: str | None,
        calibrator: "Calibrator | None" = None,
    ) -> None:
        self.model = model
        self.features = features
        self.target = target
        #: Optional post-hoc probability calibrator (R2-8); applied in _scores.
        self.calibrator = calibrator


class H2OAutoMLAdapter:
    """TrainingAdapter running H2O AutoML; the leader model is the artifact."""

    name = "h2o_automl"
    contract_version = CONTRACT_VERSION
    data_access = "path"
    supported_tasks: ClassVar[set[TaskType]] = {
        TaskType.BINARY_CLASSIFICATION,
        TaskType.REGRESSION,
    }
    #: Probed by the parser (R2-8): this adapter can post-hoc calibrate scores.
    supports_calibration: ClassVar[bool] = True
    #: AutoML rankings can flip between near-tied leaders across environments;
    #: metric-level variance stays small when runs are models-bounded.
    determinism = DeterminismTier(kind="tolerance", tolerances={"*": 0.02})

    def __init__(self, config: dict[str, Any] | None = None) -> None:
        self.config = config or {}

    # -- session management ----------------------------------------------------

    def _h2o(self, ctx: RunContext | None = None) -> Any:
        """Start (or attach to) an H2O cluster; idempotent per process."""
        global _shutdown_registered  # noqa: PLW0603 - one JVM per job process
        import h2o

        vars_ = ctx.vars if ctx is not None else {}
        backend = str(vars_.get("h2o_backend", "local"))
        if backend == "sparkling":
            self._sparkling_context(vars_)
        else:
            h2o.init(
                nthreads=int(vars_.get("h2o_nthreads", -1)),
                max_mem_size=str(vars_.get("h2o_max_mem", "4G")),
                log_level="ERRR",
                bind_to_localhost=True,
            )
        h2o.no_progress()
        if not _shutdown_registered:
            # the cluster is job-scoped; do not leave a JVM behind
            atexit.register(self._shutdown_quietly)
            _shutdown_registered = True
        return h2o

    def _sparkling_context(self, vars_: dict[str, Any]) -> Any:
        """H2O on Spark executors via PySparkling (mbt-h2o[sparkling])."""
        try:
            from pyspark.sql import SparkSession
            from pysparkling import H2OConf, H2OContext
        except ImportError as exc:
            raise RuntimeError(
                "h2o_backend=sparkling needs the sparkling extra: "
                "pip install 'mbt-h2o[sparkling]' (pins pyspark 3.5 + "
                "h2o-pysparkling-3.5; the H2O<->Spark version matrix matters)"
            ) from exc
        builder = SparkSession.builder.appName("mbt-h2o-sparkling").master(
            str(vars_.get("spark_master", "local[*]"))
        )
        conf = {str(key): str(value) for key, value in dict(vars_.get("spark_conf", {})).items()}
        # Sparkling's REST client uses the JDK-internal sun.net.www http
        # handler; Java 16+ strong encapsulation refuses that without an
        # explicit export (harmless on Java 11). Appended so user-provided
        # extraJavaOptions survive.
        exports = (
            "--add-exports=java.base/sun.net.www.protocol.http=ALL-UNNAMED "
            "--add-exports=java.base/sun.net.www.protocol.https=ALL-UNNAMED"
        )
        for key in ("spark.driver.extraJavaOptions", "spark.executor.extraJavaOptions"):
            conf[key] = f"{conf.get(key, '')} {exports}".strip()
        for key, value in conf.items():
            builder = builder.config(key, value)
        builder.getOrCreate()
        # H2OConf takes no session since sparkling 3.32: it reads the active
        # SparkSession (created above) itself.
        return H2OContext.getOrCreate(H2OConf().setInternalClusterMode())

    @staticmethod
    def _shutdown_quietly() -> None:
        try:
            import h2o

            if h2o.cluster() is not None:
                h2o.cluster().shutdown()
        except Exception:
            pass

    # -- validation ---------------------------------------------------------------

    def param_model(self, task: TaskType) -> type[BaseModel]:
        return H2OAutoMLParams

    def validate(self, spec: ModelSpec) -> list[ValidationIssue]:
        issues: list[ValidationIssue] = []
        if spec.tuning is not None:
            issues.append(
                ValidationIssue(
                    severity="error",
                    resource=spec.name,
                    field_path="/tuning",
                    message="h2o_automl performs its own model search; a tuning "
                    "block would tune the tuner",
                    hint="remove 'tuning:' - use max_models/include_algos instead",
                )
            )
        issues.extend(
            ValidationIssue(
                severity="warning",
                resource=spec.name,
                field_path="/hyperparameters",
                message=warning,
                hint="models-bounded runs (max_models + seed) are repeatable; "
                "wall-clock budgets are not (FR-RUN-06)",
            )
            for warning in self.nondeterminism_warnings(spec)
        )
        return issues

    def nondeterminism_warnings(self, spec: ModelSpec) -> list[str]:
        warnings = []
        for key in ("max_runtime_secs", "max_runtime_secs_per_model"):
            if spec.hyperparameters.get(key):
                warnings.append(f"{key} makes the AutoML search time-dependent and irreproducible")
        return warnings

    def resolve_auto(self, spec: ModelSpec, profile: DatasetProfile) -> ModelSpec:
        for key, value in spec.hyperparameters.items():
            if value == AUTO:
                raise ValueError(
                    f"h2o_automl has no '{{{{ auto }}}}' resolution for {key!r}; "
                    "AutoML already adapts to the data (try balance_classes: true)"
                )
        return spec

    # -- data plumbing --------------------------------------------------------------

    def _params(self, spec: ModelSpec) -> H2OAutoMLParams:
        try:
            return H2OAutoMLParams.model_validate(spec.hyperparameters)
        except ValidationError as exc:
            raise ValueError(f"invalid h2o_automl hyperparameters: {exc}") from exc

    def _split_file(self, data: DatasetHandle, split: str) -> Path:
        """The split's parquet file; falls back to staging one for handles
        without on-disk backing (e.g. the compliance suite's fixtures)."""
        from mbt_adapter_base.training_helpers import staged_split_path

        return staged_split_path(data, split, prefix="mbt-h2o-split-")

    def _frame(
        self, h2o: Any, data: DatasetHandle, split: str, target: str, *, classify: bool
    ) -> Any:
        frame = h2o.import_file(str(self._split_file(data, split)))
        # A factor target is what makes H2O AutoML pick a classifier; a regression
        # target must stay numeric. Only matters at train time (predict uses the
        # model's learned type), so scoring passes classify=False.
        if classify and target in frame.columns:
            frame[target] = frame[target].asfactor()
        return frame

    def _feature_columns(self, columns: list[str], spec: ModelSpec) -> list[str]:
        return [c for c in columns if c != spec.target and c not in spec.evaluation.slices]

    # -- training ----------------------------------------------------------------------

    def train(self, spec: ModelSpec, data: DatasetHandle, ctx: RunContext) -> H2OModel:
        h2o = self._h2o(ctx)
        from h2o.automl import H2OAutoML

        params = self._params(spec)
        classify = spec.task is TaskType.BINARY_CLASSIFICATION
        frame = self._frame(h2o, data, "train", spec.target, classify=classify)
        features = self._feature_columns(frame.columns, spec)

        automl = H2OAutoML(**params.automl_kwargs(seed=ctx.seed))
        automl.train(x=features, y=spec.target, training_frame=frame)
        if automl.leader is None:
            raise RuntimeError("H2O AutoML produced no models - check algos/budget settings")
        self._emit_leaderboard(automl, ctx)
        model = H2OModel(model=automl.leader, features=features, target=spec.target)
        if spec.calibration is not None:
            self._fit_calibrator(model, spec, data)
        return model

    def _fit_calibrator(self, model: H2OModel, spec: ModelSpec, data: DatasetHandle) -> None:
        """Fit a post-hoc probability calibrator on the held-out calibration
        slice (R2-8); the same mechanism as the arrow adapters, persisted in the
        MOJO bundle sidecar. Calibrated scores flow through ``_scores``, so
        evaluate/predict/the champion delta all see calibrated probabilities,
        both models carry their own, and the gate stays apples-to-apples.
        Fits on the dedicated ``calibration`` slice core carves from train
        (falling back to ``validation`` for direct calls, F17); without either
        there is no honest calibration set, so this fails loudly."""
        from mbt_adapter_base.calibration import Calibrator
        from mbt_adapter_base.training_helpers import calibration_split

        assert spec.calibration is not None  # guarded by the caller
        split = calibration_split(data)
        raw = self._scores(model, data, split)  # no calibrator yet -> raw
        labels = data.read(split).column(spec.target).to_numpy(zero_copy_only=False)
        model.calibrator = Calibrator.fit(raw, labels, spec.calibration)

    def _emit_leaderboard(self, automl: Any, ctx: RunContext) -> None:
        try:
            board = automl.leaderboard.as_data_frame(use_multi_thread=True)
            for _, row in board.head(5).iterrows():
                ctx.events.emit(
                    f"h2o leaderboard: {row.iloc[0]}  "
                    + "  ".join(f"{c}={row[c]:.4f}" for c in board.columns[1:3])
                )
        except Exception:
            pass

    # -- scoring -----------------------------------------------------------------------

    def _scores(self, model: H2OModel, data: DatasetHandle, split: str) -> "np.ndarray":
        import numpy as np

        h2o = self._h2o()
        target = str(model.target or getattr(data, "label_column", ""))
        frame = self._frame(h2o, data, split, target, classify=False)
        predictions = model.model.predict(frame)
        # binomial: [predict, p0, p1] -> P(class 1); regression: [predict] ->
        # the target-scale value. Dispatch on the presence of a p1 column.
        col = predictions["p1"] if "p1" in predictions.columns else predictions["predict"]
        raw = np.asarray(col.as_data_frame(use_multi_thread=True)).ravel()
        # Post-hoc calibration is a monotonic transform on the raw score (R2-8).
        if model.calibrator is not None:
            return model.calibrator.transform(raw)
        return raw

    def evaluate(
        self,
        model: H2OModel,
        data: DatasetHandle,
        split: str,
        metrics: list[MetricSpec],
        slices: list[str] | None = None,
    ) -> MetricResults:
        from mbt_adapter_base.training_helpers import evaluate_split

        target = str(model.target or getattr(data, "label_column", ""))
        table = data.read(split)
        return evaluate_split(table, target, self._scores(model, data, split), metrics, slices)

    def feature_importance(self, model: H2OModel) -> dict[str, float]:
        """Leader varimp percentages as fractions (FR-DOCS-02).

        Ensemble leaders have no variable importance - they yield {} and the
        model card simply omits the table (the documented optional-capability
        contract), rather than inventing numbers from a metalearner.
        """
        try:
            rows = model.model.varimp(use_pandas=False)
        except (AttributeError, ValueError):
            return {}
        if not rows:
            return {}
        # rows: (variable, relative_importance, scaled_importance, percentage).
        # GLM et al. expand categoricals to one "column.level" row per level;
        # roll those up to the column so fractions keep summing to ~1.
        by_feature: dict[str, float] = dict.fromkeys(model.features, 0.0)
        for row in rows:
            name, pct = str(row[0]), float(row[3])
            if name in by_feature:
                by_feature[name] += pct
                continue
            base = max(
                (f for f in model.features if name.startswith(f + ".")), key=len, default=None
            )
            if base is not None:
                by_feature[base] += pct
        return {name: round(value, 6) for name, value in by_feature.items()}

    def predict(self, model: H2OModel, data: DatasetHandle, split: str) -> pa.Table:
        table = data.read(split)
        scores = self._scores(model, data, split)
        return pa.Table.from_arrays(
            [*table.columns, pa.array(scores.astype("float64"))],
            names=[*table.column_names, "prediction"],
        )

    # -- artifacts (MOJO) -----------------------------------------------------------------

    def export(self, model: H2OModel, format: str, store: ArtifactStore) -> ArtifactRef:
        if format not in ("native", "h2o_mojo"):
            raise ValueError(f"unsupported export format {format!r}")
        with tempfile.TemporaryDirectory() as tmp:
            bundle = Path(tmp) / "bundle"
            bundle.mkdir()
            mojo = Path(model.model.download_mojo(path=str(bundle), get_genmodel_jar=False))
            mojo.rename(bundle / "model.mojo.zip")  # fixed name for load
            # A columns sidecar (target + features), like the sparkml adapter, so
            # load() recovers the feature context - otherwise a reloaded MOJO has
            # no features and its feature_importance/champion card table is empty.
            sidecar = "\n".join([model.target or "", *model.features])
            (bundle / "mbt_columns.txt").write_text(sidecar)
            if model.calibrator is not None:
                (bundle / "mbt_calibrator.json").write_text(model.calibrator.to_json())
            archive = shutil.make_archive(str(Path(tmp) / "h2obundle"), "zip", bundle)
            return store.put_file(Path(archive), "model.h2omojo.zip", format="h2o_mojo")

    def load(self, ref: ArtifactRef, store: ArtifactStore) -> H2OModel:
        if ref.format != "h2o_mojo":
            raise ValueError(f"h2o_automl cannot load artifact format {ref.format!r}")
        h2o = self._h2o()
        extract_dir = Path(tempfile.mkdtemp(prefix="mbt-h2o-load-"))
        atexit.register(shutil.rmtree, extract_dir, ignore_errors=True)
        shutil.unpack_archive(store.fetch(ref), extract_dir, "zip")
        columns = (extract_dir / "mbt_columns.txt").read_text().splitlines()
        imported = h2o.import_mojo(str(extract_dir / "model.mojo.zip"))
        calibrator = None
        calibrator_path = extract_dir / "mbt_calibrator.json"
        if calibrator_path.exists():
            from mbt_adapter_base.calibration import Calibrator

            calibrator = Calibrator.from_json(calibrator_path.read_text())
        return H2OModel(
            model=imported, features=columns[1:], target=columns[0], calibrator=calibrator
        )
