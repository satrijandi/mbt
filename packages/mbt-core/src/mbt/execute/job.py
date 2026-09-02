"""The training-job entrypoint: ``python -m mbt.execute.job <job.json>`` (TSD §10.3).

Everything framework- or data-heavy happens here, inside a subprocess (or a
remote job in v1): hooks, AUTO resolution, tuning, training, metric
computation for challenger *and* champion, artifact export, tracking logging.
No registry access, no gate decisions - core compares, jobs compute.

The result is written to ``<job.json>.result.json``; stdout carries the
JSON event stream the coordinator forwards.
"""

import contextlib
import json
import os
import sys
import traceback
from dataclasses import dataclass
from pathlib import Path
from typing import Any

import jinja2
import pyarrow as pa

from mbt.adapters.registry import get_registry
from mbt.contracts import (
    AUTO,
    AdapterRef,
    BootstrapDelta,
    DatasetProfile,
    HookContext,
    JobResult,
    MetricResults,
    MetricSpec,
    ModelSpec,
    MonitorStats,
    PredictionRunInfo,
    RunContext,
    ScoringSpec,
    SplitStrategy,
    TrainingJob,
    TuningResult,
)
from mbt.events import EventBus, JsonLinesSink, get_bus, set_bus
from mbt.events.models import AutoResolved, LogMessage
from mbt.exceptions import AdapterError, ConfigError, MbtError
from mbt.execute.handles import TransformedDatasetHandle
from mbt.quality.hooks import ModelHooks, load_hooks
from mbt.runtime import normalized_adapter_config
from mbt.secrets import taint
from mbt.storage import artifact_store_for
from mbt_adapter_base.datasets import InMemoryDatasetHandle

#: Fraction of the train window carved as implicit validation (TSD §13.5).
_IMPLICIT_VALIDATION_FRACTION = 0.2

#: Fraction of the train window carved as the dedicated calibration slice (F17).
#: The calibrator must fit on rows the model never trained on AND that no
#: selection step (early stopping, tuning) optimized against, so it gets its
#: own slice rather than reusing the validation split.
_CALIBRATION_FRACTION = 0.2

#: Robust tuning bootstrap (R2-7): fewer resamples than a gate's 1000 because it
#: runs once per trial during the search; the one-sided confidence matches the
#: champion-gate default.
_TUNING_BOOTSTRAP_RESAMPLES = 200
_TUNING_BOOTSTRAP_CONFIDENCE = 0.95


def _render_adapter_ref(ref: AdapterRef, job_vars: dict[str, Any]) -> AdapterRef:
    """Re-render env_var()/env()/var() in an unrendered adapter config (TSD §18)."""
    jinja_env = jinja2.Environment(undefined=jinja2.StrictUndefined, autoescape=False)

    def lookup_env(name: str, default: str | None, *, secret: bool) -> str:
        value = os.environ.get(name)
        if value is None:
            if default is None:
                raise ConfigError(
                    f"environment variable {name!r} required by the job is not set",
                    hint="the job re-resolves secrets from its own environment (TSD §18)",
                )
            return default
        return taint(value) if secret else value

    def env_var(name: str, default: str | None = None) -> str:
        return lookup_env(name, default, secret=True)

    def env(name: str, default: str | None = None) -> str:
        """Non-secret environment value: same lookup, no taint."""
        return lookup_env(name, default, secret=False)

    def var(name: str, default: Any = None) -> Any:
        return job_vars.get(name, default)

    def render(value: Any) -> Any:
        if isinstance(value, str) and ("{{" in value or "{%" in value):
            return jinja_env.from_string(value).render(env_var=env_var, env=env, var=var)
        if isinstance(value, dict):
            return {k: render(v) for k, v in value.items()}
        if isinstance(value, list):
            return [render(v) for v in value]
        return value

    return AdapterRef(adapter=ref.adapter, config=render(ref.config))


@dataclass
class _JobRuntime:
    job: TrainingJob
    spec: ModelSpec
    adapter: Any
    handle: Any  # what the adapter reads (transformed, or a path materialization)
    transformed: TransformedDatasetHandle  # always the lazy transformed view
    base_handle: Any  # pre-transform materialization (carries the time_column)
    base_profile: DatasetProfile
    hooks: ModelHooks | None
    builtin_specs: list[MetricSpec]
    hook_specs: list[MetricSpec]
    ctx: RunContext
    store: Any


def run_job(job: TrainingJob) -> JobResult:
    """Execute one training/evaluation/scoring job; never raises (returns errors)."""
    tracking = None
    run_handle = None
    try:
        if job.mode == "score":
            return _run_score(job)
        runtime = _prepare(job)
        if job.tracking is not None and job.mode == "train":
            tracking = _tracking_adapter(job)
            run_handle = tracking.start_run(job.node, dict(job.tracking_meta))

        if job.mode == "evaluate":
            return _run_evaluate(runtime)

        result = _run_train(runtime, tracking, run_handle)
        if tracking is not None and run_handle is not None:
            tracking.end_run(run_handle, "FINISHED")
        return result
    except MbtError as exc:
        _best_effort_fail(tracking, run_handle)
        return JobResult(status="error", error=str(exc))
    except Exception as exc:
        _best_effort_fail(tracking, run_handle)
        tail = traceback.format_exc(limit=8)
        return JobResult(status="error", error=f"{exc!r}\n{tail}")


def _tracking_adapter(job: TrainingJob) -> Any:
    assert job.tracking is not None
    rendered = _render_adapter_ref(job.tracking, _job_vars(job))
    config = normalized_adapter_config(rendered, Path(job.project_dir))
    return get_registry().component("tracking", rendered.adapter, config)


def _job_vars(job: TrainingJob) -> dict[str, Any]:
    return dict(job.vars)


def _materialize_for_path_adapter(handle: Any, spec: ModelSpec) -> Any:
    """Write a handle's transformed splits to parquet for adapters that
    declare ``data_access == "path"`` (JVM/cluster frameworks ingest files
    natively). Hooks and feature selection are already applied by the
    transformed handle, so path adapters see exactly what arrow adapters see.
    """
    import atexit
    import shutil
    import tempfile

    import pyarrow.parquet as pq

    from mbt_adapter_base.materialization import (
        MaterializedDatasetHandle,
        write_materialization_metadata,
    )

    directory = Path(tempfile.mkdtemp(prefix="mbt-path-data-"))
    # The handle reads these files for the rest of the job; the job process
    # is short-lived, so process exit is the natural cleanup point.
    atexit.register(shutil.rmtree, directory, ignore_errors=True)
    counts: dict[str, int] = {}
    for split in sorted(handle.splits()):
        table = handle.read(split)
        pq.write_table(table, directory / f"{split}.parquet")
        counts[split] = table.num_rows
    write_materialization_metadata(
        directory,
        snapshot_id=handle.snapshot_id,
        dataset=spec.name,
        label_column=spec.target,
        time_column=None,  # split time columns never reach adapters
        windows={},
        sample_fraction=1.0,
        row_counts=counts,
    )
    return MaterializedDatasetHandle(directory)


def _prepare(job: TrainingJob) -> _JobRuntime:
    registry = get_registry()
    spec = ModelSpec.model_validate(job.node.config)

    data_ref = _render_adapter_ref(job.data, _job_vars(job))
    data_adapter = registry.component(
        "data", data_ref.adapter, normalized_adapter_config(data_ref, Path(job.project_dir))
    )
    base_handle = data_adapter.from_locator(job.dataset)
    base_profile = base_handle.profile()

    plugin = registry.get(spec.adapter)
    if plugin.training is None:
        raise AdapterError(f"adapter {spec.adapter!r} provides no training adapter")
    adapter = plugin.training({})

    # Run-time task validation now that the dataset profile exists (TSD §5.6).
    from mbt.config.tasks import get_task_schema

    issues = get_task_schema(spec.task).validate_dataset(spec, base_profile)
    errors = [i for i in issues if i.severity == "error"]
    for issue in issues:
        if issue.severity == "warning":
            get_bus().emit(
                LogMessage(level="warn", unique_id=job.node.unique_id, message=issue.message)
            )
    if errors:
        raise ConfigError(
            "; ".join(i.message for i in errors),
            resource=job.node.unique_id,
            hint=errors[0].hint,
        )

    hooks: ModelHooks | None = None
    if job.node.hooks_path is not None:
        hooks = load_hooks(Path(job.project_dir), job.node.hooks_path)

    ctx = RunContext(
        run_id=job.run_id,
        unique_id=job.node.unique_id,
        seed=spec.seed,
        target_name=job.target_name,
        project_dir=job.project_dir,
        vars=_job_vars(job),
        events=get_bus(),
    )

    def hook_ctx(split: str) -> HookContext:
        return HookContext(spec=spec, profile=base_profile, split=split, logger=get_bus())

    time_column = getattr(base_handle, "time_column", None)
    transformed = TransformedDatasetHandle(base_handle, spec, hooks, hook_ctx, time_column)
    handle: Any = transformed
    if getattr(adapter, "data_access", "arrow") == "path":
        handle = _materialize_for_path_adapter(transformed, spec)

    return _JobRuntime(
        job=job,
        spec=spec,
        adapter=adapter,
        handle=handle,
        transformed=transformed,
        base_handle=base_handle,
        base_profile=base_profile,
        hooks=hooks,
        builtin_specs=[m for m in job.metric_specs if m.kind == "builtin"],
        hook_specs=[m for m in job.metric_specs if m.kind == "hook"],
        ctx=ctx,
        store=artifact_store_for(job.artifact_store, run_prefix=_store_prefix(job)),
    )


def _store_prefix(job: TrainingJob) -> str:
    return f"{job.node.name}/{job.run_id}"


# -- metric computation ---------------------------------------------------------


def _metrics_for(
    runtime: _JobRuntime, model: Any, split: str, *, with_slices: bool
) -> MetricResults:
    """Builtin metrics via the adapter, hook metrics via predict + hooks."""
    slices = runtime.spec.evaluation.slices if with_slices else []
    results: MetricResults = runtime.adapter.evaluate(
        model, runtime.handle, split, runtime.builtin_specs, slices=slices or None
    )
    if runtime.hook_specs:
        if runtime.hooks is None or not runtime.hooks.has_custom_metrics:
            raise ConfigError(
                "hook metrics declared but hooks.py exposes no custom_metrics",
                resource=runtime.job.node.unique_id,
            )
        predictions: pa.Table = runtime.adapter.predict(model, runtime.handle, split)
        hook_ctx = HookContext(
            spec=runtime.spec, profile=runtime.base_profile, split=split, logger=get_bus()
        )
        computed = runtime.hooks.custom_metrics(predictions, hook_ctx)
        missing = [m.name for m in runtime.hook_specs if m.name not in computed]
        if missing:
            raise ConfigError(
                f"hooks custom_metrics did not return declared metric(s): {', '.join(missing)}",
                resource=runtime.job.node.unique_id,
            )
        merged = dict(results.metrics)
        merged.update({m.name: computed[m.name] for m in runtime.hook_specs})
        results = MetricResults(metrics=merged, slices=results.slices)
    return results


def _export_baseline(runtime: _JobRuntime, model: Any) -> Any:
    """Build + export the monitoring baseline next to the model artifact (ADR-21).

    Post-hook train-split feature distributions plus the test-split score
    distribution: everything scoring-time shift monitors compare against.
    Built unconditionally on every training job - it is cheap (quantiles),
    and champions registered without one cannot be monitored later.
    """
    import tempfile

    from mbt_adapter_base.monitoring import build_baseline, write_baseline

    train = runtime.transformed.read("train")
    feature_columns = runtime.transformed.feature_columns or []
    predictions: pa.Table = runtime.adapter.predict(model, runtime.handle, "test")
    scores = predictions.column("prediction").to_numpy(zero_copy_only=False)
    baseline = build_baseline(train, feature_columns, scores, model_name=runtime.spec.name)
    with tempfile.TemporaryDirectory(prefix="mbt-baseline-") as staging:
        path = Path(staging) / "baseline.json"
        write_baseline(baseline, path)
        return runtime.store.put_file(path, "baseline.json", format="json")


def _feature_importance(runtime: _JobRuntime, model: Any) -> dict[str, float]:
    """Per-feature importance for the model card (FR-DOCS-02).

    Prefers a data-grounded SHAP importance when the adapter exposes it (the
    tree adapters), since split-gain is cardinality-biased; falls back to the
    adapter's model-intrinsic ``feature_importance``. Optional per adapter, like
    ``log_trial`` on trackers: absence simply leaves cards without the table.
    """
    adapter = runtime.adapter
    if hasattr(adapter, "shap_importance"):
        return dict(adapter.shap_importance(model, runtime.handle, "test"))
    if hasattr(adapter, "feature_importance"):
        return dict(adapter.feature_importance(model))
    return {}


def _partial_dependence(
    runtime: _JobRuntime, model: Any, importance: dict[str, float], *, top_n: int = 3
) -> dict[str, list[list[float]]]:
    """Partial dependence for the ``top_n`` most-important NUMERIC features: how
    the average prediction moves as each feature sweeps its range while the rest
    stay at their observed values (explainability). Adapter-agnostic - it
    re-predicts on the test split with one feature overridden across a quantile
    grid. Categorical features are skipped (their PD is a different idiom)."""
    import numpy as np
    import pyarrow as pa
    import pyarrow.compute as pc

    from mbt_adapter_base.datasets import InMemoryDatasetHandle

    table = runtime.handle.read("test")
    curves: dict[str, list[list[float]]] = {}
    for feature in sorted(importance, key=lambda name: -importance[name]):
        if len(curves) >= top_n or feature not in table.column_names:
            continue
        column = table.column(feature)
        is_int = pa.types.is_integer(column.type)
        if not (is_int or pa.types.is_floating(column.type)):
            continue  # numeric features only
        values = np.asarray(column.to_numpy(zero_copy_only=False), dtype=float)
        grid = np.unique(np.quantile(values, np.linspace(0.0, 1.0, 8)))
        index = table.column_names.index(feature)
        curve: list[list[float]] = []
        for point in grid:
            cell = round(float(point)) if is_int else float(point)
            override = table.set_column(
                index, feature, pa.array([cell] * table.num_rows, type=column.type)
            )
            handle = InMemoryDatasetHandle({"pd": override})
            predicted = runtime.adapter.predict(model, handle, "pd").column("prediction")
            curve.append([round(float(point), 6), round(float(pc.mean(predicted).as_py()), 6)])
        curves[feature] = curve
    return curves


def _champion_delta_bounds(
    runtime: _JobRuntime, challenger: Any, champion: Any
) -> dict[str, BootstrapDelta]:
    """Paired-bootstrap lower bounds for champion-gate deltas (ADR-18).

    Whole-split builtin-metric gates only: slice and hook-metric gates keep
    point comparisons. Both models score the same resampled rows of the
    pinned test split, so only the model difference remains in the deltas.
    """
    from mbt.quality.metrics import metric_direction
    from mbt_adapter_base.metrics import paired_bootstrap_delta

    spec = runtime.spec
    builtin_by_name = {m.name: m for m in runtime.builtin_specs}
    gates = [
        gate
        for gate in spec.evaluation.gates
        if gate.compare_to is not None
        and gate.confidence is not None
        and gate.slice is None
        and gate.metric in builtin_by_name
    ]
    if not gates:
        return {}
    challenger_table = runtime.adapter.predict(challenger, runtime.handle, "test")
    champion_table = runtime.adapter.predict(champion, runtime.handle, "test")
    y_true = challenger_table.column(spec.target).to_numpy(zero_copy_only=False).astype("float64")
    challenger_scores = challenger_table.column("prediction").to_numpy(zero_copy_only=False)
    champion_scores = champion_table.column("prediction").to_numpy(zero_copy_only=False)
    bounds: dict[str, BootstrapDelta] = {}
    for gate in gates:
        confidence = gate.confidence
        if confidence is None:  # narrowed out above; keeps mypy strict happy
            continue  # pragma: no cover - gates are pre-filtered to confidence is not None
        bounds[gate.metric] = paired_bootstrap_delta(
            builtin_by_name[gate.metric],
            y_true,
            challenger_scores,
            champion_scores,
            greater_is_better=metric_direction(gate.metric, runtime.job.metric_specs),
            confidence=confidence,
            n_resamples=gate.bootstrap_resamples,
            seed=spec.seed + 3,  # seed ladder: train, +1 tuning, +2 validation
            # carve, +3 bootstrap, +4 random k-fold, +5 calibration carve
        )
    return bounds


# -- tuning (TSD §13.5, ADR-8) ----------------------------------------------------


def _tail_carve_indices(
    raw_train: pa.Table,
    *,
    time_column: str | None,
    windows: dict[str, Any],
    time_range: tuple[Any, Any] | None,
    seed: int,
    fraction: float,
) -> tuple[list[int], list[int]]:
    """``(fit_idx, held_idx)`` slicing the train rows for a carve: the tail
    ``fraction`` of the train time span when the split is temporal, else a
    seeded random ``fraction``. Shared by the implicit-validation carve
    (``seed+2``) and the calibration carve (``seed+5``) so the two slice train
    by identical rules and differ only in seed and destination split."""
    if time_column and (time_range is not None or "train" in windows):
        from datetime import datetime

        def _naive(value: Any) -> Any:
            if isinstance(value, datetime):
                return value.replace(tzinfo=None)
            if hasattr(value, "isoformat") and not isinstance(value, datetime):  # date
                return datetime(value.year, value.month, value.day)
            return value  # pragma: no cover - split time columns are temporal by construction

        if time_range is not None:
            start, end = _naive(time_range[0]), _naive(time_range[1])
        else:
            start = datetime.fromisoformat(str(windows["train"][0]).replace("Z", "+00:00"))
            end = datetime.fromisoformat(str(windows["train"][1]).replace("Z", "+00:00"))
        boundary = _naive(start) + (_naive(end) - _naive(start)) * (1 - fraction)
        column = raw_train.column(time_column).to_pylist()
        fit_idx = [i for i, v in enumerate(column) if _naive(v) < boundary]
        held_idx = [i for i, v in enumerate(column) if _naive(v) >= boundary]
    else:
        import numpy as np

        rng = np.random.RandomState(seed)
        permutation = rng.permutation(raw_train.num_rows)
        cut = int(raw_train.num_rows * (1 - fraction))
        fit_idx = sorted(int(i) for i in permutation[:cut])
        held_idx = sorted(int(i) for i in permutation[cut:])
    return fit_idx, held_idx


def _carve_validation(
    runtime: _JobRuntime, *, time_range: tuple[Any, Any] | None = None
) -> TransformedDatasetHandle | None:
    """A handle whose train/validation splits tuning may see; test never (ADR-8).

    Returns None when the dataset already declares a validation split.
    ``time_range`` (nested temporal CV) overrides the declared train window with
    the fold's own ``[start, end]`` so the carve is fold-aware; the main path
    leaves it None and uses the declared window unchanged.
    """
    base = runtime.transformed
    if "validation" in base.splits():
        return None

    job = runtime.job
    spec = runtime.spec
    raw_train = base._base.read("train")
    time_column = getattr(base._base, "time_column", None)
    windows = job.dataset_windows.get("windows", job.dataset_windows)

    fit_idx, val_idx = _tail_carve_indices(
        raw_train,
        time_column=time_column,
        windows=windows,
        time_range=time_range,
        seed=spec.seed + 2,
        fraction=_IMPLICIT_VALIDATION_FRACTION,
    )
    if not fit_idx or not val_idx:
        raise ConfigError(
            "implicit validation carve produced an empty split",
            resource=job.node.unique_id,
            hint="declare an explicit validation split in the dataset",
        )
    carved = InMemoryDatasetHandle(
        {
            "train": raw_train.take(fit_idx),
            "validation": raw_train.take(val_idx),
        },
        snapshot_id=base.snapshot_id,
        label_column=spec.target,
        time_column=time_column,
    )

    def hook_ctx(split: str) -> HookContext:
        return HookContext(spec=spec, profile=runtime.base_profile, split=split, logger=get_bus())

    return TransformedDatasetHandle(carved, spec, runtime.hooks, hook_ctx, time_column)


def _carve_calibration(
    runtime: _JobRuntime,
    spec: ModelSpec,
    base: TransformedDatasetHandle,
    *,
    time_range: tuple[Any, Any] | None = None,
) -> TransformedDatasetHandle:
    """A handle whose train is shrunk by a dedicated ``calibration`` slice (F17).

    The calibrator must fit on rows the model never trained on and that no
    selection step optimized against - fitting it on the ``validation`` split
    that early stopping and tuning select on makes the reported ``ece``/``brier``
    optimistic and overfits the deployed calibrator. So a spec that sets
    ``calibration`` always gets its own slice carved from train (temporal tail
    or seeded random ``seed+5``, mirroring the validation carve rules); every
    other split (test, a declared validation) passes through untouched.
    ``time_range`` makes the carve fold-aware inside the backtest (F5).
    """
    job = runtime.job
    raw_train = base._base.read("train")
    time_column = getattr(base._base, "time_column", None)
    windows = job.dataset_windows.get("windows", job.dataset_windows)

    fit_idx, cal_idx = _tail_carve_indices(
        raw_train,
        time_column=time_column,
        windows=windows,
        time_range=time_range,
        seed=spec.seed + 5,
        fraction=_CALIBRATION_FRACTION,
    )
    if not fit_idx or not cal_idx:
        raise ConfigError(
            "calibration carve produced an empty split",
            resource=job.node.unique_id,
            hint="the train split is too small to carve a calibration slice "
            "from; grow the train window or drop 'calibration'",
        )
    tables = {
        "train": raw_train.take(fit_idx),
        "calibration": raw_train.take(cal_idx),
    }
    for split in base._base.splits():
        if split not in ("train", "calibration"):
            tables[split] = base._base.read(split)
    carved = InMemoryDatasetHandle(
        tables,
        snapshot_id=base.snapshot_id,
        label_column=spec.target,
        time_column=time_column,
    )

    def hook_ctx(split: str) -> HookContext:
        return HookContext(spec=spec, profile=runtime.base_profile, split=split, logger=get_bus())

    return TransformedDatasetHandle(carved, spec, runtime.hooks, hook_ctx, time_column)


def _robust_objective_value(
    runtime: _JobRuntime, model: Any, spec: ModelSpec, objective_spec: MetricSpec
) -> float:
    """Bootstrap lower bound of the validation objective metric (R2-7): the
    tuning selection is then defended against validation-window luck, not made
    on a single-carve point estimate. Predicts once on validation and resamples
    the (label, score) rows with a fixed seed, so trials are compared on the same
    resamples and the search stays reproducible."""
    import numpy as np

    from mbt_adapter_base.metrics import bootstrap_metric_lower_bound

    predictions = runtime.adapter.predict(model, runtime.handle, "validation")
    y_score = np.asarray(
        predictions.column("prediction").to_numpy(zero_copy_only=False), dtype=float
    )
    y_true = np.asarray(predictions.column(spec.target).to_numpy(zero_copy_only=False), dtype=float)
    return bootstrap_metric_lower_bound(
        objective_spec,
        y_true,
        y_score,
        confidence=_TUNING_BOOTSTRAP_CONFIDENCE,
        n_resamples=_TUNING_BOOTSTRAP_RESAMPLES,
        seed=spec.seed + 3,
    )


def _run_tuning(
    runtime: _JobRuntime,
    spec: ModelSpec,
    tracking: Any = None,
    run_handle: Any = None,
    *,
    carve_time_range: tuple[Any, Any] | None = None,
) -> tuple[ModelSpec, TuningResult | None]:
    tuning = spec.tuning
    if tuning is None:
        return spec, None
    job = runtime.job
    if job.tuning_engine is None:
        raise ConfigError(
            f"model declares tuning but no tuning engine was provided for {tuning.engine!r}",
            resource=job.node.unique_id,
        )
    engine_ref = _render_adapter_ref(job.tuning_engine, _job_vars(job))
    engine = get_registry().component("tuning", engine_ref.adapter, engine_ref.config)

    carved = _carve_validation(runtime, time_range=carve_time_range)
    tune_handle: Any = carved or runtime.handle
    if carved is not None and getattr(runtime.adapter, "data_access", "arrow") == "path":
        tune_handle = _materialize_for_path_adapter(carved, spec)
    explicit_validation = "validation" in runtime.transformed.splits()

    objective_spec = next(
        (
            m
            for m in [*runtime.builtin_specs, *runtime.hook_specs]
            if m.name == tuning.objective.metric
        ),
        None,
    )
    if objective_spec is None:
        raise ConfigError(
            f"tuning objective {tuning.objective.metric!r} is not a resolved metric",
            resource=job.node.unique_id,
        )
    if tuning.objective.robust and objective_spec.kind != "builtin":
        raise ConfigError(
            f"tuning objective {tuning.objective.metric!r} is a hook metric; "
            "robust (bootstrap) selection supports builtin metrics only",
            resource=job.node.unique_id,
        )

    tune_runtime = _JobRuntime(**{**runtime.__dict__, "handle": tune_handle})
    trial_counter = {"index": 0}

    adapter_reports = hasattr(runtime.adapter, "train_with_report")
    if tuning.pruner is not None and not adapter_reports:
        get_bus().emit(
            LogMessage(
                level="warn",
                unique_id=job.node.unique_id,
                message=(
                    f"tuning.pruner {tuning.pruner!r} is set but adapter "
                    f"{spec.adapter!r} does not report training progress "
                    "(no train_with_report); trials run to completion"
                ),
            )
        )

    def objective(trial_params: dict[str, Any], report: Any = None) -> float:
        # Trials never calibrate: the objective is scored on the same
        # validation split a trial calibrator would fit on, which would make a
        # calibration-sensitive objective (brier/ece) circularly optimal (F17).
        # The final fit calibrates on its own dedicated slice instead.
        trial_spec = spec.model_copy(
            update={
                "hyperparameters": {**spec.hyperparameters, **trial_params},
                "calibration": None,
            }
        )
        if report is not None and adapter_reports:
            # A pruning report may raise out of the training loop; the engine
            # owns that exception and marks the trial pruned.
            model = runtime.adapter.train_with_report(trial_spec, tune_handle, runtime.ctx, report)
        else:
            model = runtime.adapter.train(trial_spec, tune_handle, runtime.ctx)
        if tuning.objective.robust:
            value = _robust_objective_value(tune_runtime, model, spec, objective_spec)
        else:
            results = _metrics_for(tune_runtime, model, "validation", with_slices=False)
            value = float(results.metrics[tuning.objective.metric])
        index = trial_counter["index"]
        trial_counter["index"] += 1
        # Per-trial progress on the bus at debug (visible under --verbose or
        # --log-format json) so a long search is not opaque while it runs.
        get_bus().emit(
            LogMessage(
                level="debug",
                unique_id=job.node.unique_id,
                message=f"tuning trial {index}: {tuning.objective.metric}={value:.4f}",
            )
        )
        # Trial history as nested tracking runs where the adapter supports it
        # (FR-TUNE-03); optional so simple trackers keep working.
        if tracking is not None and run_handle is not None and hasattr(tracking, "log_trial"):
            tracking.log_trial(run_handle, index, trial_params, value)
        return value

    n_trials = tuning.n_trials
    if job.tuning_cap is not None:
        n_trials = min(n_trials, job.tuning_cap)
        if n_trials < tuning.n_trials:
            get_bus().emit(
                LogMessage(
                    unique_id=job.node.unique_id,
                    message=(
                        f"tuning capped at {n_trials} trial(s) by the target's "
                        f"max_tuning_trials (requested {tuning.n_trials}) (FR-TUNE-04)"
                    ),
                )
            )
    result = engine.tune(tuning, objective, n_trials=n_trials, seed=spec.seed + 1)

    # Final fit reabsorbs an *implicit* carve; an explicit validation split
    # stays held out because the user declared it (TSD §10.5 step 6).
    _ = explicit_validation
    tuned = spec.model_copy(
        update={"hyperparameters": {**spec.hyperparameters, **result.best_params}}
    )
    return tuned, result


# -- main paths -------------------------------------------------------------------


def _backtest_folds(
    runtime: _JobRuntime, spec: ModelSpec, base_train: pa.Table, n_folds: int
) -> list[tuple[pa.Table, pa.Table]]:
    """(train_rows, test_rows) fold pairs for the backtest, by split strategy:
    time-ordered expanding prefixes for a temporal split (train on the past,
    evaluate on the next fold), or random leave-one-fold-out k-fold for a random
    split (each fold is the test set once, train on the rest). The temporal path
    needs the base handle's time column (the transformed view strips it); the
    random fold assignment is seeded off the model seed for reproducibility.

    A temporal ``split.embargo`` (R2-7) gaps each fold's train tail from its test
    window exactly as it gaps the single train/test split - without it the
    walk-forward backtest leaks at every fold boundary (F6). A fold whose entire
    (earliest, shortest) train prefix falls inside the embargo has no
    leakage-free history to train on and is dropped."""
    import numpy as np
    import pyarrow.compute as pc

    if spec.evaluation.protocol.split is SplitStrategy.TEMPORAL:
        time_column = getattr(runtime.base_handle, "time_column", None)
        if time_column is None or time_column not in base_train.column_names:
            return []
        ordered = base_train.take(
            pc.sort_indices(base_train, sort_keys=[(time_column, "ascending")])
        )
        edges = [round(i * ordered.num_rows / n_folds) for i in range(n_folds + 1)]
        embargo = runtime.job.dataset_windows.get("embargo")
        if embargo is None:
            return [
                (ordered.slice(0, edges[i]), ordered.slice(edges[i], edges[i + 1] - edges[i]))
                for i in range(1, n_folds)
            ]
        # Embargo each internal fold boundary: drop training rows whose label
        # horizon could reach into that fold's test window (mirrors the
        # single-split embargo in compile/compiler.py).
        from mbt.compile.windows import subtract_duration

        times = ordered.column(time_column)
        folds: list[tuple[pa.Table, pa.Table]] = []
        for i in range(1, n_folds):
            cutoff = subtract_duration(times[edges[i]].as_py(), embargo)
            prefix = ordered.slice(0, edges[i])
            prefix = prefix.filter(
                pc.less(prefix.column(time_column), pa.scalar(cutoff, type=times.type))
            )
            if prefix.num_rows == 0:
                continue  # the embargo consumed this fold's entire train prefix
            folds.append((prefix, ordered.slice(edges[i], edges[i + 1] - edges[i])))
        return folds
    # random split: k-fold cross-validation (each fold is held out as the test set)
    n = base_train.num_rows
    perm = np.random.default_rng(spec.seed + 4).permutation(n)
    edges = [round(i * n / n_folds) for i in range(n_folds + 1)]
    return [
        (
            base_train.take(np.concatenate([perm[: edges[i]], perm[edges[i + 1] :]])),
            base_train.take(perm[edges[i] : edges[i + 1]]),
        )
        for i in range(n_folds)
    ]


def _walk_forward_backtest(
    runtime: _JobRuntime, spec: ModelSpec, n_folds: int, *, nested: bool = False
) -> tuple[dict[str, float], dict[str, float]]:
    """Fold-based cross-validated evaluation over the training window (R2-7):
    time-ordered walk-forward for a temporal split, random k-fold for a random
    split (see ``_backtest_folds``). Returns ``(means, stds)`` of each builtin
    metric across the folds: the mean is the de-luckified point estimate (a
    single lucky split cannot flatter it) and the population std is the standard
    CV stability signal. Report-only unless a gate uses ``source: backtest``.

    With ``nested`` (nested CV), each fold re-runs ``_run_tuning`` on the fold's
    train only and refits with the fold-tuned params, so the estimate is unbiased
    for the TUNED model - the outer-test fold never informs the selection. Without
    it, refits use ``spec`` (already AUTO-resolved and tuned). Either way the fold
    goes through the SAME ``TransformedDatasetHandle`` pipeline as the real fit.
    Returns {} when there is too little data (or, for a temporal split, no usable
    time column) to form a fold.
    """
    from mbt_adapter_base.datasets import InMemoryDatasetHandle

    base_train = runtime.base_handle.read("train")
    folds = _backtest_folds(runtime, spec, base_train, n_folds)
    if not folds:
        return {}, {}
    time_column = getattr(runtime.base_handle, "time_column", None)

    def hook_ctx(split: str) -> HookContext:
        return HookContext(spec=spec, profile=runtime.base_profile, split=split, logger=get_bus())

    per_fold: dict[str, list[float]] = {}
    for train_rows, test_rows in folds:
        if train_rows.num_rows == 0 or test_rows.num_rows == 0:
            continue
        base = InMemoryDatasetHandle(
            {"train": train_rows, "test": test_rows},
            label_column=spec.target,
            time_column=time_column,
        )
        handle = TransformedDatasetHandle(base, spec, runtime.hooks, hook_ctx, time_column)

        def fold_time_range(rows: pa.Table = train_rows) -> tuple[Any, Any] | None:
            if time_column is None or time_column not in rows.column_names:
                return None
            # a temporal fold's carve must use THIS fold's time span (an
            # expanding prefix), not the whole train window
            import pyarrow.compute as pc

            col = rows.column(time_column)
            return (pc.min(col).as_py(), pc.max(col).as_py())

        fold_spec = spec
        if nested:
            # Nested CV: re-tune on this fold's train only, so the outer-test fold
            # never informs the selection (an unbiased estimate of the tuning).
            fold_runtime = _JobRuntime(
                **{
                    **runtime.__dict__,
                    "handle": handle,
                    "transformed": handle,
                    "base_handle": base,
                    "base_profile": base.profile(),
                }
            )
            fold_spec, _ = _run_tuning(fold_runtime, spec, carve_time_range=fold_time_range())
        fit_handle = handle
        if spec.calibration is not None:
            # Each fold calibrates on its own carved slice, exactly as the
            # production fit does, so a `source: backtest` gate compares
            # calibrated fold models against a calibrated final model (F5).
            fit_handle = _carve_calibration(
                runtime, fold_spec, handle, time_range=fold_time_range()
            )
        fold_model = runtime.adapter.train(fold_spec, fit_handle, runtime.ctx)
        result = runtime.adapter.evaluate(fold_model, handle, "test", runtime.builtin_specs)
        for name, value in result.metrics.items():
            per_fold.setdefault(name, []).append(float(value))
    if not per_fold:
        return {}, {}
    import statistics

    # mean is the de-luckified point estimate; the population std across folds is
    # the standard CV stability signal (a big std means the mean is not to be
    # trusted - the model's score swings with the split).
    means = {name: round(statistics.fmean(vals), 6) for name, vals in per_fold.items()}
    stds = {name: round(statistics.pstdev(vals), 6) for name, vals in per_fold.items()}
    return means, stds


def _run_train(runtime: _JobRuntime, tracking: Any, run_handle: Any) -> JobResult:
    job = runtime.job
    bus = get_bus()

    # 1. AUTO resolution from the dataset profile (FR-RES-10)
    spec = runtime.adapter.resolve_auto(runtime.spec, runtime.base_profile)
    resolved_auto = {
        key: spec.hyperparameters[key]
        for key, value in runtime.spec.hyperparameters.items()
        if value == AUTO
    }
    leftover = [k for k, v in spec.hyperparameters.items() if v == AUTO]
    if leftover:
        raise AdapterError(
            f"adapter left AUTO sentinels unresolved: {', '.join(leftover)}",
            resource=job.node.unique_id,
        )
    for key, value in resolved_auto.items():
        bus.emit(AutoResolved(unique_id=job.node.unique_id, param=key, value=str(value)))

    # Nested CV re-tunes per fold, so it needs the pre-tuning (search-space) spec.
    pre_tuning_spec = spec

    # 2. tuning (never sees the test split, ADR-8)
    spec, tuning_result = _run_tuning(runtime, spec, tracking, run_handle)
    tuning_cfg = runtime.spec.tuning
    if tuning_result is not None and tuning_cfg is not None:
        # A tuning search is otherwise silent on success (trial history goes
        # only to the tracking backend); surface a one-line summary.
        bus.emit(
            LogMessage(
                unique_id=job.node.unique_id,
                message=(
                    f"tuning complete: {tuning_result.n_trials} trial(s), "
                    f"{tuning_result.n_pruned} pruned, best "
                    f"{tuning_cfg.objective.metric}={tuning_result.best_value:.4f}"
                ),
            )
        )

    # 3. final fit on the declared train window; a calibrated spec fits on
    # train minus a dedicated calibration slice so the calibrator never sees
    # training rows nor the validation split selection optimized against (F17)
    fit_handle = runtime.handle
    if spec.calibration is not None:
        carved = _carve_calibration(runtime, spec, runtime.transformed)
        fit_handle = carved
        if getattr(runtime.adapter, "data_access", "arrow") == "path":
            fit_handle = _materialize_for_path_adapter(carved, spec)
    model = runtime.adapter.train(spec, fit_handle, runtime.ctx)

    # 4. evaluate challenger and (if provided) champion on the SAME test split
    challenger = _metrics_for(runtime, model, "test", with_slices=True)
    champion_metrics: MetricResults | None = None
    delta_bounds: dict[str, BootstrapDelta] = {}
    if job.champion is not None:
        champion_model = runtime.adapter.load(job.champion, runtime.store)
        champion_metrics = _metrics_for(runtime, champion_model, "test", with_slices=True)
        delta_bounds = _champion_delta_bounds(runtime, model, champion_model)

    # 5. export the artifact, plus the monitoring baseline (ADR-21)
    artifact = runtime.adapter.export(model, "native", runtime.store)
    baseline = _export_baseline(runtime, model)
    importance = _feature_importance(runtime, model)
    partial_dependence = _partial_dependence(runtime, model, importance)
    backtest_folds = spec.evaluation.protocol.backtest_folds
    nested = spec.evaluation.protocol.nested_cv
    backtest_metrics, backtest_std = (
        _walk_forward_backtest(
            runtime, pre_tuning_spec if nested else spec, backtest_folds, nested=nested
        )
        if backtest_folds is not None
        else ({}, {})
    )

    # 6. tracking: params, metrics, artifacts, tuning history
    tracking_run_id: str | None = None
    if tracking is not None and run_handle is not None:
        tracking_run_id = run_handle.run_id
        params = {k: str(v) for k, v in spec.hyperparameters.items()}
        params["seed"] = str(spec.seed)
        tracking.log(
            run_handle,
            params=params,
            metrics=dict(challenger.metrics),
            artifacts=[artifact],
        )
        if tuning_result is not None:
            tracking.log(
                run_handle,
                metrics={"tuning.best_value": tuning_result.best_value},
                tags={
                    "mbt.tuning.n_trials": str(tuning_result.n_trials),
                    "mbt.tuning.n_pruned": str(tuning_result.n_pruned),
                    "mbt.tuning.best_params": json.dumps(tuning_result.best_params),
                },
            )

    return JobResult(
        status="success",
        metrics=challenger,
        champion_metrics=champion_metrics,
        champion_delta_bounds=delta_bounds,
        feature_importance=importance,
        partial_dependence=partial_dependence,
        backtest_metrics=backtest_metrics,
        backtest_std=backtest_std,
        resolved_auto=resolved_auto,
        tuning=tuning_result,
        artifact=artifact,
        baseline=baseline,
        tracking_run_id=tracking_run_id,
    )


def _run_score(job: TrainingJob) -> JobResult:
    """Score one batch with a registered champion (mode="score", ADR-20/21).

    The coordinator resolved the champion, baseline, and run_key; this job
    loads the artifact, applies the model's hooks + feature selection to the
    unlabeled input, predicts, computes shift statistics, and writes the
    prediction run. Core applies the monitor thresholds afterwards.
    """
    registry = get_registry()
    if job.model_node is None or job.output is None or job.run_key is None:
        raise ConfigError(
            "score mode requires model_node, output, and run_key",
            resource=job.node.unique_id,
        )
    if job.artifact is None:
        raise ConfigError(
            "score mode requires the champion artifact reference", resource=job.node.unique_id
        )
    scoring_spec = ScoringSpec.model_validate(job.node.config)
    model_spec = ModelSpec.model_validate(job.model_node.config)

    data_ref = _render_adapter_ref(job.data, _job_vars(job))
    data_adapter = registry.component(
        "data", data_ref.adapter, normalized_adapter_config(data_ref, Path(job.project_dir))
    )
    base_handle = data_adapter.from_locator(job.dataset)
    base_profile = base_handle.profile()

    plugin = registry.get(model_spec.adapter)
    if plugin.training is None:
        raise AdapterError(f"adapter {model_spec.adapter!r} provides no training adapter")
    adapter = plugin.training({})

    hooks: ModelHooks | None = None
    if job.model_node.hooks_path is not None:
        hooks = load_hooks(Path(job.project_dir), job.model_node.hooks_path)

    def hook_ctx(split: str) -> HookContext:
        return HookContext(spec=model_spec, profile=base_profile, split=split, logger=get_bus())

    time_column = getattr(base_handle, "time_column", None)
    transformed = TransformedDatasetHandle(
        base_handle, model_spec, hooks, hook_ctx, time_column, require_target=False
    )
    handle: Any = transformed
    if getattr(adapter, "data_access", "arrow") == "path":
        handle = _materialize_for_path_adapter(transformed, model_spec)

    raw = base_handle.read("score")
    passthrough = scoring_spec.passthrough_columns
    missing = [c for c in passthrough if c not in raw.column_names]
    if missing:
        raise ConfigError(
            f"passthrough column(s) missing from the scoring input: {', '.join(missing)}",
            resource=job.node.unique_id,
            hint="output.columns, ground_truth.join_key, and input.time_column "
            "must exist in the input table",
        )

    store = artifact_store_for(job.artifact_store, run_prefix=_store_prefix(job))
    model = adapter.load(job.artifact, store)

    if raw.num_rows > 0:
        predictions: pa.Table = adapter.predict(model, handle, "score")
        if predictions.num_rows != raw.num_rows:
            raise AdapterError(
                f"hooks changed the scoring input's row count "
                f"({raw.num_rows} rows in, {predictions.num_rows} predictions out)",
                resource=job.node.unique_id,
                hint="transform_features must be row-stable (no filtering or "
                "reordering) for scoring (ADR-20)",
            )
        prediction_column = predictions.column("prediction")
    else:
        prediction_column = pa.chunked_array([pa.array([], type=pa.float64())])
    output_table = raw.select(passthrough).append_column("prediction", prediction_column)
    if job.output.explain_top_k is not None:
        # Per-prediction local attribution (explainability): the top-k features
        # by |SHAP| for each row, as a JSON string, so a consumer can answer
        # "why did THIS row score this way".
        if not hasattr(adapter, "explain"):
            raise ConfigError(
                f"output.explain_top_k is set but the {model_spec.adapter!r} adapter does not "
                "support per-prediction SHAP explanations",
                resource=job.node.unique_id,
                hint="use a tree adapter (xgboost/lightgbm), or drop explain_top_k",
            )
        rows = (
            adapter.explain(model, handle, "score", job.output.explain_top_k)
            if raw.num_rows
            else []
        )
        output_table = output_table.append_column("explanation", pa.array(rows, type=pa.string()))
    # job.output is the coordinator-resolved output: a string decision_threshold
    # (a champion operating-point metric) has already been resolved to a float
    # from the champion's tags (R2-5), unlike the raw spec in node.config.
    threshold = job.output.decision_threshold
    if threshold is not None:
        # The deployable operating point (R2-5): emit a 0/1 decision alongside
        # the probability so downstream consumers get a decision rule.
        decision = (prediction_column.to_numpy(zero_copy_only=False) >= threshold).astype("int8")
        output_table = output_table.append_column("decision", pa.array(decision))

    stats: MonitorStats | None = None
    if scoring_spec.monitors is not None:
        if job.baseline is None:
            stats = MonitorStats(baseline_missing=True)
        elif raw.num_rows == 0:
            stats = MonitorStats()  # nothing to compare; the 0-row WARN already fired
        else:
            from mbt_adapter_base.monitoring import compute_monitor_stats, read_baseline

            baseline = read_baseline(store.fetch(job.baseline))
            scores = prediction_column.to_numpy(zero_copy_only=False)
            stats = compute_monitor_stats(
                baseline, transformed.read("score"), scores, scoring_spec.monitors
            )

    prediction_store = data_adapter.open_predictions(job.output)
    persisted: PredictionRunInfo = prediction_store.write_run(
        output_table,
        PredictionRunInfo(
            run_key=job.run_key,
            uri="",
            scored_at=job.tracking_meta.get("mbt.anchor", ""),
            run_id=job.run_id,
            model_name=model_spec.registration.name if model_spec.registration else model_spec.name,
            model_version=job.model_version or "",
            row_count=0,  # set by the store
            meta={
                "config_hash": job.node.config_hash,
                "input_hash": job.node.input_hash,
                "snapshot_id": job.node.snapshot_id or "",
                **({"decision_threshold": str(threshold)} if threshold is not None else {}),
            },
        ),
    )

    tracking_run_id: str | None = None
    if job.tracking is not None:
        tracking = _tracking_adapter(job)
        run_handle = tracking.start_run(job.node, dict(job.tracking_meta))
        tracking_run_id = run_handle.run_id
        metrics: dict[str, float] = {"predictions.rows": float(persisted.row_count)}
        if stats is not None:
            if stats.prediction_shift is not None:
                metrics["monitor.prediction_shift"] = stats.prediction_shift.value
            if stats.feature_shift:
                metrics["monitor.feature_shift.max"] = max(
                    s.value for s in stats.feature_shift.values()
                )
        tracking.log(
            run_handle,
            metrics=metrics,
            tags={
                "mbt.model_version": job.model_version or "",
                "mbt.run_key": persisted.run_key,
            },
        )
        tracking.end_run(run_handle, "FINISHED")

    return JobResult(
        status="success",
        predictions=persisted,
        monitor_stats=stats,
        tracking_run_id=tracking_run_id,
    )


def _run_evaluate(runtime: _JobRuntime) -> JobResult:
    job = runtime.job
    if job.artifact is None:
        raise ConfigError(
            "evaluate mode requires an artifact reference", resource=job.node.unique_id
        )
    model = runtime.adapter.load(job.artifact, runtime.store)
    results = _metrics_for(runtime, model, "test", with_slices=True)
    champion_metrics: MetricResults | None = None
    delta_bounds: dict[str, BootstrapDelta] = {}
    if job.champion is not None:
        champion_model = runtime.adapter.load(job.champion, runtime.store)
        champion_metrics = _metrics_for(runtime, champion_model, "test", with_slices=True)
        delta_bounds = _champion_delta_bounds(runtime, model, champion_model)
    return JobResult(
        status="success",
        metrics=results,
        champion_metrics=champion_metrics,
        champion_delta_bounds=delta_bounds,
        feature_importance=_feature_importance(runtime, model),
        artifact=job.artifact,
    )


def _best_effort_fail(tracking: Any, run_handle: Any) -> None:
    if tracking is not None and run_handle is not None:
        with contextlib.suppress(Exception):
            tracking.end_run(run_handle, "FAILED")


def main(argv: list[str] | None = None) -> int:
    args = argv if argv is not None else sys.argv[1:]
    if len(args) != 1:
        sys.stderr.write("usage: python -m mbt.execute.job <job.json>\n")
        return 2
    job_path = Path(args[0])
    set_bus(EventBus(sinks=[JsonLinesSink(sys.stdout)]))
    job = TrainingJob.model_validate_json(job_path.read_text())
    get_bus().run_id = job.run_id
    result = run_job(job)
    from mbt.adapters.local.compute import result_path_for
    from mbt.secrets import redact

    # The job taints its rendered adapter configs, so an exception message
    # embedding a credential (connection strings in tracebacks) is masked
    # before it ever reaches disk - same guarantee as the event sinks.
    result_path_for(job_path).write_text(redact(result.model_dump_json()))
    return 0 if result.status == "success" else 3


if __name__ == "__main__":
    raise SystemExit(main())
