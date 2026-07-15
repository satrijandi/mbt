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


def _render_adapter_ref(ref: AdapterRef, job_vars: dict[str, Any]) -> AdapterRef:
    """Re-render env_var()/var() in an unrendered adapter config (TSD §18)."""
    env = jinja2.Environment(undefined=jinja2.StrictUndefined, autoescape=False)

    def env_var(name: str, default: str | None = None) -> str:
        value = os.environ.get(name)
        if value is None:
            if default is None:
                raise ConfigError(
                    f"environment variable {name!r} required by the job is not set",
                    hint="the job re-resolves secrets from its own environment (TSD §18)",
                )
            return default
        return taint(value)

    def var(name: str, default: Any = None) -> Any:
        return job_vars.get(name, default)

    def render(value: Any) -> Any:
        if isinstance(value, str) and ("{{" in value or "{%" in value):
            return env.from_string(value).render(env_var=env_var, var=var)
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
    """Per-feature importance when the adapter exposes it (FR-DOCS-02).

    Optional per adapter, like ``log_trial`` on trackers: absence simply
    leaves model cards without an importance table.
    """
    if not hasattr(runtime.adapter, "feature_importance"):
        return {}
    return dict(runtime.adapter.feature_importance(model))


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
            seed=spec.seed + 3,  # seed: train, +1: tuning, +2: validation carve, +3: bootstrap
        )
    return bounds


# -- tuning (TSD §13.5, ADR-8) ----------------------------------------------------


def _carve_validation(runtime: _JobRuntime) -> TransformedDatasetHandle | None:
    """A handle whose train/validation splits tuning may see; test never (ADR-8).

    Returns None when the dataset already declares a validation split.
    """
    base = runtime.transformed
    if "validation" in base.splits():
        return None

    job = runtime.job
    spec = runtime.spec
    raw_train = base._base.read("train")
    time_column = getattr(base._base, "time_column", None)
    windows = job.dataset_windows.get("windows", job.dataset_windows)

    if time_column and "train" in windows:
        from datetime import datetime, timedelta

        start = datetime.fromisoformat(str(windows["train"][0]).replace("Z", "+00:00"))
        end = datetime.fromisoformat(str(windows["train"][1]).replace("Z", "+00:00"))
        boundary = (start + (end - start) * (1 - _IMPLICIT_VALIDATION_FRACTION)).replace(
            tzinfo=None
        )
        column = raw_train.column(time_column).to_pylist()

        def _naive(value: Any) -> Any:
            if isinstance(value, datetime):
                return value.replace(tzinfo=None)
            if hasattr(value, "isoformat") and not isinstance(value, datetime):  # date
                return datetime(value.year, value.month, value.day)
            return value  # pragma: no cover - split time columns are temporal by construction

        fit_idx = [i for i, v in enumerate(column) if _naive(v) < boundary]
        val_idx = [i for i, v in enumerate(column) if _naive(v) >= boundary]
        _ = timedelta  # keep import local and explicit
    else:
        import numpy as np

        rng = np.random.RandomState(spec.seed + 2)
        permutation = rng.permutation(raw_train.num_rows)
        cut = int(raw_train.num_rows * (1 - _IMPLICIT_VALIDATION_FRACTION))
        fit_idx = sorted(int(i) for i in permutation[:cut])
        val_idx = sorted(int(i) for i in permutation[cut:])

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


def _run_tuning(
    runtime: _JobRuntime,
    spec: ModelSpec,
    tracking: Any = None,
    run_handle: Any = None,
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

    carved = _carve_validation(runtime)
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
        trial_spec = spec.model_copy(
            update={"hyperparameters": {**spec.hyperparameters, **trial_params}}
        )
        if report is not None and adapter_reports:
            # A pruning report may raise out of the training loop; the engine
            # owns that exception and marks the trial pruned.
            model = runtime.adapter.train_with_report(trial_spec, tune_handle, runtime.ctx, report)
        else:
            model = runtime.adapter.train(trial_spec, tune_handle, runtime.ctx)
        results = _metrics_for(tune_runtime, model, "validation", with_slices=False)
        value = float(results.metrics[tuning.objective.metric])
        index = trial_counter["index"]
        trial_counter["index"] += 1
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

    # 2. tuning (never sees the test split, ADR-8)
    spec, tuning_result = _run_tuning(runtime, spec, tracking, run_handle)

    # 3. final fit on the declared train window
    model = runtime.adapter.train(spec, runtime.handle, runtime.ctx)

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
