"""The scoring, evaluation and after-test-check steps of a job."""

import contextlib
from pathlib import Path
from typing import Any

import pyarrow as pa

from mbt.adapters.registry import get_registry
from mbt.events import get_bus
from mbt.events.bus import HookEventSink
from mbt.exceptions import AdapterError, ConfigError
from mbt.execute import training_report as report_step
from mbt.execute.handles import TransformedDatasetHandle
from mbt.execute.job_metrics import _champion_delta_bounds, _metrics_for
from mbt.execute.job_runtime import JobRuntime
from mbt.execute.job_setup import (
    _job_vars,
    _materialize_for_path_adapter,
    _render_adapter_ref,
    _store_prefix,
    _tracking_adapter,
)
from mbt.quality.hooks import ModelHooks, load_hooks
from mbt.runtime import normalized_adapter_config
from mbt.storage import artifact_store_for
from mbt_adapter_base import (
    BootstrapDelta,
    HookContext,
    JobResult,
    MetricResults,
    ModelSpec,
    MonitorStats,
    PredictionRunInfo,
    ScoringSpec,
    TrainingJob,
)
from mbt_adapter_base.capabilities import Capability, supports


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
    # The champion's own exported spec wins over the working tree's (ADR-28);
    # the coordinator falls back to model_node.config, loudly, only for a
    # champion registered before mbt exported one.
    model_spec = ModelSpec.model_validate(job.champion_spec or job.model_node.config)

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
        return HookContext(
            spec=model_spec, profile=base_profile, split=split, logger=HookEventSink(get_bus())
        )

    time_column = getattr(base_handle, "time_column", None)
    transformed = TransformedDatasetHandle(
        base_handle,
        model_spec,
        hooks,
        hook_ctx,
        time_column,
        require_target=False,
        pinned_features=job.champion_feature_columns,
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
        if not supports(adapter, Capability.EXPLAIN, model_spec):
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
            scored_at=job.anchor,
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

    # No tracking run: scoring is inference, not an experiment (ADR-28). The
    # batch's row count, shift statistics, model version and run key all live
    # in the prediction sidecar this job just wrote, which is where an
    # operator asking about a batch already looks.
    return JobResult(status="success", predictions=persisted, monitor_stats=stats)


def _run_evaluate(runtime: JobRuntime) -> JobResult:
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
        champion_metrics = _metrics_for(
            runtime, champion_model, "test", with_slices=True, handle=runtime.champion_handle
        )
        delta_bounds = _champion_delta_bounds(runtime, model, champion_model)
    return JobResult(
        status="success",
        metrics=results,
        champion_metrics=champion_metrics,
        champion_delta_bounds=delta_bounds,
        feature_importance=report_step.feature_importance(runtime, model),
        artifact=job.artifact,
    )


def _run_oot_check(runtime: JobRuntime) -> JobResult:
    """The pre-deploy check (mode="oot_check", ADR-30).

    Loads a registered version, scores its recorded test window and the rows
    since, and appends the report to the version's training run under
    ``evaluations/<run_id>/``. It opens no run and logs no parameters: those
    are immutable, and this run's windows are not the training run's.
    """
    job = runtime.job
    if job.artifact is None:
        raise ConfigError(
            "oot_check mode requires the version's artifact reference",
            resource=job.node.unique_id,
        )
    model = runtime.adapter.load(job.artifact, runtime.store)
    tracking = None
    run_handle = None
    if job.tracking is not None and job.tracking_run_id:
        tracking = _tracking_adapter(job)
        run_handle = tracking.resume(job.tracking_run_id)
    summary = report_step.produce_report(
        runtime,
        runtime.spec,
        model,
        kind="check",
        stage=_materialize_for_path_adapter,
        include_train=False,
        tracking=tracking,
        run_handle=run_handle,
        artifact_path=f"evaluations/{job.run_id}",
    ).summary
    if tracking is not None and run_handle is not None:
        metrics = report_step.report_metrics(summary)
        with contextlib.suppress(Exception):  # the verdict does not hang on the tracker
            tracking.log(
                run_handle,
                metrics={f"oot_check.{name}": value for name, value in metrics.items()},
            )
    # The check measures the rows after the test window, so its headline
    # numbers are the declared metrics over that whole window (mature rows).
    window = next((cell for cell in summary.periods if cell.period == "window"), None)
    declared = {metric.name for metric in job.metric_specs}
    after_test = {
        f"oot_{name}": value
        for name, value in (window.metrics if window is not None else {}).items()
        if name in declared
    }
    return JobResult(
        status="success",
        metrics=MetricResults(metrics=after_test),
        artifact=job.artifact,
        report=summary,
        tracking_run_id=job.tracking_run_id,
    )
