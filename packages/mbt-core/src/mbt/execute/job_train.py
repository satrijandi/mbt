"""The training step: tune, fit, calibrate, evaluate, export, log (TSD §10.3)."""

import contextlib
import json
from collections.abc import Iterator
from pathlib import Path
from typing import Any

from mbt.events import get_bus
from mbt.events.models import AutoResolved, LogMessage
from mbt.exceptions import AdapterError
from mbt.execute import training_report as report_step
from mbt.execute.job_backtest import _walk_forward_backtest
from mbt.execute.job_carve import _carve_calibration
from mbt.execute.job_metrics import _champion_delta_bounds, _metrics_for
from mbt.execute.job_runtime import JobRuntime
from mbt.execute.job_setup import _materialize_for_path_adapter
from mbt.execute.job_tuning import _run_tuning
from mbt.utils import canonical_json
from mbt_adapter_base import (
    AUTO,
    BootstrapDelta,
    JobResult,
    MetricResults,
    ModelSpec,
)
from mbt_adapter_base.capabilities import Capability, supports


def _check_operating_point_support(
    runtime: JobRuntime, metrics: dict[str, float], test: Any
) -> dict[str, int]:
    """Say how many rows each deployable cutoff rests on, loudly when few (D-1).

    A ``threshold_at_precision_<p>`` value is a FITTED parameter selected by
    scanning the test split's own PR curve, and the precision reported for it
    is then measured on the same rows. The point bias is negligible while the
    cutoff is supported by roughly a hundred rows and serious when it is not -
    at 7 rows a reported 0.78 is a realized 0.65 on fresh data.

    That is the rare-positive, high-precision corner, which is exactly the
    retention-campaign shape the feature was built for. ``churn_demo`` runs
    ``threshold_at_precision_0.35`` at a ~20% base rate, comfortably outside
    the danger zone, which is why no test ever caught this.

    Returns the support per metric so the caller can record it.
    """
    from mbt_adapter_base.metrics import MIN_OPERATING_POINT_SUPPORT, operating_point_support

    support: dict[str, int] = {}
    for name, cutoff in metrics.items():
        if not name.startswith(("threshold_at_precision_", "threshold_at_recall_")):
            continue
        rows = operating_point_support(test.scores, cutoff)
        support[name] = rows
        if rows >= MIN_OPERATING_POINT_SUPPORT:
            _detail(runtime, f"operating point {name}={cutoff:.6g} rests on {rows} test row(s)")
            continue
        get_bus().emit(
            LogMessage(
                level="warn",
                unique_id=runtime.job.node.unique_id,
                message=(
                    f"operating point {name}={cutoff:.6g} rests on only {rows} test row(s) "
                    f"(under {MIN_OPERATING_POINT_SUPPORT}): the cutoff is fitted on the same "
                    "rows that report its precision, so on fresh data the realized precision "
                    "will typically be materially lower than the number shown - do not deploy "
                    "this as a decision rule without a larger test window or a lower target"
                ),
            )
        )
    return support


def _export_baseline(runtime: JobRuntime, test_scores: Any) -> Any:
    """Build + export the monitoring baseline next to the model artifact (ADR-21).

    Post-hook train-split feature distributions plus the test-split score
    distribution: everything scoring-time shift monitors compare against.
    Built unconditionally on every training job - it is cheap (quantiles),
    and champions registered without one cannot be monitored later.
    ``test_scores`` are the report's test predictions, scored once.
    """
    import tempfile

    from mbt_adapter_base.monitoring import build_baseline, write_baseline

    train = runtime.transformed.read("train")
    feature_columns = runtime.transformed.feature_columns()
    baseline = build_baseline(train, feature_columns, test_scores, model_name=runtime.spec.name)
    with tempfile.TemporaryDirectory(prefix="mbt-baseline-") as staging:
        path = Path(staging) / "baseline.json"
        write_baseline(baseline, path)
        return runtime.store.put_file(path, "baseline.json", format="json")


def _export_inference_config(
    runtime: JobRuntime,
    artifact: Any,
    baseline: Any,
    metrics: dict[str, float],
    *,
    hyperparameters: dict[str, Any] | None = None,
    report: Any = None,
) -> Any:
    """Export the champion's own inference config next to the artifact (ADR-28).

    What a later scoring run reads the model's spec from, so scoring follows
    the champion it loaded rather than the working tree. Built unconditionally
    like the baseline above: a champion registered without one falls back to
    the local manifest with a warning, and that fallback should be a
    migration story, not the normal path.
    """
    import tempfile

    from mbt.execute.inference_config import build_inference_config
    from mbt.execute.training_report import dataset_record

    document = build_inference_config(
        node=runtime.job.node,
        project=runtime.job.project,
        run_id=runtime.job.run_id,
        feature_columns=runtime.transformed.feature_columns(),
        metrics=metrics,
        artifact=artifact,
        baseline=baseline,
        meta=dict(runtime.job.tracking_meta),
        hyperparameters=hyperparameters,
        dataset=dataset_record(runtime),
        report=report,
    )
    with tempfile.TemporaryDirectory(prefix="mbt-inference-config-") as staging:
        path = Path(staging) / "inference_config.json"
        path.write_text(canonical_json(document) + "\n")
        return runtime.store.put_file(path, "inference_config.json", format="json")


def _log_run_documents(
    runtime: JobRuntime, tracking: Any, run_handle: Any, inference_config: Any
) -> None:
    """Attach the readable record of what this run trained (ADR-28).

    ``log(artifacts=...)`` uploads only a ``file://`` ArtifactRef, because a
    model binary belongs in mbt's artifact store with a pointer in the tracker
    (ADR-21) - which leaves a run on an ``s3://`` store carrying no documents
    at all. These two are small, mbt wrote them, and they are what makes the
    run describe its model: the inference config, and the hooks source whose
    hash the config records.

    Optional capability, probed like ``prepare`` and ``log_trial``: a tracker
    without it simply keeps the tags it always had.
    """
    if not supports(tracking, Capability.LOG_DOCUMENT):
        return
    import tempfile

    with contextlib.suppress(Exception):  # a tracker hiccup must not fail training
        local = runtime.store.fetch(inference_config)
        tracking.log_document(run_handle, local)
        hooks_path = runtime.job.node.hooks_path
        if hooks_path is not None:
            source = Path(runtime.job.project_dir) / hooks_path
            if source.is_file():
                # Copied under its declared basename so the run shows
                # "wide_hooks.py", not whatever the checkout nests it under.
                with tempfile.TemporaryDirectory(prefix="mbt-hooks-") as staging:
                    staged = Path(staging) / Path(hooks_path).name
                    staged.write_bytes(source.read_bytes())
                    tracking.log_document(run_handle, staged)


def _partial_dependence(
    runtime: JobRuntime, model: Any, importance: dict[str, float], *, top_n: int = 3
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


def _detail(runtime: JobRuntime, message: str) -> None:
    """A debug-level account of what the job did: hidden on the console
    unless ``--verbose``, always in the node's run log (ADR-30)."""
    get_bus().emit(LogMessage(level="debug", unique_id=runtime.job.node.unique_id, message=message))


@contextlib.contextmanager
def _phase(runtime: JobRuntime, name: str) -> Iterator[None]:
    import time

    started = time.monotonic()
    yield
    _detail(runtime, f"{name} took {time.monotonic() - started:.2f}s")


def _describe_inputs(runtime: JobRuntime) -> None:
    """Split sizes, label balance, time span and feature set, for the log."""
    profile = runtime.materialization.profile()
    rows = ", ".join(f"{split}={n}" for split, n in sorted(profile.n_rows.items()))
    _detail(runtime, f"dataset {runtime.job.dataset.uri}: {rows}")
    if profile.label_balance:
        balance = ", ".join(f"{k}={v:.3%}" for k, v in sorted(profile.label_balance.items()))
        _detail(runtime, f"label balance (train): {balance}")
    if profile.time_range:
        _detail(runtime, f"time span: {profile.time_range[0]} .. {profile.time_range[1]}")
    windows = runtime.job.dataset_windows.get("windows") or {}
    for split, bounds in sorted(windows.items(), key=lambda item: str(item[1][0])):
        _detail(runtime, f"window {split}: [{bounds[0]}, {bounds[1]})")  # oldest first
    test_window = runtime.job.node.resolved.get("test_window")
    if test_window:
        _detail(runtime, f"test narrowed by test_window to [{test_window[0]}, {test_window[1]})")
    features = runtime.transformed.feature_columns()
    shown = ", ".join(features[:30]) + (
        f", ... ({len(features) - 30} more)" if len(features) > 30 else ""
    )
    _detail(runtime, f"{len(features)} feature(s): {shown}")


def _metric_line(metrics: dict[str, float]) -> str:
    return ", ".join(f"{name}={value:.4f}" for name, value in sorted(metrics.items()))


def _tracking_params(runtime: JobRuntime, spec: ModelSpec) -> dict[str, str]:
    """Bare hyperparameters and ``seed`` - what runs have always carried, kept
    so history stays comparable - plus the flat model and dataset configs
    (ADR-30)."""
    from mbt.execute.training_report import dataset_record
    from mbt.reporting.flatten import dataset_params, model_params

    params = {k: str(v) for k, v in spec.hyperparameters.items()}
    params["seed"] = str(spec.seed)
    params.update(model_params(spec.model_dump(mode="json"), runtime.transformed.feature_columns()))
    node = runtime.job.dataset_node
    record = dataset_record(runtime)
    if node is not None and record is not None:
        windows = dict(record["windows"])
        if record["test_window"]:
            windows["test"] = record["test_window"]
        params.update(
            dataset_params(
                node.config,
                windows=windows,
                anchor=record["anchor"],
                sample_fraction=record["sample_fraction"],
                row_counts=record["row_counts"],
            )
        )
    return params


def _run_train(runtime: JobRuntime, tracking: Any, run_handle: Any) -> JobResult:
    job = runtime.job
    bus = get_bus()
    _describe_inputs(runtime)

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
        _detail(
            runtime,
            f"calibration ({spec.calibration}) carve: fit on {carved.base.read('train').num_rows} "
            f"rows, calibrate on {carved.base.read('calibration').num_rows}",
        )
        fit_handle = carved
        if getattr(runtime.adapter, "data_access", "arrow") == "path":
            fit_handle = _materialize_for_path_adapter(carved, spec)
    _detail(
        runtime, f"hyperparameters: {json.dumps(spec.hyperparameters, sort_keys=True, default=str)}"
    )
    with _phase(runtime, "fit"):
        model = runtime.adapter.train(spec, fit_handle, runtime.ctx)

    # 4. evaluate challenger and (if provided) champion on the SAME test split
    with _phase(runtime, "test evaluation"):
        challenger = _metrics_for(runtime, model, "test", with_slices=True)
    _detail(runtime, "test metrics: " + _metric_line(challenger.metrics))
    champion_metrics: MetricResults | None = None
    delta_bounds: dict[str, BootstrapDelta] = {}
    if job.champion is not None:
        champion_model = runtime.adapter.load(job.champion, runtime.store)
        champion_metrics = _metrics_for(
            runtime, champion_model, "test", with_slices=True, handle=runtime.champion_handle
        )
        _detail(
            runtime, "champion on the same test split: " + _metric_line(champion_metrics.metrics)
        )
        delta_bounds = _champion_delta_bounds(runtime, model, champion_model)

    # 5. export the artifact and the monitoring baseline (ADR-21); score every
    #    report split once (ADR-30) and reuse the test scores for the baseline
    artifact = runtime.adapter.export(model, "native", runtime.store)
    _detail(runtime, f"artifact {artifact.uri} ({artifact.size_bytes} bytes, {artifact.format})")
    with _phase(runtime, "scoring the report splits and building the training report"):
        report_run = report_step.produce_report(
            runtime,
            spec,
            model,
            kind="training",
            stage=_materialize_for_path_adapter,
            tracking=tracking,
            run_handle=run_handle,
        )
    _check_operating_point_support(runtime, challenger.metrics, report_run.scored["test"])
    baseline = _export_baseline(runtime, report_run.scored["test"].scores)
    partial_dependence = _partial_dependence(runtime, model, report_run.importance)
    backtest_folds = spec.evaluation.protocol.backtest_folds
    nested = spec.evaluation.protocol.nested_cv
    backtest_metrics, backtest_std = (
        _walk_forward_backtest(
            runtime, pre_tuning_spec if nested else spec, backtest_folds, nested=nested
        )
        if backtest_folds is not None
        else ({}, {})
    )

    # 6. the inference config a later scoring run reads its spec from (ADR-28)
    summary = report_run.summary
    _detail(
        runtime,
        f"training report: {len(summary.documents)} file(s) at {summary.report_uri}; "
        f"{summary.out_of_time_rows} after-test row(s), "
        f"{sum(1 for c in summary.periods if c.period != 'window')} period cell(s)",
    )
    inference_config = _export_inference_config(
        runtime,
        artifact,
        baseline,
        dict(challenger.metrics),
        hyperparameters=dict(spec.hyperparameters),
        report=summary,
    )

    # 7. tracking: params, metrics, artifacts, documents, tuning history
    tracking_run_id: str | None = None
    if tracking is not None and run_handle is not None:
        tracking_run_id = run_handle.run_id
        tracking.log(
            run_handle,
            params=_tracking_params(runtime, spec),
            metrics={**challenger.metrics, **report_step.report_metrics(summary)},
            artifacts=[artifact],
        )
        _log_run_documents(runtime, tracking, run_handle, inference_config)
        report_step.log_config_documents(runtime, tracking, run_handle, spec)
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
        feature_importance=report_run.importance,
        partial_dependence=partial_dependence,
        backtest_metrics=backtest_metrics,
        backtest_std=backtest_std,
        resolved_auto=resolved_auto,
        tuning=tuning_result,
        artifact=artifact,
        baseline=baseline,
        inference_config=inference_config,
        report=summary,
        tracking_run_id=tracking_run_id,
    )
