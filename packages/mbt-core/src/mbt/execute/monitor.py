"""``mbt monitor``: delayed ground-truth evaluation of prediction runs (ADR-21).

Runs coordinator-side, in-process: no model loads, no training frameworks.
For every scoring node with a ``ground_truth`` block it finds matured,
not-yet-evaluated prediction runs (``scored_at + maturity <= anchor``),
joins the matured labels, computes realized metrics with the shared metric
engine, applies the declared threshold gates, and writes a ``ground_truth``
ledger marker into the prediction store so each run is evaluated exactly
once. Gate breaches set ``monitor_failed`` (exit code 2).
"""

import time
from datetime import UTC, datetime
from typing import Any

import mbt
from mbt.adapters.registry import AdapterRegistry
from mbt.artifacts.manifest import Manifest
from mbt.artifacts.run_results import (
    MonitorResult,
    NodeResult,
    RunResults,
    RunResultsMetadata,
)
from mbt.compile.windows import parse_window
from mbt.contracts import (
    ManifestNode,
    MetricSpec,
    ModelSpec,
    PredictionRunInfo,
    ScoringInputSpec,
    ScoringSpec,
)
from mbt.events import get_bus
from mbt.events.models import LogMessage, NodeFinished, NodeStarted, RunFinished, RunStarted
from mbt.exceptions import ConfigError, MbtError
from mbt.execute.orchestrator import (
    InvocationOptions,
    _require_scoring_capability,
    prepare,
)
from mbt.execute.planner import plan_execution
from mbt.execute.runners import ExecutionContext, _BuildContext, materialization_key
from mbt.quality.metrics import resolve_metric
from mbt.quality.monitors import all_monitors_passed, evaluate_ground_truth_gates

#: Ledger marker name in the prediction store (ADR-21).
GROUND_TRUTH_MARKER = "ground_truth"


def _parse_ts(value: str) -> datetime:
    return datetime.fromisoformat(value.replace("Z", "+00:00"))


def run_monitor(opts: InvocationOptions, *, registry: AdapterRegistry | None = None) -> RunResults:
    """Evaluate matured prediction runs against arrived labels; never trains."""
    started_monotonic = time.monotonic()
    started_at = datetime.now(tz=UTC).isoformat()
    prepared = prepare(opts, registry=registry)
    manifest = prepared.manifest
    bus = get_bus()
    bus.run_id = prepared.run_id

    plan = plan_execution(manifest, opts.select, opts.exclude, None, executable=("scoring",))
    bus.emit(
        RunStarted(command="monitor", target=manifest.metadata.target, selected=len(plan.selected))
    )

    ctx = ExecutionContext(
        manifest=manifest,
        profiles=prepared.profiles,
        registry=prepared.registry,
        project_dir=opts.project_dir.resolve(),
        run_id=prepared.run_id,
        command="monitor",
        cli_vars=opts.cli_vars,
        python_tests=[],
        total_nodes=len(plan.execution_set),
    )
    _require_scoring_capability(ctx)
    anchor = _parse_ts(manifest.metadata.anchor)

    results: list[NodeResult] = []
    for uid in plan.order:
        node = manifest.nodes[uid]
        index = ctx.next_index()
        bus.emit(
            NodeStarted(unique_id=uid, resource_type="scoring", index=index, total=ctx.total_nodes)
        )
        node_started = time.monotonic()
        try:
            result = _monitor_node(ctx, node, anchor)
        except MbtError as exc:
            result = NodeResult(unique_id=uid, status="error", message=str(exc))
        result.execution_time_s = time.monotonic() - node_started
        bus.emit(
            NodeFinished(
                unique_id=uid,
                resource_type="scoring",
                status=result.status,
                execution_time_s=result.execution_time_s,
                index=index,
                total=ctx.total_nodes,
                message=result.message,
            )
        )
        results.append(result)

    run_results = RunResults(
        metadata=RunResultsMetadata(
            run_id=prepared.run_id,
            mbt_version=mbt.__version__,
            target=manifest.metadata.target,
            manifest_hash=manifest.manifest_hash(),
            anchor=manifest.metadata.anchor,
            started_at=started_at,
            elapsed_s=round(time.monotonic() - started_monotonic, 3),
            command="monitor",
            selector=" ".join(opts.select) if opts.select else None,
        ),
        results=results,
    )
    run_results.write(opts.project_dir / "target" / "run_results.json")
    statuses: list[str] = [r.status for r in results]
    bus.emit(
        RunFinished(
            command="monitor",
            status={0: "success", 1: "error", 2: "quality_failure"}[run_results.exit_code()],
            succeeded=statuses.count("success"),
            failed=sum(statuses.count(s) for s in ("error", "monitor_failed")),
            skipped=statuses.count("skipped"),
            elapsed_s=run_results.metadata.elapsed_s,
        )
    )
    return run_results


def _monitor_node(ctx: ExecutionContext, node: ManifestNode, anchor: datetime) -> NodeResult:
    uid = node.unique_id
    spec = ScoringSpec.model_validate(node.config)
    if spec.ground_truth is None:
        return NodeResult(unique_id=uid, status="skipped", message="no ground_truth block declared")
    ground_truth = spec.ground_truth

    store = ctx.data_adapter.open_predictions(spec.output)
    matured = _matured_unevaluated(store, ground_truth.maturity, anchor)
    if not matured:
        return NodeResult(
            unique_id=uid, status="success", message="0 matured prediction runs to evaluate"
        )

    metric_specs = _metric_specs(spec, node, ctx.manifest)
    labels = _materialize_labels(ctx, node, spec)

    monitors: list[MonitorResult] = []
    newest_metrics: dict[str, float] = {}
    evaluated = 0
    for run in matured:
        outcome = _evaluate_run(ctx, node, spec, run, store, labels, metric_specs, anchor)
        if outcome is None:
            continue
        metrics, gate_results = outcome
        evaluated += 1
        newest_metrics = metrics  # matured list is scored_at-ascending
        monitors.extend(gate_results)

    passed = all_monitors_passed(monitors)
    failures = [m.message for m in monitors if not m.passed and m.message]
    message = f"evaluated {evaluated} of {len(matured)} matured prediction run(s)"
    if failures:
        message += "; gate breach: " + "; ".join(failures)
    return NodeResult(
        unique_id=uid,
        status="success" if passed else "monitor_failed",
        metrics=newest_metrics,
        monitors=monitors,
        message=message,
    )


def _matured_unevaluated(store: Any, maturity: str, anchor: datetime) -> list[PredictionRunInfo]:
    """Runs whose maturity lag has passed and whose ledger marker is absent."""
    delta = parse_window(maturity).start.delta
    assert delta is not None  # validated at parse time
    matured: list[PredictionRunInfo] = []
    for run in store.list_runs():
        if run.row_count == 0:
            continue
        scored_at = _parse_ts(run.scored_at)
        if scored_at - delta > anchor:  # delta is negative: scored_at + |maturity|
            continue
        if store.read_marker(run.run_key, GROUND_TRUTH_MARKER) is not None:
            continue
        matured.append(run)
    return matured


def _metric_specs(spec: ScoringSpec, node: ManifestNode, manifest: Manifest) -> list[MetricSpec]:
    """Resolve ground-truth metrics against the model's task schema."""
    from mbt.config.tasks import get_task_schema

    assert spec.ground_truth is not None
    model_uid = next(d for d in node.depends_on if d.startswith("model."))
    model_node = manifest.nodes[model_uid]
    model_spec = ModelSpec.model_validate(model_node.config)
    declared = {
        name: MetricSpec.model_validate(payload) for name, payload in manifest.metrics.items()
    }
    task_schema = get_task_schema(model_spec.task)
    resolved: list[MetricSpec] = []
    errors: list[str] = []
    for name in spec.ground_truth.metrics:
        outcome = resolve_metric(name, declared, task_schema, has_hooks=False)
        if isinstance(outcome, str):
            errors.append(outcome)
        elif outcome.kind != "builtin":
            errors.append(f"ground-truth metric {name!r} must be a builtin")
        else:
            resolved.append(outcome)
    if errors:
        raise ConfigError("; ".join(errors), resource=node.unique_id)
    return resolved


def _materialize_labels(ctx: ExecutionContext, node: ManifestNode, spec: ScoringSpec) -> Any:
    """Build the matured-label table through the data adapter (no new contract).

    The label source is pinned on the manifest's sources (observability),
    but deliberately not part of the scoring node's identity (ADR-20); the
    build verifies against the label table's own manifest pin.
    """
    assert spec.ground_truth is not None
    label_uid = spec.ground_truth.label.source
    source = ctx.manifest.sources.get(label_uid)
    if source is None:
        raise ConfigError(
            f"ground-truth label source {label_uid!r} missing from the manifest",
            resource=node.unique_id,
            hint="recompile: the manifest and spec disagree",
        )
    label_node = node.model_copy(update={"snapshot_id": source.snapshot_id, "resolved": {}})
    key = materialization_key(label_node)
    build_ctx = _BuildContext(
        node=label_node,
        source=source.config,
        source_tables={label_uid: source.config},
        resolved_windows={},
        sample_fraction=1.0,  # never sample labels: evaluation wants them all
        deep_snapshot=ctx.manifest.metadata.deep_snapshot,
        output_dir=ctx.project_dir / "target" / "scoring_inputs" / f"{node.name}_labels" / key,
        events=get_bus(),
    )
    handle = ctx.data_adapter.build_scoring_input(ScoringInputSpec(source=label_uid), build_ctx)
    table = handle.read("score")
    needed = [*spec.ground_truth.join_columns, spec.ground_truth.label.column]
    missing = [c for c in needed if c not in table.column_names]
    if missing:
        raise ConfigError(
            f"ground-truth label table lacks column(s): {', '.join(missing)}",
            resource=node.unique_id,
            hint="check ground_truth.join_key and ground_truth.label.column",
        )
    return table.select(needed)


def _evaluate_run(
    ctx: ExecutionContext,
    node: ManifestNode,
    spec: ScoringSpec,
    run: PredictionRunInfo,
    store: Any,
    labels: Any,
    metric_specs: list[MetricSpec],
    anchor: datetime,
) -> tuple[dict[str, float], list[MonitorResult]] | None:
    """Evaluate one matured prediction run; None when it cannot be evaluated.

    Runs that cannot be evaluated (no label coverage, single-class labels)
    are NOT marked: they retry on the next monitor run once labels arrive.
    """
    import duckdb

    assert spec.ground_truth is not None
    uid = node.unique_id
    join_columns = spec.ground_truth.join_columns
    label_column = spec.ground_truth.label.column

    predictions = store.read(run.run_key, columns=[*join_columns, "prediction"])
    con = duckdb.connect()
    try:
        con.register("mbt_predictions", predictions)
        con.register("mbt_labels", labels)
        using = ", ".join(f'"{c}"' for c in join_columns)
        # con.sql(...).to_arrow_table() (the Relation API) works at the duckdb>=1.0
        # floor and is not deprecated; Connection.to_arrow_table only exists from a
        # later duckdb, and Connection.fetch_arrow_table warns on current duckdb.
        joined = con.sql(
            f'SELECT p."prediction", l."{label_column}" FROM mbt_predictions p '
            f"JOIN mbt_labels l USING ({using})"
        ).to_arrow_table()
    finally:
        con.close()

    coverage = joined.num_rows / max(run.row_count, 1)
    if joined.num_rows == 0:
        get_bus().emit(
            LogMessage(
                level="warn",
                unique_id=uid,
                message=(
                    f"run {run.run_key}: no matured labels joined "
                    f"(join_key: {', '.join(join_columns)}); will retry next monitor run"
                ),
            )
        )
        return None
    if coverage < 1.0:
        get_bus().emit(
            LogMessage(
                level="warn",
                unique_id=uid,
                message=(
                    f"run {run.run_key}: labels cover {coverage:.1%} of {run.row_count} scored rows"
                ),
            )
        )

    try:
        from mbt_adapter_base.metrics import compute_binary_results
    except ImportError as exc:  # pragma: no cover - env-specific
        raise ConfigError(
            f"ground-truth evaluation needs the metric engine: {exc}",
            hint="install mbt-adapter-base[metrics] (numpy + scikit-learn)",
        ) from exc
    import numpy as np

    y_score = joined.column("prediction").to_numpy(zero_copy_only=False).astype("float64")
    y_true = joined.column(label_column).to_numpy(zero_copy_only=False).astype("float64")
    if len(np.unique(y_true)) < 2:
        get_bus().emit(
            LogMessage(
                level="warn",
                unique_id=uid,
                message=(
                    f"run {run.run_key}: matured labels are single-class; realized "
                    "metrics are undefined - will retry next monitor run"
                ),
            )
        )
        return None

    metrics = dict(compute_binary_results(metric_specs, y_true, y_score).metrics)
    gate_results = evaluate_ground_truth_gates(
        spec.ground_truth.gates, metrics, metric_specs, run_key=run.run_key
    )

    store.write_marker(
        run.run_key,
        GROUND_TRUTH_MARKER,
        {
            "evaluated_at": anchor.strftime("%Y-%m-%dT%H:%M:%SZ"),
            "monitor_run_id": ctx.run_id,
            "model_version": run.model_version,
            "metrics": metrics,
            "gates_passed": all(g.passed for g in gate_results),
            "matched_rows": joined.num_rows,
            "coverage": round(coverage, 4),
        },
    )

    _log_tracking(ctx, node, run, metrics, coverage)
    return metrics, gate_results


def _log_tracking(
    ctx: ExecutionContext,
    node: ManifestNode,
    run: PredictionRunInfo,
    metrics: dict[str, float],
    coverage: float,
) -> None:
    """One tracking run per evaluated prediction run: the realized-metric
    time series accumulates in the tracking backend (MLflow et al.)."""
    try:
        tracking = ctx.tracking()
        handle = tracking.start_run(
            node,
            {
                "mbt.monitor": GROUND_TRUTH_MARKER,
                "mbt.run_key": run.run_key,
                "mbt.model_version": run.model_version,
                "mbt.scored_at": run.scored_at,
            },
        )
        tracking.log(handle, metrics={**metrics, "ground_truth.coverage": coverage})
        tracking.end_run(handle, "FINISHED")
    except Exception as exc:
        get_bus().emit(
            LogMessage(level="warn", message=f"could not log monitor metrics to tracking: {exc}")
        )


__all__ = ["GROUND_TRUTH_MARKER", "run_monitor"]
