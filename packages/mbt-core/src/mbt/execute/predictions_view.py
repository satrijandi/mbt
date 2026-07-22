"""Read surface over the prediction store: ``mbt predictions ls/show`` (R2-12).

Coordinator-side and read-only: opens each scoring node's prediction store and
reports run + ground-truth-ledger state (matured?, evaluated?, realized
metrics) without touching training or the monitor's write path. ``matured`` is
computed exactly as ``mbt monitor`` does, so ``ls`` previews what the next
monitor run would pick up.
"""

from dataclasses import dataclass, field
from datetime import datetime
from typing import Any

from mbt.adapters.registry import AdapterRegistry
from mbt.compile.windows import parse_window
from mbt.contracts import ScoringSpec
from mbt.execute.monitor import GROUND_TRUTH_MARKER
from mbt.execute.orchestrator import (
    InvocationOptions,
    prepare,
    prepare_readonly,
    require_scoring_capability,
)
from mbt.execute.runners import ExecutionContext


@dataclass
class PredictionRunView:
    """One prediction run's state as seen through the store (no ML deps)."""

    scoring: str  # scoring node unique_id
    run_key: str
    scored_at: str
    model_name: str
    model_version: str
    row_count: int
    matured: bool | None  # None when the node declares no ground_truth block
    evaluated: bool
    realized: dict[str, float] = field(default_factory=dict)
    coverage: float | None = None
    marker: dict[str, Any] | None = None  # the full ground_truth marker, for `show`


def _parse_ts(value: str) -> datetime:
    return datetime.fromisoformat(value.replace("Z", "+00:00"))


def _prepared_context(
    opts: InvocationOptions, registry: AdapterRegistry | None
) -> tuple[ExecutionContext, datetime]:
    # ls/show are read-only: reuse the already-built target/manifest.json rather
    # than recompiling (which would overwrite that build artifact and couple this
    # read to a clean parse + live snapshot pinning, so a transient source outage
    # during an incident would fail `ls` even though the store reads fine). A
    # fresh compile happens only if nothing is built yet, or --manifest is given.
    built = opts.project_dir / "target" / "manifest.json"
    if opts.manifest_path is None and built.is_file():
        prepared = prepare_readonly(opts, built, registry=registry)
    else:
        prepared = prepare(opts, registry=registry)
    ctx = ExecutionContext(
        manifest=prepared.manifest,
        profiles=prepared.profiles,
        registry=prepared.registry,
        project_dir=opts.project_dir.resolve(),
        run_id=prepared.run_id,
        command="predictions",
        cli_vars=opts.cli_vars,
        python_tests=[],
        total_nodes=0,
    )
    require_scoring_capability(ctx)
    return ctx, _parse_ts(prepared.manifest.metadata.anchor)


def _views(ctx: ExecutionContext, anchor: datetime) -> list[PredictionRunView]:
    views: list[PredictionRunView] = []
    for uid, node in ctx.manifest.nodes.items():
        if node.resource_type != "scoring":
            continue
        spec = ScoringSpec.model_validate(node.config)
        store = ctx.data_adapter.open_predictions(spec.output)
        mature_by: datetime | None = None
        if spec.ground_truth is not None:
            mature_by = parse_window(spec.ground_truth.maturity).start.resolve(anchor)
        for run in store.list_runs():
            matured: bool | None = None
            # A 0-row run is never evaluable, and `mbt monitor` skips it outright
            # (it can never become `evaluated`), so report maturity as unknown
            # rather than True - otherwise `ls` shows a 0-row run as matured=yes /
            # evaluated=no forever, contradicting what monitor will actually do.
            if mature_by is not None and run.row_count > 0:
                try:
                    matured = _parse_ts(run.scored_at) <= mature_by
                except ValueError:
                    matured = None  # unparseable sidecar (R2-19); report unknown
            marker = store.read_marker(run.run_key, GROUND_TRUTH_MARKER)
            metrics = (marker or {}).get("metrics", {})
            views.append(
                PredictionRunView(
                    scoring=uid,
                    run_key=run.run_key,
                    scored_at=run.scored_at,
                    model_name=run.model_name,
                    model_version=run.model_version,
                    row_count=run.row_count,
                    matured=matured,
                    evaluated=marker is not None,
                    realized={k: float(v) for k, v in metrics.items()},
                    coverage=(marker or {}).get("coverage"),
                    marker=marker,
                )
            )
    return views


def list_prediction_runs(
    opts: InvocationOptions, *, registry: AdapterRegistry | None = None
) -> list[PredictionRunView]:
    """Every prediction run across the project's scoring nodes, newest last."""
    ctx, anchor = _prepared_context(opts, registry)
    return sorted(_views(ctx, anchor), key=lambda v: (v.scoring, v.scored_at, v.run_key))


def show_prediction_run(
    opts: InvocationOptions, run_key: str, *, registry: AdapterRegistry | None = None
) -> PredictionRunView | None:
    """One run by key, or None if no scoring node's store holds it."""
    ctx, anchor = _prepared_context(opts, registry)
    return next((v for v in _views(ctx, anchor) if v.run_key == run_key), None)
