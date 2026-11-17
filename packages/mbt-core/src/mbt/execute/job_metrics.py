"""Metric computation shared by tuning, training and evaluation."""

from typing import Any

import pyarrow as pa

from mbt.events import get_bus
from mbt.events.bus import HookEventSink
from mbt.exceptions import ConfigError
from mbt.execute.job_runtime import JobRuntime
from mbt.execute.seeds import (
    champion_bootstrap_seed,
)
from mbt_adapter_base import (
    BootstrapDelta,
    HookContext,
    MetricResults,
)


def _metrics_for(
    runtime: JobRuntime, model: Any, split: str, *, with_slices: bool, handle: Any = None
) -> MetricResults:
    """Builtin metrics via the adapter, hook metrics via predict + hooks.

    ``handle`` is what the model reads; the challenger's by default, the
    champion's own view for the champion gate."""
    handle = runtime.handle if handle is None else handle
    slices = runtime.spec.evaluation.slices if with_slices else []
    results: MetricResults = runtime.adapter.evaluate(
        model, handle, split, runtime.builtin_specs, slices=slices or None
    )
    if runtime.hook_specs:
        if runtime.hooks is None or not runtime.hooks.has_custom_metrics:
            raise ConfigError(
                "hook metrics declared but hooks.py exposes no custom_metrics",
                resource=runtime.job.node.unique_id,
            )
        predictions: pa.Table = runtime.adapter.predict(model, handle, split)
        hook_ctx = HookContext(
            spec=runtime.spec,
            profile=runtime.base_profile,
            split=split,
            logger=HookEventSink(get_bus()),
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


def _champion_delta_bounds(
    runtime: JobRuntime, challenger: Any, champion: Any
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
    champion_table = runtime.adapter.predict(champion, runtime.champion_handle, "test")
    if champion_table.num_rows != challenger_table.num_rows:
        raise ConfigError(
            f"the champion scored {champion_table.num_rows} test row(s) and the challenger "
            f"{challenger_table.num_rows}; a paired comparison needs the same rows",
            resource=runtime.job.node.unique_id,
            hint="a hooks.py that filters rows must filter the same rows for every model",
        )
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
            seed=champion_bootstrap_seed(spec.seed),  # the ladder: execute/seeds.py
        )
    return bounds
