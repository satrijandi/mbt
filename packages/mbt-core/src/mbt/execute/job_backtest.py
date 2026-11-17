"""Rolling and walk-forward backtests over the train window."""

from typing import Any

import pyarrow as pa

from mbt.events import get_bus
from mbt.events.bus import HookEventSink
from mbt.events.models import LogMessage
from mbt.execute.handles import TransformedDatasetHandle
from mbt.execute.job_carve import _carve_calibration
from mbt.execute.job_runtime import JobRuntime
from mbt.execute.job_tuning import _run_tuning
from mbt.execute.seeds import (
    random_kfold_seed,
)
from mbt_adapter_base import (
    HookContext,
    ModelSpec,
    SplitStrategy,
)


def _backtest_folds(
    runtime: JobRuntime, spec: ModelSpec, base_train: pa.Table, n_folds: int
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
            # SplitSpec makes time_column mandatory for a temporal split and the
            # parser forces the protocol to match it, so the only way here is a
            # transform_features hook that dropped the column. Skipping the
            # backtest silently would report "no folds" as if none were asked
            # for (F6's neighbour).
            get_bus().emit(
                LogMessage(
                    level="warn",
                    unique_id=runtime.job.node.unique_id,
                    message=(
                        f"walk-forward backtest skipped: split time column "
                        f"{time_column!r} is not in the train split after hooks. "
                        "transform_features must preserve the split time column"
                    ),
                )
            )
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
    perm = np.random.default_rng(random_kfold_seed(spec.seed)).permutation(n)
    edges = [round(i * n / n_folds) for i in range(n_folds + 1)]
    return [
        (
            base_train.take(np.concatenate([perm[: edges[i]], perm[edges[i + 1] :]])),
            base_train.take(perm[edges[i] : edges[i + 1]]),
        )
        for i in range(n_folds)
    ]


def _walk_forward_backtest(
    runtime: JobRuntime, spec: ModelSpec, n_folds: int, *, nested: bool = False
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
        return HookContext(
            spec=spec, profile=runtime.base_profile, split=split, logger=HookEventSink(get_bus())
        )

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
            fold_runtime = JobRuntime(
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
