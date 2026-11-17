"""Carving validation and calibration slices out of the train split (TSD §13.5, F17)."""

from typing import Any

import pyarrow as pa

from mbt.events import get_bus
from mbt.events.bus import HookEventSink
from mbt.exceptions import ConfigError
from mbt.execute.handles import TransformedDatasetHandle
from mbt.execute.job_runtime import JobRuntime
from mbt.execute.seeds import (
    calibration_carve_seed,
    validation_carve_seed,
)
from mbt_adapter_base import (
    HookContext,
    ModelSpec,
)
from mbt_adapter_base.datasets import InMemoryDatasetHandle

#: Fraction of the train window carved as implicit validation (TSD §13.5).
_IMPLICIT_VALIDATION_FRACTION = 0.2


#: Fraction of the train window carved as the dedicated calibration slice (F17).
#: The calibrator must fit on rows the model never trained on AND that no
#: selection step (early stopping, tuning) optimized against, so it gets its
#: own slice rather than reusing the validation split.
_CALIBRATION_FRACTION = 0.2


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
    seeded random ``fraction``. Shared by the implicit-validation carve and the
    calibration carve so the two slice train by identical rules and differ only
    in seed (their rungs in ``execute/seeds.py``) and destination split."""
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
    runtime: JobRuntime, *, time_range: tuple[Any, Any] | None = None
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
    raw_train = base.base.read("train")
    time_column = getattr(base.base, "time_column", None)
    windows = job.dataset_windows.get("windows", job.dataset_windows)

    fit_idx, val_idx = _tail_carve_indices(
        raw_train,
        time_column=time_column,
        windows=windows,
        time_range=time_range,
        seed=validation_carve_seed(spec.seed),
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
        return HookContext(
            spec=spec, profile=runtime.base_profile, split=split, logger=HookEventSink(get_bus())
        )

    return TransformedDatasetHandle(carved, spec, runtime.hooks, hook_ctx, time_column)


def _carve_calibration(
    runtime: JobRuntime,
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
    or its own seeded random draw, mirroring the validation carve rules); every
    other split (test, a declared validation) passes through untouched.
    ``time_range`` makes the carve fold-aware inside the backtest (F5).
    """
    job = runtime.job
    raw_train = base.base.read("train")
    time_column = getattr(base.base, "time_column", None)
    windows = job.dataset_windows.get("windows", job.dataset_windows)

    fit_idx, cal_idx = _tail_carve_indices(
        raw_train,
        time_column=time_column,
        windows=windows,
        time_range=time_range,
        seed=calibration_carve_seed(spec.seed),
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
    for split in base.base.splits():
        if split not in ("train", "calibration"):
            tables[split] = base.base.read(split)
    carved = InMemoryDatasetHandle(
        tables,
        snapshot_id=base.snapshot_id,
        label_column=spec.target,
        time_column=time_column,
    )

    def hook_ctx(split: str) -> HookContext:
        return HookContext(
            spec=spec, profile=runtime.base_profile, split=split, logger=HookEventSink(get_bus())
        )

    return TransformedDatasetHandle(carved, spec, runtime.hooks, hook_ctx, time_column)
