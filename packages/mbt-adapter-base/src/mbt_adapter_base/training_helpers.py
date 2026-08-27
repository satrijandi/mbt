"""Shared implementation helpers for training adapters.

Each helper factors a pattern every binary-classification adapter was
re-implementing by hand - identically, or worse, almost identically (the
fake adapter rounded ``scale_pos_weight`` to 4 decimals while the real ones
used 6): the ``evaluate()`` body, ``'{{ auto }}'`` scale_pos_weight
resolution, and the staged-parquet fallback for ``data_access="path"``
frameworks. Heavy imports stay inside functions (ADR-14).
"""

import atexit
import shutil
import tempfile
from pathlib import Path
from typing import TYPE_CHECKING, Any

from mbt_adapter_base.interchange import DatasetProfile, MetricResults
from mbt_adapter_base.specs import MetricSpec

if TYPE_CHECKING:
    import numpy as np
    import pyarrow as pa


def top_k_explanations(shap_values: "np.ndarray", features: list[str], top_k: int) -> list[str]:
    """Per-row local attribution: the ``top_k`` features by |SHAP| for each row,
    as a JSON string ``[[feature, contribution], ...]`` ordered by descending
    |contribution| (explainability). Shared by the SHAP-capable adapters so the
    scoring ``explanation`` column has one format regardless of framework."""
    import json

    import numpy as np

    result: list[str] = []
    for row in shap_values:
        order = np.argsort(-np.abs(row))[:top_k]
        result.append(json.dumps([[features[i], round(float(row[i]), 6)] for i in order]))
    return result


def evaluate_split(
    table: "pa.Table",
    target: str,
    y_score: Any,
    metrics: list[MetricSpec],
    slices: list[str] | None = None,
) -> MetricResults:
    """The shared ``evaluate()`` body: y_true from the split table, the
    adapter's scores, declared slice columns, through the one metric engine -
    so metric semantics stay identical across adapters by construction."""
    import numpy as np

    from mbt_adapter_base.metrics import compute_results

    y_true = table.column(target).to_numpy(zero_copy_only=False).astype(np.float64)
    slice_columns = {
        name: table.column(name).to_numpy(zero_copy_only=False)
        for name in (slices or [])
        if name in table.column_names
    }
    return compute_results(metrics, y_true, np.asarray(y_score, dtype=np.float64), slice_columns)


def calibration_split(data: Any) -> str:
    """The split a post-hoc calibrator must fit on.

    Core carves a dedicated ``calibration`` slice from train whenever the spec
    sets ``calibration`` (F17: the calibrator must not fit on the validation
    split that early stopping and tuning select on). Direct adapter calls (the
    compliance suite, notebooks) may instead pass their own held-out
    ``validation`` split, so that remains the documented fallback. Without
    either there is no honest calibration set, so this fails loudly.
    """
    splits = data.splits()
    if "calibration" in splits:
        return "calibration"
    if "validation" in splits:
        return "validation"
    raise ValueError(
        "calibration needs a 'calibration' (or 'validation') split to fit on; "
        "core carves one from train when the model spec sets 'calibration'"
    )


def positive_rate(profile: DatasetProfile) -> float | None:
    """The positive-class share from a profile's label balance, if present."""
    balance = profile.label_balance or {}
    for key in ("1", "1.0", "true", "True"):
        if key in balance:
            return balance[key]
    return None


def resolve_scale_pos_weight(profile: DatasetProfile) -> float:
    """``(1 - p) / p`` at 6 decimal places; raises without a positive balance."""
    positive = positive_rate(profile)
    if positive is None or positive <= 0:
        raise ValueError(
            "cannot auto-resolve scale_pos_weight: the dataset profile has "
            "no positive-class balance"
        )
    return round((1.0 - positive) / positive, 6)


def staged_split_path(data: Any, split: str, *, prefix: str) -> Path:
    """The split's on-disk parquet for ``data_access="path"`` adapters.

    Uses the handle's own file when it has one (``split_path``); otherwise
    stages a copy in a temp dir that is removed at process exit (training
    jobs are short-lived subprocesses).
    """
    split_path = getattr(data, "split_path", None)
    if callable(split_path):
        return Path(split_path(split))
    import pyarrow.parquet as pq

    directory = Path(tempfile.mkdtemp(prefix=prefix))
    atexit.register(shutil.rmtree, directory, ignore_errors=True)
    out = directory / f"{split}.parquet"
    pq.write_table(data.read(split), out)
    return out
