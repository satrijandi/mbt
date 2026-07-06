"""Shared binary-classification metric computation for tabular adapters.

Training adapters compute metrics; core compares them (TSD §10.3). This
module keeps that computation identical across adapters (XGBoost, LightGBM)
so champion/challenger deltas are apples to apples.

numpy/scikit-learn load lazily: importing this module is cheap (ADR-14).
Requires the ``mbt-adapter-base[metrics]`` extra at call time.
"""

from typing import TYPE_CHECKING, Any

from mbt_adapter_base.interchange import MetricResults
from mbt_adapter_base.specs import MetricSpec

if TYPE_CHECKING:
    import numpy as np

#: Builtin metric base names for binary classification, shared by task schema
#: validation (core) and computation (adapters).
BINARY_METRIC_BASES = frozenset(
    {
        "roc_auc",
        "pr_auc",
        "logloss",
        "brier",
        "accuracy",
        "ece",
        "recall_at_precision",
        "precision_at_recall",
    }
)


def parse_metric_sugar(name: str) -> tuple[str, dict[str, Any]] | None:
    """Parse sugar like ``recall_at_precision_0.9`` into (base, params).

    Returns None when the name is not a parameterized builtin (TSD §5.7).
    """
    for base, param in (
        ("recall_at_precision", "precision"),
        ("precision_at_recall", "recall"),
    ):
        prefix = base + "_"
        if name.startswith(prefix):
            try:
                value = float(name.removeprefix(prefix))
            except ValueError:
                return None
            if not 0.0 < value <= 1.0:
                return None
            return base, {param: value}
    return None


def is_builtin_binary_metric(name: str) -> bool:
    """True when a metric name (with or without sugar) is a binary builtin."""
    if name in BINARY_METRIC_BASES:
        return True
    return parse_metric_sugar(name) is not None


def _expected_calibration_error(
    y_true: "np.ndarray", y_score: "np.ndarray", n_bins: int = 10
) -> float:
    import numpy as np

    bins = np.linspace(0.0, 1.0, n_bins + 1)
    indices = np.clip(np.digitize(y_score, bins[1:-1]), 0, n_bins - 1)
    ece = 0.0
    total = len(y_true)
    for b in range(n_bins):
        mask = indices == b
        count = int(mask.sum())
        if count == 0:
            continue
        confidence = float(y_score[mask].mean())
        accuracy = float(y_true[mask].mean())
        ece += (count / total) * abs(confidence - accuracy)
    return float(ece)


def _recall_at_precision(
    y_true: "np.ndarray", y_score: "np.ndarray", min_precision: float
) -> float:
    from sklearn.metrics import precision_recall_curve

    precision, recall, _ = precision_recall_curve(y_true, y_score)
    achievable = recall[precision >= min_precision]
    return float(achievable.max()) if achievable.size else 0.0


def _precision_at_recall(y_true: "np.ndarray", y_score: "np.ndarray", min_recall: float) -> float:
    from sklearn.metrics import precision_recall_curve

    precision, recall, _ = precision_recall_curve(y_true, y_score)
    achievable = precision[recall >= min_recall]
    return float(achievable.max()) if achievable.size else 0.0


def compute_binary_metric(spec: MetricSpec, y_true: "np.ndarray", y_score: "np.ndarray") -> float:
    """Compute one builtin binary-classification metric."""
    import numpy as np
    from sklearn.metrics import (
        accuracy_score,
        average_precision_score,
        brier_score_loss,
        log_loss,
        roc_auc_score,
    )

    base = spec.name
    params = dict(spec.params)
    if base not in BINARY_METRIC_BASES:
        sugar = parse_metric_sugar(spec.name)
        if sugar is None:
            raise ValueError(f"unknown builtin binary metric: {spec.name!r}")
        base, params = sugar[0], {**sugar[1], **params}

    if base == "roc_auc":
        return float(roc_auc_score(y_true, y_score))
    if base == "pr_auc":
        return float(average_precision_score(y_true, y_score))
    if base == "logloss":
        return float(log_loss(y_true, np.clip(y_score, 1e-15, 1 - 1e-15)))
    if base == "brier":
        return float(brier_score_loss(y_true, y_score))
    if base == "accuracy":
        threshold = float(params.get("threshold", 0.5))
        return float(accuracy_score(y_true, y_score >= threshold))
    if base == "ece":
        return _expected_calibration_error(y_true, y_score, int(params.get("n_bins", 10)))
    if base == "recall_at_precision":
        return _recall_at_precision(y_true, y_score, float(params["precision"]))
    if base == "precision_at_recall":
        return _precision_at_recall(y_true, y_score, float(params["recall"]))
    raise ValueError(f"unhandled builtin binary metric: {base!r}")  # pragma: no cover


def compute_binary_results(
    metric_specs: list[MetricSpec],
    y_true: "np.ndarray",
    y_score: "np.ndarray",
    slice_columns: dict[str, "np.ndarray"] | None = None,
) -> MetricResults:
    """Compute all requested metrics, plus per-slice values (FR-TEST-04).

    Hook metrics (``kind == "hook"``) are computed by the caller and merged
    afterwards; this function skips them.
    """
    import numpy as np

    builtin = [s for s in metric_specs if s.kind == "builtin"]
    metrics = {s.name: compute_binary_metric(s, y_true, y_score) for s in builtin}

    slices: dict[str, dict[str, float]] = {}
    for column, values in (slice_columns or {}).items():
        for value in sorted({str(v) for v in values.tolist()}):
            mask = np.asarray([str(v) == value for v in values.tolist()])
            if int(mask.sum()) == 0 or len(set(y_true[mask].tolist())) < 2:
                # Degenerate slice: single-class metrics are undefined; skip.
                continue
            key = f"{column}={value}"
            slices[key] = {
                s.name: compute_binary_metric(s, y_true[mask], y_score[mask]) for s in builtin
            }
    return MetricResults(metrics=metrics, slices=slices)
