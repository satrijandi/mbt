"""Shared metric computation for tabular adapters (binary + regression).

Training adapters compute metrics; core compares them (TSD §10.3). This
module keeps that computation identical across adapters (XGBoost, LightGBM)
so champion/challenger deltas are apples to apples. ``compute_metric``
dispatches on the metric name (the binary and regression name sets are
disjoint), so no task context is threaded through the adapter layer.

numpy/scikit-learn load lazily: importing this module is cheap (ADR-14).
Requires the ``mbt-adapter-base[metrics]`` extra at call time.
"""

from typing import TYPE_CHECKING, Any

from mbt_adapter_base.interchange import BootstrapDelta, MetricResults
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
        "threshold_at_precision",
        "threshold_at_recall",
        "lift",
        "gain",
    }
)

#: Builtin metric base names for regression. Disjoint from the binary set, so
#: the engine dispatches on the metric name alone - no task needs to be threaded
#: through the adapter ``evaluate()`` layer.
REGRESSION_METRIC_BASES = frozenset({"rmse", "mae", "r2", "mape"})


def is_builtin_regression_metric(name: str) -> bool:
    """True when a metric name is a builtin regression metric (no sugar forms)."""
    return name in REGRESSION_METRIC_BASES


def parse_metric_sugar(name: str) -> tuple[str, dict[str, Any]] | None:
    """Parse sugar like ``recall_at_precision_0.9`` or ``lift_at_0.1`` into
    (base, params).

    Returns None when the name is not a parameterized builtin (TSD §5.7).
    """
    for prefix, base, param in (
        ("recall_at_precision_", "recall_at_precision", "precision"),
        ("precision_at_recall_", "precision_at_recall", "recall"),
        ("threshold_at_precision_", "threshold_at_precision", "precision"),
        ("threshold_at_recall_", "threshold_at_recall", "recall"),
        ("lift_at_", "lift", "fraction"),
        ("gain_at_", "gain", "fraction"),
    ):
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

    # Equal-frequency (adaptive) bins: each holds ~total/n_bins samples, taken
    # as contiguous slices of the score-sorted samples. Fixed-width bins are
    # noisy on skewed score distributions (typical for churn), piling most
    # samples into one bin and leaving others empty; equal-mass bins avoid that.
    total = len(y_true)
    order = np.argsort(y_score, kind="stable")
    ece = 0.0
    for chunk in np.array_split(order, n_bins):
        if chunk.size == 0:  # more bins than samples: the tail bins are empty
            continue
        confidence = float(y_score[chunk].mean())
        accuracy = float(y_true[chunk].mean())
        ece += (chunk.size / total) * abs(confidence - accuracy)
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


def _threshold_at_precision(
    y_true: "np.ndarray", y_score: "np.ndarray", min_precision: float
) -> float:
    """The deployable operating point for a precision target: the smallest
    score threshold whose precision meets it (maximal coverage at the
    required precision). Returns 1.0 when unattainable or degenerate -
    "predict nothing" is the only rule that honors the target."""
    from sklearn.metrics import precision_recall_curve

    if float(y_true.sum()) == 0.0:
        return 1.0
    precision, _, thresholds = precision_recall_curve(y_true, y_score)
    # precision[i] pairs with thresholds[i]; the final curve point has none
    achievable = thresholds[precision[:-1] >= min_precision]
    return float(achievable.min()) if achievable.size else 1.0


def _threshold_at_recall(y_true: "np.ndarray", y_score: "np.ndarray", min_recall: float) -> float:
    """The operating point for a coverage target: the largest score threshold
    whose recall meets it (best precision at the required recall). Returns
    0.0 when degenerate - "predict everything" is the only safe rule."""
    from sklearn.metrics import precision_recall_curve

    if float(y_true.sum()) == 0.0:
        return 0.0
    _, recall, thresholds = precision_recall_curve(y_true, y_score)
    achievable = thresholds[recall[:-1] >= min_recall]
    return float(achievable.max()) if achievable.size else 0.0


def _top_fraction_indices(y_score: "np.ndarray", fraction: float) -> "np.ndarray":
    """Row indices of the top-scoring ``fraction``; stable sort so ties break
    by row order, deterministically."""
    import numpy as np

    k = max(1, round(len(y_score) * fraction))
    order: np.ndarray = np.argsort(-y_score, kind="stable")
    return order[:k]


def _lift(y_true: "np.ndarray", y_score: "np.ndarray", fraction: float) -> float:
    """Positive rate in the top ``fraction`` over the overall positive rate."""
    base_rate = float(y_true.mean()) if len(y_true) else 0.0
    if base_rate == 0.0:
        return 0.0
    top = _top_fraction_indices(y_score, fraction)
    return float(y_true[top].mean() / base_rate)


def _gain(y_true: "np.ndarray", y_score: "np.ndarray", fraction: float) -> float:
    """Fraction of all positives captured in the top ``fraction`` (cumulative
    gain, the y-axis of a gain chart)."""
    total = float(y_true.sum()) if len(y_true) else 0.0
    if total == 0.0:
        return 0.0
    top = _top_fraction_indices(y_score, fraction)
    return float(y_true[top].sum() / total)


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
    if base == "threshold_at_precision":
        return _threshold_at_precision(y_true, y_score, float(params["precision"]))
    if base == "threshold_at_recall":
        return _threshold_at_recall(y_true, y_score, float(params["recall"]))
    if base == "lift":
        return _lift(y_true, y_score, float(params.get("fraction", 0.1)))
    if base == "gain":
        return _gain(y_true, y_score, float(params.get("fraction", 0.1)))
    raise ValueError(f"unhandled builtin binary metric: {base!r}")  # pragma: no cover


def compute_regression_metric(
    spec: MetricSpec, y_true: "np.ndarray", y_pred: "np.ndarray"
) -> float:
    """Compute one builtin regression metric on target-scale predictions."""
    import numpy as np
    from sklearn.metrics import (
        mean_absolute_error,
        mean_absolute_percentage_error,
        mean_squared_error,
        r2_score,
    )

    base = spec.name
    if base == "rmse":
        return float(np.sqrt(mean_squared_error(y_true, y_pred)))
    if base == "mae":
        return float(mean_absolute_error(y_true, y_pred))
    if base == "r2":
        return float(r2_score(y_true, y_pred))
    if base == "mape":
        return float(mean_absolute_percentage_error(y_true, y_pred))
    raise ValueError(f"unknown builtin regression metric: {spec.name!r}")


def compute_metric(spec: MetricSpec, y_true: "np.ndarray", y_score: "np.ndarray") -> float:
    """Compute one builtin metric, dispatched by name (binary or regression).

    The binary and regression metric-name sets are disjoint, so the metric name
    alone selects the engine - the caller needs no task context. ``y_score`` is
    a probability for binary metrics and a target-scale prediction for
    regression; the adapters put both in the same ``prediction`` column.
    """
    if is_builtin_regression_metric(spec.name):
        return compute_regression_metric(spec, y_true, y_score)
    return compute_binary_metric(spec, y_true, y_score)


def compute_results(
    metric_specs: list[MetricSpec],
    y_true: "np.ndarray",
    y_score: "np.ndarray",
    slice_columns: dict[str, "np.ndarray"] | None = None,
) -> MetricResults:
    """Compute all requested builtin metrics (binary or regression, dispatched
    by name), plus per-slice values (FR-TEST-04).

    Hook metrics (``kind == "hook"``) are computed by the caller and merged
    afterwards; this function skips them.
    """
    builtin = [s for s in metric_specs if s.kind == "builtin"]
    metrics = {s.name: compute_metric(s, y_true, y_score) for s in builtin}

    slices: dict[str, dict[str, float]] = {}
    for column, values in (slice_columns or {}).items():
        for label, mask in _slice_groups(values).items():
            if int(mask.sum()) == 0 or len(set(y_true[mask].tolist())) < 2:
                # Degenerate slice: a single distinct label makes classification
                # metrics and R^2 undefined; skip it for either task.
                continue
            slices[f"{column}={label}"] = {
                s.name: compute_metric(s, y_true[mask], y_score[mask]) for s in builtin
            }
    return MetricResults(metrics=metrics, slices=slices)


#: A numeric slice column with more than this many distinct values is binned
#: into quantile ranges instead of one slice per value (R2-9).
_MAX_CATEGORICAL_SLICE_VALUES = 12
_SLICE_QUANTILE_BINS = 4


def _slice_groups(values: "np.ndarray") -> "dict[str, np.ndarray]":
    """Group a slice column into ``label -> row mask`` pairs.

    Categorical (or low-cardinality) columns give one group per distinct value.
    A high-cardinality numeric column is binned into quantile ranges (e.g.
    ``[25, 40)``) so slicing by a continuous feature (age, tenure) stays usable
    instead of exploding into hundreds of one-row slices (R2-9); a distribution
    too concentrated to yield distinct quantile edges falls back to per-value.
    """
    import numpy as np

    strings = [str(v) for v in values.tolist()]
    distinct = sorted(set(strings))
    if np.issubdtype(values.dtype, np.number) and len(distinct) > _MAX_CATEGORICAL_SLICE_VALUES:
        numeric = values.astype(float)
        edges = np.unique(np.quantile(numeric, np.linspace(0.0, 1.0, _SLICE_QUANTILE_BINS + 1)))
        if len(edges) >= 3:  # enough distinct quantiles to bin meaningfully
            bucket = np.digitize(numeric, edges[1:-1], right=False)
            last = len(edges) - 2
            return {
                f"[{edges[b]:g}, {edges[b + 1]:g}{']' if b == last else ')'}": bucket == b
                for b in range(len(edges) - 1)
            }
    array = np.asarray(strings)
    return {value: array == value for value in distinct}


def paired_bootstrap_delta(
    spec: MetricSpec,
    y_true: "np.ndarray",
    challenger_scores: "np.ndarray",
    champion_scores: "np.ndarray",
    *,
    greater_is_better: bool,
    confidence: float,
    n_resamples: int,
    seed: int,
) -> BootstrapDelta:
    """Paired bootstrap of a challenger-champion metric delta (ADR-18).

    Every resample draws rows with replacement ONCE and scores both models
    on those same rows, so sampling noise cancels and the delta distribution
    reflects only the model difference. Returns the one-sided lower bound at
    ``confidence``; the champion-gate criterion is ``lower >= min_delta``.

    Resamples that degenerate to a single class are skipped (ranking metrics
    are undefined there); if every resample degenerates, ``lower`` falls back
    to the point delta and ``n_resamples`` reports 0.
    """
    import numpy as np

    def _delta(indices: "np.ndarray") -> float:
        challenger = compute_metric(spec, y_true[indices], challenger_scores[indices])
        champion = compute_metric(spec, y_true[indices], champion_scores[indices])
        return (challenger - champion) if greater_is_better else (champion - challenger)

    n = len(y_true)
    point = _delta(np.arange(n))
    rng = np.random.default_rng(seed)
    deltas: list[float] = []
    for _ in range(n_resamples):
        indices = rng.integers(0, n, size=n)
        if np.unique(y_true[indices]).size < 2:
            continue
        deltas.append(_delta(indices))
    if not deltas:
        return BootstrapDelta(point=point, lower=point, confidence=confidence, n_resamples=0)
    lower = float(np.quantile(np.asarray(deltas), 1.0 - confidence))
    return BootstrapDelta(point=point, lower=lower, confidence=confidence, n_resamples=len(deltas))


def bootstrap_metric_lower_bound(
    spec: MetricSpec,
    y_true: "np.ndarray",
    y_score: "np.ndarray",
    *,
    confidence: float,
    n_resamples: int,
    seed: int,
) -> float:
    """A single model's metric under row resampling, oriented pessimistically
    (R2-7): the same bootstrap idea ADR-18 applies to the champion delta, used to
    defend a TUNING selection against validation-window luck.

    Returns the pessimistic bound at ``confidence`` - the lower percentile for a
    higher-is-better metric, the upper percentile for a lower-is-better one - so
    the tuning engine, optimizing in the metric's own direction, prefers params
    that are robustly good rather than merely lucky on the point estimate.
    Degenerate (single-class) resamples are skipped; if every resample
    degenerates, falls back to the point estimate.
    """
    import numpy as np

    n = len(y_true)
    point = compute_metric(spec, y_true, y_score)
    rng = np.random.default_rng(seed)
    values: list[float] = []
    for _ in range(n_resamples):
        indices = rng.integers(0, n, size=n)
        if np.unique(y_true[indices]).size < 2:
            continue
        values.append(compute_metric(spec, y_true[indices], y_score[indices]))
    if not values:
        return point
    quantile = (1.0 - confidence) if spec.greater_is_better else confidence
    return float(np.quantile(np.asarray(values), quantile))
