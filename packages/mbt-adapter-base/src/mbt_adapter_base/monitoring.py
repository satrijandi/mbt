"""Distribution-shift monitoring: baselines and PSI/KS statistics (ADR-20/21).

Vocabulary: this codebase reserves "drift" for data-snapshot drift (data
versioning, ADR-11); distribution monitoring is called "shift".

Training jobs build a ``MonitoringBaseline`` from the post-hook train split
and the test-split score distribution; scoring jobs compute ``ShiftStat``s
against it; core applies the declared thresholds ("jobs compute, core
compares", ADR-3).

Everything here is deterministic by construction: quantile grids, stable
category ordering, no sampling. numpy loads lazily so importing this module
stays cheap (ADR-14).
"""

import json
from fnmatch import fnmatchcase
from pathlib import Path
from typing import TYPE_CHECKING, Literal

import pyarrow as pa
from pydantic import BaseModel, ConfigDict, Field

from mbt_adapter_base.interchange import MonitorStats, ShiftStat
from mbt_adapter_base.specs import FeatureShiftSpec, MonitorsSpec, PredictionShiftSpec

if TYPE_CHECKING:
    import numpy as np

BASELINE_SCHEMA_VERSION = 1

#: p0..p100 grid size; PSI bins derive from the deciles of this grid.
_GRID_POINTS = 101

#: Categorical baselines keep at most this many distinct values; the rest
#: pool into ``_OTHER`` (bounds baseline size on high-cardinality columns).
_MAX_CATEGORIES = 50

_OTHER = "__other__"

#: Epsilon smoothing for PSI bin proportions (avoids log(0) blowups).
_PSI_EPSILON = 1e-4


class _BaselineModel(BaseModel):
    model_config = ConfigDict(extra="forbid")


class FeatureBaseline(_BaselineModel):
    """Reference distribution of one post-hook feature column."""

    kind: Literal["numeric", "categorical"]
    quantiles: list[float] | None = None  # numeric: p0..p100 of the train split
    categories: list[str] | None = None  # categorical: values + "__other__"
    proportions: list[float] | None = None  # categorical: per-category share
    null_fraction: float = 0.0
    n: int


class ScoreBaseline(_BaselineModel):
    """Reference distribution of the champion's test-split scores."""

    quantiles: list[float]  # p0..p100
    n: int


class MonitoringBaseline(_BaselineModel):
    """The baseline artifact exported next to the model artifact (ADR-21)."""

    baseline_schema_version: int = BASELINE_SCHEMA_VERSION
    model_name: str
    feature_columns: list[str]  # post-hook selected features, in order
    features: dict[str, FeatureBaseline] = Field(default_factory=dict)
    score: ScoreBaseline


def _is_numeric(dtype: pa.DataType) -> bool:
    return bool(pa.types.is_integer(dtype) or pa.types.is_floating(dtype))


def _is_categorical(dtype: pa.DataType) -> bool:
    return bool(
        pa.types.is_string(dtype)
        or pa.types.is_large_string(dtype)
        or pa.types.is_boolean(dtype)
        or pa.types.is_dictionary(dtype)
    )


def _numeric_values(column: pa.ChunkedArray) -> "np.ndarray":
    """Non-null values as a float64 array."""
    import numpy as np

    return np.asarray(column.drop_null().cast(pa.float64()).to_numpy(zero_copy_only=False))


def _categorical_values(column: pa.ChunkedArray) -> list[str]:
    """Non-null values as strings (dictionary/bool columns included)."""
    dtype = column.type
    if pa.types.is_dictionary(dtype):
        column = column.cast(dtype.value_type)
    return [str(v) for v in column.drop_null().cast(pa.string()).to_pylist()]


def _numeric_baseline(column: pa.ChunkedArray) -> FeatureBaseline | None:
    import numpy as np

    values = _numeric_values(column)
    if values.size == 0:
        return None
    quantiles = np.quantile(values, np.linspace(0.0, 1.0, _GRID_POINTS))
    return FeatureBaseline(
        kind="numeric",
        quantiles=[float(q) for q in quantiles],
        null_fraction=float(column.null_count / max(len(column), 1)),
        n=int(values.size),
    )


def _categorical_baseline(column: pa.ChunkedArray) -> FeatureBaseline | None:
    values = _categorical_values(column)
    if not values:
        return None
    counts: dict[str, int] = {}
    for value in values:
        counts[value] = counts.get(value, 0) + 1
    # Deterministic top-K: by count desc, then name asc.
    ranked = sorted(counts.items(), key=lambda kv: (-kv[1], kv[0]))
    kept = ranked[:_MAX_CATEGORIES]
    other = sum(count for _, count in ranked[_MAX_CATEGORIES:])
    total = len(values)
    categories = [name for name, _ in kept] + [_OTHER]
    proportions = [count / total for _, count in kept] + [other / total]
    return FeatureBaseline(
        kind="categorical",
        categories=categories,
        proportions=proportions,
        null_fraction=float(column.null_count / max(len(column), 1)),
        n=total,
    )


def build_baseline(
    features: pa.Table,
    feature_columns: list[str],
    test_scores: "np.ndarray",
    *,
    model_name: str,
) -> MonitoringBaseline:
    """Build the monitoring baseline from the post-hook train split.

    Columns that are neither numeric nor categorical (or are entirely null)
    are omitted; scoring runs report them as skipped.
    """
    import numpy as np

    baselines: dict[str, FeatureBaseline] = {}
    for name in feature_columns:
        if name not in features.column_names:
            continue
        column = features.column(name)
        baseline: FeatureBaseline | None = None
        if _is_numeric(column.type):
            baseline = _numeric_baseline(column)
        elif _is_categorical(column.type):
            baseline = _categorical_baseline(column)
        if baseline is not None:
            baselines[name] = baseline

    scores = np.asarray(test_scores, dtype=np.float64)
    score_quantiles = np.quantile(scores, np.linspace(0.0, 1.0, _GRID_POINTS))
    return MonitoringBaseline(
        model_name=model_name,
        feature_columns=list(feature_columns),
        features=baselines,
        score=ScoreBaseline(quantiles=[float(q) for q in score_quantiles], n=int(scores.size)),
    )


def write_baseline(baseline: MonitoringBaseline, path: Path) -> None:
    path.write_text(json.dumps(baseline.model_dump(), sort_keys=True, indent=1) + "\n")


def read_baseline(path: Path) -> MonitoringBaseline:
    return MonitoringBaseline.model_validate(json.loads(path.read_text()))


def _grid_ecdf(quantiles: list[float], edge: float) -> float:
    """Baseline ECDF at ``edge``, approximated from the stored quantile grid."""
    below = sum(1 for q in quantiles if q <= edge)
    return below / len(quantiles)


def _numeric_psi(quantiles: list[float], current: "np.ndarray") -> float:
    import numpy as np

    # 10 equal-frequency bins from the baseline deciles; duplicate edges
    # (heavy ties) collapse, and the expected mass per collapsed bin is
    # recovered from the full quantile grid. Bins are (left, right] with
    # open ends, matching the ``<=`` ECDF convention, so ties on an edge
    # land in the same bin on both sides of the comparison.
    deciles = quantiles[:: (_GRID_POINTS - 1) // 10]
    inner = sorted(set(deciles[1:-1]))
    expected: list[float] = []
    previous = 0.0
    for edge in inner:
        ecdf = _grid_ecdf(quantiles, edge)
        expected.append(ecdf - previous)
        previous = ecdf
    expected.append(1.0 - previous)

    indices = np.searchsorted(np.asarray(inner), current, side="left")
    counts = np.bincount(indices, minlength=len(inner) + 1)
    actual = counts / current.size
    value = 0.0
    for expected_share, actual_share in zip(expected, actual, strict=True):
        p = max(expected_share, _PSI_EPSILON)
        q = max(float(actual_share), _PSI_EPSILON)
        value += (q - p) * float(np.log(q / p))
    return float(value)


def _categorical_psi(baseline: FeatureBaseline, current: list[str]) -> float:
    import numpy as np

    categories = baseline.categories or []
    proportions = baseline.proportions or []
    known = set(categories) - {_OTHER}
    counts = dict.fromkeys(categories, 0)
    for observed in current:
        counts[observed if observed in known else _OTHER] += 1
    total = len(current)
    statistic = 0.0
    for name, share in zip(categories, proportions, strict=True):
        p = max(share, _PSI_EPSILON)
        q = max(counts[name] / total, _PSI_EPSILON)
        statistic += (q - p) * float(np.log(q / p))
    return float(statistic)


def _numeric_ks(quantiles: list[float], current: "np.ndarray") -> float:
    import numpy as np

    # KS statistic over the stored grid: at baseline quantile q_i the
    # baseline ECDF is i/(grid-1); compare against the current ECDF.
    ordered = np.sort(current)
    statistic = 0.0
    for i, q in enumerate(quantiles):
        baseline_ecdf = i / (_GRID_POINTS - 1)
        current_ecdf = float(np.searchsorted(ordered, q, side="right")) / ordered.size
        statistic = max(statistic, abs(current_ecdf - baseline_ecdf))
    return float(statistic)


def _categorical_ks(baseline: FeatureBaseline, current: list[str]) -> float:
    """Categorical analog of KS: max per-category proportion difference."""
    categories = baseline.categories or []
    proportions = baseline.proportions or []
    known = set(categories) - {_OTHER}
    counts = dict.fromkeys(categories, 0)
    for value in current:
        counts[value if value in known else _OTHER] += 1
    total = len(current)
    return max(
        abs(counts[name] / total - p) for name, p in zip(categories, proportions, strict=True)
    )


def _feature_shift(
    baseline: FeatureBaseline, column: pa.ChunkedArray, method: Literal["psi", "ks"]
) -> ShiftStat | None:
    """One feature's shift statistic; None when the column can't be compared."""
    if baseline.kind == "numeric":
        if not _is_numeric(column.type):
            return None
        values = _numeric_values(column)
        if values.size == 0:
            return None
        quantiles = baseline.quantiles or []
        value = (
            _numeric_psi(quantiles, values) if method == "psi" else _numeric_ks(quantiles, values)
        )
        return ShiftStat(
            method=method, value=value, n_current=int(values.size), n_baseline=baseline.n
        )
    if not _is_categorical(column.type):
        return None
    strings = _categorical_values(column)
    if not strings:
        return None
    value = (
        _categorical_psi(baseline, strings)
        if method == "psi"
        else _categorical_ks(baseline, strings)
    )
    return ShiftStat(method=method, value=value, n_current=len(strings), n_baseline=baseline.n)


def _score_shift(
    baseline: ScoreBaseline, scores: "np.ndarray", method: Literal["psi", "ks"]
) -> ShiftStat:
    value = (
        _numeric_psi(baseline.quantiles, scores)
        if method == "psi"
        else _numeric_ks(baseline.quantiles, scores)
    )
    return ShiftStat(method=method, value=value, n_current=int(scores.size), n_baseline=baseline.n)


def _selected_features(baseline: MonitoringBaseline, spec: FeatureShiftSpec) -> list[str]:
    selected = []
    for name in baseline.feature_columns:
        if not any(fnmatchcase(name, glob) for glob in spec.include):
            continue
        if any(fnmatchcase(name, glob) for glob in spec.exclude):
            continue
        selected.append(name)
    return selected


def compute_monitor_stats(
    baseline: MonitoringBaseline,
    features: pa.Table,
    scores: "np.ndarray",
    monitors: MonitorsSpec,
) -> MonitorStats:
    """Compute every declared shift statistic for one scoring batch.

    An empty batch yields empty statistics (nothing to compare); the
    zero-row condition is warned about at materialization time.
    """
    import numpy as np

    feature_stats: dict[str, ShiftStat] = {}
    skipped: list[str] = []
    spec: FeatureShiftSpec | None = monitors.feature_shift
    if spec is not None and features.num_rows > 0:
        for name in _selected_features(baseline, spec):
            feature_baseline = baseline.features.get(name)
            if feature_baseline is None or name not in features.column_names:
                skipped.append(name)
                continue
            stat = _feature_shift(feature_baseline, features.column(name), spec.method)
            if stat is None:
                skipped.append(name)
            else:
                feature_stats[name] = stat

    prediction_stat: ShiftStat | None = None
    shift_spec: PredictionShiftSpec | None = monitors.prediction_shift
    score_values = np.asarray(scores, dtype=np.float64)
    if shift_spec is not None and score_values.size > 0:
        prediction_stat = _score_shift(baseline.score, score_values, shift_spec.method)

    return MonitorStats(
        feature_shift=feature_stats,
        prediction_shift=prediction_stat,
        skipped_features=skipped,
    )
