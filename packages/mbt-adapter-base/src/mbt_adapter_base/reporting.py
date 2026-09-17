"""Training-report numerics (ADR-30): time axes, bins, period cells.

Pure array code shared by the training job and any report engine plugin.
Everything is deterministic: no sampling without an explicit key, stable
orderings, and bin edges fitted once on the test split and then frozen.
numpy loads lazily so importing this module stays cheap (ADR-14).
"""

import calendar
import hashlib
from dataclasses import dataclass
from datetime import UTC, datetime, timedelta
from typing import TYPE_CHECKING, Any

import pyarrow as pa
import pyarrow.compute as pc

from mbt_adapter_base.specs import (
    DEFAULT_CUSTOM_EDGES,
    BinningSpec,
    CustomBinning,
    FixedWidthBinning,
    QuantileBinning,
    ReportPeriod,
    TopPercentBinning,
    parse_time_offset,
)

if TYPE_CHECKING:
    import numpy as np

#: The resolution time columns are compared at. Resolved windows are UTC
#: instants with second precision (ADR-12); microseconds cover every source.
_TIME_TYPE = pa.timestamp("us")

#: Equal-width bands a regression prediction's test range is cut into when a
#: ``fixed_width`` strategy declares no width.
_REGRESSION_FIXED_BANDS = 20

#: A strategy producing more bins than this is a configuration mistake (a
#: width in the wrong unit), not a table anyone reads.
MAX_BINS = 1000

#: ``day_of_month`` needs daily data: in a typical month, at least this many
#: distinct days must carry rows, or most day slots compare nothing.
_DAILY_DAYS_PER_MONTH = 14

#: The conventional PSI reading: below 0.1 stable, 0.1-0.25 worth a look,
#: above 0.25 shifted. Display bands only - gates use declared thresholds.
PSI_WARN_BAND = 0.1
PSI_FAIL_BAND = 0.25


# -- time ------------------------------------------------------------------------------


def parse_window_bound(iso: str) -> datetime:
    """A resolved window bound (``2026-06-30T00:00:00Z``) as a naive UTC datetime.

    Naive because the data adapters compare ``CAST(time AS TIMESTAMP)``
    against these bounds, which is a comparison of wall-clock values in UTC.
    """
    parsed = datetime.fromisoformat(iso.replace("Z", "+00:00"))
    if parsed.tzinfo is not None:
        parsed = parsed.astimezone(UTC).replace(tzinfo=None)
    return parsed


def naive_timestamps(column: pa.ChunkedArray | pa.Array) -> pa.ChunkedArray | pa.Array:
    """A time column as naive UTC ``timestamp[us]``, whatever it was stored as.

    Dates become midnight, ISO strings are parsed, and tz-aware timestamps are
    shifted to UTC - the same reading the data adapters' split predicates use.
    """
    if pa.types.is_dictionary(column.type):
        column = pc.cast(column, column.type.value_type)
    return pc.cast(column, _TIME_TYPE)


def window_mask(
    column: pa.ChunkedArray | pa.Array, start: str, end: str
) -> pa.ChunkedArray | pa.Array:
    """Rows whose time lies in the resolved window ``[start, end)``; nulls are out."""
    times = naive_timestamps(column)
    low = pa.scalar(parse_window_bound(start), type=_TIME_TYPE)
    high = pa.scalar(parse_window_bound(end), type=_TIME_TYPE)
    inside = pc.and_(pc.greater_equal(times, low), pc.less(times, high))
    return pc.fill_null(inside, False)


def time_values(column: pa.ChunkedArray | pa.Array) -> "np.ndarray":
    """A time column as ``datetime64[us]`` (NaT for nulls), for grouping."""
    import numpy as np

    times = naive_timestamps(column)
    if isinstance(times, pa.ChunkedArray):
        times = times.combine_chunks()
    return np.asarray(times.to_numpy(zero_copy_only=False), dtype="datetime64[us]")


# -- periods -----------------------------------------------------------------------------


def week_of_month(day: int) -> int:
    """W1 is days 1-7, W2 days 8-14, ... W5 days 29-31."""
    return (day - 1) // 7 + 1


@dataclass(frozen=True)
class PeriodCodes:
    """Per-row integer codes for one grain, and which rows have a time at all.

    Integers rather than strings because a panel can hold millions of rows:
    grouping happens on the codes, and only the few distinct cells present
    are turned into labels. Codes of rows where ``valid`` is False are
    meaningless.
    """

    cells: "np.ndarray"
    slots: "np.ndarray"
    valid: "np.ndarray"


def _calendar_parts(times: "np.ndarray") -> tuple["np.ndarray", "np.ndarray", "np.ndarray"]:
    """(months since the epoch, day of month, null mask) per row."""
    import numpy as np

    null = np.isnat(times)
    filled = np.where(null, np.datetime64(0, "us"), times)
    months = filled.astype("datetime64[M]")
    days = (filled.astype("datetime64[D]") - months.astype("datetime64[D]")).astype(np.int64) + 1
    return months.astype(np.int64), days, null


def period_codes(times: "np.ndarray", period: ReportPeriod) -> PeriodCodes:
    """The cell each time falls in, and the reference slot it is compared on.

    ``month`` cells are calendar months, all on slot 0 (the whole reference
    window); ``week_of_month`` cells are (month, W) on slot W; ``day_of_month``
    cells are (month, day) on slot day. :func:`cell_label` and
    :func:`slot_label` turn codes into ``2026-06``, ``2026-06/W1``, ``W1``,
    ``2026-06-05`` and ``D05``.
    """
    import numpy as np

    months, days, null = _calendar_parts(times)
    if period == "month":
        cells, slots = months, np.zeros_like(months)
    elif period == "week_of_month":
        slots = (days - 1) // 7 + 1
        cells = months * 10 + slots
    else:
        slots = days
        cells = months * 100 + days
    return PeriodCodes(cells=cells, slots=slots, valid=~null)


def _month_key(months_since_epoch: int) -> str:
    year, month = divmod(int(months_since_epoch), 12)
    return f"{1970 + year:04d}-{month + 1:02d}"


def cell_label(code: int, period: ReportPeriod) -> str:
    """``2026-06``, ``2026-06/W1`` or ``2026-06-05`` for a cell code."""
    if period == "month":
        return _month_key(code)
    if period == "week_of_month":
        return f"{_month_key(code // 10)}/W{code % 10}"
    return f"{_month_key(code // 100)}-{code % 100:02d}"


def slot_label(code: int, period: ReportPeriod) -> str:
    """``all``, ``W1`` or ``D05`` for a slot code."""
    if period == "month":
        return "all"
    if period == "week_of_month":
        return f"W{code}"
    return f"D{code:02d}"


def _month_bounds(key: str) -> tuple[datetime, datetime]:
    year, month = (int(part) for part in key[:7].split("-"))
    start = datetime(year, month, 1)
    days = calendar.monthrange(year, month)[1]
    return start, start + timedelta(days=days)


def cell_bounds(key: str, period: ReportPeriod) -> tuple[datetime, datetime]:
    """The calendar ``[start, end)`` of a cell label from :func:`cell_label`."""
    month_start, month_end = _month_bounds(key)
    if period == "month":
        return month_start, month_end
    if period == "week_of_month":
        week = int(key.rsplit("/W", 1)[1])
        start = month_start + timedelta(days=7 * (week - 1))
        return start, min(start + timedelta(days=7), month_end)
    start = datetime(month_start.year, month_start.month, int(key[-2:]))
    return start, start + timedelta(days=1)


def unsupported_periods(times: "np.ndarray", periods: list[ReportPeriod]) -> dict[str, str]:
    """Grains finer than the data's time resolution, with the reason.

    A monthly panel (one date per month) has no week or day view; a weekly
    panel has week slots but its day slots would compare nothing, because the
    same weekday lands on a different date next month.
    """
    import numpy as np

    valid = times[~np.isnat(times)]
    if valid.size == 0:
        return {}
    days = np.unique(valid.astype("datetime64[D]"))
    months = days.astype("datetime64[M]")
    per_month = np.unique(months, return_counts=True)[1]
    skipped: dict[str, str] = {}
    if "week_of_month" in periods and int(per_month.max()) < 2:
        skipped["week_of_month"] = "the time column holds at most one date per month"
    if "day_of_month" in periods and float(np.median(per_month)) < _DAILY_DAYS_PER_MONTH:
        skipped["day_of_month"] = (
            f"the time column is not daily (fewer than {_DAILY_DAYS_PER_MONTH} "
            "distinct days in a typical month)"
        )
    return skipped


def horizon_delta(horizon: str) -> tuple[int, timedelta]:
    """``label.horizon`` as (calendar months, fixed delta) for maturity checks."""
    count, unit = parse_time_offset(horizon)
    count = abs(count)
    if unit == "mo":
        return count, timedelta(0)
    fixed = {"d": timedelta(days=count), "w": timedelta(weeks=count), "h": timedelta(hours=count)}
    return 0, fixed[unit]


def _add_months(times: "np.ndarray", months: int) -> "np.ndarray":
    """Shift each time by calendar months, clamping the day like the window
    grammar does (``Jan 31 + 1mo -> Feb 28``) and keeping the time of day."""
    import numpy as np

    if months == 0:
        return times
    start = times.astype("datetime64[M]")
    day_index = (times.astype("datetime64[D]") - start.astype("datetime64[D]")).astype(np.int64)
    time_of_day = times - times.astype("datetime64[D]").astype(times.dtype)
    target = start + np.timedelta64(months, "M")
    length = (
        (target + np.timedelta64(1, "M")).astype("datetime64[D]") - target.astype("datetime64[D]")
    ).astype(np.int64)
    clamped = np.minimum(day_index, length - 1)
    shifted = (target.astype("datetime64[D]") + clamped.astype("timedelta64[D]")).astype(
        times.dtype
    )
    return np.asarray(shifted + time_of_day)


def maturity_mask(
    times: "np.ndarray",
    labelled: "np.ndarray",
    *,
    horizon: str | None,
    anchor: datetime,
) -> "np.ndarray":
    """Rows whose outcome has been observed by ``anchor``.

    With a declared ``label.horizon`` a row is mature once ``time + horizon``
    is at or before the anchor, AND its label is present - an upstream that
    fills a placeholder before the outcome lands must not pass for mature.
    Without one, a present label is the only evidence available.
    """
    import numpy as np

    mask = np.asarray(np.asarray(labelled, dtype=bool) & ~np.isnat(times))
    if horizon is None:
        return mask
    months, fixed = horizon_delta(horizon)
    filled = np.where(np.isnat(times), np.datetime64(0, "us"), times).astype("datetime64[us]")
    observed = _add_months(filled, months) + np.timedelta64(int(fixed.total_seconds() * 1e6), "us")
    cutoff = np.datetime64(anchor.replace(tzinfo=None), "us")
    return np.asarray(mask & (observed <= cutoff))


# -- metrics beyond the engine -----------------------------------------------------------


def ks_statistic(y_true: "np.ndarray", y_score: "np.ndarray") -> float:
    """Kolmogorov-Smirnov separation: max |TPR - FPR| over score thresholds."""
    import numpy as np

    y = np.asarray(y_true, dtype=float)
    s = np.asarray(y_score, dtype=float)
    positives = float((y == 1).sum())
    negatives = float((y == 0).sum())
    if positives == 0 or negatives == 0:
        return float("nan")
    order = np.argsort(-s, kind="mergesort")
    s, y = s[order], y[order]
    tpr = np.cumsum(y == 1) / positives
    fpr = np.cumsum(y == 0) / negatives
    # Only compare at the end of each run of tied scores: a threshold cannot
    # separate rows that share a score.
    last_of_tie = np.append(s[1:] != s[:-1], True)
    return float(np.max(np.abs(tpr - fpr)[last_of_tie]))


# -- binning -----------------------------------------------------------------------------


@dataclass(frozen=True)
class FittedBins:
    """One strategy's bins, fitted on the reference (test) scores and frozen.

    Band strategies hold ascending ``edges`` ``e0 < e1 < ... < ek``: bin ``i``
    covers ``(e_i, e_{i+1}]``, the lowest is closed on both sides, and a score
    outside ``[e0, ek]`` counts in the outermost bin so no row drops out.
    ``top_percent`` instead holds each cutoff and the reference score a row
    must reach to be in that top slice.
    """

    name: str
    strategy: str
    edges: tuple[float, ...] = ()
    cutoffs: tuple[float, ...] = ()
    thresholds: tuple[float, ...] = ()

    @property
    def n_bins(self) -> int:
        return len(self.cutoffs) if self.strategy == "top_percent" else len(self.edges) - 1


def _format_number(value: float) -> str:
    return f"{value:g}"


def fit_bins(spec: BinningSpec, reference: "np.ndarray", *, probability: bool) -> FittedBins | None:
    """Fit one strategy on the reference scores.

    Returns None only for ``custom`` without edges on a regression model,
    which ``binning: all`` can produce and the report then skips (a declared
    one is a parse error). Raises ValueError for a strategy that would
    produce an unreadable number of bins.
    """
    import numpy as np

    scores = np.asarray(reference, dtype=float)
    scores = scores[~np.isnan(scores)]
    if scores.size == 0:
        raise ValueError("cannot fit bins on an empty reference split")
    low, high = (0.0, 1.0) if probability else (float(scores.min()), float(scores.max()))
    if isinstance(spec, TopPercentBinning):
        thresholds = tuple(
            float(np.quantile(scores, 1.0 - cutoff / 100.0, method="higher"))
            for cutoff in spec.cutoffs
        )
        return FittedBins(
            name="top_percent",
            strategy="top_percent",
            cutoffs=tuple(spec.cutoffs),
            thresholds=thresholds,
        )
    if isinstance(spec, QuantileBinning):
        # "lower" keeps every edge an observed score: an interpolated edge
        # between two tied groups would invent an empty band.
        inner = np.quantile(scores, np.linspace(0.0, 1.0, spec.bins + 1), method="lower")[1:-1]
        interior = sorted({float(edge) for edge in inner if low < edge < high})
        edges = (low, *interior, high)
        name = f"quantile_{spec.bins}"
    elif isinstance(spec, FixedWidthBinning):
        if spec.width is None and not probability:
            width = (high - low) / _REGRESSION_FIXED_BANDS if high > low else 1.0
            name = "fixed_width"
        else:
            width = spec.width if spec.width is not None else 0.05
            name = f"fixed_width_{_format_number(width)}"
        count = max(1, int(np.ceil((high - low) / width - 1e-9)))
        if count > MAX_BINS:
            raise ValueError(
                f"fixed_width {width:g} cuts the score range [{low:g}, {high:g}] into "
                f"{count} bins (more than {MAX_BINS}); widen it"
            )
        edges = (*(round(low + width * i, 12) for i in range(count)), high)
    else:
        assert isinstance(spec, CustomBinning)
        declared = spec.edges if spec.edges is not None else None
        if declared is None:
            if not probability:
                return None
            declared = list(DEFAULT_CUSTOM_EDGES)
        edges = tuple(float(edge) for edge in declared)
        name = "custom"
    if len(edges) - 1 > MAX_BINS:  # pragma: no cover - quantile bins are capped at 100
        raise ValueError(f"{name} produces more than {MAX_BINS} bins")
    return FittedBins(name=name, strategy=spec.strategy, edges=edges)


def assign_bins(bins: FittedBins, scores: "np.ndarray") -> "np.ndarray":
    """Band index per score, 0 = lowest band (band strategies only)."""
    import numpy as np

    interior = np.asarray(bins.edges[1:-1], dtype=float)
    indices = np.searchsorted(interior, np.asarray(scores, dtype=float), side="left")
    return np.clip(indices, 0, bins.n_bins - 1)


def _rate(numerator: float, denominator: float) -> float | None:
    return numerator / denominator if denominator else None


def bin_rows(
    bins: FittedBins,
    scores: "np.ndarray",
    labels: "np.ndarray",
    labelled: "np.ndarray",
    *,
    binary: bool,
) -> list[dict[str, Any]]:
    """One table for one split or cell: rows ordered from the top score down.

    ``labels`` may hold NaN where ``labelled`` is False; those rows count in
    ``n`` and ``mean_score`` (what was scored) but not in any label statistic
    (what was observed).
    """
    if bins.strategy == "top_percent":
        return _top_percent_rows(bins, scores, labels, labelled, binary=binary)
    import numpy as np

    scores = np.asarray(scores, dtype=float)
    labels = np.asarray(labels, dtype=float)
    labelled = np.asarray(labelled, dtype=bool)
    index = assign_bins(bins, scores) if scores.size else np.zeros(0, dtype=int)
    total_n = int(scores.size)
    total_labelled = int(labelled.sum())
    total_label = float(labels[labelled].sum()) if total_labelled else 0.0
    total_negative = float(total_labelled - total_label) if binary else 0.0
    base_rate = _rate(total_label, total_labelled)
    rows: list[dict[str, Any]] = []
    cum_n = cum_labelled = 0
    cum_label = cum_negative = 0.0
    for rank, band in enumerate(range(bins.n_bins - 1, -1, -1), start=1):
        in_band = index == band
        observed = in_band & labelled
        n = int(in_band.sum())
        n_labelled = int(observed.sum())
        label_sum = float(labels[observed].sum()) if n_labelled else 0.0
        cum_n += n
        cum_labelled += n_labelled
        cum_label += label_sum
        rate = _rate(label_sum, n_labelled)
        row: dict[str, Any] = {
            "bin": rank,
            "lower": bins.edges[band],
            "upper": bins.edges[band + 1],
            "n": n,
            "share": _rate(n, total_n),
            "mean_score": float(scores[in_band].mean()) if n else None,
            "n_labelled": n_labelled,
            "label_rate" if binary else "label_mean": rate,
            "lift": (rate / base_rate) if rate is not None and base_rate else None,
            "cum_share": _rate(cum_n, total_n),
        }
        if binary:
            negatives = n_labelled - label_sum
            cum_negative += negatives
            cum_rate = _rate(cum_label, cum_labelled)
            row.update(
                {
                    "positives": int(label_sum),
                    "cum_positives": int(cum_label),
                    "capture": _rate(cum_label, total_label),
                    "cum_label_rate": cum_rate,
                    "cum_lift": (cum_rate / base_rate)
                    if cum_rate is not None and base_rate
                    else None,
                    "ks": abs(cum_label / total_label - cum_negative / total_negative)
                    if total_label and total_negative
                    else None,
                }
            )
        rows.append(row)
    return rows


def _top_percent_rows(
    bins: FittedBins,
    scores: "np.ndarray",
    labels: "np.ndarray",
    labelled: "np.ndarray",
    *,
    binary: bool,
) -> list[dict[str, Any]]:
    import numpy as np

    scores = np.asarray(scores, dtype=float)
    labels = np.asarray(labels, dtype=float)
    labelled = np.asarray(labelled, dtype=bool)
    total_n = int(scores.size)
    total_labelled = int(labelled.sum())
    total_label = float(labels[labelled].sum()) if total_labelled else 0.0
    base_rate = _rate(total_label, total_labelled)
    rows: list[dict[str, Any]] = []
    for cutoff, threshold in zip(bins.cutoffs, bins.thresholds, strict=True):
        selected = scores >= threshold
        observed = selected & labelled
        n = int(selected.sum())
        n_labelled = int(observed.sum())
        label_sum = float(labels[observed].sum()) if n_labelled else 0.0
        rate = _rate(label_sum, n_labelled)
        row: dict[str, Any] = {
            "top_percent": cutoff,
            "threshold": threshold,
            "n": n,
            # The share actually selected: on the reference it is the cutoff up
            # to ties; on a later period its drift IS the stability signal.
            "share": _rate(n, total_n),
            "n_labelled": n_labelled,
            "precision" if binary else "label_mean": rate,
            "lift": (rate / base_rate) if rate is not None and base_rate else None,
        }
        if binary:
            row["positives"] = int(label_sum)
            row["capture"] = _rate(label_sum, total_label)
        rows.append(row)
    return rows


def histogram_rows(
    scores: "np.ndarray", labels: "np.ndarray", labelled: "np.ndarray", *, bins: FittedBins
) -> list[dict[str, Any]]:
    """Score distribution per band, split by observed class (binary only)."""
    import numpy as np

    scores = np.asarray(scores, dtype=float)
    labels = np.asarray(labels, dtype=float)
    labelled = np.asarray(labelled, dtype=bool)
    index = assign_bins(bins, scores) if scores.size else np.zeros(0, dtype=int)
    positive = labelled & (labels == 1)
    negative = labelled & (labels == 0)
    n_pos = int(positive.sum())
    n_neg = int(negative.sum())
    rows = []
    for band in range(bins.n_bins):
        in_band = index == band
        rows.append(
            {
                "lower": bins.edges[band],
                "upper": bins.edges[band + 1],
                "n": int(in_band.sum()),
                "share": _rate(int(in_band.sum()), int(scores.size)),
                "share_positive": _rate(int((in_band & positive).sum()), n_pos),
                "share_negative": _rate(int((in_band & negative).sum()), n_neg),
            }
        )
    return rows


def score_summary(scores: "np.ndarray") -> dict[str, float | int | None]:
    """Count, mean, spread and the percentiles a reviewer reads first."""
    import numpy as np

    values = np.asarray(scores, dtype=float)
    values = values[~np.isnan(values)]
    summary: dict[str, float | int | None] = {"n": int(values.size)}
    if values.size == 0:
        return summary
    summary.update({"mean": float(values.mean()), "std": float(values.std())})
    for pct in (0, 1, 5, 25, 50, 75, 95, 99, 100):
        summary[f"p{pct}"] = float(np.percentile(values, pct))
    return summary


# -- deterministic sampling ------------------------------------------------------------


def key_hash_buckets(table: pa.Table, key_columns: list[str]) -> list[int]:
    """The canonical row digest per row: unsigned low 64 bits of the md5 of the
    '|'-joined key values (nulls as ''), the preimage the data adapters hash
    for sampling (F19), so a capped sample is a pure function of the keys."""
    columns = [table.column(name).to_pylist() for name in key_columns]
    buckets = []
    for values in zip(*columns, strict=True):
        preimage = "|".join("" if v is None else str(v) for v in values)
        digest = hashlib.md5(preimage.encode(), usedforsecurity=False).hexdigest()
        buckets.append(int(digest[16:32], 16))
    return buckets


def capped_indices(table: pa.Table, key_columns: list[str], max_rows: int | None) -> list[int]:
    """Row indices to keep: all of them, or the ``max_rows`` smallest digests
    (ties by position), returned in the table's original order.

    Deterministic - the same rows always give the same sample, and a smaller
    cap keeps a subset of a larger one. A table that grows can displace kept
    rows with newer ones whose digests are smaller."""
    if max_rows is None or table.num_rows <= max_rows:
        return list(range(table.num_rows))
    buckets = key_hash_buckets(table, key_columns)
    ranked = sorted(range(table.num_rows), key=lambda i: (buckets[i], i))
    return sorted(ranked[:max_rows])
