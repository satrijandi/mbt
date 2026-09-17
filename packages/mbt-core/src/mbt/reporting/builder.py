"""Compute the training report (ADR-30) from scored, aligned splits.

The job predicts each split once and hands this module a ``ScoredSplit`` per
split: the table the model consumed, the raw key and time columns, labels and
scores, all row-aligned. Everything here is arithmetic over those arrays; the
files it produces are written by ``mbt.reporting.writer``.

Reference is always the test split: bin edges and top-percent cutoffs are
fitted on it, period cells compare with its matching slot, and stability is
measured against a baseline built from it.
"""

from dataclasses import dataclass, field
from datetime import datetime
from pathlib import Path
from typing import TYPE_CHECKING, Any, Literal

import pyarrow as pa

from mbt.contracts import (
    OUT_OF_TIME_SPLIT,
    FeatureShiftSpec,
    MetricSpec,
    MonitorsSpec,
    MonitorStats,
    PeriodCell,
    PredictionShiftSpec,
    ReportSpec,
    ReportSummary,
    StabilityCell,
    StabilitySpec,
)
from mbt_adapter_base.reporting import (
    PSI_FAIL_BAND,
    PSI_WARN_BAND,
    FittedBins,
    bin_rows,
    cell_bounds,
    cell_label,
    fit_bins,
    histogram_rows,
    ks_statistic,
    maturity_mask,
    period_codes,
    score_summary,
    slot_label,
    unsupported_periods,
)
from mbt_adapter_base.specs import FixedWidthBinning, GatePeriod, ReportPeriod

if TYPE_CHECKING:
    import numpy as np

#: Always reported per cell, whatever the model declares: the numbers a
#: reviewer compares first. Declared builtin metrics are added to them.
_DEFAULT_BINARY_METRICS = ("roc_auc", "pr_auc")
_DEFAULT_REGRESSION_METRICS = ("rmse", "mae")

#: The report grains in display order; ``window`` is always computed.
_GRAINS: tuple[ReportPeriod, ...] = ("month", "week_of_month", "day_of_month")

#: A table is a list of flat rows; the writer turns each into a CSV.
Table = list[dict[str, Any]]


@dataclass
class ScoredSplit:
    """One split, scored once and aligned row by row."""

    name: str
    features: pa.Table  # what the model consumed: features, target, slices
    passthrough: pa.Table  # the raw key columns, then the time column
    key_columns: list[str]  # the dataset's sample_key
    labels: "np.ndarray"  # float; NaN where the label is null
    scores: "np.ndarray"  # P(class 1), or the regression prediction
    times: "np.ndarray | None"  # datetime64[us]; None without a time column

    @property
    def n_rows(self) -> int:
        return int(self.scores.size)


@dataclass
class ReportInputs:
    """Everything the report needs besides the scored splits."""

    model_name: str
    binary: bool
    report: ReportSpec
    feature_columns: list[str]
    importance: dict[str, float]
    metric_specs: list[MetricSpec]
    anchor: datetime
    horizon: str | None
    stability: StabilitySpec | None = None
    gate_periods: set[GatePeriod] = field(default_factory=set)
    #: The resolved after-test window, which bounds the window cell; its
    #: first and last row stand in when it is unknown.
    after_window: tuple[str, str] | None = None


@dataclass
class ReportData:
    """The computed report: the summary core gates on, plus every table."""

    summary: ReportSummary
    tables: dict[str, Table] = field(default_factory=dict)
    bins: list[FittedBins] = field(default_factory=list)
    predictions: dict[str, pa.Table] = field(default_factory=dict)
    #: The fewest labelled rows a cell or bin needs before a rate is shown.
    min_rows: int = 30
    #: The report engine whose drift reports ``files`` holds, if any.
    engine: str | None = None
    #: Rendered files to copy into the report, by their path inside it.
    files: dict[str, Path] = field(default_factory=dict)


def build_report(inputs: ReportInputs, splits: dict[str, ScoredSplit]) -> ReportData:
    """The whole report; ``splits`` must hold ``test`` (the reference)."""
    import numpy as np

    test = splits["test"]
    after = splits.get(OUT_OF_TIME_SPLIT)
    if after is not None and after.n_rows == 0:
        after = None
    warnings: list[str] = []
    data = ReportData(
        summary=ReportSummary(
            anchor=inputs.anchor.isoformat(),
            reference_rows=test.n_rows,
            out_of_time_rows=after.n_rows if after is not None else 0,
        ),
        min_rows=inputs.report.min_rows,
    )
    data.predictions = _predictions(inputs, splits)
    data.summary.split_metrics = {
        name: _metrics(inputs, *_observed(inputs, split))
        for name, split in splits.items()
        if split.n_rows
    }
    data.tables["split_metrics"] = [
        {"split": name, **metrics} for name, metrics in data.summary.split_metrics.items()
    ]
    data.bins = _fit_all_bins(inputs, test, warnings)
    data.tables.update(_binning_tables(inputs, data.bins, splits))
    data.tables.update(_distribution_tables(inputs, splits, test))
    data.tables["feature_importance"] = _importance_table(inputs)
    data.summary.importance = dict(_ranked_importance(inputs)[: inputs.report.importance.top_n])

    if after is not None and test.times is not None and after.times is not None:
        grains = _requested_grains(inputs)
        both = np.concatenate([test.times, after.times])
        skipped = unsupported_periods(both, list(grains))
        data.summary.skipped_periods = skipped
        warnings.extend(f"{grain} skipped: {reason}" for grain, reason in skipped.items())
        active = [grain for grain in grains if grain not in skipped]
        cells, rows = _period_cells(inputs, test, after, active)
        data.summary.periods = cells
        data.tables["performance_by_period"] = rows
        stability, score_rows, feature_rows = _stability(inputs, test, after, active)
        data.summary.stability = stability
        data.tables["stability_scores"] = score_rows
        data.tables["stability_features"] = feature_rows
    elif OUT_OF_TIME_SPLIT in splits:
        warnings.append("the after-test window holds no rows; nothing to compare yet")
    data.summary.warnings = warnings
    return data


# -- splits and metrics ---------------------------------------------------------------------


def _observed(inputs: ReportInputs, split: ScoredSplit) -> tuple["np.ndarray", "np.ndarray"]:
    """Labels and scores of the rows whose outcome is known (see ``_labelled_mask``)."""
    labelled = _labelled_mask(inputs, split)
    return split.labels[labelled], split.scores[labelled]


def _metric_specs(inputs: ReportInputs) -> list[MetricSpec]:
    defaults = _DEFAULT_BINARY_METRICS if inputs.binary else _DEFAULT_REGRESSION_METRICS
    declared = [spec for spec in inputs.metric_specs if spec.kind == "builtin"]
    names = {spec.name for spec in declared}
    return [*(MetricSpec(name=n) for n in defaults if n not in names), *declared]


def _metrics(inputs: ReportInputs, labels: "np.ndarray", scores: "np.ndarray") -> dict[str, float]:
    """The cell's metrics, or none when they are undefined for it.

    Undefined means no labelled rows, or - for a classifier - a single class:
    every ranking metric divides by the missing class.
    """
    import numpy as np

    from mbt_adapter_base.metrics import compute_metric

    if labels.size == 0:
        return {}
    if inputs.binary and np.unique(labels).size < 2:
        return {}
    out: dict[str, float] = {}
    for spec in _metric_specs(inputs):
        try:
            value = float(compute_metric(spec, labels, scores))
        except (ValueError, ZeroDivisionError):  # undefined on this cell
            continue
        if np.isfinite(value):
            out[spec.name] = value
    if inputs.binary:
        out["ks"] = ks_statistic(labels, scores)
        out["label_rate"] = float(labels.mean())
    else:
        out["label_mean"] = float(labels.mean())
    return out


def _predictions(inputs: ReportInputs, splits: dict[str, ScoredSplit]) -> dict[str, pa.Table]:
    """Row-level predictions per split, when the project opted in."""
    import numpy as np

    from mbt_adapter_base.reporting import capped_indices

    settings = inputs.report.predictions
    if not settings.enabled:
        return {}
    out: dict[str, pa.Table] = {}
    for name, split in splits.items():
        names = pa.array([name] * split.n_rows, type=pa.string())
        if split.passthrough.num_columns and split.passthrough.num_rows == split.n_rows:
            table = split.passthrough.append_column("split", names)
        else:  # nothing to carry: no keys, no time column
            table = pa.table({"split": names})
        label = pa.array(split.labels, type=pa.float64(), from_pandas=True)
        observed = split.labels[~np.isnan(split.labels)]
        if inputs.binary and np.array_equal(observed, np.round(observed)):
            label = label.cast(pa.int64())  # 0 / 1 as the source has them; missing stays null
        table = table.append_column("label", label)
        if inputs.binary:
            table = table.append_column("p0", pa.array(1.0 - split.scores, type=pa.float64()))
            table = table.append_column("p1", pa.array(split.scores, type=pa.float64()))
        else:
            table = table.append_column("prediction", pa.array(split.scores, type=pa.float64()))
        if settings.max_rows is not None and split.key_columns:
            keep = capped_indices(split.passthrough, split.key_columns, settings.max_rows)
            table = table.take(pa.array(keep, type=pa.int64()))
        elif settings.max_rows is not None and table.num_rows > settings.max_rows:
            # No key to sample by (a dataset from before sample_key was
            # required): the first rows, which is at least deterministic.
            table = table.slice(0, settings.max_rows)
        out[name] = table
    return out


# -- binning -------------------------------------------------------------------------------


def _fit_all_bins(inputs: ReportInputs, test: ScoredSplit, warnings: list[str]) -> list[FittedBins]:
    fitted: list[FittedBins] = []
    for spec in inputs.report.binning_strategies:
        try:
            bins = fit_bins(spec, test.scores, probability=inputs.binary)
        except ValueError as exc:
            warnings.append(f"{spec.strategy} binning skipped: {exc}")
            continue
        if bins is None:
            warnings.append(
                "custom binning skipped: a regression prediction has no default edges; "
                "declare them under report.binning"
            )
            continue
        fitted.append(bins)
    return fitted


def _binning_tables(
    inputs: ReportInputs, fitted: list[FittedBins], splits: dict[str, ScoredSplit]
) -> dict[str, Table]:
    """One long table per strategy: every split, and every after-test month."""
    import numpy as np

    tables: dict[str, Table] = {}
    for bins in fitted:
        rows: Table = []
        for name, split in splits.items():
            if not split.n_rows:
                continue
            labelled = _labelled_mask(inputs, split)
            for row in bin_rows(bins, split.scores, split.labels, labelled, binary=inputs.binary):
                rows.append({"split": name, "cell": "all", **row})
            if name != OUT_OF_TIME_SPLIT or split.times is None:
                continue
            codes = period_codes(split.times, "month")
            for code in np.unique(codes.cells[codes.valid]):
                mask = codes.valid & (codes.cells == code)
                for row in bin_rows(
                    bins,
                    split.scores[mask],
                    split.labels[mask],
                    labelled[mask],
                    binary=inputs.binary,
                ):
                    rows.append({"split": name, "cell": cell_label(int(code), "month"), **row})
        tables[f"binning_{bins.name}"] = rows
    return tables


def _labelled_mask(inputs: ReportInputs, split: ScoredSplit) -> "np.ndarray":
    """Rows whose outcome is known.

    The after-test split applies the maturity rule; train and test labels are
    complete by construction (the dataset checks guarantee it), so there only
    nulls drop.
    """
    import numpy as np

    labelled = np.asarray(~np.isnan(split.labels))
    if split.name == OUT_OF_TIME_SPLIT and split.times is not None:
        return maturity_mask(split.times, labelled, horizon=inputs.horizon, anchor=inputs.anchor)
    return labelled


# -- distribution ------------------------------------------------------------------------------


def _distribution_tables(
    inputs: ReportInputs, splits: dict[str, ScoredSplit], test: ScoredSplit
) -> dict[str, Table]:
    tables: dict[str, Table] = {
        "score_summary": [
            {"split": name, **score_summary(split.scores)} for name, split in splits.items()
        ]
    }
    bands = fit_bins(FixedWidthBinning(), test.scores, probability=inputs.binary)
    assert bands is not None  # fixed_width always fits
    if inputs.binary:
        tables["score_histogram"] = histogram_rows(
            test.scores, test.labels, _labelled_mask(inputs, test), bins=bands
        )
    tables["test_features"] = _feature_profile(inputs, splits, test)
    return tables


def _feature_profile(
    inputs: ReportInputs, splits: dict[str, ScoredSplit], test: ScoredSplit
) -> Table:
    """What the model saw on the test split, feature by feature, with the
    train-to-test PSI beside it (is the test set even like the train set?)."""
    import numpy as np
    import pyarrow.compute as pc

    from mbt_adapter_base.monitoring import build_baseline, compute_monitor_stats

    columns = [c for c in inputs.feature_columns if c in test.features.column_names]
    train_psi: dict[str, float] = {}
    train = splits.get("train")
    if train is not None and train.n_rows and test.n_rows:
        baseline = build_baseline(
            train.features, columns, train.scores, model_name=inputs.model_name
        )
        stats = compute_monitor_stats(
            baseline, test.features, test.scores, _display_monitors("psi")
        )
        train_psi = {name: stat.value for name, stat in stats.feature_shift.items()}
    rows: Table = []
    for name in columns:
        column = test.features.column(name)
        row: dict[str, Any] = {
            "feature": name,
            "type": str(column.type),
            "n": len(column),
            "null_share": column.null_count / len(column) if len(column) else None,
            "distinct": pc.count_distinct(column).as_py(),
            "psi_vs_train": train_psi.get(name),
        }
        if pa.types.is_integer(column.type) or pa.types.is_floating(column.type):
            values = np.asarray(
                column.drop_null().cast(pa.float64()).to_numpy(zero_copy_only=False)
            )
            if values.size:
                row.update(
                    {
                        "mean": float(values.mean()),
                        "std": float(values.std()),
                        **{
                            f"p{pct}": float(np.percentile(values, pct))
                            for pct in (1, 5, 25, 50, 75, 95, 99)
                        },
                    }
                )
        else:
            counts = pc.value_counts(column.drop_null()).to_pylist()
            top = sorted(counts, key=lambda item: (-item["counts"], str(item["values"])))[:5]
            row["top_values"] = ", ".join(f"{item['values']} ({item['counts']})" for item in top)
        rows.append(row)
    return rows


# -- importance ---------------------------------------------------------------------------------


def _ranked_importance(inputs: ReportInputs) -> list[tuple[str, float]]:
    return sorted(inputs.importance.items(), key=lambda item: (-item[1], item[0]))


def _importance_table(inputs: ReportInputs) -> Table:
    ranked = _ranked_importance(inputs)
    total = sum(value for _, value in ranked) or 1.0
    top_n = inputs.report.importance.top_n
    rows: Table = []
    cumulative = 0.0
    for rank, (name, value) in enumerate(ranked, start=1):
        cumulative += value / total
        rows.append(
            {
                "rank": rank,
                "feature": name,
                "importance": value,
                "share": value / total,
                "cum_share": cumulative,
                "top_n": rank <= top_n,
            }
        )
    return rows


# -- periods -------------------------------------------------------------------------------------


def _requested_grains(inputs: ReportInputs) -> list[ReportPeriod]:
    wanted: set[str] = set(inputs.report.periods) | set(inputs.gate_periods)
    if inputs.stability is not None:
        wanted.add("month")
    return [grain for grain in _GRAINS if grain in wanted]


def _period_cells(
    inputs: ReportInputs,
    test: ScoredSplit,
    after: ScoredSplit,
    grains: list[ReportPeriod],
) -> tuple[list[PeriodCell], Table]:
    """The window cell, then every cell of every active grain."""
    import numpy as np

    assert test.times is not None and after.times is not None
    after_labelled = _labelled_mask(inputs, after)
    test_labelled = ~np.isnan(test.labels)
    cells: list[PeriodCell] = []
    rows: Table = []

    reference = _metrics(inputs, test.labels[test_labelled], test.scores[test_labelled])
    rows.append(
        _period_row("window", "test window", "all", test.n_rows, reference, role="reference")
    )
    bounds = (
        (_bound(inputs.after_window[0]), _bound(inputs.after_window[1]))
        if inputs.after_window is not None
        else (_min_time(after.times), _max_time(after.times))
    )
    window = _cell(
        inputs,
        period="window",
        key="window",
        slot="all",
        bounds=bounds,
        mask=np.ones(after.n_rows, dtype=bool),
        after=after,
        labelled=after_labelled,
        reference_rows=test.n_rows,
        reference_metrics=reference,
    )
    cells.append(window)
    rows.append(_cell_row(window))

    for grain in grains:
        test_codes = period_codes(test.times, grain)
        after_codes = period_codes(after.times, grain)
        slot_reference: dict[int, tuple[int, dict[str, float]]] = {}
        for slot in np.unique(test_codes.slots[test_codes.valid]):
            in_slot = test_codes.valid & (test_codes.slots == slot)
            observed = in_slot & test_labelled
            metrics = _metrics(inputs, test.labels[observed], test.scores[observed])
            slot_reference[int(slot)] = (int(in_slot.sum()), metrics)
            if grain == "month":
                continue  # the one month slot is the whole test window, listed above
            rows.append(
                _period_row(
                    grain,
                    f"test {slot_label(int(slot), grain)}",
                    slot_label(int(slot), grain),
                    int(in_slot.sum()),
                    metrics,
                    role="reference",
                )
            )
        for code in np.unique(after_codes.cells[after_codes.valid]):
            mask = after_codes.valid & (after_codes.cells == code)
            slot = int(after_codes.slots[mask][0])
            key = cell_label(int(code), grain)
            reference_rows, reference_metrics = slot_reference.get(slot, (0, {}))
            cell = _cell(
                inputs,
                period=grain,
                key=key,
                slot=slot_label(slot, grain),
                bounds=cell_bounds(key, grain),
                mask=mask,
                after=after,
                labelled=after_labelled,
                reference_rows=reference_rows,
                reference_metrics=reference_metrics,
            )
            cells.append(cell)
            rows.append(_cell_row(cell))
    return cells, rows


def _bound(iso: str) -> datetime:
    from mbt_adapter_base.reporting import parse_window_bound

    return parse_window_bound(iso)


def _min_time(times: "np.ndarray") -> datetime:
    import numpy as np

    return datetime.fromisoformat(str(np.nanmin(times).astype("datetime64[us]")))


def _max_time(times: "np.ndarray") -> datetime:
    import numpy as np

    return datetime.fromisoformat(str(np.nanmax(times).astype("datetime64[us]")))


def _cell(
    inputs: ReportInputs,
    *,
    period: GatePeriod,
    key: str,
    slot: str,
    bounds: tuple[datetime, datetime],
    mask: "np.ndarray",
    after: ScoredSplit,
    labelled: "np.ndarray",
    reference_rows: int,
    reference_metrics: dict[str, float],
) -> PeriodCell:
    import numpy as np

    n_rows = int(mask.sum())
    observed = mask & labelled
    n_labelled = int(observed.sum())
    mature = n_labelled == n_rows
    metrics = _metrics(inputs, after.labels[observed], after.scores[observed])
    note = ""
    if n_labelled == 0:
        note = "no mature labels yet"
    elif not metrics:
        note = "a single class: ranking metrics are undefined"
    elif not mature:
        note = f"{n_rows - n_labelled} row(s) not mature yet"
    if n_rows and n_labelled < inputs.report.min_rows and not note:
        note = f"fewer than {inputs.report.min_rows} labelled rows"
    if metrics:
        metrics["mean_score"] = float(np.mean(after.scores[mask]))
    return PeriodCell(
        period=period,
        key=key,
        slot=slot,
        start=bounds[0].isoformat(),
        end=bounds[1].isoformat(),
        n_rows=n_rows,
        n_labelled=n_labelled,
        mature=mature,
        metrics=metrics,
        reference_rows=reference_rows,
        reference_metrics=reference_metrics,
        note=note,
    )


def _period_row(
    period: str,
    cell: str,
    slot: str,
    n_rows: int,
    metrics: dict[str, float],
    *,
    role: Literal["reference", "cell"],
) -> dict[str, Any]:
    return {
        "role": role,
        "period": period,
        "cell": cell,
        "slot": slot,
        "n_rows": n_rows,
        **metrics,
    }


def _cell_row(cell: PeriodCell) -> dict[str, Any]:
    row = _period_row(cell.period, cell.key, cell.slot, cell.n_rows, cell.metrics, role="cell")
    row.update(
        {
            "start": cell.start,
            "end": cell.end,
            "n_labelled": cell.n_labelled,
            "mature": cell.mature,
            "reference_rows": cell.reference_rows,
            "note": cell.note,
        }
    )
    for name, value in cell.metrics.items():
        base = cell.reference_metrics.get(name)
        if base is not None:
            row[f"delta_{name}"] = value - base
    return row


# -- stability -----------------------------------------------------------------------------------


def _display_monitors(method: Literal["psi", "ks"]) -> MonitorsSpec:
    """Every feature and the score, one method, no thresholds that matter."""
    return MonitorsSpec(
        feature_shift=FeatureShiftSpec(method=method, threshold=1.0),
        prediction_shift=PredictionShiftSpec(method=method, threshold=1.0),
    )


def _stability(
    inputs: ReportInputs,
    test: ScoredSplit,
    after: ScoredSplit,
    grains: list[ReportPeriod],
) -> tuple[list[StabilityCell], Table, Table]:
    """Scores and features of each after-test cell against the test split.

    The window and every month compare every feature; week and day slots
    compare the most important ones against their own reference slot.
    """
    import numpy as np

    from mbt_adapter_base.monitoring import build_baseline

    assert test.times is not None and after.times is not None
    reference = build_baseline(
        test.features, inputs.feature_columns, test.scores, model_name=inputs.model_name
    )
    cells: list[StabilityCell] = []
    score_rows: Table = []
    feature_rows: Table = []

    def judge(period: GatePeriod, key: str, slot: str, mask: "np.ndarray", baseline: Any) -> None:
        frame = after.features.filter(pa.array(mask))
        scores = after.scores[mask]
        psi = _stats(baseline, frame, scores, _display_monitors("psi"))
        ks = _stats(baseline, frame, scores, _display_monitors("ks"))
        gate = None
        stability = inputs.stability
        if stability is not None and stability.period == period:
            gate = _stats(baseline, frame, scores, stability.monitors)
        cell = _stability_cell(period, key, slot, int(mask.sum()), baseline.score.n, psi, ks, gate)
        cells.append(cell)
        score_rows.append(
            {
                "period": period,
                "cell": key,
                "slot": slot,
                "n_rows": cell.n_rows,
                "reference_rows": cell.reference_rows,
                "score_psi": cell.score_psi,
                "score_ks": cell.score_ks,
                "max_feature_psi": cell.max_feature_psi,
                "max_feature": cell.max_feature,
                "features_psi_over_warn": cell.features_over_warn,
                "features_psi_over_fail": cell.features_over_fail,
            }
        )
        for name, stat in sorted(psi.feature_shift.items()):
            ks_stat = ks.feature_shift.get(name)
            feature_rows.append(
                {
                    "period": period,
                    "cell": key,
                    "feature": name,
                    "psi": stat.value,
                    "ks": ks_stat.value if ks_stat is not None else None,
                    "n_rows": stat.n_current,
                    "reference_rows": stat.n_baseline,
                }
            )

    judge("window", "window", "all", np.ones(after.n_rows, dtype=bool), reference)
    month_codes = period_codes(after.times, "month")
    for code in np.unique(month_codes.cells[month_codes.valid]):
        mask = month_codes.valid & (month_codes.cells == code)
        judge("month", cell_label(int(code), "month"), "all", mask, reference)

    top = [name for name, _ in _ranked_importance(inputs) if name in inputs.feature_columns][
        : inputs.report.stability.feature_top_n
    ]
    for grain in grains:
        if grain == "month":
            continue
        test_codes = period_codes(test.times, grain)
        after_codes = period_codes(after.times, grain)
        slot_baselines: dict[int, Any] = {}
        for code in np.unique(after_codes.cells[after_codes.valid]):
            mask = after_codes.valid & (after_codes.cells == code)
            slot = int(after_codes.slots[mask][0])
            if slot not in slot_baselines:
                in_slot = test_codes.valid & (test_codes.slots == slot)
                if not in_slot.any():
                    continue  # no reference rows in this slot: nothing to compare
                slot_baselines[slot] = build_baseline(
                    test.features.filter(pa.array(in_slot)),
                    top,
                    test.scores[in_slot],
                    model_name=inputs.model_name,
                )
            judge(
                grain,
                cell_label(int(code), grain),
                slot_label(slot, grain),
                mask,
                slot_baselines[slot],
            )
    return cells, score_rows, feature_rows


def _stats(
    baseline: Any, frame: pa.Table, scores: "np.ndarray", monitors: MonitorsSpec
) -> MonitorStats:
    from mbt_adapter_base.monitoring import compute_monitor_stats

    return compute_monitor_stats(baseline, frame, scores, monitors)


def _stability_cell(
    period: GatePeriod,
    key: str,
    slot: str,
    n_rows: int,
    reference_rows: int,
    psi: MonitorStats,
    ks: MonitorStats,
    gate: MonitorStats | None,
) -> StabilityCell:
    features = psi.feature_shift
    worst = max(features.items(), key=lambda item: (item[1].value, item[0]), default=None)
    return StabilityCell(
        period=period,
        key=key,
        slot=slot,
        n_rows=n_rows,
        reference_rows=reference_rows,
        score_psi=psi.prediction_shift.value if psi.prediction_shift else None,
        score_ks=ks.prediction_shift.value if ks.prediction_shift else None,
        max_feature_psi=worst[1].value if worst else None,
        max_feature=worst[0] if worst else None,
        features_over_warn=sum(1 for stat in features.values() if stat.value > PSI_WARN_BAND),
        features_over_fail=sum(1 for stat in features.values() if stat.value > PSI_FAIL_BAND),
        gate=gate,
    )
