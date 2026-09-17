"""Write a computed training report to a directory (ADR-30).

Layout, relative to the report directory::

    report.html                              the page a reviewer opens
    summary.json                             what core gated on
    evaluation/metrics_by_split.csv
    evaluation/feature_importance.csv
    evaluation/binning/<strategy>.csv        every split, and every after-test month
    evaluation/distribution/score_summary.csv
    evaluation/distribution/score_histogram.csv   (classifiers)
    evaluation/distribution/test_features.csv
    performance/by_period.csv                window, month, week and day cells
    stability/scores_by_period.csv
    stability/features_by_period.csv
    predictions/<split>.parquet              only when the project opted in
"""

import csv
import json
import shutil
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Literal

import pyarrow.parquet as pq

from mbt.reporting.builder import ReportData, Table
from mbt.reporting.render import (
    Column,
    LineChart,
    Series,
    column_chart,
    esc,
    fmt_number,
    fmt_share,
    fmt_text,
    hbar_chart,
    line_chart,
    render_page,
    table,
    tiles,
)
from mbt_adapter_base.reporting import PSI_FAIL_BAND, PSI_WARN_BAND
from mbt_adapter_base.types import OUT_OF_TIME_SPLIT

#: Table name -> file, relative to the report directory.
_TABLE_FILES = {
    "split_metrics": "evaluation/metrics_by_split.csv",
    "feature_importance": "evaluation/feature_importance.csv",
    "score_summary": "evaluation/distribution/score_summary.csv",
    "score_histogram": "evaluation/distribution/score_histogram.csv",
    "test_features": "evaluation/distribution/test_features.csv",
    "performance_by_period": "performance/by_period.csv",
    "stability_scores": "stability/scores_by_period.csv",
    "stability_features": "stability/features_by_period.csv",
}

#: Splits in their fixed chart order and colour slot.
_SPLIT_SLOTS = {"train": 0, "test": 1, "out_of_time": 2}
_SPLIT_NAMES = {"train": "train (in-sample)", "test": "test", "out_of_time": "after test"}


@dataclass
class ReportMeta:
    """What the page says about where the numbers came from."""

    model: str
    run: str
    anchor: str
    binary: bool
    kind: Literal["training", "pre-deploy check"] = "training"
    dataset: str | None = None
    windows: dict[str, list[str]] = field(default_factory=dict)
    row_counts: dict[str, int] = field(default_factory=dict)

    @property
    def primary(self) -> str:
        return "roc_auc" if self.binary else "rmse"


def write_report(data: ReportData, meta: ReportMeta, out_dir: Path) -> list[str]:
    """Write every file; returns their paths relative to ``out_dir``, sorted."""
    out_dir.mkdir(parents=True, exist_ok=True)
    written: list[str] = []
    engine_files = {
        "engine_drift": f"stability/{data.engine}/drift_by_column.csv",
        "engine_drift_summary": f"stability/{data.engine}/drift_by_period.csv",
    }
    for name, rows in data.tables.items():
        relative = (
            _TABLE_FILES.get(name)
            or engine_files.get(name)
            or f"evaluation/binning/{name.removeprefix('binning_')}.csv"
        )
        if rows:
            _write_csv(out_dir / relative, rows)
            written.append(relative)
    for relative, source in data.files.items():
        (out_dir / relative).parent.mkdir(parents=True, exist_ok=True)
        shutil.copyfile(source, out_dir / relative)
        written.append(relative)
    for split, predictions in data.predictions.items():
        relative = f"predictions/{split}.parquet"
        (out_dir / relative).parent.mkdir(parents=True, exist_ok=True)
        pq.write_table(predictions, out_dir / relative)
        written.append(relative)
    written.extend(["report.html", "summary.json"])
    data.summary.documents = sorted(written)
    (out_dir / "summary.json").write_text(
        json.dumps(data.summary.model_dump(mode="json"), indent=1, sort_keys=True) + "\n"
    )
    (out_dir / "report.html").write_text(_page(data, meta), encoding="utf-8")
    return data.summary.documents


def _write_csv(path: Path, rows: Table) -> None:
    columns: dict[str, None] = {}
    for row in rows:
        columns.update(dict.fromkeys(row))
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=list(columns), extrasaction="ignore")
        writer.writeheader()
        for row in rows:
            writer.writerow({k: ("" if v is None else v) for k, v in row.items()})


# -- the page -----------------------------------------------------------------------------------


def _page(data: ReportData, meta: ReportMeta) -> str:
    title = f"{'Pre-deploy check' if meta.kind != 'training' else 'Training report'}: {meta.model}"
    sections = [
        f"<h1>{esc(title)}</h1>",
        f"<p class='muted'>run <code>{esc(meta.run)}</code> &middot; anchor "
        f"<code>{esc(meta.anchor)}</code>"
        + (f" &middot; dataset <code>{esc(meta.dataset)}</code>" if meta.dataset else "")
        + "</p>",
        _windows(meta),
        _warnings(data),
        _summary_tiles(data, meta),
        "<h2>Metrics by split</h2>",
        _split_metrics(data, meta),
        _importance(data),
        _distribution(data, meta),
        _binning(data, meta),
        _after_test(data, meta),
        _files(data),
    ]
    return render_page(title, "".join(part for part in sections if part))


def _windows(meta: ReportMeta) -> str:
    if not meta.windows:
        return ""
    rows = [
        {
            "split": _SPLIT_NAMES.get(split, split),
            "start": bounds[0],
            "end": bounds[1],
            "rows": meta.row_counts.get(split),
        }
        for split, bounds in sorted(meta.windows.items(), key=lambda kv: kv[1][0])
    ]
    return table(
        rows,
        [
            Column("split", "split", str, numeric=False),
            Column("start", "from", str, numeric=False),
            Column("end", "to (exclusive)", str, numeric=False),
            Column("rows", "rows"),
        ],
    )


def _warnings(data: ReportData) -> str:
    if not data.summary.warnings:
        return ""
    items = "".join(f"<li>{esc(message)}</li>" for message in data.summary.warnings)
    return f"<ul class='warnings muted'>{items}</ul>"


def _window_cell(data: ReportData) -> Any:
    return next((c for c in data.summary.periods if c.period == "window"), None)


def _window_stability(data: ReportData) -> Any:
    return next((c for c in data.summary.stability if c.period == "window"), None)


def _summary_tiles(data: ReportData, meta: ReportMeta) -> str:
    metrics = data.summary.split_metrics
    primary = meta.primary
    # train, test, after test: the order every table on the page uses
    items = []
    if "train" in metrics:
        items.append((f"train {primary}", fmt_number(metrics["train"].get(primary)), "in-sample"))
    items.append(
        (
            f"test {primary}",
            fmt_number(metrics.get("test", {}).get(primary)),
            f"{data.summary.reference_rows:,} rows",
        )
    )
    window = _window_cell(data)
    if window is not None:
        items.append(
            (
                f"after-test {primary}",
                fmt_number(window.metrics.get(primary)),
                f"{window.n_labelled:,} mature of {window.n_rows:,} rows",
            )
        )
    stability = _window_stability(data)
    if stability is not None:
        items.append(
            (
                "after-test score PSI",
                fmt_number(stability.score_psi),
                f"vs test; {PSI_WARN_BAND} watch, {PSI_FAIL_BAND} shifted",
            )
        )
    return tiles(items)


#: Numbers the report adds beside the declared metrics; they go last.
_REPORT_METRICS = ("ks", "label_rate", "label_mean", "mean_score")


def _metric_columns(meta: ReportMeta, rows: Table, first: list[Column]) -> list[Column]:
    keys: dict[str, None] = {}
    for row in rows:
        keys.update(dict.fromkeys(row))
    skip = {c.key for c in first}
    declared = sorted(k for k in keys if k != meta.primary and k not in _REPORT_METRICS)
    ordered = [meta.primary, *declared, *_REPORT_METRICS]
    return [*first, *(Column(k, k) for k in ordered if k in keys and k not in skip)]


def _split_metrics(data: ReportData, meta: ReportMeta) -> str:
    rows = [
        {
            "split": _SPLIT_NAMES.get(r["split"], r["split"]),
            **{k: v for k, v in r.items() if k != "split"},
        }
        for r in data.tables.get("split_metrics", [])
    ]
    columns = _metric_columns(meta, rows, [Column("split", "split", str, numeric=False)])
    return table(rows, columns)


def _importance(data: ReportData) -> str:
    rows = data.tables.get("feature_importance", [])
    if not rows:
        return (
            "<h2>Feature importance</h2><p class='muted'>the adapter reports no "
            "importance for this model</p>"
        )
    top = [r for r in rows if r["top_n"]]
    chart = hbar_chart(
        f"Share of total importance, top {len(top)} features",
        [(str(r["feature"]), float(r["share"])) for r in top],
        fmt_share,
    )
    columns = [
        Column("rank", "rank"),
        Column("feature", "feature", str, numeric=False),
        Column("importance", "importance"),
        Column("share", "share", fmt_share),
        Column("cum_share", "cumulative", fmt_share),
    ]
    return (
        "<h2>Feature importance</h2>"
        + chart
        + f"<details><summary>all {len(rows)} features</summary>{table(rows, columns)}</details>"
    )


def _distribution(data: ReportData, meta: ReportMeta) -> str:
    parts = ["<h2>Score distribution</h2>"]
    histogram = data.tables.get("score_histogram", [])
    if histogram:
        categories = [f"{r['lower']:g}-{r['upper']:g}" for r in histogram]
        parts.append(
            line_chart(
                LineChart(
                    caption="Test split: share of each observed class per 0.05 score band",
                    categories=categories,
                    series=[
                        Series("negatives", [r["share_negative"] for r in histogram], slot=0),
                        Series("positives", [r["share_positive"] for r in histogram], slot=1),
                    ],
                    fmt=fmt_share,
                    zero_based=True,
                    ceiling=1.0,
                )
            )
        )
    summary_rows = [
        {**r, "split": _SPLIT_NAMES.get(r["split"], r["split"])}
        for r in data.tables.get("score_summary", [])
    ]
    parts.append(
        table(
            summary_rows,
            [
                Column("split", "split", str, numeric=False),
                Column("n", "rows"),
                Column("mean", "mean"),
                Column("std", "std"),
                *(Column(f"p{p}", f"p{p}") for p in (1, 5, 25, 50, 75, 95, 99)),
            ],
        )
    )
    features = data.tables.get("test_features", [])
    if features:
        # A pre-deploy check reads no train split, so there is nothing to shift from.
        against_train = any(r.get("psi_vs_train") is not None for r in features)
        columns = [
            Column("feature", "feature", str, numeric=False),
            Column("type", "type", str, numeric=False),
            Column("null_share", "null", fmt_share),
            Column("distinct", "distinct"),
            *([Column("psi_vs_train", "PSI vs train")] if against_train else []),
            Column("mean", "mean"),
            Column("p5", "p5"),
            Column("p50", "p50"),
            Column("p95", "p95"),
            Column("top_values", "top values", str, numeric=False),
        ]
        title = "test split, feature by feature" + (
            " (with PSI against train)" if against_train else ""
        )
        parts.append(f"<details><summary>{title}</summary>{table(features, columns)}</details>")
    return "".join(parts)


def _strategy_title(name: str) -> str:
    """``quantile_10`` -> ``Quantile bins (10)``, and so on."""
    if name.startswith("quantile_"):
        return f"Quantile bins ({name.removeprefix('quantile_')})"
    if name.startswith("fixed_width_"):
        return f"Fixed-width bands ({name.removeprefix('fixed_width_')})"
    return {
        "fixed_width": "Fixed-width bands",
        "custom": "Custom edges",
        "top_percent": "Top-percent cutoffs",
    }.get(name, name)


def _occupied(rows: Table) -> set[int]:
    """Bins holding rows in some split: the band tables trim empty ends."""
    return {int(r["bin"]) for r in rows if "bin" in r and r["n"]}


def _trim(rows: Table, occupied: set[int]) -> Table:
    if not occupied or not rows or "bin" not in rows[0]:
        return rows
    low, high = min(occupied), max(occupied)
    return [r for r in rows if low <= int(r["bin"]) <= high]


def _bin_label(row: dict[str, Any]) -> str:
    if "top_percent" in row:
        return f"top {row['top_percent']:g}%"
    # two significant digits keep ten decile labels apart on the axis; the
    # table beside the chart carries the edges in full
    return f"{row['lower']:.2g}-{row['upper']:.2g}"


def _binning(data: ReportData, meta: ReportMeta) -> str:
    rate = "label_rate" if meta.binary else "label_mean"
    parts = ["<h2>Binning</h2>"]
    for bins in data.bins:
        rows = data.tables[f"binning_{bins.name}"]
        # Bands nobody's score reached (a probability model that never
        # scores above 0.6 has eight empty 0.05 bands) stay in the CSV only.
        rows = _trim(rows, _occupied(rows))
        whole = [r for r in rows if r["cell"] == "all"]
        # never empty: data adapters refuse an empty test split
        test_rows = [r for r in whole if r["split"] == "test"]
        key = "precision" if bins.strategy == "top_percent" and meta.binary else rate
        categories = [_bin_label(r) for r in test_rows]
        series = []
        for split in ("train", "test", "out_of_time"):
            split_rows = [r for r in whole if r["split"] == split]
            if split_rows:
                # A rate over a handful of rows is noise that reads as signal
                # on a line; the tables below still show every bin.
                values = [
                    r.get(key) if (r.get("n_labelled") or 0) >= data.min_rows else None
                    for r in split_rows
                ]
                series.append(Series(_SPLIT_NAMES[split], values, slot=_SPLIT_SLOTS[split]))
        noun = "positive rate" if meta.binary else "mean label"
        what = "precision" if key == "precision" else noun
        parts.append(f"<h3>{esc(_strategy_title(bins.name))}</h3>")
        parts.append(
            line_chart(
                LineChart(
                    caption=f"{what.capitalize()} by bin, edges fitted on test "
                    f"(highest scores first; bins under {data.min_rows} labelled rows "
                    "are left off the chart)",
                    categories=categories,
                    series=series,
                    fmt=fmt_share if meta.binary else fmt_number,
                    zero_based=meta.binary,
                    ceiling=1.0 if meta.binary else None,
                )
            )
        )
        columns = _bin_columns(meta, bins.strategy)
        parts.append(table(test_rows, columns))
        others = [r for r in rows if not (r["split"] == "test" and r["cell"] == "all")]
        if others:
            labelled = [{**r, "split": _SPLIT_NAMES.get(r["split"], r["split"])} for r in others]
            parts.append(
                f"<details><summary>{esc(_other_bins_title(others))}</summary>"
                + table(
                    labelled,
                    [
                        Column("split", "split", str, numeric=False),
                        Column("cell", "period", str, numeric=False),
                        *columns,
                    ],
                )
                + "</details>"
            )
    return "".join(parts) if len(parts) > 1 else ""


def _other_bins_title(rows: Table) -> str:
    """Name what the collapsed bin table holds, e.g. no train on a check."""
    splits = {r["split"] for r in rows}
    names = ["train"] if "train" in splits else []
    if OUT_OF_TIME_SPLIT in splits:
        names.append("the after-test window")
        if any(r["cell"] != "all" for r in rows):
            names.append("each after-test month")
    return ", ".join(names[:-1]) + (" and " if len(names) > 1 else "") + names[-1]


def _bin_columns(meta: ReportMeta, strategy: str) -> list[Column]:
    if strategy == "top_percent":
        return [
            Column("top_percent", "top %", lambda v: f"{v:g}%"),
            Column("threshold", "score >="),
            Column("n", "rows"),
            Column("share", "share", fmt_share),
            Column("precision", "precision", fmt_share),
            Column("label_mean", "mean label"),
            Column("capture", "capture", fmt_share),
            Column("lift", "lift", lambda v: fmt_number(v, 2)),
        ]
    return [
        Column("bin", "bin"),
        Column("lower", "from"),
        Column("upper", "to"),
        Column("n", "rows"),
        Column("share", "share", fmt_share),
        Column("mean_score", "mean score"),
        Column("label_rate", "positive rate", fmt_share),
        Column("label_mean", "mean label"),
        Column("lift", "lift", lambda v: fmt_number(v, 2)),
        Column("capture", "cum. capture", fmt_share),
        Column("cum_lift", "cum. lift", lambda v: fmt_number(v, 2)),
        Column("ks", "KS"),
    ]


def _after_test(data: ReportData, meta: ReportMeta) -> str:
    summary = data.summary
    if not summary.periods:
        return ""
    primary = meta.primary
    parts = ["<h2>After the test window</h2>"]
    window = _window_cell(data)
    months = [c for c in summary.periods if c.period == "month"]
    if months:
        reference = window.reference_metrics.get(primary) if window is not None else None
        parts.append(
            line_chart(
                LineChart(
                    caption=f"{primary} by month after the test window "
                    "(mature, labelled rows only)",
                    categories=[c.key for c in months],
                    series=[Series(primary, [c.metrics.get(primary) for c in months], slot=0)],
                    references=[("test", reference)] if reference is not None else [],
                    # applies only when every value is at most 1, so an AUC
                    # of 0.99 gets no 1.005 tick and an RMSE of 3 is untouched
                    ceiling=1.0,
                )
            )
        )
    rows = data.tables.get("performance_by_period", [])
    columns = [
        Column("cell", "period", str, numeric=False),
        Column("slot", "slot", str, numeric=False),
        Column("n_rows", "rows"),
        Column("n_labelled", "mature"),
        Column(primary, primary),
        Column(f"delta_{primary}", "vs test"),
        Column("ks", "KS"),
        Column(
            "label_rate" if meta.binary else "label_mean",
            "label rate" if meta.binary else "mean label",
            fmt_share if meta.binary else fmt_number,
        ),
        Column("mean_score", "mean score"),
        Column("reference_rows", "test rows"),
        Column("note", "note", fmt_text, numeric=False),
    ]
    by_period: dict[str, Table] = {}
    for row in rows:
        by_period.setdefault(str(row["period"]), []).append(row)
    parts.append(table(by_period.get("window", []) + by_period.get("month", []), columns))
    for period, label in (("week_of_month", "Week of month"), ("day_of_month", "Day of month")):
        if period in by_period:
            parts.append(
                f"<details><summary>{label}: each slot against the same slot of the test "
                f"window</summary>{table(by_period[period], columns)}</details>"
            )
    for period, reason in summary.skipped_periods.items():
        parts.append(f"<p class='muted'>{esc(period)} not shown: {esc(reason)}</p>")
    parts.append(_stability(data))
    parts.append(_engine_drift(data))
    return "".join(parts)


def _stability(data: ReportData) -> str:
    scores = data.tables["stability_scores"]  # set with the period cells, window row first
    months = [r for r in scores if r["period"] == "month"]
    parts = ["<h3>Stability against the test split</h3>"]
    if months:
        parts.append(
            column_chart(
                "Score PSI by month",
                [str(r["cell"]) for r in months],
                [r["score_psi"] for r in months],
                fmt_number,
                references=[("watch", PSI_WARN_BAND), ("shifted", PSI_FAIL_BAND)],
            )
        )
    columns = [
        Column("cell", "period", str, numeric=False),
        Column("slot", "slot", str, numeric=False),
        Column("n_rows", "rows"),
        Column("reference_rows", "test rows"),
        Column("score_psi", "score PSI"),
        Column("score_ks", "score KS"),
        Column("max_feature_psi", "worst feature PSI"),
        Column("max_feature", "worst feature", str, numeric=False),
        Column("features_psi_over_warn", f"features > {PSI_WARN_BAND}"),
        Column("features_psi_over_fail", f"features > {PSI_FAIL_BAND}"),
    ]
    whole = [r for r in scores if r["period"] in ("window", "month")]
    parts.append(table(whole, columns))
    slots = [r for r in scores if r["period"] not in ("window", "month")]
    if slots:
        parts.append(
            "<details><summary>week and day slots (top features only)</summary>"
            + table(slots, columns)
            + "</details>"
        )
    features = [r for r in data.tables.get("stability_features", []) if r["period"] == "window"]
    if features:
        worst = sorted(features, key=lambda r: -(r["psi"] or 0.0))[:20]
        parts.append(
            "<details><summary>most shifted features over the whole window</summary>"
            + table(
                worst,
                [
                    Column("feature", "feature", str, numeric=False),
                    Column("psi", "PSI"),
                    Column("ks", "KS"),
                    Column("n_rows", "rows"),
                ],
            )
            + "</details>"
        )
    return "".join(parts)


def _engine_drift(data: ReportData) -> str:
    """The report engine's own drift verdicts, shown beside mbt's (ADR-30)."""
    cells = data.tables.get("engine_drift_summary", [])
    if not cells:
        return ""
    engine = esc((data.engine or "").capitalize())
    parts = [
        f"<h3>{engine} drift report</h3>",
        f"<p class='muted'>{engine}'s own tests, for reading beside the tables above; "
        "the gates use mbt's PSI and KS. Each cell's full report is a separate "
        "file on the run.</p>",
        table(
            cells,
            [
                Column("cell", "period", str, numeric=False),
                Column("n_rows", "rows"),
                Column("reference_rows", "test rows"),
                Column("drifted_share", "columns drifted", fmt_share),
                Column("file", "report", str, numeric=False),
            ],
        ),
    ]
    columns = data.tables.get("engine_drift", [])
    if columns:
        parts.append(
            "<details><summary>column by column</summary>"
            + table(
                columns,
                [
                    Column("cell", "period", str, numeric=False),
                    Column("column", "column", str, numeric=False),
                    Column("method", "test", str, numeric=False),
                    Column("score", "score"),
                    Column("threshold", "threshold"),
                    Column("drifted", "drifted"),
                ],
            )
            + "</details>"
        )
    return "".join(parts)


def _files(data: ReportData) -> str:
    items = "".join(f"<li><code>{esc(name)}</code></li>" for name in data.summary.documents)
    return f"<h2>Files</h2><ul>{items}</ul>"
