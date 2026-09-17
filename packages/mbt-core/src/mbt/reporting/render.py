"""The training report page (ADR-30): one self-contained HTML file.

No script, no CDN, no web font: the page is read inside a tracking UI's
sandboxed iframe as often as on its own. Charts are inline SVG styled by role
variables, so light and dark are two token sets over one markup. Every chart
has a table twin carrying the same numbers, and every mark a native tooltip -
hover enhances, it never gates.
"""

import html
import math
from collections.abc import Callable, Sequence
from dataclasses import dataclass, field
from itertools import pairwise
from typing import Any

from mbt.docsgen.html import page

#: Chart roles. Series slots follow a fixed order (identity follows the
#: split, never its rank), validated for both surfaces with the dataviz
#: palette checker; the light aqua is below 3:1, so series always carry a
#: direct label or legend and a table.
_CHART_CSS = """
:root { --surface:#ffffff; --ink:#1f2430; --ink-2:#52514e; --ink-muted:#6b7280;
        --grid:#e8e7e1; --axis:#c3c2b7; --series-1:#2a78d6; --series-2:#eb6834;
        --series-3:#1baf7a; --good:#0ca30c; --warning:#b77900; --critical:#d03b3b; }
/* Same switch as the base palette: the page follows the reader's OS. */
@media (prefers-color-scheme: dark) {
  :root { --surface:#0f1419; --ink:#e6e8eb; --ink-2:#c3c2b7; --ink-muted:#9aa4b2;
          --grid:#242c38; --axis:#3a4657; --series-1:#3987e5; --series-2:#d95926;
          --series-3:#199e70; --good:#5dc98a; --warning:#e0af68; --critical:#f07178; }
}
h3 { font-size: 1rem; margin: 1.5rem 0 .4rem; }
.tiles { display: grid; grid-template-columns: repeat(auto-fit, minmax(170px, 1fr));
         gap: .75rem; margin: 1rem 0; }
.tile { border: 1px solid var(--line); border-radius: 8px; padding: .7rem .9rem; min-width: 0; }
.tile .label { color: var(--muted); font-size: .82rem; }
.tile .value { font-size: 1.6rem; font-weight: 600; line-height: 1.2; }
.tile .sub { color: var(--muted); font-size: .8rem; }
figure { margin: .75rem 0 1rem; }
figcaption { color: var(--muted); font-size: .85rem; margin-bottom: .35rem; }
.legend { display: flex; flex-wrap: wrap; gap: .25rem 1rem; font-size: .82rem;
          color: var(--ink-2); margin: 0 0 .25rem; }
.legend .key { display: inline-block; width: 14px; height: 2px; vertical-align: middle;
               margin-right: .35rem; border-radius: 1px; }
/* Drawn at the width they are shown at, so chart text matches the page's. */
.chart { width: 100%; max-width: 960px; height: auto; overflow: visible; }
.chart text { font: 12px -apple-system, "Segoe UI", Roboto, sans-serif; fill: var(--ink-muted); }
.chart .label { fill: var(--ink-2); }
.chart .grid { stroke: var(--grid); stroke-width: 1; }
.chart .axis { stroke: var(--axis); stroke-width: 1; }
.chart .reference { stroke: var(--ink-muted); stroke-width: 1; }
.chart .line { fill: none; stroke-width: 2; stroke-linejoin: round; stroke-linecap: round; }
.chart .dot { stroke: var(--surface); stroke-width: 2; }
.chart .hit { fill: transparent; }
.chart .bar:hover, .chart .hit:hover + .dot { opacity: .8; }
.scroll { overflow-x: auto; max-width: 100%; }
.scroll table { min-width: 100%; width: max-content; }
td.num, th.num { text-align: right; font-variant-numeric: tabular-nums; }
details { margin: .4rem 0; }
summary { cursor: pointer; color: var(--accent); }
ul.warnings { padding-left: 1.2rem; }
"""

#: Vertical room one direct label needs, in chart units.
_LABEL_GAP = 14.0

#: Chart width in SVG units; the CSS caps the rendered width at the same
#: number, so one unit is one pixel and 12px text stays 12px.
_WIDTH = 960

#: Advance widths of 12px system sans-serif, rounded up: digits are
#: tabular, punctuation narrow, and anything else is taken as a wide letter.
_NARROW = {".": 3.5, ",": 3.5, ":": 3.5, " ": 3.5, "-": 4.5}
_DIGIT_WIDTH = 6.8
_OTHER_WIDTH = 7.5
#: Clear space kept between two axis labels.
_LABEL_SPACE = 12.0

#: The fixed series order: train, test, after-test.
SERIES = ("var(--series-1)", "var(--series-2)", "var(--series-3)")

Number = float | int | None


# -- formatting -----------------------------------------------------------------------------


def fmt_number(value: Any, digits: int = 4) -> str:
    if value is None or (isinstance(value, float) and math.isnan(value)):
        return "-"
    if isinstance(value, bool):
        return "yes" if value else "no"
    if isinstance(value, int):
        return f"{value:,}"
    if isinstance(value, float):
        return f"{value:,.{digits}f}"
    return str(value)


def fmt_share(value: Any) -> str:
    if value is None or (isinstance(value, float) and math.isnan(value)):
        return "-"
    return f"{float(value):.1%}"


def fmt_text(value: Any) -> str:
    return "" if value is None else str(value)


def esc(value: Any) -> str:
    return html.escape(str(value), quote=True)


# -- tables --------------------------------------------------------------------------------


@dataclass
class Column:
    key: str
    title: str
    fmt: Callable[[Any], str] = fmt_number
    numeric: bool = True


def table(rows: Sequence[dict[str, Any]], columns: Sequence[Column]) -> str:
    """A scroll-safe table; columns absent from every row are dropped."""
    present = [c for c in columns if any(c.key in row for row in rows)]
    if not rows or not present:
        return "<p class='muted'>nothing to show</p>"
    head = "".join(f"<th class='{'num' if c.numeric else ''}'>{esc(c.title)}</th>" for c in present)
    body = "".join(
        "<tr>"
        + "".join(
            f"<td class='{'num' if c.numeric else ''}'>{esc(c.fmt(row.get(c.key)))}</td>"
            for c in present
        )
        + "</tr>"
        for row in rows
    )
    return f"<div class='scroll'><table><tr>{head}</tr>{body}</table></div>"


def tiles(items: Sequence[tuple[str, str, str]]) -> str:
    """Stat tiles: (label, value, sub-line)."""
    cells = "".join(
        f"<div class='tile'><div class='label'>{esc(label)}</div>"
        f"<div class='value'>{esc(value)}</div><div class='sub'>{esc(sub)}</div></div>"
        for label, value, sub in items
    )
    return f"<div class='tiles'>{cells}</div>"


# -- charts -----------------------------------------------------------------------------------


def _nice_ticks(low: float, high: float, count: int = 4) -> list[float]:
    if high <= low:
        high = low + 1.0
    raw = (high - low) / count
    magnitude = 10 ** math.floor(math.log10(raw))
    step = next(m * magnitude for m in (1, 2, 2.5, 5, 10) if m * magnitude >= raw)
    start = math.floor(low / step) * step
    ticks = []
    value = start
    while value <= high + step * 1e-9:
        ticks.append(round(value, 10))
        value += step
    if ticks[-1] < high:
        ticks.append(round(ticks[-1] + step, 10))
    return ticks


def _legend(entries: Sequence[tuple[str, int]]) -> str:
    """Line keys for two or more series; one series is named by its caption."""
    if len(entries) < 2:
        return ""
    keys = "".join(
        f"<span><span class='key' style='background:{SERIES[slot % len(SERIES)]}'></span>"
        f"{esc(name)}</span>"
        for name, slot in entries
    )
    return f"<div class='legend'>{keys}</div>"


def _figure(caption: str, svg: str, legend: str = "") -> str:
    return f"<figure role='group'><figcaption>{esc(caption)}</figcaption>{legend}{svg}</figure>"


@dataclass
class Series:
    name: str
    values: list[Number]
    slot: int = 0


@dataclass
class LineChart:
    caption: str
    categories: list[str]
    series: list[Series]
    fmt: Callable[[Any], str] = fmt_number
    references: list[tuple[str, float]] = field(default_factory=list)
    zero_based: bool = False
    #: The largest value the measure can take (1.0 for a rate), so padding
    #: never draws an axis past it. A power of ten, so a nice tick lands on it.
    ceiling: float | None = None


def _text_width(text: str) -> float:
    return sum(_NARROW.get(char, _DIGIT_WIDTH if char.isdigit() else _OTHER_WIDTH) for char in text)


def _label_stride(labels: Sequence[str], slot: float) -> int:
    """Label every n-th category, so no two axis labels touch."""
    widest = max((_text_width(label) for label in labels), default=0.0) + _LABEL_SPACE
    return max(1, math.ceil(widest / slot))


def line_chart(chart: LineChart) -> str:
    """One shared x axis of categories, one y axis, up to three series."""
    values = [v for s in chart.series for v in s.values if v is not None]
    values += [value for _, value in chart.references]
    if not values or not chart.categories:
        return ""
    width, height = _WIDTH, 260
    left, right, top, bottom = 60, 120, 12, 36
    low = 0.0 if chart.zero_based else min(values)
    high = max(values)
    pad = (high - low) * 0.08 or 0.05
    top_value = high + pad
    if chart.ceiling is not None and high <= chart.ceiling:
        top_value = min(top_value, chart.ceiling)
    ticks = _nice_ticks(low if chart.zero_based else low - pad, top_value)
    y0, y1 = ticks[0], ticks[-1]
    plot_w, plot_h = width - left - right, height - top - bottom
    n = len(chart.categories)

    def x_at(i: int) -> float:
        return left + (plot_w * (i + 0.5) / n)

    def y_at(v: float) -> float:
        return top + plot_h * (1 - (v - y0) / (y1 - y0))

    parts = [
        f"<svg class='chart' viewBox='0 0 {width} {height}' role='img' "
        f"aria-label='{esc(chart.caption)}' xmlns='http://www.w3.org/2000/svg'>"
    ]
    for tick in ticks:
        y = y_at(tick)
        parts.append(
            f"<line class='grid' x1='{left}' x2='{left + plot_w}' y1='{y:.1f}' y2='{y:.1f}'/>"
        )
        parts.append(
            f"<text x='{left - 6}' y='{y + 4:.1f}' text-anchor='end'>{esc(chart.fmt(tick))}</text>"
        )
    parts.append(
        f"<line class='axis' x1='{left}' x2='{left + plot_w}' "
        f"y1='{top + plot_h}' y2='{top + plot_h}'/>"
    )
    stride = _label_stride(chart.categories, plot_w / n)
    for i, label in enumerate(chart.categories):
        if i % stride:
            continue
        parts.append(
            f"<text x='{x_at(i):.1f}' y='{top + plot_h + 16}' "
            f"text-anchor='middle'>{esc(label)}</text>"
        )
    label_rows: list[float] = []
    for label, value in chart.references:
        y = y_at(value)
        label_rows.append(y)
        parts.append(
            f"<line class='reference' x1='{left}' x2='{left + plot_w}' y1='{y:.1f}' y2='{y:.1f}'/>"
        )
        parts.append(
            f"<text class='label' x='{left + plot_w + 6}' y='{y + 4:.1f}'>"
            f"{esc(label)} {esc(chart.fmt(value))}</text>"
        )
    end_labels: list[str] = []
    for series in chart.series:
        color = SERIES[series.slot % len(SERIES)]
        points = [(x_at(i), y_at(v), v, i) for i, v in enumerate(series.values) if v is not None]
        # A missing value breaks the line: joining across it would draw a
        # trend through bins or periods that have no data.
        runs: list[list[tuple[float, float]]] = []
        previous = -2
        for x, y, _, i in points:
            if i != previous + 1:
                runs.append([])
            runs[-1].append((x, y))
            previous = i
        for run in runs:
            if len(run) > 1:
                path = " ".join(f"{x:.1f},{y:.1f}" for x, y in run)
                parts.append(f"<polyline class='line' points='{path}' style='stroke:{color}'/>")
        for x, y, value, i in points:
            tip = f"{series.name} - {chart.categories[i]}: {chart.fmt(value)}"
            parts.append(
                f"<g><title>{esc(tip)}</title>"
                f"<circle class='hit' cx='{x:.1f}' cy='{y:.1f}' r='12'/>"
                f"<circle class='dot' cx='{x:.1f}' cy='{y:.1f}' r='4' style='fill:{color}'/></g>"
            )
        if points:
            x, y, value, _ = points[-1]
            label_rows.append(y)
            end_labels.append(
                f"<text class='label' x='{x + 9:.1f}' y='{y + 4:.1f}'>"
                f"{esc(series.name)} {esc(chart.fmt(value))}</text>"
            )
    # Direct end labels only when every one has its own line of space: nudged
    # apart they detach from their series, and the legend already names them.
    rows = sorted(label_rows)
    if all(b - a >= _LABEL_GAP for a, b in pairwise(rows)):
        parts.extend(end_labels)
    parts.append("</svg>")
    return _figure(chart.caption, "".join(parts), _legend([(s.name, s.slot) for s in chart.series]))


def _bar_path(x: float, y: float, w: float, h: float, radius: float, end: str) -> str:
    """A bar with a rounded data end and a square baseline end."""
    r = max(0.0, min(radius, w / 2 if end == "top" else h / 2, h if end == "top" else w))
    if end == "right":
        return (
            f"M{x:.1f},{y:.1f} H{x + w - r:.1f} Q{x + w:.1f},{y:.1f} {x + w:.1f},{y + r:.1f} "
            f"V{y + h - r:.1f} Q{x + w:.1f},{y + h:.1f} {x + w - r:.1f},{y + h:.1f} "
            f"H{x:.1f} Z"
        )
    return (
        f"M{x:.1f},{y + h:.1f} V{y + r:.1f} Q{x:.1f},{y:.1f} {x + r:.1f},{y:.1f} "
        f"H{x + w - r:.1f} Q{x + w:.1f},{y:.1f} {x + w:.1f},{y + r:.1f} V{y + h:.1f} Z"
    )


def hbar_chart(caption: str, rows: Sequence[tuple[str, float]], fmt: Callable[[Any], str]) -> str:
    """Ranked horizontal bars, one series, value at each tip."""
    if not rows:
        return ""
    bar, gap = 14, 8
    label_w, value_w = 220, 70
    width = _WIDTH
    height = len(rows) * (bar + gap) + gap
    plot_w = width - label_w - value_w
    peak = max(value for _, value in rows) or 1.0
    parts = [
        f"<svg class='chart' viewBox='0 0 {width} {height}' role='img' "
        f"aria-label='{esc(caption)}' xmlns='http://www.w3.org/2000/svg'>",
        f"<line class='axis' x1='{label_w}' x2='{label_w}' y1='0' y2='{height}'/>",
    ]
    for i, (name, value) in enumerate(rows):
        y = gap + i * (bar + gap)
        w = max(1.0, plot_w * value / peak)
        shown = name if len(name) <= 28 else name[:27] + "..."
        parts.append(
            f"<g><title>{esc(name)}: {esc(fmt(value))}</title>"
            f"<text class='label' x='{label_w - 8}' y='{y + bar - 3}' text-anchor='end'>"
            f"{esc(shown)}</text>"
            f"<path class='bar' d='{_bar_path(label_w + 1, y, w, bar, 4, 'right')}' "
            f"style='fill:var(--series-1)'/>"
            f"<text x='{label_w + 1 + w + 6:.1f}' y='{y + bar - 3}'>{esc(fmt(value))}</text></g>"
        )
    parts.append("</svg>")
    return _figure(caption, "".join(parts))


def column_chart(
    caption: str,
    categories: Sequence[str],
    values: Sequence[Number],
    fmt: Callable[[Any], str],
    references: Sequence[tuple[str, float]] = (),
) -> str:
    """One series of columns from a zero baseline, with labelled reference lines."""
    present = [v for v in values if v is not None]
    if not present:
        return ""
    width, height = _WIDTH, 220
    left, right, top, bottom = 60, 120, 12, 36
    plot_w, plot_h = width - left - right, height - top - bottom
    ticks = _nice_ticks(0.0, max([*present, *(v for _, v in references)]) * 1.1)
    y1 = ticks[-1]
    n = len(categories)
    slot = plot_w / n
    col = min(24.0, slot * 0.6)

    def y_at(v: float) -> float:
        return top + plot_h * (1 - v / y1)

    parts = [
        f"<svg class='chart' viewBox='0 0 {width} {height}' role='img' "
        f"aria-label='{esc(caption)}' xmlns='http://www.w3.org/2000/svg'>"
    ]
    for tick in ticks:
        y = y_at(tick)
        parts.append(
            f"<line class='grid' x1='{left}' x2='{left + plot_w}' y1='{y:.1f}' y2='{y:.1f}'/>"
        )
        parts.append(
            f"<text x='{left - 6}' y='{y + 4:.1f}' text-anchor='end'>{esc(fmt(tick))}</text>"
        )
    stride = _label_stride(categories, slot)
    for i, (label, value) in enumerate(zip(categories, values, strict=True)):
        cx = left + slot * (i + 0.5)
        if i % stride == 0:
            parts.append(
                f"<text x='{cx:.1f}' y='{top + plot_h + 16}' "
                f"text-anchor='middle'>{esc(label)}</text>"
            )
        if value is None:
            continue
        h = max(1.0, plot_h * value / y1)
        parts.append(
            f"<g><title>{esc(label)}: {esc(fmt(value))}</title>"
            f"<rect class='hit' x='{cx - slot / 2:.1f}' y='{top}' width='{slot:.1f}' "
            f"height='{plot_h}'/>"
            f"<path class='bar' d='{_bar_path(cx - col / 2, top + plot_h - h, col, h, 4, 'top')}' "
            f"style='fill:var(--series-1)'/></g>"
        )
    parts.append(
        f"<line class='axis' x1='{left}' x2='{left + plot_w}' "
        f"y1='{top + plot_h}' y2='{top + plot_h}'/>"
    )
    for label, value in references:
        y = y_at(value)
        parts.append(
            f"<line class='reference' x1='{left}' x2='{left + plot_w}' y1='{y:.1f}' y2='{y:.1f}'/>"
        )
        parts.append(
            f"<text class='label' x='{left + plot_w + 6}' y='{y + 4:.1f}'>"
            f"{esc(label)} {value:g}</text>"
        )
    parts.append("</svg>")
    return _figure(caption, "".join(parts))


def render_page(title: str, body: str) -> str:
    return page(title, body, extra_css=_CHART_CSS)
