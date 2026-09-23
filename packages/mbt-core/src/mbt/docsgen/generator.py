"""Static docs site: DAG lineage + one model card per model (FR-DOCS-01..03).

Self-contained output (inline CSS, no CDN) so the site works on corp
networks and artifact hosting. Input: manifest + latest run_results.
"""

import html
import json
from pathlib import Path
from typing import Any

import networkx as nx

from mbt.artifacts.manifest import Manifest
from mbt.artifacts.run_results import NodeResult, RunResults
from mbt.docsgen.html import page, sparkline
from mbt.secrets import redact
from mbt_adapter_base import (
    AUTO,
)


def _lineage_svg(manifest: Manifest) -> str:
    """Server-side layered DAG rendering: no JS graph library needed."""
    graph = manifest.graph()
    if not graph.nodes:
        return "<p class='muted'>no nodes</p>"
    layers: dict[str, int] = {}
    for uid in nx.lexicographical_topological_sort(graph):
        preds = list(graph.predecessors(uid))
        layers[uid] = max((layers[p] + 1 for p in preds), default=0)
    by_layer: dict[int, list[str]] = {}
    for uid, layer in layers.items():
        by_layer.setdefault(layer, []).append(uid)

    box_w, box_h, gap_x, gap_y, pad = 240, 34, 60, 18, 20
    positions: dict[str, tuple[int, int]] = {}
    for layer, uids in sorted(by_layer.items()):
        for row, uid in enumerate(sorted(uids)):
            positions[uid] = (pad + layer * (box_w + gap_x), pad + row * (box_h + gap_y))
    width = pad * 2 + (max(by_layer) + 1) * (box_w + gap_x) - gap_x
    height = pad * 2 + max(len(v) for v in by_layer.values()) * (box_h + gap_y) - gap_y

    parts = [
        f'<svg viewBox="0 0 {width} {height}" width="100%" xmlns="http://www.w3.org/2000/svg">',
        '<defs><marker id="arrow" viewBox="0 0 10 10" refX="9" refY="5" '
        'markerWidth="7" markerHeight="7" orient="auto-start-reverse">'
        # var(--edge), not a literal, so the arrowhead follows the dark palette
        # along with the edge it terminates.
        '<path d="M 0 0 L 10 5 L 0 10 z" fill="var(--edge)"/></marker></defs>',
    ]
    for u, v in graph.edges:
        x1, y1 = positions[u][0] + box_w, positions[u][1] + box_h // 2
        x2, y2 = positions[v][0], positions[v][1] + box_h // 2
        mx = (x1 + x2) / 2
        parts.append(f'<path class="edge" d="M {x1} {y1} C {mx} {y1}, {mx} {y2}, {x2} {y2}"/>')
    kinds = {uid: data.get("resource_type", "model") for uid, data in graph.nodes(data=True)}
    for uid, (x, y) in positions.items():
        kind = kinds.get(uid, "model")
        label = html.escape(uid.split(".", 2)[-1])
        name = uid.rsplit(".", 1)[-1]
        link_open = link_close = ""
        if kind == "model":
            link_open, link_close = f'<a href="model_{html.escape(name)}.html">', "</a>"
        parts.append(
            f'<g class="node {kind}">{link_open}'
            f'<rect x="{x}" y="{y}" width="{box_w}" height="{box_h}"/>'
            f'<text x="{x + 10}" y="{y + 21}">{html.escape(kind)}: {label}</text>'
            f"{link_close}</g>"
        )
    parts.append("</svg>")
    return "".join(parts)


def _treatment_row(name: str, features: dict[str, Any]) -> str:
    """One column's declared treatment, in the order it is applied (ADR-27)."""
    policy = (features.get("categorical") or {}).get(name)
    transform = (features.get("transforms") or {}).get(name) or {}
    parts: list[str] = []
    if policy is not None:
        parts.append("categorical")
        if policy.get("levels"):
            parts.append(f"levels {policy['levels']}")
        if policy.get("min_frequency") is not None:
            parts.append(f"pool below {policy['min_frequency']}")
        if policy.get("max_levels") is not None:
            parts.append(f"max {policy['max_levels']} levels")
        if policy.get("null_as_level"):
            parts.append("null is a level")
    cap = transform.get("cap") or {}
    if cap.get("min") is not None:
        parts.append(f"floor at {cap['min']}")
    if cap.get("max") is not None:
        parts.append(f"cap at {cap['max']}")
    if transform.get("log"):
        parts.append("log1p")
    if transform.get("percentile"):
        parts.append(f"percentile within {transform['percentile']}")
    direction = (features.get("monotonic") or {}).get(name) or transform.get("monotonic")
    if direction:
        parts.append(f"monotone {direction}")
    return (
        f"<tr><td><code>{html.escape(name)}</code></td>"
        f"<td>{html.escape(', '.join(parts))}</td></tr>"
    )


def _treatment_table(features: dict[str, Any]) -> str:
    """The per-column treatment block, so a card reader can see what the model
    actually consumed without opening the YAML."""
    named = [
        *(features.get("categorical") or {}),
        *(features.get("transforms") or {}),
        *(features.get("monotonic") or {}),
    ]
    ordered = list(dict.fromkeys(named))
    if not ordered:
        return ""
    rows = "".join(_treatment_row(name, features) for name in ordered)
    return f"<table><tr><th>column</th><th>treatment</th></tr>{rows}</table>"


def _metric_table(result: NodeResult | None) -> str:
    if result is None or not result.metrics:
        # Naming the file matters: this used to read "run mbt build" and was
        # shown to users who HAD just built, because a later `mbt score`
        # overwrote the shared results file (FEEDBACK v3 A-2).
        return (
            "<p class='muted'>no metrics for this model in "
            "<code>target/run_results.build.json</code> - run <code>mbt build</code> "
            "(or <code>mbt run</code>), then <code>mbt docs generate</code></p>"
        )
    # A cross-validated fold mean sits beside the single-split value (R2-7), so
    # an optimistic single split is visible at a glance.
    backtest = result.backtest_metrics
    backtest_std = result.backtest_std
    header = (
        "<tr><th>metric</th><th>value</th>"
        + ("<th>backtest (cross-validated mean &pm; std)</th>" if backtest else "")
        + "</tr>"
    )
    rows = ""
    for k, v in sorted(result.metrics.items()):
        cell = ""
        if backtest:
            bt = backtest.get(k)
            if bt is not None:
                # the std (fold-to-fold spread) shows whether the mean is stable
                std = backtest_std.get(k)
                text = f"{bt:.4f} &pm; {std:.4f}" if std is not None else f"{bt:.4f}"
                cell = f"<td>{text}</td>"
            else:
                cell = "<td class='muted'>-</td>"
        rows += f"<tr><td><code>{html.escape(k)}</code></td><td>{v:.4f}</td>{cell}</tr>"
    out = f"<table>{header}{rows}</table>"
    if result.slices:
        slice_rows = ""
        for slice_key, metrics in sorted(result.slices.items()):
            for metric, value in sorted(metrics.items()):
                slice_rows += (
                    f"<tr><td><code>{html.escape(slice_key)}</code></td>"
                    f"<td><code>{html.escape(metric)}</code></td><td>{value:.4f}</td></tr>"
                )
        out += (
            "<h2>Slices</h2><table><tr><th>slice</th><th>metric</th><th>value</th></tr>"
            f"{slice_rows}</table>"
        )
    return out


#: The card lists as many features as the training report ranks by default.
_IMPORTANCE_ROWS = 20


def _importance_table(result: NodeResult | None) -> str:
    if result is None or not result.feature_importance:
        return ""
    top = sorted(result.feature_importance.items(), key=lambda kv: (-kv[1], kv[0]))[
        :_IMPORTANCE_ROWS
    ]
    rows = "".join(
        f"<tr><td><code>{html.escape(name)}</code></td><td>{share:.1%}</td></tr>"
        for name, share in top
    )
    return (
        "<h2>Feature importance (normalized, latest run)</h2>"
        f"<table><tr><th>feature</th><th>share</th></tr>{rows}</table>"
    )


def _partial_dependence_section(result: NodeResult | None) -> str:
    if result is None or not result.partial_dependence:
        return ""
    rows = "".join(
        f"<tr><td><code>{html.escape(feature)}</code></td>"
        f"<td>{sparkline(curve)}</td>"
        f"<td>{curve[0][1]:.3f} &rarr; {curve[-1][1]:.3f}</td></tr>"
        for feature, curve in result.partial_dependence.items()
    )
    return (
        "<h2>Partial dependence (avg prediction across each feature's range)</h2>"
        f"<table><tr><th>feature</th><th>response</th><th>low &rarr; high</th></tr>{rows}</table>"
    )


def _gate_table(result: NodeResult | None) -> str:
    if result is None or not result.gates:
        return ""
    rows = ""
    for gate in result.gates:
        badge = "ok" if gate.passed else "bad"
        expected = (
            f"threshold {gate.expected}"
            if gate.kind == "threshold"
            else f"champion v{gate.champion_version or '-'} + {gate.min_delta}"
        )
        if gate.period is not None:
            where = f" at {gate.cell}" if gate.cell else ""
            expected += f" (worst {gate.period}{where})"
        actual = "-" if gate.actual is None else f"{gate.actual:.4f}"
        if not gate.applicable:
            badge, actual = "warn", "not applicable"
        rows += (
            f"<tr><td><code>{html.escape(gate.metric)}</code></td>"
            f"<td>{html.escape(expected)}</td><td>{actual}</td>"
            f"<td><span class='badge {badge}'>{_gate_label(gate)}</span></td></tr>"
        )
    return (
        "<h2>Gate history (latest run)</h2>"
        f"<table><tr><th>metric</th><th>gate</th><th>actual</th><th>result</th></tr>{rows}</table>"
    )


def _gate_label(gate: Any) -> str:
    if not gate.applicable:
        return "N/A"
    return "PASS" if gate.passed else "FAIL"


def _stability_table(result: NodeResult | None) -> str:
    """The after-test stability gates (ADR-30), one row per judged statistic."""
    if result is None or not result.monitors:
        return ""
    rows = "".join(
        f"<tr><td><code>{html.escape(m.monitor)}</code></td>"
        f"<td><code>{html.escape(m.subject or '-')}</code></td>"
        f"<td>{html.escape(m.measure)}</td>"
        f"<td>{'-' if m.value is None else f'{m.value:.4f}'}</td><td>{m.threshold:.4f}</td>"
        f"<td><span class='badge {'ok' if m.passed else 'bad'}'>"
        f"{'PASS' if m.passed else 'FAIL'}</span></td></tr>"
        for m in result.monitors
    )
    return (
        "<h2>Stability after the test window (latest run)</h2>"
        "<table><tr><th>monitor</th><th>period: subject</th><th>measure</th><th>value</th>"
        f"<th>fail above</th><th>result</th></tr>{rows}</table>"
    )


def _model_card(manifest: Manifest, uid: str, result: NodeResult | None) -> str:
    node = manifest.nodes[uid]
    config: dict[str, Any] = node.config
    dataset_uid = next((d for d in node.depends_on if d.startswith("dataset.")), None)
    dataset = manifest.nodes.get(dataset_uid) if dataset_uid else None

    # The manifest keeps the AUTO sentinel verbatim (ADR-12); the card is a
    # human presentation layer, so show the keyword the user wrote ("auto")
    # rather than the internal "__mbt_auto__" token.
    hyper_rows = "".join(
        f"<tr><td><code>{html.escape(str(k))}</code></td>"
        f"<td><code>{html.escape('auto' if v == AUTO else str(v))}</code></td>"
        f"<td>{html.escape(str(result.resolved_auto.get(k, ''))) if result else ''}</td></tr>"
        for k, v in sorted(config.get("hyperparameters", {}).items())
    )
    features = config.get("features", {})
    windows = dataset.resolved.get("windows", {}) if dataset else {}
    window_rows = "".join(
        f"<tr><td>{html.escape(split)}</td><td><code>{html.escape(str(bounds[0]))}</code></td>"
        f"<td><code>{html.escape(str(bounds[1]))}</code></td></tr>"
        for split, bounds in sorted(windows.items())
    )
    registration = config.get("registration") or {}
    reg_line = ""
    if result and result.registration:
        reg_line = (
            f"<p>Registered as <code>{html.escape(result.registration.name)}</code> "
            f"v{result.registration.version} → "
            f"<span class='badge plain'>{html.escape(result.registration.stage)}</span></p>"
        )
    elif registration:
        reg_line = f"<p>Registers as <code>{html.escape(str(registration.get('name')))}</code></p>"
    tracking_line = (
        f"<p>Tracking run: <code>{html.escape(result.tracking_run_id)}</code></p>"
        if result and result.tracking_run_id
        else ""
    )

    body = f"""
    <p><a href="index.html">← lineage</a></p>
    <h1>{html.escape(node.name)}</h1>
    <p class="muted">{html.escape(str(config.get("description", "")))}</p>
    <p>
      <span class="badge plain">{html.escape(str(node.task))}</span>
      <span class="badge plain">adapter: {html.escape(str(node.adapter))}</span>
      owner: <code>{html.escape(str(config.get("owner", "")))}</code>
      tags: {" ".join(f"<code>{html.escape(t)}</code>" for t in node.tags) or "-"}
    </p>
    {reg_line}{tracking_line}
    <h2>Identity</h2>
    <table>
      <tr><th>config_hash</th><td><code>{node.config_hash}</code></td></tr>
      <tr><th>input_hash</th><td><code>{node.input_hash}</code></td></tr>
      <tr><th>seed</th><td><code>{node.seed}</code></td></tr>
      <tr><th>dataset</th><td><code>{html.escape(dataset_uid or "-")}</code></td></tr>
      <tr><th>data snapshot</th>
      <td><code>{html.escape(str(dataset.snapshot_id if dataset else "-"))}</code></td></tr>
    </table>
    <h2>Data window</h2>
    {
        (f"<table><tr><th>split</th><th>start</th><th>end</th></tr>{window_rows}</table>")
        if window_rows
        else "<p class='muted'>random split</p>"
    }
    <h2>Features</h2>
    <p>include: <code>{html.escape(str(features.get("include", ["*"])))}</code><br>
       exclude: <code>{html.escape(str(features.get("exclude", [])))}</code></p>
    {_treatment_table(features)}
    {_importance_table(result)}
    {_partial_dependence_section(result)}
    <h2>Hyperparameters</h2>
    <table><tr><th>param</th><th>value</th><th>resolved auto</th></tr>{hyper_rows}</table>
    <h2>Metrics (latest run)</h2>
    {_metric_table(result)}
    {_gate_table(result)}
    {_stability_table(result)}
    <p class="muted">The training report - bin tables, per-period performance, stability and
    feature importance - is on the tracking run under <code>report/</code> (ADR-30).</p>
    """
    return page(f"{node.name} - mbt model card", body)


def generate_docs(
    manifest: Manifest,
    run_results: RunResults | None,
    output_dir: Path,
) -> Path:
    """Render the static site into ``output_dir``; returns the index path."""
    output_dir.mkdir(parents=True, exist_ok=True)
    results_by_id: dict[str, NodeResult] = {}
    if run_results is not None:
        results_by_id = {r.unique_id: r for r in run_results.results}

    model_rows = ""
    for uid, node in sorted(manifest.nodes.items()):
        if node.resource_type != "model":
            continue
        result = results_by_id.get(uid)
        status = result.status if result else "-"
        badge = {"success": "ok", "-": "plain"}.get(status, "bad")
        model_rows += (
            f"<tr><td><a href='model_{html.escape(node.name)}.html'>"
            f"<code>{html.escape(node.name)}</code></a></td>"
            f"<td>{html.escape(str(node.task))}</td>"
            f"<td>{html.escape(str(node.adapter))}</td>"
            f"<td><code>{html.escape(str(node.config.get('owner', '')))}</code></td>"
            f"<td><span class='badge {badge}'>{html.escape(status)}</span></td></tr>"
        )
        # Redact tainted env_var() values that rendered into spec config
        # (description, owner, hyperparameters, ...): docs are published, so a
        # leak here is public. Redacting the assembled page catches every field.
        (output_dir / f"model_{node.name}.html").write_text(
            redact(_model_card(manifest, uid, result))
        )

    exposure_rows = "".join(
        f"<tr><td><code>{html.escape(e.name)}</code></td>"
        f"<td>{html.escape(str(e.config.get('type', '')))}</td>"
        f"<td>{', '.join(f'<code>{html.escape(d)}</code>' for d in e.depends_on)}</td>"
        f"<td><code>{html.escape(str(e.config.get('owner', '')))}</code></td></tr>"
        for e in sorted(manifest.exposures.values(), key=lambda e: e.name)
    )

    meta = manifest.metadata
    index_body = f"""
    <h1>{html.escape(meta.project_name)} <span class="muted">- mbt docs</span></h1>
    <p class="muted">target <code>{html.escape(meta.target)}</code> ·
       anchor <code>{html.escape(meta.anchor)}</code> ·
       mbt {html.escape(meta.mbt_version)} ·
       git <code>{html.escape(str(meta.git.commit or "-")[:12])}</code></p>
    <h2>Lineage</h2>
    {_lineage_svg(manifest)}
    <h2>Models</h2>
    <table><tr><th>model</th><th>task</th><th>adapter</th><th>owner</th><th>last status</th></tr>
    {model_rows or "<tr><td colspan='5' class='muted'>none</td></tr>"}</table>
    <h2>Exposures</h2>
    <table><tr><th>exposure</th><th>type</th><th>depends on</th><th>owner</th></tr>
    {exposure_rows or "<tr><td colspan='4' class='muted'>none</td></tr>"}</table>
    <script type="application/json" id="lineage-data">{
        json.dumps(
            {"nodes": sorted(manifest.nodes), "edges": [[u, v] for u, v in manifest.graph().edges]}
        )
    }</script>
    """
    index = output_dir / "index.html"
    index.write_text(redact(page(f"{meta.project_name} - mbt docs", index_body)))
    return index
