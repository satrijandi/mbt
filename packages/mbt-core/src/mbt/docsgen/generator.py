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

_CSS = """
:root { --bg:#ffffff; --fg:#1f2430; --muted:#6b7280; --line:#e5e7eb;
        --accent:#2563eb; --ok:#15803d; --bad:#b91c1c; --warn:#a16207; }
* { box-sizing: border-box; }
body { font: 15px/1.5 -apple-system, "Segoe UI", Roboto, sans-serif;
       color: var(--fg); background: var(--bg); margin: 0; }
main { max-width: 1080px; margin: 0 auto; padding: 2rem 1.5rem 4rem; }
h1 { font-size: 1.6rem; margin: 0 0 .25rem; }
h2 { font-size: 1.15rem; margin: 2rem 0 .5rem; border-bottom: 1px solid var(--line);
     padding-bottom: .25rem; }
a { color: var(--accent); text-decoration: none; }
a:hover { text-decoration: underline; }
.muted { color: var(--muted); }
table { border-collapse: collapse; width: 100%; margin: .5rem 0 1rem; }
th, td { text-align: left; padding: .35rem .6rem; border-bottom: 1px solid var(--line);
         font-size: .92rem; vertical-align: top; }
th { color: var(--muted); font-weight: 600; }
code, .mono { font-family: ui-monospace, SFMono-Regular, Menlo, monospace;
              font-size: .85em; background: #f3f4f6; padding: .1em .35em;
              border-radius: 4px; }
.badge { display: inline-block; padding: .1em .55em; border-radius: 999px;
         font-size: .8rem; font-weight: 600; }
.badge.ok { background: #dcfce7; color: var(--ok); }
.badge.bad { background: #fee2e2; color: var(--bad); }
.badge.warn { background: #fef9c3; color: var(--warn); }
.badge.plain { background: #eef2ff; color: var(--accent); }
svg .node rect { fill: #eef2ff; stroke: var(--accent); rx: 6; }
svg .node.dataset rect { fill: #ecfdf5; stroke: var(--ok); }
svg .node.source rect { fill: #f3f4f6; stroke: var(--muted); }
svg .node.exposure rect { fill: #fef3c7; stroke: var(--warn); }
svg text { font: 12px ui-monospace, Menlo, monospace; fill: var(--fg); }
svg .edge { stroke: #cbd5e1; stroke-width: 1.2; fill: none; marker-end: url(#arrow); }
"""


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
        f'<svg viewBox="0 0 {width} {height}" width="100%" '
        f'xmlns="http://www.w3.org/2000/svg">',
        '<defs><marker id="arrow" viewBox="0 0 10 10" refX="9" refY="5" '
        'markerWidth="7" markerHeight="7" orient="auto-start-reverse">'
        '<path d="M 0 0 L 10 5 L 0 10 z" fill="#cbd5e1"/></marker></defs>',
    ]
    for u, v in graph.edges:
        x1, y1 = positions[u][0] + box_w, positions[u][1] + box_h // 2
        x2, y2 = positions[v][0], positions[v][1] + box_h // 2
        mx = (x1 + x2) / 2
        parts.append(
            f'<path class="edge" d="M {x1} {y1} C {mx} {y1}, {mx} {y2}, {x2} {y2}"/>'
        )
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


def _page(title: str, body: str) -> str:
    return (
        "<!DOCTYPE html><html lang='en'><head><meta charset='utf-8'>"
        f"<title>{html.escape(title)}</title>"
        "<meta name='viewport' content='width=device-width, initial-scale=1'>"
        f"<style>{_CSS}</style></head><body><main>{body}</main></body></html>"
    )


def _metric_table(result: NodeResult | None) -> str:
    if result is None or not result.metrics:
        return "<p class='muted'>no run results yet - run <code>mbt build</code></p>"
    rows = "".join(
        f"<tr><td><code>{html.escape(k)}</code></td><td>{v:.4f}</td></tr>"
        for k, v in sorted(result.metrics.items())
    )
    out = f"<table><tr><th>metric</th><th>value</th></tr>{rows}</table>"
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
        actual = "-" if gate.actual is None else f"{gate.actual:.4f}"
        rows += (
            f"<tr><td><code>{html.escape(gate.metric)}</code></td>"
            f"<td>{html.escape(expected)}</td><td>{actual}</td>"
            f"<td><span class='badge {badge}'>{'PASS' if gate.passed else 'FAIL'}</span></td></tr>"
        )
    return (
        "<h2>Gate history (latest run)</h2>"
        f"<table><tr><th>metric</th><th>gate</th><th>actual</th><th>result</th></tr>{rows}</table>"
    )


def _model_card(
    manifest: Manifest, uid: str, result: NodeResult | None
) -> str:
    node = manifest.nodes[uid]
    config: dict[str, Any] = node.config
    dataset_uid = next((d for d in node.depends_on if d.startswith("dataset.")), None)
    dataset = manifest.nodes.get(dataset_uid) if dataset_uid else None

    hyper_rows = "".join(
        f"<tr><td><code>{html.escape(str(k))}</code></td><td><code>{html.escape(str(v))}</code></td>"
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
      <tr><th>data snapshot</th><td><code>{html.escape(str(dataset.snapshot_id if dataset else "-"))}</code></td></tr>
    </table>
    <h2>Data window</h2>
    {f"<table><tr><th>split</th><th>start</th><th>end</th></tr>{window_rows}</table>" if window_rows else "<p class='muted'>random split</p>"}
    <h2>Features</h2>
    <p>include: <code>{html.escape(str(features.get("include", ["*"])))}</code><br>
       exclude: <code>{html.escape(str(features.get("exclude", [])))}</code></p>
    <h2>Hyperparameters</h2>
    <table><tr><th>param</th><th>value</th><th>resolved auto</th></tr>{hyper_rows}</table>
    <h2>Metrics (latest run)</h2>
    {_metric_table(result)}
    {_gate_table(result)}
    """
    return _page(f"{node.name} - mbt model card", body)


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
        (output_dir / f"model_{node.name}.html").write_text(
            _model_card(manifest, uid, result)
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
    <script type="application/json" id="lineage-data">{json.dumps(
        {"nodes": sorted(manifest.nodes), "edges": [[u, v] for u, v in manifest.graph().edges]}
    )}</script>
    """
    index = output_dir / "index.html"
    index.write_text(_page(f"{meta.project_name} - mbt docs", index_body))
    return index
