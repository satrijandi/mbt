"""Shared HTML chrome for mbt's generated pages: model cards and training reports.

Self-contained by design (inline CSS, no CDN, no fonts) so a page renders on a
corporate network, in an artifact store, and inside a tracking UI's iframe.
"""

import html

BASE_CSS = """
:root { --bg:#ffffff; --fg:#1f2430; --muted:#6b7280; --line:#e5e7eb;
        --accent:#2563eb; --ok:#15803d; --bad:#b91c1c; --warn:#a16207;
        --chip:#f3f4f6; --chip-ok:#dcfce7; --chip-bad:#fee2e2;
        --chip-warn:#fef9c3; --chip-accent:#eef2ff; --chip-ds:#ecfdf5;
        --chip-exp:#fef3c7; --edge:#cbd5e1; }
/* Model cards are read on whatever the reader's OS is set to; a card that is
   a white rectangle at night is the one part of mbt's output nobody can
   configure. Every colour above is a variable so this override is complete -
   the palette shifts, the markup does not.
   Both palettes were checked against WCAG: every text/background pair clears
   AA, and body, muted, accent, and code text clear AAA in the dark one. If you
   retune a colour, re-check the pair it is used against rather than eyeballing
   it - the badge foregrounds sit on tinted chips, not on --bg. */
@media (prefers-color-scheme: dark) {
  :root { --bg:#0f1419; --fg:#e6e8eb; --muted:#9aa4b2; --line:#242c38;
          --accent:#7aa2f7; --ok:#5dc98a; --bad:#f07178; --warn:#e0af68;
          --chip:#1b2230; --chip-ok:#123524; --chip-bad:#3b1a1d;
          --chip-warn:#3a2f14; --chip-accent:#1a2436; --chip-ds:#12301f;
          --chip-exp:#332813; --edge:#3a4657; }
}
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
              font-size: .85em; background: var(--chip); padding: .1em .35em;
              border-radius: 4px; }
.badge { display: inline-block; padding: .1em .55em; border-radius: 999px;
         font-size: .8rem; font-weight: 600; }
.badge.ok { background: var(--chip-ok); color: var(--ok); }
.badge.bad { background: var(--chip-bad); color: var(--bad); }
.badge.warn { background: var(--chip-warn); color: var(--warn); }
.badge.plain { background: var(--chip-accent); color: var(--accent); }
svg .node rect { fill: var(--chip-accent); stroke: var(--accent); rx: 6; }
svg .node.dataset rect { fill: var(--chip-ds); stroke: var(--ok); }
svg .node.source rect { fill: var(--chip); stroke: var(--muted); }
svg .node.exposure rect { fill: var(--chip-exp); stroke: var(--warn); }
svg text { font: 12px ui-monospace, Menlo, monospace; fill: var(--fg); }
svg .edge { stroke: var(--edge); stroke-width: 1.2; fill: none; marker-end: url(#arrow); }
"""


def page(title: str, body: str, *, extra_css: str = "") -> str:
    """A complete HTML document around ``body``."""
    return (
        "<!DOCTYPE html><html lang='en'><head><meta charset='utf-8'>"
        f"<title>{html.escape(title)}</title>"
        "<meta name='viewport' content='width=device-width, initial-scale=1'>"
        f"<style>{BASE_CSS}{extra_css}</style></head><body><main>{body}</main></body></html>"
    )


def sparkline(curve: list[list[float]], width: int = 120, height: int = 24) -> str:
    ys = [point[1] for point in curve]
    low = min(ys)
    span = (max(ys) - low) or 1.0
    last = max(len(curve) - 1, 1)
    points = " ".join(
        f"{round(i / last * width, 1)},{round(height - (y - low) / span * height, 1)}"
        for i, y in enumerate(ys)
    )
    return (
        f'<svg viewBox="0 0 {width} {height}" width="{width}" height="{height}" '
        f'xmlns="http://www.w3.org/2000/svg"><polyline points="{points}" fill="none" '
        'stroke="var(--accent)" stroke-width="1.5"/></svg>'
    )
