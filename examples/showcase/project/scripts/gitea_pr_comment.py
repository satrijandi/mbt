"""mbt PR comment bot for Gitea - a faithful port of the scaffold's
pr_comment.js to Gitea's issue-comments API (DESIGN.md section 6).

Renders metrics vs champion, the gate table, and retrained nodes purely from
target/run_results.json + target/state_diff.json - no re-computation in CI
scripts. Updates its own comment in place (one marker, never stacks); the
GitHub runner cost line is replaced with total execution time.

Reads Woodpecker's step environment: CI_FORGE_URL, CI_REPO_OWNER,
CI_REPO_NAME, CI_COMMIT_PULL_REQUEST, plus GITEA_TOKEN (a repo secret).
"""

import json
import os
import sys
import urllib.request
from pathlib import Path

MARKER = "<!-- mbt-pr-comment -->"


def load_json(path: str) -> dict | None:
    try:
        return json.loads(Path(path).read_text())
    except (OSError, ValueError):
        return None


def fmt(value: float | None, digits: int = 4) -> str:
    return "-" if value is None else f"{value:.{digits}f}"


def gate_row(node_id: str, gate: dict) -> str:
    status = "PASS" if gate.get("passed") else "**FAIL**"
    if gate.get("kind") == "champion":
        champ = (
            f"v{gate['champion_version']} = {fmt(gate.get('champion_value'))}"
            if gate.get("champion_version")
            else "none (bootstrap)"
        )
        delta = fmt(gate.get("actual_delta"))
        return (
            f"| {node_id} | {gate.get('metric')} | champion ({champ}) "
            f"| {fmt(gate.get('actual'))} | {delta} >= {gate.get('min_delta')} | {status} |"
        )
    return (
        f"| {node_id} | {gate.get('metric')} | threshold {gate.get('expected')} "
        f"| {fmt(gate.get('actual'))} | - | {status} |"
    )


def build_body() -> str:
    results = load_json("target/run_results.json")
    diff = load_json("target/state_diff.json")
    body = f"{MARKER}\n## mbt build report\n\n"

    if not results:
        return body + (
            "No `run_results.json` produced - the build failed before execution. "
            "Check the pipeline logs.\n"
        )

    nodes = results.get("results", [])
    trained = [r for r in nodes if r.get("unique_id", "").startswith("model.")]
    failed = [
        r
        for r in nodes
        if r.get("status") in ("error", "gate_failed", "test_failed", "monitor_failed")
    ]

    meta = results.get("metadata", {})
    body += f"**Target:** `{meta.get('target')}` · **Command:** `{meta.get('command')}`"
    if meta.get("selector"):
        body += f" · **Selector:** `{meta.get('selector')}`"
    body += "\n\n"

    if diff is not None:
        modified = [
            f"`{d.get('unique_id')}` ({', '.join(d.get('components', []))})"
            for d in diff.get("modified", [])
        ]
        added = [f"`{d.get('unique_id')}`" for d in diff.get("added", [])]
        body += "### Changed vs production\n"
        if modified or added:
            lines = [f"- added: {a}" for a in added] + [f"- modified: {m}" for m in modified]
            body += "\n".join(lines) + "\n\n"
        else:
            body += "Nothing modified - no retraining needed.\n\n"
        if (diff.get("env") or {}).get("changed"):
            body += (
                "> ⚠️ environment digest changed vs the reference manifest "
                "(not treated as modified by default).\n\n"
            )

    if nodes:
        body += "### Nodes\n| node | status | time |\n|---|---|---|\n"
        for r in nodes:
            seconds = r.get("execution_time_s") or 0
            body += f"| `{r.get('unique_id')}` | {r.get('status')} | {seconds:.1f}s |\n"
        body += "\n"

    gate_rows = [gate_row(r["unique_id"], g) for r in nodes for g in (r.get("gates") or [])]
    if gate_rows:
        body += (
            "### Gates (metrics vs champion)\n"
            "| node | metric | gate | actual | delta | result |\n|---|---|---|---|---|---|\n"
        )
        body += "\n".join(gate_rows) + "\n\n"

    total_seconds = sum(r.get("execution_time_s") or 0 for r in nodes)
    body += f"### Time\nExecution time: **{total_seconds:.0f}s** across {len(trained)} model(s).\n"

    if failed:
        body += f"\n> ❌ {len(failed)} node(s) failed - registration blocked.\n"
    return body


def api(method: str, url: str, token: str, payload: dict | None = None) -> object:
    data = json.dumps(payload).encode() if payload is not None else None
    req = urllib.request.Request(
        url,
        data=data,
        method=method,
        headers={"Authorization": f"token {token}", "Content-Type": "application/json"},
    )
    with urllib.request.urlopen(req, timeout=30) as resp:
        raw = resp.read()
    return json.loads(raw) if raw else None


def main() -> int:
    forge = os.environ["CI_FORGE_URL"].rstrip("/")
    owner = os.environ["CI_REPO_OWNER"]
    repo = os.environ["CI_REPO_NAME"]
    index = os.environ["CI_COMMIT_PULL_REQUEST"]
    token = os.environ["GITEA_TOKEN"]

    body = build_body()
    base = f"{forge}/api/v1/repos/{owner}/{repo}"
    comments = api("GET", f"{base}/issues/{index}/comments", token)
    existing = next((c for c in comments if MARKER in (c.get("body") or "")), None)
    if existing:
        # update in place instead of stacking
        api("PATCH", f"{base}/issues/comments/{existing['id']}", token, {"body": body})
        print(f"gitea_pr_comment: updated comment {existing['id']} on PR #{index}")
    else:
        api("POST", f"{base}/issues/{index}/comments", token, {"body": body})
        print(f"gitea_pr_comment: created comment on PR #{index}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
