"""Push mbt run results to the Pushgateway (runs inside the runner image).

Implements the metric spec documented in docs/tutorial.md step 14, parsing
target/run_results.json (the machine-readable integration contract - shift
and realized-metric values never travel as typed events):

  mbt_node_success, mbt_node_duration_seconds, mbt_test_metric{metric=},
  mbt_realized_metric{metric=}, mbt_gate_passed,
  mbt_gate_margin{kind=threshold|champion|ground_truth},
  mbt_shift_value{monitor=,subject=,measure=}, mbt_shift_threshold
  (push_time_seconds comes free per group)

Grouped per (job, project, target, command, node); series carry the spec's
owner when the manifest exposes one. Push failures are deliberately
best-effort (warn + exit 0): observability observes, it must never fail a
pipeline (mbt's own exit codes enforce).

Usage: push_metrics.py [project_dir] (default: cwd)
"""

import json
import math
import os
import sys
import urllib.error
import urllib.parse
import urllib.request
from pathlib import Path

GATEWAY = os.environ.get("MBT_PUSHGATEWAY", "http://pushgateway:9091")


def _fmt(value: float) -> str:
    return "NaN" if value is None or math.isnan(value) else repr(float(value))


def _labels(pairs: dict) -> str:
    inner = ",".join(
        f'{key}="{str(val).replace(chr(92), chr(92) * 2).replace(chr(34), chr(92) + chr(34))}"'
        for key, val in pairs.items()
        if val is not None
    )
    return "{" + inner + "}" if inner else ""


def _owner_by_node(project_dir: Path) -> dict:
    """Best-effort owner lookup from the manifest; absent owners are omitted."""
    owners: dict = {}
    try:
        manifest = json.loads((project_dir / "target" / "manifest.json").read_text())
        nodes = manifest.get("nodes", {})
        if isinstance(nodes, dict):
            items = nodes.items()
        else:
            items = ((n.get("unique_id"), n) for n in nodes)
        for unique_id, node in items:
            blob = json.dumps(node)
            marker = '"owner": "'
            idx = blob.find(marker)
            if idx >= 0:
                owners[unique_id] = blob[idx + len(marker) : blob.index('"', idx + len(marker))]
    except Exception:
        pass
    return owners


def render_node(result: dict, owner: str | None) -> str:
    base = {"owner": owner} if owner else {}
    lines = [
        f"mbt_node_success{_labels(base)} {1 if result.get('status') == 'success' else 0}",
    ]
    if result.get("execution_time_s") is not None:
        lines.append(f"mbt_node_duration_seconds{_labels(base)} {_fmt(result['execution_time_s'])}")

    metric_name = (
        "mbt_realized_metric" if result.get("_command") == "monitor" else "mbt_test_metric"
    )
    for name, value in (result.get("metrics") or {}).items():
        if isinstance(value, (int, float)):
            lines.append(f"{metric_name}{_labels({**base, 'metric': name})} {_fmt(value)}")

    gates = result.get("gates") or []
    if gates:
        all_passed = all(g.get("passed") for g in gates)
        lines.append(f"mbt_gate_passed{_labels(base)} {1 if all_passed else 0}")
        for gate in gates:
            kind = gate.get("kind") or ("champion" if gate.get("champion_version") else "threshold")
            margin = gate.get("margin")
            actual, threshold = gate.get("actual"), gate.get("threshold")
            if margin is None and actual is not None and threshold is not None:
                margin = actual - threshold
            if margin is None and gate.get("actual_delta") is not None:
                margin = gate["actual_delta"] - (gate.get("min_delta") or 0.0)
            if margin is not None:
                lines.append(f"mbt_gate_margin{_labels({**base, 'kind': kind})} {_fmt(margin)}")

    for mon in result.get("monitors") or []:
        which = mon.get("monitor")
        if which in ("feature_shift", "prediction_shift"):
            labels = {**base, "monitor": which}
            labels.update({"subject": mon.get("subject"), "measure": mon.get("measure")})
            if mon.get("value") is not None:
                lines.append(f"mbt_shift_value{_labels(labels)} {_fmt(mon['value'])}")
            if mon.get("threshold") is not None:
                lines.append(f"mbt_shift_threshold{_labels(labels)} {_fmt(mon['threshold'])}")
        elif which == "ground_truth" and mon.get("value") is not None:
            labels = {**base, "metric": mon.get("measure"), "run_key": mon.get("subject")}
            lines.append(f"mbt_realized_metric{_labels(labels)} {_fmt(mon['value'])}")
    return "\n".join(lines) + "\n"


def main() -> int:
    project_dir = Path(sys.argv[1]) if len(sys.argv) > 1 else Path.cwd()
    results_path = project_dir / "target" / "run_results.json"
    if not results_path.exists():
        print(f"push_metrics: {results_path} not found, nothing to push", file=sys.stderr)
        return 0
    payload = json.loads(results_path.read_text())
    meta = payload.get("metadata", {})
    owners = _owner_by_node(project_dir)

    project = "unknown"
    try:
        import yaml

        project = yaml.safe_load((project_dir / "mbt_project.yml").read_text())["name"]
    except Exception:
        pass

    pushed = 0
    for result in payload.get("results", []):
        result["_command"] = meta.get("command", "")
        group = "/".join(
            [
                "metrics/job/mbt",
                "project/" + urllib.parse.quote(project, safe=""),
                "target/" + urllib.parse.quote(str(meta.get("target", "unknown")), safe=""),
                "command/" + urllib.parse.quote(str(meta.get("command", "unknown")), safe=""),
                "node/" + urllib.parse.quote(str(result.get("unique_id", "unknown")), safe=""),
            ]
        )
        body = render_node(result, owners.get(result.get("unique_id"))).encode()
        req = urllib.request.Request(f"{GATEWAY}/{group}", data=body, method="PUT")
        try:
            urllib.request.urlopen(req, timeout=5)
            pushed += 1
        except (urllib.error.URLError, OSError) as exc:
            print(f"push_metrics: WARN pushgateway unreachable ({exc}); skipping", file=sys.stderr)
            return 0
    print(f"push_metrics: pushed {pushed} node group(s) for command={meta.get('command')}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
