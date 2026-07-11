#!/usr/bin/env bash
# Exit-code fidelity wrapper (DESIGN.md section 6). Woodpecker collapses any
# nonzero exit into "failed", erasing mbt's 1-vs-2 contract (1 hard error,
# 2 quality failure), so every executing CI step runs mbt through here:
#
#   1. run the wrapped command and capture its exit code;
#   2. write target/ci_exit_class (one line: "<class> <label>") - the PR
#      comment step, the alert below, and the test tier all read this file;
#   3. classify the alert: exit 2 notifies the failed nodes' owner (a model
#      quality verdict), any other nonzero pages on-call (infrastructure);
#   4. push metrics best-effort (an absent Pushgateway never fails CI);
#   5. re-exit with the original code.
#
# Usage: bash scripts/run_mbt.sh mbt build ...
# Env:   MBT_ALERT_WEBHOOK - webhook-sink URL (alerts are skipped when unset)

set -uo pipefail

"$@"
code=$?

mkdir -p target
case "$code" in
  0) class="0 ok" ;;
  2) class="2 quality-failure" ;;
  *) class="1 hard-error" ;;
esac
echo "$class" > target/ci_exit_class

if [ "$code" -ne 0 ] && [ -n "${MBT_ALERT_WEBHOOK:-}" ]; then
  payload=$(python3 - "$code" <<'PY'
import json
import os
import sys

code = int(sys.argv[1])

def load(path):
    try:
        with open(path) as fh:
            return json.load(fh)
    except Exception:
        return {}

results = load("target/run_results.json")
manifest = load("target/manifest.json")

failed = [
    r["unique_id"]
    for r in results.get("results", [])
    if r.get("status") in ("error", "gate_failed", "test_failed", "monitor_failed")
]
owners = set()
nodes = manifest.get("nodes", {})
node_items = nodes.items() if isinstance(nodes, dict) else ((n.get("unique_id"), n) for n in nodes)
for unique_id, node in node_items:
    if unique_id in failed:
        blob = json.dumps(node)
        marker = '"owner": "'
        idx = blob.find(marker)
        if idx >= 0:
            owners.add(blob[idx + len(marker) : blob.index('"', idx + len(marker))])

quality = code == 2
print(json.dumps({
    "source": "mbt-ci",
    "class": "quality-failure" if quality else "hard-error",
    "notify": "owner" if quality else "on-call",
    "owner": sorted(owners)[0] if owners else None,
    "failed_nodes": failed,
    "repo": os.environ.get("CI_REPO", ""),
    "pipeline": os.environ.get("CI_PIPELINE_NUMBER", ""),
    "event": os.environ.get("CI_PIPELINE_EVENT", ""),
}))
PY
  )
  curl -fsS -m 2 -X POST -H 'Content-Type: application/json' \
    -d "$payload" "$MBT_ALERT_WEBHOOK" || true
fi

python3 scripts/push_metrics.py . || true

exit "$code"
