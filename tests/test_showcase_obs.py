"""Showcase observability loop (SHOW-14): run_results -> Pushgateway ->
Prometheus -> alert firing on injected shift.

Runs after the lifecycle module (alphabetical order), so a production
champion already exists; scores its own runs regardless, because the loop
under test here is metrics, not models.
"""

import time

from showcase_utils import ANCHOR, SHOWCASE_MARKS

pytestmark = SHOWCASE_MARKS


def _query(stack, promql: str) -> list:
    import urllib.parse

    url = (
        f"http://localhost:{stack.ports['SHOWCASE_PROMETHEUS_PORT']}"
        f"/api/v1/query?query={urllib.parse.quote(promql)}"
    )
    payload = stack.http_json(url)
    assert payload["status"] == "success", payload
    return payload["data"]["result"]


def _wait_for(stack, promql: str, deadline_s: int = 120) -> list:
    end = time.time() + deadline_s
    result: list = []
    while time.time() < end:
        result = _query(stack, promql)
        if result:
            return result
        time.sleep(5)
    raise AssertionError(f"no series for {promql!r} within {deadline_s}s")


def test_metrics_flow_and_alert_rules_loaded(showcase_stack) -> None:
    stack = showcase_stack

    # A scored run pushed through the exporter shows up in Prometheus with
    # the documented names and group labels.
    stack.sync_lake()
    stack.mbt("score", "--target", "prod_score", "--anchor", ANCHOR, "--deep-snapshot")
    push = stack.exec("python", "/workspace/project/scripts/push_metrics.py", "/workspace/project")
    assert "pushed" in push.stdout, push.stdout

    series = _wait_for(stack, 'mbt_node_success{command="score"}')
    assert any("retention_scoring" in s["metric"].get("node", "") for s in series), series
    _wait_for(stack, 'mbt_shift_value{monitor="feature_shift"}')

    # All four canonical rules are loaded.
    rules_payload = stack.http_json(
        f"http://localhost:{stack.ports['SHOWCASE_PROMETHEUS_PORT']}/api/v1/rules"
    )
    names = {rule["name"] for group in rules_payload["data"]["groups"] for rule in group["rules"]}
    assert {"MbtGateFailed", "MbtScheduleStale", "MbtGateNearBreach", "MbtShiftBreach"} <= names

    # Grafana is up with the provisioned dashboard.
    health = stack.http_json(f"http://localhost:{stack.ports['SHOWCASE_GRAFANA_PORT']}/api/health")
    assert health.get("database") == "ok", health


def test_injected_shift_breaches_and_alert_fires(showcase_stack) -> None:
    stack = showcase_stack

    # Poison the scoring batch, score again: mbt itself enforces (exit 2)...
    stack.exec("python", "/workspace/bootstrap/inject_drift.py")
    stack.mbt(
        "score", "--target", "prod_score", "--anchor", ANCHOR, "--deep-snapshot", expect_exit=2
    )
    scoring = stack.result_for("scoring.churn_lake.retention_scoring")
    assert scoring["status"] == "monitor_failed", scoring
    breached = [
        m
        for m in scoring["monitors"]
        if m["monitor"] == "feature_shift" and not m["passed"] and m["value"] is not None
    ]
    assert breached, scoring["monitors"]

    # ...and observability observes: the pushed breach fires MbtShiftBreach.
    stack.exec("python", "/workspace/project/scripts/push_metrics.py", "/workspace/project")
    _wait_for(stack, "mbt_shift_value >= mbt_shift_threshold")
    firing = _wait_for(stack, 'ALERTS{alertname="MbtShiftBreach"}', deadline_s=120)
    assert firing, "MbtShiftBreach never entered pending/firing"

    # Restore clean data for anyone poking at the stack after the tests.
    stack.sync_lake()
