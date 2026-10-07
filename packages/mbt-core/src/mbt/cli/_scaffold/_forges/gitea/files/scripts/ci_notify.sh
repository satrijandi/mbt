#!/usr/bin/env bash
# The pipelines' two outbound signals, in one place:
#
#   ci_notify.sh heartbeat        ping MBT_HEARTBEAT_URL (a cron monitor such
#                                 as healthchecks.io or Cronitor): a schedule
#                                 that silently stops is caught by the
#                                 missing ping
#   ci_notify.sh alert "TEXT"     POST {"text": "TEXT: <pipeline url>"} to
#                                 MBT_ALERT_WEBHOOK (Slack, Teams, PagerDuty)
#
# Woodpecker refuses to start a pipeline whose step names a secret that does
# not exist, so these secrets always exist; the value `disabled` (or an empty
# one) turns the signal off instead. A failed ping or post never fails the
# pipeline: the dead-man's-switch alerts on the missing ping anyway.

set -uo pipefail

kind="${1:?usage: ci_notify.sh heartbeat|alert TEXT}"
url_of() { case "${1:-}" in ""|disabled) echo "" ;; *) echo "$1" ;; esac; }
run_url="${CI_PIPELINE_URL:-${CI_PIPELINE_FORGE_URL:-}}"

case "$kind" in
  heartbeat)
    url="$(url_of "${MBT_HEARTBEAT_URL:-}")"
    [ -z "$url" ] && { echo "mbt_heartbeat_url disabled; skipping heartbeat"; exit 0; }
    curl -fsS -m 10 --retry 3 "$url" >/dev/null \
      || echo "heartbeat ping failed; the dead-man's-switch will alert on the missing ping"
    ;;
  alert)
    url="$(url_of "${MBT_ALERT_WEBHOOK:-}")"
    [ -z "$url" ] && { echo "mbt_alert_webhook disabled; skipping alert"; exit 0; }
    text="${2:?usage: ci_notify.sh alert TEXT}: ${run_url}"
    payload=$(python3 -c 'import json, sys; print(json.dumps({"text": sys.argv[1]}))' "$text")
    curl -fsS -m 10 -X POST -H 'Content-Type: application/json' -d "$payload" "$url" >/dev/null \
      || echo "alert post failed"
    ;;
  *)
    echo "ci_notify.sh: unknown signal $kind (heartbeat|alert)" >&2
    exit 1
    ;;
esac
