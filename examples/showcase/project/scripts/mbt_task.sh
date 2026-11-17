#!/usr/bin/env bash
# mbt as one Airflow task (the showcase DAGs' mbt_executable, mbt-airflow ADR-34).
#
# Runs mbt exactly as given, then pushes that task's metrics to the Pushgateway
# from where mbt left its results, and exits with mbt's own code. The exit code
# is the contract (0 ok, 1 hard error, 2 quality verdict): Airflow fails a 2 at
# once and retries anything else, so nothing here may change it. The push is
# best-effort for the same reason - observability observes, it never fails a task.
#
# mbt-airflow tells every process where its task's state is:
#   MBT_AIRFLOW_TARGET_PATH  the task's private target directory (run_results.json)
#   MBT_AIRFLOW_MANIFEST     the run's manifest, set on node tasks only
#
# Usage: scripts/mbt_task.sh <mbt arguments...>   (run from the project directory)
set -uo pipefail

mbt "$@"
code=$?

# A compile has no node results to push.
if [ "${1:-}" != compile ]; then
  args=(--target-dir "${MBT_AIRFLOW_TARGET_PATH:-target}")
  if [ -n "${MBT_AIRFLOW_MANIFEST:-}" ]; then
    args+=(--manifest "$MBT_AIRFLOW_MANIFEST")
  fi
  python3 scripts/push_metrics.py "${args[@]}" . || true
fi

exit "$code"
