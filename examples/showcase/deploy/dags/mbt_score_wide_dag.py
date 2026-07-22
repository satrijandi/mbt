"""Batch-monthly wide scoring (SHOW-20): sync the lake, score the
`tag:wide` pipeline with its run-time production champion, then run the
Evidently serving-phase stability gate on the scored batch.

Manual-trigger in the showcase (schedule=None) so the test tier and demos
stay deterministic; in a real deployment wire `schedule="0 0 1 * *"` here -
the wide cadence predicts every 1st of the month at 00:00, matching the
month-start population snapshots. The anchor is a param for the same
reason (fixed-date seed data, DESIGN.md 4.5).

Task containers are ephemeral (`run_in_unit` starts a fresh unit per task
and `/app/project/target` dies with it), so the score task copies the
scoring batch out to the mounted /workspace/monitoring/ in the SAME
container, and the gate compares it against the reference exported there
by the train-phase gate (`make wide` or the CI flow). Gate exit 2 is a
quality verdict: AirflowFailException, model owner notified, no retries.
"""

import sys
from datetime import datetime, timedelta
from pathlib import Path

from airflow.sdk import DAG, task

# git-sync serves the dags folder through a flipping symlink; Airflow's
# DAG processor imports files under the resolved worktree path, where the
# symlinked dags folder is NOT on sys.path - anchor sibling imports here.
sys.path.insert(0, str(Path(__file__).resolve().parent))
from showcase_dag_utils import ANCHOR, run_in_unit

with DAG(
    dag_id="mbt_score_wide",
    description="mbt score --select tag:wide + evidently serving gate (batch-monthly cadence)",
    schedule=None,
    start_date=datetime(2026, 1, 1),
    catchup=False,
    params={"anchor": ANCHOR},
) as dag:

    @task(retries=1, retry_delay=timedelta(seconds=5))
    def sync_lake() -> None:
        run_in_unit(["python3", "/workspace/bootstrap/sync_lake.py"])

    @task(retries=1, retry_delay=timedelta(seconds=5))
    def score(**context) -> None:
        # The copy-out chains in the same container because target/ is
        # ephemeral; a fresh unit has exactly one run-key dir to glob.
        run_in_unit(
            [
                "bash",
                "-c",
                "mbt score --target prod_score --select tag:wide "
                f"--anchor {context['params']['anchor']} --deep-snapshot "
                "&& mkdir -p /workspace/monitoring "
                "&& cp target/scoring_inputs/wide_retention_scoring/*/score.parquet "
                "/workspace/monitoring/wide_current.parquet",
            ]
        )

    @task(retries=1, retry_delay=timedelta(seconds=5))
    def drift_gate() -> None:
        run_in_unit(
            [
                "python3",
                "scripts/evidently_gate.py",
                "--phase",
                "serving",
                "--reference",
                "/workspace/monitoring/wide_reference.parquet",
                "--current",
                "/workspace/monitoring/wide_current.parquet",
                "--out",
                "/workspace/monitoring/wide_drift_report.html",
            ]
        )

    sync_lake() >> score() >> drift_gate()
