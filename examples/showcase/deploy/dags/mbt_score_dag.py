"""Batch scoring: score the newest cohort of the lake table with the run-time
production champion, straight off the object store on the cluster-free
`batch` target (ADR-20: a promotion changes the NEXT run of this DAG, zero
redeploy).

Manual-trigger in the showcase (schedule=None) so the test tier and demos
stay deterministic; wire a cron here in a real deployment. The anchor
defaults to the lake's own (the day after its newest cohort lands); set the
param to pin one.
"""

import sys
from datetime import datetime, timedelta
from pathlib import Path

from airflow.sdk import DAG, task

# git-sync serves the dags folder through a flipping symlink; Airflow's
# DAG processor imports files under the resolved worktree path, where the
# symlinked dags folder is NOT on sys.path - anchor sibling imports here.
sys.path.insert(0, str(Path(__file__).resolve().parent))
from showcase_dag_utils import anchor_arg, run_in_unit

with DAG(
    dag_id="mbt_score",
    description="mbt score --target batch with the run-time champion",
    schedule=None,
    start_date=datetime(2026, 1, 1),
    catchup=False,
    params={"anchor": ""},
) as dag:

    @task(retries=1, retry_delay=timedelta(seconds=5))
    def score(**context) -> None:
        run_in_unit(
            ["mbt", "score", "--target", "batch"],
            anchor=anchor_arg(context["params"]["anchor"]),
        )

    score()
