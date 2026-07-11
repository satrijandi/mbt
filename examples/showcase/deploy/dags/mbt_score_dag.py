"""Daily batch scoring: sync the lake to the local scoring plane, then
score every `tag:daily` pipeline with its run-time production champion
(ADR-20: a promotion changes the NEXT run of this DAG, zero redeploy).

Manual-trigger in the showcase (schedule=None) so the test tier and demos
stay deterministic; wire a cron here in a real deployment. The anchor is a
param for the same reason (fixed-date seed data, DESIGN.md 4.5).
"""

import sys
from datetime import datetime, timedelta
from pathlib import Path

from airflow.decorators import task
from airflow.models.dag import DAG

# git-sync serves the dags folder through a flipping symlink; Airflow's
# DAG processor imports files under the resolved worktree path, where the
# symlinked dags folder is NOT on sys.path - anchor sibling imports here.
sys.path.insert(0, str(Path(__file__).resolve().parent))
from showcase_dag_utils import ANCHOR, run_in_unit

with DAG(
    dag_id="mbt_score",
    description="mbt score --select tag:daily with the run-time champion",
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
        run_in_unit(
            [
                "mbt",
                "score",
                "--target",
                "prod_score",
                "--select",
                "tag:daily",
                "--anchor",
                context["params"]["anchor"],
                "--deep-snapshot",
            ]
        )

    sync_lake() >> score()
