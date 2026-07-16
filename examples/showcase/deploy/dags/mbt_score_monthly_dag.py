"""Monthly batch scoring (SHOW-17): sync the lake to the local scoring
plane, then score the `tag:monthly` pipeline with its run-time production
champion - the whole cadence stays on the DuckDB plane, no cluster.

Manual-trigger in the showcase (schedule=None) so the test tier and demos
stay deterministic; wire a monthly cron here in a real deployment. The
anchor is a param for the same reason (fixed-date seed data, DESIGN.md 4.5).
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
    dag_id="mbt_score_monthly",
    description="mbt score --select tag:monthly with the run-time champion (DuckDB plane)",
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
                "tag:monthly",
                "--anchor",
                context["params"]["anchor"],
                "--deep-snapshot",
            ]
        )

    sync_lake() >> score()
