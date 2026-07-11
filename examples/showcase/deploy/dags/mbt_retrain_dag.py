"""Weekly retrain: build the `tag:weekly` models on the PROD target - real
cluster pushdown, sparkling H2O inside the executors - from a scheduled,
pinned deployable unit.

This is the cluster-from-CI path (DESIGN.md section 5 step 7): the unit
container joins the compose network, mounts the shared /workspace (ADR-17
driver-local staging), and advertises its own address as the Spark driver
host (the runner entrypoint exports SPARK_DRIVER_HOST).

A retrain that fails its gates exits 2: quality verdict, no retry, owner
notified - same routing as monitoring.
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
    dag_id="mbt_retrain",
    description="mbt build --target prod --select tag:weekly (cluster + sparkling)",
    schedule=None,
    start_date=datetime(2026, 1, 1),
    catchup=False,
    params={"anchor": ANCHOR, "select": "+churn_automl,tag:weekly"},
) as dag:

    @task(retries=1, retry_delay=timedelta(seconds=5))
    def retrain(**context) -> None:
        params = context["params"]
        run_in_unit(
            [
                "mbt",
                "build",
                "--target",
                "prod",
                "--select",
                params["select"],
                "--anchor",
                params["anchor"],
            ]
        )

    retrain()
