"""Weekly ground-truth monitoring: evaluate matured prediction runs against
arrived labels (ADR-21, exactly-once per prediction run).

Exit-code routing is the point (DESIGN.md section 5 step 7): a realized-
metric gate breach exits 2 -> AirflowFailException -> the task fails on
try 1 with NO retry (quality verdicts are deterministic) and the model
owner is notified. A hard error (exit 1) retries first, then pages on-call.

`vars` param exists so operators (and the test tier) can override gate
floors for one evaluation, e.g. {"vars": "pr_auc_floor: 0.99"}.
"""

import sys
from datetime import datetime, timedelta
from pathlib import Path

from airflow.sdk import DAG, task

# git-sync serves the dags folder through a flipping symlink; Airflow's
# DAG processor imports files under the resolved worktree path, where the
# symlinked dags folder is NOT on sys.path - anchor sibling imports here.
sys.path.insert(0, str(Path(__file__).resolve().parent))
from showcase_dag_utils import MONITOR_ANCHOR, run_in_unit

with DAG(
    dag_id="mbt_monitor",
    description="mbt monitor: realized metrics with exactly-once evaluation",
    schedule=None,
    start_date=datetime(2026, 1, 1),
    catchup=False,
    params={"anchor": MONITOR_ANCHOR, "vars": "", "target": "prod_score"},
) as dag:

    @task(retries=1, retry_delay=timedelta(seconds=5))
    def monitor(**context) -> None:
        params = context["params"]
        args = [
            "mbt",
            "monitor",
            "--target",
            params["target"],
            "--anchor",
            params["anchor"],
            "--deep-snapshot",
        ]
        if params["vars"]:
            args += ["--vars", params["vars"]]
        run_in_unit(args)

    monitor()
