"""Retrain: build the models on the PROD target - real cluster pushdown, sparkling H2O
inside the executors - from a scheduled, pinned deployable unit.

This is the cluster-from-CI path (DESIGN.md section 5 step 7): every task's container
joins the compose network, mounts the shared /workspace (ADR-17 driver-local staging),
and advertises its own address as the Spark driver host (the runner entrypoint exports
SPARK_DRIVER_HOST).

A node is a group of steps (ADR-33, ADR-35), each its own task with its own state, log
and retry: the dataset is materialized and then checked, and the model is trained, judged
against its gates, and registered. A registry blip after a long training retries the
`register` step alone, and a failed gate is the `gate` step's verdict.

A retrain that fails its gates exits 2: quality verdict, no retry, owner notified -
same routing as monitoring.
"""

import sys
from datetime import datetime
from pathlib import Path

# git-sync serves the dags folder through a flipping symlink; Airflow's
# DAG processor imports files under the resolved worktree path, where the
# symlinked dags folder is NOT on sys.path - anchor sibling imports here.
sys.path.insert(0, str(Path(__file__).resolve().parent))
from mbt_airflow import MbtDag
from showcase_dag_utils import EXECUTION, LAKE_ANCHOR, baked, project

if baked():
    mbt_retrain = MbtDag(
        "mbt_retrain",
        project=project("prod"),
        execution=EXECUTION,
        command="build",
        select=["+churn_automl"],
        anchor_command=LAKE_ANCHOR,
        description="mbt build --target prod (cluster + sparkling)",
        start_date=datetime(2026, 1, 1),
    )
