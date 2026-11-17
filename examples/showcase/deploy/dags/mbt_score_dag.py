"""Batch scoring: score the newest cohort of the lake table with the run-time
production champion, straight off the object store on the cluster-free
`batch` target (ADR-20: a promotion changes the NEXT run of this DAG, zero
redeploy).

A scoring node reads the registry's champion when it runs, so this DAG has no edge
to a training task (ADR-33): a daily score never waits for a weekly retrain.

Manual-trigger in the showcase (schedule=None) so the test tier and demos
stay deterministic; wire a cron here in a real deployment. The anchor
defaults to the lake's own (the day after its newest cohort lands); set the
param to pin one.
"""

import sys
from datetime import datetime
from pathlib import Path

# See mbt_retrain_dag.py for why this path is inserted.
sys.path.insert(0, str(Path(__file__).resolve().parent))
from mbt_airflow import MbtDag
from showcase_dag_utils import EXECUTION, LAKE_ANCHOR, baked, project

if baked():
    mbt_score = MbtDag(
        "mbt_score",
        project=project("batch"),
        execution=EXECUTION,
        command="score",
        anchor_command=LAKE_ANCHOR,
        description="mbt score --target batch with the run-time champion",
        start_date=datetime(2026, 1, 1),
    )
