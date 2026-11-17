"""Retrain the XGBoost baseline alone, on the PROD target: the deterministic workhorse.

Same unit, same cluster pushdown as mbt_retrain, without the sparkling-H2O AutoML
search. A DAG is drawn for its selection (ADR-33), so a narrower retrain is a DAG of
its own and not a run-time parameter; this is the one the scheduling tier triggers,
keeping sparkling-on-cluster out of scheduled paths (SHOW-04 owns it, DESIGN's
flake-isolation rule).
"""

import sys
from datetime import datetime
from pathlib import Path

# See mbt_retrain_dag.py for why this path is inserted.
sys.path.insert(0, str(Path(__file__).resolve().parent))
from mbt_airflow import MbtDag
from showcase_dag_utils import EXECUTION, LAKE_ANCHOR, baked, project

if baked():
    mbt_retrain_baseline = MbtDag(
        "mbt_retrain_baseline",
        project=project("prod"),
        execution=EXECUTION,
        command="build",
        select=["+churn_baseline_xgb"],
        anchor_command=LAKE_ANCHOR,
        description="mbt build --target prod, the XGBoost baseline alone",
        start_date=datetime(2026, 1, 1),
    )
