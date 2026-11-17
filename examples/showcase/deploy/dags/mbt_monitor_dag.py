"""Ground-truth monitoring: evaluate matured prediction runs against
arrived labels (ADR-21, exactly-once per prediction run).

Exit-code routing is the point (DESIGN.md section 5 step 7): a realized-
metric gate breach exits 2 -> the task fails on try 1 with NO retry (quality
verdicts are deterministic) and the model owner is notified. A hard error
(exit 1) retries first, then pages on-call.

The `vars` param exists so operators (and the test tier) can override gate
floors for one evaluation, e.g. {"vars": "pr_auc_floor: 0.99"}: it goes to the
compile task, and every task after it runs that manifest.
"""

import sys
from datetime import datetime
from pathlib import Path

# See mbt_retrain_dag.py for why this path is inserted.
sys.path.insert(0, str(Path(__file__).resolve().parent))
from mbt_airflow import MbtDag
from showcase_dag_utils import EXECUTION, LAKE_ANCHOR_MONITOR, baked, project

if baked():
    mbt_monitor = MbtDag(
        "mbt_monitor",
        project=project("batch"),
        execution=EXECUTION,
        command="monitor",
        anchor_command=LAKE_ANCHOR_MONITOR,
        description="mbt monitor: realized metrics with exactly-once evaluation",
        start_date=datetime(2026, 1, 1),
    )
