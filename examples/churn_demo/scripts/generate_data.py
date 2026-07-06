"""Deterministic demo data for examples/churn_demo.

The parquet output is committed; regenerate only when the schema must change
(golden manifests hash the file bytes via --deep-snapshot).
"""

from datetime import datetime, timedelta
from pathlib import Path
from random import Random

import pyarrow as pa
import pyarrow.parquet as pq

#: Fixed so committed bytes and golden manifests stay stable.
BASE = datetime(2026, 1, 1)
N_ROWS = 2400


def main() -> None:
    rng = Random(1234)
    plans = ["basic", "pro", "enterprise"]
    rows: dict[str, list] = {
        "user_id": [],
        "snapshot_date": [],
        "is_active": [],
        "tenure_days": [],
        "monthly_usage": [],
        "support_tickets": [],
        "plan_type": [],
        "churned_90d": [],
        "upgraded_90d": [],
    }
    for i in range(N_ROWS):
        tenure = rng.randint(1, 1000)
        usage = max(0.0, rng.gauss(120, 60))
        tickets = rng.randint(0, 6)
        plan = plans[rng.randrange(3)]
        churn_p = max(0.02, min(0.9, 0.25 - usage / 1000 + tickets * 0.06 - tenure / 5000))
        upsell_p = max(0.02, min(0.8, usage / 400 + (0.08 if plan == "basic" else 0.02)))
        rows["user_id"].append(i)
        rows["snapshot_date"].append(BASE + timedelta(days=rng.randrange(180)))
        rows["is_active"].append(rng.random() > 0.05)
        rows["tenure_days"].append(tenure)
        rows["monthly_usage"].append(round(usage, 2))
        rows["support_tickets"].append(tickets)
        rows["plan_type"].append(plan)
        rows["churned_90d"].append(1 if rng.random() < churn_p else 0)
        rows["upgraded_90d"].append(1 if rng.random() < upsell_p else 0)

    out = Path(__file__).resolve().parent.parent / "data" / "subscribers"
    out.mkdir(parents=True, exist_ok=True)
    pq.write_table(pa.table(rows), out / "part-000.parquet")
    print(f"wrote {N_ROWS} rows")


if __name__ == "__main__":
    main()
