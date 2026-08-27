"""Generate deterministic sample subscriber data for the quickstart.

Usage: python scripts/generate_sample_data.py [n_rows]
"""

import sys
from datetime import datetime, timedelta
from pathlib import Path
from random import Random

import pyarrow as pa
import pyarrow.parquet as pq


def main(n_rows: int = 5000) -> None:
    rng = Random(7)
    now = datetime.now().replace(hour=0, minute=0, second=0, microsecond=0)
    base = now - timedelta(days=200)
    plans = ["basic", "pro", "enterprise"]

    rows: dict[str, list[object]] = {
        "user_id": [],
        "snapshot_date": [],
        "is_active": [],
        "tenure_days": [],
        "monthly_usage": [],
        "support_tickets": [],
        "plan_type": [],
        "churned_90d": [],
    }
    for i in range(n_rows):
        tenure = rng.randint(1, 1000)
        usage = max(0.0, rng.gauss(120, 60))
        tickets = rng.randint(0, 6)
        plan = plans[rng.randrange(3)]
        # churn is likelier for low usage, short tenure, many tickets
        churn_score = 0.25 - usage / 1000 + tickets * 0.06 - tenure / 5000
        churned = 1 if rng.random() < max(0.02, min(0.9, churn_score)) else 0
        rows["user_id"].append(i)
        rows["snapshot_date"].append(base + timedelta(days=rng.randrange(200)))
        rows["is_active"].append(rng.random() > 0.05)
        rows["tenure_days"].append(tenure)
        rows["monthly_usage"].append(round(usage, 2))
        rows["support_tickets"].append(tickets)
        rows["plan_type"].append(plan)
        rows["churned_90d"].append(churned)

    out = Path(__file__).resolve().parent.parent / "data" / "subscribers"
    out.mkdir(parents=True, exist_ok=True)
    pq.write_table(pa.table(rows), out / "part-000.parquet")
    print(f"wrote {n_rows} rows to {out / 'part-000.parquet'}")

    generate_scoring_data(now, plans, max(n_rows // 10, 50))


def generate_scoring_data(now: datetime, plans: list[str], n_rows: int) -> None:
    """A fresh unlabeled batch (last 7 days) plus its future outcomes.

    Separate RNG so the training data above stays unchanged; outcomes share
    the same generative signal so realized metrics are non-degenerate.
    """
    rng = Random(11)
    batch: dict[str, list[object]] = {
        "user_id": [],
        "snapshot_date": [],
        "is_active": [],
        "tenure_days": [],
        "monthly_usage": [],
        "support_tickets": [],
        "plan_type": [],
    }
    outcomes: dict[str, list[object]] = {"user_id": [], "churned_90d": []}
    for i in range(n_rows):
        user_id = 100_000 + i
        tenure = rng.randint(1, 1000)
        usage = max(0.0, rng.gauss(120, 60))
        tickets = rng.randint(0, 6)
        churn_score = 0.25 - usage / 1000 + tickets * 0.06 - tenure / 5000
        batch["user_id"].append(user_id)
        batch["snapshot_date"].append(now - timedelta(days=1 + rng.randrange(6)))
        batch["is_active"].append(rng.random() > 0.05)
        batch["tenure_days"].append(tenure)
        batch["monthly_usage"].append(round(usage, 2))
        batch["support_tickets"].append(tickets)
        batch["plan_type"].append(plans[rng.randrange(3)])
        outcomes["user_id"].append(user_id)
        outcomes["churned_90d"].append(1 if rng.random() < max(0.02, min(0.9, churn_score)) else 0)

    data_dir = Path(__file__).resolve().parent.parent / "data"
    for name, table in (("scoring_batch", batch), ("churn_outcomes", outcomes)):
        out = data_dir / name
        out.mkdir(parents=True, exist_ok=True)
        pq.write_table(pa.table(table), out / "part-000.parquet")
        print(f"wrote {n_rows} rows to {out / 'part-000.parquet'}")


if __name__ == "__main__":
    main(int(sys.argv[1]) if len(sys.argv) > 1 else 5000)
