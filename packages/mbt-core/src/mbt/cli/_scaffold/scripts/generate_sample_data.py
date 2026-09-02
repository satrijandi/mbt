"""Generate deterministic sample subscriber data for the quickstart.

Usage: python scripts/generate_sample_data.py [n_rows]

The signal is a logistic model over tenure, usage, support tickets, and plan
tier. Measured on the default 5000 rows, the scaffold's XGBoost model fits it
to **0.81 ROC AUC / 0.50 PR AUC** at a ~20% base rate (the Bayes-optimal
ceiling for this generator is 0.88 / 0.65, so the demo model leaves visible
headroom rather than pretending to be perfect).

That is deliberate: this is the first model anyone sees, and an earlier
version's near-flat score function produced 0.66 ROC AUC / 0.30 PR AUC, which
made the quickstart look like the tool could not learn and made its example
gate (a 0.25 PR AUC threshold) teach nothing about what a real gate does.
"""

import sys
from datetime import datetime, timedelta
from math import exp
from pathlib import Path
from random import Random

import pyarrow as pa
import pyarrow.parquet as pq

#: Plan tier shifts the log-odds. Without this, plan_type is pure noise, yet
#: the model card slices on it - so the demo showed three identical slices.
PLAN_EFFECT = {"basic": 1.1, "pro": 0.0, "enterprise": -1.0}


def churn_probability(tenure: int, usage: float, tickets: int, plan: str) -> float:
    """P(churn) for one subscriber; shared by the training and scoring data.

    Both generators must use the SAME function, or `mbt monitor` compares
    predictions against outcomes drawn from a different world and realized
    metrics come out degenerate.
    """
    log_odds = -0.8 + tickets * 0.62 - usage / 55.0 + PLAN_EFFECT[plan] - tenure / 420.0
    return 1.0 / (1.0 + exp(-log_odds))


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
        churned = 1 if rng.random() < churn_probability(tenure, usage, tickets, plan) else 0
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
        plan = plans[rng.randrange(3)]
        batch["user_id"].append(user_id)
        batch["snapshot_date"].append(now - timedelta(days=1 + rng.randrange(6)))
        batch["is_active"].append(rng.random() > 0.05)
        batch["tenure_days"].append(tenure)
        batch["monthly_usage"].append(round(usage, 2))
        batch["support_tickets"].append(tickets)
        batch["plan_type"].append(plan)
        outcomes["user_id"].append(user_id)
        probability = churn_probability(tenure, usage, tickets, plan)
        outcomes["churned_90d"].append(1 if rng.random() < probability else 0)

    data_dir = Path(__file__).resolve().parent.parent / "data"
    for name, table in (("scoring_batch", batch), ("churn_outcomes", outcomes)):
        out = data_dir / name
        out.mkdir(parents=True, exist_ok=True)
        pq.write_table(pa.table(table), out / "part-000.parquet")
        print(f"wrote {n_rows} rows to {out / 'part-000.parquet'}")


if __name__ == "__main__":
    main(int(sys.argv[1]) if len(sys.argv) > 1 else 5000)
