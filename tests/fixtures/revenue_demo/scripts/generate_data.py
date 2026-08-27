"""Deterministic demo data for tests/fixtures/revenue_demo (the regression vertical).

A spend-forecast regression twin of churn_demo: the target `spend_next_30d` is
a continuous dollar amount driven by a strong, learnable signal (usage, plan,
tenure, support load) plus Gaussian noise, so an XGBoost regressor comfortably
beats the rmse gate while a mean predictor does not.

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
N_ROWS = 6000

#: Monthly revenue premium per plan tier - a secondary, categorical signal.
PLAN_BONUS = {"basic": 0.0, "pro": 6.0, "enterprise": 14.0}


def spend_next_30d(rng: Random, usage: float, tenure: int, tickets: int, plan: str) -> float:
    """The generative model of next-30-day spend: a linear signal + noise.

    The dominant drivers are numeric (usage, tenure, support load) so the demo
    is robust to how any adapter encodes the categorical plan tier; a mean
    predictor scores rmse ~35 (the target's own spread) while a real regressor
    reaches ~5, so the `rmse_ceiling: 12` gate is meaningful.
    """
    signal = 10.0 + 0.5 * usage + 0.06 * tenure - 3.0 * tickets + PLAN_BONUS[plan]
    return round(max(0.0, signal + rng.gauss(0.0, 4.0)), 2)


def main() -> None:
    rng = Random(24601)
    plans = ["basic", "pro", "enterprise"]
    channels = ["organic", "paid", "referral", "partner"]
    rows: dict[str, list] = {
        "user_id": [],
        "snapshot_date": [],
        "is_active": [],
        "tenure_days": [],
        "monthly_usage": [],
        "support_tickets": [],
        "plan_type": [],
        # noise features: no target effect by construction; the model must cope
        # with irrelevant numeric and categorical inputs
        "weekly_logins": [],
        "signup_channel": [],
        "spend_next_30d": [],
    }
    for i in range(N_ROWS):
        tenure = rng.randint(1, 1000)
        usage = max(0.0, rng.gauss(120, 60))
        tickets = rng.randint(0, 6)
        plan = plans[rng.randrange(3)]
        rows["user_id"].append(i)
        rows["snapshot_date"].append(BASE + timedelta(days=rng.randrange(180)))
        rows["is_active"].append(rng.random() > 0.05)
        rows["tenure_days"].append(tenure)
        rows["monthly_usage"].append(round(usage, 2))
        rows["support_tickets"].append(tickets)
        rows["plan_type"].append(plan)
        rows["weekly_logins"].append(rng.randint(0, 40))
        rows["signup_channel"].append(channels[rng.randrange(4)])
        rows["spend_next_30d"].append(spend_next_30d(rng, usage, tenure, tickets, plan))

    out = Path(__file__).resolve().parent.parent / "data" / "subscribers"
    out.mkdir(parents=True, exist_ok=True)
    pq.write_table(pa.table(rows), out / "part-000.parquet")
    print(f"wrote {N_ROWS} rows")

    generate_scoring_data(plans, channels)


def generate_scoring_data(plans: list[str], channels: list[str], n_rows: int = 800) -> None:
    """A fresh unlabeled batch in the week before REVENUE_ANCHOR (2026-06-30),
    plus its realized spend for `mbt monitor` (ADR-21).

    Independent RNG so the committed subscribers bytes stay untouched. The
    outcomes share the generative spend signal, so realized regression metrics
    are non-degenerate.
    """
    rng = Random(13579)
    scoring_base = datetime(2026, 6, 23)  # REVENUE_ANCHOR minus 7 days
    batch: dict[str, list] = {
        "user_id": [],
        "snapshot_date": [],
        "is_active": [],
        "tenure_days": [],
        "monthly_usage": [],
        "support_tickets": [],
        "plan_type": [],
        "weekly_logins": [],
        "signup_channel": [],
    }
    outcomes: dict[str, list] = {"user_id": [], "spend_next_30d": []}
    for i in range(n_rows):
        user_id = 100_000 + i
        tenure = rng.randint(1, 1000)
        usage = max(0.0, rng.gauss(120, 60))
        tickets = rng.randint(0, 6)
        plan = plans[rng.randrange(3)]
        batch["user_id"].append(user_id)
        batch["snapshot_date"].append(scoring_base + timedelta(days=rng.randrange(7)))
        batch["is_active"].append(rng.random() > 0.05)
        batch["tenure_days"].append(tenure)
        batch["monthly_usage"].append(round(usage, 2))
        batch["support_tickets"].append(tickets)
        batch["plan_type"].append(plan)
        batch["weekly_logins"].append(rng.randint(0, 40))
        batch["signup_channel"].append(channels[rng.randrange(4)])
        outcomes["user_id"].append(user_id)
        outcomes["spend_next_30d"].append(spend_next_30d(rng, usage, tenure, tickets, plan))

    data_dir = Path(__file__).resolve().parent.parent / "data"
    for name, table in (("scoring_batch", batch), ("spend_outcomes", outcomes)):
        out = data_dir / name
        out.mkdir(parents=True, exist_ok=True)
        pq.write_table(pa.table(table), out / "part-000.parquet")
        print(f"wrote {n_rows} rows to {name}")


if __name__ == "__main__":
    main()
