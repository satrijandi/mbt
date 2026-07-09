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
N_ROWS = 6000


def main() -> None:
    rng = Random(1234)
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
        # noise features: no label effect by construction; models must cope
        # with irrelevant numeric and categorical inputs
        "weekly_logins": [],
        "signup_channel": [],
        # deliberate leakage, shipped as a teaching asset: account_status is
        # measured AFTER the 90-day outcome and literally encodes the label.
        # The always-on label_leakage_scan catches it at V=1.000 the moment
        # the dataset's reviewed exclusion is removed (see the spec).
        "account_status": [],
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
        churned = 1 if rng.random() < churn_p else 0
        rows["user_id"].append(i)
        rows["snapshot_date"].append(BASE + timedelta(days=rng.randrange(180)))
        rows["is_active"].append(rng.random() > 0.05)
        rows["tenure_days"].append(tenure)
        rows["monthly_usage"].append(round(usage, 2))
        rows["support_tickets"].append(tickets)
        rows["plan_type"].append(plan)
        rows["weekly_logins"].append(rng.randint(0, 40))
        rows["signup_channel"].append(channels[rng.randrange(4)])
        rows["account_status"].append("cancelled" if churned else "active")
        rows["churned_90d"].append(churned)
        rows["upgraded_90d"].append(1 if rng.random() < upsell_p else 0)

    out = Path(__file__).resolve().parent.parent / "data" / "subscribers"
    out.mkdir(parents=True, exist_ok=True)
    pq.write_table(pa.table(rows), out / "part-000.parquet")
    print(f"wrote {N_ROWS} rows")

    generate_scoring_data(plans, channels)


def generate_scoring_data(plans: list[str], channels: list[str], n_rows: int = 800) -> None:
    """A fresh unlabeled batch in the week before DEMO_ANCHOR (2026-06-30),
    plus its future outcomes for `mbt monitor` (ADR-21).

    Independent RNG: the committed subscribers bytes stay untouched. The
    outcomes share the generative churn signal, so realized metrics are
    non-degenerate. No account_status column: at scoring time the
    post-outcome status does not exist yet (it is the demo's teaching leak).
    """
    rng = Random(5678)
    scoring_base = datetime(2026, 6, 23)  # DEMO_ANCHOR minus 7 days
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
    outcomes: dict[str, list] = {"user_id": [], "churned_90d": []}
    for i in range(n_rows):
        user_id = 100_000 + i
        tenure = rng.randint(1, 1000)
        usage = max(0.0, rng.gauss(120, 60))
        tickets = rng.randint(0, 6)
        churn_p = max(0.02, min(0.9, 0.25 - usage / 1000 + tickets * 0.06 - tenure / 5000))
        batch["user_id"].append(user_id)
        batch["snapshot_date"].append(scoring_base + timedelta(days=rng.randrange(7)))
        batch["is_active"].append(rng.random() > 0.05)
        batch["tenure_days"].append(tenure)
        batch["monthly_usage"].append(round(usage, 2))
        batch["support_tickets"].append(tickets)
        batch["plan_type"].append(plans[rng.randrange(3)])
        batch["weekly_logins"].append(rng.randint(0, 40))
        batch["signup_channel"].append(channels[rng.randrange(4)])
        outcomes["user_id"].append(user_id)
        outcomes["churned_90d"].append(1 if rng.random() < churn_p else 0)

    data_dir = Path(__file__).resolve().parent.parent / "data"
    for name, table in (("scoring_batch", batch), ("churn_outcomes", outcomes)):
        out = data_dir / name
        out.mkdir(parents=True, exist_ok=True)
        pq.write_table(pa.table(table), out / "part-000.parquet")
        print(f"wrote {n_rows} rows to {name}")


if __name__ == "__main__":
    main()
