"""Deterministic monthly-cadence demo data for the showcase (SHOW-17).

Writes ../data/monthly_{subscribers,scoring_batch,churn_outcomes}: month-start
snapshots (2025-07-01 .. 2026-06-01) of ~2500 subscribers with a learnable
30-day churn signal (low usage, many tickets, short tenure => churn), a fresh
unlabeled batch at the newest month start, and its realized outcomes for
`mbt monitor`.
The parquet output is committed; regenerate only when the schema must change.

Dates are chosen so the showcase's pinned anchors work unchanged: with the
build/score anchor 2026-06-30 the training split covers the ten oldest
snapshots and evaluates on the two freshest, the scoring window catches
exactly the 2026-06-01 batch, and its 30-day labels mature before the
monitor anchor 2026-07-20.

Schema mirrors examples/churn_demo minus the teaching-leak column
(account_status); the monthly path carries no leak on purpose - one teaching
asset in the project is enough, and the always-on `label_leakage_scan`
guards this dataset without needing a reviewed exclusion.
"""

from datetime import datetime
from pathlib import Path
from random import Random

import pyarrow as pa
import pyarrow.parquet as pq

OUT = Path(__file__).resolve().parent.parent / "data"
PLANS = ["basic", "pro", "enterprise"]
CHANNELS = ["organic", "paid", "referral", "partner"]
N_USERS = 2500
N_SCORING = 700

MONTHS = [(2025, m) for m in range(7, 13)] + [(2026, m) for m in range(1, 7)]


def churn_probability(usage: float, tickets: int, tenure: int) -> float:
    # Deliberately strong contrasts: a demo model must clear its PR-AUC gate
    # comfortably, so tickets and low usage dominate the noise.
    p = 0.01 + tickets * 0.10 + max(0.0, (110 - usage)) / 220 - tenure / 15000
    return max(0.005, min(0.95, p))


def user_profile(rng: Random) -> dict:
    return {
        "plan": PLANS[rng.randrange(3)],
        "channel": CHANNELS[rng.randrange(4)],
        "base_usage": max(5.0, rng.gauss(120, 55)),
        "tenure0": rng.randint(30, 900),
    }


def snapshot_row(rng: Random, uid: int, prof: dict, when: datetime, months_in: int) -> dict:
    usage = max(0.0, rng.gauss(prof["base_usage"], 25))
    tickets = min(8, max(0, int(rng.gauss(1.2 + (100 - usage) / 80, 1.2))))
    tenure = prof["tenure0"] + months_in * 30
    return {
        "user_id": uid,
        "inference_date": when,
        "is_active": rng.random() > 0.04,
        "tenure_days": tenure,
        "monthly_usage": round(usage, 2),
        "support_tickets": tickets,
        "plan_type": prof["plan"],
        "weekly_logins": max(0, int(usage / 10 + rng.gauss(0, 3))),
        "signup_channel": prof["channel"],
        "_churn_p": churn_probability(usage, tickets, tenure),
    }


def write_table(rows: list[dict], table: str) -> None:
    dest = OUT / table
    dest.mkdir(parents=True, exist_ok=True)
    pq.write_table(pa.table({k: [r[k] for r in rows] for k in rows[0]}), dest / "part-000.parquet")


def main() -> None:
    rng = Random(20260716)

    # -- monthly training snapshots ------------------------------------------
    profiles = {uid: user_profile(rng) for uid in range(N_USERS)}
    churned_at: dict[int, int] = {}  # uid -> month index it churned in
    rows: list[dict] = []
    for m_idx, (year, month) in enumerate(MONTHS):
        when = datetime(year, month, 1)
        for uid, prof in profiles.items():
            if uid in churned_at and churned_at[uid] < m_idx:
                continue  # left the base in an earlier month
            row = snapshot_row(rng, uid, prof, when, m_idx)
            churned = rng.random() < row.pop("_churn_p")
            row["churned_30d"] = int(churned)
            if churned:
                churned_at[uid] = m_idx
            rows.append(row)
    write_table(rows, "monthly_subscribers")
    positives = sum(r["churned_30d"] for r in rows)
    print(f"monthly_subscribers: {len(rows)} rows, {positives / len(rows):.1%} churn rate")

    # -- fresh month-start scoring batch + its future outcomes ----------------
    batch_when = datetime(2026, 6, 1)
    batch_rows, outcome_rows = [], []
    uid = 10_000
    # The batch must look like the population the champion saw, on BOTH
    # monitored axes (each tripped a monitor when violated during design):
    # - risk mix: survivors of ~10 months, like the freshest training
    #   snapshots - fresh unconditioned profiles are riskier and shifted the
    #   score distribution (PSI 0.44);
    # - tenure support: months_in 0..9 like the train split - an
    #   all-at-max-tenure batch shifted tenure_days (PSI 1.85).
    while len(batch_rows) < N_SCORING:
        prof = user_profile(rng)
        survived = True
        for m_idx in range(10):
            probe = snapshot_row(rng, uid, prof, batch_when, m_idx)
            if rng.random() < probe["_churn_p"]:
                survived = False
                break
        uid += 1
        if not survived:
            continue
        row = snapshot_row(rng, uid - 1, prof, batch_when, rng.randrange(10))
        churn_p = row.pop("_churn_p")
        batch_rows.append(row)
        outcome_rows.append({"user_id": uid - 1, "churned_30d": int(rng.random() < churn_p)})
    write_table(batch_rows, "monthly_scoring_batch")
    write_table(outcome_rows, "monthly_churn_outcomes")
    outcome_rate = sum(r["churned_30d"] for r in outcome_rows) / len(outcome_rows)
    print(f"monthly_scoring_batch: {len(batch_rows)} rows @ {batch_when:%Y-%m-%d}")
    print(f"monthly_churn_outcomes: {len(outcome_rows)} rows, {outcome_rate:.1%} churned")


if __name__ == "__main__":
    main()
