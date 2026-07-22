"""Deterministic wide multi-table demo data for the showcase (SHOW-19).

The realistic churn shape (ADR-22): examples come from a monthly population
table carrying the entity crosswalk, features live in three history tables
joined by DIFFERENT keys, and the outcome is observed one calendar month
after the prediction snapshot.

    monthly_population    customer_id, safe_id, snapshot_date (month starts)
    monthly_labels        customer_id, snapshot_date (population's + 1mo), is_churn
    demographic_history   customer_id, snapshot_date, profile + filler columns
    login_history         customer_id, snapshot_date, activity + filler columns
    transaction_history   safe_id,     snapshot_date, spend + filler columns
    wide_churn_outcomes   customer_id, is_churn - the matured outcomes of the
                          newest snapshot's cohort, for `mbt monitor` (ADR-21)

demographic_history carries one NUMERIC-CODED categorical on purpose:
contract_code (int8, 0 = month-to-month ... 3 = two-year). Its churn effect
is deliberately non-monotone (highest hazard at 0, second-highest at 3), so
treating the code as a number costs real signal - the wide models' shared
hooks file (project/models/wide_hooks.py) casts it to string before
training, which is the showcase's DS-declared-categorical pattern.

Signal lives in a handful of named columns (low login/transaction activity
and rising inactivity churn); every `*_f##` filler column is pure noise, so
LightGBM gain importance demonstrably prunes them during feature selection.
Churned customers leave the population in later months, which keeps the
newest (scoring) cohort's risk mix consistent with late training months.

Dates are chosen so the showcase's pinned anchors work unchanged: with the
build/score anchor 2026-06-30 the training window covers 2025-07..2026-03,
the test window 2026-04..2026-05, and the 2026-06-01 snapshot is the
scoring batch whose labels (recorded at 2026-07-01) mature before the
monitor anchor 2026-07-20.

The default output is committed (~3000 customers, small filler counts);
regenerate only when the schema must change. ``--customers`` and
``--filler-columns`` synthesize the same shape at stress scale (the real
scenario is ~7M rows x up to 2000 columns per table) - stress output is for
local experiments and is never committed; generation is numpy-vectorized so
scale costs disk, not hours.
"""

import argparse
from datetime import datetime
from pathlib import Path

import numpy as np
import pyarrow as pa
import pyarrow.parquet as pq

OUT = Path(__file__).resolve().parent.parent / "data"
SEED = 20260717

MONTHS = [datetime(2025, m, 1) for m in range(7, 13)] + [datetime(2026, m, 1) for m in range(1, 7)]
LABEL_MONTHS = [*MONTHS[1:], datetime(2026, 7, 1)]

REGIONS = np.array(["north", "south", "east", "west", "central"])
INCOME_BANDS = np.array(["low", "mid", "upper_mid", "high"])
PLAN_TIERS = np.array(["basic", "plus", "premium"])
TOP_CATEGORIES = np.array(["grocery", "travel", "dining", "utilities", "retail", "other"])


def _filler(rng: np.random.Generator, prefix: str, count: int, rows: int) -> dict[str, np.ndarray]:
    """Pure-noise numeric columns; feature selection should discard these."""
    return {f"{prefix}_f{i:02d}": rng.normal(0.0, 1.0, rows).round(4) for i in range(count)}


def generate(customers: int, filler_columns: int, out: Path) -> None:
    rng = np.random.default_rng(SEED)

    # -- fixed per-customer profiles ------------------------------------------
    # The pool is bigger than the starting base: every month ~6% fresh
    # customers join while churners leave. Without acquisition, 12 months of
    # survivor culling would leave only low-hazard customers, and the newest
    # (scoring) cohort would carry almost no discriminable signal - realized
    # metrics then sit at the base rate no matter how good the model is.
    pool = customers * 2
    customer_id = np.arange(pool, dtype=np.int64)
    safe_id = np.array([f"sf-{i:08d}" for i in customer_id])
    age_years = rng.integers(18, 75, pool)
    region = REGIONS[rng.integers(0, len(REGIONS), pool)]
    income_band = INCOME_BANDS[rng.integers(0, len(INCOME_BANDS), pool)]
    plan_tier = PLAN_TIERS[rng.integers(0, len(PLAN_TIERS), pool)]
    top_category = TOP_CATEGORIES[rng.integers(0, len(TOP_CATEGORIES), pool)]
    # Numeric-coded contract term (0 = month-to-month ... 3 = two-year): the
    # DS-declared categorical that wide_hooks.py casts to string at train time.
    contract_code = rng.integers(0, 4, pool).astype(np.int8)
    login_base = np.clip(rng.normal(16.0, 7.0, pool), 0.5, 30.0)
    txn_base = np.clip(rng.normal(28.0, 14.0, pool), 1.0, 120.0)

    population = {k: [] for k in ("customer_id", "safe_id", "snapshot_date")}
    labels = {k: [] for k in ("customer_id", "snapshot_date", "is_churn")}
    demo_parts: list[dict] = []
    login_parts: list[dict] = []
    txn_parts: list[dict] = []

    active = np.zeros(pool, dtype=bool)
    active[:customers] = True
    next_joiner = customers
    joiners_per_month = int(customers * 0.06)
    newest_cohort: np.ndarray | None = None
    newest_outcome: np.ndarray | None = None
    for month_idx, when in enumerate(MONTHS):
        if month_idx > 0:
            fresh = customer_id[next_joiner : next_joiner + joiners_per_month]
            active[fresh] = True
            next_joiner += joiners_per_month
        idx = np.flatnonzero(active)
        n = idx.size

        # monthly activity around each base, decaying for at-risk customers
        month_noise = rng.normal(1.0, 0.28, n)
        login_days = np.clip(login_base[idx] * month_noise, 0.0, 30.0).round(1)
        days_since_login = np.clip(rng.exponential(30.0 / (1.0 + login_days)), 0.0, 30.0).round(1)
        txn_cnt = np.clip(txn_base[idx] * rng.normal(1.0, 0.22, n), 0.0, None).round(0)
        txn_amt_sum = (txn_cnt * np.clip(rng.normal(42.0, 15.0, n), 5.0, None)).round(2)

        # churn hazard: inactivity dominates; income adds a mild tilt. The
        # contrasts are deliberately strong so demo models clear their
        # PR-AUC gates comfortably (same reasoning as the monthly tables).
        # contract_code's effect is NON-monotone by design (see docstring):
        # month-to-month (0) churns most, two-year (3) second-most.
        hazard = (
            0.015
            + 0.30 * (login_days < 8.0)
            + 0.15 * (days_since_login > 10.0)
            + 0.24 * (txn_cnt < 14.0)
            + 0.03 * (income_band[idx] == "low")
            + 0.12 * (contract_code[idx] == 0)
            + 0.06 * (contract_code[idx] == 3)
        )
        churned = rng.random(n) < hazard

        population["customer_id"].append(customer_id[idx])
        population["safe_id"].append(safe_id[idx])
        population["snapshot_date"].append(np.full(n, when, dtype="datetime64[us]"))
        labels["customer_id"].append(customer_id[idx])
        labels["snapshot_date"].append(np.full(n, LABEL_MONTHS[month_idx], dtype="datetime64[us]"))
        labels["is_churn"].append(churned.astype(np.int64))

        demo_parts.append(
            {
                "customer_id": customer_id[idx],
                "snapshot_date": np.full(n, when, dtype="datetime64[us]"),
                "age_years": age_years[idx],
                "region": region[idx],
                "income_band": income_band[idx],
                "plan_tier": plan_tier[idx],
                "contract_code": contract_code[idx],
                "household_size": rng.integers(1, 6, n),
                "tenure_months": np.full(n, month_idx) + rng.integers(1, 60, n),
                **_filler(rng, "dem", filler_columns, n),
            }
        )
        login_parts.append(
            {
                "customer_id": customer_id[idx],
                "snapshot_date": np.full(n, when, dtype="datetime64[us]"),
                "login_days_30d": login_days,
                "days_since_login": days_since_login,
                "sessions_30d": (login_days * np.clip(rng.normal(2.2, 0.6, n), 0.5, None)).round(0),
                "avg_session_min": np.clip(rng.normal(9.0, 3.5, n), 0.5, None).round(1),
                **_filler(rng, "log", filler_columns, n),
            }
        )
        txn_parts.append(
            {
                "safe_id": safe_id[idx],
                "snapshot_date": np.full(n, when, dtype="datetime64[us]"),
                "txn_cnt_30d": txn_cnt,
                "txn_amt_sum_30d": txn_amt_sum,
                "txn_amt_avg_90d": np.clip(rng.normal(40.0, 12.0, n), 1.0, None).round(2),
                "merchant_diversity": np.clip(
                    (txn_cnt / 4.0) + rng.normal(0, 1.5, n), 1.0, None
                ).round(0),
                "top_category": top_category[idx],
                **_filler(rng, "txn", filler_columns, n),
            }
        )

        if when == MONTHS[-1]:
            newest_cohort = customer_id[idx]
            newest_outcome = churned.astype(np.int64)
        active[idx[churned]] = False

    def _write(name: str, parts: list[dict] | dict) -> None:
        if isinstance(parts, list):
            columns = {k: np.concatenate([p[k] for p in parts]) for k in parts[0]}
        else:
            columns = parts
        dest = out / name
        dest.mkdir(parents=True, exist_ok=True)
        pq.write_table(pa.table(columns), dest / "part-000.parquet")

    _write("monthly_population", {k: np.concatenate(v) for k, v in population.items()})
    _write("monthly_labels", {k: np.concatenate(v) for k, v in labels.items()})
    _write("demographic_history", demo_parts)
    _write("login_history", login_parts)
    _write("transaction_history", txn_parts)
    assert newest_cohort is not None and newest_outcome is not None
    _write("wide_churn_outcomes", {"customer_id": newest_cohort, "is_churn": newest_outcome})

    total = sum(len(p["customer_id"]) for p in demo_parts)
    rate = np.concatenate(labels["is_churn"]).mean()
    width = len(demo_parts[0]) + len(login_parts[0]) + len(txn_parts[0]) - 4
    print(f"population rows: {total}, churn rate: {rate:.1%}, joined feature columns: ~{width}")
    print(f"newest cohort (scoring batch {MONTHS[-1]:%Y-%m-%d}): {newest_cohort.size} customers")


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    parser.add_argument("--customers", type=int, default=3000)
    parser.add_argument(
        "--filler-columns",
        type=int,
        default=16,
        help="noise columns PER feature table (stress: hundreds; default committed: 16)",
    )
    parser.add_argument("--out", type=Path, default=OUT)
    args = parser.parse_args()
    generate(args.customers, args.filler_columns, args.out)


if __name__ == "__main__":
    main()
