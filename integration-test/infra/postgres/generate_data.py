#!/usr/bin/env python3
"""
Generate CSV seed data for the MBT integration-test warehouse database.

Tables:
  - customers_to_score: customer_id, snapshot_date
  - features_a: customer_id, snapshot_date, feature_a .. feature_z  (26 features)
  - features_b: customer_id, snapshot_date, feature_1 .. feature_1000 (1000 features)
  - labels: customer_id, snapshot_date, is_churn

Data models a realistic telecom churn prediction scenario:
  - snapshot_date = monthly cutoff / prediction date
  - is_churn = whether customer churns within next 1 month
  - 20 customers x 3 monthly snapshots (historical) + 10 scoring customers
  - ~18% churn rate (realistic for telecom)

Usage:
  python3 generate_data.py --output-dir /tmp/mbt-data
"""

import argparse
import csv
import os
import random
import sys
from datetime import date
from pathlib import Path

random.seed(42)

# ── Configuration ──────────────────────────────────────────────────────────────
NUM_CUSTOMERS = 20
SNAPSHOT_DATES = [
    date(2025, 10, 1),
    date(2025, 11, 1),
    date(2025, 12, 1),
]
SCORE_SNAPSHOT = date(2026, 1, 1)
NUM_SCORE_CUSTOMERS = 10

FEATURE_A_COLS = [f"feature_{chr(ord('a') + i)}" for i in range(26)]
FEATURE_B_COLS = [f"feature_{i}" for i in range(1, 1001)]


# ── Customer profile generation ────────────────────────────────────────────────

def generate_customer_profile(cid: int) -> dict:
    """Generate a stable customer profile that drives feature generation."""
    tenure_months = random.randint(1, 72)
    plan_tier = random.choice(["basic", "standard", "premium", "enterprise"])
    tier_charge = {"basic": 30, "standard": 55, "premium": 85, "enterprise": 120}[plan_tier]
    monthly_charge = round(tier_charge + random.gauss(0, 5), 2)
    monthly_charge = max(20, monthly_charge)
    age = random.randint(18, 75)
    num_lines = random.randint(1, 5)
    has_international = random.random() < 0.3
    has_streaming = random.random() < 0.5
    has_device_protection = random.random() < 0.4
    contract_type = random.choices(
        ["month-to-month", "one-year", "two-year"],
        weights=[0.5, 0.3, 0.2],
    )[0]
    payment_method = random.choices(
        ["credit_card", "bank_transfer", "electronic_check", "mailed_check"],
        weights=[0.35, 0.25, 0.25, 0.15],
    )[0]

    # Base churn probability driven by realistic risk factors
    churn_prob = 0.15
    if contract_type == "month-to-month":
        churn_prob += 0.14
    if tenure_months < 6:
        churn_prob += 0.15
    elif tenure_months < 12:
        churn_prob += 0.07
    elif tenure_months > 36:
        churn_prob -= 0.05
    if payment_method == "electronic_check":
        churn_prob += 0.08
    if monthly_charge > 80:
        churn_prob += 0.06
    if has_device_protection:
        churn_prob -= 0.03
    if has_streaming:
        churn_prob -= 0.02
    if num_lines > 2:
        churn_prob -= 0.03
    churn_prob = max(0.03, min(0.65, churn_prob))

    return {
        "customer_id": f"CUST_{cid:05d}",
        "tenure_months": tenure_months,
        "plan_tier": plan_tier,
        "monthly_charge": monthly_charge,
        "age": age,
        "num_lines": num_lines,
        "has_international": has_international,
        "has_streaming": has_streaming,
        "has_device_protection": has_device_protection,
        "contract_type": contract_type,
        "payment_method": payment_method,
        "churn_prob": churn_prob,
    }


def generate_features_a(profile: dict, snapshot: date) -> list[float]:
    """
    Generate 26 features (a-z) representing demographic/account attributes.

    Mapping:
      a=tenure_months, b=monthly_charge, c=total_charges, d=age,
      e=num_lines, f=has_international(0/1), g=has_streaming(0/1),
      h=has_device_protection(0/1), i=contract_months_remaining,
      j=payment_risk_score, k=avg_monthly_usage_gb, l=peak_usage_gb,
      m=off_peak_usage_gb, n=support_tickets_last_30d, o=days_since_last_ticket,
      p=avg_call_duration_min, q=num_calls_last_30d, r=sms_count_last_30d,
      s=data_overage_count, t=late_payment_count, u=discount_pct,
      v=referral_count, w=satisfaction_score(1-10), x=nps_score(-100..100),
      y=account_age_days, z=monthly_arpu
    """
    noise = lambda scale=1.0: random.gauss(0, scale)
    month_offset = (snapshot.year - 2025) * 12 + snapshot.month - 10

    tenure = profile["tenure_months"] + month_offset
    monthly = profile["monthly_charge"]
    total = round(monthly * tenure + noise(50), 2)
    total = max(monthly, total)

    contract_map = {"month-to-month": 0, "one-year": 12, "two-year": 24}
    contract_remaining = max(0, contract_map[profile["contract_type"]] - month_offset)

    payment_risk = {"credit_card": 0.1, "bank_transfer": 0.2,
                    "electronic_check": 0.6, "mailed_check": 0.4}[profile["payment_method"]]
    payment_risk = round(payment_risk + noise(0.05), 3)
    payment_risk = max(0, min(1, payment_risk))

    usage_base = 15 + profile["num_lines"] * 8 + (20 if profile["has_streaming"] else 0)
    avg_usage = round(usage_base + noise(3), 2)
    peak_usage = round(avg_usage * random.uniform(1.2, 1.8), 2)
    off_peak = round(avg_usage * random.uniform(0.3, 0.7), 2)

    support_tickets = max(0, int(profile["churn_prob"] * 10 + noise(1.5)))
    days_since_ticket = random.randint(1, 90) if support_tickets > 0 else random.randint(30, 180)

    avg_call_dur = round(max(0.5, 5 + noise(2)), 2)
    num_calls = max(0, int(30 + profile["num_lines"] * 10 + noise(8)))
    sms_count = max(0, int(50 + noise(20)))
    data_overage = max(0, int(noise(1.5)))
    late_payments = max(0, int(profile["churn_prob"] * 5 + noise(1)))
    discount = round(max(0, min(50, 10 * (tenure / 24) + noise(3))), 1)
    referrals = max(0, int(2 * (1 - profile["churn_prob"]) + noise(0.8)))
    satisfaction = max(1, min(10, round(7 - profile["churn_prob"] * 8 + noise(0.8))))
    nps = max(-100, min(100, round(40 - profile["churn_prob"] * 120 + noise(10))))
    account_age_days = tenure * 30 + random.randint(-5, 5)
    arpu = round(monthly + noise(2), 2)

    return [
        float(tenure), monthly, total, float(profile["age"]),
        float(profile["num_lines"]),
        1.0 if profile["has_international"] else 0.0,
        1.0 if profile["has_streaming"] else 0.0,
        1.0 if profile["has_device_protection"] else 0.0,
        float(contract_remaining), payment_risk,
        max(0, avg_usage), max(0, peak_usage), max(0, off_peak),
        float(support_tickets), float(days_since_ticket),
        avg_call_dur, float(num_calls), float(sms_count),
        float(data_overage), float(late_payments), discount,
        float(referrals), float(satisfaction), float(nps),
        float(account_age_days), arpu,
    ]


def generate_features_b(profile: dict, snapshot: date) -> list[float]:
    """
    Generate 1000 behavioral/interaction features.

    Groups:
      1-100:   Daily usage patterns (30-day rolling stats)
      101-200: Call detail records aggregates
      201-300: Network quality metrics
      301-400: Billing & payment behavior
      401-500: Customer service interactions
      501-600: Digital channel engagement (app, web)
      601-700: Product/service usage
      701-800: Location-based features
      801-900: Social/network features
      901-1000: Derived/interaction features
    """
    noise = lambda s=1.0: random.gauss(0, s)
    cp = profile["churn_prob"]
    tenure = profile["tenure_months"]
    monthly = profile["monthly_charge"]
    features = []

    for i in range(100):  # 1-100: daily usage
        base = 5 + monthly / 20 + noise(2)
        if cp > 0.3:
            base *= random.uniform(0.5, 0.9)
        features.append(round(max(0, base), 4))

    for i in range(100):  # 101-200: call records
        base = 10 + profile["num_lines"] * 3 + noise(3)
        features.append(round(max(0, base), 4))

    for i in range(100):  # 201-300: network quality
        base = 2 + cp * 5 + noise(1)
        features.append(round(max(0, base), 4))

    for i in range(100):  # 301-400: billing behavior
        base = monthly / 10 + noise(1.5)
        if profile["payment_method"] == "electronic_check":
            base += noise(0.5)
        features.append(round(max(0, base), 4))

    for i in range(100):  # 401-500: customer service
        base = 1 + cp * 4 + noise(0.8)
        features.append(round(max(0, base), 4))

    for i in range(100):  # 501-600: digital engagement
        base = 8 - cp * 3 + noise(2)
        features.append(round(max(0, base), 4))

    for i in range(100):  # 601-700: product usage
        base = 5 + (1 if profile["has_streaming"] else 0) * 3
        base += (1 if profile["has_international"] else 0) * 2
        base += noise(1.5)
        features.append(round(max(0, base), 4))

    for i in range(100):  # 701-800: location
        base = random.uniform(0, 10) + noise(1)
        features.append(round(max(0, base), 4))

    for i in range(100):  # 801-900: social/network
        base = profile["num_lines"] * 2 + noise(1.5)
        features.append(round(max(0, base), 4))

    for i in range(100):  # 901-1000: derived/interaction
        base = (tenure / 12) * (monthly / 50) + noise(1)
        base *= (1 - cp * 0.5)
        features.append(round(max(0, base), 4))

    return features


def generate_label(profile: dict, snapshot: date) -> int:
    """Determine if customer churns within 1 month of snapshot_date."""
    # Use hashlib for deterministic hashing across Python processes
    import hashlib
    salt = hashlib.sha256(f"{profile['customer_id']}:{snapshot}".encode()).hexdigest()
    rng = random.Random(salt)
    return 1 if rng.random() < profile["churn_prob"] else 0


# ── CSV generation ─────────────────────────────────────────────────────────────

def generate_all(output_dir: str) -> dict:
    """Generate all CSV files and DDL SQL. Returns summary stats."""
    out = Path(output_dir)
    out.mkdir(parents=True, exist_ok=True)

    profiles = [generate_customer_profile(i + 1) for i in range(NUM_CUSTOMERS)]
    score_profiles = [generate_customer_profile(10000 + i + 1) for i in range(NUM_SCORE_CUSTOMERS)]

    # ── DDL ────────────────────────────────────────────────────────────────────
    ddl_lines = []
    ddl_lines.append("-- Drop existing tables if any")
    ddl_lines.append("DROP TABLE IF EXISTS churn_predictions, labels, features_b, features_a, customers_to_score CASCADE;")
    ddl_lines.append("")

    ddl_lines.append("CREATE TABLE customers_to_score (")
    ddl_lines.append("    customer_id VARCHAR(20) NOT NULL,")
    ddl_lines.append("    snapshot_date DATE NOT NULL,")
    ddl_lines.append("    PRIMARY KEY (customer_id, snapshot_date)")
    ddl_lines.append(");")
    ddl_lines.append("")

    ddl_lines.append("CREATE TABLE features_a (")
    ddl_lines.append("    customer_id VARCHAR(20) NOT NULL,")
    ddl_lines.append("    snapshot_date DATE NOT NULL,")
    for col in FEATURE_A_COLS:
        ddl_lines.append(f"    {col} DOUBLE PRECISION,")
    ddl_lines.append("    PRIMARY KEY (customer_id, snapshot_date)")
    ddl_lines.append(");")
    ddl_lines.append("")

    ddl_lines.append("CREATE TABLE features_b (")
    ddl_lines.append("    customer_id VARCHAR(20) NOT NULL,")
    ddl_lines.append("    snapshot_date DATE NOT NULL,")
    for col in FEATURE_B_COLS:
        ddl_lines.append(f"    {col} DOUBLE PRECISION,")
    ddl_lines.append("    PRIMARY KEY (customer_id, snapshot_date)")
    ddl_lines.append(");")
    ddl_lines.append("")

    ddl_lines.append("CREATE TABLE labels (")
    ddl_lines.append("    customer_id VARCHAR(20) NOT NULL,")
    ddl_lines.append("    snapshot_date DATE NOT NULL,")
    ddl_lines.append("    is_churn INT NOT NULL CHECK (is_churn IN (0, 1)),")
    ddl_lines.append("    PRIMARY KEY (customer_id, snapshot_date)")
    ddl_lines.append(");")
    ddl_lines.append("")

    ddl_lines.append("CREATE TABLE churn_predictions (")
    ddl_lines.append("    customer_id VARCHAR(20),")
    ddl_lines.append("    snapshot_date DATE,")
    ddl_lines.append("    prediction INT,")
    ddl_lines.append("    prediction_probability DOUBLE PRECISION,")
    ddl_lines.append("    execution_date TIMESTAMP DEFAULT NOW(),")
    ddl_lines.append("    model_run_id VARCHAR(100),")
    ddl_lines.append("    serving_run_id VARCHAR(100)")
    ddl_lines.append(");")
    ddl_lines.append("")

    ddl_lines.append("-- Permissions")
    ddl_lines.append("GRANT ALL PRIVILEGES ON ALL TABLES IN SCHEMA public TO mbt_user;")
    ddl_lines.append("GRANT USAGE ON SCHEMA public TO metabase_user;")
    ddl_lines.append("GRANT SELECT ON ALL TABLES IN SCHEMA public TO metabase_user;")
    ddl_lines.append("ALTER DEFAULT PRIVILEGES IN SCHEMA public GRANT SELECT ON TABLES TO metabase_user;")

    (out / "ddl.sql").write_text("\n".join(ddl_lines) + "\n")

    # ── customers_to_score CSV ─────────────────────────────────────────────────
    with open(out / "customers_to_score.csv", "w", newline="") as f:
        w = csv.writer(f)
        w.writerow(["customer_id", "snapshot_date"])
        for snap in SNAPSHOT_DATES:
            for p in profiles:
                w.writerow([p["customer_id"], snap.isoformat()])
        for p in score_profiles:
            w.writerow([p["customer_id"], SCORE_SNAPSHOT.isoformat()])

    # ── features_a CSV ─────────────────────────────────────────────────────────
    with open(out / "features_a.csv", "w", newline="") as f:
        w = csv.writer(f)
        w.writerow(["customer_id", "snapshot_date"] + FEATURE_A_COLS)
        for snap in SNAPSHOT_DATES:
            for p in profiles:
                vals = generate_features_a(p, snap)
                w.writerow([p["customer_id"], snap.isoformat()] + vals)
        for p in score_profiles:
            vals = generate_features_a(p, SCORE_SNAPSHOT)
            w.writerow([p["customer_id"], SCORE_SNAPSHOT.isoformat()] + vals)

    # ── features_b CSV ─────────────────────────────────────────────────────────
    with open(out / "features_b.csv", "w", newline="") as f:
        w = csv.writer(f)
        w.writerow(["customer_id", "snapshot_date"] + FEATURE_B_COLS)
        for snap in SNAPSHOT_DATES:
            for p in profiles:
                vals = generate_features_b(p, snap)
                w.writerow([p["customer_id"], snap.isoformat()] + vals)
        for p in score_profiles:
            vals = generate_features_b(p, SCORE_SNAPSHOT)
            w.writerow([p["customer_id"], SCORE_SNAPSHOT.isoformat()] + vals)

    # ── labels CSV ─────────────────────────────────────────────────────────────
    churn_count = 0
    total_count = 0
    with open(out / "labels.csv", "w", newline="") as f:
        w = csv.writer(f)
        w.writerow(["customer_id", "snapshot_date", "is_churn"])
        for snap in SNAPSHOT_DATES:
            for p in profiles:
                label = generate_label(p, snap)
                churn_count += label
                total_count += 1
                w.writerow([p["customer_id"], snap.isoformat(), label])

    churn_rate = churn_count / total_count * 100 if total_count > 0 else 0

    return {
        "customers_historical": NUM_CUSTOMERS,
        "customers_scoring": NUM_SCORE_CUSTOMERS,
        "snapshots_historical": len(SNAPSHOT_DATES),
        "features_a_cols": len(FEATURE_A_COLS),
        "features_b_cols": len(FEATURE_B_COLS),
        "label_rows": total_count,
        "churn_count": churn_count,
        "churn_rate": churn_rate,
        "total_rows_customers_to_score": NUM_CUSTOMERS * len(SNAPSHOT_DATES) + NUM_SCORE_CUSTOMERS,
        "total_rows_features_a": NUM_CUSTOMERS * len(SNAPSHOT_DATES) + NUM_SCORE_CUSTOMERS,
        "total_rows_features_b": NUM_CUSTOMERS * len(SNAPSHOT_DATES) + NUM_SCORE_CUSTOMERS,
    }


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Generate warehouse seed data CSVs")
    parser.add_argument("--output-dir", required=True, help="Directory to write CSV files")
    args = parser.parse_args()

    stats = generate_all(args.output_dir)

    print(f"Generated data in {args.output_dir}/")
    print(f"  customers_to_score: {stats['total_rows_customers_to_score']} rows")
    print(f"  features_a:         {stats['total_rows_features_a']} rows x {stats['features_a_cols']} features")
    print(f"  features_b:         {stats['total_rows_features_b']} rows x {stats['features_b_cols']} features")
    print(f"  labels:             {stats['label_rows']} rows (churn rate: {stats['churn_rate']:.1f}%)")

    # List generated files
    for f in sorted(Path(args.output_dir).iterdir()):
        size = f.stat().st_size
        if size > 1024 * 1024:
            print(f"  {f.name}: {size / 1024 / 1024:.1f} MB")
        else:
            print(f"  {f.name}: {size / 1024:.1f} KB")
