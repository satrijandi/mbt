"""Seed the five snowflake_wide demo tables - entirely server-side.

Creates CUSTOMER_POPULATION, CHURN_LABELS, DEMOGRAPHIC_FEATURES,
ENGAGEMENT_FEATURES, and BILLING_FEATURES in the database.schema named by
the SNOWFLAKE_* environment variables (the same ones profiles.yml reads),
so `mbt build --project-dir examples/snowflake_wide` runs against them
unmodified.

Every table is one CREATE TABLE ... AS SELECT over TABLE(GENERATOR(...)):
rows are synthesized inside Snowflake, so nothing is uploaded and the size
knobs cost warehouse-seconds, not bandwidth - seeding 10M customers is the
same one-statement round trip as seeding 10k. The data is deterministic
(HASH-based, no RANDOM), so reruns produce identical tables.

Shapes worth noticing:

- The population spine is NOT all (customer, month) pairs: each customer
  becomes active in a deterministic start month, so the spine grows month
  over month and genuinely decides which rows exist (ADR-22).
- The FEATURE tables cover a strict superset of the population/label
  universe - extra customers AND an extra month - exactly like real gold
  tables built for the whole company rather than one model's cohort.
  ENGAGEMENT_FEATURES additionally carries its own mid-month snapshot
  cadence. mbt joins everything onto the spine by exact key, so none of
  those extra rows ever enter a training set or scoring batch.
- Churn is driven by a per-customer propensity that also shapes the
  engagement and billing features, so a model can learn it.

Usage (after `set -a; source .env; set +a`, see the README):

    uv run python examples/snowflake_wide/seed_demo_tables.py --dry-run
    uv run python examples/snowflake_wide/seed_demo_tables.py
    uv run python examples/snowflake_wide/seed_demo_tables.py --customers 2000000
    uv run python examples/snowflake_wide/seed_demo_tables.py --drop

Refuses to touch tables that already exist unless --force is given; use a
scratch schema, not a shared gold one.
"""

from __future__ import annotations

import argparse
import os
import sys
from typing import Any

#: First month-start snapshot; --months extends forward from here. The
#: committed dataset windows (train 2026-01/03, test 2026-03/05) and the
#: scoring anchor in the README assume this origin.
START_MONTH = "2026-01-01"

#: The five demo tables, matching sources.yml identifiers exactly.
TABLES = (
    "CUSTOMER_POPULATION",
    "CHURN_LABELS",
    "DEMOGRAPHIC_FEATURES",
    "ENGAGEMENT_FEATURES",
    "BILLING_FEATURES",
)

#: Deterministic per-customer churn propensity in [0, 1); the label and the
#: signal-bearing features all derive from it.
_PROPENSITY = "MOD(ABS(HASH(customer_id)), 1000) / 1000.0"
#: Per-(customer, month) noise in [0, 1).
_NOISE = "MOD(ABS(HASH(customer_id, snapshot_date)), 1000) / 1000.0"


def _all_pairs(customers: int, months: int) -> str:
    """CTE producing every (customer_id, snapshot_date) pair, generated
    server-side. ROW_NUMBER (not bare SEQ4, which may leave gaps) guarantees
    every statement generates the IDENTICAL id set, so the five independently
    created tables agree on their keys."""
    return f"""WITH customers AS (
  SELECT ROW_NUMBER() OVER (ORDER BY SEQ4()) - 1 AS customer_id
  FROM TABLE(GENERATOR(ROWCOUNT => {customers}))
), months AS (
  SELECT DATEADD(MONTH, ROW_NUMBER() OVER (ORDER BY SEQ4()) - 1, DATE '{START_MONTH}')
         AS snapshot_date
  FROM TABLE(GENERATOR(ROWCOUNT => {months}))
), pairs AS (
  SELECT c.customer_id, m.snapshot_date
  FROM customers c CROSS JOIN months m
)"""


def feature_superset(customers: int, months: int) -> tuple[int, int]:
    """The universe FEATURE tables cover: more customers and one more month
    than the population/label universe. Real gold feature tables are exactly
    this - built for the whole company, not for one model's cohort - and the
    spine join is what makes that safe: rows outside the population never
    enter a training set or a scoring batch."""
    return customers + max(customers // 10, 1), months + 1


def render_statements(database: str, schema: str, customers: int, months: int) -> dict[str, str]:
    """The five CREATE OR REPLACE TABLE statements, keyed by table name.

    Pure string rendering (no connection), so tests can assert the DDL
    matches sources.yml without a warehouse.
    """
    prefix = f"{database}.{schema}"
    pairs = _all_pairs(customers, months)
    feature_pairs = _all_pairs(*feature_superset(customers, months))
    return {
        # The spine: customer c is active from its deterministic start month
        # onward, so later snapshots hold more customers (ADR-22: the spine
        # decides which rows exist).
        "CUSTOMER_POPULATION": f"""CREATE OR REPLACE TABLE {prefix}.CUSTOMER_POPULATION AS
{pairs}
SELECT customer_id, snapshot_date
FROM pairs
WHERE MOD(ABS(HASH(customer_id, 'start')), {months})
      <= DATEDIFF(MONTH, DATE '{START_MONTH}', snapshot_date)""",
        "CHURN_LABELS": f"""CREATE OR REPLACE TABLE {prefix}.CHURN_LABELS AS
{pairs}
SELECT customer_id, snapshot_date,
       CASE WHEN 0.8 * ({_PROPENSITY}) + 0.2 * ({_NOISE}) > 0.65
            THEN 1 ELSE 0 END AS is_churn
FROM pairs""",
        "DEMOGRAPHIC_FEATURES": f"""CREATE OR REPLACE TABLE {prefix}.DEMOGRAPHIC_FEATURES AS
{feature_pairs}
SELECT customer_id, snapshot_date,
       20 + MOD(ABS(HASH(customer_id, 'age')), 50) AS age,
       MOD(ABS(HASH(customer_id, 'tenure')), 36) AS tenure_months
FROM pairs""",
        # Engagement carries the signal (low-propensity customers log in more)
        # and, on top of the superset universe, its OWN snapshot cadence: a
        # mid-month row 14 days after every month-start. The dataset joins on
        # exact (customer_id, snapshot_date), so only spine-matching dates
        # appear in the panel; the mid-month rows stay in the warehouse.
        "ENGAGEMENT_FEATURES": f"""CREATE OR REPLACE TABLE {prefix}.ENGAGEMENT_FEATURES AS
{feature_pairs}
, cadence AS (
  SELECT customer_id, snapshot_date FROM pairs
  UNION ALL
  SELECT customer_id, DATEADD(DAY, 14, snapshot_date) FROM pairs
)
SELECT customer_id, snapshot_date,
       GREATEST(0, ROUND(40 * (1 - ({_PROPENSITY})))
                   - MOD(ABS(HASH(customer_id, snapshot_date, 'l')), 8)) AS logins_30d,
       (1 + 30 * (1 - ({_PROPENSITY})) + ({_NOISE}) * 5)::FLOAT AS avg_session_min
FROM cadence""",
        # Billing carries a bookkeeping column (etl_loaded_at) on purpose: the
        # dataset and scoring specs drop it with a per-table `exclude:`
        # (ADR-25), so it is pruned inside the warehouse query and never
        # reaches a training set or scoring batch.
        "BILLING_FEATURES": f"""CREATE OR REPLACE TABLE {prefix}.BILLING_FEATURES AS
{feature_pairs}
SELECT customer_id, snapshot_date,
       (10 + 60 * ({_PROPENSITY}) + ({_NOISE}) * 20)::FLOAT AS monthly_spend,
       CASE MOD(ABS(HASH(customer_id, 'plan')), 3)
            WHEN 0 THEN 'basic' WHEN 1 THEN 'pro' ELSE 'enterprise'
       END AS plan_tier,
       DATEADD(DAY, 2, snapshot_date) AS etl_loaded_at
FROM pairs""",
    }


def _connection_config() -> dict[str, Any]:
    """Connection kwargs from the same SNOWFLAKE_* env vars profiles.yml
    reads: account/user/warehouse/database/schema required, exactly one of
    password / externalbrowser / key-pair for auth, role optional."""
    required = ("account", "user", "warehouse", "database", "schema")
    missing = [k for k in required if not os.environ.get(f"SNOWFLAKE_{k.upper()}")]
    if missing:
        names = ", ".join(f"SNOWFLAKE_{k.upper()}" for k in missing)
        raise SystemExit(
            f"missing environment: {names} (copy packages/mbt-snowflake/.env.example "
            "to .env, edit it, then `set -a; source .env; set +a`)"
        )
    config: dict[str, Any] = {k: os.environ[f"SNOWFLAKE_{k.upper()}"] for k in required}
    for key in ("password", "role", "authenticator"):
        value = os.environ.get(f"SNOWFLAKE_{key.upper()}")
        if value:
            config[key] = value
    if os.environ.get("SNOWFLAKE_PRIVATE_KEY_FILE"):
        config["private_key_file"] = os.environ["SNOWFLAKE_PRIVATE_KEY_FILE"]
        if os.environ.get("SNOWFLAKE_PRIVATE_KEY_FILE_PWD"):
            config["private_key_file_pwd"] = os.environ["SNOWFLAKE_PRIVATE_KEY_FILE_PWD"]
    if str(config.get("authenticator", "")).lower() == "externalbrowser":
        # One browser prompt for the whole run (and, with mbt-snowflake[sso]
        # installed, for the following mbt session too).
        config["client_store_temporary_credential"] = True
    if not any(k in config for k in ("password", "authenticator", "private_key_file")):
        raise SystemExit(
            "no auth configured: set SNOWFLAKE_AUTHENTICATOR=externalbrowser (SSO), "
            "SNOWFLAKE_PASSWORD, or SNOWFLAKE_PRIVATE_KEY_FILE"
        )
    return config


def _existing_tables(cursor: Any, database: str, schema: str) -> list[str]:
    names = ", ".join(f"'{t}'" for t in TABLES)
    cursor.execute(
        f"SELECT table_name FROM {database}.INFORMATION_SCHEMA.TABLES "
        f"WHERE table_schema = '{schema.upper()}' AND table_name IN ({names})"
    )
    return sorted(row[0] for row in cursor.fetchall())


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    parser.add_argument(
        "--customers", type=int, default=50_000, help="customers to generate (default 50k)"
    )
    parser.add_argument(
        "--months",
        type=int,
        default=5,
        help=f"month-start snapshots from {START_MONTH} (default 5: Jan..May 2026, "
        "matching the committed train/test/score windows)",
    )
    parser.add_argument(
        "--dry-run", action="store_true", help="print the SQL and exit (no connection)"
    )
    parser.add_argument(
        "--force", action="store_true", help="replace demo tables that already exist"
    )
    parser.add_argument("--drop", action="store_true", help="drop the five demo tables and exit")
    args = parser.parse_args(argv)

    if args.dry_run:
        database = os.environ.get("SNOWFLAKE_DATABASE", "ANALYTICS")
        schema = os.environ.get("SNOWFLAKE_SCHEMA", "SANDBOX")
        for sql in render_statements(database, schema, args.customers, args.months).values():
            print(sql + ";\n")
        return 0

    config = _connection_config()
    database, schema = config["database"], config["schema"]
    import snowflake.connector

    connection = snowflake.connector.connect(**config)
    try:
        cursor = connection.cursor()
        try:
            if args.drop:
                for table in TABLES:
                    cursor.execute(f"DROP TABLE IF EXISTS {database}.{schema}.{table}")
                    print(f"dropped {database}.{schema}.{table}")
                return 0
            statements = render_statements(database, schema, args.customers, args.months)
            existing = _existing_tables(cursor, database, schema)
            if existing and not args.force:
                raise SystemExit(
                    f"refusing to replace existing tables in {database}.{schema}: "
                    f"{', '.join(existing)} (rerun with --force, or point "
                    "SNOWFLAKE_SCHEMA at a scratch schema)"
                )
            for table, sql in statements.items():
                cursor.execute(sql)
                cursor.execute(f"SELECT COUNT(*) FROM {database}.{schema}.{table}")
                row = cursor.fetchone()
                count = row[0] if row else 0
                print(f"created {database}.{schema}.{table}: {count} rows")
        finally:
            cursor.close()
    finally:
        connection.close()
    return 0


if __name__ == "__main__":
    sys.exit(main())
