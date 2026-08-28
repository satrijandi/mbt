"""Load the showcase's demo tables into Snowflake - the warehouse data plane.

The showcase project (DESIGN.md section 11) declares BOTH addresses for every
source: a `path:` for the SeaweedFS/DuckDB planes and an `identifier:` for the
Snowflake plane. This script fills in the Snowflake side, so
`mbt build --target snowflake` reads the very same cadence the lake targets do,
from the warehouse instead of parquet.

Deliberately an UPLOAD, not a server-side generator (unlike
examples/snowflake_wide/seed_demo_tables.py, whose GENERATOR() SQL synthesizes
rows in-warehouse). The showcase's data is only ~8 MB, and loading the exact
same bytes both planes read is what makes them comparable: the cross-plane
equivalence test in tests/test_showcase_snowflake.py asserts the two
materialized panels agree row for row, which would be meaningless if the
warehouse held independently generated data.

Sources, mirroring the Makefile's `workspace` target:

    examples/showcase/data/               the 9 monthly + wide tables
    tests/fixtures/churn_demo/data/       the 3 daily tables

COLUMN CASE IS LOAD-BEARING: tables are written with quote_identifiers=False,
so column names fold to Snowflake's default UPPERCASE. The adapter's generated
SQL emits unquoted lowercase identifiers (which resolve to the same uppercase)
and lowercases the Arrow batches on the way back via `normalize_case`. Quoting
here would create lowercase columns that the adapter's SQL cannot resolve.

Usage (from the repo root, with SNOWFLAKE_* exported - see
packages/mbt-snowflake/.env.example):

    uv run python examples/showcase/scripts/seed_snowflake.py --dry-run
    uv run python examples/showcase/scripts/seed_snowflake.py
    uv run python examples/showcase/scripts/seed_snowflake.py --force
    uv run python examples/showcase/scripts/seed_snowflake.py --drop

Refuses to replace tables that already exist unless --force is given; use a
scratch schema, not a shared gold one.
"""

from __future__ import annotations

import argparse
import os
import sys
from pathlib import Path
from typing import Any

#: Prefix mirroring the live suite's MBT_LIVE_ convention, so these demo
#: tables can never be confused with real ones. Must match the identifiers
#: in examples/showcase/project/sources.yml.
PREFIX = "MBT_SHOWCASE_"

REPO_ROOT = Path(__file__).resolve().parents[3]
SHOWCASE_DATA = REPO_ROOT / "examples" / "showcase" / "data"
CHURN_DEMO_DATA = REPO_ROOT / "tests" / "fixtures" / "churn_demo" / "data"

#: source-table name -> directory of parquet parts. Keys match sources.yml
#: table names exactly; the Snowflake table is PREFIX + key.upper().
TABLES: dict[str, Path] = {
    # daily cadence (shared with the churn_demo fixture)
    "subscribers": CHURN_DEMO_DATA / "subscribers",
    "scoring_batch": CHURN_DEMO_DATA / "scoring_batch",
    "churn_outcomes": CHURN_DEMO_DATA / "churn_outcomes",
    # monthly cadence
    "monthly_subscribers": SHOWCASE_DATA / "monthly_subscribers",
    "monthly_scoring_batch": SHOWCASE_DATA / "monthly_scoring_batch",
    "monthly_churn_outcomes": SHOWCASE_DATA / "monthly_churn_outcomes",
    # wide multi-table cadence (ADR-22) - what the Snowflake plane trains on
    "monthly_population": SHOWCASE_DATA / "monthly_population",
    "monthly_labels": SHOWCASE_DATA / "monthly_labels",
    "demographic_history": SHOWCASE_DATA / "demographic_history",
    "login_history": SHOWCASE_DATA / "login_history",
    "transaction_history": SHOWCASE_DATA / "transaction_history",
    "wide_churn_outcomes": SHOWCASE_DATA / "wide_churn_outcomes",
}


def table_name(source: str) -> str:
    return f"{PREFIX}{source.upper()}"


def read_table(directory: Path) -> Any:
    """Every parquet part under ``directory``, concatenated, as a DataFrame."""
    import pandas as pd

    parts = sorted(directory.glob("*.parquet"))
    if not parts:
        raise SystemExit(
            f"no parquet under {directory} - generate it first "
            "(examples/showcase/scripts/generate_wide_data.py, or `make workspace`)"
        )
    return pd.concat([pd.read_parquet(p) for p in parts], ignore_index=True)


def _connection_config() -> dict[str, Any]:
    """Connection kwargs from the same SNOWFLAKE_* env vars profiles.yml reads.

    Mirrors examples/snowflake_wide/seed_demo_tables.py so one .env drives both
    examples.
    """
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
        # One browser prompt for this script; whether the NEXT process reuses
        # it depends on the account's ALLOW_ID_TOKEN (docs/troubleshooting.md).
        config["client_store_temporary_credential"] = True
    if not any(k in config for k in ("password", "authenticator", "private_key_file")):
        raise SystemExit(
            "no auth configured: set SNOWFLAKE_AUTHENTICATOR=externalbrowser (SSO), "
            "SNOWFLAKE_PASSWORD, or SNOWFLAKE_PRIVATE_KEY_FILE"
        )
    return config


def existing_tables(cursor: Any, database: str, schema: str) -> list[str]:
    names = ", ".join(f"'{table_name(s)}'" for s in TABLES)
    cursor.execute(
        f"SELECT table_name FROM {database}.INFORMATION_SCHEMA.TABLES "
        f"WHERE table_schema = '{schema.upper()}' AND table_name IN ({names})"
    )
    return sorted(row[0] for row in cursor.fetchall())


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    parser.add_argument(
        "--dry-run",
        action="store_true",
        help="print what would be loaded, with row/column counts, and exit (no connection)",
    )
    parser.add_argument(
        "--force", action="store_true", help="replace demo tables that already exist"
    )
    parser.add_argument(
        "--drop", action="store_true", help="drop the showcase demo tables and exit"
    )
    args = parser.parse_args(argv)

    if args.dry_run:
        database = os.environ.get("SNOWFLAKE_DATABASE", "ANALYTICS")
        schema = os.environ.get("SNOWFLAKE_SCHEMA", "SANDBOX")
        total = 0
        for source, directory in TABLES.items():
            frame = read_table(directory)
            total += len(frame)
            print(
                f"{database}.{schema}.{table_name(source):40s} "
                f"{len(frame):>8,} rows x {len(frame.columns):>3} cols  <- {directory}"
            )
        print(f"\n{len(TABLES)} tables, {total:,} rows total")
        return 0

    config = _connection_config()
    database, schema = config["database"], config["schema"]
    import snowflake.connector
    from snowflake.connector.pandas_tools import write_pandas

    connection = snowflake.connector.connect(**config)
    try:
        cursor = connection.cursor()
        try:
            if args.drop:
                for source in TABLES:
                    name = table_name(source)
                    cursor.execute(f"DROP TABLE IF EXISTS {database}.{schema}.{name}")
                    print(f"dropped {database}.{schema}.{name}")
                return 0

            present = existing_tables(cursor, database, schema)
            if present and not args.force:
                listed = ", ".join(present)
                raise SystemExit(
                    f"these tables already exist in {database}.{schema}: {listed}\n"
                    "re-run with --force to replace them (or --drop to remove them)"
                )
        finally:
            cursor.close()

        for source, directory in TABLES.items():
            name = table_name(source)
            frame = read_table(directory)
            # quote_identifiers=False: columns must fold to UPPERCASE so the
            # adapter's unquoted lowercase SQL resolves them (see module docstring).
            success, _chunks, rows, _output = write_pandas(
                connection,
                frame,
                table_name=name,
                database=database,
                schema=schema,
                auto_create_table=True,
                overwrite=True,
                quote_identifiers=False,
            )
            if not success:
                raise SystemExit(f"failed loading {database}.{schema}.{name}")
            print(f"loaded {database}.{schema}.{name:40s} {rows:>8,} rows")
    finally:
        connection.close()

    print(
        f"\n{len(TABLES)} tables ready in {database}.{schema}.\n"
        "Next: uv run mbt build --project-dir examples/showcase/project "
        "--target snowflake --select tag:wide"
    )
    return 0


if __name__ == "__main__":
    sys.exit(main())
