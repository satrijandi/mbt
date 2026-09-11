"""Load the showcase's demo tables into Snowflake - the warehouse data plane.

The showcase project (DESIGN.md section 11) declares BOTH addresses for every
source: a `path:` for the SeaweedFS/DuckDB planes and an `identifier:` for the
Snowflake plane. This script fills in the Snowflake side, so
`mbt build --target snowflake` reads the very same cadence the lake targets do,
from the warehouse instead of parquet.

Deliberately an UPLOAD rather than a server-side generator. The showcase's
data is only ~8 MB, and loading the exact same bytes both planes read is what
makes them comparable: the cross-plane
equivalence test in tests/test_showcase_snowflake.py asserts the two
materialized panels agree row for row, which would be meaningless if the
warehouse held independently generated data.

Sources, mirroring the Makefile's `workspace` target:

    examples/showcase/data/               the 9 monthly + wide tables
    tests/fixtures/churn_demo/data/       the 3 daily tables

On top of those 12 tables it materializes the 2 panels mbt actually reads
(ADR-29), joining the five gold tables IN the warehouse with a plain CTAS.
Seeding therefore needs no privilege beyond CREATE TABLE - see the PANELS
block below for why a dynamic table is the right answer in a real deployment
and the wrong one here.

COLUMN CASE IS LOAD-BEARING: tables are written with quote_identifiers=False,
so column names fold to Snowflake's default UPPERCASE. The adapter's generated
SQL emits unquoted lowercase identifiers (which resolve to the same uppercase)
and lowercases the Arrow batches on the way back via `normalize_case`. Quoting
here would create lowercase columns that the adapter's SQL cannot resolve.

TIMESTAMPS TAKE TWO SETTINGS, and getting either wrong is silent - the load
reports success and the full row count, then temporal windows match nothing:

1. `create_table_sql` declares TIMESTAMP_NTZ explicitly instead of letting
   auto_create_table ask INFER_SCHEMA (which answers NUMBER).
2. `use_logical_type=True` on the load, so Snowflake honors the parquet
   TIMESTAMP(MICROS) annotation instead of reading the physical INT64.

With only (1), epoch integers land in a timestamp column and Snowsight shows
"Invalid date". With only (2), the column is typed by inference. Both are
pinned by tests in packages/mbt-snowflake/tests/test_showcase_snowflake_plane.py.

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

#: The only tables the Snowflake plane reads: the wide cadence's training
#: inputs plus its ground-truth outcomes. Everything else in TABLES belongs to
#: the daily/monthly cadences, which live on the lake planes.
#:
#: They still get CREATED (empty), because compile pins a snapshot for every
#: source referenced by ANY dataset or scoring node regardless of --select - a
#: missing table fails the compile before selection narrows anything. Pinning
#: is a metadata call (SYSTEM$LAST_CHANGE_COMMIT_TIME), so an empty table
#: satisfies it without putting ~26k rows of unrelated demo data in the user's
#: sandbox schema.
#:
#: Kept in sync with the specs by
#: packages/mbt-snowflake/tests/test_showcase_snowflake_plane.py.
WIDE_TABLES: frozenset[str] = frozenset(
    {
        "monthly_population",
        "monthly_labels",
        "demographic_history",
        "login_history",
        "transaction_history",
        "wide_churn_outcomes",
    }
)


#: The panels mbt actually reads on this plane (ADR-29): one relation per
#: dataset, joined upstream over the six tables above. That is the
#: warehouse-native stand-in for the dbt model a real deployment owns.
#:
#: A PLAIN TABLE (CREATE TABLE AS SELECT), not a dynamic table. ADR-29's
#: requirement is that the panel be a PHYSICAL relation rather than a view: a
#: view's change token is DDL-blind, so re-deploying it with an extra column
#: from an already-loaded table moves neither of mbt's hashes and a pinned
#: manifest re-verifies clean against a relation that changed shape. A
#: materialized table has its own rows, so the commit-time token and
#: --deep-snapshot's HASH_AGG both mean what they say - exactly as a dynamic
#: table would. What it gives up is auto-refresh, which this plane never uses:
#: the seeded data is static and every showcase run pins the same anchor.
#:
#: The payoff is the privilege floor. Seeding needs nothing but CREATE TABLE,
#: so the warehouse plane runs on a sandbox role that cannot create a dynamic
#: table. A real deployment should still own the panel as a dbt model, where
#: a dynamic table earns its keep: incremental refresh shared across every
#: model reading it, guarded consumer-side by a `freshness: {max_lag: ...}`
#: check.
#:
#: `_scoring` is the serving twin: the same joins, minus the label and its
#: INNER join, so the newest cohort survives. They share one definition here
#: for the reason ADR-29 gives - two relations that must stay in lockstep
#: should not be written twice.
PANEL_FEATURE_JOINS = """
  JOIN (SELECT * EXCLUDE (etl_loaded_at) FROM {demographic_history}) AS demo
       USING (customer_id, inference_date)
  JOIN (SELECT * EXCLUDE (etl_loaded_at) FROM {login_history}) AS login
       USING (customer_id, inference_date)
  JOIN (SELECT * EXCLUDE (etl_loaded_at) FROM {transaction_history}) AS txn
       USING (safe_id, inference_date)
"""

#: Panel name -> the label join that distinguishes it (empty for serving).
PANELS: dict[str, str] = {
    "monthly_panel": "JOIN {monthly_labels} AS lbl USING (customer_id, inference_date)",
    "monthly_panel_scoring": "",
}


def panel_sql(database: str, schema: str, panel: str) -> str:
    """The DDL for one panel, materialized as a plain table."""
    refs = {source: f"{database}.{schema}.{table_name(source)}" for source in TABLES}
    joins = PANEL_FEATURE_JOINS.format(**refs)
    label_join = PANELS[panel].format(**refs)
    return (
        f"CREATE OR REPLACE TABLE {database}.{schema}.{table_name(panel)}\n"
        f"AS SELECT * FROM {refs['monthly_population']} AS pop\n"
        f"  {label_join}{joins}"
    )


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


#: pandas dtype family -> Snowflake column type. Deliberately explicit: see
#: create_table_sql for why INFER_SCHEMA cannot be trusted with this data.
_TYPE_MAP: tuple[tuple[str, str], ...] = (
    ("datetime64", "TIMESTAMP_NTZ"),
    ("bool", "BOOLEAN"),
    ("int", "NUMBER(38,0)"),
    ("uint", "NUMBER(38,0)"),
    ("float", "FLOAT"),
)


def snowflake_type(dtype: Any) -> str:
    """Snowflake column type for a pandas dtype.

    Anything unrecognized becomes VARCHAR, which is the safe default for the
    object columns (plan tiers, ids) these tables carry.
    """
    name = str(dtype)
    for prefix, sf_type in _TYPE_MAP:
        if name.startswith(prefix):
            return sf_type
    return "VARCHAR"


def create_table_sql(database: str, schema: str, table: str, frame: Any) -> str:
    """Explicit DDL for one demo table.

    NOT left to ``write_pandas(auto_create_table=True)``. That path stages the
    frame as parquet and asks Snowflake's INFER_SCHEMA for the column types,
    and INFER_SCHEMA typed our tz-naive datetime64 columns as NUMBER - the
    tables loaded fine and every timestamp arrived as epoch MICROSECONDS
    (2025-07-01 became 1751328000000000).

    Nothing downstream survives that: mbt's temporal split pushes down
    ``CAST(inference_date AS TIMESTAMP_NTZ) >= TO_TIMESTAMP_NTZ('...')``, the
    label join matches on the same column, and `mbt monitor` does maturity
    arithmetic on it. Declaring the types here makes the load deterministic
    and independent of connector/INFER_SCHEMA behavior.

    Columns are UNQUOTED so they fold to uppercase, matching the
    quote_identifiers=False load and the adapter's generated SQL.
    """
    columns = ",\n  ".join(
        f"{name} {snowflake_type(dtype)}" for name, dtype in frame.dtypes.items()
    )
    return f"CREATE OR REPLACE TABLE {database}.{schema}.{table} (\n  {columns}\n)"


#: Env vars read VERBATIM, never trimmed: leading or trailing whitespace can be
#: significant in a secret, and silently stripping it would turn a working
#: credential into an authentication failure with no way to see why.
VERBATIM_ENV: frozenset[str] = frozenset({"password", "private_key_file_pwd"})


def _env(key: str) -> str | None:
    """``SNOWFLAKE_<KEY>`` from the environment, whitespace-trimmed.

    Trimming is not cosmetic tidying - a padded identifier reaches Snowflake by
    two paths from this script and they disagree about whitespace:

    1. ``create_table_sql`` interpolates the schema straight into DDL, where the
       SQL tokenizer ignores the padding, so every CREATE TABLE SUCCEEDS.
    2. ``write_pandas`` binds the same location through ``COPY INTO
       identifier(?)``, where the padding IS part of the identifier.

    So a ``SNOWFLAKE_SCHEMA`` pasted with trailing spaces (column-aligned output
    and quoted .env values both do it) failed with ``Schema
    'DB."SANDBOX          "' does not exist or not authorized`` - AFTER six
    tables had already been created, and naming a schema that looks correct in
    the message unless you notice the quotes. Observed against a real account.

    Secrets are exempt (:data:`VERBATIM_ENV`).
    """
    value = os.environ.get(f"SNOWFLAKE_{key.upper()}")
    if value is None or key in VERBATIM_ENV:
        return value
    return value.strip()


def _connection_config() -> dict[str, Any]:
    """Connection kwargs from the same SNOWFLAKE_* env vars profiles.yml reads.

    Same keys the live Snowflake suite reads, so one .env drives both.
    """
    required = ("account", "user", "warehouse", "database", "schema")
    # Checked on the TRIMMED value, so a whitespace-only var reads as unset
    # rather than as a connection attempt against an identifier of spaces.
    missing = [k for k in required if not _env(k)]
    if missing:
        names = ", ".join(f"SNOWFLAKE_{k.upper()}" for k in missing)
        raise SystemExit(
            f"missing environment: {names} (copy packages/mbt-snowflake/.env.example "
            "to .env, edit it, then `set -a; source .env; set +a`)"
        )
    config: dict[str, Any] = {k: _env(k) for k in required}
    for key in ("password", "role", "authenticator"):
        value = _env(key)
        if value:
            config[key] = value
    private_key_file = _env("private_key_file")
    if private_key_file:
        config["private_key_file"] = private_key_file
        key_password = _env("private_key_file_pwd")
        if key_password:
            config["private_key_file_pwd"] = key_password
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
    """Every showcase relation already present, panels included.

    The panels are covered so a half-seeded schema trips the --force guard
    instead of being silently half-replaced.
    """
    names = ", ".join(f"'{table_name(s)}'" for s in (*TABLES, *PANELS))
    cursor.execute(
        f"SELECT table_name FROM {database}.INFORMATION_SCHEMA.TABLES "
        f"WHERE table_schema = '{schema.upper()}' AND table_name IN ({names})"
    )
    return sorted(row[0] for row in cursor.fetchall())


def dynamic_panels(cursor: Any, database: str, schema: str) -> set[str]:
    """Panel names currently held by a DYNAMIC table.

    This script used to create the panels as dynamic tables, and Snowflake's
    DDL is kind-specific in both directions: DROP TABLE refuses a dynamic
    table, and CREATE OR REPLACE TABLE will not replace one either. So an
    account seeded before that change needs the old kind removed by name
    before either path here can proceed. Probing the kind beats guessing at
    IF EXISTS semantics across kinds - on a schema that never held one this
    returns an empty set and nothing else changes.
    """
    cursor.execute(f"SHOW DYNAMIC TABLES LIKE '{PREFIX}%' IN SCHEMA {database}.{schema}")
    # SHOW returns the object name in column 1 ("name"); column 0 is created_on.
    held = {row[1] for row in cursor.fetchall()}
    return {panel for panel in PANELS if table_name(panel) in held}


def drop_relation_sql(database: str, schema: str, name: str, *, dynamic: bool) -> str:
    """DROP for one relation, in the kind Snowflake will accept."""
    kind = "DYNAMIC TABLE" if dynamic else "TABLE"
    return f"DROP {kind} IF EXISTS {database}.{schema}.{name}"


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
    parser.add_argument(
        "--all-cadences",
        action="store_true",
        help="also load rows for the daily/monthly cadences (default: create them "
        "empty, since the Snowflake plane only reads the wide cadence)",
    )
    args = parser.parse_args(argv)

    def loads_rows(source: str) -> bool:
        return args.all_cadences or source in WIDE_TABLES

    if args.dry_run:
        database = _env("database") or "ANALYTICS"
        schema = _env("schema") or "SANDBOX"
        total = 0
        for source, directory in TABLES.items():
            frame = read_table(directory)
            rows = len(frame) if loads_rows(source) else 0
            total += rows
            note = "" if loads_rows(source) else "  (empty: not read by the wide cadence)"
            print(
                f"{database}.{schema}.{table_name(source):40s} "
                f"{rows:>8,} rows x {len(frame.columns):>3} cols{note}"
            )
        loaded = sum(1 for s in TABLES if loads_rows(s))
        print(
            f"\n{len(TABLES)} tables created, {loaded} with data, {total:,} rows total"
            + ("" if args.all_cadences else "  (--all-cadences loads the rest)")
        )
        # The DDL is the part worth eyeballing: timestamp columns MUST come out
        # as TIMESTAMP_NTZ or every temporal window silently matches nothing.
        sample = "monthly_population"
        print(f"\nDDL for {table_name(sample)} (the rest follow the same mapping):\n")
        print(create_table_sql(database, schema, table_name(sample), read_table(TABLES[sample])))
        print("\nThe panels mbt reads (ADR-29) - one relation per dataset:\n")
        for panel in PANELS:
            print(panel_sql(database, schema, panel))
            print()
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
                # Panels first, then the tables they read: the reverse of the
                # order they are created in. A panel left behind is the leak
                # this used to have - it iterated TABLES only, so both panels
                # survived every --drop.
                stale = dynamic_panels(cursor, database, schema)
                for relation in (*PANELS, *TABLES):
                    name = table_name(relation)
                    cursor.execute(
                        drop_relation_sql(database, schema, name, dynamic=relation in stale)
                    )
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
            # Create the table ourselves, then load into it. auto_create_table
            # would hand the typing decision to INFER_SCHEMA, which types these
            # datetime64 columns as NUMBER (see create_table_sql).
            ddl_cursor = connection.cursor()
            try:
                ddl_cursor.execute(create_table_sql(database, schema, name, frame))
            finally:
                ddl_cursor.close()
            if not loads_rows(source):
                # Created but not loaded: the wide cadence never reads this
                # table, and compile only needs it to EXIST so snapshot pinning
                # can resolve it (a metadata call, so an empty table answers).
                print(f"created {database}.{schema}.{name:40s}    empty (other cadence)")
                continue
            # quote_identifiers=False: columns must fold to UPPERCASE so the
            # adapter's unquoted lowercase SQL resolves them (see module docstring).
            #
            # use_logical_type=True is the OTHER half of the timestamp fix, and
            # it is not optional. write_pandas stages the frame as parquet and
            # COPYs it through a generated FILE FORMAT; its default (None)
            # leaves USE_LOGICAL_TYPE unset, and Snowflake's PARQUET default for
            # that is FALSE - so Snowflake ignores the TIMESTAMP(MICROS)
            # annotation and reads the physical INT64. That is what typed these
            # columns as NUMBER under auto_create_table, and with the explicit
            # TIMESTAMP_NTZ DDL it instead lands epoch integers in a timestamp
            # column, which renders as 'Invalid date'. The connector only warns
            # about this for tz-AWARE columns; ours are tz-naive, so it is
            # silent.
            success, _chunks, rows, _output = write_pandas(
                connection,
                frame,
                table_name=name,
                database=database,
                schema=schema,
                auto_create_table=False,
                overwrite=False,  # the CREATE OR REPLACE above already emptied it
                quote_identifiers=False,
                use_logical_type=True,
            )
            if not success:
                raise SystemExit(f"failed loading {database}.{schema}.{name}")
            print(f"loaded {database}.{schema}.{name:40s} {rows:>8,} rows")

        # The panels come last: the CTAS reads the tables loaded above, so it
        # would otherwise materialize an empty panel.
        panel_cursor = connection.cursor()
        try:
            # A panel left over from a pre-CTAS seeding is a dynamic table,
            # which CREATE OR REPLACE TABLE will not replace.
            stale = dynamic_panels(panel_cursor, database, schema)
            for panel in PANELS:
                name = table_name(panel)
                if panel in stale:
                    panel_cursor.execute(drop_relation_sql(database, schema, name, dynamic=True))
                    print(f"dropped {database}.{schema}.{name:40s}    stale dynamic table")
                panel_cursor.execute(panel_sql(database, schema, panel))
                print(f"created {database}.{schema}.{name:40s}    table (CTAS)")
        finally:
            panel_cursor.close()
    finally:
        connection.close()

    print(
        f"\n{len(TABLES)} tables and {len(PANELS)} panels ready in {database}.{schema}.\n"
        "Next: uv run mbt build --project-dir examples/showcase/project "
        "--target snowflake --select tag:wide"
    )
    return 0


if __name__ == "__main__":
    sys.exit(main())
