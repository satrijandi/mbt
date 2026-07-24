"""Verify mbt's warehouse pulls against YOUR Snowflake - cheaply.

Renders the exact SQL mbt pushes down for this project's training splits and
scoring batch (the five-table join, per-table column projections, temporal
windows, entity sampling), straight from the committed specs. With
``--execute`` it then runs BOUNDED probes against the real warehouse:

- ``SELECT COUNT(*)`` per split/batch (aggregates in-warehouse, tiny result)
- ``SELECT * ... LIMIT n`` to show the actual columns and a few rows
- an INFORMATION_SCHEMA comparison per source table: which columns exist in
  the warehouse vs which enter the panel, so per-table pruning (ADR-25) is
  verified against production tables, not trusted from docs.

Nothing is written anywhere; total cost is a handful of small queries. Uses
the same SNOWFLAKE_* env vars as profiles.yml (browser SSO included).

    uv run python examples/snowflake_wide/verify_warehouse_pull.py
    uv run python examples/snowflake_wide/verify_warehouse_pull.py --execute
    uv run python examples/snowflake_wide/verify_warehouse_pull.py \\
        --execute --anchor 2026-06-01T00:00:00Z --sample-fraction 0.01
"""

from __future__ import annotations

import argparse
import importlib.util
import os
import re
import sys
from datetime import UTC, datetime
from pathlib import Path
from typing import Any

import yaml
from mbt_snowflake.sql import base_relation, sampling_predicate, scoring_query, split_queries

from mbt.compile.windows import format_ts, parse_window
from mbt_adapter_base import DatasetSpec, ScoringInputSpec

HERE = Path(__file__).resolve().parent


def _source_name(ref: str) -> str:
    """'source('snowflake', 'churn_labels')' -> 'churn_labels'."""
    return re.findall(r"'([^']*)'", ref)[1]


def _load_specs() -> tuple[DatasetSpec, ScoringInputSpec]:
    """The dataset spec and the scoring node's INPUT spec (the rest of the
    scoring node - gates, monitors - carries jinja vars and is irrelevant to
    query construction)."""
    dataset_doc = yaml.safe_load((HERE / "datasets" / "wide_churn_training.yml").read_text())
    scoring_doc = yaml.safe_load((HERE / "scoring" / "wide_churn_scoring.yml").read_text())
    return (
        DatasetSpec.model_validate(dataset_doc["datasets"][0]),
        ScoringInputSpec.model_validate(scoring_doc["scoring"][0]["input"]),
    )


def _table_refs() -> dict[str, str]:
    """source name -> fully qualified table ref, exactly as the adapter
    qualifies them: qualified identifiers pass through, bare ones inherit
    DATABASE.SCHEMA from the environment (the profile's config)."""
    database = os.environ.get("SNOWFLAKE_DATABASE", "ANALYTICS")
    schema = os.environ.get("SNOWFLAKE_SCHEMA", "SANDBOX")
    doc = yaml.safe_load((HERE / "sources.yml").read_text())
    refs: dict[str, str] = {}
    for table in doc["sources"][0]["tables"]:
        identifier = table["identifier"]
        refs[table["name"]] = (
            identifier if "." in identifier else f"{database}.{schema}.{identifier}"
        )
    return refs


def _resolve(expression: str, anchor: datetime) -> tuple[str, str]:
    start, end = parse_window(expression).resolve(anchor)
    return format_ts(start), format_ts(end)


def _dataset_queries(
    spec: DatasetSpec, refs_by_ref: dict[str, str], anchor: datetime, fraction: float
) -> dict[str, str]:
    """Mirrors SnowflakeDataAdapter.build_dataset's query construction."""
    where = [f"({f})" for f in spec.filters]
    if fraction < 1.0:
        where.append(sampling_predicate(spec.sample_key_columns, fraction))
    windows = {
        "train": _resolve(spec.split.train, anchor),
        "test": _resolve(spec.split.test, anchor),
    }
    relation, exclude = base_relation(spec, refs_by_ref)
    return split_queries(spec, relation, where, windows, exclude)


def _scoring_batch_query(
    spec: ScoringInputSpec, refs_by_ref: dict[str, str], anchor: datetime, fraction: float
) -> str:
    """Mirrors SnowflakeDataAdapter.build_scoring_input's construction."""
    where = [f"({f})" for f in spec.filters]
    if fraction < 1.0:
        where.append(sampling_predicate(spec.sample_key_columns, fraction))
    window = _resolve(spec.window, anchor) if spec.window is not None else None
    return scoring_query(spec, refs_by_ref, where, window)


def _connect() -> Any:
    """Reuse the seed script's env-driven connection config (SSO included)."""
    seed_spec = importlib.util.spec_from_file_location(
        "seed_demo_tables", HERE / "seed_demo_tables.py"
    )
    assert seed_spec is not None and seed_spec.loader is not None
    seed = importlib.util.module_from_spec(seed_spec)
    seed_spec.loader.exec_module(seed)
    import snowflake.connector

    return snowflake.connector.connect(**seed._connection_config())


def _probe(cursor: Any, label: str, query: str, limit: int) -> list[str]:
    """COUNT + LIMIT probes for one generated query; returns panel columns."""
    cursor.execute(f"SELECT COUNT(*) FROM ({query})")
    count = cursor.fetchone()[0]
    cursor.execute(f"SELECT * FROM ({query}) LIMIT {limit}")
    columns = [d[0].lower() for d in cursor.description]
    rows = cursor.fetchall()
    print(f"== {label}: {count} rows ==")
    print(f"   columns ({len(columns)}): {', '.join(columns)}")
    for row in rows:
        print(f"   {row}")
    print()
    return columns


def _warehouse_columns(cursor: Any, ref: str) -> list[str]:
    database, schema, table = ref.split(".")
    cursor.execute(
        f"SELECT column_name FROM {database}.INFORMATION_SCHEMA.COLUMNS "
        f"WHERE table_schema = '{schema.upper()}' AND table_name = '{table.upper()}' "
        "ORDER BY ordinal_position"
    )
    return [row[0].lower() for row in cursor.fetchall()]


def _report_pruning(cursor: Any, refs: dict[str, str], panel_columns: set[str]) -> None:
    print("== per-table pruning (warehouse columns vs the panel) ==")
    for name, ref in refs.items():
        in_warehouse = _warehouse_columns(cursor, ref)
        if not in_warehouse:
            print(f"   {name} ({ref}): NOT FOUND in information_schema - check grants/identifier")
            continue
        pruned = [c for c in in_warehouse if c not in panel_columns]
        note = f"pruned at source: {', '.join(pruned)}" if pruned else "all columns enter"
        print(f"   {name} ({ref}): {len(in_warehouse)} columns; {note}")
    print()


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    parser.add_argument(
        "--execute",
        action="store_true",
        help="run bounded probes against the warehouse (default: print SQL only)",
    )
    parser.add_argument(
        "--anchor",
        default="2026-06-01T00:00:00Z",
        help="ISO anchor the windows resolve against (default matches the seeded demo)",
    )
    parser.add_argument(
        "--sample-fraction",
        type=float,
        default=0.01,
        help="entity sample fraction for the probes (default 0.01; use 1.0 for none)",
    )
    parser.add_argument("--limit", type=int, default=5, help="sample rows to fetch per probe")
    args = parser.parse_args(argv)

    anchor = datetime.fromisoformat(args.anchor.replace("Z", "+00:00")).astimezone(UTC)
    dataset, scoring = _load_specs()
    refs_by_name = _table_refs()
    assert dataset.inputs is not None and scoring.inputs is not None
    refs_by_ref = {
        ref: refs_by_name[_source_name(ref)]
        for ref in (
            dataset.inputs.spine,
            dataset.inputs.label_source,
            *dataset.inputs.feature_sources,
            scoring.inputs.spine,
            *scoring.inputs.feature_sources,
        )
    }

    queries = _dataset_queries(dataset, refs_by_ref, anchor, args.sample_fraction)
    batch = _scoring_batch_query(scoring, refs_by_ref, anchor, args.sample_fraction)

    print(f"Anchor {format_ts(anchor)}, sample_fraction {args.sample_fraction}\n")
    for split, sql in sorted(queries.items()):
        print(f"-- training split '{split}' --\n{sql}\n")
    print(f"-- scoring batch --\n{batch}\n")

    if not args.execute:
        print("(dry run - pass --execute to run bounded probes against the warehouse)")
        return 0

    connection = _connect()
    try:
        cursor = connection.cursor()
        try:
            panel: set[str] = set()
            for split, sql in sorted(queries.items()):
                panel.update(_probe(cursor, f"training '{split}'", sql, args.limit))
            panel.update(_probe(cursor, "scoring batch", batch, args.limit))
            _report_pruning(cursor, refs_by_name, panel)
        finally:
            cursor.close()
    finally:
        connection.close()
    print("done: joins, windows, sampling, and per-table pruning verified in-warehouse")
    return 0


if __name__ == "__main__":
    sys.exit(main())
