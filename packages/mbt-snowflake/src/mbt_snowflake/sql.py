"""SQL assembly for the Snowflake adapter (pure functions, golden-testable).

Everything user-controllable that lands in SQL is either validated as an
identifier or passed through deliberately (dataset ``filters`` are raw SQL
fragments by design, same trust model as the local DuckDB adapter).

Sampling and random splits hash a declared key with ``MD5_NUMBER_LOWER64``
(an official Snowflake function returning the lower 64 bits of an MD5 as an
integer): fully deterministic across runs, warehouses, and releases -
unlike ``SAMPLE ... REPEATABLE``, which is only stable for block sampling
over unchanging physical layout.
"""

import re
from collections.abc import Mapping
from datetime import datetime

from mbt_adapter_base import DatasetSpec, FeatureEntry, ScoringInputSpec, parse_time_offset
from mbt_adapter_base.materialization import SAMPLE_MODULUS

_IDENTIFIER_RE = re.compile(r"^[A-Za-z_][A-Za-z0-9_$]*$")
_TABLE_REF_RE = re.compile(r"^[A-Za-z_][A-Za-z0-9_$]*(\.[A-Za-z_][A-Za-z0-9_$]*){0,2}$")


class SnowflakeSQLError(ValueError):
    """Invalid identifier or unbuildable query."""


def validate_column(name: str) -> str:
    if not _IDENTIFIER_RE.match(name):
        raise SnowflakeSQLError(
            f"invalid column identifier {name!r}: only letters, digits, '_' and '$' "
            "are allowed (unquoted Snowflake identifiers)"
        )
    return name


def qualify_table(identifier: str, database: str | None, schema: str | None) -> str:
    """Qualify a table identifier against the configured database/schema."""
    if not _TABLE_REF_RE.match(identifier):
        raise SnowflakeSQLError(
            f"invalid table identifier {identifier!r}: expected "
            "[DATABASE.][SCHEMA.]TABLE with plain identifiers"
        )
    dots = identifier.count(".")
    if dots == 2:
        return identifier
    if dots == 1:
        if not database:
            raise SnowflakeSQLError(
                f"table {identifier!r} needs a database: qualify it fully or set "
                "'database' in the adapter config"
            )
        return f"{database}.{identifier}"
    if not database or not schema:
        raise SnowflakeSQLError(
            f"table {identifier!r} needs database and schema: qualify it or set "
            "'database'/'schema' in the adapter config"
        )
    return f"{database}.{schema}.{identifier}"


def key_hash_expr(key_columns: list[str], salt: str = "") -> str:
    """Deterministic 0..SAMPLE_MODULUS-1 bucket from a stable row key.

    ``MD5_NUMBER_LOWER64`` (the unsigned lower 64 bits of the md5) over the
    '|'-joined preimage is the canonical cross-adapter digest (F19): the local
    DuckDB adapter and Spark compute the identical value, so the same key
    lands in the same bucket on every backend.
    """
    if not key_columns:
        raise SnowflakeSQLError("a non-empty key is required for hashing")
    parts = ", ".join(f"COALESCE(CAST({validate_column(c)} AS VARCHAR), '')" for c in key_columns)
    if salt:
        safe_salt = salt.replace("'", "''")
        parts = f"'{safe_salt}', {parts}"
    return f"MOD(MD5_NUMBER_LOWER64(CONCAT_WS('|', {parts})), {SAMPLE_MODULUS})"


def sampling_predicate(key_columns: list[str], fraction: float) -> str:
    """Push-down reproducible sampling: same fraction -> same rows; smaller
    fractions are subsets of larger ones (threshold hashing)."""
    threshold = int(fraction * SAMPLE_MODULUS)
    return f"{key_hash_expr(key_columns)} < {threshold}"


#: time_offset units -> SQL interval keywords (calendar month included).
_INTERVAL_UNITS = {"mo": "MONTH", "d": "DAY", "w": "WEEK", "h": "HOUR"}


def _interval_sql(count: int, unit: str) -> str:
    """``(1, "mo")`` -> ``+ INTERVAL '1 MONTH'`` (sign as the operator).

    The quoted-interval spelling is shared by Snowflake and DuckDB, so the
    emulation tests run the generated SQL verbatim.
    """
    operator = "-" if count < 0 else "+"
    return f"{operator} INTERVAL '{abs(count)} {_INTERVAL_UNITS[unit]}'"


def _feature_relation(table_ref: str, entry: FeatureEntry) -> str:
    """The joinable relation for one feature table: the bare table, or a
    projecting subquery when the entry declares ``columns``/``exclude``
    (ADR-25). The projection runs INSIDE the warehouse query, so pruned
    columns of a wide gold table are never scanned into the panel."""
    keep = entry.keep_columns
    if keep is not None:
        cols = ", ".join(validate_column(c) for c in keep)
        return f"(SELECT {cols} FROM {table_ref})"
    if entry.exclude is not None:
        cols = ", ".join(validate_column(c) for c in entry.exclude)
        return f"(SELECT * EXCLUDE ({cols}) FROM {table_ref})"
    return table_ref


def _spine_relation(spec: DatasetSpec, table_refs: Mapping[str, str]) -> str:
    """The spine + feature USING joins, before any label join (shared by
    ``base_relation`` and the label-join coverage counts, F21)."""
    assert spec.inputs is not None
    join_kind = "LEFT JOIN" if spec.inputs.join == "left" else "JOIN"
    sql = f"{table_refs[spec.inputs.spine]} AS mbt_spine"
    for i, entry in enumerate(spec.inputs.feature_entries):
        using = ", ".join(validate_column(c) for c in entry.using)
        relation = _feature_relation(table_refs[entry.source], entry)
        sql += f" {join_kind} {relation} AS mbt_f{i} USING ({using})"
    return sql


def base_relation(spec: DatasetSpec, table_refs: Mapping[str, str]) -> tuple[str, list[str]]:
    """FROM clause plus columns to project away afterwards.

    Single table, or spine + feature USING joins in declaration order; a
    population-spine label joins last through a rename-project subquery
    (ADR-22): its join columns are renamed, matched with ON so the
    time_offset can shift the spine's time column, and excluded from the
    output by the caller.
    """
    if spec.inputs is None:
        assert spec.source is not None
        return table_refs[spec.source], []
    sql = _spine_relation(spec, table_refs)
    if spec.inputs.population is None:
        return sql, []
    renames = {
        validate_column(c): f"__mbt_lbl{i}" for i, c in enumerate(spec.inputs.label_join_columns)
    }
    rename_sql = ", ".join(f"{c} AS {alias}" for c, alias in renames.items())
    offset = spec.inputs.label_time_offset
    conditions = []
    for column, alias in renames.items():
        if offset is not None and column == spec.split.time_column:
            count, unit = parse_time_offset(offset)
            conditions.append(
                f"CAST({alias} AS TIMESTAMP) = "
                f"CAST({column} AS TIMESTAMP) {_interval_sql(count, unit)}"
            )
        else:
            conditions.append(f"{alias} = {column}")
    sql += (
        f" JOIN (SELECT * RENAME ({rename_sql}) FROM "
        f"{table_refs[spec.inputs.label_source]}) AS mbt_label "
        f"ON {' AND '.join(conditions)}"
    )
    return sql, list(renames.values())


def _iso_to_ntz(iso: str) -> str:
    """ISO-8601 (Z-suffixed, UTC) -> TIMESTAMP_NTZ literal text."""
    ts = datetime.fromisoformat(iso.replace("Z", "+00:00"))
    return ts.replace(tzinfo=None).isoformat(sep=" ")


def split_queries(
    spec: DatasetSpec,
    relation: str,
    where: list[str],
    resolved_windows: Mapping[str, tuple[str, str]],
    exclude: list[str] | None = None,
) -> dict[str, str]:
    """One SELECT per split, filters/sampling/split predicates pushed down."""
    queries: dict[str, str] = {}
    base_where = list(where)
    select = f"* EXCLUDE ({', '.join(exclude)})" if exclude else "*"

    if spec.split.strategy.value == "temporal":
        assert spec.split.time_column is not None
        time_sql = f"CAST({validate_column(spec.split.time_column)} AS TIMESTAMP_NTZ)"
        for split, (start, end) in sorted(resolved_windows.items()):
            predicates = [
                *base_where,
                f"{time_sql} >= TO_TIMESTAMP_NTZ('{_iso_to_ntz(start)}')",
                f"{time_sql} < TO_TIMESTAMP_NTZ('{_iso_to_ntz(end)}')",
            ]
            queries[split] = f"SELECT {select} FROM {relation} WHERE {' AND '.join(predicates)}"
        return queries

    # random strategy: deterministic hash buckets over the sample key.
    # Proportions are approximate (each row lands in a split independently);
    # exact-fraction ranking is not worth a full-table window sort in the
    # warehouse. Deterministic and leak-free is what matters here.
    keys = spec.sample_key_columns
    if not keys:
        raise SnowflakeSQLError(
            "a random split on Snowflake needs 'sample_key' (or inputs.join_key) "
            "as the stable row identity to hash"
        )
    bucket = key_hash_expr(keys, salt=str(spec.split.seed or 0))
    fractions: dict[str, float] = {"train": float(spec.split.train)}
    if spec.split.validation is not None:
        fractions["validation"] = float(spec.split.validation)
    fractions["test"] = float(spec.split.test)
    low = 0.0
    entries = list(fractions.items())
    for index, (split, fraction) in enumerate(entries):
        lo = int(low * SAMPLE_MODULUS)
        # the final split's upper bound is pinned to the modulus so no bucket
        # can fall through a float-accumulation gap (mirrored by the local
        # adapter, so the two backends compute identical edges - F19)
        hi = SAMPLE_MODULUS if index == len(entries) - 1 else int((low + fraction) * SAMPLE_MODULUS)
        predicates = [*base_where, f"{bucket} >= {lo}", f"{bucket} < {hi}"]
        queries[split] = f"SELECT {select} FROM {relation} WHERE {' AND '.join(predicates)}"
        low += fraction
    return queries


def coverage_queries(spec: DatasetSpec, table_refs: Mapping[str, str]) -> tuple[str, str] | None:
    """``(spine_count_sql, matched_count_sql)`` for the label-join coverage
    statistic (F21), or None when the dataset has no population-spine label
    join. Counted before filters/sampling/windows so the ratio isolates the
    inner label join's silent drop (labels off the offset grid)."""
    if spec.inputs is None or spec.inputs.population is None:
        return None
    matched, _ = base_relation(spec, table_refs)
    spine = _spine_relation(spec, table_refs)
    return (f"SELECT COUNT(*) FROM {spine}", f"SELECT COUNT(*) FROM {matched}")


def scoring_relation(spec: ScoringInputSpec, table_refs: Mapping[str, str]) -> str:
    """FROM clause for an unlabeled scoring batch (ADR-20): a single source, or a
    spine + feature USING joins in declaration order. Never a label join - a
    scoring input has no label by design (contrast ``base_relation``)."""
    if spec.inputs is None:
        assert spec.source is not None
        return table_refs[spec.source]
    join_kind = "LEFT JOIN" if spec.inputs.join == "left" else "JOIN"
    sql = f"{table_refs[spec.inputs.spine]} AS mbt_spine"
    for i, entry in enumerate(spec.inputs.feature_entries):
        using = ", ".join(validate_column(c) for c in entry.using)
        relation = _feature_relation(table_refs[entry.source], entry)
        sql += f" {join_kind} {relation} AS mbt_f{i} USING ({using})"
    return sql


def scoring_query(
    spec: ScoringInputSpec,
    table_refs: Mapping[str, str],
    where: list[str],
    window: tuple[str, str] | None,
) -> str:
    """One SELECT materializing the unlabeled scoring batch (ADR-20/23).

    Filters push down; a ``[start, end)`` window on ``time_column`` is applied
    when the scoring node resolved a ``score`` window (same half-open temporal
    predicate as the training splits)."""
    predicates = list(where)
    if window is not None and spec.time_column is not None:
        start, end = window
        time_sql = f"CAST({validate_column(spec.time_column)} AS TIMESTAMP_NTZ)"
        predicates += [
            f"{time_sql} >= TO_TIMESTAMP_NTZ('{_iso_to_ntz(start)}')",
            f"{time_sql} < TO_TIMESTAMP_NTZ('{_iso_to_ntz(end)}')",
        ]
    clause = f" WHERE {' AND '.join(predicates)}" if predicates else ""
    return f"SELECT * FROM {scoring_relation(spec, table_refs)}{clause}"
