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

from mbt_adapter_base import DatasetSpec, ScoringInputSpec
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


def base_relation(spec: DatasetSpec, table_refs: Mapping[str, str]) -> tuple[str, list[str]]:
    """FROM clause plus columns to project away afterwards.

    One relation, whatever built it (ADR-29). The second element is the
    project-away list, kept because ``split_queries`` still takes it and the
    scoring path shares the shape; nothing populates it today.
    """
    return table_refs[spec.source], []


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
    # sample_key is required and validated non-empty on the spec (ADR-29), so
    # there is no keyless path left to reject here.
    bucket = key_hash_expr(spec.sample_key_columns, salt=str(spec.split.seed or 0))
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


def scoring_relation(spec: ScoringInputSpec, table_refs: Mapping[str, str]) -> str:
    """FROM clause for an unlabeled scoring batch (ADR-20): one relation, the
    serving twin of the training panel. Never a label join - a scoring input
    has no label by design (ADR-29)."""
    return table_refs[spec.source]


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
