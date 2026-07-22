"""Built-in dataset checks (TSD §11.1, FR-TEST-05).

Checks run in the coordinator against the materialized dataset, right after
a dataset build. Failures set node status ``test_failed`` (exit code 2).
"""

import math
from dataclasses import dataclass, field
from datetime import datetime
from typing import Any, Protocol

import duckdb
import pyarrow as pa

from mbt.contracts import CheckSpec, DatasetSpec, ScoringSpec, TestResult
from mbt.events import get_bus
from mbt.events.models import CheckEvaluated, LogMessage

_TIMESTAMP_TYPES = ("timestamp", "date")


class _CheckableHandle(Protocol):
    def splits(self) -> set[str]: ...

    def read(self, split: str, columns: list[str] | None = None) -> pa.Table: ...


@dataclass(frozen=True)
class SourceAccess:
    """Pre-join source reach for source-level checks (F2/F21).

    Checks normally see only the materialized splits; ``unique`` with a
    ``source:`` param and ``relationships`` need the RAW source tables before
    the join fans anything out, so the runner threads in the node's source
    tables (keyed ``group.name``) plus the resolved data adapter, which is
    probed for the optional source-check methods
    (``count_source_duplicates`` / ``read_source_distinct``).
    """

    tables: dict[str, Any] = field(default_factory=dict)
    adapter: Any = None

    def resolve(self, name: str, check: str) -> tuple[Any, str]:
        """The source table for ``name`` (``group.name``, or a bare table name
        when unambiguous among this dataset's sources)."""
        if name in self.tables:
            return self.tables[name], name
        matches = [key for key in self.tables if key.split(".", 1)[-1] == name]
        if len(matches) == 1:
            return self.tables[matches[0]], matches[0]
        raise ValueError(
            f"{check}: unknown source {name!r}; this resource's sources: "
            f"{', '.join(sorted(self.tables)) or '(none)'}"
        )


def _normalize(check: CheckSpec) -> tuple[str, dict[str, Any]]:
    if isinstance(check, str):
        return check, {}
    name = next(iter(check))
    return name, dict(check[name] or {})


def run_checks(
    spec: DatasetSpec,
    handle: _CheckableHandle,
    resolved_windows: dict[str, Any],
    *,
    resource: str,
    sources: SourceAccess | None = None,
) -> list[TestResult]:
    checks: list[CheckSpec] = list(spec.checks)
    declared = {_normalize(check)[0] for check in checks}
    if "label_leakage_scan" not in declared:
        # Leakage guards are on by default: every dataset build scans for
        # label-associated features; declare the check to tune thresholds,
        # exclude reviewed columns, or opt out (`enabled: false`).
        checks.append("label_leakage_scan")
    return _run_named_checks(checks, spec, handle, resolved_windows, resource, sources)


def run_scoring_checks(
    spec: ScoringSpec,
    handle: _CheckableHandle,
    resolved_windows: dict[str, Any],
    *,
    resource: str,
    sources: SourceAccess | None = None,
) -> list[TestResult]:
    """Label-free checks on a scoring input (ADR-20).

    No ``label_leakage_scan`` auto-append: there is no label to leak. The
    parser restricts scoring checks to the label-free subset and requires
    explicit columns for ``not_null``.
    """
    return _run_named_checks(list(spec.checks), spec, handle, resolved_windows, resource, sources)


def _run_named_checks(
    checks: list[CheckSpec],
    spec: Any,
    handle: _CheckableHandle,
    resolved_windows: dict[str, Any],
    resource: str,
    sources: SourceAccess | None = None,
) -> list[TestResult]:
    bus = get_bus()
    results: list[TestResult] = []
    for check in checks:
        name, params = _normalize(check)
        if not params.get("enabled", True):
            result = TestResult(name=name, passed=True, message="check disabled")
        else:
            runner = _CHECKS[name]
            try:
                result = runner(spec, handle, resolved_windows, params, resource, sources)
            except Exception as exc:
                result = TestResult(name=name, passed=False, message=f"check raised: {exc!r}")
        results.append(result)
        # One event per check so the pass/fail stream is visible on the bus,
        # mirroring GateEvaluated (a check with no message renders "PASS").
        bus.emit(
            CheckEvaluated(
                unique_id=resource,
                check=result.name,
                passed=result.passed,
                message=result.message,
            )
        )
    return results


def _connect_splits(handle: _CheckableHandle) -> tuple["duckdb.DuckDBPyConnection", list[str]]:
    con = duckdb.connect()
    splits = sorted(handle.splits())
    for split in splits:
        con.register(f"split_{split}", handle.read(split))
    return con, splits


def _check_schema(
    spec: Any,
    handle: _CheckableHandle,
    windows: dict[str, Any],
    params: dict[str, Any],
    resource: str,
    sources: "SourceAccess | None" = None,
) -> TestResult:
    """Declared columns exist (and match declared arrow types when given)."""
    columns = params.get("columns", {})
    split = "train" if "train" in handle.splits() else min(sorted(handle.splits()))
    table = handle.read(split)
    actual = {field.name: str(field.type) for field in table.schema}
    problems: list[str] = []
    if isinstance(columns, list):
        for name in columns:
            if name not in actual:
                problems.append(f"missing column {name!r}")
    else:
        for name, expected_type in columns.items():
            if name not in actual:
                problems.append(f"missing column {name!r}")
            elif expected_type and expected_type != actual[name]:
                problems.append(
                    f"column {name!r} has type {actual[name]!r}, expected {expected_type!r}"
                )
    return TestResult(name="schema", passed=not problems, message="; ".join(problems))


def _check_not_null(
    spec: Any,
    handle: _CheckableHandle,
    windows: dict[str, Any],
    params: dict[str, Any],
    resource: str,
    sources: "SourceAccess | None" = None,
) -> TestResult:
    columns = list(params.get("columns") or [spec.label.column])
    con, splits = _connect_splits(handle)
    try:
        problems = []
        for split in splits:
            for column in columns:
                row = con.execute(
                    f'SELECT count(*) FROM split_{split} WHERE "{column}" IS NULL'
                ).fetchone()
                nulls = int(row[0]) if row else 0
                if nulls:
                    problems.append(f"{split}.{column}: {nulls} null(s)")
        return TestResult(name="not_null", passed=not problems, message="; ".join(problems))
    finally:
        con.close()


def _check_unique(
    spec: Any,
    handle: _CheckableHandle,
    windows: dict[str, Any],
    params: dict[str, Any],
    resource: str,
    sources: "SourceAccess | None" = None,
) -> TestResult:
    """Each listed column must hold no duplicated (non-null) value in any split.

    dbt-parity ``unique``: this catches a multi-table join that fanned out the
    population spine (a feature/label table that is not unique on its join key
    multiplies rows), which otherwise silently over-weights an entity in both
    training and the reported metric (F2). Nulls are ignored, as in dbt - pair
    with ``not_null`` when the key must also be present.

    With ``source: <group.name>`` the check runs PRE-JOIN against the raw
    source table instead, treating ``columns`` as one composite key - the 1:1
    join-cardinality contract that stops the fan-out before it happens (F2):
    a source unique on its ``using`` key cannot multiply the spine.
    """
    columns = list(params.get("columns") or [])
    if not columns:
        return TestResult(
            name="unique",
            passed=False,
            message="unique check requires explicit 'columns', e.g. unique: {columns: [user_id]}",
        )
    source_name = params.get("source")
    if source_name:
        if sources is None:
            return TestResult(
                name="unique",
                passed=False,
                message="source-level unique needs source access; run it via mbt build/test",
            )
        table, label = sources.resolve(str(source_name), "unique")
        counter = getattr(sources.adapter, "count_source_duplicates", None)
        if counter is None:
            return TestResult(
                name="unique",
                passed=False,
                message=(
                    f"data adapter {getattr(sources.adapter, 'name', '?')!r} does not "
                    "support source-level checks (count_source_duplicates)"
                ),
            )
        duplicates = int(counter(table, columns))
        key = ", ".join(columns)
        if duplicates:
            return TestResult(
                name="unique",
                passed=False,
                message=(
                    f"source {label}: composite key ({key}): {duplicates} duplicated "
                    "key(s) - a non-unique join key fans out the spine (F2)"
                ),
            )
        return TestResult(name="unique", passed=True, message=f"source {label}: ({key}) unique")
    con, splits = _connect_splits(handle)
    try:
        problems = []
        for split in splits:
            for column in columns:
                row = con.execute(
                    f'SELECT count(*) FROM (SELECT "{column}" FROM split_{split} '
                    f'WHERE "{column}" IS NOT NULL GROUP BY "{column}" HAVING count(*) > 1)'
                ).fetchone()
                dup = int(row[0]) if row else 0
                if dup:
                    problems.append(f"{split}.{column}: {dup} duplicated value(s)")
        return TestResult(name="unique", passed=not problems, message="; ".join(problems))
    finally:
        con.close()


def _check_accepted_values(
    spec: Any,
    handle: _CheckableHandle,
    windows: dict[str, Any],
    params: dict[str, Any],
    resource: str,
    sources: "SourceAccess | None" = None,
) -> TestResult:
    """A column's non-null values must all lie in an allowed set, in every split.

    dbt-parity ``accepted_values``: a categorical feature or label that drifts to
    an unexpected level (a new plan code, a typo, an upstream enum change) is
    caught here rather than silently trained as a bogus category or dropped
    through a downstream join (F21). Nulls are ignored, as in dbt - pair with
    ``not_null`` when the value must also be present.
    """
    column = params.get("column")
    values = list(params.get("values") or [])
    if not column or not values:
        return TestResult(
            name="accepted_values",
            passed=False,
            message=(
                "accepted_values check requires a 'column' and non-empty 'values', "
                "e.g. accepted_values: {column: plan, values: [basic, pro]}"
            ),
        )
    con, splits = _connect_splits(handle)
    try:
        placeholders = ", ".join("?" for _ in values)
        problems = []
        for split in splits:
            row = con.execute(
                f'SELECT count(DISTINCT "{column}") FROM split_{split} '
                f'WHERE "{column}" IS NOT NULL AND "{column}" NOT IN ({placeholders})',
                values,
            ).fetchone()
            unexpected = int(row[0]) if row else 0
            if unexpected:
                problems.append(f"{split}.{column}: {unexpected} unexpected value(s)")
        return TestResult(name="accepted_values", passed=not problems, message="; ".join(problems))
    finally:
        con.close()


def _check_row_count(
    spec: Any,
    handle: _CheckableHandle,
    windows: dict[str, Any],
    params: dict[str, Any],
    resource: str,
    sources: "SourceAccess | None" = None,
) -> TestResult:
    """The materialized dataset's total row count must lie within declared bounds.

    A volume floor/ceiling contract (F21): the temporal label join is exact
    equality on ``time_column + offset``, so if label timestamps drift off the
    offset grid the rows drop silently through the inner join and the only signal
    is a degraded metric (or, if all drop, a bare '0 rows' error). Declaring
    ``row_count: {min: N}`` turns a catastrophic volume drop into a loud build
    failure. Counts every split, since they are proportional slices of one build.
    """
    minimum = params.get("min")
    maximum = params.get("max")
    if minimum is None and maximum is None:
        return TestResult(
            name="row_count",
            passed=False,
            message="row_count check requires 'min' and/or 'max', e.g. row_count: {min: 1000}",
        )
    total = sum(handle.read(split).num_rows for split in handle.splits())
    problems = []
    if minimum is not None and total < minimum:
        problems.append(f"{total} rows below the minimum {minimum}")
    if maximum is not None and total > maximum:
        problems.append(f"{total} rows above the maximum {maximum}")
    return TestResult(name="row_count", passed=not problems, message="; ".join(problems))


def _check_freshness(
    spec: Any,
    handle: _CheckableHandle,
    windows: dict[str, Any],
    params: dict[str, Any],
    resource: str,
    sources: "SourceAccess | None" = None,
) -> TestResult:
    """The newest materialized row must be within ``max_lag`` of the anchor.

    A source-freshness / staleness guard (F21): a scheduled retrain has no
    upstream-is-stale signal ("freshness arrives as new snapshots"), so if a
    source stops updating the retrain silently trains on stale data. The dataset
    windows end at the manifest anchor ("now"), so the newest materialized value
    of the split time column lagging that anchor by more than ``max_lag`` means
    the upstream is stale. Temporal only (needs a time column + windows); works on
    a dataset (``split.time_column``) and a scoring input (``input.time_column``),
    so a stale nightly SCORING batch is caught too, not just a stale retrain.
    """
    max_lag = params.get("max_lag")
    time_column = getattr(getattr(spec, "split", None), "time_column", None) or getattr(
        getattr(spec, "input", None), "time_column", None
    )
    resolved = windows.get("windows", windows) or {}
    ends = [
        datetime.fromisoformat(str(bounds[1]).replace("Z", "+00:00")).replace(tzinfo=None)
        for bounds in resolved.values()
        if isinstance(bounds, (list, tuple))
    ]
    if not max_lag or not time_column or not ends:
        return TestResult(
            name="freshness",
            passed=False,
            message=(
                "freshness check needs 'max_lag' and a temporal split "
                "(time_column + windows), e.g. freshness: {max_lag: 2d}"
            ),
        )
    anchor = max(ends)
    con = duckdb.connect()
    try:
        newest = None
        for split in sorted(handle.splits()):
            con.register(f"f_{split}", handle.read(split))
            row = con.execute(
                f'SELECT CAST(max("{time_column}") AS TIMESTAMP) FROM f_{split}'
            ).fetchone()
            candidate = row[0] if row else None
            if candidate is not None and (newest is None or candidate > newest):
                newest = candidate
    finally:
        con.close()
    from mbt.compile.windows import subtract_duration

    cutoff = subtract_duration(anchor, max_lag)
    if newest is not None and newest >= cutoff:
        return TestResult(name="freshness", passed=True, message="")
    latest = newest.isoformat() if newest is not None else "no rows"
    return TestResult(
        name="freshness",
        passed=False,
        message=(
            f"newest {time_column} is {latest}, more than {max_lag} before the anchor "
            f"{anchor.isoformat()} - the upstream may be stale"
        ),
    )


def _check_no_future_columns(
    spec: Any,
    handle: _CheckableHandle,
    windows: dict[str, Any],
    params: dict[str, Any],
    resource: str,
    sources: "SourceAccess | None" = None,
) -> TestResult:
    """No timestamp column may exceed its own split's window end.

    Checking each split against its OWN boundary catches train/test overlap
    (a train row carrying a timestamp inside the test window is temporal
    leakage), not just absolutely-future values beyond the newest window.
    """
    resolved = windows.get("windows", windows) or {}
    boundaries = {
        split: datetime.fromisoformat(str(bounds[1]).replace("Z", "+00:00")).replace(tzinfo=None)
        for split, bounds in resolved.items()
        if isinstance(bounds, (list, tuple))
    }
    if not boundaries:
        return TestResult(
            name="no_future_columns",
            passed=True,
            message="no temporal windows to check against",
        )
    problems = []
    con = duckdb.connect()
    try:
        for split, boundary in sorted(boundaries.items()):
            if split not in handle.splits():
                continue
            table = handle.read(split)
            con.register(f"t_{split}", table)
            for field in table.schema:
                if not str(field.type).startswith(_TIMESTAMP_TYPES):
                    continue
                row = con.execute(
                    f'SELECT CAST(max("{field.name}") AS TIMESTAMP) FROM t_{split}'
                ).fetchone()
                maximum = row[0] if row else None
                if maximum is not None and maximum > boundary:
                    problems.append(
                        f"{split}.{field.name} reaches {maximum.isoformat()} beyond the "
                        f"{split} window end {boundary.isoformat()} (temporal leakage)"
                    )
    finally:
        con.close()
    return TestResult(name="no_future_columns", passed=not problems, message="; ".join(problems))


def _cramers_v(con: "duckdb.DuckDBPyConnection", column: str, label: str) -> float | None:
    """Cramér's V of a categorical column against the label, from the
    contingency table (pure arithmetic; mbt-core carries no scipy).

    For two binary variables V equals |phi| - the Pearson correlation of the
    indicators - so the numeric scan's thresholds transfer unchanged.
    Returns None for degenerate columns (fewer than two levels or classes)
    and for quasi-identifiers (more than half the rows unique), whose V
    saturates spuriously without indicating leakage.
    """
    rows = con.execute(
        f'SELECT CAST("{column}" AS VARCHAR), CAST("{label}" AS VARCHAR), COUNT(*) FROM t '
        f'WHERE "{column}" IS NOT NULL AND "{label}" IS NOT NULL GROUP BY 1, 2'
    ).fetchall()
    counts = {(str(level), str(cls)): int(n) for level, cls, n in rows}
    levels = sorted({key[0] for key in counts})
    classes = sorted({key[1] for key in counts})
    total = sum(counts.values())
    if len(levels) < 2 or len(classes) < 2 or total == 0:
        return None
    if len(levels) > total * 0.5:
        return None  # quasi-identifier, not a categorical feature
    level_totals = {lv: sum(counts.get((lv, c), 0) for c in classes) for lv in levels}
    class_totals = {c: sum(counts.get((lv, c), 0) for lv in levels) for c in classes}
    chi2 = 0.0
    for lv in levels:
        for c in classes:
            expected = level_totals[lv] * class_totals[c] / total
            chi2 += (counts.get((lv, c), 0) - expected) ** 2 / expected
    return math.sqrt(chi2 / (total * (min(len(levels), len(classes)) - 1)))


def _check_label_leakage_scan(
    spec: Any,
    handle: _CheckableHandle,
    windows: dict[str, Any],
    params: dict[str, Any],
    resource: str,
    sources: "SourceAccess | None" = None,
) -> TestResult:
    """Flag features suspiciously associated with the label.

    Numeric columns are screened with ``|corr|``, categorical (string)
    columns with Cramér's V - the same 0-1 scale, so one two-tier bar covers
    both: ``>= max_abs_correlation`` (default 0.95) fails the build; the
    warn band ``[warn_abs_correlation, max)`` (default 0.85) logs a warning
    and is recorded without failing. ``exclude`` skips reviewed columns.
    Runs on every dataset build unless opted out (``enabled``).
    """
    threshold = float(params.get("max_abs_correlation", 0.95))
    warn_threshold = float(params.get("warn_abs_correlation", 0.85))
    excluded = set(params.get("exclude") or [])
    label = spec.label.column
    table = handle.read("train")
    con = duckdb.connect()
    try:
        con.register("t", table)
        problems: list[str] = []
        suspects: list[str] = []
        for field in table.schema:
            if field.name == label or field.name in excluded:
                continue
            type_name = str(field.type)
            association: float | None
            if type_name.startswith(("int", "uint", "float", "double", "decimal")) or (
                type_name == "bool"
            ):
                row = con.execute(
                    f'SELECT corr(CAST("{field.name}" AS DOUBLE), CAST("{label}" AS DOUBLE)) FROM t'
                ).fetchone()
                corr = row[0] if row else None
                association = None if corr is None else abs(corr)
                stat = "|corr|"
            elif type_name in ("string", "large_string") or type_name.startswith("dictionary"):
                association = _cramers_v(con, field.name, label)
                stat = "V"
            else:
                continue
            if association is None:
                continue
            if association >= threshold:
                problems.append(f"{field.name} ({stat}={association:.3f})")
            elif association >= warn_threshold:
                suspects.append(f"{field.name} ({stat}={association:.3f})")
    finally:
        con.close()
    parts: list[str] = []
    if problems:
        parts.append("suspiciously label-associated features: " + ", ".join(problems))
    if suspects:
        warn = "features in the warn band: " + ", ".join(suspects)
        parts.append(warn)
        get_bus().emit(
            LogMessage(level="warn", unique_id=resource, message=f"label_leakage_scan: {warn}")
        )
    return TestResult(name="label_leakage_scan", passed=not problems, message="; ".join(parts))


def _check_class_balance_report(
    spec: Any,
    handle: _CheckableHandle,
    windows: dict[str, Any],
    params: dict[str, Any],
    resource: str,
    sources: "SourceAccess | None" = None,
) -> TestResult:
    """Report-only: emits the label balance, never fails (TSD §11.1)."""
    profile = getattr(handle, "profile", None)
    balance: dict[str, float] | None = None
    if callable(profile):
        balance = profile().label_balance
    message = (
        "label balance (train): "
        + ", ".join(f"{k}={v:.3%}" for k, v in sorted((balance or {}).items()))
        if balance
        else "label balance unavailable"
    )
    get_bus().emit(LogMessage(unique_id=resource, message=message))
    return TestResult(name="class_balance_report", passed=True, message=message)


#: Name -> implementation. Its keys must match BUILTIN_CHECK_NAMES exactly
#: (pinned by test_check_names_match_dispatch_table); the names live in the
#: import-light check_names module so the parser can validate against them
#: without importing this module's duckdb/pyarrow dependencies.
def _check_relationships(
    spec: Any,
    handle: _CheckableHandle,
    windows: dict[str, Any],
    params: dict[str, Any],
    resource: str,
    sources: "SourceAccess | None" = None,
) -> TestResult:
    """dbt-parity ``relationships`` (foreign key): every non-null value of
    ``column`` in the materialized splits must exist in ``field`` of the
    referenced source table ``to`` (F2/F21).

    The parent side is pulled as DISTINCT values through the data adapter
    (``read_source_distinct``), so the check works on any backend; size it for
    dimension-like parents (customers, plans), not fact tables.
    """
    column = params.get("column")
    to = params.get("to")
    field_name = params.get("field")
    if not (column and to and field_name):
        return TestResult(
            name="relationships",
            passed=False,
            message=(
                "relationships requires 'column', 'to', and 'field', e.g. "
                "relationships: {column: plan_id, to: lakehouse.plans, field: id}"
            ),
        )
    if sources is None:
        return TestResult(
            name="relationships",
            passed=False,
            message="relationships needs source access; run it via mbt build/test",
        )
    table, label = sources.resolve(str(to), "relationships")
    reader = getattr(sources.adapter, "read_source_distinct", None)
    if reader is None:
        return TestResult(
            name="relationships",
            passed=False,
            message=(
                f"data adapter {getattr(sources.adapter, 'name', '?')!r} does not "
                "support source-level checks (read_source_distinct)"
            ),
        )
    parent = set(reader(table, str(field_name)).column(0).to_pylist())
    con, splits = _connect_splits(handle)
    try:
        problems = []
        for split in splits:
            rows = con.execute(
                f'SELECT DISTINCT "{column}" FROM split_{split} WHERE "{column}" IS NOT NULL'
            ).fetchall()
            orphans = sorted(str(row[0]) for row in rows if row[0] not in parent)
            if orphans:
                sample = ", ".join(orphans[:3])
                problems.append(
                    f"{split}.{column}: {len(orphans)} value(s) not in "
                    f"{label}.{field_name} (e.g. {sample})"
                )
        return TestResult(name="relationships", passed=not problems, message="; ".join(problems))
    finally:
        con.close()


def _check_label_join_coverage(
    spec: Any,
    handle: _CheckableHandle,
    windows: dict[str, Any],
    params: dict[str, Any],
    resource: str,
    sources: "SourceAccess | None" = None,
) -> TestResult:
    """The label join must retain at least ``min_fraction`` of the spine (F21).

    The temporal label join is exact equality on ``time_column + offset``, so
    labels drifting off the offset grid silently drop spine rows through the
    inner join; the build records ``{spine_rows, matched_rows}`` (measured
    before filters/sampling/windows, so the ratio isolates the join drop) and
    this check turns a quiet partial drop into a loud failure.
    """
    min_fraction = params.get("min_fraction")
    if not isinstance(min_fraction, int | float) or not 0.0 < float(min_fraction) <= 1.0:
        return TestResult(
            name="label_join_coverage",
            passed=False,
            message=(
                "label_join_coverage requires min_fraction in (0, 1], e.g. "
                "label_join_coverage: {min_fraction: 0.95}"
            ),
        )
    coverage = getattr(handle, "label_join_coverage", None)
    if coverage is None:
        return TestResult(
            name="label_join_coverage",
            passed=False,
            message=(
                "no label-join coverage recorded: only population-spine datasets "
                "measure it (and older materializations lack it - rebuild)"
            ),
        )
    spine = int(coverage["spine_rows"])
    matched = int(coverage["matched_rows"])
    if spine == 0:
        return TestResult(
            name="label_join_coverage",
            passed=False,
            message="the population spine had 0 rows before the label join",
        )
    fraction = matched / spine
    return TestResult(
        name="label_join_coverage",
        passed=fraction >= float(min_fraction),
        message=(
            f"label join matched {matched} of {spine} spine rows "
            f"({fraction:.1%}); floor {float(min_fraction):.0%}"
        ),
    )


_CHECKS = {
    "schema": _check_schema,
    "not_null": _check_not_null,
    "unique": _check_unique,
    "accepted_values": _check_accepted_values,
    "relationships": _check_relationships,
    "row_count": _check_row_count,
    "freshness": _check_freshness,
    "label_join_coverage": _check_label_join_coverage,
    "no_future_columns": _check_no_future_columns,
    "label_leakage_scan": _check_label_leakage_scan,
    "class_balance_report": _check_class_balance_report,
}
