"""Built-in dataset checks (TSD §11.1, FR-TEST-05).

Checks run in the coordinator against the materialized dataset, right after
a dataset build. Failures set node status ``test_failed`` (exit code 2).
"""

import math
from datetime import datetime
from typing import Any, Protocol

import duckdb
import pyarrow as pa

from mbt.contracts import CheckSpec, DatasetSpec, ScoringSpec, TestResult
from mbt.events import get_bus
from mbt.events.models import LogMessage

_TIMESTAMP_TYPES = ("timestamp", "date")


class _CheckableHandle(Protocol):
    def splits(self) -> set[str]: ...

    def read(self, split: str, columns: list[str] | None = None) -> pa.Table: ...


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
) -> list[TestResult]:
    checks: list[CheckSpec] = list(spec.checks)
    declared = {_normalize(check)[0] for check in checks}
    if "label_leakage_scan" not in declared:
        # Leakage guards are on by default: every dataset build scans for
        # label-associated features; declare the check to tune thresholds,
        # exclude reviewed columns, or opt out (`enabled: false`).
        checks.append("label_leakage_scan")
    return _run_named_checks(checks, spec, handle, resolved_windows, resource)


def run_scoring_checks(
    spec: ScoringSpec,
    handle: _CheckableHandle,
    resolved_windows: dict[str, Any],
    *,
    resource: str,
) -> list[TestResult]:
    """Label-free checks on a scoring input (ADR-20).

    No ``label_leakage_scan`` auto-append: there is no label to leak. The
    parser restricts scoring checks to the label-free subset and requires
    explicit columns for ``not_null``.
    """
    return _run_named_checks(list(spec.checks), spec, handle, resolved_windows, resource)


def _run_named_checks(
    checks: list[CheckSpec],
    spec: Any,
    handle: _CheckableHandle,
    resolved_windows: dict[str, Any],
    resource: str,
) -> list[TestResult]:
    results: list[TestResult] = []
    for check in checks:
        name, params = _normalize(check)
        if not params.get("enabled", True):
            results.append(TestResult(name=name, passed=True, message="check disabled"))
            continue
        runner = _CHECKS[name]
        try:
            results.append(runner(spec, handle, resolved_windows, params, resource))
        except Exception as exc:
            results.append(TestResult(name=name, passed=False, message=f"check raised: {exc!r}"))
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


def _check_no_future_columns(
    spec: Any,
    handle: _CheckableHandle,
    windows: dict[str, Any],
    params: dict[str, Any],
    resource: str,
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
_CHECKS = {
    "schema": _check_schema,
    "not_null": _check_not_null,
    "no_future_columns": _check_no_future_columns,
    "label_leakage_scan": _check_label_leakage_scan,
    "class_balance_report": _check_class_balance_report,
}
