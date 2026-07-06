"""Built-in dataset checks (TSD §11.1, FR-TEST-05).

Checks run in the coordinator against the materialized dataset, right after
a dataset build. Failures set node status ``test_failed`` (exit code 2).
"""

from datetime import datetime
from typing import Any, Protocol

import duckdb
import pyarrow as pa

from mbt.contracts import CheckSpec, DatasetSpec, TestResult
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
    results: list[TestResult] = []
    for check in spec.checks:
        name, params = _normalize(check)
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
    spec: DatasetSpec,
    handle: _CheckableHandle,
    windows: dict[str, Any],
    params: dict[str, Any],
    resource: str,
) -> TestResult:
    """Declared columns exist (and match declared arrow types when given)."""
    columns = params.get("columns", {})
    table = handle.read("train")
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
    spec: DatasetSpec,
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
    spec: DatasetSpec,
    handle: _CheckableHandle,
    windows: dict[str, Any],
    params: dict[str, Any],
    resource: str,
) -> TestResult:
    """No timestamp column may exceed the split boundary (leakage guard)."""
    resolved = windows.get("windows", windows) or {}
    ends = [bounds[1] for bounds in resolved.values() if isinstance(bounds, (list, tuple))]
    if not ends:
        return TestResult(
            name="no_future_columns",
            passed=True,
            message="no temporal windows to check against",
        )
    boundary = max(datetime.fromisoformat(str(e).replace("Z", "+00:00")) for e in ends)
    boundary_naive = boundary.replace(tzinfo=None)
    table = handle.read("train")
    problems = []
    con = duckdb.connect()
    try:
        con.register("t", table)
        for field in table.schema:
            if not str(field.type).startswith(_TIMESTAMP_TYPES):
                continue
            row = con.execute(f'SELECT CAST(max("{field.name}") AS TIMESTAMP) FROM t').fetchone()
            maximum = row[0] if row else None
            if maximum is not None and maximum > boundary_naive:
                problems.append(
                    f"column {field.name!r} reaches {maximum.isoformat()} "
                    f"beyond the split boundary {boundary_naive.isoformat()}"
                )
    finally:
        con.close()
    return TestResult(name="no_future_columns", passed=not problems, message="; ".join(problems))


def _check_label_leakage_scan(
    spec: DatasetSpec,
    handle: _CheckableHandle,
    windows: dict[str, Any],
    params: dict[str, Any],
    resource: str,
) -> TestResult:
    """Flag numeric features suspiciously associated with the label."""
    threshold = float(params.get("max_abs_correlation", 0.95))
    label = spec.label.column
    table = handle.read("train")
    con = duckdb.connect()
    try:
        con.register("t", table)
        problems = []
        for field in table.schema:
            if field.name == label:
                continue
            if not (
                str(field.type).startswith(("int", "uint", "float", "double", "decimal"))
                or str(field.type) == "bool"
            ):
                continue
            row = con.execute(
                f'SELECT corr(CAST("{field.name}" AS DOUBLE), CAST("{label}" AS DOUBLE)) FROM t'
            ).fetchone()
            corr = row[0] if row else None
            if corr is not None and abs(corr) >= threshold:
                problems.append(f"{field.name} (|corr|={abs(corr):.3f})")
    finally:
        con.close()
    return TestResult(
        name="label_leakage_scan",
        passed=not problems,
        message=("suspiciously label-associated features: " + ", ".join(problems))
        if problems
        else "",
    )


def _check_class_balance_report(
    spec: DatasetSpec,
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


_CHECKS = {
    "schema": _check_schema,
    "not_null": _check_not_null,
    "no_future_columns": _check_no_future_columns,
    "label_leakage_scan": _check_label_leakage_scan,
    "class_balance_report": _check_class_balance_report,
}

BUILTIN_CHECK_NAMES = frozenset(_CHECKS)
