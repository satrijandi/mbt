"""Unit tests for built-in dataset checks (mbt/quality/checks.py)."""

import pyarrow as pa

from mbt.contracts import DatasetSpec
from mbt.quality.checks import run_checks
from mbt_adapter_base.datasets import InMemoryDatasetHandle


def _spec(checks: list) -> DatasetSpec:
    return DatasetSpec.model_validate(
        {
            "name": "unit_ds",
            "source": "source('a', 'b')",
            "label": {"column": "y"},
            "split": {
                "strategy": "temporal",
                "time_column": "t",
                "train": "-30d:-7d",
                "test": "-7d:now",
            },
            "checks": checks,
        }
    )


def _handle() -> InMemoryDatasetHandle:
    # 'x' is constant so the auto-appended leakage scan sees a NULL correlation
    table = pa.table({"x": [1.0, 1.0, 1.0, 1.0], "y": [0, 1, 0, 1]})
    return InMemoryDatasetHandle({"train": table}, label_column="y")


def test_schema_check_reports_missing_typed_column() -> None:
    spec = _spec([{"schema": {"columns": {"absent": "int64", "y": "int64"}}}])
    results = {r.name: r for r in run_checks(spec, _handle(), {}, resource="dataset.unit")}
    schema = results["schema"]
    assert not schema.passed
    assert "missing column 'absent'" in schema.message


def test_no_future_columns_without_windows_passes() -> None:
    spec = _spec(["no_future_columns"])
    results = {r.name: r for r in run_checks(spec, _handle(), {}, resource="dataset.unit")}
    check = results["no_future_columns"]
    assert check.passed
    assert "no temporal windows" in check.message


def test_no_future_columns_skips_absent_splits() -> None:
    spec = _spec(["no_future_columns"])
    windows = {"windows": {"validation": ["2026-01-01T00:00:00Z", "2026-02-01T00:00:00Z"]}}
    results = {r.name: r for r in run_checks(spec, _handle(), windows, resource="dataset.unit")}
    check = results["no_future_columns"]
    assert check.passed and check.message == ""
