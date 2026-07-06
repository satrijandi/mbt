"""Built-in dataset check tests, positive and negative (S4-06)."""

from datetime import datetime

import pyarrow as pa

from mbt.contracts import DatasetSpec
from mbt.quality.checks import run_checks
from mbt_adapter_base.datasets import InMemoryDatasetHandle


def _handle(**overrides) -> InMemoryDatasetHandle:
    base = {
        "user_id": [1, 2, 3, 4],
        "snapshot_date": [datetime(2026, 1, d) for d in (1, 2, 3, 4)],
        "feature": [1.0, 2.0, 3.0, 4.0],
        "churned": [0, 1, 0, 1],
    }
    base.update(overrides)
    table = pa.table(base)
    return InMemoryDatasetHandle(
        {"train": table, "test": table}, label_column="churned", time_column="snapshot_date"
    )


def _spec(checks) -> DatasetSpec:
    return DatasetSpec.model_validate(
        {
            "name": "d",
            "source": "source('a', 'b')",
            "label": {"column": "churned"},
            "split": {"time_column": "snapshot_date", "train": "-30d:-7d", "test": "-7d:now"},
            "checks": checks,
        }
    )


WINDOWS = {
    "windows": {
        "train": ["2026-01-01T00:00:00Z", "2026-01-03T00:00:00Z"],
        "test": ["2026-01-03T00:00:00Z", "2026-01-05T00:00:00Z"],
    }
}


def _run(checks, handle):
    return {r.name: r for r in run_checks(_spec(checks), handle, WINDOWS, resource="d")}


def test_not_null_passes_and_fails() -> None:
    ok = _run([{"not_null": {"columns": ["churned"]}}], _handle())
    assert ok["not_null"].passed
    bad = _run(
        [{"not_null": {"columns": ["feature"]}}],
        _handle(feature=[1.0, None, 3.0, None]),
    )
    assert not bad["not_null"].passed
    assert "2 null" in bad["not_null"].message


def test_schema_check() -> None:
    ok = _run([{"schema": {"columns": ["churned", "feature"]}}], _handle())
    assert ok["schema"].passed
    missing = _run([{"schema": {"columns": ["nonexistent"]}}], _handle())
    assert not missing["schema"].passed
    typed = _run([{"schema": {"columns": {"feature": "double"}}}], _handle())
    assert typed["schema"].passed
    wrong_type = _run([{"schema": {"columns": {"feature": "int64"}}}], _handle())
    assert not wrong_type["schema"].passed


def test_no_future_columns() -> None:
    ok = _run(["no_future_columns"], _handle())
    assert ok["no_future_columns"].passed
    leaky = _handle(
        snapshot_date=[
            datetime(2026, 1, 1),
            datetime(2026, 1, 2),
            datetime(2026, 1, 3),
            datetime(2026, 9, 9),
        ]
    )
    bad = _run(["no_future_columns"], leaky)
    assert not bad["no_future_columns"].passed
    assert "snapshot_date" in bad["no_future_columns"].message


def test_label_leakage_scan() -> None:
    ok = _run(["label_leakage_scan"], _handle())
    assert ok["label_leakage_scan"].passed
    # a feature perfectly correlated with the label must be flagged
    leaky = _handle(feature=[0.0, 1.0, 0.0, 1.0])
    bad = _run(["label_leakage_scan"], leaky)
    assert not bad["label_leakage_scan"].passed
    assert "feature" in bad["label_leakage_scan"].message


def test_class_balance_report_never_fails() -> None:
    result = _run(["class_balance_report"], _handle())
    assert result["class_balance_report"].passed
    assert "label balance" in result["class_balance_report"].message
