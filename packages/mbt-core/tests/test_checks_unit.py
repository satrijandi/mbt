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


def test_check_names_match_dispatch_table() -> None:
    """The name registry and the impl dispatch table are one source of truth:
    every declared name has an implementation and vice versa, so a check added
    to only one side fails loudly here instead of becoming a confusing
    'unknown dataset check' at parse time."""
    from mbt.quality.check_names import BUILTIN_CHECK_NAMES, SCORING_CHECK_NAMES
    from mbt.quality.checks import _CHECKS

    assert set(_CHECKS) == set(BUILTIN_CHECK_NAMES)
    assert SCORING_CHECK_NAMES <= BUILTIN_CHECK_NAMES


def test_parser_validates_against_shared_check_names() -> None:
    """The parser derives its valid-check sets from the authoritative module
    rather than re-listing them (the drift the shared source removes)."""
    from mbt.parsing import project_parser
    from mbt.quality.check_names import BUILTIN_CHECK_NAMES, SCORING_CHECK_NAMES

    assert project_parser._BUILTIN_CHECKS is BUILTIN_CHECK_NAMES
    assert project_parser._SCORING_CHECKS is SCORING_CHECK_NAMES
