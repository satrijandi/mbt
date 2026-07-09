"""Unit tests for Python data-test discovery and execution (mbt/quality/python_tests.py)."""

from pathlib import Path

import pyarrow as pa
from core_helpers import write

from mbt.contracts import DatasetSpec
from mbt.parsing.errors import ParseReport
from mbt.quality.python_tests import discover_python_tests, run_python_tests


def _dataset_spec() -> DatasetSpec:
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
        }
    )


def test_discovery_skips_underscored_broken_and_empty_files(tmp_path: Path) -> None:
    write(tmp_path / "tests" / "_helper.py", "def test_hidden(dataset, spec): return True\n")
    write(tmp_path / "tests" / "broken.py", "def test_x(:\n")
    write(tmp_path / "tests" / "no_tests_here.py", "def helper():\n    return 1\n")
    write(tmp_path / "tests" / "real_checks.py", "def test_rows(dataset, spec): return True\n")
    report = ParseReport()
    found = discover_python_tests(tmp_path, ["tests", "missing_dir"], report)
    assert [f.rel for f in found] == ["tests/real_checks.py"]
    assert found[0].selector is None  # no '# mbt: select=' header
    assert found[0].test_names == ("test_rows",)
    severities = [issue.severity for issue in report.issues]
    assert severities.count("error") == 1  # broken.py
    assert severities.count("warning") == 1  # no_tests_here.py
    assert "invalid Python" in report.issues[0].message


def test_run_python_tests_outcome_shapes(tmp_path: Path) -> None:
    write(
        tmp_path / "tests" / "outcome_shapes.py",
        """
        from mbt.contracts import TestResult

        def test_true(dataset, spec):
            return True

        def test_false(dataset, spec):
            return False

        def test_none(dataset, spec):
            assert dataset.num_rows >= 0

        def test_raises(dataset, spec):
            raise ValueError("boom")

        def test_wrong_type(dataset, spec):
            return 42

        def test_result(dataset, spec):
            return TestResult(name="test_result", passed=True, message="ok")
        """,
    )
    report = ParseReport()
    (test_file,) = discover_python_tests(tmp_path, ["tests"], report)
    table = pa.table({"y": [0, 1]})
    results = {r.name: r for r in run_python_tests(test_file, table, _dataset_spec())}
    assert results["test_true"].passed
    assert not results["test_false"].passed
    assert results["test_none"].passed
    assert not results["test_raises"].passed
    assert "boom" in (results["test_raises"].message or "")
    assert not results["test_wrong_type"].passed
    assert "returned int" in (results["test_wrong_type"].message or "")
    assert results["test_result"].passed and results["test_result"].message == "ok"


def test_run_python_tests_only_filter(tmp_path: Path) -> None:
    write(
        tmp_path / "tests" / "only_filter.py",
        """
        def test_kept(dataset, spec):
            return True

        def test_skipped(dataset, spec):
            return True
        """,
    )
    report = ParseReport()
    (test_file,) = discover_python_tests(tmp_path, ["tests"], report)
    table = pa.table({"y": [0, 1]})
    results = run_python_tests(test_file, table, _dataset_spec(), only={"test_kept"})
    assert [r.name for r in results] == ["test_kept"]
