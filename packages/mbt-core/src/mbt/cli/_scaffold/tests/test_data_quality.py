# mbt: select=churn_training_set
"""Python data tests run against the materialized dataset (FR-RES-05)."""

from mbt.contracts import TestResult


def test_label_is_binary(dataset, spec):  # type: ignore[no-untyped-def]
    values = set(dataset.column(spec.label.column).to_pylist())
    return TestResult(
        name="test_label_is_binary",
        passed=values <= {0, 1},
        message=f"label classes: {sorted(values)}",
    )


def test_minimum_rows(dataset, spec):  # type: ignore[no-untyped-def]
    return TestResult(
        name="test_minimum_rows",
        passed=dataset.num_rows >= 100,
        message=f"{dataset.num_rows} rows",
    )
