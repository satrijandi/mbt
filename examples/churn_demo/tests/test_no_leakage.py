# mbt: select=churn_training_set
"""Data tests for the churn training set (FR-RES-05)."""

from mbt.contracts import TestResult


def test_label_is_binary(dataset, spec):
    values = set(dataset.column(spec.label.column).to_pylist())
    return TestResult(
        name="test_label_is_binary",
        passed=values <= {0, 1},
        message=f"label classes: {sorted(values)}",
    )


def test_only_active_subscribers(dataset, spec):
    actives = dataset.column("is_active").to_pylist()
    return TestResult(
        name="test_only_active_subscribers",
        passed=all(actives),
        message=f"{sum(1 for a in actives if not a)} inactive rows leaked through filters",
    )
