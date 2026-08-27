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


def test_planted_leak_is_quarantined(dataset, spec):
    """account_status is the demo's deliberate leak: generated post-outcome,
    it encodes the label exactly. This test pins BOTH facts - the leak stays
    leaky (so the teaching asset cannot silently rot) and the dataset's
    reviewed scan exclusion stays declared (so removing it is a visible,
    build-blocking spec change, not an accident)."""
    status = dataset.column("account_status").to_pylist()
    labels = dataset.column(spec.label.column).to_pylist()
    encodes = all(
        (value == "cancelled") == bool(label) for value, label in zip(status, labels, strict=True)
    )
    excluded = any(
        isinstance(check, dict)
        and "account_status" in (check.get("label_leakage_scan") or {}).get("exclude", [])
        for check in spec.checks
    )
    return TestResult(
        name="test_planted_leak_is_quarantined",
        passed=encodes and excluded,
        message=f"leak encodes label: {encodes}; scan exclusion declared: {excluded}",
    )
