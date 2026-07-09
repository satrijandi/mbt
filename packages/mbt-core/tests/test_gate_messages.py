"""Gate-failure node message: names the specific failing gate(s) for the
results table, JSON run_results, and the GitOps PR comment."""

from mbt.artifacts.run_results import GateResult
from mbt.execute.runners import _gate_failure_summary


def test_threshold_gate_failure_names_metric_and_criterion() -> None:
    gate = GateResult(metric="pr_auc", kind="threshold", passed=False, expected=0.99, actual=0.3161)
    assert _gate_failure_summary([gate]) == "gate breach: pr_auc=0.3161 failed threshold 0.99"


def test_only_failing_gates_are_named_with_their_slice() -> None:
    passing = GateResult(metric="roc_auc", kind="threshold", passed=True, expected=0.5, actual=0.7)
    slice_fail = GateResult(
        metric="pr_auc",
        kind="threshold",
        slice="plan=basic",
        passed=False,
        expected=0.4,
        actual=0.2,
    )
    assert (
        _gate_failure_summary([passing, slice_fail])
        == "gate breach: pr_auc [plan=basic]=0.2000 failed threshold 0.4"
    )


def test_champion_gate_failure_reports_the_bootstrap_bound() -> None:
    gate = GateResult(
        metric="pr_auc", kind="champion", passed=False, delta_lower=-0.01, min_delta=0.0
    )
    assert "challenger delta lower bound -0.0100 < required 0.0" in _gate_failure_summary([gate])


def test_no_failing_gate_falls_back_to_generic_message() -> None:
    passing = GateResult(metric="roc_auc", kind="threshold", passed=True, expected=0.5, actual=0.7)
    assert _gate_failure_summary([passing]) == "one or more gates failed"
