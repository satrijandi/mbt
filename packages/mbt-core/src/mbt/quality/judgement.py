"""Judging one job result: gates, stability, verdict, pass, summary (C-1).

The five-call sequence lived at ``runners.py:795-820`` and ``oot_check.py:210-223``,
written out twice. Each leaf had a tight unit test; the COMPOSITION was tested
only through ``run_command`` - and the composition is where the coupling was.

Two defects came out of that, and both are fixed here:

- ``after_test_verdict`` identified after-test gates by ``gate.period is not
  None``, an internal detail of how ``_out_of_time_result`` is built.
  ``GateSpec.source == "out_of_time"`` is the real discriminator and is
  available on the spec, so a future gate kind populating ``period`` would
  have silently changed every recorded verdict while both the leaf test and
  ``_out_of_time_result`` kept passing.
- ``evaluate_stability`` returned ``[]`` for two different facts, and the
  caller consumed ``not stability`` as if it meant one. That is exactly what
  separates ``not_gated`` from ``true``; it is ``StabilityOutcome`` now.
"""

from dataclasses import dataclass
from typing import Any

from mbt.artifacts.run_results import GateResult, MonitorResult
from mbt.quality.gates import all_gates_passed
from mbt.quality.monitors import (
    StabilityOutcome,
    all_monitors_passed,
    evaluate_stability,
)
from mbt_adapter_base import (
    ModelSpec,
)
from mbt_adapter_base.champion import AfterTestVerdict

#: ``GateSpec.source`` value that marks an after-test gate (ADR-30).
AFTER_TEST_SOURCE = "out_of_time"


def after_test_verdict(
    spec: ModelSpec, gates: list[GateResult], stability: StabilityOutcome
) -> str | None:
    """``true`` / ``false`` / ``not_gated`` for the after-test gates (ADR-30).

    None when the model declares none. ``not_gated`` means they were declared
    but nothing was mature enough to judge - a promotion that requires the
    check refuses it, because a gate that judged nothing proved nothing.
    """
    declared = spec.evaluation.stability is not None or any(
        gate.source == "out_of_time" for gate in spec.evaluation.gates
    )
    if not declared:
        return None
    # The gate's OWN source, not ``period is not None`` (C-1): period is an
    # internal detail of how an after-test result happens to be built, so any
    # future gate kind that populated it would silently have changed every
    # recorded verdict.
    after = [gate for gate in gates if gate.source == AFTER_TEST_SOURCE]
    if not any(gate.applicable for gate in after) and not stability.judged:
        return "not_gated"
    passed = all(gate.passed for gate in after) and all_monitors_passed(stability.results)
    return "true" if passed else "false"


def after_test_tags(verdict: str, anchor: str, source: str) -> dict[str, str]:
    """The registry and tracking tags that record an after-test verdict (ADR-30).

    The spelling lives in ``ChampionRecord``; this stays as the one-call shape
    the tracking path wants, where there is no full record to pack.
    """
    return AfterTestVerdict(passed=verdict, anchor=anchor, source=source).pack()


def gate_failure_summary(
    gates: list[GateResult], monitors: list[MonitorResult] | None = None
) -> str:
    """A specific, reviewer-facing summary of which gate(s) failed - feeds the
    node message shown in the results table, the JSON run_results, and the
    GitOps PR comment, consistent with the monitor path's ``gate breach: ...``.
    """
    parts: list[str] = []
    for gate in gates:
        if gate.passed:
            continue
        where = f" [{gate.slice}]" if gate.slice else ""
        if gate.kind == "champion" and gate.delta_lower is not None:
            parts.append(
                f"{gate.metric}{where}: challenger delta lower bound "
                f"{gate.delta_lower:.4f} < required {gate.min_delta}"
            )
        elif gate.actual is not None:
            cell = f" [{gate.cell}]" if gate.cell else ""
            parts.append(
                f"{gate.metric}{where}{cell}={gate.actual:.4f} failed threshold {gate.expected}"
            )
        else:
            parts.append(f"{gate.metric}{where}")
    for monitor in monitors or []:
        if not monitor.passed:
            parts.append(f"stability {monitor.message or monitor.subject}")
    return "gate breach: " + "; ".join(parts) if parts else "one or more gates failed"


@dataclass(frozen=True)
class Judgement:
    """Whether a job result passes, and everything recorded about why."""

    gates: list[GateResult]
    stability: StabilityOutcome
    #: ``true`` / ``false`` / ``not_gated``, or None when none are declared.
    verdict: str | None
    passed: bool
    #: Reviewer-facing summary; None when it passed.
    failure_summary: str | None


def judge(
    spec: ModelSpec,
    gates: list[GateResult],
    report: Any | None,
    *,
    resource: str,
) -> Judgement:
    """Judge one job result against its spec.

    Gates come in already evaluated, because HOW they are evaluated genuinely
    differs between the two callers - the pre-deploy check re-runs only the
    after-test ones. Everything downstream of that is identical, and now
    exists once.
    """
    stability = evaluate_stability(spec.evaluation.stability, report, resource=resource)
    verdict = after_test_verdict(spec, gates, stability)
    passed = all_gates_passed(gates) and all_monitors_passed(stability.results)
    return Judgement(
        gates=gates,
        stability=stability,
        verdict=verdict,
        passed=passed,
        failure_summary=None if passed else gate_failure_summary(gates, stability.results),
    )


__all__ = [
    "AFTER_TEST_SOURCE",
    "Judgement",
    "after_test_verdict",
    "gate_failure_summary",
    "judge",
]
