"""Guard for the dependency audit wrapper (scripts/audit_dependencies.py).

The wrapper exists so an accepted vulnerability cannot quietly become a
permanent blind spot: it fails both on a finding nobody accepted AND on an
acceptance that no longer fires. Both directions are asserted here, on fixture
reports rather than a live pip-audit run, so the suite stays offline.
"""

import importlib.util
from pathlib import Path

import pytest

REPO_ROOT = Path(__file__).resolve().parent.parent
_spec = importlib.util.spec_from_file_location(
    "audit_dependencies", REPO_ROOT / "scripts" / "audit_dependencies.py"
)
assert _spec is not None and _spec.loader is not None
audit = importlib.util.module_from_spec(_spec)
_spec.loader.exec_module(audit)


def _report(*vulns_by_package: tuple[str, list[str]]) -> dict:
    return {
        "dependencies": [
            {"name": name, "version": "1.0", "vulns": [{"id": v} for v in ids]}
            for name, ids in vulns_by_package
        ]
    }


def test_findings_are_collected_across_packages() -> None:
    found = audit.findings_from(_report(("a", ["PYSEC-1", "PYSEC-2"]), ("b", ["PYSEC-3"])))
    assert found == {"PYSEC-1", "PYSEC-2", "PYSEC-3"}


def test_packages_without_vulns_contribute_nothing() -> None:
    assert audit.findings_from(_report(("a", []))) == set()
    assert audit.findings_from({}) == set()


def _vuln_report(vuln: dict) -> dict:
    return {"dependencies": [{"name": "a", "version": "1.0", "vulns": [vuln]}]}


def test_an_advisory_is_matched_by_alias_not_just_primary_id() -> None:
    """pip-audit's primary id is not stable across resolutions.

    The same MLflow SSRF advisory reports as CVE-2026-71211 against the locked
    environment and as GHSA-h7x2-h6g9-p789 against the floors one, each
    demoting the other spelling to an alias. Matching the raw id made one
    ACCEPTED entry cover the `test` job and miss the `floors` job on the
    identical advisory, so no single entry could turn both green.
    """
    accepted = sorted(set(audit.ACCEPTED) - audit.FLOOR_ONLY)[0]
    found = audit.findings_from(_vuln_report({"id": "OTHER-SPELLING-1", "aliases": [accepted]}))
    assert found == {accepted}
    unaccepted, _ = audit.classify(found)
    assert unaccepted == []


def test_an_unaccepted_finding_keeps_pip_audits_own_id() -> None:
    """The operator searches for the id pip-audit printed, so do not rewrite it."""
    vuln = {"id": "PYSEC-9999-3", "aliases": ["CVE-9999-3"]}
    assert audit.findings_from(_vuln_report(vuln)) == {"PYSEC-9999-3"}


def test_a_null_alias_list_is_tolerated() -> None:
    """pip-audit emits `aliases: null` for some sources; that is not an alias."""
    assert audit.findings_from(_vuln_report({"id": "PYSEC-9999-4", "aliases": None})) == {
        "PYSEC-9999-4"
    }


def test_an_unaccepted_finding_fails() -> None:
    unaccepted, stale = audit.classify({"PYSEC-9999-1"})
    assert unaccepted == ["PYSEC-9999-1"]
    assert audit.report(unaccepted, stale, check_stale=True) == 1


def test_an_accepted_finding_passes_while_it_still_fires() -> None:
    unaccepted, stale = audit.classify(set(audit.ACCEPTED))
    assert (unaccepted, stale) == ([], [])
    assert audit.report(unaccepted, stale, check_stale=True) == 0


def test_an_acceptance_that_stopped_firing_fails_so_it_gets_deleted() -> None:
    """The whole point: pip-audit alone would go on passing here forever."""
    checkable = sorted(set(audit.ACCEPTED) - audit.FLOOR_ONLY)
    still_applies = checkable[:1]
    unaccepted, stale = audit.classify(set(still_applies))
    assert stale == sorted(set(checkable) - set(still_applies))
    assert audit.report(unaccepted, stale, check_stale=True) == 1


def test_stale_acceptances_are_only_a_note_when_the_check_is_off() -> None:
    """The floors job resolves different versions than the lock, so an advisory
    may legitimately not apply there; that must not fail the build."""
    unaccepted, stale = audit.classify(set())
    assert stale == sorted(set(audit.ACCEPTED) - audit.FLOOR_ONLY)
    assert audit.report(unaccepted, stale, check_stale=False) == 0


def test_unaccepted_findings_fail_even_with_the_stale_check_off() -> None:
    unaccepted, stale = audit.classify(set(audit.ACCEPTED) | {"PYSEC-9999-2"})
    assert audit.report(unaccepted, stale, check_stale=False) == 1


def test_every_acceptance_documents_why_and_what_ends_it() -> None:
    """An entry with no rationale is the blind spot this script exists to stop."""
    for advisory, reason in audit.ACCEPTED.items():
        assert len(reason) > 80, f"{advisory} needs a real rationale, got: {reason!r}"
        assert "Ends when" in reason, f"{advisory} must say what ends the acceptance"


def test_a_non_json_report_is_a_hard_error_not_a_silent_pass(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """A broken pip-audit must never look like a clean audit."""

    class _Proc:
        returncode = 2
        stdout = "Traceback: pip-audit exploded"
        stderr = "boom"

    monkeypatch.setattr(audit.subprocess, "run", lambda *a, **k: _Proc())
    with pytest.raises(SystemExit, match="did not produce a JSON report"):
        audit.run_pip_audit()


def test_floor_only_entries_are_real_acceptances() -> None:
    """FLOOR_ONLY narrows an acceptance; it cannot create one. An id listed
    there but absent from ACCEPTED would be exempt from the staleness check
    while never having been justified at all."""
    assert set(audit.ACCEPTED) >= audit.FLOOR_ONLY


def test_floor_only_entries_are_exempt_from_the_staleness_check() -> None:
    """They are accepted BECAUSE the shipped resolution is clear of them, so
    demanding that they fire against the lock would be permanently red."""
    _, stale = audit.classify(set())
    assert not (audit.FLOOR_ONLY & set(stale))


def test_a_floor_only_entry_firing_against_the_lock_fails() -> None:
    """The exemption must not become a way to silence a live finding: if the
    shipped resolution DOES have one, the floor-only story is false."""
    advisory = sorted(audit.FLOOR_ONLY)[0]
    misfiled = audit.misfiled_floor_only({advisory})
    assert misfiled == [advisory]
    assert audit.report([], [], check_stale=True, misfiled=misfiled) == 1


def test_a_clean_locked_run_reports_no_misfiled_floor_only() -> None:
    unaccepted, stale = audit.classify(set(audit.ACCEPTED))
    assert audit.misfiled_floor_only(set(audit.ACCEPTED) - audit.FLOOR_ONLY) == []
    assert audit.report(unaccepted, stale, check_stale=True, misfiled=[]) == 0


def test_a_floor_only_entry_that_stopped_firing_at_the_floors_fails() -> None:
    """The half nothing used to check.

    A FLOOR_ONLY acceptance claims the declared lower bound still carries the
    advisory. The floors job runs --no-check-stale, so the ordinary staleness
    check skips these by design; --require-floor-only is what makes the claim
    falsifiable in the one environment that can judge it.
    """
    advisory = sorted(audit.FLOOR_ONLY)[0]
    still_firing = set(audit.FLOOR_ONLY)
    assert audit.unearned_floor_only(still_firing) == []

    risen_past = still_firing - {advisory}
    assert audit.unearned_floor_only(risen_past) == [advisory]
    assert audit.report([], [], check_stale=False, unearned=[advisory]) == 1


def test_require_floor_only_is_opt_in(monkeypatch: pytest.MonkeyPatch) -> None:
    """Without the flag, the floors job's own resolution must stay green.

    Only the floors environment can judge these; the locked audit and the
    upstream tier resolve something else, so demanding the entries fire there
    would be permanently red for the wrong reason.
    """
    report = {
        "dependencies": [
            {"vulns": [{"id": advisory, "aliases": []}]} for advisory in audit.ACCEPTED
        ]
    }
    monkeypatch.setattr(audit, "run_pip_audit", lambda: report)
    # The floors environment: every acceptance fires, floor-only ones included.
    assert audit.main(["--no-check-stale"]) == 0
    assert audit.main(["--no-check-stale", "--require-floor-only"]) == 0
    # The same report judged as if it were the LOCK is a different verdict:
    # a floor-only entry firing there means the floor-only story is false.
    assert audit.main([]) == 1

    # Now the floors have risen past one of them: only the opt-in run notices.
    advisory = sorted(audit.FLOOR_ONLY)[0]
    thinner = {
        "dependencies": [
            {"vulns": [{"id": a, "aliases": []}]} for a in audit.ACCEPTED if a != advisory
        ]
    }
    monkeypatch.setattr(audit, "run_pip_audit", lambda: thinner)
    assert audit.main(["--no-check-stale"]) == 0
    assert audit.main(["--no-check-stale", "--require-floor-only"]) == 1
