"""Known-vulnerability audit whose accepted-findings list cannot rot.

``pip-audit --ignore-vuln`` keeps passing forever once an advisory stops
applying, so a suppression outlives the reason it was added and nobody
notices - the entry becomes a permanent blind spot, and the next person reads
it as "we looked at this recently".

This wrapper treats the accepted list as an assertion in BOTH directions:

* a finding that is not accepted fails the build, as usual;
* an accepted finding that NO LONGER FIRES also fails, telling the operator to
  delete the entry.

So the only way to keep the list is to keep earning it. Every entry states why
it is accepted and what would end that acceptance.
"""

from __future__ import annotations

import argparse
import json
import subprocess
import sys

#: Shared reason for the three h2o advisories that only the declared floor hits.
_SPARKLING_FLOOR = (
    "CVE-2024-45758 / CVE-2025-6544 / CVE-2026-3960: deserialization and RCE "
    "through h2o's unauthenticated ImportSQLTable REST endpoints, reachable by "
    "setting a JDBC connection URL. Fixed in 3.46.0.10, and the resolution mbt "
    "SHIPS is already past it (uv.lock pins 3.46.0.11, which reports none of "
    "the three). They survive only at the declared FLOOR: mbt-h2o[sparkling] "
    "needs the h2o python client to match the backend embedded in "
    "h2o-pysparkling-3.5, whose newest release (3.46.0.6.post1) embeds 3.46.0.6 "
    "- so a floor above that makes the sparkling extra uninstallable outright, "
    "which is how this was found (a showcase image build failing on "
    "ResolutionImpossible). Not in mbt's execute path either way: mbt-h2o never "
    "references JDBC or ImportSQLTable, and starts the cluster with "
    "bind_to_localhost=True as a job-scoped process torn down at exit, so the "
    "endpoint is never a network service. Ends when h2o-pysparkling-3.5 ships "
    "a build on a newer H2O and the mbt-h2o floor can rise with it."
)

#: Advisory ID -> why shipping with it is acceptable, and what ends that.
#:
#: An entry belongs here only when BOTH hold: the fix is unreachable for us,
#: and the vulnerable code is not in mbt's execute path. Reachability is the
#: load-bearing half - "not exploitable by us" is a perishable claim, so each
#: entry names the code path it depends on, not just a severity opinion.
ACCEPTED: dict[str, str] = {
    "PYSEC-2026-1443": (
        "CVE-2024-7768: DoS of h2o's own REST endpoint (/3/ImportFiles). "
        "mbt-h2o runs that server only as a local, ephemeral, training-time "
        "process, never as a network service. No fixed h2o release exists "
        "(<= 3.46.1 all affected). Ends when a fixed h2o ships."
    ),
    "PYSEC-2026-3552": (
        "CVE-2026-69247: Bleichenbacher oracle in cryptography's PKCS#7 "
        "decryption (pkcs7_decrypt_der/pem/smime). mbt never imports "
        "cryptography and decrypts no PKCS#7; it reaches us only as a "
        "transitive dep of mlflow, pyopenssl and snowflake-connector-python. "
        "The fix is 50.0.0 and mlflow 3.15.2 (newest) still pins "
        "cryptography<50, so it is unreachable. pyopenssl 26.4.0 already "
        "allows <51, leaving mlflow as the sole blocker. Already fixed "
        "upstream: mlflow#24871 -> PR mlflow#24872 raises the cap to "
        "cryptography<51, merged to master and still unreleased as of the "
        "3.15.2 patch. Ends when the next mlflow minor ships and Renovate "
        "bumps us onto it; this script's staleness check will then fail and "
        "force the entry out."
    ),
    "CVE-2026-71211": (
        "GHSA-h7x2-h6g9-p789: SSRF in MLflow's AI Gateway - a request to the "
        "gateway can reach an arbitrary host because api_base is not validated. "
        "No fixed release exists: the advisory covers mlflow >= 3.13.0 <= 3.15.2 "
        "and 3.15.2 is the newest, with first_patched_version null. Not in mbt's "
        "execute path: the AI Gateway is a separate server mbt neither runs nor "
        "configures, and mbt-mlflow imports exactly two things from mlflow - "
        "mlflow.tracking.MlflowClient and mlflow.exceptions.MlflowException - "
        "with no reference anywhere to mlflow.gateway, mlflow.deployments or "
        "api_base. Ends when an mlflow release ships the fix and Renovate bumps "
        "us onto it; this script's staleness check will then force the entry out."
    ),
    "PYSEC-2026-352": _SPARKLING_FLOOR,
    "PYSEC-2026-349": _SPARKLING_FLOOR,
    "PYSEC-2026-2180": _SPARKLING_FLOOR,
}

#: Accepted only because a DECLARED FLOOR must stay on a version that has them.
#:
#: These do not apply to what we ship - the locked resolution is h2o 3.46.0.11,
#: which is clear of all three - so the staleness check must not demand that
#: they fire in the locked environment. `misfiled_floor_only` asserts the other
#: direction: if one of these ever DOES fire against the lock, the "floor only"
#: story is wrong and it must become a real acceptance.
#:
#: The one rot risk this cannot close: if the floor later rises past them,
#: nothing here forces the entry out, because the environment that would prove
#: it (the floors job) deliberately runs with --no-check-stale. Raising an h2o
#: floor is the moment to re-read this set.
FLOOR_ONLY: frozenset[str] = frozenset({"PYSEC-2026-352", "PYSEC-2026-349", "PYSEC-2026-2180"})


def findings_from(report: dict) -> set[str]:
    """Advisory IDs reported against non-editable dependencies.

    Each finding is reported under whichever identifier ACCEPTED already knows
    it by, because pip-audit's primary ``id`` is NOT stable across resolutions:
    the same MLflow SSRF advisory reports as CVE-2026-71211 against the locked
    environment and as GHSA-h7x2-h6g9-p789 against the floors one, each
    demoting the other spelling to an alias. Matching the raw ``id`` therefore
    let an acceptance hold in the `test` job and fail in the `floors` job on the
    identical advisory - so one entry could not turn both green.

    Anything not accepted keeps pip-audit's own id, which is the string the
    operator will search for.
    """
    found: set[str] = set()
    for dependency in report.get("dependencies", []):
        for vuln in dependency.get("vulns", []):
            names = {vuln["id"], *(vuln.get("aliases") or [])}
            known = sorted(names & set(ACCEPTED))
            found.add(known[0] if known else vuln["id"])
    return found


def classify(found: set[str]) -> tuple[list[str], list[str]]:
    """Split into (findings we have not accepted, acceptances that are stale).

    FLOOR_ONLY entries are exempt from the staleness half: they are accepted
    precisely because they do NOT apply to the resolution we ship, so demanding
    that they fire here would make the locked audit permanently red.
    """
    checkable = set(ACCEPTED) - FLOOR_ONLY
    return sorted(found - set(ACCEPTED)), sorted(checkable - found)


def misfiled_floor_only(found: set[str]) -> list[str]:
    """FLOOR_ONLY entries that fire in THIS resolution.

    Only meaningful where the resolution is the locked one. Each of these
    entries claims "the shipped resolution is clear of this, only the declared
    floor is not"; if one fires against the locked environment that claim is
    false and it needs a real acceptance, not a floor-shaped one. Without this
    the exemption above would be a way to silence a live finding.
    """
    return sorted(FLOOR_ONLY & found)


def run_pip_audit() -> dict:
    """Run pip-audit and return its JSON report.

    A nonzero exit only means "vulnerabilities were found", which is the normal
    case here, so the report is parsed regardless; only unparseable output is
    treated as the tool itself failing.
    """
    proc = subprocess.run(
        [sys.executable, "-m", "pip_audit", "--skip-editable", "--format", "json"],
        capture_output=True,
        text=True,
        check=False,
    )
    try:
        return json.loads(proc.stdout)
    except json.JSONDecodeError:
        raise SystemExit(
            f"pip-audit did not produce a JSON report (exit {proc.returncode}).\n"
            f"--- stdout ---\n{proc.stdout}\n--- stderr ---\n{proc.stderr}"
        ) from None


def report(
    unaccepted: list[str], stale: list[str], check_stale: bool, misfiled: list[str] | None = None
) -> int:
    misfiled = misfiled or []
    for advisory in unaccepted:
        print(f"FAIL new vulnerability, not accepted: {advisory}")
    if unaccepted:
        print(
            "\nFix it by raising the dependency, or - only if the fix is "
            "unreachable AND the code is not in mbt's execute path - add it to "
            f"ACCEPTED in {__file__} with both of those stated."
        )
    if stale and check_stale:
        for advisory in stale:
            print(f"FAIL accepted vulnerability no longer applies: {advisory}")
        print(
            f"\nDelete those entries from ACCEPTED in {__file__}: whatever made "
            "them acceptable has changed, so the note is now misleading."
        )
    elif stale:
        print(f"note: not reported by this environment: {', '.join(stale)}")

    for advisory in misfiled:
        print(f"FAIL accepted as floor-only but firing against the lock: {advisory}")
    if misfiled:
        print(
            f"\nThose entries are in FLOOR_ONLY in {__file__}, which claims the "
            "shipped resolution is clear of them. It is not. Either raise the "
            "dependency, or move the entry into a full acceptance that states "
            "why the fix is unreachable for the version we actually ship."
        )

    if unaccepted or misfiled or (stale and check_stale):
        return 1
    accepted = ", ".join(sorted(ACCEPTED)) or "none"
    print(f"pip-audit clean; still-accepted advisories: {accepted}")
    return 0


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--no-check-stale",
        action="store_true",
        help=(
            "Do not fail on accepted advisories that this environment does not "
            "report. Use where the resolution is not the locked one (the floors "
            "job resolves different versions, so an advisory may legitimately "
            "not apply there)."
        ),
    )
    args = parser.parse_args(argv)
    found = findings_from(run_pip_audit())
    unaccepted, stale = classify(found)
    # Only the locked resolution can judge a floor-only claim; the floors job
    # and the upstream tier deliberately resolve something else.
    misfiled = [] if args.no_check_stale else misfiled_floor_only(found)
    return report(unaccepted, stale, check_stale=not args.no_check_stale, misfiled=misfiled)


if __name__ == "__main__":
    raise SystemExit(main())
