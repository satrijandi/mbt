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
        "The fix is 50.0.0 and mlflow 3.15.1 (newest) still pins "
        "cryptography<50, so it is unreachable. Already fixed upstream: "
        "mlflow#24871 -> PR mlflow#24872 raises the cap to cryptography<51, "
        "merged to master, unreleased as of 3.15.1. Ends when the next mlflow "
        "minor ships and Renovate bumps us onto it; this script's staleness "
        "check will then fail and force the entry out."
    ),
}


def findings_from(report: dict) -> set[str]:
    """Advisory IDs reported against non-editable dependencies."""
    return {
        vuln["id"]
        for dependency in report.get("dependencies", [])
        for vuln in dependency.get("vulns", [])
    }


def classify(found: set[str]) -> tuple[list[str], list[str]]:
    """Split into (findings we have not accepted, acceptances that are stale)."""
    return sorted(found - set(ACCEPTED)), sorted(set(ACCEPTED) - found)


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


def report(unaccepted: list[str], stale: list[str], check_stale: bool) -> int:
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

    if unaccepted or (stale and check_stale):
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
    unaccepted, stale = classify(findings_from(run_pip_audit()))
    return report(unaccepted, stale, check_stale=not args.no_check_stale)


if __name__ == "__main__":
    raise SystemExit(main())
