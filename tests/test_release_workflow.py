"""The release workflow must never publish immutable wheels for a red commit.

R2-1: `release.yml` had no test gate and no `skip-existing`, so pushing a
`vX.Y.Z` tag on a commit that would fail CI published broken, unfixable wheels
for all 10 packages, and a re-run after a partial upload hard-failed on the
packages that had already landed. This turns the fix into a permanent
invariant, in the spirit of the version-sync and cli-reference drift guards:
the release stays gated on the repo's real CI, and the publish stays idempotent.
"""

from pathlib import Path

import yaml

WORKFLOWS = Path(__file__).resolve().parent.parent / ".github" / "workflows"


def _load(name: str) -> dict:
    return yaml.safe_load((WORKFLOWS / name).read_text())


def _triggers(spec: dict) -> dict:
    # PyYAML parses the unquoted `on:` key as the boolean True (YAML 1.1).
    return spec.get("on", spec.get(True, {}))


def test_release_publish_is_gated_on_ci() -> None:
    release = _load("release.yml")
    jobs = release["jobs"]
    # the publishing job must depend on a gate...
    assert "ci" in jobs["release"].get("needs", []), jobs["release"].get("needs")
    # ...and that gate must be the repo's real CI called reusably, never a
    # hand-maintained subset that can drift from what actually guards main.
    assert jobs["ci"]["uses"].endswith("/ci.yml"), jobs["ci"].get("uses")


def test_ci_is_reusable_so_the_release_gate_stays_honest() -> None:
    assert "workflow_call" in _triggers(_load("ci.yml"))


def test_publish_is_idempotent_on_reruns() -> None:
    steps = _load("release.yml")["jobs"]["release"]["steps"]
    publish = next(s for s in steps if "pypi-publish" in str(s.get("uses", "")))
    assert (publish.get("with") or {}).get("skip-existing") is True, publish


def _declared_permissions(spec: dict) -> dict[str, str]:
    """Every permission the workflow declares anywhere (top level or any job),
    at the highest level requested for each key."""
    rank = {"read": 0, "write": 1}
    merged: dict[str, str] = {}

    def absorb(block: dict | None) -> None:
        for key, level in (block or {}).items():
            if key not in merged or rank.get(level, 0) > rank.get(merged[key], 0):
                merged[key] = level

    absorb(spec.get("permissions"))
    for job in spec.get("jobs", {}).values():
        absorb(job.get("permissions"))
    return merged


def test_release_ci_call_grants_the_callee_permission_envelope() -> None:
    """GitHub validates a reusable call's permission envelope at STARTUP: the
    caller job must grant every permission the called workflow declares
    anywhere - even on jobs whose ``if:`` would skip them for this event. The
    first ``v0.1.0`` tag failed with ``startup_failure`` on exactly this
    (ci.yml's docs-publish declares pages/id-token write; the ``ci`` caller
    job granted only ``contents: read``), which no YAML linter can catch."""
    rank = {"read": 0, "write": 1}
    callee = _declared_permissions(_load("ci.yml"))
    caller = _load("release.yml")["jobs"]["ci"].get("permissions") or {}
    for key, level in callee.items():
        granted = caller.get(key)
        assert granted is not None and rank[granted] >= rank[level], (
            f"release.yml's ci job grants {caller}, but ci.yml declares "
            f"{key}: {level} - the reusable call would fail at startup"
        )
