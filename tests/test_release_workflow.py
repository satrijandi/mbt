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
