"""The release workflow must never publish immutable wheels for a red commit.

R2-1: `release.yml` had no test gate and no `skip-existing`, so pushing a
`vX.Y.Z` tag on a commit that would fail CI published broken, unfixable wheels
for all 11 packages, and a re-run after a partial upload hard-failed on the
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


def test_github_release_precedes_the_pypi_publish_and_publish_is_opt_in() -> None:
    """The v0.1.0 tag proved two things at once: an unconfigured PyPI publisher
    fails `invalid-publisher`, and ordered first it blocked the GitHub release
    behind it - the artifact users actually consume via the tag pins. So the
    GitHub release must come before the publish, and the publish must be
    gated on the explicit opt-in variable so the workflow is genuinely inert
    (green, release created) until the PyPI side exists."""
    steps = _load("release.yml")["jobs"]["release"]["steps"]
    index = {
        kind: next(i for i, s in enumerate(steps) if kind in str(s.get("uses", "")))
        for kind in ("gh-release", "pypi-publish")
    }
    assert index["gh-release"] < index["pypi-publish"], index
    publish = steps[index["pypi-publish"]]
    assert "PYPI_TRUSTED_PUBLISHING" in str(publish.get("if", "")), publish


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


# -- the one-time setup a human must do (CONTRIBUTING.md) ----------------------

CONTRIBUTING = WORKFLOWS.parent.parent / "CONTRIBUTING.md"


def _publish_step() -> dict:
    release = yaml.safe_load((WORKFLOWS / "release.yml").read_text())
    (job,) = (j for j in release["jobs"].values() if "environment" in j)
    (step,) = (s for s in job["steps"] if "pypi-publish" in str(s.get("uses", "")))
    assert isinstance(job, dict) and isinstance(step, dict)
    return {"environment": job["environment"], "gate": step.get("if", "")}


def test_the_documented_trusted_publisher_matches_the_workflow() -> None:
    """Trusted Publishing is configured by typing four values into PyPI, and a
    wrong one fails as `invalid-publisher` at the end of a tagged release - the
    least convenient moment to discover a typo.

    CONTRIBUTING.md spells the values out so nobody reverse-engineers them from
    YAML; this keeps that table honest if the workflow is ever renamed or the
    environment changes.
    """
    contributing = CONTRIBUTING.read_text()
    published = _publish_step()

    assert "`release.yml`" in contributing, (
        "CONTRIBUTING no longer names the workflow file PyPI must be told about"
    )
    assert f"| Environment | `{published['environment']}` |" in contributing, (
        f"CONTRIBUTING documents a different environment than release.yml's "
        f"{published['environment']!r}"
    )


def test_the_documented_opt_in_variable_is_the_one_that_gates_publishing() -> None:
    gate = _publish_step()["gate"]
    assert "PYPI_TRUSTED_PUBLISHING" in gate, "the publish gate variable changed"
    assert "PYPI_TRUSTED_PUBLISHING" in CONTRIBUTING.read_text(), (
        "CONTRIBUTING does not name the variable that actually enables publishing"
    )


def test_every_package_is_listed_for_publisher_configuration() -> None:
    """All ten need their own PyPI project AND their own publisher; a missing
    one publishes partially, which `skip-existing` will not rescue."""
    import tomllib

    root = WORKFLOWS.parent.parent
    names = {
        tomllib.loads(path.read_text())["project"]["name"]
        for path in (root / "packages").glob("*/pyproject.toml")
    }
    assert len(names) == 11, sorted(names)
    contributing = CONTRIBUTING.read_text()
    missing = sorted(name for name in names if f"`{name}`" not in contributing)
    assert not missing, f"CONTRIBUTING's publisher list is missing {missing}"


def test_the_recommended_branch_protection_checks_are_real_ci_jobs() -> None:
    """A required check that no job produces blocks every PR forever, and one
    that is misspelled protects nothing - both fail silently in the settings
    UI, so pin the list to ci.yml.

    `docs-publish` must stay OFF the list: it is main-only, so requiring it
    would deadlock every pull request.
    """
    ci = yaml.safe_load((WORKFLOWS / "ci.yml").read_text())
    contributing = CONTRIBUTING.read_text()

    expected: set[str] = set()
    for name, job in ci["jobs"].items():
        if "github.ref == 'refs/heads/main'" in str(job.get("if", "")):
            assert f"`{name}`," not in contributing, (
                f"{name} is main-only; requiring it would deadlock every PR"
            )
            continue
        versions = job.get("strategy", {}).get("matrix", {}).get("python")
        expected |= {f"{name} ({v})" for v in versions} if versions else {name}

    missing = sorted(check for check in expected if f"`{check}`" not in contributing)
    assert not missing, (
        f"CONTRIBUTING's branch-protection list does not mention {missing}; "
        f"an unlisted job is a gate nobody is actually requiring"
    )


def test_every_published_artifact_carries_build_provenance() -> None:
    """Wheels users install must be attestable (FEEDBACK v3 D-1).

    The repo already scans dependencies, sources, and git history, and the
    showcase publishes oras provenance for its deployable unit. The artifacts
    that actually leave this repo had none, which is the wrong way round. The
    attestation must cover the same file set the release attaches, and must
    run before the publish so a failure cannot ship unattested wheels.
    """
    job = _load("release.yml")["jobs"]["release"]
    steps = job["steps"]
    attest = next(s for s in steps if "attest-build-provenance" in str(s.get("uses", "")))
    subjects = (attest.get("with") or {}).get("subject-path", "")
    assert "dist/*.whl" in subjects and "dist/*.tar.gz" in subjects, attest

    order = {
        kind: next(i for i, s in enumerate(steps) if kind in str(s.get("uses", "")))
        for kind in ("attest-build-provenance", "gh-release", "pypi-publish")
    }
    assert order["attest-build-provenance"] < order["gh-release"] < order["pypi-publish"], order


def test_the_attestation_permission_is_actually_granted() -> None:
    """`attestations: write` is not implied by `contents: write`; without it the
    attest step fails at run time, which is exactly the class of error that
    made the first v0.1.0 tag fail at startup."""
    release = _load("release.yml")
    granted = {
        **(release.get("permissions") or {}),
        **(release["jobs"]["release"].get("permissions") or {}),
    }
    assert granted.get("attestations") == "write", granted
    assert granted.get("id-token") == "write", granted
