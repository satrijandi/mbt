"""Guards for the upstream-resolution tier (.github/workflows/upstream.yml).

That tier's whole value is that it does NOT use uv.lock: it re-resolves to the
newest versions our constraints allow and runs the e2e seams against them,
which is the only thing that would have caught h2o moving MOJO export behind a
paid tier. Every invariant asserted here is one that, if quietly "cleaned up",
turns the job into a slower duplicate of ci.yml that detects nothing.

Also covers scripts/lock_versions.py, which renders the resolution diff the
tier reports; without a legible diff a failure is undebuggable.
"""

import importlib.util
from pathlib import Path

import pytest
import yaml

REPO_ROOT = Path(__file__).resolve().parent.parent
WORKFLOW = REPO_ROOT / ".github" / "workflows" / "upstream.yml"

_spec = importlib.util.spec_from_file_location(
    "lock_versions", REPO_ROOT / "scripts" / "lock_versions.py"
)
assert _spec is not None and _spec.loader is not None
lock_versions = importlib.util.module_from_spec(_spec)
_spec.loader.exec_module(lock_versions)


@pytest.fixture(scope="module")
def workflow() -> dict:
    # PyYAML reads a bare `on:` key as the boolean True (YAML 1.1), which is
    # why the trigger lookup below goes through it.
    return yaml.safe_load(WORKFLOW.read_text())


@pytest.fixture(scope="module")
def steps(workflow: dict) -> list[dict]:
    jobs = workflow["jobs"]
    assert len(jobs) == 1, "one job keeps the resolution and the tests in one env"
    return next(iter(jobs.values()))["steps"]


def _run_commands(steps: list[dict]) -> str:
    return "\n".join(step.get("run", "") for step in steps)


def test_it_discards_the_lock_and_re_resolves(steps: list[dict]) -> None:
    """Without --upgrade this job just re-runs ci.yml against the same pins."""
    assert "uv lock --upgrade" in _run_commands(steps)


def test_it_runs_the_e2e_tier(steps: list[dict]) -> None:
    """floors already re-resolves but runs only `-m "not e2e"`, so it could not
    see the h2o MOJO paywall. Running e2e here is the entire point."""
    assert "-m e2e" in _run_commands(steps)


def test_the_uv_cache_is_explicitly_disabled(steps: list[dict]) -> None:
    """Cached index metadata is a stale view of what is newest."""
    setup_uv = [s for s in steps if "setup-uv" in s.get("uses", "")]
    assert setup_uv, "the tier needs uv"
    assert setup_uv[0]["with"]["enable-cache"] is False


def test_it_is_scheduled_and_dispatchable(workflow: dict) -> None:
    triggers = workflow[True]  # the bare `on:` key
    assert "schedule" in triggers, "an upstream watch that never runs is not a watch"
    assert "workflow_dispatch" in triggers, "must be runnable on demand to check a fix"


def test_it_never_commits_the_re_resolved_lock(steps: list[dict]) -> None:
    """This tier reports on the world; it does not decide what we ship."""
    commands = _run_commands(steps)
    assert "git commit" not in commands
    assert "git push" not in commands


def test_failure_is_surfaced_rather_than_left_in_a_log(steps: list[dict]) -> None:
    """A scheduled job going red unnoticed is how the live tier sat broken for
    six nights; this tier raises an issue instead."""
    notify = [s for s in steps if "gh issue" in s.get("run", "")]
    assert notify, "a red nightly nobody reads is not a detector"
    assert notify[0].get("if") == "always()", "must also run on success, to close the issue"
    assert "gh issue comment" in notify[0]["run"], "update one issue, do not open a new one nightly"
    assert "gh issue close" in notify[0]["run"], "and close it when upstream goes green"


def test_lock_versions_renders_sorted_name_version_pairs() -> None:
    lock = """
        version = 1
        [[package]]
        name = "zulu"
        version = "2.0"
        [[package]]
        name = "alpha"
        version = "1.0"
    """
    assert lock_versions.versions(lock) == ["alpha==1.0", "zulu==2.0"]


def test_lock_versions_handles_a_lock_with_no_packages() -> None:
    assert lock_versions.versions("version = 1\n") == []


def test_lock_versions_reports_usage_instead_of_traceback() -> None:
    assert lock_versions.main([]) == 2
    assert lock_versions.main(["a", "b"]) == 2


def test_lock_versions_reads_the_real_lock() -> None:
    """The rendering has to work on the actual lock, not just a fixture."""
    rendered = lock_versions.versions((REPO_ROOT / "uv.lock").read_text())
    assert len(rendered) > 100
    assert rendered == sorted(rendered)
    assert any(line.startswith("h2o==") for line in rendered)
