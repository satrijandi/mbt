"""Guards for the workflow failure notices (ci.yml, live.yml).

The live tier is the only thing that runs mbt against real external systems,
and it runs on no PR - so anything it catches is visible only to whoever opens
the Actions tab. Twice that was nobody: six consecutive red nightlies on the
0600 atomic-write bug, then a full day of red on an unpinned transitive that
would have cleared itself.

ci.yml has the same hole for a different reason: main is not watched either.
Its floors job broke on 2026-08-27 and main stayed red for eight consecutive
commits. Branch protection does not close that by itself, because required
checks are deliberately not enforced for admins.

The notice jobs close both holes, and each is shell inside YAML: nothing would
execute it until the day it matters, and `set -euo pipefail` gives it several
ways to fail on the *green* path - which would turn every passing run red and
train everyone to ignore the signal. So these tests actually RUN the scripts
against a stubbed `gh` rather than reading them. The green case is the one
worth the machinery, and the one static review is worst at, since it is the
branch that does nothing.
"""

import os
import shutil
import subprocess
from pathlib import Path

import pytest
import yaml

REPO_ROOT = Path(__file__).resolve().parent.parent
WORKFLOWS = REPO_ROOT / ".github" / "workflows"
WORKFLOW = WORKFLOWS / "live.yml"


@pytest.fixture(scope="module")
def workflow() -> dict:
    return yaml.safe_load(WORKFLOW.read_text())


@pytest.fixture(scope="module")
def ci() -> dict:
    return yaml.safe_load((WORKFLOWS / "ci.yml").read_text())


def test_the_notice_waits_for_every_job_and_runs_even_when_they_fail(workflow: dict) -> None:
    notify = workflow["jobs"]["notify"]
    tiers = {name for name in workflow["jobs"] if name != "notify"}
    assert set(notify["needs"]) == tiers, (
        f"the notice must depend on every tier ({sorted(tiers)}), or a new job "
        f"can go red without telling anyone"
    )
    # `needs` alone would skip the job when a dependency fails, which is the
    # only case it exists for.
    assert notify["if"] == "always()"


def test_the_workflow_can_actually_file_an_issue(workflow: dict) -> None:
    """`issues: write` is not the default for a scheduled workflow; without it
    the notice step fails with a 403 that reads like a gh bug."""
    assert workflow["permissions"]["issues"] == "write"


# -- the script itself ---------------------------------------------------------


def _notice_script(workflow: dict) -> str:
    steps = workflow["jobs"]["notify"]["steps"]
    runs = [step["run"] for step in steps if "run" in step]
    assert len(runs) == 1, "expected exactly one run step in the notice job"
    return runs[0]


def _run_notice(
    script: str, tmp_path: Path, *, snowflake: str, showcase: str, existing: str = ""
) -> tuple[int, list[str]]:
    """Execute the real notice script with `gh` stubbed out.

    The stub logs each invocation to a file and answers `gh issue list` with
    `existing` (an issue number, or empty for "no open issue").
    """
    bin_dir = tmp_path / "bin"
    bin_dir.mkdir()
    calls = tmp_path / "gh-calls.txt"
    (bin_dir / "gh").write_text(
        "#!/usr/bin/env bash\n"
        f'echo "$@" >> {calls}\n'
        'if [ "$1" = "issue" ] && [ "$2" = "list" ]; then\n'
        f'  printf "%s" "{existing}"\n'
        "fi\n"
        "exit 0\n"
    )
    (bin_dir / "gh").chmod(0o755)

    env = {
        **os.environ,
        "PATH": f"{bin_dir}:{os.environ['PATH']}",
        "GH_TOKEN": "stub",
        "GH_REPO": "satrijandi/mbt",
        "TITLE": "Live integration is failing",
        "SNOWFLAKE_OUTCOME": snowflake,
        "SHOWCASE_OUTCOME": showcase,
        "RUN_URL": "https://example.invalid/run/1",
    }
    completed = subprocess.run(
        ["bash", "-c", script], env=env, capture_output=True, text=True, check=False
    )
    assert not completed.stderr, completed.stderr
    logged = calls.read_text().splitlines() if calls.exists() else []
    return completed.returncode, logged


@pytest.mark.skipif(shutil.which("bash") is None, reason="needs bash")
def test_a_green_nightly_files_nothing_and_exits_zero(workflow: dict, tmp_path: Path) -> None:
    """The case this file exists for: a notifier that fails on success turns
    every green nightly red, which is worse than having no notifier at all."""
    code, calls = _run_notice(
        _notice_script(workflow), tmp_path, snowflake="success", showcase="success"
    )
    assert code == 0
    assert not [call for call in calls if call.startswith("issue create")]
    assert not [call for call in calls if call.startswith("issue comment")]


@pytest.mark.skipif(shutil.which("bash") is None, reason="needs bash")
def test_a_skipped_or_cancelled_job_is_not_reported_as_a_failure(
    workflow: dict, tmp_path: Path
) -> None:
    """`always()` means this job also runs when a tier was skipped or the run
    was cancelled; neither is evidence that mbt is broken."""
    code, calls = _run_notice(
        _notice_script(workflow), tmp_path, snowflake="skipped", showcase="cancelled"
    )
    assert code == 0
    assert not [call for call in calls if call.startswith("issue create")]


@pytest.mark.skipif(shutil.which("bash") is None, reason="needs bash")
def test_a_failing_tier_opens_one_issue_naming_it(workflow: dict, tmp_path: Path) -> None:
    code, calls = _run_notice(
        _notice_script(workflow), tmp_path, snowflake="success", showcase="failure"
    )
    assert code == 0
    created = [call for call in calls if call.startswith("issue create")]
    assert len(created) == 1, calls
    assert "showcase" in created[0]
    assert "snowflake" not in created[0].split("Failing job(s):")[1][:40]


@pytest.mark.skipif(shutil.which("bash") is None, reason="needs bash")
def test_a_second_red_night_comments_rather_than_duplicating(
    workflow: dict, tmp_path: Path
) -> None:
    code, calls = _run_notice(
        _notice_script(workflow), tmp_path, snowflake="failure", showcase="failure", existing="42"
    )
    assert code == 0
    assert not [call for call in calls if call.startswith("issue create")]
    commented = [call for call in calls if call.startswith("issue comment 42")]
    assert len(commented) == 1, calls


@pytest.mark.skipif(shutil.which("bash") is None, reason="needs bash")
def test_recovery_closes_the_open_issue(workflow: dict, tmp_path: Path) -> None:
    """Otherwise the tracking issue outlives the problem and the next reader
    cannot tell whether the tier is red now or was red in July."""
    code, calls = _run_notice(
        _notice_script(workflow), tmp_path, snowflake="success", showcase="success", existing="42"
    )
    assert code == 0
    closed = [call for call in calls if call.startswith("issue close 42")]
    assert len(closed) == 1, calls


# -- ci.yml's main-is-red notice -----------------------------------------------


def _ci_notice_script(ci: dict) -> str:
    steps = ci["jobs"]["notify"]["steps"]
    (run,) = [step["run"] for step in steps if "run" in step]
    assert isinstance(run, str)
    return run


def _run_ci_notice(
    script: str, tmp_path: Path, *, results: str, existing: str = ""
) -> tuple[int, list[str]]:
    bin_dir = tmp_path / "bin"
    bin_dir.mkdir()
    calls = tmp_path / "gh-calls.txt"
    (bin_dir / "gh").write_text(
        "#!/usr/bin/env bash\n"
        f'echo "$@" >> {calls}\n'
        'if [ "$1" = "issue" ] && [ "$2" = "list" ]; then\n'
        f'  printf "%s" "{existing}"\n'
        "fi\n"
        "exit 0\n"
    )
    (bin_dir / "gh").chmod(0o755)
    env = {
        **os.environ,
        "PATH": f"{bin_dir}:{os.environ['PATH']}",
        "GH_TOKEN": "stub",
        "GH_REPO": "satrijandi/mbt",
        "TITLE": "CI is failing on main",
        "RESULTS": results,
        "RUN_URL": "https://example.invalid/run/1",
        "SHA": "deadbeef",
    }
    done = subprocess.run(
        ["bash", "-c", script], env=env, capture_output=True, text=True, check=False
    )
    assert not done.stderr, done.stderr
    return done.returncode, (calls.read_text().splitlines() if calls.exists() else [])


def test_the_ci_notice_watches_every_gating_job(ci: dict) -> None:
    """A job left out of `needs` can go red without the notice firing, which
    is precisely the failure being fixed. docs-publish is excluded on purpose:
    it depends on docs and only runs on main, so needing it would be circular
    noise rather than a gate."""
    notify = ci["jobs"]["notify"]
    gating = {
        name
        for name, job in ci["jobs"].items()
        if name not in {"notify", "docs-publish"} and "uses" not in job
    }
    assert set(notify["needs"]) == gating, (
        f"ci.yml's notify watches {sorted(notify['needs'])} but the gating jobs "
        f"are {sorted(gating)}"
    )
    assert notify["if"].startswith("always()")
    assert notify["permissions"]["issues"] == "write"


@pytest.mark.skipif(shutil.which("bash") is None, reason="needs bash")
def test_a_green_main_files_nothing(ci: dict, tmp_path: Path) -> None:
    code, calls = _run_ci_notice(
        _ci_notice_script(ci), tmp_path, results="success success success success"
    )
    assert code == 0
    assert not [c for c in calls if c.startswith("issue create")]


@pytest.mark.skipif(shutil.which("bash") is None, reason="needs bash")
def test_a_skipped_job_is_not_a_red_main(ci: dict, tmp_path: Path) -> None:
    """`success skipped success` must not read as failure - the substring
    match is on " failure ", and getting that wrong in either direction is
    the classic bug in this shape of script."""
    code, calls = _run_ci_notice(_ci_notice_script(ci), tmp_path, results="success skipped success")
    assert code == 0
    assert not [c for c in calls if c.startswith("issue create")]


@pytest.mark.skipif(shutil.which("bash") is None, reason="needs bash")
def test_a_failing_job_opens_the_issue(ci: dict, tmp_path: Path) -> None:
    for results in ("failure success success", "success success failure", "failure"):
        target = tmp_path / results.replace(" ", "_")
        target.mkdir()
        code, calls = _run_ci_notice(_ci_notice_script(ci), target, results=results)
        assert code == 0, results
        created = [c for c in calls if c.startswith("issue create")]
        assert len(created) == 1, (results, calls)


@pytest.mark.skipif(shutil.which("bash") is None, reason="needs bash")
def test_a_recovered_main_closes_the_issue(ci: dict, tmp_path: Path) -> None:
    code, calls = _run_ci_notice(
        _ci_notice_script(ci), tmp_path, results="success success", existing="7"
    )
    assert code == 0
    assert len([c for c in calls if c.startswith("issue close 7")]) == 1, calls
