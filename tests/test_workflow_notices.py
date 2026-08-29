"""Guards for the live tier's failure notice (.github/workflows/live.yml).

The live tier is the only thing that runs mbt against real external systems,
and it runs on no PR - so anything it catches is visible only to whoever opens
the Actions tab. Twice that was nobody: six consecutive red nightlies on the
0600 atomic-write bug, then a full day of red on an unpinned transitive that
would have cleared itself.

The notice job closes that hole, and it is shell inside YAML: nothing would
execute it until the night it matters, and its `set -euo pipefail` gives it
several ways to fail on the *green* path, which would turn every passing
nightly red and train everyone to ignore the tier. So these tests actually RUN
the script against a stubbed `gh` rather than reading it - the green-nightly
case is the one worth the machinery, and it is the one static review is worst
at, since it is the branch that does nothing.
"""

import os
import shutil
import subprocess
from pathlib import Path

import pytest
import yaml

REPO_ROOT = Path(__file__).resolve().parent.parent
WORKFLOW = REPO_ROOT / ".github" / "workflows" / "live.yml"


@pytest.fixture(scope="module")
def workflow() -> dict:
    return yaml.safe_load(WORKFLOW.read_text())


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
