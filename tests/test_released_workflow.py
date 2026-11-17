"""Guards for the released-tag tier (.github/workflows/released.yml), FEEDBACK v7 F-1.

That tier's whole value is that it installs the PUBLISHED tag the way
docs/installation.md says and walks docs/quickstart.md to the end. v0.2.0 went
stale when DuckDB 2.0 arrived, and every fresh install failed its first build
while main and every other tier stayed green. Each invariant asserted here is
one that, if quietly "cleaned up", turns the job into a slower duplicate of
ci.yml, which tests main.
"""

import os
import re
import shutil
import subprocess
from pathlib import Path

import pytest
import yaml

REPO_ROOT = Path(__file__).resolve().parent.parent
WORKFLOW = REPO_ROOT / ".github" / "workflows" / "released.yml"

#: The packages docs/installation.md installs for the quickstart.
INSTALLED = ("mbt-adapter-base", "mbt-core", "mbt-xgboost", "mbt-mlflow")

#: docs/quickstart.md, in order. Each must appear as a command of the job.
QUICKSTART = (
    "mbt init",
    "generate_sample_data.py",
    "mbt parse",
    "mbt build",
    "mbt run --manifest target/manifest.json",
    "mbt promote --model churn_classifier --to production",
    "mbt score",
    "mbt monitor",
    'mbt monitor --anchor "$later"',
    "mbt predictions ls",
)


@pytest.fixture(scope="module")
def workflow() -> dict:
    return yaml.safe_load(WORKFLOW.read_text())


@pytest.fixture(scope="module")
def steps(workflow: dict) -> list[dict]:
    jobs = workflow["jobs"]
    assert len(jobs) == 1, "one job keeps the install and the quickstart in one environment"
    return next(iter(jobs.values()))["steps"]


def _step(steps: list[dict], name_part: str) -> dict:
    (found,) = [s for s in steps if name_part in s.get("name", "")]
    return found


def test_it_runs_nightly_and_on_demand(workflow: dict) -> None:
    # PyYAML reads a bare `on:` key as the boolean True (YAML 1.1).
    triggers = workflow.get("on") or workflow[True]
    assert triggers["schedule"], "a release goes stale between releases, so this has to be nightly"
    assert "workflow_dispatch" in triggers


def test_it_checks_nothing_out_of_the_repository(steps: list[dict]) -> None:
    """A checkout would be main, and main is what every other tier tests. The
    whole point is the published tag; anything read from a working tree could
    quietly turn this into a test of the wrong thing."""
    assert not [s for s in steps if "actions/checkout" in s.get("uses", "")]
    commands = "\n".join(s.get("run", "") for s in steps)
    assert "./packages" not in commands and "pip install ." not in commands


def test_it_installs_the_tag_by_git_reference_with_no_lock(steps: list[dict]) -> None:
    install = _step(steps, "Install it")["run"]
    for package in INSTALLED:
        assert re.search(
            rf'"{package} @ git\+\$REPO_URL@\$TAG#subdirectory=packages/{package}"', install
        ), package
    assert "uv.lock" not in install
    assert "--frozen" not in install and "uv sync" not in install


def test_the_uv_cache_is_explicitly_disabled(steps: list[dict]) -> None:
    """Cached index metadata is a stale view of what a user resolves today."""
    setup = next(s for s in steps if "setup-uv" in s.get("uses", ""))
    assert setup["with"]["enable-cache"] is False


def test_it_walks_the_quickstart_in_order_and_fails_on_any_step(steps: list[dict]) -> None:
    run = _step(steps, "quickstart")["run"]
    assert "set -euo pipefail" in run
    positions = [run.find(command) for command in QUICKSTART]
    assert -1 not in positions, dict(zip(QUICKSTART, positions, strict=True))
    assert positions == sorted(positions), "the steps must run in the documented order"


def test_it_runs_in_an_empty_home(steps: list[dict]) -> None:
    """Nothing of the runner's own ~/.mbt may rescue a broken install."""
    assert "export HOME=" in _step(steps, "quickstart")["run"]


def test_the_quickstart_it_walks_is_the_documented_one() -> None:
    """If docs/quickstart.md gains or drops a step, this tier must follow."""
    doc = (REPO_ROOT / "docs" / "quickstart.md").read_text()
    for command in (
        "mbt init my_models",
        "mbt parse",
        "mbt build",
        "mbt run --manifest target/manifest.json",
        "mbt promote --model churn_classifier --to production",
        "mbt score",
        "mbt monitor",
        "mbt predictions ls",
    ):
        assert command in doc, f"docs/quickstart.md no longer says {command!r}"


def test_the_workflow_can_actually_file_an_issue(workflow: dict) -> None:
    assert workflow["permissions"]["issues"] == "write"


# -- the notice script, run for real against a stubbed gh -------------------------


def _notice(steps: list[dict]) -> str:
    run = _step(steps, "tracking issue")["run"]
    assert isinstance(run, str)
    return run


def _run_notice(
    script: str, tmp_path: Path, *, outcome: str, existing: str = ""
) -> tuple[int, list[str]]:
    tmp_path.mkdir(parents=True, exist_ok=True)
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
        "TITLE": "The newest release fails the quickstart on a fresh install",
        "OUTCOME": outcome,
        "TAG": "v9.9.9",
        "RUN_URL": "https://example.invalid/run/1",
    }
    done = subprocess.run(
        ["bash", "-c", script], env=env, capture_output=True, text=True, check=False
    )
    assert not done.stderr, done.stderr
    return done.returncode, calls.read_text().splitlines() if calls.exists() else []


@pytest.mark.skipif(shutil.which("bash") is None, reason="needs bash")
def test_a_green_night_files_nothing_and_exits_zero(steps: list[dict], tmp_path: Path) -> None:
    code, calls = _run_notice(_notice(steps), tmp_path, outcome="success")
    assert code == 0
    assert not [c for c in calls if c.startswith(("issue create", "issue comment", "issue close"))]


@pytest.mark.skipif(shutil.which("bash") is None, reason="needs bash")
def test_a_red_night_opens_one_issue_naming_the_tag(steps: list[dict], tmp_path: Path) -> None:
    code, calls = _run_notice(_notice(steps), tmp_path, outcome="failure")
    assert code == 0
    created = [c for c in calls if c.startswith("issue create")]
    assert len(created) == 1, calls
    assert "v9.9.9" in created[0]


@pytest.mark.skipif(shutil.which("bash") is None, reason="needs bash")
def test_a_second_red_night_comments_and_recovery_closes(steps: list[dict], tmp_path: Path) -> None:
    code, calls = _run_notice(_notice(steps), tmp_path / "red", outcome="failure", existing="7")
    assert code == 0
    assert not [c for c in calls if c.startswith("issue create")]
    assert len([c for c in calls if c.startswith("issue comment 7")]) == 1
    code, calls = _run_notice(_notice(steps), tmp_path / "green", outcome="success", existing="7")
    assert code == 0
    assert len([c for c in calls if c.startswith("issue close 7")]) == 1
