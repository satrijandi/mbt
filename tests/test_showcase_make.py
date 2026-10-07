"""The runbook itself, exercised (SHOW-18): drive the README golden path
through `make` exactly as a human would - doctor, up, demo (a refused
challenger included), ci, clone, protect, lifecycle (its promotion a reviewed
PR), score, outcomes, monitor, inject-drift + recovery, week, down, clean.

Every other module tests the platform through its own harness; this one
tests that the COMMANDS THE README TELLS A HUMAN TO TYPE still work, so the
runbook cannot drift from reality silently.

Extra gate (like the k3d tier): MBT_LIVE_SHOWCASE_MAKE=1 on top of the
usual double gate. The module boots a full second stack via the Makefile,
so it must run in its own pytest invocation, never concurrently with the
session-stack modules (two full stacks would exceed the RAM guardrails).
Isolation: SHOWCASE_PROJECT/SHOWCASE_WORKSPACE/port overrides keep it away
from any real `make up` stack a developer has running.
"""

import os
import subprocess
import uuid
from pathlib import Path

import pytest
from showcase_utils import (
    DS_USER,
    GITEA_PASSWORD,
    GITEA_USER,
    SHOWCASE_DIR,
    SHOWCASE_MARKS,
    docker_sock_gid,
    free_port,
    service_log_tails,
)

pytestmark = [
    *SHOWCASE_MARKS,
    pytest.mark.skipif(
        os.environ.get("MBT_LIVE_SHOWCASE_MAKE") != "1",
        reason="make-runbook tier is separately opt-in: set MBT_LIVE_SHOWCASE_MAKE=1",
    ),
]

#: The Makefile boots every profile, so a log dump has to name them all too.
PROFILES = ("core", "spark", "dev", "obs", "ci", "orch")

PORT_VARS = (
    "SHOWCASE_S3_PORT",
    "SHOWCASE_FILER_PORT",
    "SHOWCASE_MLFLOW_PORT",
    "SHOWCASE_SPARK_UI_PORT",
    "SHOWCASE_JUPYTER_PORT",
    "SHOWCASE_PUSHGW_PORT",
    "SHOWCASE_ALERTMANAGER_PORT",
    "SHOWCASE_PROMETHEUS_PORT",
    "SHOWCASE_GRAFANA_PORT",
    "SHOWCASE_GITEA_PORT",
    "SHOWCASE_WOODPECKER_PORT",
    "SHOWCASE_WEBHOOK_PORT",
    "SHOWCASE_ZOT_PORT",
    "SHOWCASE_AIRFLOW_PORT",
)


class MakeRunner:
    """Run make targets against an isolated project/workspace/port set."""

    def __init__(self, workspace: Path) -> None:
        self.project = f"mbt-make-{uuid.uuid4().hex[:8]}"
        self.workspace = workspace
        self.env = os.environ.copy()
        self.env.update({name: str(free_port()) for name in PORT_VARS})
        self.env.update(
            {
                "SHOWCASE_PROJECT": self.project,
                "SHOWCASE_NETWORK": f"{self.project}_default",
                "SHOWCASE_WORKSPACE": str(workspace),
                "DOCKER_SOCK_GID": str(docker_sock_gid()),
            }
        )

    def make(self, target: str, timeout: int = 2400) -> None:
        proc = subprocess.run(
            ["make", "-C", str(SHOWCASE_DIR), target],
            env=self.env,
            capture_output=True,
            text=True,
            timeout=timeout,
            check=False,
        )
        if proc.returncode != 0:
            logs = service_log_tails(self.compose)
            pytest.fail(
                f"make {target} exited {proc.returncode}\n--- stdout ---\n{proc.stdout[-8000:]}"
                f"\n--- stderr ---\n{proc.stderr[-8000:]}\n--- stack logs ---\n{logs}"
            )

    def compose(self, *args: str) -> str:
        """A compose subcommand against this runner's project; its stdout."""
        return subprocess.run(
            [
                "docker",
                "compose",
                "-p",
                self.project,
                "-f",
                str(SHOWCASE_DIR / "compose" / "docker-compose.yml"),
                *(flag for profile in PROFILES for flag in ("--profile", profile)),
                *args,
            ],
            env=self.env,
            capture_output=True,
            text=True,
            timeout=120,
            check=False,
        ).stdout

    def containers(self) -> list[str]:
        proc = subprocess.run(
            ["docker", "ps", "-aq", "--filter", f"name={self.project}"],
            capture_output=True,
            text=True,
            timeout=60,
            check=False,
        )
        return [line for line in proc.stdout.splitlines() if line]


def _git(repo: Path, *args: str, check: bool = True) -> subprocess.CompletedProcess:
    # credential.helper= keeps a developer's keychain out of it: the clone's
    # remote URL is the only credential `make clone` promises.
    proc = subprocess.run(
        ["git", "-C", str(repo), "-c", "credential.helper=", *args],
        capture_output=True,
        text=True,
        timeout=120,
        check=False,
    )
    if check and proc.returncode != 0:
        pytest.fail(f"git {' '.join(args)} in {repo} exited {proc.returncode}:\n{proc.stderr}")
    return proc


@pytest.fixture(scope="module")
def runbook(tmp_path_factory: pytest.TempPathFactory):
    runner = MakeRunner(tmp_path_factory.mktemp("showcase-make-ws"))
    try:
        yield runner
    finally:
        # Idempotent even after the test's own down/clean.
        subprocess.run(
            ["make", "-C", str(SHOWCASE_DIR), "clean"],
            env=runner.env,
            capture_output=True,
            timeout=600,
            check=False,
        )


def test_runbook_golden_path(runbook) -> None:
    """README top to bottom: every documented make target exits 0 and leaves
    the artifacts it promises."""
    runner = runbook
    ws = runner.workspace

    runner.make("doctor")
    runner.make("up")
    assert (ws / "project" / "mbt_project.yml").exists(), "workspace was not staged"

    # The narrated lifecycle on the one lake table: dev + prod builds, the
    # champion promoted, the newest cohort scored, its outcomes landed, and
    # ground truth monitored.
    runner.make("demo")
    runs = list((ws / "predictions" / "retention_scores").glob("*/_SUCCESS"))
    assert runs, "demo left no prediction runs"
    assert list((ws / "predictions" / "retention_scores").glob("*/ground_truth.marker.json")), (
        "demo's monitor evaluated nothing after the outcomes landed"
    )

    # The CI seeding target, then the two things its output tells a human
    # to do: open the repo URL and log into Woodpecker with the Gitea
    # account (the OAuth dance against the host-published ports).
    runner.make("ci")
    import json
    import sys

    import requests

    gitea_port = runner.env["SHOWCASE_GITEA_PORT"]
    repo_page = requests.get(f"http://localhost:{gitea_port}/mbt-showcase/churn", timeout=30)
    assert repo_page.ok, f"churn repo page -> {repo_page.status_code}"
    login = subprocess.run(
        [
            sys.executable,
            str(SHOWCASE_DIR / "scripts" / "ci_bootstrap.py"),
            "login",
            "--gitea-url",
            f"http://localhost:{gitea_port}",
            "--woodpecker-url",
            f"http://localhost:{runner.env['SHOWCASE_WOODPECKER_PORT']}",
            "--user",
            GITEA_USER,
            "--password",
            GITEA_PASSWORD,
        ],
        capture_output=True,
        text=True,
        timeout=300,
        check=False,
    )
    assert login.returncode == 0, f"browser login failed:\n{login.stdout}\n{login.stderr}"
    assert json.loads(login.stdout.strip().splitlines()[-1])["woodpecker_token"]

    # The walkthrough's setup targets: a working copy per persona that can
    # push (branch pushes - no pipeline to wait for), then the governance
    # rule, which must refuse the data scientist's direct push to main.
    work = ws.parent / "showcase-work"
    runner.env["SHOWCASE_WORK"] = str(work)
    runner.make("clone")
    runner.make("clone")  # an existing clone is left alone
    for clone, user in ((work / "ops", GITEA_USER), (work / "ds", "mbtds")):
        assert _git(clone, "config", "user.name").stdout.strip() == user
        _git(clone, "checkout", "-q", "-b", f"hello-{user}")
        _git(clone, "commit", "-q", "--allow-empty", "-m", f"{user} can push")
        _git(clone, "push", "-q", "origin", f"hello-{user}")

    runner.make("protect")
    runner.make("protect")  # re-running resets the rule, never fails
    api = f"http://localhost:{gitea_port}/api/v1/repos/mbt-showcase/churn"
    auth = (GITEA_USER, GITEA_PASSWORD)
    rule = requests.get(f"{api}/branch_protections/main", auth=auth, timeout=30).json()
    assert rule["block_admin_merge_override"] is True, rule
    assert rule["required_approvals"] == 1, rule
    assert rule["approvals_whitelist_username"] == [GITEA_USER], rule
    owners = requests.get(f"{api}/contents/CODEOWNERS?ref=main", auth=auth, timeout=30)
    assert owners.ok, owners.status_code

    ds = work / "ds"
    _git(ds, "checkout", "-q", "main")
    _git(ds, "pull", "-q")
    _git(ds, "commit", "-q", "--allow-empty", "-m", "straight to main")
    refused = _git(ds, "push", "origin", "main", check=False)
    assert refused.returncode != 0
    assert "protected branch" in refused.stderr, refused.stderr

    # The scheduled lifecycle: no unit is pinned yet (the pushes above were
    # branches), so the target pushes to main as the owner - allowed past the
    # protection rule - bakes the first unit, then runs every DAG in the
    # pinned unit: retrain, score, monitor before and after the outcomes.
    runner.make("lifecycle", timeout=3600)
    deploy = requests.get(
        f"http://localhost:{gitea_port}/mbt-showcase/deploy/raw/branch/main/images.env",
        timeout=30,
    )
    assert "@sha256:" in deploy.text, deploy.text
    af = f"http://localhost:{runner.env['SHOWCASE_AIRFLOW_PORT']}"
    af_token = requests.post(
        f"{af}/auth/token", json={"username": "admin", "password": "admin"}, timeout=60
    ).json()["access_token"]
    for dag_id, runs in (("mbt_retrain", 1), ("mbt_score", 1), ("mbt_monitor", 2)):
        payload = requests.get(
            f"{af}/api/v2/dags/{dag_id}/dagRuns",
            headers={"Authorization": f"Bearer {af_token}"},
            timeout=60,
        ).json()
        states = [run["state"] for run in payload["dag_runs"]]
        assert states == ["success"] * runs, (dag_id, states)
    assert list((ws / "predictions" / "retention_scores").glob("*/ground_truth.marker.json"))
    # The promotion went the reviewed way: mbtds's promotions.yml PR, merged by
    # the code owner, and the version it pins is the one production serves.
    pulls = requests.get(f"{api}/pulls?state=closed", auth=auth, timeout=30).json()
    promotion = next(p for p in pulls if p["title"].startswith("promote churn_automl v"))
    assert promotion["merged"] and promotion["user"]["login"] == DS_USER, promotion
    version = promotion["title"].rsplit("v", 1)[1]
    mlflow = f"http://localhost:{runner.env['SHOWCASE_MLFLOW_PORT']}"
    alias = requests.get(
        f"{mlflow}/api/2.0/mlflow/registered-models/alias",
        params={"name": "churn_automl", "alias": "production"},
        timeout=30,
    ).json()
    assert alias["model_version"]["version"] == version, alias

    # The standalone targets rerun cleanly on the same anchors, from the
    # seeded table: reset, score, a week passes, monitor.
    runner.make("reset")
    runner.make("score")
    runner.make("outcomes")
    runner.make("monitor")

    # Drift injection breaches (tolerated by the target); reset + score
    # recovers - the runbook's documented poison/recover loop.
    runner.make("inject-drift")
    runner.make("reset")
    runner.make("score")

    # A week passes (the seeded 2026-09-28 lake moves to 2026-10-05, which
    # has happened on any date this runs) and its operations run: monitor,
    # the weekly retrain against the champion gate, promote-if-staged, score.
    runner.make("week")

    runner.make("down")
    assert runner.containers() == [], "make down left containers behind"

    # Re-staging over a previous run has to work: `make up` is the documented
    # way back into the stack and does not imply `make clean` first. Everything
    # above wrote into the workspace as root, so on native Linux the host
    # cannot unlink project/target - the target clears it through a container
    # the way `clean` does, where a plain rsync --delete failed outright.
    assert (ws / "project" / "target").is_dir(), "expected build output to re-stage over"
    runner.make("workspace")
    assert not (ws / "project" / "target").exists(), (
        "re-staging left the previous run's root-owned build output behind"
    )
    assert (ws / "project" / "mbt_project.yml").exists(), "re-staging lost the project"

    runner.make("clean")
    assert not ws.exists(), "make clean left the workspace behind"
