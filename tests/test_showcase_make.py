"""The runbook itself, exercised (SHOW-18): drive the README golden path
through `make` exactly as a human would - up, demo, monthly, score, monitor,
inject-drift + recovery, down, clean.

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
    GITEA_PASSWORD,
    GITEA_USER,
    SHOWCASE_DIR,
    SHOWCASE_MARKS,
    docker_sock_gid,
    free_port,
)

pytestmark = [
    *SHOWCASE_MARKS,
    pytest.mark.skipif(
        os.environ.get("MBT_LIVE_SHOWCASE_MAKE") != "1",
        reason="make-runbook tier is separately opt-in: set MBT_LIVE_SHOWCASE_MAKE=1",
    ),
]

PORT_VARS = (
    "SHOWCASE_S3_PORT",
    "SHOWCASE_FILER_PORT",
    "SHOWCASE_MLFLOW_PORT",
    "SHOWCASE_SPARK_UI_PORT",
    "SHOWCASE_JUPYTER_PORT",
    "SHOWCASE_PUSHGW_PORT",
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
            logs = subprocess.run(
                [
                    "docker",
                    "compose",
                    "-p",
                    self.project,
                    "logs",
                    "--no-color",
                    "--tail",
                    "100",
                ],
                capture_output=True,
                text=True,
                timeout=120,
                check=False,
            ).stdout
            pytest.fail(
                f"make {target} exited {proc.returncode}\n--- stdout ---\n{proc.stdout[-8000:]}"
                f"\n--- stderr ---\n{proc.stderr[-8000:]}\n--- stack logs ---\n{logs[-8000:]}"
            )

    def containers(self) -> list[str]:
        proc = subprocess.run(
            ["docker", "ps", "-aq", "--filter", f"name={self.project}"],
            capture_output=True,
            text=True,
            timeout=60,
            check=False,
        )
        return [line for line in proc.stdout.splitlines() if line]


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

    runner.make("up")
    assert (ws / "project" / "mbt_project.yml").exists(), "workspace was not staged"

    # The narrated lifecycle: dev + prod builds, both champions promoted,
    # both cadences scored, ground truth monitored.
    runner.make("demo")
    daily_runs = list((ws / "lake_local" / "predictions" / "retention_scores").glob("*/_SUCCESS"))
    assert daily_runs, "demo left no daily prediction runs"
    monthly_runs = list(
        (ws / "lake_local" / "predictions" / "monthly_retention_scores").glob("*/_SUCCESS")
    )
    assert monthly_runs, "demo left no monthly prediction runs"

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

    # Standalone cadence targets rerun cleanly on the same anchors.
    runner.make("monthly")
    runner.make("score")
    runner.make("monitor")

    # Drift injection breaches (tolerated by the target), a plain score
    # recovers - the runbook's documented poison/recover loop.
    runner.make("inject-drift")
    runner.make("score")

    runner.make("down")
    assert runner.containers() == [], "make down left containers behind"

    runner.make("clean")
    assert not ws.exists(), "make clean left the workspace behind"
