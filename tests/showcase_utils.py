"""Shared harness for the docker-compose showcase tier (examples/showcase).

Opt-in gating mirrors the live Snowflake suite (double gate): module-level
skipif unless MBT_LIVE_SHOWCASE=1, then loud pytest.fail once opted in if
docker/compose are missing. Modules using this helper apply SHOWCASE_MARKS
as their pytestmark.

Module ordering and coupling: the modules share one session-scoped stack
and run in collection (alphabetical) order. Exactly ONE ordering constraint
is load-bearing: test_showcase_ci must be the first forge consumer (its
bootstrap test asserts a virgin Woodpecker). Every other module is
standalone-safe by construction - it provisions or promotes whatever it
needs (ensure_seeded, ensure_daily_champion, the scheduling fixture's
promote-if-missing loop) - and every score/monitor invocation is
cadence-scoped (--select tag:...) so adding a cadence never breaks a
neighbor.

Everything the stack writes lives under the pytest tmp workspace or in
compose-project-scoped docker volumes; teardown is `down -v` in a finally.
"""

import json
import os
import shutil
import socket
import subprocess
import sys
import uuid
from pathlib import Path

import pytest

REPO_ROOT = Path(__file__).resolve().parent.parent
SHOWCASE_DIR = REPO_ROOT / "examples" / "showcase"
CHURN_DEMO_DATA = REPO_ROOT / "tests" / "fixtures" / "churn_demo" / "data"

SKIP_REASON = (
    "showcase stack tests are opt-in: set MBT_LIVE_SHOWCASE=1 with docker running "
    "(examples/showcase/README.md)"
)
SHOWCASE_MARKS = [
    pytest.mark.live,
    pytest.mark.live_showcase,
    pytest.mark.timeout(3600),
    pytest.mark.skipif(os.environ.get("MBT_LIVE_SHOWCASE") != "1", reason=SKIP_REASON),
]

ANCHOR = "2026-06-30T00:00:00Z"
MONITOR_ANCHOR = "2026-07-20T00:00:00Z"
RUNNER_IMAGE = os.environ.get("MBT_SHOWCASE_RUNNER_IMAGE", "mbt-showcase-runner:dev")


def require_docker() -> None:
    """Gate 2: opted in but docker unusable must FAIL loudly, never skip."""
    if shutil.which("docker") is None:
        pytest.fail("MBT_LIVE_SHOWCASE=1 but docker is not on PATH (install/start Docker)")
    probe = subprocess.run(
        ["docker", "info", "--format", "{{.ServerVersion}}"],
        capture_output=True,
        text=True,
        timeout=60,
        check=False,
    )
    if probe.returncode != 0:
        pytest.fail("MBT_LIVE_SHOWCASE=1 but the docker daemon is not reachable:\n" + probe.stderr)


def free_port() -> int:
    with socket.socket() as sock:
        sock.bind(("127.0.0.1", 0))
        return sock.getsockname()[1]


def docker_sock_gid() -> int:
    """GID of the group owning /var/run/docker.sock as containers see it.

    airflow-scheduler's non-root user joins this group to run DAG tasks.
    Native Linux bind-mounts the host socket (root:docker), so the host
    stat is authoritative; Docker Desktop resolves the mount inside its VM
    where the socket is group 0, and the host stat would be wrong.
    """
    sock = Path("/var/run/docker.sock")
    if sys.platform == "linux" and sock.exists():
        return sock.stat().st_gid
    return 0


class ComposeStack:
    """One compose project (core+spark+dev+obs+ci profiles), tmp workspace."""

    def __init__(self, workspace: Path) -> None:
        self.project_name = f"mbt-show-{uuid.uuid4().hex[:8]}"
        self.workspace = workspace
        self.ports = {
            "SHOWCASE_S3_PORT": free_port(),
            "SHOWCASE_FILER_PORT": free_port(),
            "SHOWCASE_MLFLOW_PORT": free_port(),
            "SHOWCASE_SPARK_UI_PORT": free_port(),
            "SHOWCASE_JUPYTER_PORT": free_port(),
            "SHOWCASE_PUSHGW_PORT": free_port(),
            "SHOWCASE_PROMETHEUS_PORT": free_port(),
            "SHOWCASE_GRAFANA_PORT": free_port(),
            "SHOWCASE_GITEA_PORT": free_port(),
            "SHOWCASE_WOODPECKER_PORT": free_port(),
            "SHOWCASE_WEBHOOK_PORT": free_port(),
            "SHOWCASE_ZOT_PORT": free_port(),
            "SHOWCASE_AIRFLOW_PORT": free_port(),
        }
        self.env = os.environ.copy()
        self.env.update({k: str(v) for k, v in self.ports.items()})
        self.env["SHOWCASE_WORKSPACE"] = str(workspace)
        self.env["SHOWCASE_RUNNER_IMAGE"] = RUNNER_IMAGE
        # Woodpecker's agent spawns step containers OUTSIDE compose; this is
        # how they join the session network and resolve gitea/seaweedfs/
        # mlflow/webhook-sink by name (compose names it <project>_default).
        self.env["SHOWCASE_NETWORK"] = f"{self.project_name}_default"
        self.env["DOCKER_SOCK_GID"] = str(docker_sock_gid())

    # -- docker plumbing -----------------------------------------------------
    def compose(self, *args: str, timeout: int = 300) -> subprocess.CompletedProcess[str]:
        cmd = [
            "docker",
            "compose",
            "-p",
            self.project_name,
            "-f",
            str(SHOWCASE_DIR / "compose" / "docker-compose.yml"),
            "--profile",
            "core",
            "--profile",
            "spark",
            "--profile",
            "dev",
            "--profile",
            "obs",
            "--profile",
            "ci",
            "--profile",
            "orch",
            *args,
        ]
        return subprocess.run(
            cmd, capture_output=True, text=True, timeout=timeout, env=self.env, check=False
        )

    def logs(self) -> str:
        # --tail is per SERVICE: a chatty service (spark, airflow) must not
        # scroll the interesting one out of the single flat truncation.
        return self.compose("logs", "--no-color", "--tail", "150", timeout=120).stdout[-60000:]

    def up(self) -> None:
        self._stage_workspace()
        proc = self.compose("up", "-d", "--wait", timeout=900)
        if proc.returncode != 0:
            self.down()
            pytest.fail(
                f"compose up failed:\n{proc.stdout}\n{proc.stderr}\n--- logs ---\n{self.logs()}"
            )

    def down(self) -> None:
        self.compose("down", "-v", "--remove-orphans", timeout=300)

    def _stage_workspace(self) -> None:
        # tmp/ and monitoring/ are created here, by the host, on purpose: every
        # container runs as root, so whichever one reached the directory first
        # would own it and the host could no longer write into it. monitoring/
        # holds the persisted serving baseline (the DAG's unit containers are
        # ephemeral) and the poisoned batch the serving-gate test writes.
        for shared in ("tmp", "monitoring"):
            (self.workspace / shared).mkdir(parents=True, exist_ok=True)
        for src, dest in (
            (SHOWCASE_DIR / "project", self.workspace / "project"),
            (CHURN_DEMO_DATA, self.workspace / "seed"),
            (SHOWCASE_DIR / "bootstrap", self.workspace / "bootstrap"),
        ):
            if dest.exists():
                shutil.rmtree(dest)
            shutil.copytree(src, dest)
        # The monthly tables live beside the showcase (SHOW-17); merge them
        # into the same seed dir the churn_demo tables come from.
        shutil.copytree(SHOWCASE_DIR / "data", self.workspace / "seed", dirs_exist_ok=True)

    # -- in-container execution ----------------------------------------------
    def exec(
        self,
        *args: str,
        workdir: str = "/workspace/project",
        expect_exit: int = 0,
        timeout: int = 1200,
    ) -> subprocess.CompletedProcess[str]:
        proc = self.compose(
            "exec", "-T", "--workdir", workdir, "jupyterlab", *args, timeout=timeout
        )
        assert proc.returncode == expect_exit, (
            f"exec {args} exited {proc.returncode} (wanted {expect_exit})\n"
            f"--- stdout ---\n{proc.stdout}\n--- stderr ---\n{proc.stderr}"
        )
        return proc

    def mbt(
        self, *args: str, expect_exit: int = 0, timeout: int = 1200
    ) -> subprocess.CompletedProcess[str]:
        return self.exec("mbt", *args, expect_exit=expect_exit, timeout=timeout)

    def seed_lake(self) -> None:
        self.exec("python", "/workspace/bootstrap/seed_lake.py", workdir="/workspace")

    def sync_lake(self) -> None:
        self.exec("python", "/workspace/bootstrap/sync_lake.py", workdir="/workspace")

    # -- assertions helpers ----------------------------------------------------
    def run_results(self) -> dict:
        path = self.workspace / "project" / "target" / "run_results.json"
        return json.loads(path.read_text())

    def result_for(self, unique_id: str) -> dict:
        matches = [r for r in self.run_results()["results"] if r.get("unique_id") == unique_id]
        assert matches, f"{unique_id} not in run_results: " + ", ".join(
            r.get("unique_id", "?") for r in self.run_results()["results"]
        )
        return matches[0]

    def mlflow_url(self) -> str:
        return f"http://localhost:{self.ports['SHOWCASE_MLFLOW_PORT']}"

    def s3_url(self) -> str:
        return f"http://localhost:{self.ports['SHOWCASE_S3_PORT']}"

    def filer_url(self) -> str:
        return f"http://localhost:{self.ports['SHOWCASE_FILER_PORT']}"

    def gitea_url(self) -> str:
        return f"http://localhost:{self.ports['SHOWCASE_GITEA_PORT']}"

    def woodpecker_url(self) -> str:
        return f"http://localhost:{self.ports['SHOWCASE_WOODPECKER_PORT']}"

    def webhook_url(self) -> str:
        return f"http://localhost:{self.ports['SHOWCASE_WEBHOOK_PORT']}"

    def zot_url(self) -> str:
        return f"http://localhost:{self.ports['SHOWCASE_ZOT_PORT']}"

    def airflow_url(self) -> str:
        return f"http://localhost:{self.ports['SHOWCASE_AIRFLOW_PORT']}"

    def http_json(self, url: str, headers: dict | None = None) -> dict:
        import urllib.request

        request = urllib.request.Request(url, headers=headers or {})
        with urllib.request.urlopen(request, timeout=30) as resp:
            return json.loads(resp.read().decode())


def ensure_daily_champion(stack: ComposeStack) -> None:
    """Standalone-safety: modules that score `tag:daily` need a production
    churn_automl champion. A full session inherits the lifecycle module's;
    a solo module run trains and promotes one here instead (dev target,
    spark snapshot scheme - no --deep-snapshot)."""
    from mlflow.tracking import MlflowClient

    client = MlflowClient(tracking_uri=stack.mlflow_url())
    try:
        client.get_model_version_by_alias("churn_automl", "production")
        return
    except Exception:
        pass
    stack.sync_lake()
    stack.mbt(
        "build", "--target", "dev", "--select", "churn_automl", "--anchor", ANCHOR, timeout=1800
    )
    stack.mbt("promote", "--model", "churn_automl", "--to", "production", timeout=300)


def build_runner_image() -> None:
    """Idempotent: no-op when the image already exists."""
    proc = subprocess.run(
        [str(SHOWCASE_DIR / "scripts" / "build_image.sh")],
        capture_output=True,
        text=True,
        timeout=2400,
        check=False,
    )
    if proc.returncode != 0:
        pytest.fail(f"runner image build failed:\n{proc.stdout[-8000:]}\n{proc.stderr[-8000:]}")


# -- CI tier harness (gitea + woodpecker + zot + webhook-sink + airflow) ------
#
# Session-scoped via the `showcase_ci` fixture in conftest.py: the CI,
# provenance, and scheduling modules all drive the same seeded forge.

ORG = "mbt-showcase"
REPO = "churn"
DEPLOY_REPO = "deploy"
GITEA_USER = "mbtops"
GITEA_PASSWORD = "mbtops-showcase-password"
DS_USER = "mbtds"
DS_PASSWORD = "mbtds-showcase-password"
PR_COMMENT_MARKER = "<!-- mbt-pr-comment -->"
PIPELINE_TERMINAL = {"success", "failure", "error", "killed", "declined", "canceled"}


def run_git(cwd: Path, *args: str, expect_ok: bool = True) -> subprocess.CompletedProcess:
    proc = subprocess.run(
        [
            "git",
            "-c",
            "user.name=showcase-test",
            "-c",
            "user.email=showcase-test@example.com",
            *args,
        ],
        cwd=cwd,
        capture_output=True,
        text=True,
        timeout=120,
        env={"GIT_TERMINAL_PROMPT": "0", "PATH": "/usr/bin:/bin:/usr/local/bin"},
        check=False,
    )
    if expect_ok:
        assert proc.returncode == 0, f"git {args} failed:\n{proc.stdout}\n{proc.stderr}"
    return proc


class CiHarness:
    """Drive the seeded forge the way a user would: git, PRs, and the Gitea,
    Woodpecker, zot, webhook-sink, and Airflow APIs."""

    def __init__(
        self,
        stack: ComposeStack,
        gitea_token: str,
        ds_token: str,
        woodpecker_token: str,
        repo_id: int,
        workdir: Path,
    ) -> None:
        self.stack = stack
        self.gitea_token = gitea_token
        self.ds_token = ds_token
        self.woodpecker_token = woodpecker_token
        self.repo_id = repo_id
        self.workdir = workdir
        self.seen_pipelines: set = set()
        self._af_token: str | None = None

    # -- gitea -----------------------------------------------------------------
    def gitea_api(
        self,
        method: str,
        path: str,
        payload: dict | None = None,
        token: str | None = None,
        ok_statuses: tuple = (),
    ):
        import requests

        resp = requests.request(
            method,
            f"{self.stack.gitea_url()}/api/v1{path}",
            json=payload,
            headers={"Authorization": f"token {token or self.gitea_token}"},
            timeout=30,
        )
        assert resp.ok or resp.status_code in ok_statuses, (
            f"{method} {path} -> {resp.status_code}: {resp.text[:2000]}"
        )
        return resp.json() if resp.ok and resp.content else None

    def clone_url(self, repo: str = REPO, user: str = GITEA_USER, token: str | None = None) -> str:
        port = self.stack.ports["SHOWCASE_GITEA_PORT"]
        return f"http://{user}:{token or self.gitea_token}@localhost:{port}/{ORG}/{repo}.git"

    def fresh_clone(
        self, name: str, repo: str = REPO, user: str = GITEA_USER, token: str | None = None
    ) -> Path:
        dest = self.workdir / f"{name}-{uuid.uuid4().hex[:6]}"
        run_git(self.workdir, "clone", self.clone_url(repo, user, token), str(dest))
        return dest

    def pr_comment(self, index: int) -> str:
        comments = self.gitea_api("GET", f"/repos/{ORG}/{REPO}/issues/{index}/comments")
        bodies = [c["body"] for c in comments if PR_COMMENT_MARKER in (c.get("body") or "")]
        assert len(bodies) == 1, f"expected exactly one marker comment, got {len(bodies)}"
        return bodies[0]

    def merge_pr(self, index: int, deadline_s: int = 60) -> None:
        # Gitea computes mergeability asynchronously; retry until it settles.
        import time

        import requests

        end = time.time() + deadline_s
        while True:
            resp = requests.post(
                f"{self.stack.gitea_url()}/api/v1/repos/{ORG}/{REPO}/pulls/{index}/merge",
                json={"Do": "merge"},
                headers={"Authorization": f"token {self.gitea_token}"},
                timeout=30,
            )
            if resp.ok:
                return
            if time.time() > end:
                pytest.fail(f"PR #{index} merge kept failing: {resp.status_code} {resp.text[:500]}")
            time.sleep(2)

    def raw_file(self, repo: str, path: str, ref: str) -> bytes:
        import requests

        resp = requests.get(
            f"{self.stack.gitea_url()}/api/v1/repos/{ORG}/{repo}/raw/{path}",
            params={"ref": ref},
            headers={"Authorization": f"token {self.gitea_token}"},
            timeout=30,
        )
        assert resp.ok, f"no {path} at {repo}@{ref}: {resp.status_code}"
        return resp.content

    def state_manifest(self, ref: str = "mbt-state") -> dict:
        return json.loads(self.raw_file(REPO, "manifest.json", ref))

    def state_commits(self) -> list:
        return self.gitea_api("GET", f"/repos/{ORG}/{REPO}/commits?sha=mbt-state&limit=50")

    def deploy_commits(self) -> list:
        return self.gitea_api("GET", f"/repos/{ORG}/{DEPLOY_REPO}/commits?sha=main&limit=50")

    def images_env(self, ref: str = "main") -> dict:
        conf = {}
        for line in self.raw_file(DEPLOY_REPO, "images.env", ref).decode().splitlines():
            if line and not line.startswith("#") and "=" in line:
                key, _, value = line.partition("=")
                conf[key] = value
        return conf

    # -- woodpecker --------------------------------------------------------------
    def oauth_login(self, user: str, password: str) -> str:
        """The human login path: complete the Gitea OAuth dance against the
        HOST-published ports (exactly what a browser does) and mint a
        Woodpecker API token for the given persona."""
        result = _run_ci_bootstrap(
            [
                "login",
                "--gitea-url",
                self.stack.gitea_url(),
                "--woodpecker-url",
                self.stack.woodpecker_url(),
                "--user",
                user,
                "--password",
                password,
            ]
        )
        return result["woodpecker_token"]

    def wp_api(self, path: str):
        import requests

        resp = requests.get(
            f"{self.stack.woodpecker_url()}/api{path}",
            headers={"Authorization": f"Bearer {self.woodpecker_token}"},
            timeout=30,
        )
        assert resp.ok, f"GET {path} -> {resp.status_code}: {resp.text[:2000]}"
        return resp.json()

    def wait_pipeline(self, event: str, timeout_s: int = 900) -> dict:
        """Wait for a NEW pipeline of the given event to reach a terminal state."""
        import time

        end = time.time() + timeout_s
        while time.time() < end:
            pipelines = self.wp_api(f"/repos/{self.repo_id}/pipelines?perPage=50") or []
            fresh = [
                p
                for p in pipelines
                if p["event"] == event and p["number"] not in self.seen_pipelines
            ]
            done = [p for p in fresh if p["status"] in PIPELINE_TERMINAL]
            if done:
                pipeline = sorted(done, key=lambda p: p["number"])[0]
                self.seen_pipelines.add(pipeline["number"])
                if pipeline["status"] != "success":
                    # Non-success is a legitimate verdict for some tests, but
                    # when it is NOT the expected one the step logs are the
                    # only evidence (the stack tears down before a human can
                    # look) - same rationale as _dump_failed_task_logs.
                    self._dump_pipeline_logs(pipeline["number"])
                return pipeline
            time.sleep(3)
        # Timeout: whatever pipeline is stuck (or never turned terminal) is
        # the only evidence there is - dump it before failing.
        stuck = [
            p
            for p in (self.wp_api(f"/repos/{self.repo_id}/pipelines?perPage=50") or [])
            if p["number"] not in self.seen_pipelines
        ]
        if stuck:
            self._dump_pipeline_logs(max(p["number"] for p in stuck))
        pytest.fail(f"no new terminal {event} pipeline within {timeout_s}s")

    def _dump_pipeline_logs(self, number: int) -> None:
        # Best-effort: a broken log fetch must not mask the caller's verdict.
        import base64

        try:
            detail = self.wp_api(f"/repos/{self.repo_id}/pipelines/{number}")
        except Exception as exc:
            print(f"(could not fetch pipeline {number} detail: {exc})")
            return
        for workflow in detail.get("workflows", []):
            for step in workflow.get("children", []):
                if step.get("state") == "success":
                    continue
                print(
                    f"--- pipeline {number} workflow {workflow.get('name')!r} "
                    f"step {step.get('name')!r} state={step.get('state')} ---"
                )
                try:
                    entries = self.wp_api(f"/repos/{self.repo_id}/logs/{number}/{step['id']}")
                    log = "".join(
                        base64.b64decode(entry.get("data") or "").decode(errors="replace")
                        for entry in entries or []
                    )
                    print(log[-4000:] if log else "(no log output)")
                except Exception as exc:
                    print(f"(could not fetch step logs: {exc})")

    def ensure_seeded(self, timeout_s: int = 1200) -> bool:
        """First-ever push to main: full bootstrap build (fetch_state exit 3),
        baseline publish, first bake. Idempotent across modules."""
        if self.wp_api(f"/repos/{self.repo_id}/pipelines?perPage=1"):
            return False
        clone = self.fresh_clone("seed")
        readme = clone / "README.md"
        readme.write_text("# churn (showcase CI seed)\n")
        run_git(clone, "add", "README.md")
        run_git(clone, "commit", "-m", "seed: trigger first prod build")
        run_git(clone, "push", "origin", "main")
        pipeline = self.wait_pipeline("push", timeout_s=timeout_s)
        assert pipeline["status"] == "success", pipeline
        return True

    # -- webhook sink --------------------------------------------------------------
    def alerts(self) -> list:
        return self.stack.http_json(f"{self.stack.webhook_url()}/requests")

    def reset_alerts(self) -> None:
        import requests

        requests.delete(f"{self.stack.webhook_url()}/requests", timeout=10).raise_for_status()

    # -- zot -------------------------------------------------------------------------
    def provenance_files(self, tag: str) -> dict:
        """Fetch the oras artifact's files {name: bytes} via the OCI API."""
        import requests

        base = f"{self.stack.zot_url()}/v2/mbt/churn/provenance"
        manifest = requests.get(
            f"{base}/manifests/{tag}",
            headers={"Accept": "application/vnd.oci.image.manifest.v1+json"},
            timeout=30,
        )
        assert manifest.ok, f"no provenance manifest for {tag}: {manifest.status_code}"
        files = {}
        for layer in manifest.json().get("layers", []):
            name = (layer.get("annotations") or {}).get("org.opencontainers.image.title")
            blob = requests.get(f"{base}/blobs/{layer['digest']}", timeout=30)
            assert blob.ok, f"blob {layer['digest']} for {tag}: {blob.status_code}"
            files[name] = blob.content
        return files

    # -- airflow ----------------------------------------------------------------------
    def af_token(self, *, force: bool = False) -> str:
        """Mint (and cache) a JWT for the v2 API; Airflow 3 dropped basic auth."""
        import requests

        if force or self._af_token is None:
            resp = requests.post(
                f"{self.stack.airflow_url()}/auth/token",
                json={"username": "admin", "password": "admin"},
                timeout=60,
            )
            assert resp.ok, f"airflow token mint -> {resp.status_code}: {resp.text[:2000]}"
            self._af_token = resp.json()["access_token"]
        return self._af_token

    def af_api(self, method: str, path: str, payload: dict | None = None):
        import requests

        for attempt in (1, 2):  # one retry with a fresh token on expiry
            resp = requests.request(
                method,
                f"{self.stack.airflow_url()}/api/v2{path}",
                json=payload,
                headers={"Authorization": f"Bearer {self.af_token(force=attempt > 1)}"},
                timeout=60,
            )
            if resp.status_code != 401:
                break
        assert resp.ok, f"airflow {method} {path} -> {resp.status_code}: {resp.text[:2000]}"
        return resp.json() if resp.content else None

    def wait_dag(self, dag_id: str, timeout_s: int = 300) -> None:
        """Wait until git-sync + the DAG processor have registered the DAG."""
        import time

        import requests

        end = time.time() + timeout_s
        while time.time() < end:
            resp = requests.get(
                f"{self.stack.airflow_url()}/api/v2/dags/{dag_id}",
                headers={"Authorization": f"Bearer {self.af_token()}"},
                timeout=30,
            )
            if resp.status_code == 401:
                self.af_token(force=True)
            elif resp.ok and not resp.json().get("is_paused", True):
                return
            time.sleep(5)
        pytest.fail(f"DAG {dag_id} never appeared unpaused within {timeout_s}s")

    def trigger_dag(self, dag_id: str, conf: dict | None = None) -> str:
        # logical_date is a required (nullable) field in the v2 trigger body.
        run = self.af_api(
            "POST", f"/dags/{dag_id}/dagRuns", {"logical_date": None, "conf": conf or {}}
        )
        return run["dag_run_id"]

    def wait_dag_run(self, dag_id: str, run_id: str, timeout_s: int = 1800) -> str:
        import time
        import urllib.parse

        encoded = urllib.parse.quote(run_id, safe="")
        end = time.time() + timeout_s
        while time.time() < end:
            state = self.af_api("GET", f"/dags/{dag_id}/dagRuns/{encoded}")["state"]
            if state == "failed":
                # Failed is a legitimate verdict for some tests, but when it
                # is NOT the expected one the task log is the only evidence
                # (the stack is torn down before a human can look) - dump it.
                self._dump_failed_task_logs(dag_id, encoded)
                return state
            if state == "success":
                return state
            time.sleep(5)
        pytest.fail(f"DAG run {dag_id}/{run_id} still not terminal after {timeout_s}s")

    def _dump_failed_task_logs(self, dag_id: str, encoded_run_id: str) -> None:
        # Best-effort: a broken log fetch must not mask the caller's verdict.
        try:
            tis = self.af_api("GET", f"/dags/{dag_id}/dagRuns/{encoded_run_id}/taskInstances")
            for ti in tis["task_instances"]:
                if ti["state"] != "failed":
                    continue
                for attempt in range(1, (ti["try_number"] or 1) + 1):
                    # v2 serves structured log events, not text/plain: each
                    # content item is a StructuredLogMessage (or a string).
                    payload = self.af_api(
                        "GET",
                        f"/dags/{dag_id}/dagRuns/{encoded_run_id}/taskInstances"
                        f"/{ti['task_id']}/logs/{attempt}?full_content=true",
                    )
                    log = "\n".join(
                        event.get("event", str(event)) if isinstance(event, dict) else str(event)
                        for event in payload["content"]
                    )
                    print(
                        f"--- {dag_id}.{ti['task_id']} try {attempt}/{ti['try_number']} "
                        f"(failed) ---\n{log[-4000:]}"
                    )
        except Exception as exc:
            print(f"(could not fetch failed-task logs for {dag_id}: {exc})")

    def task_instances(self, dag_id: str, run_id: str) -> dict:
        import urllib.parse

        encoded = urllib.parse.quote(run_id, safe="")
        payload = self.af_api("GET", f"/dags/{dag_id}/dagRuns/{encoded}/taskInstances")
        return {ti["task_id"]: ti for ti in payload["task_instances"]}


def _run_ci_bootstrap(script_args: list[str]) -> dict:
    import sys

    proc = subprocess.run(
        [sys.executable, str(SHOWCASE_DIR / "scripts" / "ci_bootstrap.py"), *script_args],
        capture_output=True,
        text=True,
        timeout=600,
        cwd=REPO_ROOT,
        check=False,
    )
    assert proc.returncode == 0, (
        f"ci_bootstrap {script_args[0]} failed:\n{proc.stdout}\n{proc.stderr}"
    )
    return json.loads(proc.stdout.strip().splitlines()[-1])


def bootstrap_ci(stack: ComposeStack, workdir: Path) -> CiHarness:
    """Seed gitea, re-up woodpecker with real OAuth creds, mint its token."""
    pre = _run_ci_bootstrap(
        [
            "pre",
            "--gitea-url",
            stack.gitea_url(),
            "--gitea-container",
            f"{stack.project_name}-gitea-1",
            "--woodpecker-url",
            stack.woodpecker_url(),
            "--project-dir",
            str(stack.workspace / "project"),
            "--network",
            stack.env["SHOWCASE_NETWORK"],
            "--workspace",
            str(stack.workspace),
        ]
    )

    # Recreate woodpecker with the real OAuth app credentials (it booted on
    # placeholders). The agent recreates too: its stored agent ID references
    # the old server's database.
    stack.env["SHOWCASE_WOODPECKER_CLIENT"] = pre["client_id"]
    stack.env["SHOWCASE_WOODPECKER_SECRET"] = pre["client_secret"]
    up = stack.compose(
        "up",
        "-d",
        "--wait",
        "--force-recreate",
        "woodpecker-server",
        "woodpecker-agent",
        timeout=300,
    )
    assert up.returncode == 0, f"woodpecker re-up failed:\n{up.stdout}\n{up.stderr}"

    post = _run_ci_bootstrap(
        [
            "post",
            "--gitea-url",
            stack.gitea_url(),
            "--woodpecker-url",
            stack.woodpecker_url(),
            "--gitea-token",
            pre["gitea_token"],
            "--zot-ref",
            f"localhost:{stack.ports['SHOWCASE_ZOT_PORT']}/mbt/churn",
        ]
    )
    return CiHarness(
        stack=stack,
        gitea_token=pre["gitea_token"],
        ds_token=pre["ds_token"],
        woodpecker_token=post["woodpecker_token"],
        repo_id=post["repo_id"],
        workdir=workdir,
    )
