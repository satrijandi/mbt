"""Drive the scheduled half of the showcase lifecycle (`make lifecycle`).

`make demo` runs mbt by hand inside JupyterLab; this is the same lifecycle
the way production runs it: Woodpecker bakes the deployable unit and pins
its digest in the deploy repo, git-sync hands the deploy repo's DAGs to
Airflow, and every retrain, score and monitor is an Airflow DAG run that
executes mbt inside that pinned unit (showcase_dag_utils.run_in_unit).

Runs on the HOST against the published ports, like ci_bootstrap.py (the dev
venv has `requests` via mlflow). Two phases, which the Makefile narrates:

  unit - make sure a deployable unit is pinned. If the deploy repo's
         images.env has no digest yet, commit to the churn repo's main (a
         push event, exactly what a merge is), wait for Woodpecker's
         prod-build to bake the unit, then wait for Airflow to register
         and unpause every DAG. Idempotent: an existing pin is kept.
  run  - trigger one DAG with an optional conf, wait for a terminal state,
         print the task log (mbt's own output from inside the unit) and the
         run's Airflow URL. Exits 0 on `success`, 1 otherwise.

Usage:
  lifecycle.py unit --gitea-url URL --woodpecker-url URL --airflow-url URL
  lifecycle.py run  --airflow-url URL DAG_ID [--conf JSON]
"""

import argparse
import base64
import json
import sys
import time
import urllib.parse
from collections.abc import Iterable
from datetime import UTC, datetime
from pathlib import Path
from typing import Any

import requests

sys.path.insert(0, str(Path(__file__).resolve().parent))
from ci_bootstrap import DEPLOY_REPO, ORG, PASSWORD, REPO, USER, gitea_api, oauth_login

DAGS = ("mbt_retrain", "mbt_score", "mbt_monitor")
PIPELINE_TERMINAL = {"success", "failure", "error", "killed", "declined", "canceled"}
#: The marker file the unit phase commits to trigger the first prod-build.
MARKER = "LIFECYCLE.md"


def say(message: str) -> None:
    print(f"    {message}", flush=True)


def pinned_image(gitea_url: str) -> str:
    """The deploy repo's pinned unit (``repo@sha256:...``), or '' if none."""
    resp = requests.get(f"{gitea_url}/{ORG}/{DEPLOY_REPO}/raw/branch/main/images.env", timeout=30)
    if not resp.ok:
        raise SystemExit(
            f"no {ORG}/{DEPLOY_REPO} repo on Gitea ({resp.status_code}) - run `make ci` first"
        )
    for line in resp.text.splitlines():
        if line.startswith("IMAGE=") and "@sha256:" in line:
            return line.partition("=")[2].strip()
    return ""


def event_text(event: object) -> str:
    """One task-log event as text, with the exception chain if it has one.

    Airflow 3 serves each log line as a structured event (or a plain string).
    A failure's message is just ``Task failed with exception``; the exception
    itself rides in ``error_detail``, which printing ``event`` alone drops -
    so a failed run printed no cause at all.
    """
    if not isinstance(event, dict):
        return str(event)
    text = str(event.get("event", event))
    for exc in event.get("error_detail") or []:
        text += f"\n  {exc.get('exc_type')}: {exc.get('exc_value')}"
    return text


def mbt_output(lines: Iterable[str]) -> list[str]:
    """The task log minus Airflow's framing and Spark's console progress bars.

    Airflow wraps the task in ``::group::`` sections (log source, DAG
    parsing) and ends it with ``Done. Returned value was: None``; Spark draws
    ``[Stage N:>`` bars with carriage returns, which a captured log keeps as
    lines of their own. What is left is the unit container's output: mbt's.
    """
    kept: list[str] = []
    in_group = False
    for raw in lines:
        for line in raw.replace("\r", "\n").split("\n"):
            if line.startswith("::group::"):
                in_group = True
            elif line.startswith("::endgroup::"):
                in_group = False
            elif not (
                in_group
                or not line.strip()
                or line.lstrip().startswith("[Stage ")
                or line.startswith("Done. Returned value was:")
            ):
                kept.append(line)
    return kept


def list_pipelines(woodpecker_url: str, repo_id: int, headers: dict) -> list[dict] | None:
    """The repo's Woodpecker pipelines, or None when this poll got no answer.

    Woodpecker's SQLite store locks under the stack's load while a pipeline
    starts ("database is locked" in its log), and the API then answers with a
    non-JSON error body. That is a transient for a poll, not a failed bake:
    calling ``.json()`` on it unguarded crashed `make lifecycle` mid-build.
    """
    try:
        resp = requests.get(
            f"{woodpecker_url}/api/repos/{repo_id}/pipelines", headers=headers, timeout=30
        )
        if resp.ok:
            return resp.json() or []
        reason = f"HTTP {resp.status_code}: {resp.text.strip()[:200]}"
    except (requests.ConnectionError, requests.Timeout, ValueError) as exc:
        # ValueError covers requests' JSONDecodeError on a 2xx non-JSON body.
        reason = f"{type(exc).__name__}: {exc}"
    say(f"Woodpecker did not list pipelines ({reason}); polling again")
    return None


class Airflow:
    """The v2 REST API with a JWT (Airflow 3 dropped basic auth)."""

    def __init__(self, url: str) -> None:
        self.url = url.rstrip("/")
        self._token: str | None = None

    def token(self, *, force: bool = False) -> str:
        if force or self._token is None:
            resp = requests.post(
                f"{self.url}/auth/token",
                json={"username": "admin", "password": "admin"},
                timeout=60,
            )
            if not resp.ok:
                raise SystemExit(f"airflow token mint -> {resp.status_code}: {resp.text[:500]}")
            self._token = resp.json()["access_token"]
        return self._token

    def api(
        self, method: str, path: str, payload: dict | None = None, ok_statuses: tuple = ()
    ) -> Any:
        for attempt in (1, 2):  # one retry with a fresh token on expiry
            resp = requests.request(
                method,
                f"{self.url}/api/v2{path}",
                json=payload,
                headers={"Authorization": f"Bearer {self.token(force=attempt > 1)}"},
                timeout=60,
            )
            if resp.status_code != 401:
                break
        if not resp.ok and resp.status_code not in ok_statuses:
            raise SystemExit(f"airflow {method} {path} -> {resp.status_code}: {resp.text[:500]}")
        return resp.json() if resp.ok and resp.content else None

    def wait_dags(self, timeout_s: int = 300) -> None:
        """git-sync + the DAG processor register the deploy repo's DAGs."""
        end = time.time() + timeout_s
        pending = set(DAGS)
        while pending and time.time() < end:
            for dag_id in sorted(pending):
                dag = self.api("GET", f"/dags/{dag_id}", ok_statuses=(404,))
                if dag and not dag.get("is_paused", True):
                    pending.discard(dag_id)
            if pending:
                time.sleep(5)
        if pending:
            raise SystemExit(f"Airflow never registered {sorted(pending)} within {timeout_s}s")

    def task_log(self, dag_id: str, run_id: str) -> str:
        encoded = urllib.parse.quote(run_id, safe="")
        tis = self.api("GET", f"/dags/{dag_id}/dagRuns/{encoded}/taskInstances")
        chunks = []
        for ti in tis["task_instances"]:
            base = f"/dags/{dag_id}/dagRuns/{encoded}/taskInstances/{ti['task_id']}"
            # Each attempt's own state: the task instance's is the latest
            # try's, which labelled a failed-then-retried try 1 "success".
            tries = self.api("GET", f"{base}/tries")["task_instances"]
            for attempt in sorted(tries, key=lambda t: t["try_number"]):
                number = attempt["try_number"]
                # v2 serves structured log events: each content item is a
                # StructuredLogMessage dict (or a plain string).
                payload = self.api("GET", f"{base}/logs/{number}?full_content=true")
                lines = mbt_output(event_text(event) for event in payload["content"])
                chunks.append(
                    f"--- {dag_id}.{ti['task_id']} try {number} ({attempt['state']}) ---\n"
                    + "\n".join(lines)
                )
        return "\n".join(chunks)


def phase_unit(args: argparse.Namespace) -> int:
    image = pinned_image(args.gitea_url)
    if image:
        say(f"deployable unit already pinned: {image}")
    else:
        say("no deployable unit pinned yet: pushing to main so prod-build bakes the first one")
        token = oauth_login(args.gitea_url, args.woodpecker_url, USER, PASSWORD)
        wp = {"Authorization": f"Bearer {token}"}
        lookup = requests.get(
            f"{args.woodpecker_url}/api/repos/lookup/{ORG}/{REPO}", headers=wp, timeout=30
        )
        if not lookup.ok:
            raise SystemExit(f"{ORG}/{REPO} is not active on Woodpecker - run `make ci` first")
        repo_id = lookup.json()["id"]
        end = time.time() + args.timeout
        listed = list_pipelines(args.woodpecker_url, repo_id, wp)
        while listed is None:
            if time.time() >= end:
                raise SystemExit("Woodpecker never listed the churn repo's pipelines")
            time.sleep(10)
            listed = list_pipelines(args.woodpecker_url, repo_id, wp)
        seen = {p["number"] for p in listed}

        # A commit on main IS what a merge produces; the contents API makes
        # one without a working copy.
        path = f"/repos/{ORG}/{REPO}/contents/{MARKER}"
        existing = gitea_api(
            args.gitea_url, "GET", f"{path}?ref=main", auth=(USER, PASSWORD), ok_statuses=(404,)
        )
        stamp = datetime.now(UTC).strftime("%Y-%m-%dT%H:%M:%SZ")
        body = f"# lifecycle\n\n`make lifecycle` pushed this at {stamp} to bake the first unit.\n"
        gitea_api(
            args.gitea_url,
            "PUT" if existing else "POST",
            path,
            auth=(USER, PASSWORD),
            payload={
                "branch": "main",
                "content": base64.b64encode(body.encode()).decode(),
                "message": "lifecycle: bake the first deployable unit",
                **({"sha": existing["sha"]} if existing else {}),
            },
        )

        pipeline = None
        end = time.time() + args.timeout
        while time.time() < end:
            listed = list_pipelines(args.woodpecker_url, repo_id, wp) or []
            fresh = [p for p in listed if p["number"] not in seen and p["event"] == "push"]
            if fresh:
                newest = max(fresh, key=lambda p: p["number"])
                if pipeline is None:
                    say(
                        f"prod-build pipeline #{newest['number']}: "
                        f"{args.woodpecker_url}/repos/{repo_id}/pipeline/{newest['number']}"
                    )
                pipeline = newest
                if pipeline["status"] in PIPELINE_TERMINAL:
                    break
            time.sleep(10)
        if pipeline is None or pipeline["status"] != "success":
            status = pipeline["status"] if pipeline else "never started"
            raise SystemExit(f"prod-build did not succeed ({status}); see the Woodpecker URL above")
        image = pinned_image(args.gitea_url)
        if not image:
            raise SystemExit("prod-build succeeded but the deploy repo still has no IMAGE pin")
        say(f"deployable unit baked and pinned: {image}")

    af = Airflow(args.airflow_url)
    af.wait_dags()
    say(f"Airflow has {', '.join(DAGS)} registered and unpaused")
    return 0


def phase_run(args: argparse.Namespace) -> int:
    af = Airflow(args.airflow_url)
    af.wait_dags()
    conf = json.loads(args.conf) if args.conf else {}
    # logical_date is a required (nullable) field in the v2 trigger body.
    run = af.api("POST", f"/dags/{args.dag_id}/dagRuns", {"logical_date": None, "conf": conf})
    run_id = run["dag_run_id"]
    url = f"{af.url}/dags/{args.dag_id}/runs/{urllib.parse.quote(run_id, safe='')}"
    say(f"triggered {args.dag_id} run {run_id}" + (f" with conf {conf}" if conf else ""))
    say(url)

    started = time.time()
    state = "queued"
    last = None
    while time.time() - started < args.timeout:
        encoded = urllib.parse.quote(run_id, safe="")
        state = af.api("GET", f"/dags/{args.dag_id}/dagRuns/{encoded}")["state"]
        if state != last:
            say(f"{args.dag_id}: {state} ({int(time.time() - started)}s)")
            last = state
        if state in ("success", "failed"):
            break
        time.sleep(5)

    log = af.task_log(args.dag_id, run_id)
    lines = log.splitlines()
    if len(lines) > args.log_lines:
        print(f"    ... ({len(lines) - args.log_lines} earlier log lines; full log at {url})")
    print("\n".join(lines[-args.log_lines :]), flush=True)
    if state != "success":
        print(f"{args.dag_id} run {run_id} ended {state}: {url}", file=sys.stderr)
        return 1
    say(f"{args.dag_id} succeeded in {int(time.time() - started)}s")
    return 0


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    sub = parser.add_subparsers(dest="phase", required=True)
    unit = sub.add_parser("unit", help="make sure a deployable unit is pinned and DAGs are live")
    unit.add_argument("--gitea-url", required=True)
    unit.add_argument("--woodpecker-url", required=True)
    unit.add_argument("--airflow-url", required=True)
    unit.add_argument("--timeout", type=int, default=1800, help="seconds to wait for prod-build")
    run = sub.add_parser("run", help="trigger one DAG run and wait for it")
    run.add_argument("dag_id", choices=DAGS)
    run.add_argument("--airflow-url", required=True)
    run.add_argument("--conf", help="the run's conf, as JSON")
    run.add_argument("--timeout", type=int, default=3600, help="seconds to wait for the run")
    run.add_argument("--log-lines", type=int, default=60, help="task log lines to print")
    args = parser.parse_args()
    return phase_unit(args) if args.phase == "unit" else phase_run(args)


if __name__ == "__main__":
    sys.exit(main())
