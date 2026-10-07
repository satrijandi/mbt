"""Shared plumbing for the showcase DAGs: run mbt inside the pinned
deployable unit, with the tutorial's exit-code routing rule.

Exit-code fidelity at the scheduler (DESIGN.md section 5 step 7):

- exit 2 is a QUALITY verdict (gate/check/monitor said no). Deterministic -
  retrying cannot change it - so the task raises AirflowFailException,
  which fails immediately without consuming retries; the model owner is
  notified, never on-call.
- any other nonzero exit is a hard error (infra, code): raise a retryable
  AirflowException so Airflow's retry policy runs before paging on-call.
- metrics push best-effort regardless of outcome, from inside the task
  container (push_metrics.py warns and exits 0 when the gateway is away).

The unit's digest and the session wiring come from ../images.env, committed
in this repo: bumping IMAGE (or `git revert`) IS the deploy.

Anchors come from the data: a DAG's `anchor` param left empty resolves, inside
the unit at run time, to the lake's own anchor (scripts/lake_anchor.py - the
day after its newest weekly cohort, or ten days later to monitor), so a run
always targets the newest cohort. Set the param to pin one.
"""

import shlex
from pathlib import Path

from airflow.sdk.exceptions import AirflowException, AirflowFailException

CONF_PATH = Path(__file__).resolve().parent.parent / "images.env"


def load_conf() -> dict:
    conf = {}
    for raw in CONF_PATH.read_text().splitlines():
        line = raw.strip()
        if line and not line.startswith("#") and "=" in line:
            key, _, value = line.partition("=")
            conf[key] = value
    return conf


def anchor_arg(anchor: str, *, monitor: bool = False) -> str:
    """Shell text for ``--anchor``: the pinned param, or the lake's own."""
    if anchor:
        return f"--anchor {shlex.quote(anchor)}"
    flag = " --monitor" if monitor else ""
    return f'--anchor "$(python3 scripts/lake_anchor.py{flag})"'


def run_in_unit(mbt_args: list[str], *, anchor: str, alert_class_2_owner: bool = True) -> None:
    """Run one mbt command in the pinned unit; classify the exit code.

    ``anchor`` is shell text from ``anchor_arg``, appended unquoted so the
    lake's anchor resolves inside the unit."""
    import docker

    conf = load_conf()
    if not conf.get("IMAGE"):
        raise AirflowException("images.env has no IMAGE pin yet - run a prod build first")

    command = " ".join(shlex.quote(a) for a in mbt_args) + " " + anchor
    wrapped = (
        f"{command}; rc=$?; "
        "MBT_PUSHGATEWAY=http://pushgateway:9091 python3 scripts/push_metrics.py . || true; "
        "exit $rc"
    )
    # docker-py's default 60s API timeout is shorter than a container create
    # can take on a busy daemon (Spark + H2O + the rest of the stack in one
    # Docker Desktop/OrbStack VM): the create timed out client-side and the
    # task failed before mbt ran. wait() below sets its own, longer timeout.
    client = docker.from_env(timeout=600)
    container = client.containers.run(
        conf["IMAGE"],
        ["bash", "-c", wrapped],
        network=conf["NETWORK"],
        working_dir="/app/project",
        volumes={conf["WORKSPACE"]: {"bind": "/workspace", "mode": "rw"}},
        environment={
            "AWS_ACCESS_KEY_ID": "mbtadmin",
            "AWS_SECRET_ACCESS_KEY": "mbtsecret",
            "AWS_DEFAULT_REGION": "us-east-1",
            "AWS_ENDPOINT_URL_S3": "http://seaweedfs:8333",
            "TMPDIR": "/workspace/tmp",
        },
        detach=True,
    )
    try:
        status = container.wait(timeout=3000)
        print(container.logs().decode(errors="replace"))
    finally:
        container.remove(force=True)

    code = status.get("StatusCode", 1)
    if code == 0:
        return
    if code == 2 and alert_class_2_owner:
        raise AirflowFailException(
            "mbt exited 2: quality verdict (gate/check/monitor). Deterministic - "
            "not retried; notify the model owner, not on-call."
        )
    raise AirflowException(f"mbt exited {code}: hard error - Airflow retries, then on-call.")
