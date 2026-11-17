"""Shared wiring for the showcase DAGs: the pinned deployable unit, drawn by mbt-airflow.

Each DAG is one mbt command drawn as the steps of each node (ADR-33, ADR-35), and every task runs
mbt inside the deployable unit (ADR-34). Exit-code fidelity at the scheduler
(DESIGN.md section 5 step 7) is the package's:

- exit 2 is a QUALITY verdict (gate/check/monitor said no). Deterministic -
  retrying cannot change it - so the task fails immediately without consuming
  retries; the model owner is notified, never on-call.
- any other nonzero exit is a hard error (infra, code): the task is retried before
  anyone is paged.
- metrics push best-effort after every task, from inside the task's container
  (scripts/mbt_task.sh; push_metrics.py warns and exits 0 when the gateway is away).

The unit's digest and the session wiring come from ../images.env, and the graph is
drawn from ../manifest.json: prod-build writes BOTH in the one commit that pins a new
unit, so the DAG a person sees is always the DAG of the code that will run. Bumping
IMAGE (or `git revert`) IS the deploy.

Until prod-build has baked the first unit there is no manifest, so there are no DAGs;
they appear once git-sync hands Airflow that first commit.

Anchors come from the data: a DAG's `anchor` param left empty resolves, inside the
unit at run time, to the lake's own anchor (scripts/lake_anchor.py - the day after
its newest weekly cohort, or ten days later to monitor), so a run always targets the
newest cohort. Set the param to pin one.
"""

from datetime import timedelta
from pathlib import Path

from airflow.sdk.exceptions import AirflowException
from mbt_airflow import DockerUnit, MbtDocker, MbtExecution, MbtProject

#: The deploy repo's worktree. The paths are resolved, so they name the commit
#: git-sync handed this process, not the symlink it flips.
DEPLOY_DIR = Path(__file__).resolve().parent.parent
CONF_PATH = DEPLOY_DIR / "images.env"
MANIFEST_PATH = DEPLOY_DIR / "manifest.json"

#: Inside the unit: where the project is, and the volume every task shares.
PROJECT_DIR = "/app/project"
WORKSPACE = "/workspace"

#: What prints the lake's anchor (scripts/lake_anchor.py), and what a monitor adds.
LAKE_ANCHOR = ["python3", "scripts/lake_anchor.py"]
LAKE_ANCHOR_MONITOR = [*LAKE_ANCHOR, "--monitor"]


def load_conf() -> dict[str, str]:
    conf = {}
    for raw in CONF_PATH.read_text().splitlines():
        line = raw.strip()
        if line and not line.startswith("#") and "=" in line:
            key, _, value = line.partition("=")
            conf[key] = value
    return conf


def unit() -> DockerUnit:
    """The pinned deployable unit, read when a task runs, not when the file parses."""
    conf = load_conf()
    if not conf.get("IMAGE"):
        # CONF_PATH is resolved, so it names git-sync's worktree: the deploy
        # commit this task read, to compare with the deploy repo's main.
        raise AirflowException(
            f"images.env has no IMAGE pin yet ({CONF_PATH}) - run a prod build first; "
            "if one just pinned a unit, git-sync has not synced it here yet"
        )
    return DockerUnit(
        image=conf["IMAGE"],
        network=conf["NETWORK"],
        volumes={conf["WORKSPACE"]: WORKSPACE},
        working_dir=PROJECT_DIR,
        env={
            "AWS_ACCESS_KEY_ID": "mbtadmin",
            "AWS_SECRET_ACCESS_KEY": "mbtsecret",
            "AWS_DEFAULT_REGION": "us-east-1",
            "AWS_ENDPOINT_URL_S3": "http://seaweedfs:8333",
            "MBT_PUSHGATEWAY": "http://pushgateway:9091",
            "TMPDIR": f"{WORKSPACE}/tmp",
        },
    )


EXECUTION = MbtExecution(
    docker=MbtDocker(unit),
    # Pushes the task's metrics after mbt exits, then exits with mbt's own code.
    mbt_executable="scripts/mbt_task.sh",
    workspace_root=f"{WORKSPACE}/mbt-airflow",
    retries=1,
    retry_delay=timedelta(seconds=5),
)


def project(target: str) -> MbtProject:
    """The churn project, drawn from the manifest prod-build published."""
    return MbtProject(project_dir=PROJECT_DIR, manifest_path=MANIFEST_PATH, target=target)


def baked() -> bool:
    """Whether prod-build has published a unit (and with it, a manifest) yet."""
    return MANIFEST_PATH.is_file()
