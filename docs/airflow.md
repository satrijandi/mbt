# Running on Airflow

`mbt-airflow` draws an mbt project as an Airflow DAG, the way `dbt-cosmos` does for dbt: one task per node, with the edges taken from the manifest.
A weekly retrain stops being one opaque box.
Each dataset and each model is its own task, with its own state, its own retry, and its own place in the graph.

```bash
pip install mbt-airflow   # on the scheduler and on every worker; mbt-core comes with it
```

Airflow 3.3.1 or later is required.
Every worker also needs the `mbt` executable and whatever your models import (XGBoost, MLflow, and so on).
Airflow itself needs only the package, because the tasks run the real CLI in a subprocess ([ADR-33](adr/0033-mbt-project-as-an-airflow-dag.md)), or in a container ([below](#running-each-task-in-a-container)).

## The smallest DAG

```python
from datetime import datetime

from mbt_airflow import MbtDag, MbtProject

project = MbtProject(
    project_dir="/opt/churn_models",
    manifest_path="/opt/churn_models/manifest.json",
)

churn_weekly = MbtDag(
    "churn_weekly",
    project=project,
    select=["tag:weekly"],
    schedule="0 6 * * 1",
    start_date=datetime(2026, 1, 1),
)

churn_score = MbtDag(
    "churn_score",
    project=project,
    command="score",
    select=["tag:daily"],
    schedule="@daily",
    start_date=datetime(2026, 1, 1),
)
```

Put that in your DAGs folder.
Any `DAG` argument (`schedule`, `start_date`, `default_args`, `tags`, ...) goes straight through.

`select` and `exclude` use the CLI's [selector grammar](cli-reference.md#selectors).
`state:` selectors are the one exception: a DAG is drawn before any run, so there is no reference state to compare against.

## Publish the manifest the DAG is drawn from

The graph is drawn when Airflow parses the file, from a manifest compiled ahead of time.
Publish one next to the project whenever its nodes change, for example from CI:

```bash
mbt compile --anchor "$(date -u +%Y-%m-%dT%H:%M:%SZ)" --target-path build/manifest
cp build/manifest/manifest.json /opt/churn_models/manifest.json
```

Only its shape is used: which nodes exist and what depends on what.
Every run compiles again at its own anchor, so the data a run sees is never the data the manifest was compiled against.

If the project gains or loses a node the selector matches and the manifest is not republished, the run's compile task fails and names the difference, rather than quietly running a different graph than the one you see.

## What gets drawn

One DAG is one command's.

| `command` | Draws | Each task runs |
|---|---|---|
| `build` (the default), `run` | datasets and models, each as a group of steps ([below](#a-node-is-a-group-of-steps)) | `mbt <command> --select <node> --phase <step> --no-auto-materialize` |
| `test` | datasets and models | `mbt test --select <node> --no-auto-materialize` |
| `score`, `monitor` | scoring nodes | `mbt <command> --select <node>` |

They are separate DAGs because mbt's commands partition the nodes that way.
A scoring node depends on its model in the manifest, but it reads the registry's champion at run time, on its own cadence, so a daily score must not wait for a weekly retrain.
The score DAG therefore has no edge to a training task.

A model selection is drawn with the datasets it needs, because a model task reads its dataset instead of building it.
For `churn_demo`, `select=["tag:weekly"]` selects two models, and with `phases=False` (one task per node) draws five tasks:

```text
mbt_compile --+--> churn_training_set --> churn_classifier
              |
              +--> upsell_training_set --> upsell_classifier
```

### A node is a group of steps

By default a node of `build` or `run` is not one task but one per step mbt can run on its own ([ADR-35](adr/0035-phases-of-a-node.md)):

| Node | Steps, in order |
|---|---|
| dataset | `materialize`, `checks` |
| model | `train`, `gate`, `register` |

so the same selection is eleven tasks, each node a collapsible group in Airflow's graph view:

```text
mbt_compile --+--> churn_training_set [ materialize --> checks ] --> churn_classifier [ train --> gate --> register ]
              |
              +--> upsell_training_set [ materialize --> checks ] --> upsell_classifier [ train --> gate --> register ]
```

A node waits for the last step of each node it reads, so a model trains only once its dataset's checks have passed.
The steps have different failure stories, which is the point of drawing them:

- **`register`** is a registry write. If the registry blips after a two-hour training, the retry registers the trained artifact; it does not train again.
- **`gate`** carries the quality verdict. A red `gate` is exit 2: it is not retried and goes to the model's owner, while a red `train` or `register` is infrastructure.
- **`checks`** can be re-run without rebuilding the dataset, and `train` is the only step that needs the training pool, queue or GPU. `MbtExecution.phase_args` gives a step operator arguments of its own: `phase_args={"train": {"pool": "gpu"}, "register": {"retries": 5}}`.

The steps of one node share a target directory and hand their work over through a state file there, so every step of a node must see the same workspace ([ADR-35](adr/0035-phases-of-a-node.md)).
`phases=False` draws one task per node.
Preprocessing is part of `train`: it runs inside the training job, through the model's hooks.

Promotion is not a task.
It stays a reviewed step between the build DAG and the score DAG ([GitOps & CI](gitops.md)).

## What each task does

The first task, `mbt_compile`, compiles the project once at the run's anchor.
Every node task then executes that manifest verbatim (`--manifest`), so snapshots, windows, and seeds agree across the whole run.

Each node gets its own target directory, shared by its steps, and one dataset cache is shared with the rest of the run ([ADR-32](adr/0032-target-path-and-exact-node-execution.md)).
A task's result is read from its own `run_results.json` and returned as its XCom: the node's status, its metrics, and any message.
After `train` that is the metrics and the artifact; after `register` it is the finished node, registration included.

The anchor is the run's `anchor` param when you set one, else what your `anchor_command` prints, else the end of its data interval, else its logical date.
The `vars` param is passed to the compile task only, which is what keeps the dataset and the model tasks agreeing on cache keys.

`anchor_command` is for a project whose clock is its data, such as a lake that grows a cohort a week and should be scored as of its newest one.
It is a command run where mbt runs (in the container, with `MbtDocker`) that prints exactly one anchor, and only the compile task runs it, once per run:

```python
churn_score = MbtDag(
    "churn_score",
    project=project,
    command="score",
    anchor_command=["python3", "scripts/lake_anchor.py"],
    start_date=datetime(2026, 1, 1),
)
```

A pinned `anchor` param still wins, so a run can be pointed at any date by hand.

## When a task fails

| mbt exits | Airflow does | Why |
|---|---|---|
| `0` | succeeds | the node passed |
| `2` | fails at once, no retry | a gate, check, test, or monitor said no; retrying cannot change a verdict, so it goes to the model owner |
| anything else | retries (once by default), then fails | a hard error: infrastructure, data, or code. The last lines of mbt's own output are in the exception |
| no `mbt` executable | fails at once, no retry | a retry cannot install it |

A failed task leaves every task downstream of it in `upstream_failed`; independent branches keep going.

## Configuration

`MbtProject` says what to draw, and `MbtExecution` says how to run it.

```python
from datetime import timedelta

from mbt_airflow import MbtExecution, MbtProject

project = MbtProject(
    project_dir="/opt/churn_models",
    manifest_path="/opt/churn_models/manifest.json",
    target="prod",  # --target
    profiles_dir="/etc/mbt",  # --profiles-dir
)

execution = MbtExecution(
    mbt_executable="/opt/mbt/bin/mbt",
    workspace_root="/mnt/shared/mbt-airflow",
    env={"MLFLOW_TRACKING_URI": "http://mlflow:5000"},
    retries=2,
    retry_delay=timedelta(minutes=2),
    cleanup_on_success=True,
    operator_args={"pool": "training"},
)
```

| `MbtExecution` field | Meaning |
|---|---|
| `mbt_executable` | The `mbt` to run, `mbt` on the worker's `PATH` by default |
| `workspace_root` | Where each run keeps its target directories and dataset cache. Defaults to `$MBT_AIRFLOW_WORKSPACE`, else a temp directory |
| `docker` | Run every task in a container instead of on the worker: an `MbtDocker` ([below](#running-each-task-in-a-container)) |
| `env` | Extra environment for the mbt processes. The worker's own is inherited, which is where secrets read by `env_var()` in `profiles.yml` come from |
| `retries`, `retry_delay` | How a hard error is retried. A quality verdict never is |
| `cleanup_on_success` | Add a final task that removes the run's workspace, but only if every node succeeded, so a failed run keeps its logs and results |
| `operator_args` | Passed to every operator: `pool`, `queue`, `execution_timeout`, and any other `BaseOperator` argument. They win over the named settings |
| `phase_args` | The same, for the tasks of one step, by phase name (`train`, `gate`, `register`, `materialize`, `checks`). They win over `operator_args` |

**The workspace must be visible to every worker.**
On one machine a temp directory is fine.
On several, point `workspace_root` at a shared volume, because a model task reads the dataset another task built.
An object-store location is not supported.

## Running each task in a container

By default the tasks run `mbt` on the worker.
`MbtDocker` runs each one in a container instead, started through the Docker API, which is how a team that deploys its project as a pinned image keeps the image the code that runs ([ADR-34](adr/0034-airflow-tasks-run-in-a-container.md)):

```bash
pip install 'mbt-airflow[docker]'   # the Docker SDK; imported only when a container starts
```

```python
from datetime import timedelta
from pathlib import Path

from mbt_airflow import DockerUnit, MbtDocker, MbtExecution, MbtProject


def unit() -> DockerUnit:
    pin = Path("/opt/deploy/image.digest").read_text().strip()
    return DockerUnit(
        image=pin,  # registry/churn@sha256:...
        network="mbt_default",
        volumes={"/srv/mbt-workspace": "/workspace"},  # host path -> path in the container
        working_dir="/app/project",
        env={"MLFLOW_TRACKING_URI": "http://mlflow:5000"},
    )


execution = MbtExecution(
    docker=MbtDocker(unit),
    workspace_root="/workspace/mbt-airflow",  # a path IN the container
    retries=1,
    retry_delay=timedelta(seconds=5),
)
project = MbtProject(
    project_dir="/app/project",  # also in the container
    manifest_path="/opt/deploy/manifest.json",  # on the worker: the DAG is drawn from it
    target="prod",
)
```

| `DockerUnit` field | Meaning |
|---|---|
| `image` | The image to run, normally pinned by digest |
| `network` | The docker network to join, where the services the project talks to live |
| `volumes` | Host path to container path. The run's workspace lives on one of these, because a model task reads the dataset another task built |
| `working_dir` | Where the project is in the container |
| `env` | The container's whole environment. The worker's is **not** inherited: a container starts clean, so the credentials it needs are named here |

`MbtDocker(unit)` accepts a function as well as a `DockerUnit`, and calls it when the task runs, not when the DAG file is parsed.
Read the pin there: a pin that changes between parsing and running is then the one the task uses, and a missing pin fails the task with your message.

What differs from running locally:

- **`workspace_root` is required.** It is a path in the container, and the worker's temp directory is not somewhere a container can see.
- **The worker needs the Docker socket**, which is root on the host.
- **The task log is written when the container exits**, not as it runs: a stream would time out on a quiet stretch of a long training. Killing the task stops the container.
- **Results come back through Docker.** The worker reads a node's `run_results.json` out of the finished container, so it needs no mount of the workspace, and a failed task's message still carries the gate detail. `cleanup_on_success` removes the workspace by running `rm -rf` in the unit.
- **Failure handling is unchanged.** Exit 2 fails at once, any other nonzero exit is retried, and the last lines of mbt's stderr are in the exception.

### Around `mbt`

Every mbt process is told where its task's results are, in `MBT_AIRFLOW_TARGET_PATH` (the task's private target directory) and, for a node task, `MBT_AIRFLOW_MANIFEST` (the run's manifest).
Give `mbt_executable` a wrapper script that runs `mbt "$@"` and acts on them, then exits with mbt's own code:

```bash
#!/usr/bin/env bash
mbt "$@"
code=$?
python3 scripts/push_metrics.py --target-dir "$MBT_AIRFLOW_TARGET_PATH" --manifest "$MBT_AIRFLOW_MANIFEST" . || true
exit "$code"
```

The configured `env` cannot shadow these two.

## Inside a DAG you already have

`MbtTaskGroup` draws the same tasks inside a DAG of your own, so you can chain them with other work:

```python
from datetime import datetime

from airflow.sdk import DAG, BaseOperator

from mbt_airflow import MbtProject, MbtTaskGroup

project = MbtProject("/opt/churn_models", "/opt/churn_models/manifest.json")

with DAG("nightly", schedule="@daily", start_date=datetime(2026, 1, 1)):
    ingest = BaseOperator(task_id="ingest")
    train = MbtTaskGroup("train", project=project, select=["tag:weekly"])
    publish = BaseOperator(task_id="publish")
    ingest >> train >> publish
```

Task ids are prefixed by the group (`train.churn_classifier`), and each group has its own workspace, so several groups can share one DAG.
`anchor` and `mbt_vars` are literals or Jinja templates.

## Limits

- **Local and Docker execution only.** A Kubernetes pod per task is not built; the runner seam ([ADR-34](adr/0034-airflow-tasks-run-in-a-container.md)) is where it would go.
- **Parallel branches need a real tracking server.** Independent models run at once under a real executor, and two training jobs writing one sqlite MLflow database at once is the first thing that goes wrong. Use an MLflow server for anything beyond a laptop.
- **The manifest has to be republished** when the project's nodes change. A stale one fails the run, as described above, instead of drifting silently.
