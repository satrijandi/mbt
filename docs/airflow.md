# Running on Airflow

`mbt-airflow` draws an mbt project as an Airflow DAG, the way `dbt-cosmos` does for dbt: one task per node, with the edges taken from the manifest.
A weekly retrain stops being one opaque box.
Each dataset and each model is its own task, with its own state, its own retry, and its own place in the graph.

```bash
pip install mbt-airflow   # on the scheduler and on every worker; mbt-core comes with it
```

Airflow 3.3.1 or later is required.
Every worker also needs the `mbt` executable and whatever your models import (XGBoost, MLflow, and so on).
Airflow itself needs only the package, because the tasks run the real CLI in a subprocess ([ADR-33](adr/0033-mbt-project-as-an-airflow-dag.md)).

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
| `build` (the default), `run`, `test` | datasets and models | `mbt <command> --select <node> --no-auto-materialize` |
| `score`, `monitor` | scoring nodes | `mbt <command> --select <node>` |

They are separate DAGs because mbt's commands partition the nodes that way.
A scoring node depends on its model in the manifest, but it reads the registry's champion at run time, on its own cadence, so a daily score must not wait for a weekly retrain.
The score DAG therefore has no edge to a training task.

A model selection is drawn with the datasets it needs, because a model task reads its dataset instead of building it.
For `churn_demo`, `select=["tag:weekly"]` selects two models and draws five tasks:

```text
mbt_compile --+--> churn_training_set --> churn_classifier
              |
              +--> upsell_training_set --> upsell_classifier
```

Promotion is not a task.
It stays a reviewed step between the build DAG and the score DAG ([GitOps & CI](gitops.md)).

## What each task does

The first task, `mbt_compile`, compiles the project once at the run's anchor.
Every node task then executes that manifest verbatim (`--manifest`), so snapshots, windows, and seeds agree across the whole run.

Each node task gets its own target directory and shares one dataset cache with the rest of the run ([ADR-32](adr/0032-target-path-and-exact-node-execution.md)).
A task's result is read from its own `run_results.json` and returned as its XCom: the node's status, its metrics, and any message.

The anchor is the run's `anchor` param when you set one, else the end of its data interval, else its logical date.
The `vars` param is passed to the compile task only, which is what keeps the dataset and the model tasks agreeing on cache keys.

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
| `env` | Extra environment for the mbt processes. The worker's own is inherited, which is where secrets read by `env_var()` in `profiles.yml` come from |
| `retries`, `retry_delay` | How a hard error is retried. A quality verdict never is |
| `cleanup_on_success` | Add a final task that removes the run's workspace, but only if every node succeeded, so a failed run keeps its logs and results |
| `operator_args` | Passed to every operator: `pool`, `queue`, `execution_timeout`, and any other `BaseOperator` argument. They win over the named settings |

**The workspace must be visible to every worker.**
On one machine a temp directory is fine.
On several, point `workspace_root` at a shared volume, because a model task reads the dataset another task built.
An object-store location is not supported.

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

- **Local execution only.** The tasks run `mbt` on the worker. Running each in a container is a later addition.
- **Parallel branches need a real tracking server.** Independent models run at once under a real executor, and two training jobs writing one sqlite MLflow database at once is the first thing that goes wrong. Use an MLflow server for anything beyond a laptop.
- **The manifest has to be republished** when the project's nodes change. A stale one fails the run, as described above, instead of drifting silently.
