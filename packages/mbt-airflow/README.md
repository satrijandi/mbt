# mbt-airflow

An [mbt](https://github.com/satrijandi/mbt) project as an Apache Airflow DAG, the way `dbt-cosmos` draws a dbt project: one task per node, with the edges taken from the manifest.
A weekly retrain stops being one opaque box.
Each dataset and each model is its own task, with its own state, its own retry, and its own place in the graph.

```bash
pip install mbt-airflow        # on the scheduler and every worker; brings mbt-core
```

mbt is not on PyPI yet; see [Installation](https://satrijandi.github.io/mbt/installation/) for installing from a release tag.
Airflow 3.3.1 or later is required (`airflow.sdk`).
Every worker also needs the `mbt` executable and whatever your models import, because the tasks run the real CLI in a subprocess.

## A DAG in a few lines

```python
from datetime import datetime

from mbt_airflow import MbtDag, MbtProject

project = MbtProject(
    project_dir="/opt/churn_models",
    manifest_path="/opt/churn_models/manifest.json",  # from `mbt compile`, published by CI
)

churn_weekly = MbtDag(
    "churn_weekly",
    project=project,
    select=["tag:weekly"],
    schedule="0 6 * * 1",
    start_date=datetime(2026, 1, 1),
)
```

`select=["tag:weekly"]` on the `churn_demo` fixture project draws five tasks:

```text
mbt_compile --+--> churn_training_set --> churn_classifier
              |
              +--> upsell_training_set --> upsell_classifier
```

A model is drawn with the datasets it needs, and the first task compiles the project once at the run's anchor, so every other task executes that one manifest.
`select` and `exclude` are mbt's own selector grammar.

## What it does

- **One DAG per command.** `build` (also `run`, `test`) draws datasets and models; `score` and `monitor` draw scoring nodes. A daily score never waits for a weekly retrain.
- **Exit codes keep their meaning.** A failed gate, check, test, or monitor (exit `2`) fails the task at once with no retry, and the failure message carries the gate detail. Any other failure is retried, and quotes mbt's own last lines.
- **Private state per task.** Each task owns a target directory and shares one dataset cache with the rest of the run, so tasks do not fight over `target/` and a model reads the dataset another task built.
- **A stale DAG refuses to run.** If the project gained a node the selector matches after the DAG was drawn, the run fails and names it, instead of quietly running a different graph than the one you see.
- **Local or in a container.** By default the tasks run `mbt` on the worker. `MbtDocker` runs each one in a container started through the Docker API (`pip install 'mbt-airflow[docker]'`), from an image you pin, with the results read back through Docker. Failures keep their meaning either way.
- **An anchor from your data.** `anchor_command` names a command that prints the run's anchor (a lake's newest cohort, say), run once by the compile task, with a pinned `anchor` param always winning.
- **`MbtTaskGroup`** draws the same tasks inside a DAG you already have.

The workspace holding those directories must be visible to every worker, so on more than one machine it is a shared volume.
Parallel branches need a real MLflow tracking server, not a sqlite file.

The full guide, with configuration, failure handling and limits, is [Running on Airflow](https://satrijandi.github.io/mbt/airflow/), and the reasoning is [ADR-33](https://satrijandi.github.io/mbt/adr/0033-mbt-project-as-an-airflow-dag/).
