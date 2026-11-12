"""mbt-airflow end to end: a real Airflow run of the real CLI over churn_demo (ADR-33).

``dag.test()`` runs the DAG in-process with Airflow's own task runner: real
operators, task states, retries and XCom, but no scheduler or executor. Each task
runs the real ``mbt`` in a subprocess, so this is the full path: real XGBoost
training, MLflow on sqlite, one private target directory per task over one
shared dataset cache.

Airflow reads its settings at import, so every scenario runs in a fresh
subprocess with its own AIRFLOW_HOME.
"""

import json
import os
import subprocess
import sys
from pathlib import Path
from typing import Any

import pytest
from e2e_utils import DEMO_ANCHOR, run_mbt

pytestmark = pytest.mark.e2e

BIN = Path(sys.executable).parent

#: The DAG file a deployment would carry: it reads its settings from the environment
#: so each scenario can reuse it. ``dag.test()`` syncs DAGs from the configured dags
#: folder, so the DAG has to live in a real file there.
_DAG_FILE = """
import json, os
from datetime import UTC, datetime, timedelta
from mbt_airflow import MbtDag, MbtExecution, MbtProject

spec = json.loads(os.environ["MBT_E2E_SPEC"])
dag = MbtDag(
    spec["dag_id"],
    project=MbtProject(spec["project"], spec["manifest"]),
    execution=MbtExecution(
        mbt_executable=spec["mbt"],
        workspace_root=spec["workspace"],
        retry_delay=timedelta(seconds=1),
    ),
    command=spec["command"],
    select=spec["select"],
    start_date=datetime(2026, 1, 1, tzinfo=UTC),
)
"""

#: Runs the DAG in that file and writes {state, tasks: {id: {state, tries}}}.
_RUNNER = """
import json, os, runpy

spec = json.loads(os.environ["MBT_E2E_SPEC"])
dag = runpy.run_path(os.path.join(os.environ["AIRFLOW__CORE__DAGS_FOLDER"], "mbt_dag.py"))["dag"]
run = dag.test(run_conf=spec["conf"])
tasks = {ti.task_id: {"state": str(ti.state), "tries": ti.try_number}
         for ti in run.get_task_instances()}
open(spec["out"], "w").write(json.dumps({"state": str(run.state), "tasks": tasks}))
"""


@pytest.fixture(scope="module")
def airflow_home(tmp_path_factory: pytest.TempPathFactory) -> Path:
    """An Airflow metadata database, migrated once for the module."""
    home = tmp_path_factory.mktemp("airflow-home")
    proc = subprocess.run(
        [str(BIN / "airflow"), "db", "migrate"],
        env=_airflow_env(home),
        capture_output=True,
        text=True,
        timeout=300,
        check=False,
    )
    assert proc.returncode == 0, proc.stderr
    return home


def _airflow_env(home: Path, dags: Path | None = None) -> dict[str, str]:
    env = {
        **os.environ,
        "AIRFLOW_HOME": str(home),
        "AIRFLOW__CORE__LOAD_EXAMPLES": "False",
        "AIRFLOW__LOGGING__LOGGING_LEVEL": "WARNING",
    }
    if dags is not None:
        env["AIRFLOW__CORE__DAGS_FOLDER"] = str(dags)
    return env


class Project:
    """A disposable churn_demo, compiled once for drawing, run through Airflow."""

    def __init__(self, root: Path, project: Path, home: Path) -> None:
        self.root, self.project, self.home = root, project, home
        self.manifest = root / "drawn" / "manifest.json"
        self.dags = root / "dags"
        self.dags.mkdir()
        (self.dags / "mbt_dag.py").write_text(_DAG_FILE)
        run_mbt(
            ["compile", "--anchor", DEMO_ANCHOR, "--target-path", str(self.manifest.parent)],
            project,
        )

    def run(
        self,
        dag_id: str,
        *,
        command: str = "build",
        select: list[str],
        conf: dict[str, str] | None = None,
    ) -> dict[str, Any]:
        out = self.root / f"{dag_id}.json"
        spec = {
            "dag_id": dag_id,
            "project": str(self.project),
            "manifest": str(self.manifest),
            "mbt": str(BIN / "mbt"),
            "workspace": str(self.root / "runs"),
            "command": command,
            "select": select,
            "conf": conf or {},
            "out": str(out),
        }
        proc = subprocess.run(
            [sys.executable, "-c", _RUNNER],
            cwd=self.root,
            env={**_airflow_env(self.home, self.dags), "MBT_E2E_SPEC": json.dumps(spec)},
            capture_output=True,
            text=True,
            timeout=900,
            stdin=subprocess.DEVNULL,
            check=False,
        )
        assert out.is_file(), (
            f"the DAG never finished\nstdout:\n{proc.stdout}\nstderr:\n{proc.stderr}"
        )
        result: dict[str, Any] = json.loads(out.read_text())
        result["workspace"] = next((self.root / "runs" / dag_id).iterdir())
        return result


@pytest.fixture()
def project(demo_copy: Path, airflow_home: Path) -> Project:
    return Project(demo_copy.parent, demo_copy, airflow_home)


def states(result: dict[str, Any]) -> dict[str, str]:
    return {task: info["state"] for task, info in result["tasks"].items()}


def run_results(result: dict[str, Any], task: str) -> dict[str, dict]:
    payload = json.loads((result["workspace"] / task / "run_results.json").read_text())
    return {r["unique_id"]: r for r in payload["results"]}


WEEKLY = ["tag:weekly"]
MODELS = ("churn_classifier", "upsell_classifier")
DATASETS = ("churn_training_set", "upsell_training_set")


def test_the_whole_lifecycle_runs_as_airflow_tasks(project: Project) -> None:
    """build -> promote (a reviewed step, deliberately not a task) -> score -> monitor."""
    built = project.run("weekly", select=WEEKLY, conf={"anchor": DEMO_ANCHOR})
    assert built["state"] == "success", built
    assert set(states(built).values()) == {"success"}
    assert set(built["tasks"]) == {"mbt_compile", *DATASETS, *MODELS}
    assert all(info["tries"] == 1 for info in built["tasks"].values())

    workspace: Path = built["workspace"]
    # one private target directory per task, holding that task's node and nothing else
    for model in MODELS:
        assert set(run_results(built, model)) == {f"model.churn_demo.{model}"}
        assert not (workspace / model / "datasets").exists()  # read from the shared cache
    # ...and each dataset was built once, into the one shared cache
    assert sorted(p.name for p in (workspace / "datasets").iterdir()) == sorted(DATASETS)

    run_mbt(["promote", "--model", "churn_classifier", "--to", "production"], project.project)

    scored = project.run(
        "score", command="score", select=["tag:daily"], conf={"anchor": DEMO_ANCHOR}
    )
    assert scored["state"] == "success", scored
    assert set(scored["tasks"]) == {"mbt_compile", "retention_scoring"}  # no retrain in front of it

    monitored = project.run(
        "monitor",
        command="monitor",
        select=["tag:daily"],
        conf={"anchor": "2026-07-31T00:00:00Z"},  # a month on: the labels have matured
    )
    assert monitored["state"] == "success", monitored
    outcome = run_results(monitored, "retention_scoring")["scoring.churn_demo.retention_scoring"]
    assert outcome["message"] == "evaluated 1 of 1 matured prediction run(s)"


def test_a_failed_gate_is_a_verdict_and_is_not_retried(project: Project) -> None:
    result = project.run(
        "gates", select=WEEKLY, conf={"anchor": DEMO_ANCHOR, "vars": "pr_auc_floor: 0.99"}
    )
    assert result["state"] == "failed"
    assert states(result) == {
        "mbt_compile": "success",
        **dict.fromkeys(DATASETS, "success"),
        **dict.fromkeys(MODELS, "failed"),
    }
    for model in MODELS:
        assert result["tasks"][model]["tries"] == 1  # exit 2 fails at once
        node = run_results(result, model)[f"model.churn_demo.{model}"]
        assert node["status"] == "gate_failed"


def test_a_hard_error_is_retried_and_everything_after_it_is_skipped(project: Project) -> None:
    result = project.run("hard", select=WEEKLY, conf={"anchor": "2020-01-01T00:00:00Z"})
    assert result["state"] == "failed"
    assert result["tasks"]["mbt_compile"] == {"state": "failed", "tries": 2}  # retries=1
    assert {s for t, s in states(result).items() if t != "mbt_compile"} == {"upstream_failed"}


def test_a_dag_drawn_from_an_older_project_refuses_to_run(project: Project) -> None:
    """A model gains the selected tag after the DAG was drawn: it would never run."""
    deep = project.project / "models" / "churn_classifier_deep.yml"
    deep.write_text(deep.read_text().replace("tags: [churn]", "tags: [churn, weekly]"))
    result = project.run("stale", select=WEEKLY, conf={"anchor": DEMO_ANCHOR})
    assert result["state"] == "failed"
    assert result["tasks"]["mbt_compile"] == {"state": "failed", "tries": 1}  # deterministic
    assert {s for t, s in states(result).items() if t != "mbt_compile"} == {"upstream_failed"}
