"""MbtDag and MbtTaskGroup: the tasks, the edges, and what the templates render to."""

import tempfile
from datetime import UTC, datetime, timedelta
from pathlib import Path
from typing import Any

import pytest
from airflow.sdk import DAG, BaseOperator
from mbt_airflow import (
    MbtCleanupOperator,
    MbtCompileOperator,
    MbtDag,
    MbtExecution,
    MbtNodeOperator,
    MbtProject,
    MbtTaskGroup,
)
from mbt_airflow.dag import WORKSPACE_ENV

from mbt.exceptions import ConfigError, StateError

START = datetime(2026, 1, 1, tzinfo=UTC)
WEEKLY = ["mbt_compile", "churn_training_set", "upsell_training_set"]
WEEKLY += ["churn_classifier", "upsell_classifier"]


@pytest.fixture()
def project(churn_demo: tuple[Path, Path]) -> MbtProject:
    return MbtProject(project_dir=churn_demo[0], manifest_path=churn_demo[1])


def make_dag(project: MbtProject, **kwargs: Any) -> MbtDag:
    kwargs.setdefault("select", ["tag:weekly"])
    kwargs.setdefault("start_date", START)
    kwargs.setdefault("execution", MbtExecution(workspace_root="/ws"))
    return MbtDag("churn", project=project, **kwargs)


def render(dag: DAG, task_id: str, **context: Any) -> BaseOperator:
    """Render a task's templated fields the way a run would, then return it."""
    values: dict[str, Any] = {
        "params": {"anchor": "", "vars": ""},
        "run_id": "manual__2026-11-12T10:00:00+00:00",
        "dag": dag,
        "data_interval_end": None,
        "logical_date": datetime(2026, 11, 12, 10, tzinfo=UTC),
        "run_after": datetime(2026, 11, 12, 11, tzinfo=UTC),
    }
    values.update(context)
    task = dag.task_dict[task_id]
    task.render_template_fields(values)
    return task


# -- the DAG ------------------------------------------------------------------------


def test_the_dag_has_one_task_per_node_wired_from_the_manifest(project: MbtProject) -> None:
    dag = make_dag(project)
    tasks = dag.task_dict
    assert sorted(tasks) == sorted(WEEKLY)
    assert isinstance(tasks["mbt_compile"], MbtCompileOperator)
    assert all(isinstance(tasks[t], MbtNodeOperator) for t in WEEKLY[1:])
    assert tasks["mbt_compile"].downstream_task_ids == {"churn_training_set", "upsell_training_set"}
    assert tasks["churn_training_set"].downstream_task_ids == {"churn_classifier"}
    assert tasks["upsell_classifier"].upstream_task_ids == {"upsell_training_set"}
    assert set(dag.tags) == {"mbt", "build", "churn_demo"}


def test_a_score_dag_is_just_compile_then_the_scoring_node(project: MbtProject) -> None:
    dag = make_dag(project, command="score", select=["tag:daily"])
    assert sorted(dag.task_dict) == ["mbt_compile", "retention_scoring"]
    assert dag.task_dict["retention_scoring"].upstream_task_ids == {"mbt_compile"}
    assert "score" in dag.tags


def test_the_dag_is_manual_and_does_not_backfill_unless_told_otherwise(project: MbtProject) -> None:
    dag = make_dag(project)
    assert dag.catchup is False and dag.schedule is None
    cron = make_dag(project, schedule="0 6 * * 1", catchup=True, tags=["team"])
    assert cron.catchup is True and cron.schedule == "0 6 * * 1" and "team" in cron.tags


def test_the_dag_params_are_the_anchor_and_the_vars(project: MbtProject) -> None:
    assert make_dag(project).params["anchor"] == ""
    assert make_dag(project).params["vars"] == ""
    pinned = make_dag(project, anchor="2026-06-30T00:00:00Z", params={"owner": "ds"})
    assert pinned.params["anchor"] == "2026-06-30T00:00:00Z"
    assert pinned.params["owner"] == "ds"


def test_a_missing_manifest_says_where_it_looked(project: MbtProject, tmp_path: Path) -> None:
    gone = MbtProject(project_dir=project.project_dir, manifest_path=tmp_path / "nope.json")
    with pytest.raises(StateError, match="not found") as info:
        make_dag(gone)
    assert "mbt compile" in (info.value.hint or "")


def test_a_selection_that_draws_nothing_fails_the_dag_file_loudly(project: MbtProject) -> None:
    with pytest.raises(ConfigError, match="matched no dataset/model nodes"):
        make_dag(project, select=["tag:daily"])


# -- what the templates render to -----------------------------------------------------


def test_a_manual_run_takes_its_anchor_from_the_run(project: MbtProject) -> None:
    task = render(make_dag(project), "mbt_compile")
    assert task.anchor == "2026-11-12T10:00:00+00:00"  # the logical date: no interval to end


def test_a_scheduled_run_anchors_at_the_end_of_its_data_interval(project: MbtProject) -> None:
    end = datetime(2026, 11, 9, tzinfo=UTC)
    assert render(make_dag(project), "mbt_compile", data_interval_end=end).anchor == (
        "2026-11-09T00:00:00+00:00"
    )


def test_a_run_with_neither_falls_back_to_when_it_was_created(project: MbtProject) -> None:
    task = render(make_dag(project), "mbt_compile", data_interval_end=None, logical_date=None)
    assert task.anchor == "2026-11-12T11:00:00+00:00"


def test_the_anchor_param_pins_the_run(project: MbtProject) -> None:
    pinned = {"anchor": "2026-06-30T00:00:00Z", "vars": ""}
    assert render(make_dag(project), "mbt_compile", params=pinned).anchor == pinned["anchor"]


def test_vars_reach_the_compile_task_and_the_workspace_is_path_safe(project: MbtProject) -> None:
    dag = make_dag(project)
    compiled = render(dag, "mbt_compile", params={"anchor": "", "vars": "pr_auc_floor: 0.9"})
    assert compiled.mbt_vars == "pr_auc_floor: 0.9"
    node = render(make_dag(project), "churn_classifier")
    # no ':' or '+' from the run id, which would be awkward in a path on some filesystems
    assert node.workspace == "/ws/churn/manual__2026-11-12T10-00-00_00-00"


def test_every_task_of_a_run_shares_one_workspace(project: MbtProject) -> None:
    dag = make_dag(project)
    rendered = {render(dag, tid).workspace for tid in dag.task_dict}
    assert len(rendered) == 1


# -- configuration ------------------------------------------------------------------


def test_the_project_and_execution_settings_reach_every_task(project: MbtProject) -> None:
    configured = MbtProject(
        project.project_dir, project.manifest_path, target="prod", profiles_dir="/profiles"
    )
    execution = MbtExecution(
        mbt_executable="/opt/venv/bin/mbt",
        workspace_root="/ws",
        env={"MLFLOW_TRACKING_URI": "http://mlflow:5000"},
        retries=3,
        retry_delay=timedelta(seconds=10),
        operator_args={"pool": "mbt", "retries": 5},
    )
    dag = make_dag(configured, execution=execution)
    for task in dag.task_dict.values():
        assert task.mbt_executable == "/opt/venv/bin/mbt"
        assert (task.target, task.profiles_dir) == ("prod", "/profiles")
        assert task.env == {"MLFLOW_TRACKING_URI": "http://mlflow:5000"}
        assert task.retry_delay == timedelta(seconds=10)
        assert task.pool == "mbt"
        assert task.retries == 5  # operator_args win over the named settings


def test_by_default_a_hard_error_is_retried_once(project: MbtProject) -> None:
    task = make_dag(project).task_dict["churn_classifier"]
    assert (task.retries, task.retry_delay) == (1, timedelta(seconds=30))
    assert task.target is None and task.profiles_dir is None and task.env is None


def test_the_default_workspace_root_follows_the_environment(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    monkeypatch.setenv(WORKSPACE_ENV, "/mnt/shared/mbt")
    assert str(MbtExecution().workspace_root) == "/mnt/shared/mbt"
    monkeypatch.delenv(WORKSPACE_ENV)
    assert Path(MbtExecution().workspace_root) == Path(tempfile.gettempdir()) / "mbt-airflow"


# -- cleanup ------------------------------------------------------------------------


def test_cleanup_waits_for_the_last_tasks_only(project: MbtProject) -> None:
    dag = make_dag(project, execution=MbtExecution(workspace_root="/ws", cleanup_on_success=True))
    cleanup = dag.task_dict["mbt_cleanup"]
    assert isinstance(cleanup, MbtCleanupOperator)
    assert cleanup.upstream_task_ids == {"churn_classifier", "upsell_classifier"}
    assert cleanup.trigger_rule == "all_success"  # a failed run keeps its workspace
    rendered = render(dag, "mbt_cleanup")
    assert rendered.workspace == "/ws/churn/manual__2026-11-12T10-00-00_00-00"
    assert rendered.workspace_root == "/ws"


def test_there_is_no_cleanup_task_unless_asked_for(project: MbtProject) -> None:
    assert "mbt_cleanup" not in make_dag(project).task_dict


# -- the task group -----------------------------------------------------------------


def test_a_task_group_chains_with_your_own_tasks(project: MbtProject) -> None:
    with DAG("host", schedule=None, start_date=START) as dag:
        ingest = BaseOperator(task_id="ingest")
        group = MbtTaskGroup(
            "weekly",
            project=project,
            select=["tag:weekly"],
            execution=MbtExecution(workspace_root="/ws"),
        )
        publish = BaseOperator(task_id="publish")
        ingest >> group >> publish

    assert set(dag.task_dict) == {"ingest", "publish", *(f"weekly.{t}" for t in WEEKLY)}
    assert dag.task_dict["ingest"].downstream_task_ids == {"weekly.mbt_compile"}
    assert dag.task_dict["publish"].upstream_task_ids == {
        "weekly.churn_classifier",
        "weekly.upsell_classifier",
    }
    # the nodes' own names, not the group-prefixed ids, name their private directories
    node = render(dag, "weekly.churn_classifier")
    assert node.target_path() == Path(node.workspace) / "churn_classifier"


def test_two_groups_in_one_dag_do_not_share_a_workspace(project: MbtProject) -> None:
    with DAG("host", schedule=None, start_date=START) as dag:
        kwargs = {"project": project, "execution": MbtExecution(workspace_root="/ws")}
        MbtTaskGroup("weekly", select=["tag:weekly"], **kwargs)
        MbtTaskGroup("score", command="score", select=["tag:daily"], **kwargs)
    weekly = render(dag, "weekly.churn_classifier").workspace
    score = render(dag, "score.retention_scoring").workspace
    assert weekly.endswith("/weekly") and score.endswith("/score")
    assert weekly != score


def test_a_task_group_takes_a_pinned_anchor_and_vars(project: MbtProject) -> None:
    with DAG("host", schedule=None, start_date=START) as dag:
        MbtTaskGroup(
            "g",
            project=project,
            select=["tag:weekly"],
            anchor="{{ ds }}T00:00:00Z",
            mbt_vars="pr_auc_floor: 0.5",
        )
    compiled = render(dag, "g.mbt_compile", ds="2026-06-30")
    assert compiled.anchor == "2026-06-30T00:00:00Z"
    assert compiled.mbt_vars == "pr_auc_floor: 0.5"


def test_a_task_group_defaults_to_the_run_derived_anchor(project: MbtProject) -> None:
    with DAG("host", schedule=None, start_date=START) as dag:
        MbtTaskGroup("g", project=project, select=["tag:weekly"])
    assert render(dag, "g.mbt_compile").anchor == "2026-11-12T10:00:00+00:00"
