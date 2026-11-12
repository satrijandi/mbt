"""The operators: what they run, and how each exit code fails (ADR-33).

The tasks run a fake ``mbt`` (see airflow_helpers) so every exit code is reachable
without training anything; the real CLI is exercised by tests/test_e2e_airflow.py.
"""

import json
import threading
from pathlib import Path
from typing import Any

import pytest
from airflow.sdk.exceptions import AirflowException, AirflowFailException
from airflow_helpers import ANCHOR, logged_calls, write_fake_mbt
from mbt_airflow import (
    MbtCleanupOperator,
    MbtCompileOperator,
    MbtNodeOperator,
    TaskSpec,
    plan_tasks,
)

MODEL = TaskSpec(
    "churn_classifier",
    "model.churn_demo.churn_classifier",
    "model",
    "build",
    ("churn_training_set",),
)
SCORING = TaskSpec(
    "retention_scoring",
    "scoring.churn_demo.retention_scoring",
    "scoring",
    "score",
    ("mbt_compile",),
)


@pytest.fixture()
def fake(tmp_path: Path, churn_demo: tuple[Path, Path], monkeypatch: pytest.MonkeyPatch) -> str:
    """The fake mbt, logging every call, serving the real churn_demo manifest."""
    monkeypatch.setenv("FAKE_MBT_LOG", str(tmp_path / "calls.jsonl"))
    monkeypatch.setenv("FAKE_MBT_MANIFEST", str(churn_demo[1]))
    return write_fake_mbt(tmp_path)


def node(tmp_path: Path, fake: str, spec: TaskSpec = MODEL, **kwargs: Any) -> MbtNodeOperator:
    return MbtNodeOperator(
        task_id=spec.task_id,
        spec=spec,
        project_dir="/proj",
        workspace=str(tmp_path / "ws"),
        anchor=ANCHOR,
        mbt_executable=fake,
        **kwargs,
    )


def compile_op(tmp_path: Path, fake: str, **kwargs: Any) -> MbtCompileOperator:
    options: dict[str, Any] = {
        "plan_command": "build",
        "plan_select": ["tag:weekly"],
        "plan_exclude": None,
        "rendered_for": [
            "mbt_compile",
            "churn_training_set",
            "upsell_training_set",
            "churn_classifier",
            "upsell_classifier",
        ],
    }
    options.update(kwargs)
    return MbtCompileOperator(
        task_id="mbt_compile",
        project_dir="/proj",
        workspace=str(tmp_path / "ws"),
        anchor=ANCHOR,
        mbt_executable=fake,
        **options,
    )


# -- what they run ----------------------------------------------------------------


def test_a_build_node_runs_exactly_its_node_against_the_runs_manifest(tmp_path: Path) -> None:
    ws = tmp_path / "ws"
    assert node(tmp_path, "mbt").argv() == [
        *["mbt", "build", "--project-dir", "/proj"],
        *["--manifest", f"{ws}/mbt_compile/manifest.json"],
        *["--select", "model.churn_demo.churn_classifier", "--threads", "1"],
        *["--target-path", f"{ws}/churn_classifier"],
        *["--no-auto-materialize", "--datasets-path", f"{ws}/datasets"],
    ]


def test_a_score_node_has_no_dataset_cache(tmp_path: Path) -> None:
    argv = node(tmp_path, "mbt", SCORING).argv()
    assert argv[:2] == ["mbt", "score"]
    assert "--no-auto-materialize" not in argv and "--datasets-path" not in argv
    assert argv[argv.index("--select") + 1] == SCORING.uid


def test_profile_and_target_are_passed_on_every_command(tmp_path: Path) -> None:
    extra = {"target": "prod", "profiles_dir": "/profiles"}
    compiled = compile_op(tmp_path, "mbt", **extra).argv()
    built = node(tmp_path, "mbt", **extra).argv()
    for argv in (compiled, built):
        assert argv[argv.index("--target") + 1] == "prod"
        assert argv[argv.index("--profiles-dir") + 1] == "/profiles"


def test_the_compile_task_owns_the_vars_and_the_anchor(tmp_path: Path) -> None:
    ws = tmp_path / "ws"
    assert compile_op(tmp_path, "mbt").argv() == [
        *["mbt", "compile", "--project-dir", "/proj", "--anchor", ANCHOR],
        *["--target-path", f"{ws}/mbt_compile"],
    ]
    with_vars = compile_op(tmp_path, "mbt", mbt_vars="pr_auc_floor: 0.9").argv()
    assert with_vars[-2:] == ["--vars", "pr_auc_floor: 0.9"]
    # the node tasks never take --vars: the manifest carries them, so the cache keys agree
    assert "--vars" not in node(tmp_path, "mbt", mbt_vars="pr_auc_floor: 0.9").argv()


# -- how they fail ------------------------------------------------------------------


def test_a_passing_node_returns_its_own_result(
    tmp_path: Path, fake: str, monkeypatch: pytest.MonkeyPatch
) -> None:
    monkeypatch.setenv("FAKE_MBT_METRICS", json.dumps({"pr_auc": 0.5}))
    outcome = node(tmp_path, fake).execute({})
    assert outcome == {
        "node": MODEL.uid,
        "status": "success",
        "metrics": {"pr_auc": 0.5},
    }
    [call] = logged_calls(tmp_path / "calls.jsonl")
    assert call["argv"][0] == "build"


def test_a_quality_verdict_fails_at_once_with_the_gate_detail(
    tmp_path: Path, fake: str, monkeypatch: pytest.MonkeyPatch
) -> None:
    """Exit 2 is deterministic: AirflowFailException is what makes Airflow skip the retry."""
    monkeypatch.setenv("FAKE_MBT_EXIT", "2")
    monkeypatch.setenv("FAKE_MBT_STATUS", "gate_failed")
    monkeypatch.setenv("FAKE_MBT_MESSAGE", "gate breach: pr_auc=0.3470 failed threshold 0.99")
    with pytest.raises(AirflowFailException, match="quality verdict") as info:
        node(tmp_path, fake).execute({})
    assert "pr_auc=0.3470 failed threshold 0.99" in str(info.value)


def test_a_hard_error_is_retryable_and_quotes_mbts_own_words(
    tmp_path: Path, fake: str, monkeypatch: pytest.MonkeyPatch
) -> None:
    monkeypatch.setenv("FAKE_MBT_EXIT", "1")
    monkeypatch.setenv("FAKE_MBT_RESULTS", "no")
    monkeypatch.setenv(
        "FAKE_MBT_STDERR", "one\ntwo\nthree\nError: window resolves to an empty range\nhint: fix it"
    )
    with pytest.raises(AirflowException, match="hard error") as info:
        node(tmp_path, fake).execute({})
    assert not isinstance(info.value, AirflowFailException)  # Airflow's retry policy applies
    assert "Error: window resolves to an empty range | hint: fix it" in str(info.value)
    assert "one" not in str(info.value)  # only the tail


def test_a_missing_executable_fails_without_a_retry(tmp_path: Path) -> None:
    with pytest.raises(AirflowFailException, match=r"cannot run .*no-such-mbt"):
        node(tmp_path, str(tmp_path / "no-such-mbt")).execute({})


def test_the_executable_sees_the_configured_environment(
    tmp_path: Path, fake: str, monkeypatch: pytest.MonkeyPatch
) -> None:
    monkeypatch.setenv("FAKE_MBT_MARKER", "from the worker")
    operator = node(tmp_path, fake, env={"FAKE_MBT_MARKER": "from the config"})
    operator.execute({})
    assert logged_calls(tmp_path / "calls.jsonl")[0]["marker"] == "from the config"


def test_a_node_missing_from_its_results_is_reported_not_guessed(
    tmp_path: Path, fake: str, monkeypatch: pytest.MonkeyPatch
) -> None:
    monkeypatch.setenv("FAKE_MBT_NODE", "model.churn_demo.somebody_else")
    assert node(tmp_path, fake).execute({})["status"] == "missing from run_results.json"
    monkeypatch.setenv("FAKE_MBT_RESULTS", "no")
    fresh = tmp_path / "second-run"  # the first call left a run_results.json in its own
    assert node(fresh, fake).execute({})["status"] == "no run_results.json"


# -- the compile task ---------------------------------------------------------------


def test_compile_returns_where_the_manifest_is(tmp_path: Path, fake: str) -> None:
    outcome = compile_op(tmp_path, fake).execute({})
    assert outcome == {"manifest": str(tmp_path / "ws" / "mbt_compile" / "manifest.json")}
    assert (tmp_path / "ws" / "mbt_compile" / "manifest.json").is_file()


def test_a_dag_that_no_longer_matches_the_project_refuses_to_run(tmp_path: Path, fake: str) -> None:
    """A model gained the tag the DAG selects on after the DAG was drawn: it would
    never run, behind a green run. A node the project dropped is the other half."""
    drawn = ["mbt_compile", "churn_training_set", "churn_classifier", "gone_model"]
    with pytest.raises(AirflowFailException, match="stale") as info:
        compile_op(tmp_path, fake, rendered_for=drawn).execute({})
    message = str(info.value)
    assert "drawn but gone ['gone_model']" in message
    assert "in the project but not drawn ['upsell_classifier', 'upsell_training_set']" in message


def test_what_compile_checks_is_what_plan_tasks_draws(
    tmp_path: Path, fake: str, manifest: Any
) -> None:
    drawn = [s.task_id for s in plan_tasks(manifest, "build", ["tag:weekly"])]
    compile_op(tmp_path, fake, rendered_for=drawn).execute({})  # no raise


# -- killing a task -----------------------------------------------------------------


def test_killing_the_task_stops_mbt(
    tmp_path: Path, fake: str, monkeypatch: pytest.MonkeyPatch
) -> None:
    """Airflow calls on_kill on a timeout or a clear; mbt must not be left training."""
    started = tmp_path / "started"
    monkeypatch.setenv("FAKE_MBT_STARTED", str(started))
    monkeypatch.setenv("FAKE_MBT_SLEEP", "60")
    operator = node(tmp_path, fake)
    failures: list[BaseException] = []

    def run() -> None:
        try:
            operator.execute({})
        except AirflowException as exc:
            failures.append(exc)

    runner = threading.Thread(target=run)
    runner.start()
    for _ in range(300):
        if started.exists():
            break
        threading.Event().wait(0.1)
    assert started.exists(), "the fake mbt never started"
    operator.on_kill()
    runner.join(30)
    assert not runner.is_alive()
    assert len(failures) == 1 and "hard error" in str(failures[0])  # SIGTERM -> nonzero exit


def test_killing_a_task_that_never_started_is_harmless(tmp_path: Path) -> None:
    node(tmp_path, "mbt").on_kill()


# -- cleanup ------------------------------------------------------------------------


def test_cleanup_removes_the_run_and_only_the_run(tmp_path: Path) -> None:
    root = tmp_path / "root"
    run, other = root / "dag" / "run-1", root / "dag" / "run-2"
    for directory in (run, other):
        (directory / "datasets").mkdir(parents=True)
        (directory / "datasets" / "x").write_text("cache")
    operator = MbtCleanupOperator(task_id="c", workspace=str(run), workspace_root=str(root))
    assert operator.execute({}) == str(run.resolve())
    assert not run.exists()
    assert other.is_dir()


@pytest.mark.parametrize("where", ["the root itself", "outside the root"])
def test_cleanup_refuses_to_delete_anything_but_a_child_of_the_root(
    tmp_path: Path, where: str
) -> None:
    root, outside = tmp_path / "root", tmp_path / "elsewhere"
    root.mkdir()
    outside.mkdir()
    (outside / "precious").write_text("keep")
    target = root if where == "the root itself" else outside
    operator = MbtCleanupOperator(task_id="c", workspace=str(target), workspace_root=str(root))
    with pytest.raises(AirflowFailException, match="refusing to remove"):
        operator.execute({})
    assert target.is_dir()
    assert (outside / "precious").read_text() == "keep"
