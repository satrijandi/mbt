"""One node per task (ADR-32): a private target path, a shared dataset cache, exact plans.

These drive the real CLI over the fake adapters the way an orchestrator would:
a dataset task, then a model task that reads what the dataset task built,
each with its own ``--target-path`` over one ``--datasets-path``.
"""

import json
import multiprocessing
import os
import threading
from pathlib import Path

import pytest
from cli_unit_helpers import (  # noqa: F401 - autouse fixture import
    ANCHOR,
    cli_process_state,
    debug,
    invoke,
)
from exec_unit_helpers import DATASET_UID, make_execution_context, recording_bus

from mbt.adapters.registry import AdapterRegistry
from mbt.exceptions import StateError
from mbt.execute.planner import plan_execution
from mbt.execute.runners import DatasetRunner
from mbt.state.lock import LOCK_FILE, materialization_lock, target_lock
from mbt_adapter_base import DatasetSpec

MODEL_UID = "model.demo.churn_model"


def build(project: Path, *args: str, command: str = "build", env: dict | None = None):
    return invoke([command, "--project-dir", str(project), "--anchor", ANCHOR, *args], env=env)


def results_of(target: Path) -> dict[str, dict]:
    payload = json.loads((target / "run_results.json").read_text())
    return {r["unique_id"]: r for r in payload["results"]}


def cache_keys(datasets: Path) -> list[Path]:
    return sorted(p for p in (datasets / "churn_training").iterdir() if p.is_dir())


# -- a private target path ----------------------------------------------------------


def test_a_target_path_takes_the_artifacts_and_the_lock_with_it(demo_project: Path) -> None:
    mine = demo_project / "task-a"
    result = build(demo_project, "--target-path", str(mine))
    assert result.exit_code == 0, debug(result)

    assert (mine / "manifest.json").is_file()
    assert set(results_of(mine)) == {DATASET_UID, MODEL_UID}
    assert (mine / LOCK_FILE).is_file()
    assert (mine / "run_logs").is_dir()
    assert cache_keys(mine / "datasets")  # the default cache moves with the state
    # nothing of this invocation's state landed in the project's own target/
    assert not (demo_project / "target" / "manifest.json").exists()
    assert not (demo_project / "target" / "run_results.json").exists()


def test_the_target_path_can_come_from_the_environment(demo_project: Path) -> None:
    """An orchestrator sets the environment of a task; it does not edit its argv."""
    mine = demo_project / "from-env"
    result = build(
        demo_project,
        env={"MBT_TARGET_PATH": str(mine), "MBT_DATASETS_PATH": str(demo_project / "cache")},
    )
    assert result.exit_code == 0, debug(result)
    assert (mine / "run_results.json").is_file()
    assert cache_keys(demo_project / "cache")
    assert not (mine / "datasets").exists()


def _hold_the_lock(target: str, ready, release) -> None:  # pragma: no cover - child
    with target_lock(Path(target), "build"):
        ready.set()
        release.wait(60)


def test_tasks_with_their_own_target_paths_do_not_contend(demo_project: Path) -> None:
    """The lock is one writer per target directory: another process holding one
    leaves every other directory free, and refuses only a second writer of its
    own. (A separate process, because the lock is reentrant within one.)"""
    busy, free = demo_project / "busy", demo_project / "free"
    context = multiprocessing.get_context("spawn")
    ready, release = context.Event(), context.Event()
    holder = context.Process(target=_hold_the_lock, args=(str(busy), ready, release))
    holder.start()
    try:
        assert ready.wait(60)
        elsewhere = build(demo_project, "--target-path", str(free))
        assert elsewhere.exit_code == 0, debug(elsewhere)
        same = build(demo_project, "--target-path", str(busy))
        assert same.exit_code == 1, debug(same)
        assert "already writing" in same.stderr
        assert "--target-path" in same.stderr
    finally:
        release.set()
        holder.join(60)


def test_a_target_path_cannot_contain_the_project(demo_project: Path) -> None:
    result = build(demo_project, "--target-path", str(demo_project))
    assert result.exit_code == 1, debug(result)
    assert "contains the project directory" in result.stderr


def test_a_target_path_must_be_a_local_directory(demo_project: Path) -> None:
    result = build(demo_project, "--target-path", "s3://bucket/state")
    assert result.exit_code == 1, debug(result)
    assert "--target-path must be a local directory" in result.stderr


def test_a_relative_target_path_is_relative_to_the_invocation_not_the_project(
    demo_project: Path, tmp_path_factory: pytest.TempPathFactory
) -> None:
    here = tmp_path_factory.mktemp("invocation")
    os.chdir(here)
    result = invoke(
        ["compile", "--project-dir", str(demo_project), "--anchor", ANCHOR, "--target-path", "out"]
    )
    assert result.exit_code == 0, debug(result)
    assert (here / "out" / "manifest.json").is_file()
    assert not (demo_project / "out").exists()


@pytest.mark.parametrize(
    "command",
    [
        ["compile"],
        ["parse", "--write-json-schema"],
        ["docs", "generate"],
    ],
)
def test_the_other_writing_commands_honor_the_target_path(
    demo_project: Path, command: list[str]
) -> None:
    mine = demo_project / "elsewhere"
    anchor = [] if command[0] == "parse" else ["--anchor", ANCHOR]
    result = invoke(
        [*command, "--project-dir", str(demo_project), *anchor, "--target-path", str(mine)]
    )
    assert result.exit_code == 0, debug(result)
    produced = {"compile": "manifest.json", "parse": "json-schemas", "docs": "docs/index.html"}
    assert (mine / produced[command[0]]).exists()
    assert not (demo_project / "target" / produced[command[0]]).exists()


def test_docs_serve_looks_where_docs_generate_wrote(demo_project: Path) -> None:
    mine = demo_project / "elsewhere"
    result = invoke(
        ["docs", "serve", "--project-dir", str(demo_project), "--target-path", str(mine)]
    )
    assert result.exit_code == 1, debug(result)
    assert f"no generated docs at {mine}" in result.stderr


def test_evaluate_and_monitor_and_predictions_honor_the_target_path(demo_project: Path) -> None:
    mine, cache = demo_project / "task", demo_project / "cache"
    flags = ["--target-path", str(mine), "--datasets-path", str(cache)]
    assert build(demo_project, *flags).exit_code == 0

    evaluated = invoke(
        [
            *["evaluate", "--project-dir", str(demo_project), "--anchor", ANCHOR],
            *["--model", "churn_model", *flags],
        ]
    )
    assert evaluated.exit_code == 0, debug(evaluated)
    assert results_of(mine)[MODEL_UID]["status"] == "success"
    assert (mine / "run_results.json").is_file()
    assert not (demo_project / "target" / "run_results.json").exists()

    monitored = invoke(
        [
            *["monitor", "--project-dir", str(demo_project), "--anchor", ANCHOR],
            *["--target-path", str(mine)],
        ]
    )
    assert monitored.exit_code == 0, debug(monitored)

    # predictions reuses the manifest the build left in THIS target path: it does
    # not rewrite it, and it does not fall back to compiling into the project's own
    built_at = (mine / "manifest.json").stat().st_mtime_ns
    listed = invoke(
        ["predictions", "ls", "--project-dir", str(demo_project), "--target-path", str(mine)]
    )
    assert listed.exit_code == 0, debug(listed)
    assert (mine / "manifest.json").stat().st_mtime_ns == built_at
    assert not (demo_project / "target" / "manifest.json").exists()
    shown = invoke(
        [
            *["predictions", "show", "nope", "--project-dir", str(demo_project)],
            *["--target-path", str(mine)],
        ]
    )
    assert shown.exit_code == 1, debug(shown)


# -- exact-node execution -----------------------------------------------------------


def test_a_model_task_reads_the_dataset_a_dataset_task_built(demo_project: Path) -> None:
    """The orchestrator shape: one task per node, each with its own state over
    one dataset cache. The model task builds nothing it was not asked to."""
    cache = demo_project / "cache"
    dataset_task, model_task = demo_project / "task-dataset", demo_project / "task-model"

    first = build(
        demo_project,
        *["--select", "churn_training", "--target-path", str(dataset_task)],
        *["--datasets-path", str(cache)],
    )
    assert first.exit_code == 0, debug(first)
    assert set(results_of(dataset_task)) == {DATASET_UID}
    [materialization] = cache_keys(cache)
    marker = materialization / "_SUCCESS"
    built_at = marker.stat().st_mtime_ns

    second = build(
        demo_project,
        *["--select", "churn_model", "--no-auto-materialize", "--target-path", str(model_task)],
        *["--datasets-path", str(cache)],
    )
    assert second.exit_code == 0, debug(second)
    only_the_model = results_of(model_task)
    assert set(only_the_model) == {MODEL_UID}
    assert only_the_model[MODEL_UID]["status"] == "success"
    assert only_the_model[MODEL_UID]["registration"] is not None
    assert marker.stat().st_mtime_ns == built_at  # read, not rebuilt
    assert not (model_task / "datasets").exists()


def test_an_exact_run_does_not_build_what_it_was_not_asked_to(demo_project: Path) -> None:
    mine = demo_project / "task"
    result = build(
        demo_project,
        *["--select", "churn_model", "--no-auto-materialize"],
        "--target-path",
        str(mine),
    )
    assert result.exit_code == 1, debug(result)  # a hard error, not a quality verdict
    node = results_of(mine)[MODEL_UID]
    assert node["status"] == "error"
    assert "dataset 'churn_training' is not materialized in" in node["message"]
    assert set(results_of(mine)) == {MODEL_UID}
    assert not (mine / "datasets").exists()


def test_without_the_flag_the_model_still_builds_its_dataset(demo_project: Path) -> None:
    """ADR-13 is untouched: the default stays cold-runner friendly."""
    mine = demo_project / "task"
    result = build(demo_project, "--select", "churn_model", "--target-path", str(mine))
    assert result.exit_code == 0, debug(result)
    assert set(results_of(mine)) == {DATASET_UID, MODEL_UID}


def test_the_flags_are_accepted_everywhere_a_plan_is_made() -> None:
    """Read from the command's own parameters, not from its rendered --help: rich
    colours the help in CI and splits an option name across escape codes."""
    import typer

    from mbt.cli.main import app

    commands = typer.main.get_command(app).commands  # type: ignore[attr-defined]
    for command in ("run", "build", "test", "score"):
        flags = {opt for param in commands[command].params for opt in param.opts}
        assert {"--no-auto-materialize", "--datasets-path", "--target-path"} <= flags, command


def test_the_planner_does_not_pull_datasets_in_unless_asked(
    demo_project: Path, fake_registry: AdapterRegistry
) -> None:
    ctx = make_execution_context(demo_project, fake_registry)
    pulled = plan_execution(ctx.manifest, ["churn_model"], None)
    assert pulled.execution_set == {MODEL_UID, DATASET_UID}
    assert pulled.auto_materialized == {DATASET_UID}

    exact = plan_execution(ctx.manifest, ["churn_model"], None, auto_materialize=False)
    assert exact.execution_set == exact.selected == {MODEL_UID}
    assert exact.auto_materialized == frozenset()
    # selecting the dataset too still builds it: only the *unselected* are not pulled in
    both = plan_execution(
        ctx.manifest, ["churn_model", "churn_training"], None, auto_materialize=False
    )
    assert both.execution_set == {MODEL_UID, DATASET_UID}


def test_a_missing_materialization_says_how_to_make_one(
    demo_project: Path, fake_registry: AdapterRegistry
) -> None:
    ctx = make_execution_context(demo_project, fake_registry)
    with pytest.raises(StateError, match=r"not materialized in .*datasets") as info:
        ctx.dataset_handle(DATASET_UID)
    assert info.value.hint is not None
    assert "mbt build --select churn_training" in info.value.hint
    assert "--datasets-path" in info.value.hint


def test_a_cached_handle_is_remembered_after_the_first_lookup(
    demo_project: Path, fake_registry: AdapterRegistry
) -> None:
    builder = make_execution_context(demo_project, fake_registry)
    assert DatasetRunner(builder).run(DATASET_UID).status == "success"

    reader = make_execution_context(demo_project, fake_registry)
    first = reader.dataset_handle(DATASET_UID)
    assert reader.dataset_handle(DATASET_UID) is first


# -- a shared cache is safe to build into -------------------------------------------


def test_a_builder_that_had_to_wait_reuses_the_winners_materialization(
    demo_project: Path, fake_registry: AdapterRegistry, monkeypatch: pytest.MonkeyPatch
) -> None:
    """Two invocations racing for one key over one cache: the loser must not
    empty the directory the winner is filling (a build wipes it first). It
    waits, says so, and then reads what the winner left."""
    winner_ctx = make_execution_context(demo_project, fake_registry)
    loser_ctx = make_execution_context(demo_project, fake_registry)
    node = winner_ctx.manifest.nodes[DATASET_UID]
    output_dir = winner_ctx.dataset_dir(node)
    holding, loser_is_waiting = threading.Event(), threading.Event()

    def winner() -> None:
        with materialization_lock(output_dir):
            holding.set()
            assert loser_is_waiting.wait(30)
            DatasetRunner(winner_ctx)._build(
                node, DatasetSpec.model_validate(node.config), output_dir
            )

    builds: list[str] = []
    real_build = DatasetRunner._build

    def counting_build(self, node, spec, output_dir):
        builds.append("loser" if self.ctx is loser_ctx else "winner")
        return real_build(self, node, spec, output_dir)

    monkeypatch.setattr(DatasetRunner, "_build", counting_build)

    thread = threading.Thread(target=winner)
    with recording_bus() as sink:
        thread.start()
        assert holding.wait(30)

        class ReleaseWhenWaiting:
            def write(self, event) -> None:
                if "waiting for it" in str(getattr(event, "message", "")):
                    loser_is_waiting.set()

        from mbt.events import get_bus

        get_bus().add_sink(ReleaseWhenWaiting())
        try:
            result = DatasetRunner(loser_ctx).run(DATASET_UID)
        finally:
            loser_is_waiting.set()
            thread.join(30)

    assert result.status == "success", result.message
    assert builds == ["winner"]  # the loser never built
    messages = sink.messages()
    assert any(
        f"another process is materializing {node.name} ({output_dir.name}); waiting for it" in m
        for m in messages
    )
    assert any(f"cache hit ({output_dir.name})" in m for m in messages)


# -- clean --------------------------------------------------------------------------


def test_clean_removes_a_target_path_mbt_has_used(demo_project: Path) -> None:
    mine = demo_project / "task"
    assert build(demo_project, "--target-path", str(mine)).exit_code == 0
    own = demo_project / "target"
    own.mkdir(exist_ok=True)

    cleaned = invoke(["clean", "--project-dir", str(demo_project), "--target-path", str(mine)])
    assert cleaned.exit_code == 0, debug(cleaned)
    assert f"removed {mine}" in cleaned.output
    assert not mine.exists()
    assert own.is_dir()  # the project's own target/ was not what was asked for


def test_clean_refuses_a_directory_mbt_never_wrote_to(demo_project: Path) -> None:
    """`mbt clean --target-path ~` must not be a way to delete a home directory."""
    stranger = demo_project / "precious"
    stranger.mkdir()
    (stranger / "thesis.tex").write_text("irreplaceable")

    refused = invoke(["clean", "--project-dir", str(demo_project), "--target-path", str(stranger)])
    assert refused.exit_code == 1, debug(refused)
    assert "refusing to delete" in refused.stderr
    assert (stranger / "thesis.tex").read_text() == "irreplaceable"


def test_clean_of_a_target_path_that_does_not_exist_is_a_no_op(demo_project: Path) -> None:
    missing = demo_project / "never-made"
    cleaned = invoke(["clean", "--project-dir", str(demo_project), "--target-path", str(missing)])
    assert cleaned.exit_code == 0, debug(cleaned)
    assert f"nothing to clean at {missing}" in cleaned.output
