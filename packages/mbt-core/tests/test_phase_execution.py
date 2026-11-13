"""One step of a node per invocation (ADR-35): the phases of a dataset and of a model.

These drive the real CLI over the fake adapters the way an orchestrator would: every
phase its own process, a model's phases over one ``--target-path`` (they share a state
file there), a dataset's over one ``--datasets-path``. The claim under test is that
the phases, run in order, leave exactly what the whole node leaves.
"""

import json
import re
from pathlib import Path

import pytest
from cli_unit_helpers import (  # noqa: F401 - autouse fixture import
    ANCHOR,
    cli_process_state,
    debug,
    invoke,
)

from mbt.exceptions import ConfigError
from mbt.execute.phases import ModelPhaseState, check_phase
from mbt.state.layout import TargetLayout

DATASET_UID = "dataset.demo.churn_training"
MODEL_UID = "model.demo.churn_model"


def run(project: Path, *args: str, command: str = "run"):
    return invoke([command, "--project-dir", str(project), "--anchor", ANCHOR, *args])


def phase(project: Path, which: str, select: str, task: Path, cache: Path, *extra: str):
    """One phase of one node, the way an orchestrator task runs it."""
    return run(
        project,
        *["--phase", which, "--select", select, "--target-path", str(task)],
        *["--datasets-path", str(cache), *extra],
    )


def results_of(target: Path) -> dict:
    return json.loads((target / "run_results.json").read_text())


def node_of(target: Path, uid: str) -> dict:
    return next(r for r in results_of(target)["results"] if r["unique_id"] == uid)


def cache_keys(cache: Path) -> list[Path]:
    """The materializations in the cache (a build's lock file sits beside its directory)."""
    return sorted(p for p in (cache / "churn_training").iterdir() if p.is_dir())


def registry_versions(project: Path) -> list[dict]:
    path = project / "target" / "fake_registry" / "churn_model.json"
    return json.loads(path.read_text()) if path.is_file() else []


@pytest.fixture()
def tasks(demo_project: Path) -> tuple[Path, Path, Path]:
    """(dataset task's target, model task's target, the shared dataset cache)."""
    return demo_project / "task-dataset", demo_project / "task-model", demo_project / "cache"


def materialized(demo_project: Path, tasks: tuple[Path, Path, Path]) -> None:
    dataset_task, _model_task, cache = tasks
    done = phase(demo_project, "materialize", "churn_training", dataset_task, cache)
    assert done.exit_code == 0, debug(done)


# -- a dataset: materialize, then checks ------------------------------------------------


def test_materialize_builds_the_dataset_and_runs_none_of_its_checks(
    demo_project: Path, tasks: tuple[Path, Path, Path]
) -> None:
    dataset_task, _model_task, cache = tasks
    materialized(demo_project, tasks)

    node = node_of(dataset_task, DATASET_UID)
    assert node["status"] == "success"
    assert node["tests"] == []  # the checks are the next phase's
    assert results_of(dataset_task)["metadata"]["phase"] == "materialize"
    [key] = cache_keys(cache)
    assert (key / "_SUCCESS").is_file()


def test_checks_read_what_materialize_built_and_build_nothing(
    demo_project: Path, tasks: tuple[Path, Path, Path]
) -> None:
    dataset_task, _model_task, cache = tasks
    materialized(demo_project, tasks)
    [key] = cache_keys(cache)
    built_at = (key / "_SUCCESS").stat().st_mtime_ns

    done = phase(demo_project, "checks", "churn_training", dataset_task, cache)
    assert done.exit_code == 0, debug(done)
    node = node_of(dataset_task, DATASET_UID)
    assert node["status"] == "success"
    assert node["tests"]  # the declared checks ran
    assert (key / "_SUCCESS").stat().st_mtime_ns == built_at


def test_checks_with_no_materialization_is_an_error_that_names_the_phase_to_run(
    demo_project: Path, tasks: tuple[Path, Path, Path]
) -> None:
    dataset_task, _model_task, cache = tasks
    done = phase(demo_project, "checks", "churn_training", dataset_task, cache)
    assert done.exit_code == 1, debug(done)
    message = node_of(dataset_task, DATASET_UID)["message"]
    assert "is not materialized" in message
    assert not cache.exists()  # an exact phase builds nothing it was not asked to


def test_the_dataset_phases_leave_what_the_whole_dataset_leaves(
    demo_project: Path, tasks: tuple[Path, Path, Path]
) -> None:
    dataset_task, _model_task, cache = tasks
    whole = demo_project / "whole"
    assert (
        run(demo_project, "--select", "churn_training", "--target-path", str(whole)).exit_code == 0
    )
    materialized(demo_project, tasks)
    assert phase(demo_project, "checks", "churn_training", dataset_task, cache).exit_code == 0

    whole_node = node_of(whole, DATASET_UID)
    split_node = node_of(dataset_task, DATASET_UID)
    assert split_node["status"] == whole_node["status"]
    assert [t["name"] for t in split_node["tests"]] == [t["name"] for t in whole_node["tests"]]


# -- a model: train, gate, register -------------------------------------------------------


def test_the_model_phases_leave_what_the_whole_model_leaves(
    demo_project: Path, tasks: tuple[Path, Path, Path]
) -> None:
    _dataset_task, model_task, cache = tasks
    materialized(demo_project, tasks)

    trained = phase(demo_project, "train", "churn_model", model_task, cache)
    assert trained.exit_code == 0, debug(trained)
    after_train = node_of(model_task, MODEL_UID)
    assert after_train["status"] == "success"
    assert after_train["artifact"] and after_train["metrics"]
    assert after_train["gates"] == [] and after_train["registration"] is None  # not judged yet
    assert results_of(model_task)["metadata"]["phase"] == "train"
    assert registry_versions(demo_project) == []  # nothing is registered by training alone

    gated = phase(demo_project, "gate", "churn_model", model_task, cache)
    assert gated.exit_code == 0, debug(gated)
    after_gate = node_of(model_task, MODEL_UID)
    assert [g["passed"] for g in after_gate["gates"]] == [True]
    assert after_gate["registration"] is None
    assert registry_versions(demo_project) == []  # judged is not registered

    registered = phase(demo_project, "register", "churn_model", model_task, cache)
    assert registered.exit_code == 0, debug(registered)
    final = node_of(model_task, MODEL_UID)
    [version] = registry_versions(demo_project)
    assert final["registration"]["version"] == version["version"]
    assert version["stage"] == final["registration"]["stage"]

    # The same model, trained whole, ends in the same place.
    whole = demo_project / "whole"
    assert run(demo_project, "--select", "churn_model", "--target-path", str(whole)).exit_code == 0
    whole_node = node_of(whole, MODEL_UID)
    assert final["status"] == whole_node["status"] == "success"
    assert final["metrics"] == whole_node["metrics"]
    assert final["gates"] == whole_node["gates"]
    assert final["registration"]["stage"] == whole_node["registration"]["stage"]
    assert len(registry_versions(demo_project)) == 2  # the whole run registered its own


def test_a_failed_gate_is_the_gate_phases_verdict_and_nothing_registers(
    demo_project: Path, tasks: tuple[Path, Path, Path]
) -> None:
    _dataset_task, model_task, cache = tasks
    impossible = ["--vars", "default_threshold: 0.99"]
    assert (
        phase(demo_project, "materialize", "churn_training", tasks[0], cache, *impossible).exit_code
        == 0
    )
    assert (
        phase(demo_project, "train", "churn_model", model_task, cache, *impossible).exit_code == 0
    )

    gated = phase(demo_project, "gate", "churn_model", model_task, cache, *impossible)
    assert gated.exit_code == 2, debug(gated)  # the quality verdict, from this phase alone
    node = node_of(model_task, MODEL_UID)
    assert node["status"] == "gate_failed"
    assert "pr_auc" in node["message"]

    again = phase(demo_project, "register", "churn_model", model_task, cache, *impossible)
    assert again.exit_code == 1, debug(again)  # a hard error: there is nothing to register
    assert "failed its gates" in node_of(model_task, MODEL_UID)["message"]
    assert registry_versions(demo_project) == []


def test_register_needs_a_training_state_and_a_judgement(
    demo_project: Path, tasks: tuple[Path, Path, Path]
) -> None:
    _dataset_task, model_task, cache = tasks
    materialized(demo_project, tasks)

    without_training = phase(demo_project, "gate", "churn_model", model_task, cache)
    assert without_training.exit_code == 1, debug(without_training)
    assert "no training state" in node_of(model_task, MODEL_UID)["message"]

    assert phase(demo_project, "train", "churn_model", model_task, cache).exit_code == 0
    unjudged = phase(demo_project, "register", "churn_model", model_task, cache)
    assert unjudged.exit_code == 1, debug(unjudged)
    assert "has not been judged" in node_of(model_task, MODEL_UID)["message"]
    assert registry_versions(demo_project) == []


def test_register_is_idempotent(demo_project: Path, tasks: tuple[Path, Path, Path]) -> None:
    """Airflow retries a task; a retried register must not register the artifact twice."""
    _dataset_task, model_task, cache = tasks
    materialized(demo_project, tasks)
    for which in ("train", "gate", "register"):
        assert phase(demo_project, which, "churn_model", model_task, cache).exit_code == 0
    [first] = registry_versions(demo_project)

    again = phase(demo_project, "register", "churn_model", model_task, cache)
    assert again.exit_code == 0, debug(again)
    assert registry_versions(demo_project) == [first]
    assert node_of(model_task, MODEL_UID)["registration"]["version"] == first["version"]


def test_a_register_that_died_after_the_registry_took_the_version_finishes_it(
    demo_project: Path, tasks: tuple[Path, Path, Path]
) -> None:
    """The window a retry exists for: the registry has the version, the stage is not set."""
    _dataset_task, model_task, cache = tasks
    materialized(demo_project, tasks)
    for which in ("train", "gate", "register"):
        assert phase(demo_project, which, "churn_model", model_task, cache).exit_code == 0

    state_path = TargetLayout.for_project(demo_project, str(model_task)).phase_state("churn_model")
    state = ModelPhaseState.model_validate_json(state_path.read_text())
    assert state.registered_version == "1"
    state.registration = None  # as if the process died before it recorded the stage
    state_path.write_text(state.model_dump_json())
    registry_file = demo_project / "target" / "fake_registry" / "churn_model.json"
    versions = json.loads(registry_file.read_text())
    versions[0]["stage"] = None
    registry_file.write_text(json.dumps(versions))

    done = phase(demo_project, "register", "churn_model", model_task, cache)
    assert done.exit_code == 0, debug(done)
    [only] = registry_versions(demo_project)  # not a second version
    assert only["stage"] is not None  # the stage the dead process never set
    assert node_of(model_task, MODEL_UID)["registration"]["version"] == "1"


def test_a_model_phase_reads_the_dataset_cache_and_never_builds_it(
    demo_project: Path, tasks: tuple[Path, Path, Path]
) -> None:
    _dataset_task, model_task, cache = tasks
    done = phase(demo_project, "train", "churn_model", model_task, cache)
    assert done.exit_code == 1, debug(done)
    assert "is not materialized" in node_of(model_task, MODEL_UID)["message"]
    assert not cache.exists()


# -- states that do not belong ------------------------------------------------------------


def test_phases_of_a_different_plan_do_not_mix(
    demo_project: Path, tasks: tuple[Path, Path, Path]
) -> None:
    """A different --vars is a different manifest: its gate must not judge this training."""
    _dataset_task, model_task, cache = tasks
    materialized(demo_project, tasks)
    assert phase(demo_project, "train", "churn_model", model_task, cache).exit_code == 0

    other = phase(
        demo_project, "gate", "churn_model", model_task, cache, "--vars", "default_threshold: 0.1"
    )
    assert other.exit_code == 1, debug(other)
    message = node_of(model_task, MODEL_UID)["message"]
    assert "left by a different plan" in message


@pytest.mark.parametrize(
    ("damage", "expected"),
    [
        (lambda s: "{not json", "is not readable"),
        (lambda s: s.replace('"state_version":1', '"state_version":99'), "is version 99"),
    ],
)
def test_a_damaged_state_says_how_to_make_a_new_one(
    demo_project: Path, tasks: tuple[Path, Path, Path], damage, expected: str
) -> None:
    _dataset_task, model_task, cache = tasks
    materialized(demo_project, tasks)
    assert phase(demo_project, "train", "churn_model", model_task, cache).exit_code == 0
    path = TargetLayout.for_project(demo_project, str(model_task)).phase_state("churn_model")
    path.write_text(damage(json.dumps(json.loads(path.read_text()), separators=(",", ":"))))

    done = phase(demo_project, "gate", "churn_model", model_task, cache)
    assert done.exit_code == 1, debug(done)
    assert expected in node_of(model_task, MODEL_UID)["message"]


# -- what a phase may be asked of ---------------------------------------------------------


def test_a_phase_is_only_for_run_and_build(
    demo_project: Path, tasks: tuple[Path, Path, Path]
) -> None:
    for command in ("test", "score"):
        done = run(demo_project, "--phase", "train", command=command)
        assert done.exit_code == 1, debug(done)
        assert "applies to 'run' and 'build'" in done.output + done.stderr


def test_a_phase_must_exist_and_fit_every_selected_node(
    demo_project: Path, tasks: tuple[Path, Path, Path]
) -> None:
    unknown = run(demo_project, "--phase", "polish", "--select", "churn_model")
    assert unknown.exit_code == 1, debug(unknown)
    assert "unknown phase 'polish'" in unknown.output + unknown.stderr

    # a model phase over a selection that includes the dataset: silence would let an
    # orchestrator believe the dataset had been built
    mixed = run(demo_project, "--phase", "train", "--select", "churn_model churn_training")
    assert mixed.exit_code == 1, debug(mixed)
    assert "is a model phase" in mixed.output + mixed.stderr

    wrong = run(demo_project, "--phase", "materialize", "--select", "churn_model")
    assert wrong.exit_code == 1, debug(wrong)
    assert "is a dataset phase" in wrong.output + wrong.stderr


def test_check_phase_is_the_one_place_that_decides(demo_project: Path) -> None:
    """The same rules, asked directly of the function the orchestrator calls."""
    from mbt.artifacts.manifest import read_manifest

    done = invoke(
        [
            "compile",
            "--project-dir",
            str(demo_project),
            "--anchor",
            ANCHOR,
            "--target-path",
            str(demo_project / "c"),
        ]
    )
    assert done.exit_code == 0, debug(done)
    manifest = read_manifest(demo_project / "c" / "manifest.json")
    check_phase("build", "train", manifest, [MODEL_UID])  # fits: no error
    check_phase("run", "checks", manifest, [DATASET_UID])
    with pytest.raises(ConfigError, match="is a model phase"):
        check_phase("build", "train", manifest, [DATASET_UID, MODEL_UID])


def test_a_model_with_no_registration_still_ends_cleanly_at_register(
    demo_project: Path, tasks: tuple[Path, Path, Path]
) -> None:
    """Nothing to register is not an error: the phase closes the node out all the same."""
    spec = demo_project / "models" / "churn_model.yml"
    text = spec.read_text()
    assert "registration:" in text
    spec.write_text(re.sub(r" *registration:\n *name: churn_model\n", "", text))
    assert "registration:" not in spec.read_text()
    _dataset_task, model_task, cache = tasks
    materialized(demo_project, tasks)
    for which in ("train", "gate", "register"):
        done = phase(demo_project, which, "churn_model", model_task, cache)
        assert done.exit_code == 0, debug(done)
    node = node_of(model_task, MODEL_UID)
    assert node["status"] == "success" and node["registration"] is None
    assert registry_versions(demo_project) == []


def test_a_phase_reports_the_nodes_time_so_far_not_only_its_own(
    demo_project: Path, tasks: tuple[Path, Path, Path]
) -> None:
    """A dashboard reads the last result a node pushed, and the last phase is the registry write."""
    _dataset_task, model_task, cache = tasks
    materialized(demo_project, tasks)
    seen = []
    for which in ("train", "gate", "register"):
        assert phase(demo_project, which, "churn_model", model_task, cache).exit_code == 0
        seen.append(node_of(model_task, MODEL_UID)["execution_time_s"])
    assert seen == sorted(seen) and seen[0] > 0  # never goes down: it is a running total
    state_path = TargetLayout.for_project(demo_project, str(model_task)).phase_state("churn_model")
    assert ModelPhaseState.model_validate_json(state_path.read_text()).elapsed_s == seen[-1]


def test_a_phase_that_failed_before_leaving_a_state_has_no_time_to_add(
    demo_project: Path, tasks: tuple[Path, Path, Path]
) -> None:
    _dataset_task, model_task, cache = tasks
    done = phase(demo_project, "train", "churn_model", model_task, cache)  # no dataset: it fails
    assert done.exit_code == 1, debug(done)
    assert node_of(model_task, MODEL_UID)["status"] == "error"
    assert (
        not TargetLayout.for_project(demo_project, str(model_task))
        .phase_state("churn_model")
        .exists()
    )
