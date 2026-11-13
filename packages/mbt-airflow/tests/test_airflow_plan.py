"""plan_tasks: a manifest and a selector become the tasks of a DAG (ADR-33)."""

import pytest
from mbt_airflow import TaskSpec, plan_tasks
from mbt_airflow.plan import COMPILE

from mbt.artifacts.manifest import Manifest
from mbt.exceptions import ConfigError, MbtError


def by_id(specs: list[TaskSpec]) -> dict[str, TaskSpec]:
    return {spec.task_id: spec for spec in specs}


def test_build_draws_every_dataset_and_model_under_one_compile_root(manifest: Manifest) -> None:
    specs = plan_tasks(manifest, "build", phases=False)
    tasks = by_id(specs)

    assert specs[0] == TaskSpec(COMPILE, "", "compile", "compile", ())
    assert set(tasks) == {
        COMPILE,
        "churn_oot_set",
        "churn_training_set",
        "upsell_training_set",
        "churn_classifier",
        "churn_classifier_deep",
        "churn_classifier_oot",
        "upsell_classifier",
    }
    # roots hang off the compile task; a model waits for its dataset, not for compile
    assert tasks["churn_training_set"].upstream == (COMPILE,)
    assert tasks["churn_classifier"].upstream == ("churn_training_set",)
    assert tasks["churn_classifier_oot"].upstream == ("churn_oot_set",)
    assert tasks["churn_classifier"].uid == "model.churn_demo.churn_classifier"
    assert {s.kind for s in specs} == {"compile", "dataset", "model"}


def test_every_task_is_listed_after_the_tasks_it_waits_for(manifest: Manifest) -> None:
    seen: set[str] = set()
    for spec in plan_tasks(manifest, "build"):
        assert set(spec.upstream) <= seen, spec
        seen.add(spec.task_id)


def test_a_model_selection_is_drawn_with_the_datasets_it_needs(manifest: Manifest) -> None:
    """Every node task runs --no-auto-materialize (ADR-32), so a model's dataset has
    to be a task of its own or the model finds an empty cache."""
    specs = plan_tasks(manifest, "build", ["tag:weekly"], phases=False)
    assert {s.task_id for s in specs} == {
        COMPILE,
        "churn_training_set",
        "upsell_training_set",
        "churn_classifier",
        "upsell_classifier",
    }


def test_exclude_subtracts_from_the_selection(manifest: Manifest) -> None:
    specs = plan_tasks(manifest, "build", ["tag:weekly"], ["upsell_classifier"], phases=False)
    assert {s.task_id for s in specs} == {COMPILE, "churn_training_set", "churn_classifier"}


def test_score_draws_scoring_nodes_without_an_edge_to_the_model(manifest: Manifest) -> None:
    """The manifest has scoring -> model, but a scoring node reads the registry's
    champion at run time (ADR-20): drawing the edge would put every daily score
    behind a weekly retrain."""
    assert (
        "model.churn_demo.churn_classifier"
        in manifest.nodes["scoring.churn_demo.retention_scoring"].depends_on
    )
    specs = plan_tasks(manifest, "score", ["tag:daily"])
    assert specs[1:] == [
        TaskSpec(
            "retention_scoring",
            "scoring.churn_demo.retention_scoring",
            "scoring",
            "score",
            (COMPILE,),
            node="retention_scoring",
        )
    ]


def test_monitor_draws_the_same_scoring_nodes(manifest: Manifest) -> None:
    specs = plan_tasks(manifest, "monitor", ["tag:daily"])
    assert [(s.task_id, s.command) for s in specs] == [
        (COMPILE, "compile"),
        ("retention_scoring", "monitor"),
    ]


@pytest.mark.parametrize("command", ["run", "test"])
def test_run_and_test_share_builds_node_universe(manifest: Manifest, command: str) -> None:
    assert {s.task_id for s in plan_tasks(manifest, command, phases=False)} == {
        s.task_id for s in plan_tasks(manifest, "build", phases=False)
    }
    assert {s.command for s in plan_tasks(manifest, command)[1:]} == {command}


def test_a_command_with_no_nodes_to_draw_is_refused(manifest: Manifest) -> None:
    with pytest.raises(ConfigError, match="cannot render the 'evaluate' command") as info:
        plan_tasks(manifest, "evaluate")
    assert "score" in (info.value.hint or "")


def test_a_selection_the_command_never_executes_is_refused(manifest: Manifest) -> None:
    """`tag:daily` selects only a scoring node, which `build` does not run: a DAG of
    just a compile task would pass green having run nothing."""
    with pytest.raises(ConfigError, match=r"matched no dataset/model nodes for 'mbt build'"):
        plan_tasks(manifest, "build", ["tag:daily"])


def test_a_selector_that_matches_nothing_is_the_planners_own_error(manifest: Manifest) -> None:
    with pytest.raises(MbtError, match="nope"):
        plan_tasks(manifest, "build", ["nope"])


def test_state_selectors_cannot_draw_a_dag(manifest: Manifest) -> None:
    """A DAG is drawn before any run, so there is no reference state to compare to."""
    with pytest.raises(MbtError, match="--state"):
        plan_tasks(manifest, "build", ["state:modified"])


def test_two_nodes_sharing_a_name_would_share_a_task_id(manifest: Manifest) -> None:
    payload = manifest.model_dump(mode="json")
    payload["nodes"]["model.churn_demo.churn_classifier"]["name"] = "churn_training_set"
    # a new Manifest, not a mutated one: its derived views memoize (it is immutable)
    clash = Manifest.model_validate(payload)
    with pytest.raises(ConfigError, match="share a name") as info:
        plan_tasks(clash, "build", ["churn_training_set"])
    assert "rename" in (info.value.hint or "")


# -- phases: a node is more than one task (ADR-35) -------------------------------------


def test_build_draws_each_node_as_the_steps_mbt_can_run_alone(manifest: Manifest) -> None:
    tasks = by_id(plan_tasks(manifest, "build", ["tag:weekly"]))
    assert set(tasks) == {
        COMPILE,
        "churn_training_set.materialize",
        "churn_training_set.checks",
        "upsell_training_set.materialize",
        "upsell_training_set.checks",
        "churn_classifier.train",
        "churn_classifier.gate",
        "churn_classifier.register",
        "upsell_classifier.train",
        "upsell_classifier.gate",
        "upsell_classifier.register",
    }
    # the steps of a node run in order, from the compile task through to the registry
    assert tasks["churn_training_set.materialize"].upstream == (COMPILE,)
    assert tasks["churn_training_set.checks"].upstream == ("churn_training_set.materialize",)
    assert tasks["churn_classifier.gate"].upstream == ("churn_classifier.train",)
    assert tasks["churn_classifier.register"].upstream == ("churn_classifier.gate",)


def test_a_node_waits_for_the_last_step_of_the_nodes_it_reads(manifest: Manifest) -> None:
    """A model trains on a dataset only once the dataset's checks have passed."""
    tasks = by_id(plan_tasks(manifest, "build", ["tag:weekly"]))
    assert tasks["churn_classifier.train"].upstream == ("churn_training_set.checks",)
    assert tasks["upsell_classifier.train"].upstream == ("upsell_training_set.checks",)


def test_each_step_knows_its_node_and_its_phase(manifest: Manifest) -> None:
    tasks = by_id(plan_tasks(manifest, "run", ["churn_classifier"]))
    train = tasks["churn_classifier.train"]
    assert (train.node, train.phase, train.kind, train.command) == (
        "churn_classifier",
        "train",
        "model",
        "run",
    )
    assert train.uid == "model.churn_demo.churn_classifier"
    assert tasks[COMPILE].phase is None and tasks[COMPILE].node == ""


@pytest.mark.parametrize("command", ["score", "monitor", "test"])
def test_commands_without_phases_are_drawn_one_task_per_node(
    manifest: Manifest, command: str
) -> None:
    """Only run and build train and build; the others have a single step per node."""
    selection = ["tag:daily"] if command != "test" else None
    assert plan_tasks(manifest, command, selection) == plan_tasks(
        manifest, command, selection, phases=False
    )


def test_the_undecomposed_graph_is_still_available(manifest: Manifest) -> None:
    assert all(s.phase is None for s in plan_tasks(manifest, "build", phases=False))
