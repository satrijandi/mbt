"""Where an invocation writes (ADR-32): the layout and how the CLI feeds it."""

import os
import re
from pathlib import Path

import pytest
from cli_unit_helpers import cli_process_state  # noqa: F401 - autouse fixture import

from mbt.cli.common import CLIContext
from mbt.exceptions import ConfigError
from mbt.state.layout import TargetLayout


def test_the_default_layout_is_the_projects_own_target(tmp_path: Path) -> None:
    layout = TargetLayout.for_project(tmp_path)
    assert layout.root == tmp_path.resolve() / "target"
    assert layout.datasets == layout.root / "datasets"
    assert not layout.custom_root
    assert layout.manifest == layout.root / "manifest.json"
    assert layout.run_results == layout.root / "run_results.json"
    assert layout.gate_history == layout.root / "gate_history.json"
    assert layout.docs == layout.root / "docs"
    assert layout.json_schemas == layout.root / "json-schemas"
    assert layout.run_logs("r1") == layout.root / "run_logs" / "r1"
    assert layout.dataset_dir("panel", "k1") == layout.root / "datasets" / "panel" / "k1"
    assert (
        layout.scoring_input_dir("score", "k1") == layout.root / "scoring_inputs" / "score" / "k1"
    )


def test_a_target_path_moves_the_state_and_the_default_cache_with_it(tmp_path: Path) -> None:
    layout = TargetLayout.for_project(tmp_path / "proj", tmp_path / "task-a")
    assert layout.root == tmp_path.resolve() / "task-a"
    assert layout.datasets == layout.root / "datasets"
    assert layout.custom_root


def test_the_dataset_cache_can_live_apart_from_the_state(tmp_path: Path) -> None:
    """The point of the split: every task its own state, one shared cache."""
    shared = tmp_path / "shared" / "datasets"
    first = TargetLayout.for_project(tmp_path / "proj", tmp_path / "a", shared)
    second = TargetLayout.for_project(tmp_path / "proj", tmp_path / "b", shared)
    assert first.root != second.root
    assert first.datasets == second.datasets == shared.resolve()
    assert first.dataset_dir("panel", "k") == second.dataset_dir("panel", "k")
    # ...while scoring batches, which are never reused, stay with the state
    assert first.scoring_input_dir("s", "k") != second.scoring_input_dir("s", "k")


def test_a_relative_override_is_relative_to_the_project(tmp_path: Path) -> None:
    layout = TargetLayout.for_project(tmp_path, "out/state", "out/cache")
    assert layout.root == tmp_path.resolve() / "out" / "state"
    assert layout.datasets == tmp_path.resolve() / "out" / "cache"


@pytest.mark.parametrize("flag", ["target", "datasets"])
def test_an_object_store_location_is_refused(tmp_path: Path, flag: str) -> None:
    kwargs = {f"{flag}_path": "s3://bucket/path"}
    with pytest.raises(ConfigError, match=rf"--{flag}-path must be a local directory"):
        TargetLayout.for_project(tmp_path, **kwargs)


@pytest.mark.parametrize("which", ["the project itself", "an ancestor of it"])
def test_a_target_path_may_not_contain_the_project(tmp_path: Path, which: str) -> None:
    """mbt rewrites what is under it, and `mbt clean` would delete the project."""
    project = tmp_path / "project"
    project.mkdir()
    chosen = project if which == "the project itself" else tmp_path
    with pytest.raises(ConfigError, match="contains the project directory"):
        TargetLayout.for_project(project, chosen)


# -- the CLI feeds it ------------------------------------------------------------


def test_cli_paths_are_relative_to_where_the_user_ran_mbt(tmp_path: Path) -> None:
    """The coordinator chdirs into the project, but a path TYPED on the command
    line stays shell-relative (resolve_cli_path) - this one included."""
    project = tmp_path / "proj"
    project.mkdir()
    here = tmp_path / "here"
    here.mkdir()
    os.chdir(here)

    cli = CLIContext.enter(
        Path("../proj"), target_path="state", datasets_path="../cache"
    )  # the cwd is now the project

    layout = cli.layout
    assert layout.root == here.resolve() / "state"
    assert layout.datasets == tmp_path.resolve() / "cache"
    options = cli.invocation("build")
    assert options.target_path == str(here.resolve() / "state")
    assert options.datasets_path == str(tmp_path.resolve() / "cache")
    assert options.layout == layout


def test_invocation_options_default_to_the_project_target(tmp_path: Path) -> None:
    from mbt.execute.orchestrator import InvocationOptions

    options = InvocationOptions(command="build", project_dir=tmp_path)
    assert options.layout == TargetLayout.for_project(tmp_path)
    assert options.auto_materialize


def test_no_module_spells_the_target_directory_but_the_layout() -> None:
    """ADR-32: ``project_dir / "target" / ...`` at twenty sites is what made a
    per-task target path a twenty-site change. A new one would quietly ignore
    ``--target-path``, and the task that used it would write into the project."""
    import mbt

    spelled = re.compile(r"""["']target["']\s*/|/\s*["']target["']""")
    offenders = [
        f"{path.relative_to(Path(mbt.__file__).parent)}:{number}"
        for path in sorted(Path(mbt.__file__).parent.rglob("*.py"))
        if path.name != "layout.py" and "_scaffold" not in path.parts
        for number, line in enumerate(path.read_text().splitlines(), start=1)
        if spelled.search(line)
    ]
    assert not offenders, f"spell the target directory through TargetLayout: {offenders}"
