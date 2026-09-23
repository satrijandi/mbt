"""Direct tests for the CLI's composition root and command seams (A-2).

These call ``CLIContext.enter`` and the ``*_cmd`` functions with real inputs,
with no ``CliRunner`` anywhere. Before v5 the composition root was never called
directly by any test in the repo, and ``ls``/``show``/``clean`` held real
behaviour reachable only by invoking the runner.
"""

from pathlib import Path

import pytest
from cli_unit_helpers import cli_process_state  # noqa: F401 - autouse fixture
from core_helpers import write

from mbt.cli.common import CLIContext
from mbt.cli.inspect import clean_target_cmd, ls_cmd, show_cmd
from mbt.exceptions import ConfigError

ANCHOR = "2026-07-01T00:00:00Z"


def _project(root: Path) -> Path:
    """The smallest project that compiles: one source, one dataset."""
    import pyarrow as pa
    import pyarrow.parquet as pq

    data = root / "data"
    data.mkdir(parents=True)
    pq.write_table(
        pa.table(
            {
                "user_id": list(range(60)),
                "ts": [f"2026-0{1 + i % 5}-15T00:00:00" for i in range(60)],
                "churned": [i % 2 for i in range(60)],
            }
        ),
        data / "rows.parquet",
    )
    write(root / "mbt_project.yml", "name: inspect_demo\nversion: '0.1.0'\n")
    write(
        root / "profiles.yml",
        f"""
inspect_demo:
  target: dev
  outputs:
    dev:
      data: {{adapter: local, config: {{root: {root}}}}}
      tracking: {{adapter: fake, config: {{root: {root}/target/fake_tracking}}}}
      registry: {{adapter: fake, config: {{root: {root}/target/fake_registry}}}}
      compute: {{adapter: local}}
      artifact_store: file://{root}/target/artifacts
""",
    )
    write(
        root / "sources.yml",
        "sources:\n  - name: lake\n    tables:\n      - name: rows\n"
        "        path: data/*.parquet\n        format: parquet\n",
    )
    write(
        root / "datasets" / "panel.yml",
        """
datasets:
  - name: panel
    tags: [core]
    source: source('lake', 'rows')
    sample_key: [user_id]
    label:
      column: churned
    split:
      strategy: random
      train: '0.8'
      test: '0.2'
      seed: 7
""",
    )
    return root


def test_enter_resolves_and_chdirs_into_the_project(tmp_path: Path) -> None:
    """Establishing project-dir semantics is a construction obligation now."""
    import os

    project = _project(tmp_path / "proj")
    outside = tmp_path / "elsewhere"
    outside.mkdir()
    os.chdir(outside)

    cli = CLIContext.enter(Path("../proj"))
    assert cli.project_dir == project.resolve()
    assert Path.cwd() == project.resolve()
    # the invocation cwd is remembered, so a typed path stays shell-relative
    assert cli.invocation_cwd == outside.resolve()
    assert cli.resolve_cli_path("ref.json") == str(outside.resolve() / "ref.json")


def test_enter_rejects_a_project_dir_that_is_not_a_directory(tmp_path: Path) -> None:
    with pytest.raises(ConfigError, match="is not a directory"):
        CLIContext.enter(tmp_path / "nope")


def test_enter_without_chdir_allows_a_directory_that_does_not_exist(tmp_path: Path) -> None:
    """``mbt init``'s --project-dir is the parent to scaffold INTO."""
    cli = CLIContext.enter(tmp_path / "new", chdir=False)
    assert cli.project_dir == (tmp_path / "new").resolve()


def test_ls_returns_rows_with_tags_and_paths(tmp_path: Path) -> None:
    cli = CLIContext.enter(_project(tmp_path / "proj"))
    rows = ls_cmd(cli, anchor=ANCHOR)
    by_id = {row.unique_id: row for row in rows}
    panel = by_id["dataset.inspect_demo.panel"]
    assert panel.resource_type == "dataset"
    assert panel.tags == ("core",)
    assert panel.path.endswith("panel.yml")
    assert "source.inspect_demo.lake.rows" in by_id


def test_ls_honours_tag_and_resource_type_selectors(tmp_path: Path) -> None:
    cli = CLIContext.enter(_project(tmp_path / "proj"))
    assert [r.name for r in ls_cmd(cli, select=["tag:core"], anchor=ANCHOR)] == ["panel"]
    assert ls_cmd(cli, select=["tag:absent"], anchor=ANCHOR) == []
    types = {r.resource_type for r in ls_cmd(cli, select=["resource_type:source"], anchor=ANCHOR)}
    assert types == {"source"}


def test_ls_evaluates_state_selectors_against_a_reference_manifest(tmp_path: Path) -> None:
    """v5 live defect 3, as a regression test.

    ``mbt ls --select state:modified`` used to fail with "requires --state",
    and ``--state`` was not an option to pass - while the command's own
    docstring and docs/cli-reference.md both documented the grammar.
    """
    project = _project(tmp_path / "proj")
    cli = CLIContext.enter(project)
    reference = project / "ref.json"
    _, manifest = cli.compile(anchor=ANCHOR)
    manifest.write(reference)

    # unchanged against itself: nothing is modified
    assert ls_cmd(cli, select=["state:modified"], state=str(reference), anchor=ANCHOR) == []

    # change the split seed, which is part of the dataset's config hash
    spec = project / "datasets" / "panel.yml"
    spec.write_text(spec.read_text().replace("seed: 7", "seed: 8"))
    modified = ls_cmd(cli, select=["state:modified"], state=str(reference), anchor=ANCHOR)
    assert [row.name for row in modified] == ["panel"]


def test_ls_state_selector_without_state_still_explains_itself(tmp_path: Path) -> None:
    from mbt.dag.selector import SelectorError

    cli = CLIContext.enter(_project(tmp_path / "proj"))
    with pytest.raises(SelectorError, match="requires --state"):
        ls_cmd(cli, select=["state:modified"], anchor=ANCHOR)


def test_show_renders_yaml_and_json_and_suggests_on_a_typo(tmp_path: Path) -> None:
    cli = CLIContext.enter(_project(tmp_path / "proj"))
    assert "panel" in show_cmd(cli, "panel", anchor=ANCHOR)
    assert show_cmd(cli, "panel", output="json", anchor=ANCHOR).lstrip().startswith("{")
    with pytest.raises(ConfigError, match="did you mean 'panel'"):
        show_cmd(cli, "pannel", anchor=ANCHOR)


def test_show_is_anchored_so_two_calls_agree(tmp_path: Path) -> None:
    """Without --anchor this re-anchored to now() every invocation (A-2)."""
    cli = CLIContext.enter(_project(tmp_path / "proj"))
    assert show_cmd(cli, "panel", anchor=ANCHOR) == show_cmd(cli, "panel", anchor=ANCHOR)


def test_clean_target_reports_both_outcomes(tmp_path: Path) -> None:
    project = _project(tmp_path / "proj")
    cli = CLIContext.enter(project)
    assert "nothing to clean" in clean_target_cmd(cli)[0]
    (project / "target").mkdir()
    (project / "target" / "manifest.json").write_text("{}")
    assert "removed" in clean_target_cmd(cli)[0]
    assert not (project / "target").exists()


def test_clean_resolves_project_dir_relative_to_the_invocation_cwd(tmp_path: Path) -> None:
    """The divergence all four runner-based clean tests missed, because each
    passed an ABSOLUTE --project-dir (A-2)."""
    import os

    project = _project(tmp_path / "proj")
    (project / "target").mkdir()
    outside = tmp_path / "elsewhere"
    outside.mkdir()
    os.chdir(outside)

    cli = CLIContext.enter(Path("../proj"))
    assert "removed" in clean_target_cmd(cli)[0]
    assert not (project / "target").exists()
