"""Unit tests for the ``mbt init`` scaffold (mbt/cli/scaffold.py).

Every test passes an explicit ``home=`` so the real ~/.mbt is never touched.
"""

from pathlib import Path

import pytest

import mbt
from mbt.cli.scaffold import scaffold_project
from mbt.exceptions import ConfigError


def test_scaffold_creates_project_and_home_profiles(tmp_path: Path) -> None:
    home = tmp_path / "home"
    destination = scaffold_project("churn_models", tmp_path, home=home)
    assert destination == tmp_path / "churn_models"
    assert (destination / "mbt_project.yml").is_file()
    assert (destination / ".gitignore").is_file()  # renamed from template 'gitignore'

    project_text = (destination / "mbt_project.yml").read_text()
    assert "__PROJECT_NAME__" not in project_text
    assert "churn_models" in project_text
    pins = (destination / "requirements.in").read_text()
    assert "__MBT_VERSION__" not in pins
    assert mbt.__version__ in pins

    # profiles installed into <home>/.mbt verbatim (TSD §18)
    home_profiles = home / ".mbt" / "profiles.yml"
    assert home_profiles.read_text() == (destination / "profiles.yml").read_text()


def test_scaffold_rejects_invalid_names(tmp_path: Path) -> None:
    for bad in ("CamelCase", "1_starts_with_digit", "has-dash", ""):
        with pytest.raises(ConfigError, match="invalid project name"):
            scaffold_project(bad, tmp_path, home=tmp_path / "home")


def test_scaffold_refuses_nonempty_destination(tmp_path: Path) -> None:
    (tmp_path / "proj").mkdir()
    (tmp_path / "proj" / "keep.txt").write_text("mine")
    with pytest.raises(ConfigError, match="already exists and is not empty"):
        scaffold_project("proj", tmp_path, home=tmp_path / "home")
    assert (tmp_path / "proj" / "keep.txt").read_text() == "mine"  # untouched


def test_scaffold_appends_profiles_for_a_new_project(tmp_path: Path) -> None:
    home = tmp_path / "home"
    existing = home / ".mbt" / "profiles.yml"
    existing.parent.mkdir(parents=True)
    existing.write_text("otherproj:\n  target: dev\n")
    scaffold_project("newproj", tmp_path, home=home)
    text = existing.read_text()
    assert text.startswith("otherproj:")  # other projects' profiles survive
    assert "newproj:" in text


def test_scaffold_never_clobbers_existing_project_profiles(tmp_path: Path) -> None:
    home = tmp_path / "home"
    existing = home / ".mbt" / "profiles.yml"
    existing.parent.mkdir(parents=True)
    existing.write_text("myproj:\n  target: prod\n")
    scaffold_project("myproj", tmp_path, home=home)
    assert existing.read_text() == "myproj:\n  target: prod\n"


def test_scaffold_pins_installed_packages_and_skips_absent_ones(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    """The numerics pins are resolved from the scaffolding environment, not
    hardcoded, so they cannot rot in the template (FEEDBACK B-1).

    A package that is not installed is skipped rather than guessed at: `mbt
    init` has to work from a partial install (mbt-core alone, say), and a pin
    nobody verified is worse than no pin.
    """
    from importlib.metadata import version

    from mbt.cli import scaffold as scaffold_module

    monkeypatch.setattr(scaffold_module, "_PINNED_PACKAGES", ("duckdb", "no-such-package-anywhere"))
    project = scaffold_project("pin_probe", tmp_path, home=tmp_path / "home")

    for name in ("requirements.in", "requirements.txt"):
        pins = (project / name).read_text()
        assert f"duckdb=={version('duckdb')}" in pins
        assert "no-such-package-anywhere" not in pins
        assert "__PINNED_DEPS__" not in pins
