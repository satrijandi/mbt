"""Every workspace package must agree on one version and ship its license.

A release edits eleven pyproject.toml files in lockstep; nothing else enforces
that they stay in sync (the wheel-install e2e only checks the packages it
happens to install). In the spirit of the cli-reference drift guard, this
turns the release convention into a permanent check.
"""

import tomllib
from pathlib import Path

import mbt

REPO_ROOT = Path(__file__).resolve().parent.parent
PACKAGE_PYPROJECTS = sorted(REPO_ROOT.glob("packages/*/pyproject.toml"))


def test_workspace_has_the_expected_packages() -> None:
    assert len(PACKAGE_PYPROJECTS) == 10, [p.parent.name for p in PACKAGE_PYPROJECTS]


def test_all_package_versions_match_the_root_and_the_runtime() -> None:
    root = tomllib.loads((REPO_ROOT / "pyproject.toml").read_text())
    versions = {"pyproject.toml": root["project"]["version"], "mbt.__version__": mbt.__version__}
    for pyproject in PACKAGE_PYPROJECTS:
        versions[pyproject.parent.name] = tomllib.loads(pyproject.read_text())["project"]["version"]
    assert len(set(versions.values())) == 1, versions


def test_every_package_declares_and_ships_the_license() -> None:
    for pyproject in PACKAGE_PYPROJECTS:
        project = tomllib.loads(pyproject.read_text())["project"]
        assert project["license"] == "Apache-2.0", pyproject.parent.name
        assert project.get("authors"), f"{pyproject.parent.name}: no authors"
        assert project.get("urls"), f"{pyproject.parent.name}: no project.urls"
        # hatchling's default license-file globs pick this up into the wheel
        assert (pyproject.parent / "LICENSE").is_file(), f"{pyproject.parent.name}: no LICENSE"
    assert (REPO_ROOT / "LICENSE").is_file()
