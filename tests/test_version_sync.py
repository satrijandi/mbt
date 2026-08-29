"""Every workspace package must agree on one version and ship its license.

A release edits the version in lockstep across the root plus ten package
``pyproject.toml`` files AND each package's runtime ``__version__``; nothing
else enforces that they stay in sync (the wheel-install e2e only checks the
packages it happens to install). In the spirit of the cli-reference drift
guard, this turns the release convention into a permanent check - and covers
every package's ``__version__``, not just ``mbt`` core's. See CONTRIBUTING's
"Releasing" section for the bump procedure.
"""

import re
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
        name = pyproject.parent.name
        versions[name] = tomllib.loads(pyproject.read_text())["project"]["version"]
        # ...and the package's runtime __version__ (only mbt core's was checked
        # before, so an adapter's __version__ could silently drift on a release)
        (init,) = (pyproject.parent / "src").glob("*/__init__.py")
        match = re.search(r'__version__ = "([^"]+)"', init.read_text())
        assert match, f"{name}: no __version__ in {init.name}"
        versions[f"{name}.__version__"] = match.group(1)
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


def test_every_package_ships_a_pep_561_marker() -> None:
    """Without ``py.typed`` in the installed package, a consumer's type checker
    treats every mbt import as ``Any`` - the repo is mypy --strict across all
    ten packages internally and shipped none of that outward.

    It matters most for ``mbt-adapter-base``, whose entire purpose is the typed
    protocols third-party adapters implement (docs/adapter-authoring.md): an
    adapter author who got the signatures wrong would have learned it at
    runtime, in mbt, rather than from their own type checker.

    Hatchling includes every file under the wheel's package dir, so the marker
    reaching the source tree is what puts it in the distribution;
    tests/test_wheel_install.py proves that end of it against a real wheel.
    """
    for pyproject in PACKAGE_PYPROJECTS:
        (package_dir,) = (path.parent for path in (pyproject.parent / "src").glob("*/__init__.py"))
        marker = package_dir / "py.typed"
        assert marker.is_file(), (
            f"{pyproject.parent.name}: no {marker.relative_to(REPO_ROOT)} - PEP 561 "
            f"requires it or downstream type checkers ignore this package entirely"
        )
