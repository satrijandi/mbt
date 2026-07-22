"""Guard for the release bump script (scripts/bump_version.py).

The script edits 21 version strings in lockstep - the same surface
`test_version_sync.py` guards - so a release stops being a manual 21-file edit
(the "monorepo release tax"). The edit logic is exercised on a temp fixture
(never the real repo); a final read-only check asserts the real tree matches
the script's exact-replacement assumptions so a real bump won't fail.
"""

import importlib.util
import tomllib
from pathlib import Path

import pytest

REPO_ROOT = Path(__file__).resolve().parent.parent
_spec = importlib.util.spec_from_file_location(
    "bump_version", REPO_ROOT / "scripts" / "bump_version.py"
)
assert _spec is not None and _spec.loader is not None
bump_version = importlib.util.module_from_spec(_spec)
_spec.loader.exec_module(bump_version)


def _fake_workspace(root: Path, version: str = "0.1.0") -> None:
    (root / "pyproject.toml").write_text(f'[project]\nname = "mbt"\nversion = "{version}"\n')
    for pkg, module in (("mbt-core", "mbt"), ("mbt-xgboost", "mbt_xgboost")):
        src = root / "packages" / pkg / "src" / module
        src.mkdir(parents=True)
        (root / "packages" / pkg / "pyproject.toml").write_text(
            f'[project]\nname = "{pkg}"\nversion = "{version}"\n'
            'dependencies = ["numpy>=1.0"]\n'  # a version pin that must NOT be rewritten
        )
        (src / "__init__.py").write_text(f'__version__ = "{version}"\n')


def test_bump_updates_every_version_string_in_lockstep(tmp_path: Path) -> None:
    _fake_workspace(tmp_path)
    changed = bump_version.bump_version(tmp_path, "0.2.0")
    assert len(changed) == 5  # root + 2 package pyprojects + 2 __init__

    assert tomllib.loads((tmp_path / "pyproject.toml").read_text())["project"]["version"] == "0.2.0"
    for pkg, module in (("mbt-core", "mbt"), ("mbt-xgboost", "mbt_xgboost")):
        pp = tmp_path / "packages" / pkg / "pyproject.toml"
        assert tomllib.loads(pp.read_text())["project"]["version"] == "0.2.0"
        init = tmp_path / "packages" / pkg / "src" / module / "__init__.py"
        assert init.read_text() == '__version__ = "0.2.0"\n'
    # the dependency pin is left alone - only the project version moves
    assert "numpy>=1.0" in (tmp_path / "packages" / "mbt-core" / "pyproject.toml").read_text()


def test_bump_rejects_malformed_version(tmp_path: Path) -> None:
    _fake_workspace(tmp_path)
    with pytest.raises(ValueError, match=r"X\.Y\.Z"):
        bump_version.bump_version(tmp_path, "v0.2")


def test_bump_rejects_a_noop_bump(tmp_path: Path) -> None:
    _fake_workspace(tmp_path)
    with pytest.raises(ValueError, match="already"):
        bump_version.bump_version(tmp_path, "0.1.0")


def test_bump_fails_loudly_when_a_file_lacks_the_version(tmp_path: Path) -> None:
    _fake_workspace(tmp_path)
    # corrupt one package so its version string is absent
    (tmp_path / "packages" / "mbt-core" / "pyproject.toml").write_text(
        '[project]\nname = "mbt-core"\nversion = "9.9.9"\n'
    )
    with pytest.raises(ValueError, match="expected exactly one"):
        bump_version.bump_version(tmp_path, "0.2.0")


def test_cli_main_reports_changed_files(tmp_path: Path, capsys: pytest.CaptureFixture[str]) -> None:
    _fake_workspace(tmp_path)
    assert bump_version.main(["0.3.0", "--root", str(tmp_path)]) == 0
    out = capsys.readouterr().out
    assert "bumped to 0.3.0 across 5 files" in out
    # a bad version exits 1 without raising
    assert bump_version.main(["nope", "--root", str(tmp_path)]) == 1


def test_real_repo_matches_the_scripts_replacement_assumptions() -> None:
    # read-only: the real tree must carry the current version exactly once per
    # file, or a real `bump_version.py` run would fail the exactly-one guard.
    current = tomllib.loads((REPO_ROOT / "pyproject.toml").read_text())["project"]["version"]
    package_pyprojects = sorted(REPO_ROOT.glob("packages/*/pyproject.toml"))
    for pyproject in [REPO_ROOT / "pyproject.toml", *package_pyprojects]:
        assert pyproject.read_text().count(f'version = "{current}"') == 1, pyproject
    for pyproject in package_pyprojects:
        (init,) = (pyproject.parent / "src").glob("*/__init__.py")
        assert init.read_text().count(f'__version__ = "{current}"') == 1, init
