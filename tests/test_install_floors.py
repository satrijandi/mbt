"""Guard for the floor installer (scripts/install_floors.py).

The floors job is the only thing that ever installs a declared lower bound, and
for months it did not: `uv sync --resolution lowest-direct` resolves a virtual
workspace root to newest, so the job was a second copy of the test job and every
floor in the repo was an unverified claim. These assert the two properties that
would let that happen again silently - that the install set is derived from
pyproject rather than restated somewhere, and that --verify actually fails when
the environment is not at the floors.

Offline: everything here reads the repo's own pyproject files.
"""

import importlib.metadata
import importlib.util
from pathlib import Path

import pytest

REPO_ROOT = Path(__file__).resolve().parent.parent
_spec = importlib.util.spec_from_file_location(
    "install_floors", REPO_ROOT / "scripts" / "install_floors.py"
)
assert _spec is not None and _spec.loader is not None
floors = importlib.util.module_from_spec(_spec)
_spec.loader.exec_module(floors)


def test_every_workspace_member_is_installed_editable() -> None:
    """A member missing from the install set is a package never floor-tested."""
    args = floors.install_args(REPO_ROOT)
    editable = {args[i + 1] for i, a in enumerate(args) if a == "-e"}
    for name, path in floors.member_paths(REPO_ROOT).items():
        rel = path.relative_to(REPO_ROOT).as_posix()
        assert any(e == rel or e.startswith(f"{rel}[") for e in editable), (
            f"{name} is a workspace member but install_floors would not install it"
        )


def test_lowest_direct_is_requested() -> None:
    """Without this flag the whole exercise installs newest and proves nothing."""
    args = floors.install_args(REPO_ROOT)
    assert args[:2] == ["--resolution", "lowest-direct"]


def test_dev_group_extras_survive_into_the_install_set() -> None:
    """`mbt-adapter-base[compliance,metrics]` carries the compliance suite the
    fast tier imports; dropping the extras yields an env that cannot run it."""
    args = floors.install_args(REPO_ROOT)
    assert any(a.startswith("packages/mbt-adapter-base[") for a in args)


def test_third_party_requirements_keep_their_declared_specifier() -> None:
    """Passed through verbatim, so `lowest-direct` has a bound to resolve down
    to. A bare name would install newest and silently un-test the floor."""
    args = floors.install_args(REPO_ROOT)
    assert any(a.startswith("pytest>=") for a in args)
    assert not any(a == "pytest" for a in args)


def test_declared_floors_cover_member_dependencies_not_just_the_dev_group() -> None:
    """The floors a USER gets live in the member packages, so --verify has to
    read those too; checking only the dev group would miss pyarrow and duckdb."""
    declared = floors.declared_floors(REPO_ROOT)
    for package in ("pyarrow", "duckdb", "pydantic", "mlflow"):
        assert package in declared, f"{package} floor is not being verified"


def test_the_higher_floor_wins_when_members_disagree() -> None:
    """Two members can declare the same package; a resolver installs the higher
    bound, so that is the one --verify must expect."""
    assert floors._release("2.10") > floors._release("2.9")
    assert floors._release("5") == floors._release("5.0.0")


def test_verify_flags_a_package_installed_above_its_declared_floor(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """The regression --verify exists for is an environment resolving to newest.

    Simulated rather than read from the ambient environment: this suite runs in
    BOTH the locked env (deliberately above the floors) and the floors env
    (deliberately at them), so any assertion about the ambient versions passes
    in one and fails in the other.
    """
    monkeypatch.setattr(floors, "declared_floors", lambda root, group="dev": {"pytest": "1.0"})
    drift = floors.verify(REPO_ROOT)
    assert len(drift) == 1
    assert drift[0].startswith("pytest: declared floor 1.0, installed ")


def test_verify_is_quiet_when_the_installed_version_is_the_floor(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    installed = importlib.metadata.version("pytest")
    monkeypatch.setattr(floors, "declared_floors", lambda root, group="dev": {"pytest": installed})
    assert floors.verify(REPO_ROOT) == []


def test_verify_ignores_a_declared_package_that_is_not_installed(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """An extra we did not ask for is absent by design, not drift."""
    monkeypatch.setattr(
        floors, "declared_floors", lambda root, group="dev": {"not-a-real-package": "1.0"}
    )
    assert floors.verify(REPO_ROOT) == []
