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

import importlib.util
from pathlib import Path

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


def test_verify_flags_an_environment_that_is_not_at_the_floors() -> None:
    """The running interpreter is the LOCKED env, deliberately above the floors,
    so verify must report drift here. If this ever passes, --verify has stopped
    being able to detect the regression it exists for."""
    assert floors.verify(REPO_ROOT), (
        "verify() found no drift in the locked environment, so it would not "
        "notice the floors job silently resolving to newest either"
    )
