"""Install the workspace with every declared direct dependency at its floor.

``uv sync --resolution lowest-direct`` looks like it does this and does not. In
a virtual workspace root (``package = false``) the members' requirements are not
"direct" from the root's point of view, so they resolve to the newest version
that satisfies them - and so do the root's own dependency-group entries. CI's
floors job ran that command for months and installed pytest 9.1.1 against a
declared floor of 8.0 and pyarrow 25.0.1 against a declared floor of 15.0, i.e.
it was an expensive duplicate of the normal test job. Nothing had ever installed
a floor, so "the floors are load-bearing metadata" was an unverified claim, and
the first real run found two wrong ones (``duckdb>=1.0`` could not parse the
local adapter's own SQL, ``click>=8.1`` could not run the CLI test surface) plus
about seventy advisories against floor versions nobody had audited.

``uv pip install --resolution lowest-direct`` DOES honour floors, because there
every requirement passed on the command line is direct by construction. So this
script flattens the workspace into exactly that: each member as an editable
path (carrying whatever extras the dev group asked for), each third-party
dev-group entry passed through with its own declared specifier.

``--verify`` then asserts the environment really is at the floors, so that a
future regression in the install path fails loudly instead of quietly turning
this job back into a second copy of the test job.

The argument list is derived from pyproject.toml rather than restated in
ci.yml, so a new dependency cannot be added without also being floor-tested.

Stdlib only, so it runs before anything is installed.
"""

from __future__ import annotations

import re
import subprocess
import sys
import tomllib
from pathlib import Path

#: Workspace members the floors environment leaves out, and why. Their own floors are
#: tested by a job of their own, and the test below pins that such a job exists.
#:
#: mbt-airflow: a floors resolution (direct dependencies lowest, everything else newest)
#: is a combination Airflow does not support - it only supports installs under its own
#: published constraints. Its dependency tree also lifts pydantic and packaging above the
#: floors mbt-core declares, so with it present this job would quietly stop testing them
#: (`--verify` fails on exactly that). CI's airflow-floor job installs the Airflow floor
#: the supported way instead.
EXCLUDED: dict[str, str] = {
    "mbt-airflow": "Airflow only supports installs under its own constraints, and its "
    "dependency tree lifts pydantic and packaging above mbt-core's floors; "
    "CI's airflow-floor job tests it",
}

#: `name`, optional `[extras]`, and whatever specifier follows.
_REQUIREMENT = re.compile(r"^(?P<name>[A-Za-z0-9._-]+)\s*(?:\[(?P<extras>[^\]]*)\])?")
#: the `>=` lower bound, which is what "floor" means in this repo.
_FLOOR = re.compile(r">=\s*(?P<floor>[0-9][^,;\s]*)")
#: any bound that sets the LOWEST version a requirement allows.
_LOWER_BOUND = re.compile(r"(?:>=|==|~=|>)\s*([0-9][0-9A-Za-z.]*)")


def _canonical(name: str) -> str:
    return name.lower().replace("_", "-")


def _parse(requirement: str) -> tuple[str, str, str | None]:
    """(canonical name, extras, declared floor) for a PEP 508 requirement."""
    match = _REQUIREMENT.match(requirement.strip())
    if match is None:  # pragma: no cover - a malformed pyproject is not our bug
        raise SystemExit(f"could not parse requirement {requirement!r}")
    floor = _FLOOR.search(requirement)
    return (
        _canonical(match.group("name")),
        (match.group("extras") or "").strip(),
        floor.group("floor") if floor else None,
    )


def _release(version: str) -> tuple[int, ...]:
    """Numeric release tuple, so ``5`` and ``5.0.0`` compare equal.

    Deliberately not packaging.Version: this module stays stdlib-only so it can
    run before anything is installed. Floors in this repo are plain releases.
    """
    parts: list[int] = []
    for chunk in version.split("."):
        digits = re.match(r"\d+", chunk)
        if digits is None:
            break
        parts.append(int(digits.group()))
    while len(parts) < 4:
        parts.append(0)
    return tuple(parts[:4])


def member_paths(root: Path) -> dict[str, Path]:
    """Workspace member distribution name -> its directory.

    Read from each member's own pyproject rather than assumed from the
    directory name, so a rename cannot silently drop a package from the run.
    """
    members: dict[str, Path] = {}
    config = tomllib.loads((root / "pyproject.toml").read_text())
    for pattern in config["tool"]["uv"]["workspace"]["members"]:
        for path in sorted(root.glob(pattern)):
            pyproject = path / "pyproject.toml"
            if pyproject.is_file():
                name = tomllib.loads(pyproject.read_text())["project"]["name"]
                members[_canonical(name)] = path
    return members


def install_args(root: Path, group: str = "dev") -> list[str]:
    """``uv pip install`` arguments pinning every direct dep to its floor.

    Workspace members become editable paths (a version specifier on a local
    package would be meaningless); everything else keeps the specifier declared
    in pyproject, which is what ``lowest-direct`` then resolves downward.
    """
    config = tomllib.loads((root / "pyproject.toml").read_text())
    members = member_paths(root)
    args = ["--resolution", "lowest-direct"]
    for requirement in config["dependency-groups"][group]:
        name, extras, _ = _parse(requirement)
        if name in EXCLUDED:
            continue
        path = members.get(name)
        if path is None:
            args.append(requirement)
        else:
            suffix = f"[{extras}]" if extras else ""
            args += ["-e", f"{path.relative_to(root).as_posix()}{suffix}"]
    return args


def declared_floors(root: Path, group: str = "dev") -> dict[str, str]:
    """Third-party package -> the floor the workspace declares for it.

    Covers the dev group AND every workspace member's own dependencies, since
    the latter are the floors a *user* gets. Where two members declare the same
    package the higher floor wins, which is what a resolver would install.
    """
    config = tomllib.loads((root / "pyproject.toml").read_text())
    members = member_paths(root)
    floors: dict[str, str] = {}

    def record(requirements: list[str]) -> None:
        for requirement in requirements:
            name, _, floor = _parse(requirement)
            if name in members or floor is None:
                continue
            if name not in floors or _release(floor) > _release(floors[name]):
                floors[name] = floor

    wanted: list[str] = []
    for requirement in config["dependency-groups"][group]:
        name, extras, _ = _parse(requirement)
        if name in EXCLUDED:
            continue
        if name in members:
            wanted += [f"{name}:{extra}" for extra in extras.split(",") if extra]
        record([requirement])

    for name, path in members.items():
        if name in EXCLUDED:
            continue
        project = tomllib.loads((path / "pyproject.toml").read_text())["project"]
        record(project.get("dependencies", []))
        for extra, requirements in (project.get("optional-dependencies") or {}).items():
            if f"{name}:{extra}" in wanted:
                record(requirements)
    return floors


def forced_by(name: str, installed: str) -> str | None:
    """An installed distribution whose own requirement holds ``name`` at ``installed``.

    "Holds" means that requirement's lower bound is in the installed version's
    series, so ``installed`` is the lowest version the environment allows. This is
    how a floor is legitimately lifted: ``virtualenv`` (a transitive of pre-commit,
    resolved newest) began requiring ``packaging>=26.3``, so ``packaging`` could no
    longer sit at mbt-core's declared 23.1 in this environment, whatever the install
    path did. It is not the failure ``--verify`` exists for.
    """
    from importlib.metadata import distributions

    series = _release(installed)[:2]
    for dist in distributions():
        for requirement in dist.requires or []:
            parsed = _REQUIREMENT.match(requirement)
            if parsed is None or _canonical(parsed.group("name")) != name:
                continue
            specifier = requirement.split(";")[0]
            if "extra ==" in requirement:
                continue
            bounds = [_release(v)[:2] for v in _LOWER_BOUND.findall(specifier)]
            if bounds and max(bounds) == series:
                return str(dist.metadata["Name"])
    return None


def verify(root: Path, notes: list[str] | None = None) -> list[str]:
    """Packages whose installed version is not in the declared floor's series.

    Compared on (major, minor) rather than exactly, because the floor is a
    bound and not always an installable point: ``pytest-timeout>=2.3`` resolves
    to 2.3.1 since 2.3.0 cannot co-exist with the pytest floor. Requiring
    equality would red the job for a resolution that is doing the right thing.
    The regression this guards against - the environment silently resolving to
    newest - moves the minor every time (pytest 8.0 -> 9.1, pyarrow 15 -> 25),
    so the looser bar still catches it.
    """
    from importlib.metadata import PackageNotFoundError, version

    drift: list[str] = []
    for name, floor in sorted(declared_floors(root).items()):
        try:
            installed = version(name)
        except PackageNotFoundError:
            continue  # an extra we did not ask for, or a marker that excluded it
        if _release(installed)[:2] != _release(floor)[:2]:
            forcing = forced_by(name, installed)
            if forcing is None:
                drift.append(f"{name}: declared floor {floor}, installed {installed}")
            elif notes is not None:
                notes.append(
                    f"{name} {installed} is above its floor {floor}: {forcing} requires it"
                )
    return drift


def main(argv: list[str] | None = None) -> int:
    args = sys.argv[1:] if argv is None else argv
    root = Path(__file__).resolve().parent.parent
    if "--verify" in args:
        notes: list[str] = []
        drift = verify(root, notes)
        for line in notes:
            print(f"note: {line}")
        for line in drift:
            print(f"FAIL not at its floor: {line}")
        if drift:
            print(
                "\nThe floors job is only meaningful if the environment is AT the "
                "floors. Newest-looking versions here mean the install path "
                "stopped honouring --resolution lowest-direct."
            )
            return 1
        print(f"every declared floor is installed at its floor ({len(declared_floors(root))})")
        return 0
    command = ["uv", "pip", "install", *install_args(root)]
    print(" ".join(command), flush=True)
    if "--dry-run" in args:
        return 0
    return subprocess.run(command, cwd=root, check=False).returncode


if __name__ == "__main__":
    raise SystemExit(main())
