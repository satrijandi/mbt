"""``mbt init``: golden-path project scaffold (FR-PROJ-01, TSD §18)."""

import re
from importlib.resources import files
from pathlib import Path

import mbt
from mbt.config.project import PROJECT_NAME_PATTERN
from mbt.exceptions import ConfigError

_TOKEN = "__PROJECT_NAME__"
#: The git ref the template's requirements pin mbt to, so a scaffolded project
#: reproduces the exact toolchain that generated it (NFR-01). See ``mbt_ref``.
_REF_TOKEN = "__MBT_REF__"
#: Replaced by exact `==` pins for the packages below, at the versions installed
#: in the environment running `mbt init`.
_PINS_TOKEN = "__PINNED_DEPS__"
#: One pattern with mbt_project.yml's own `name:` field, imported rather than
#: restated: they were two copies of the same regex and could drift, so
#: `mbt init NAME` could reject a name a hand-written project accepts.
_NAME_RE = re.compile(PROJECT_NAME_PATTERN)

#: The numerics stack the scaffold pins by version, in install order.
#:
#: Pinning the mbt packages pins NONE of these, and these are the versions
#: that decide model numerics. requirements.txt's own header states the reason
#: the file exists - "a floating training environment invalidates the manifest's
#: env digest, so CI always installs from this file" - and with only the mbt
#: refs pinned that was false: env_freeze_digest (ADR-19) changed whenever any
#: of them released, which is the exact condition ADR-19 exists to detect.
#:
#: Resolved from the scaffolding environment rather than hardcoded, so the pins
#: are the versions this mbt was actually tested against and cannot go stale in
#: the template. A package that is not installed is skipped: the scaffold must
#: work from a partial install (mbt-core alone, say) rather than pin a version
#: nobody verified.
_PINNED_PACKAGES = (
    "numpy",
    "scipy",
    "pandas",
    "pyarrow",
    "scikit-learn",
    "duckdb",
    "xgboost",
    "mlflow",
)


_SHA_RE = re.compile(r"[0-9a-f]{40}")


def mbt_ref(override: str | None = None) -> str:
    """The git ref a scaffolded project's CI installs mbt from (FEEDBACK v6 A-1).

    A release build pins its own tag, ``vX.Y.Z``. A development build
    (``X.Y.Z.devN``) has no tag that contains its code: v0.1.0 pinned
    ``v0.1.0`` for 95 commits after that tag was cut, so every project
    scaffolded from ``main`` installed a release that could not read the
    scaffold it shipped with. A development build therefore pins the COMMIT it
    was installed from - ``direct_url.json``'s ``vcs_info`` for a
    ``pip install git+...`` install, ``git rev-parse HEAD`` for an editable
    checkout - and refuses to guess when it has neither.
    """
    if override:
        return override
    version = mbt.__version__
    if ".dev" not in version:
        return f"v{version}"
    commit = _installed_commit()
    if commit is None:
        raise ConfigError(
            f"this mbt is a development build ({version}) that was not installed from "
            "git, so no tag or commit is known to contain its code",
            hint="pass --mbt-ref <tag-or-commit> naming the mbt it should pin, or install "
            "mbt from a release tag or a git checkout",
        )
    return commit


def _installed_commit() -> str | None:
    """The commit mbt-core was installed from, if the install recorded one."""
    import json
    import subprocess
    from importlib.metadata import PackageNotFoundError, distribution
    from urllib.parse import unquote, urlparse

    try:
        raw = distribution("mbt-core").read_text("direct_url.json")
    except PackageNotFoundError:
        return None
    info = json.loads(raw) if raw else {}
    commit = info.get("vcs_info", {}).get("commit_id")
    if isinstance(commit, str) and _SHA_RE.fullmatch(commit):
        return commit
    url = info.get("url", "")
    if not (info.get("dir_info", {}).get("editable") and url.startswith("file://")):
        return None
    try:
        head = subprocess.run(
            ["git", "rev-parse", "HEAD"],
            cwd=unquote(urlparse(url).path),
            capture_output=True,
            text=True,
            check=True,
        ).stdout.strip()
    except (OSError, subprocess.CalledProcessError):
        return None
    return head if _SHA_RE.fullmatch(head) else None


def _pinned_requirements() -> str:
    """`name==version` lines for the numerics stack, one per installed package."""
    from importlib.metadata import PackageNotFoundError, version

    lines = []
    for package in _PINNED_PACKAGES:
        try:
            lines.append(f"{package}=={version(package)}")
        except PackageNotFoundError:
            continue  # not installed here; pin nothing rather than guess
    return "\n".join(lines)


#: Template files renamed on write (dotfiles cannot ship as package data
#: reliably across build backends).
_RENAMES = {"gitignore": ".gitignore"}

#: Never copied into a scaffolded project, and never read.
#:
#: The template is *source*, so in an editable or checked-out install anything
#: that imports `scripts/generate_sample_data.py` leaves a `__pycache__` beside
#: it. The walk below reads every file as text, so one stray `.pyc` turned
#: `mbt init` - the first command a new user ever runs - into
#: "Internal error: UnicodeDecodeError ... this is a bug in mbt".
#: Found exactly that way while verifying the sample-data generator.
_SKIP_DIRS = {"__pycache__", ".pytest_cache", ".ruff_cache", ".mypy_cache"}
_SKIP_SUFFIXES = (".pyc", ".pyo")


def _walk(root: object, prefix: str = "") -> list[tuple[str, str]]:
    """(relative_path, content) for every template file."""
    out: list[tuple[str, str]] = []
    for entry in root.iterdir():  # type: ignore[attr-defined]
        rel = f"{prefix}{entry.name}"
        if entry.is_dir():
            if entry.name not in _SKIP_DIRS:
                out.extend(_walk(entry, prefix=f"{rel}/"))
        elif not entry.name.endswith(_SKIP_SUFFIXES):
            out.append((rel, entry.read_text()))
    return out


def scaffold_project(
    name: str, parent_dir: Path, *, home: Path | None = None, ref: str | None = None
) -> Path:
    """Create a new project directory from the template; returns its path."""
    if not _NAME_RE.match(name):
        raise ConfigError(
            f"invalid project name {name!r}",
            hint="use letters, digits and underscores, starting with a letter - "
            "e.g. churn_models or LOAN_APPLY_PROPENSITY",
        )
    destination = parent_dir / name
    if destination.exists() and any(destination.iterdir()):
        raise ConfigError(
            f"directory {destination} already exists and is not empty",
            hint="choose another name or remove the directory",
        )

    template_root = files("mbt.cli") / "_scaffold"
    pinned_ref = mbt_ref(ref)
    pins = _pinned_requirements()
    for rel, content in sorted(_walk(template_root)):
        parts = rel.split("/")
        parts[-1] = _RENAMES.get(parts[-1], parts[-1])
        target = destination.joinpath(*parts)
        target.parent.mkdir(parents=True, exist_ok=True)
        rendered = (
            content.replace(_TOKEN, name).replace(_REF_TOKEN, pinned_ref).replace(_PINS_TOKEN, pins)
        )
        target.write_text(rendered)

    _install_home_profiles(name, destination, home=home)
    return destination


def _install_home_profiles(name: str, destination: Path, *, home: Path | None) -> None:
    """profiles.yml also lives in ~/.mbt so commands run outside the project
    find it (TSD §18); the project copy is committed, because CI has no ~/.mbt
    to read. Never clobber existing profiles for other projects."""
    home_dir = home or Path.home()
    home_profiles = home_dir / ".mbt" / "profiles.yml"
    project_profiles = (destination / "profiles.yml").read_text()
    if not home_profiles.exists():
        home_profiles.parent.mkdir(parents=True, exist_ok=True)
        home_profiles.write_text(project_profiles)
        return
    existing = home_profiles.read_text()
    if re.search(rf"^{re.escape(name)}:", existing, flags=re.MULTILINE):
        return  # already configured; leave the user's file alone
    home_profiles.write_text(existing.rstrip("\n") + "\n\n" + project_profiles)
