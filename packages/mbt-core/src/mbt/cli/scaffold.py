"""``mbt init``: golden-path project scaffold (FR-PROJ-01, TSD §18)."""

import re
from importlib.resources import files
from pathlib import Path

import mbt
from mbt.exceptions import ConfigError

_TOKEN = "__PROJECT_NAME__"
#: Stamped into the template's requirements pins so a scaffolded project
#: reproduces the exact toolchain version that generated it (NFR-01).
_VERSION_TOKEN = "__MBT_VERSION__"
_NAME_RE = re.compile(r"^[a-z][a-z0-9_]*$")

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


def scaffold_project(name: str, parent_dir: Path, *, home: Path | None = None) -> Path:
    """Create a new project directory from the template; returns its path."""
    if not _NAME_RE.match(name):
        raise ConfigError(
            f"invalid project name {name!r}",
            hint="use lowercase snake_case starting with a letter, e.g. churn_models",
        )
    destination = parent_dir / name
    if destination.exists() and any(destination.iterdir()):
        raise ConfigError(
            f"directory {destination} already exists and is not empty",
            hint="choose another name or remove the directory",
        )

    template_root = files("mbt.cli") / "_scaffold"
    for rel, content in sorted(_walk(template_root)):
        parts = rel.split("/")
        parts[-1] = _RENAMES.get(parts[-1], parts[-1])
        target = destination.joinpath(*parts)
        target.parent.mkdir(parents=True, exist_ok=True)
        rendered = content.replace(_TOKEN, name).replace(_VERSION_TOKEN, mbt.__version__)
        target.write_text(rendered)

    _install_home_profiles(name, destination, home=home)
    return destination


def _install_home_profiles(name: str, destination: Path, *, home: Path | None) -> None:
    """profiles.yml lives in ~/.mbt by default; the project copy is gitignored
    (TSD §18). Never clobber existing profiles for other projects."""
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
