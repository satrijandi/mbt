"""``mbt deps``: install adapter packages pinned in packages.yml (FR-PROJ-04)."""

import subprocess
import sys
from pathlib import Path

import yaml
from pydantic import BaseModel, ConfigDict, ValidationError

from mbt.events import get_bus
from mbt.events.models import LogMessage
from mbt.exceptions import ConfigError

PACKAGES_FILE = "packages.yml"


class PackagePin(BaseModel):
    model_config = ConfigDict(extra="forbid")

    package: str
    version: str | None = None  # PEP 440 specifier, e.g. "~=0.1"


class PackagesFile(BaseModel):
    model_config = ConfigDict(extra="forbid")

    packages: list[PackagePin]


def load_packages(project_dir: Path) -> list[PackagePin]:
    path = project_dir / PACKAGES_FILE
    if not path.is_file():
        raise ConfigError(
            f"no {PACKAGES_FILE} in {project_dir}",
            hint="create one: packages: [{package: mbt-xgboost, version: '~=0.1'}]",
        )
    try:
        payload = yaml.safe_load(path.read_text()) or {}
        return PackagesFile.model_validate(payload).packages
    except (yaml.YAMLError, ValidationError) as exc:
        raise ConfigError(f"invalid {PACKAGES_FILE}: {exc}", path=path) from exc


def install_packages(
    pins: list[PackagePin],
    *,
    dry_run: bool = False,
    requirements_file: Path | None = None,
) -> list[str]:
    """Install the declared adapter packages into the active environment.

    With a pinned ``requirements_file`` (scaffolded projects ship one with
    exact pins) the install is reproducible: pip installs the file, honoring
    hashes when present, and packages.yml is verified against the result.
    Without one the loose PEP 440 specifiers install whatever resolves
    today - allowed, but loudly warned: an unpinned toolchain floats across
    runs and invalidates the env digest.
    """
    requirements = [f"{pin.package}{pin.version}" if pin.version else pin.package for pin in pins]
    if dry_run or not requirements:
        return requirements
    if requirements_file is not None:
        command = [sys.executable, "-m", "pip", "install", "-r", str(requirements_file)]
        get_bus().emit(
            LogMessage(message=f"installing pinned requirements from {requirements_file.name}")
        )
    else:
        command = [sys.executable, "-m", "pip", "install", *requirements]
        get_bus().emit(
            LogMessage(
                level="warn",
                message=(
                    "packages.yml installs are unpinned (no requirements.txt in the "
                    "project); pin the toolchain, e.g. `uv pip compile requirements.in "
                    "-o requirements.txt`, for reproducible environments"
                ),
            )
        )
        get_bus().emit(LogMessage(message="installing: " + " ".join(requirements)))
    proc = subprocess.run(command, capture_output=True, text=True, check=False)
    if proc.returncode != 0:
        raise ConfigError(
            f"pip install failed (exit {proc.returncode})",
            hint=(proc.stderr or proc.stdout).strip().splitlines()[-1]
            if (proc.stderr or proc.stdout).strip()
            else "run pip manually to see the full error",
        )
    verify_pins(pins)
    return requirements


def verify_pins(pins: list[PackagePin]) -> None:
    """The installed environment must satisfy every packages.yml pin: a
    pinned requirements.txt that drifted from packages.yml fails here,
    loudly, instead of at the first import."""
    from importlib import metadata

    from packaging.requirements import Requirement

    problems = []
    for pin in pins:
        requirement = Requirement(f"{pin.package}{pin.version or ''}")
        try:
            installed = metadata.version(requirement.name)
        except metadata.PackageNotFoundError:
            problems.append(f"{requirement.name} is not installed")
            continue
        if pin.version and not requirement.specifier.contains(installed, prereleases=True):
            problems.append(f"{requirement.name}=={installed} does not satisfy {pin.version!r}")
    if problems:
        raise ConfigError(
            "installed environment does not satisfy packages.yml: " + "; ".join(problems),
            hint="regenerate requirements.txt from requirements.in, or fix packages.yml",
        )
