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


def install_packages(pins: list[PackagePin], *, dry_run: bool = False) -> list[str]:
    """pip-install each pin into the active environment; returns requirement strings."""
    requirements = [f"{pin.package}{pin.version}" if pin.version else pin.package for pin in pins]
    if dry_run or not requirements:
        return requirements
    command = [sys.executable, "-m", "pip", "install", *requirements]
    get_bus().emit(LogMessage(message="installing: " + " ".join(requirements)))
    proc = subprocess.run(command, capture_output=True, text=True, check=False)
    if proc.returncode != 0:
        raise ConfigError(
            f"pip install failed (exit {proc.returncode})",
            hint=(proc.stderr or proc.stdout).strip().splitlines()[-1]
            if (proc.stderr or proc.stdout).strip()
            else "run pip manually to see the full error",
        )
    return requirements
