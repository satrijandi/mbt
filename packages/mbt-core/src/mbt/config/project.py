"""``mbt_project.yml`` schema and loading (TSD §5.2, FR-PROJ-02)."""

from pathlib import Path
from typing import Any

import yaml
from pydantic import BaseModel, ConfigDict, Field, ValidationError

import mbt
from mbt.exceptions import ConfigError

PROJECT_FILE = "mbt_project.yml"


class ProjectConfig(BaseModel):
    """The project-level configuration."""

    # protected_namespaces=(): model_defaults/model_paths are spec vocabulary,
    # not pydantic's model_* namespace (warns on pydantic < 2.10).
    model_config = ConfigDict(extra="forbid", protected_namespaces=())

    name: str = Field(pattern=r"^[a-z][a-z0-9_]*$")
    version: str
    require_mbt_version: str | None = None  # PEP 440 specifier, checked at parse
    model_defaults: dict[str, Any] = Field(default_factory=dict)
    vars: dict[str, Any] = Field(default_factory=dict)
    model_paths: list[str] = Field(default_factory=lambda: ["models"])
    dataset_paths: list[str] = Field(default_factory=lambda: ["datasets"])
    scoring_paths: list[str] = Field(default_factory=lambda: ["scoring"])
    test_paths: list[str] = Field(default_factory=lambda: ["tests"])
    macro_paths: list[str] = Field(default_factory=lambda: ["macros"])


def _check_version_requirement(specifier: str, project_path: Path) -> None:
    from packaging.specifiers import InvalidSpecifier, SpecifierSet
    from packaging.version import Version

    try:
        spec_set = SpecifierSet(specifier)
    except InvalidSpecifier as exc:
        raise ConfigError(
            f"invalid require_mbt_version specifier: {specifier!r} ({exc})",
            path=project_path,
            hint="use a PEP 440 specifier, e.g. '>=0.1,<0.2'",
        ) from exc
    if Version(mbt.__version__) not in spec_set:
        raise ConfigError(
            f"this project requires mbt {specifier}, but mbt {mbt.__version__} is installed",
            path=project_path,
            hint="upgrade or downgrade mbt-core, or relax require_mbt_version",
        )


def load_project(project_dir: Path) -> ProjectConfig:
    """Load and validate ``mbt_project.yml`` from a project directory."""
    project_path = project_dir / PROJECT_FILE
    if not project_path.is_file():
        raise ConfigError(
            f"no {PROJECT_FILE} found in {project_dir}",
            path=project_dir,
            hint="run inside an mbt project, pass --project-dir, or scaffold one with 'mbt init'",
        )
    try:
        raw = yaml.safe_load(project_path.read_text()) or {}
    except yaml.YAMLError as exc:
        raise ConfigError(f"invalid YAML in {PROJECT_FILE}: {exc}", path=project_path) from exc
    except UnicodeDecodeError as exc:
        raise ConfigError(
            f"{PROJECT_FILE} is not valid UTF-8: {exc}",
            path=project_path,
            hint="config files must be UTF-8 encoded text",
        ) from exc
    if not isinstance(raw, dict):
        raise ConfigError(
            f"{PROJECT_FILE} must be a YAML mapping",
            path=project_path,
        )
    try:
        config = ProjectConfig.model_validate(raw)
    except ValidationError as exc:
        raise ConfigError(
            f"invalid {PROJECT_FILE}: {exc}",
            path=project_path,
        ) from exc
    if config.require_mbt_version:
        _check_version_requirement(config.require_mbt_version, project_path)
    return config
