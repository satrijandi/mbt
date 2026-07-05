"""``profiles.yml``: environments/targets (TSD §5.3, FR-PROJ-03).

Search order: ``--profiles-dir``, ``$MBT_PROFILES_DIR``, ``./profiles.yml``,
``~/.mbt/profiles.yml`` - first hit wins. Jinja (``env_var``, ``var``) is
rendered *before* validation; rendered secret values are tainted and the
unrendered mapping is kept for the manifest (TSD §18).
"""

import os
from dataclasses import dataclass
from pathlib import Path
from typing import Any

import jinja2
import yaml
from pydantic import BaseModel, ConfigDict, Field, ValidationError

from mbt.contracts import AdapterRef
from mbt.exceptions import ConfigError
from mbt.secrets import taint

PROFILES_FILE = "profiles.yml"


class TargetConfig(BaseModel):
    """One named environment (dev/staging/prod)."""

    model_config = ConfigDict(extra="forbid")

    data: AdapterRef
    tracking: AdapterRef
    registry: AdapterRef
    compute: AdapterRef = Field(default_factory=lambda: AdapterRef(adapter="local"))
    artifact_store: str  # URI: file://, s3://
    threads: int = Field(default=1, ge=1)
    vars: dict[str, Any] = Field(default_factory=dict)


class ProfilesConfig(BaseModel):
    """The per-project block inside profiles.yml."""

    model_config = ConfigDict(extra="forbid")

    target: str  # default target name
    outputs: dict[str, TargetConfig]


@dataclass(frozen=True)
class LoadedProfiles:
    """Rendered profiles plus what the manifest and jobs need."""

    path: Path
    config: ProfilesConfig
    target_name: str
    target: TargetConfig
    raw_target: dict[str, Any]  # unrendered mapping for the manifest (TSD §18)
    required_env: list[str]  # env_var names referenced by the selected target


def find_profiles_path(project_dir: Path, profiles_dir: Path | None) -> Path:
    """Resolve profiles.yml per the documented search order."""
    candidates: list[Path] = []
    if profiles_dir is not None:
        candidates.append(profiles_dir / PROFILES_FILE)
    env_dir = os.environ.get("MBT_PROFILES_DIR")
    if env_dir:
        candidates.append(Path(env_dir) / PROFILES_FILE)
    candidates.append(project_dir / PROFILES_FILE)
    candidates.append(Path.home() / ".mbt" / PROFILES_FILE)
    for candidate in candidates:
        if candidate.is_file():
            return candidate
    raise ConfigError(
        "no profiles.yml found",
        hint=(
            "searched: "
            + ", ".join(str(c) for c in candidates)
            + ". Create one (see 'mbt init') or pass --profiles-dir."
        ),
    )


def _render_profiles_text(
    text: str, path: Path, cli_vars: dict[str, Any], project_vars: dict[str, Any]
) -> tuple[str, list[str]]:
    """Render Jinja in profiles.yml; returns (rendered, env var names used)."""
    used_env: list[str] = []
    _missing = object()

    def env_var(name: str, default: str | None = None) -> str:
        used_env.append(name)
        value = os.environ.get(name, _missing)  # type: ignore[arg-type]
        if value is _missing:
            if default is None:
                raise ConfigError(
                    f"environment variable {name!r} referenced in profiles.yml is not set",
                    path=path,
                    hint=f"export {name}=... or provide a default: env_var('{name}', 'fallback')",
                )
            return default
        return taint(str(value))

    def var(name: str, default: Any = _missing) -> Any:
        if name in cli_vars:
            return cli_vars[name]
        if name in project_vars:
            return project_vars[name]
        if default is _missing:
            raise ConfigError(
                f"var {name!r} referenced in profiles.yml has no value",
                path=path,
                hint=f"pass --vars '{name}: <value>' or define it in mbt_project.yml vars",
            )
        return default

    env = jinja2.Environment(undefined=jinja2.StrictUndefined, autoescape=False)  # noqa: S701
    try:
        return env.from_string(text).render(env_var=env_var, var=var), sorted(set(used_env))
    except ConfigError:
        raise
    except jinja2.TemplateError as exc:
        raise ConfigError(f"invalid Jinja in profiles.yml: {exc}", path=path) from exc


def load_profiles(
    project_name: str,
    project_dir: Path,
    profiles_dir: Path | None = None,
    target_override: str | None = None,
    cli_vars: dict[str, Any] | None = None,
    project_vars: dict[str, Any] | None = None,
) -> LoadedProfiles:
    """Load, render, validate profiles.yml and select a target."""
    path = find_profiles_path(project_dir, profiles_dir)
    text = path.read_text()

    def parse(source: str, label: str) -> dict[str, Any]:
        try:
            data = yaml.safe_load(source) or {}
        except yaml.YAMLError as exc:
            raise ConfigError(f"invalid YAML in {label} profiles.yml: {exc}", path=path) from exc
        if not isinstance(data, dict):
            raise ConfigError(f"{label} profiles.yml must be a YAML mapping", path=path)
        return data

    raw_file = parse(text, "unrendered")
    rendered_text, used_env = _render_profiles_text(
        text, path, dict(cli_vars or {}), dict(project_vars or {})
    )
    rendered_file = parse(rendered_text, "rendered")

    if project_name not in rendered_file:
        raise ConfigError(
            f"profiles.yml has no entry for project {project_name!r}",
            path=path,
            hint=f"available: {', '.join(sorted(rendered_file)) or '(none)'}",
        )
    try:
        config = ProfilesConfig.model_validate(rendered_file[project_name])
    except ValidationError as exc:
        raise ConfigError(f"invalid profiles.yml: {exc}", path=path) from exc

    target_name = target_override or config.target
    if target_name not in config.outputs:
        raise ConfigError(
            f"target {target_name!r} not defined for project {project_name!r}",
            path=path,
            hint=f"available targets: {', '.join(sorted(config.outputs))}",
        )

    raw_project = raw_file.get(project_name, {})
    raw_outputs = raw_project.get("outputs", {}) if isinstance(raw_project, dict) else {}
    raw_target = raw_outputs.get(target_name, {}) if isinstance(raw_outputs, dict) else {}

    return LoadedProfiles(
        path=path,
        config=config,
        target_name=target_name,
        target=config.outputs[target_name],
        raw_target=raw_target if isinstance(raw_target, dict) else {},
        required_env=used_env,
    )
