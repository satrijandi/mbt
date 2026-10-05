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

from mbt.exceptions import ConfigError
from mbt.secrets import taint
from mbt.yamlio import safe_load
from mbt_adapter_base import (
    AdapterRef,
)

PROFILES_FILE = "profiles.yml"


class TargetConfig(BaseModel):
    """One named environment (dev/staging/prod)."""

    model_config = ConfigDict(extra="forbid")

    data: AdapterRef
    tracking: AdapterRef
    registry: AdapterRef
    compute: AdapterRef = Field(default_factory=lambda: AdapterRef(adapter="local"))
    #: Optional tuning-engine ops config (sampler, pruner knobs); the engine is
    #: named by the model's tuning spec, so only ``config`` here is consumed.
    tuning: AdapterRef | None = None
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


#: What an unset, default-less ``env()``/``env_var()`` renders as. Never seen
#: by an adapter: ``_raise_for_unset_env`` fails the load first whenever the
#: selected target (or the project block around it) references the variable.
_UNSET_ENV = "__mbt_unset_env_{}__"


def _render_profiles_text(
    text: str, path: Path, cli_vars: dict[str, Any], project_vars: dict[str, Any]
) -> tuple[str, list[str], dict[str, str]]:
    """Render Jinja in profiles.yml.

    Returns the rendered text, the env var names used, and the names that were
    unset with no default (mapped to the function that read them). Those are
    not an error yet: the whole file renders at once, so raising here made
    ``prod``'s ``env('MBT_DATA_ROOT')`` break every ``dev`` command.
    ``load_profiles`` raises for the ones the selected target actually reads
    (FEEDBACK v6 A-8).
    """
    used_env: list[str] = []
    unset: dict[str, str] = {}
    _missing = object()

    def lookup_env(name: str, default: str | None, *, secret: bool) -> str:
        used_env.append(name)
        value = os.environ.get(name, _missing)
        if value is _missing:
            if default is None:
                unset.setdefault(name, "env_var" if secret else "env")
                return _UNSET_ENV.format(name)
            return default
        return taint(str(value)) if secret else str(value)

    def env_var(name: str, default: str | None = None) -> str:
        return lookup_env(name, default, secret=True)

    def env(name: str, default: str | None = None) -> str:
        """Non-secret environment value: same lookup, no taint."""
        return lookup_env(name, default, secret=False)

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

    jinja_env = jinja2.Environment(undefined=jinja2.StrictUndefined, autoescape=False)
    try:
        rendered = jinja_env.from_string(text).render(env_var=env_var, env=env, var=var)
        return rendered, sorted(set(used_env)), unset
    except ConfigError:
        raise
    except jinja2.TemplateError as exc:
        raise ConfigError(f"invalid Jinja in profiles.yml: {exc}", path=path) from exc


_ENV_CALL = r"\benv(?:_var)?\(\s*['\"]{}['\"]"


def _strings(value: Any) -> list[str]:
    if isinstance(value, dict):
        return [s for k, v in value.items() for s in (*_strings(k), *_strings(v))]
    if isinstance(value, list):
        return [s for item in value for s in _strings(item)]
    return [value] if isinstance(value, str) else []


def _raise_for_unset_env(
    raw_file: dict[str, Any],
    rendered_file: dict[str, Any],
    project_name: str,
    target_override: str | None,
    unset: dict[str, str],
    path: Path,
) -> None:
    """Fail for an unset, default-less variable the selected target reads.

    Read off the UNRENDERED mapping, so a variable piped through a filter
    (``env('T') | int``) is still caught. Everything in the project block
    except the other targets counts as read: ``target: "{{ env('T') }}"``
    decides which target runs at all. If the block cannot be read that way
    (Jinja around the structure itself), every unset variable counts.
    """
    import re

    raw_project = raw_file.get(project_name)
    rendered_project = rendered_file.get(project_name)
    relevant: Any = raw_file
    if isinstance(raw_project, dict) and isinstance(rendered_project, dict):
        target_name = target_override or rendered_project.get("target")
        raw_outputs = raw_project.get("outputs")
        if isinstance(target_name, str) and isinstance(raw_outputs, dict):
            relevant = {key: value for key, value in raw_project.items() if key != "outputs"}
            relevant["outputs"] = {target_name: raw_outputs.get(target_name)}
    texts = _strings(relevant)
    in_use = sorted(
        name
        for name in unset
        if any(re.search(_ENV_CALL.format(re.escape(name)), text) for text in texts)
    )
    if in_use:
        names = ", ".join(repr(name) for name in in_use)
        noun, verb = ("variable", "is") if len(in_use) == 1 else ("variables", "are")
        raise ConfigError(
            f"environment {noun} {names} referenced in profiles.yml {verb} not set",
            path=path,
            hint=f"export {in_use[0]}=... or provide a default: "
            f"{unset[in_use[0]]}('{in_use[0]}', 'fallback')",
        )


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
    try:
        text = path.read_text()
    except UnicodeDecodeError as exc:
        raise ConfigError(
            f"profiles.yml is not valid UTF-8: {exc}",
            path=path,
            hint="config files must be UTF-8 encoded text",
        ) from exc

    def parse(source: str, label: str) -> dict[str, Any]:
        try:
            data = safe_load(source, source=f"{path} ({label})") or {}
        except yaml.YAMLError as exc:
            raise ConfigError(f"invalid YAML in {label} profiles.yml: {exc}", path=path) from exc
        if not isinstance(data, dict):
            raise ConfigError(f"{label} profiles.yml must be a YAML mapping", path=path)
        return data

    raw_file = parse(text, "unrendered")
    rendered_text, used_env, unset_env = _render_profiles_text(
        text, path, dict(cli_vars or {}), dict(project_vars or {})
    )
    rendered_file = parse(rendered_text, "rendered")
    if unset_env:
        _raise_for_unset_env(
            raw_file, rendered_file, project_name, target_override, unset_env, path
        )

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
