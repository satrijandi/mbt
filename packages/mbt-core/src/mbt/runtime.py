"""Shared runtime wiring: adapters and stores from profiles (compile + execute)."""

from pathlib import Path
from typing import Any

from mbt.adapters.registry import AdapterRegistry
from mbt.config.profiles import LoadedProfiles
from mbt.contracts import AdapterRef


def normalized_adapter_config(
    ref: AdapterRef, project_dir: Path, *, path_keys: tuple[str, ...] = ("root",)
) -> dict[str, Any]:
    """Resolve relative filesystem paths in adapter config against the project dir."""
    config = dict(ref.config)
    for key in path_keys:
        value = config.get(key)
        if isinstance(value, str) and not value.startswith(("s3://", "gs://", "file://")):
            path = Path(value)
            if not path.is_absolute():
                config[key] = str((project_dir / path).resolve())
    return config


def data_adapter(profiles: LoadedProfiles, project_dir: Path, registry: AdapterRegistry) -> Any:
    ref = profiles.target.data
    return registry.component("data", ref.adapter, normalized_adapter_config(ref, project_dir))


def tracking_adapter_config(
    ref: AdapterRef, project_dir: Path, project_name: str
) -> dict[str, Any]:
    """Adapter config with the tracking experiment name composed into it.

    Two names, both the user's: the project (``name:`` in mbt_project.yml) and
    the experiment (``experiment:`` in the target's tracking config, which the
    profiles pre-render already lets you write as ``env('EXPERIMENT_NAME')``).
    They compose to ``<project>_<experiment>``, so one tracking server holds
    many projects without collision and one project holds many modelling
    efforts. With no ``experiment:`` the project name stands alone, which is
    why a fresh project's runs land under its own name rather than a literal
    ``mbt`` (ADR-28).

    Composing here rather than inside each tracking adapter keeps the rule in
    one place and applies it to every tracker, not only MLflow.
    """
    config = normalized_adapter_config(ref, project_dir)
    declared = config.get("experiment")
    if isinstance(declared, str) or declared is None:
        # An empty string counts as unset: `env('EXPERIMENT_NAME', '')` with
        # the variable absent must not produce a trailing-underscore name, or
        # worse an experiment named "".
        config["experiment"] = f"{project_name}_{declared}" if declared else project_name
    # Anything else (a mapping, a number) is left untouched so the adapter
    # raises its own error naming the shape it wanted.
    return config


def tracking_adapter(
    profiles: LoadedProfiles, project_dir: Path, registry: AdapterRegistry, project_name: str
) -> Any:
    ref = profiles.target.tracking
    return registry.component(
        "tracking", ref.adapter, tracking_adapter_config(ref, project_dir, project_name)
    )


def registry_adapter(profiles: LoadedProfiles, project_dir: Path, registry: AdapterRegistry) -> Any:
    ref = profiles.target.registry
    return registry.component("registry", ref.adapter, normalized_adapter_config(ref, project_dir))


def compute_adapter(profiles: LoadedProfiles, registry: AdapterRegistry) -> Any:
    ref = profiles.target.compute
    return registry.component("compute", ref.adapter, dict(ref.config))


def resolve_artifact_store_uri(uri: str, project_dir: Path) -> str:
    """Make relative file:// artifact-store URIs absolute against the project."""
    if uri.startswith("file://"):
        raw = uri.removeprefix("file://")
        path = Path(raw)
        if not path.is_absolute():
            path = (project_dir / raw).resolve()
        return f"file://{path}"
    return uri
