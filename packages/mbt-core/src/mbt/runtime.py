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


def data_adapter(
    profiles: LoadedProfiles, project_dir: Path, registry: AdapterRegistry
) -> Any:
    ref = profiles.target.data
    return registry.component("data", ref.adapter, normalized_adapter_config(ref, project_dir))


def tracking_adapter(
    profiles: LoadedProfiles, project_dir: Path, registry: AdapterRegistry
) -> Any:
    ref = profiles.target.tracking
    return registry.component(
        "tracking", ref.adapter, normalized_adapter_config(ref, project_dir)
    )


def registry_adapter(
    profiles: LoadedProfiles, project_dir: Path, registry: AdapterRegistry
) -> Any:
    ref = profiles.target.registry
    return registry.component(
        "registry", ref.adapter, normalized_adapter_config(ref, project_dir)
    )


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
