"""Shared builders for the misc unit-test cluster (unique module name)."""

from typing import Any

from mbt.artifacts.manifest import Manifest, ManifestExposure, ManifestMetadata
from mbt.contracts import ManifestNode


class RecordingSink:
    """Collects events; satisfies both Sink (write) and EventSink (emit)."""

    def __init__(self) -> None:
        self.events: list[Any] = []

    def write(self, event: object) -> None:
        self.events.append(event)

    def emit(self, event: object) -> None:
        self.events.append(event)


def make_metadata(**overrides: Any) -> ManifestMetadata:
    defaults: dict[str, Any] = {
        "mbt_version": "0.1.0",
        "project_name": "demo",
        "target": "dev",
        "generated_at": "2026-07-01T00:00:00Z",
        "anchor": "2026-07-01T00:00:00Z",
    }
    defaults.update(overrides)
    return ManifestMetadata(**defaults)


def make_node(uid: str, **overrides: Any) -> ManifestNode:
    prefix = uid.split(".", 1)[0]
    defaults: dict[str, Any] = {
        "unique_id": uid,
        "resource_type": prefix if prefix in ("dataset", "model", "scoring") else "model",
        "name": uid.rsplit(".", 1)[-1],
        "path": f"{uid}.yml",
        "config": {},
    }
    defaults.update(overrides)
    return ManifestNode(**defaults)


def make_manifest(
    *nodes: ManifestNode,
    exposures: dict[str, ManifestExposure] | None = None,
    metadata: ManifestMetadata | None = None,
) -> Manifest:
    return Manifest(
        metadata=metadata or make_metadata(),
        nodes={node.unique_id: node for node in nodes},
        exposures=exposures or {},
    )
