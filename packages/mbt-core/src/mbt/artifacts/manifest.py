"""The compiled manifest: format, writer, reader (TSD §8.5, FR-COMP-01..05).

Determinism (FR-COMP-04): two compiles of the same project at the same
anchor produce byte-identical files. ``generated_at`` equals the anchor by
design - the anchor *is* the compile timestamp unless overridden - so the
only volatile fields are isolated in ``metadata`` and blanked for hashing.
"""

import json
from pathlib import Path
from typing import Any

import networkx as nx
from pydantic import BaseModel, ConfigDict, Field, ValidationError

from mbt.compile.hashing import manifest_hash
from mbt.contracts import ManifestNode, SourceTable
from mbt.dag.selector import SelectableNode
from mbt.exceptions import StateError
from mbt.secrets import redact

MANIFEST_SCHEMA_VERSION = 1
#: Core reads schema N and N-1 (TSD §19).
_READABLE_VERSIONS = (MANIFEST_SCHEMA_VERSION, MANIFEST_SCHEMA_VERSION - 1)


class GitInfo(BaseModel):
    model_config = ConfigDict(extra="forbid")

    commit: str | None = None
    branch: str | None = None
    dirty: bool = False


class ManifestMetadata(BaseModel):
    model_config = ConfigDict(extra="forbid")

    manifest_schema_version: int = MANIFEST_SCHEMA_VERSION
    mbt_version: str
    project_name: str
    target: str
    generated_at: str  # volatile; equals anchor for fresh compiles
    anchor: str  # volatile (ADR-12)
    vars: dict[str, Any] = Field(default_factory=dict)  # resolved, secrets excluded
    #: Whether snapshots were pinned with content hashing (--deep-snapshot).
    deep_snapshot: bool = False
    #: The selected target's config, UNRENDERED: env_var() expressions stay
    #: as written so secrets never enter the manifest (TSD §18, ADR-5).
    target_config: dict[str, Any] = Field(default_factory=dict)
    env_digest: str = ""
    git: GitInfo = Field(default_factory=GitInfo)


class ManifestSource(BaseModel):
    model_config = ConfigDict(extra="forbid")

    unique_id: str
    group: str
    name: str
    path: str  # spec file, relative
    config: SourceTable
    snapshot_id: str | None = None


class ManifestExposure(BaseModel):
    model_config = ConfigDict(extra="forbid")

    unique_id: str
    name: str
    path: str
    config: dict[str, Any]
    depends_on: list[str] = Field(default_factory=list)


class AdapterVersion(BaseModel):
    model_config = ConfigDict(extra="forbid")

    package: str
    version: str
    contract: str


class Manifest(BaseModel):
    model_config = ConfigDict(extra="forbid")

    metadata: ManifestMetadata
    nodes: dict[str, ManifestNode] = Field(default_factory=dict)
    sources: dict[str, ManifestSource] = Field(default_factory=dict)
    exposures: dict[str, ManifestExposure] = Field(default_factory=dict)
    metrics: dict[str, dict[str, Any]] = Field(default_factory=dict)
    adapter_versions: dict[str, AdapterVersion] = Field(default_factory=dict)

    # -- identity ----------------------------------------------------------

    def manifest_hash(self) -> str:
        return manifest_hash(self.model_dump(mode="json"))

    # -- graph views ---------------------------------------------------------

    def graph(self) -> "nx.DiGraph":
        """Reconstruct the DAG (including source and exposure nodes)."""
        graph = nx.DiGraph()
        for uid in self.sources:
            graph.add_node(uid, resource_type="source")
        for uid, node in self.nodes.items():
            graph.add_node(uid, resource_type=node.resource_type)
            for dep in node.depends_on:
                graph.add_edge(dep, uid)
        for uid, exposure in self.exposures.items():
            graph.add_node(uid, resource_type="exposure")
            for dep in exposure.depends_on:
                graph.add_edge(dep, uid)
        return graph

    def selectable_nodes(self) -> dict[str, SelectableNode]:
        """All selectable resources (nodes + sources + exposures)."""
        out: dict[str, SelectableNode] = {}
        for uid, node in self.nodes.items():
            out[uid] = SelectableNode(
                unique_id=uid,
                name=node.name,
                resource_type=node.resource_type,
                tags=tuple(node.tags),
            )
        for uid, source in self.sources.items():
            out[uid] = SelectableNode(
                unique_id=uid, name=source.name, resource_type="source", tags=()
            )
        for uid, exposure in self.exposures.items():
            tags = exposure.config.get("tags", [])
            out[uid] = SelectableNode(
                unique_id=uid,
                name=exposure.name,
                resource_type="exposure",
                tags=tuple(tags) if isinstance(tags, list) else (),
            )
        return out

    # -- serialization -------------------------------------------------------

    def to_json(self) -> str:
        payload = self.model_dump(mode="json")
        return redact(json.dumps(payload, indent=2, sort_keys=True) + "\n")

    def write(self, path: Path) -> None:
        path.parent.mkdir(parents=True, exist_ok=True)
        path.write_text(self.to_json())


def read_manifest(path_or_text: Path | str, *, source: str = "manifest") -> Manifest:
    """Read and validate a manifest, enforcing schema compatibility (TSD §19)."""
    if isinstance(path_or_text, Path):
        if not path_or_text.is_file():
            raise StateError(
                f"{source} not found: {path_or_text}",
                hint="run 'mbt compile' first, or check the --state/--manifest path",
            )
        text = path_or_text.read_text()
        location = str(path_or_text)
    else:
        text = path_or_text
        location = source
    try:
        payload = json.loads(text)
    except json.JSONDecodeError as exc:
        raise StateError(f"invalid JSON in {source}: {exc}", path=location) from exc
    schema_version = payload.get("metadata", {}).get("manifest_schema_version")
    if schema_version not in _READABLE_VERSIONS:
        raise StateError(
            f"{source} has manifest_schema_version {schema_version}; this mbt-core reads "
            f"{' and '.join(str(v) for v in _READABLE_VERSIONS if v > 0)}",
            path=location,
            hint="upgrade mbt-core to read manifests from newer releases",
        )
    try:
        return Manifest.model_validate(payload)
    except ValidationError as exc:
        raise StateError(f"invalid {source}: {exc}", path=location) from exc
