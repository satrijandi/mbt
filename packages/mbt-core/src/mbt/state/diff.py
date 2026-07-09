"""Manifest diffing (TSD §9.3, §14.1; FR-STATE-01/02; ADR-4, ADR-7).

A node is *modified* iff its ``input_hash`` differs from the same unique_id
in the reference manifest; *new* iff absent there. Because ``input_hash`` is
transitive, one comparison captures config, hooks, snapshot, and upstream
changes. ``env_digest`` changes never mark nodes modified by default; the
diff reports them prominently instead (ADR-7).
"""

from dataclasses import dataclass, field
from typing import Any

from mbt.artifacts.manifest import Manifest, read_manifest
from mbt.storage import read_uri_text


def load_state(uri_or_path: str) -> Manifest:
    """Read a reference manifest from file://, s3://, or a bare path."""
    return read_manifest(read_uri_text(uri_or_path), source=f"state manifest ({uri_or_path})")


class ManifestStateIndex:
    """Answers state:new / state:modified for selector evaluation."""

    def __init__(
        self, current: Manifest, reference: Manifest, *, include_env: bool = False
    ) -> None:
        self._current = current
        self._reference = reference
        self._env_changed = current.metadata.env_digest != reference.metadata.env_digest
        self._include_env = include_env

    @property
    def env_changed(self) -> bool:
        return self._env_changed

    def is_new(self, unique_id: str) -> bool:
        if unique_id not in self._current.nodes:
            return False
        return unique_id not in self._reference.nodes

    def is_modified(self, unique_id: str) -> bool:
        node = self._current.nodes.get(unique_id)
        if node is None:
            return False
        reference = self._reference.nodes.get(unique_id)
        if reference is None:
            return True  # new counts as modified for state:modified selection
        if self._include_env and self._env_changed:
            return True
        return node.input_hash != reference.input_hash


@dataclass(frozen=True)
class NodeDiff:
    unique_id: str
    change: str  # "added" | "removed" | "modified"
    components: tuple[str, ...] = ()  # config | hooks | snapshot | upstream

    def to_dict(self) -> dict[str, Any]:
        return {
            "unique_id": self.unique_id,
            "change": self.change,
            "components": list(self.components),
        }


@dataclass
class StateDiff:
    added: list[NodeDiff] = field(default_factory=list)
    removed: list[NodeDiff] = field(default_factory=list)
    modified: list[NodeDiff] = field(default_factory=list)
    env_changed: bool = False
    env_digest_current: str = ""
    env_digest_reference: str = ""
    env_freeze_digest_current: str = ""
    env_freeze_digest_reference: str = ""

    @property
    def is_empty(self) -> bool:
        return not (self.added or self.removed or self.modified)

    def to_dict(self) -> dict[str, Any]:
        return {
            "added": [d.to_dict() for d in self.added],
            "removed": [d.to_dict() for d in self.removed],
            "modified": [d.to_dict() for d in self.modified],
            "env": {
                "changed": self.env_changed,
                "current": self.env_digest_current,
                "reference": self.env_digest_reference,
                "freeze_current": self.env_freeze_digest_current,
                "freeze_reference": self.env_freeze_digest_reference,
            },
        }


def _components(current: Any, reference: Any) -> tuple[str, ...]:
    """Attribute *which* part of a node changed (TSD §14.1)."""
    components: list[str] = []
    hooks_changed = current.hooks_hash != reference.hooks_hash
    if hooks_changed:
        # config_hash covers hooks bytes, so a hooks edit also flips it;
        # attribute it to "hooks" alone (a simultaneous spec edit cannot be
        # told apart cheaply and rarely matters for review).
        components.append("hooks")
    elif current.config_hash != reference.config_hash:
        components.append("config")
    if current.snapshot_id != reference.snapshot_id:
        components.append("snapshot")
    if (
        current.config_hash == reference.config_hash
        and current.snapshot_id == reference.snapshot_id
        and current.input_hash != reference.input_hash
    ):
        components.append("upstream")
    return tuple(components) or ("config",)


def diff_manifests(current: Manifest, reference: Manifest) -> StateDiff:
    diff = StateDiff(
        # ADR-7: the "modifying" env signal stays keyed to the targeted
        # env_digest; the freeze digest is reported for visibility only.
        env_changed=current.metadata.env_digest != reference.metadata.env_digest,
        env_digest_current=current.metadata.env_digest,
        env_digest_reference=reference.metadata.env_digest,
        env_freeze_digest_current=current.metadata.env_freeze_digest,
        env_freeze_digest_reference=reference.metadata.env_freeze_digest,
    )
    for uid in sorted(current.nodes):
        if uid not in reference.nodes:
            diff.added.append(NodeDiff(unique_id=uid, change="added"))
        else:
            node, ref_node = current.nodes[uid], reference.nodes[uid]
            if node.input_hash != ref_node.input_hash:
                diff.modified.append(
                    NodeDiff(
                        unique_id=uid,
                        change="modified",
                        components=_components(node, ref_node),
                    )
                )
    for uid in sorted(reference.nodes):
        if uid not in current.nodes:
            diff.removed.append(NodeDiff(unique_id=uid, change="removed"))
    return diff
