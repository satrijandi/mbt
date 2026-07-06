"""Node identity hashing (TSD §8.4, ADR-4, ADR-5).

``config_hash`` covers the canonical JSON of the rendered spec plus the
hooks file bytes. Excluded: ``description``, ``owner``, ``tags`` (cosmetic),
resolved windows and the anchor (ADR-12), and everything from profiles
(ADR-5: environment must not change node identity).

``input_hash`` composes ``config_hash + snapshot_id + sorted upstream
input_hashes`` in topological order: one comparison captures config, hooks,
snapshot, and upstream changes (ADR-4).
"""

import hashlib
import sys
from importlib.metadata import PackageNotFoundError, distributions, version
from typing import Any

from mbt.utils import canonical_json

#: Spec fields that never affect node identity.
HASH_EXCLUDED_FIELDS = frozenset({"description", "owner", "tags"})


def _sha256(*chunks: bytes) -> str:
    digest = hashlib.sha256()
    for chunk in chunks:
        digest.update(chunk)
    return "sha256:" + digest.hexdigest()


def config_hash(rendered_config: dict[str, Any], hooks_bytes: bytes | None = None) -> str:
    """Identity hash of one node's rendered spec (+ hooks file bytes)."""
    hashable = {k: v for k, v in rendered_config.items() if k not in HASH_EXCLUDED_FIELDS}
    return _sha256(canonical_json(hashable).encode("utf-8"), hooks_bytes or b"")


def input_hash(
    node_config_hash: str,
    snapshot_id: str | None,
    upstream_input_hashes: list[str],
) -> str:
    """Transitive identity: everything that affects the trained artifact."""
    parts = [node_config_hash, snapshot_id or "", *sorted(upstream_input_hashes)]
    return _sha256("|".join(parts).encode("utf-8"))


def env_digest(fingerprint_packages: list[str]) -> str:
    """Environment digest: Python + mbt packages + adapter fingerprints (FR-COMP-03)."""
    lines = [f"python=={sys.version_info.major}.{sys.version_info.minor}.{sys.version_info.micro}"]
    packages = set(fingerprint_packages)
    for dist in distributions():
        name = (dist.metadata["Name"] or "").lower()
        if name in {"mbt-core", "mbt-adapter-base"} or name.startswith("mbt-"):
            packages.add(name)
    for package in sorted(packages):
        try:
            lines.append(f"{package}=={version(package)}")
        except PackageNotFoundError:
            lines.append(f"{package}==(not installed)")
    return _sha256("\n".join(lines).encode("utf-8"))


def manifest_hash(manifest_payload: dict[str, Any]) -> str:
    """Hash of the canonical manifest with volatile metadata blanked (TSD §8.5)."""
    payload = dict(manifest_payload)
    metadata = dict(payload.get("metadata", {}))
    metadata["generated_at"] = ""
    metadata["anchor"] = ""
    payload["metadata"] = metadata
    return _sha256(canonical_json(payload).encode("utf-8"))
