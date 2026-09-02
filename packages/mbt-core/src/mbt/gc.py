"""Artifact-store retention: prune old run prefixes, keep what matters.

``mbt clean --artifacts-older-than`` prunes file:// stores only; object
stores have native lifecycle rules, which are the right tool there
(docs/gitops.md). The keep-set always includes the latest run's artifacts
and every stage champion of every registered model: deleting a champion
artifact would make champion re-evaluation fail hard (ADR-10).
"""

import json
from dataclasses import dataclass
from datetime import UTC, datetime
from pathlib import Path
from typing import Any

from mbt.artifacts.run_results import command_results_path
from mbt.exceptions import MbtError


@dataclass(frozen=True)
class GcPlan:
    delete: list[Path]
    keep: list[Path]
    freed_bytes: int


def run_results_artifact_uris(project_dir: Path) -> set[str]:
    """Artifact URIs recorded by the latest *training* run.

    Prefers the per-command siblings over the shared ``run_results.json``:
    only training commands record artifacts, and a ``mbt score`` in between
    would otherwise blank this keep-set out (FEEDBACK v3 A-2). Champions are
    protected separately by :func:`champion_artifact_uris` (ADR-10), so this
    is defence in depth for the not-yet-promoted artifact of a recent build.

    Parsed as raw JSON rather than through the pydantic model on purpose:
    ``mbt clean`` must not hard-fail on a results file written by another mbt
    version, the same tolerance ``mbt monitor`` gives a malformed prediction
    sidecar (R2-19). A file it cannot read contributes nothing to the
    keep-set, and the champion half still protects what matters.
    """
    results_path = project_dir / "target" / "run_results.json"
    candidates = [
        path
        for path in (command_results_path(results_path, c) for c in ("build", "run", "evaluate"))
        if path.is_file()
    ]
    chosen = max(candidates, key=lambda p: p.stat().st_mtime) if candidates else results_path
    if not chosen.is_file():
        return set()
    try:
        payload = json.loads(chosen.read_text())
    except (json.JSONDecodeError, UnicodeDecodeError):
        return set()
    return {
        entry["artifact"]["uri"] for entry in payload.get("results", []) if entry.get("artifact")
    }


def champion_artifact_uris(parsed: Any, registry_adapter: Any) -> set[str]:
    """Artifact URIs of every stage champion of every registered model."""
    from mbt.contracts import Stage

    keep: set[str] = set()
    for resource in parsed.models.values():
        registration = getattr(resource.spec, "registration", None)
        if registration is None:
            continue
        for stage in Stage:
            champion = registry_adapter.get_champion(registration.name, stage)
            if champion is not None and champion.artifact is not None:
                keep.add(champion.artifact.uri)
    return keep


def artifact_gc_plan(store_uri: str, *, cutoff: datetime, keep_uris: set[str]) -> GcPlan:
    """Plan the prune: run-prefix directories older than ``cutoff`` and not
    holding any kept artifact are deletable."""
    if not store_uri.startswith("file://"):
        raise MbtError(
            f"artifact GC supports file:// stores, got {store_uri!r}",
            hint="use bucket lifecycle rules for s3:// stores (docs/gitops.md)",
        )
    root = Path(store_uri.removeprefix("file://"))
    if not root.is_dir():
        return GcPlan(delete=[], keep=[], freed_bytes=0)
    keep_paths = {
        Path(uri.removeprefix("file://")).resolve()
        for uri in keep_uris
        if uri.startswith("file://")
    }
    delete: list[Path] = []
    keep: list[Path] = []
    freed = 0
    for prefix_dir in sorted(p for p in root.iterdir() if p.is_dir()):
        files = [f for f in prefix_dir.rglob("*") if f.is_file()]
        newest = max((f.stat().st_mtime for f in files), default=0.0)
        referenced = any(f.resolve() in keep_paths for f in files)
        if referenced or datetime.fromtimestamp(newest, tz=UTC) >= cutoff:
            keep.append(prefix_dir)
        else:
            delete.append(prefix_dir)
            freed += sum(f.stat().st_size for f in files)
    return GcPlan(delete=delete, keep=keep, freed_bytes=freed)


def apply_gc_plan(plan: GcPlan) -> None:
    import shutil

    for path in plan.delete:
        shutil.rmtree(path)
