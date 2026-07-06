"""``mbt compile``: ParsedProject + profiles -> manifest (TSD §8.1).

Compilation touches data systems only for snapshot IDs (cheap metadata
calls, parallelized across sources); it never reads data.
"""

import hashlib
import time
from concurrent.futures import ThreadPoolExecutor
from dataclasses import dataclass
from datetime import UTC, datetime
from importlib.metadata import PackageNotFoundError, version
from typing import Any

import mbt
from mbt.adapters.registry import AdapterRegistry, get_registry
from mbt.artifacts.manifest import (
    AdapterVersion,
    GitInfo,
    Manifest,
    ManifestExposure,
    ManifestMetadata,
    ManifestSource,
)
from mbt.compile.hashing import config_hash, env_digest, input_hash
from mbt.compile.windows import format_ts, parse_window
from mbt.config.profiles import LoadedProfiles
from mbt.contracts import DatasetSpec, ManifestNode, ModelSpec, SplitStrategy
from mbt.dag.graph import topological_order
from mbt.events import get_bus
from mbt.events.models import AdapterWarning, CompileCompleted, CompileStarted
from mbt.exceptions import CompilationError, ConfigError
from mbt.gitinfo import collect_git_info
from mbt.jinja.environment import ResolveContext, TargetContext
from mbt.parsing.errors import ParseReport
from mbt.parsing.project_parser import ParsedProject, ParsedResource, validate_hyperparameters
from mbt.runtime import data_adapter as build_data_adapter
from mbt_adapter_base.materialization import combine_snapshots


@dataclass(frozen=True)
class CompileOptions:
    anchor: datetime | None = None
    deep_snapshot: bool = False


def _now_anchor() -> datetime:
    return datetime.now(tz=UTC).replace(microsecond=0)


def compile_project(
    parsed: ParsedProject,
    profiles: LoadedProfiles,
    *,
    registry: AdapterRegistry | None = None,
    options: CompileOptions | None = None,
    cli_vars: dict[str, Any] | None = None,
) -> Manifest:
    """Run the full compile pipeline of TSD §8.1 and return the manifest."""
    started = time.monotonic()
    registry = registry or get_registry()
    options = options or CompileOptions()
    cli_vars = dict(cli_vars or {})
    bus = get_bus()
    bus.emit(CompileStarted(target=profiles.target_name))

    anchor = (options.anchor or _now_anchor()).astimezone(UTC).replace(microsecond=0)
    anchor_iso = format_ts(anchor)

    resolve_ctx = _build_resolve_context(parsed, profiles, cli_vars)
    report = ParseReport()

    # 1. resolve-render every node against the target
    rendered_datasets = {
        uid: _resolve_dataset(res, parsed, resolve_ctx, anchor, report)
        for uid, res in parsed.datasets.items()
    }
    rendered_models = {
        uid: _resolve_model(res, parsed, resolve_ctx, anchor, registry, report)
        for uid, res in parsed.models.items()
    }
    report.raise_if_errors()

    # 2. pin data snapshots per source (parallel, cheap metadata calls)
    snapshots = _pin_snapshots(parsed, profiles, registry, options.deep_snapshot)

    # 3. assemble nodes and hash them in topological order
    nodes: dict[str, ManifestNode] = {}
    for uid, res in parsed.datasets.items():
        spec, config, resolved = rendered_datasets[uid]
        pinned = spec.snapshot or _dataset_snapshot(res, parsed, snapshots)
        if spec.snapshot is not None:
            current = _dataset_snapshot(res, parsed, snapshots)
            if current is not None and current != spec.snapshot:
                bus.emit(
                    AdapterWarning(
                        adapter="local",
                        unique_id=uid,
                        message=(
                            f"explicit snapshot pin {spec.snapshot} is no longer current "
                            f"(current: {current})"
                        ),
                    )
                )
        nodes[uid] = ManifestNode(
            unique_id=uid,
            resource_type="dataset",
            name=res.name,
            path=res.path,
            depends_on=res.depends_on,
            config=config,
            resolved=resolved,
            snapshot_id=pinned,
        )
    for uid, res in parsed.models.items():
        model_spec, config, resolved = rendered_models[uid]
        hooks_hash: str | None = None
        if res.hooks_path is not None:
            hooks_bytes = (parsed.project_dir / res.hooks_path).read_bytes()
            hooks_hash = "sha256:" + hashlib.sha256(hooks_bytes).hexdigest()
        nodes[uid] = ManifestNode(
            unique_id=uid,
            resource_type="model",
            name=res.name,
            path=res.path,
            depends_on=res.depends_on,
            config=config,
            resolved=resolved,
            adapter=model_spec.adapter,
            task=model_spec.task.value,
            seed=model_spec.seed,
            hooks_path=res.hooks_path,
            hooks_hash=hooks_hash,
        )

    _hash_nodes(parsed, nodes)

    # 4. environment digest and adapter versions
    _preload_target_plugins(profiles, registry)
    digest = env_digest(registry.fingerprint_packages())
    adapter_versions = _adapter_versions(parsed, profiles, registry)

    manifest = Manifest(
        metadata=ManifestMetadata(
            mbt_version=mbt.__version__,
            project_name=parsed.project.name,
            target=profiles.target_name,
            generated_at=anchor_iso,  # == anchor by design (FR-COMP-04)
            anchor=anchor_iso,
            vars=_visible_vars(parsed, profiles, cli_vars),
            deep_snapshot=options.deep_snapshot,
            target_config=profiles.raw_target,
            env_digest=digest,
            git=GitInfo.model_validate(collect_git_info(parsed.project_dir)),
        ),
        nodes=nodes,
        sources={
            uid: ManifestSource(
                unique_id=uid,
                group=entry.group,
                name=entry.table.name,
                path=entry.path,
                config=entry.table,
                snapshot_id=snapshots.get(uid),
            )
            for uid, entry in parsed.sources.items()
        },
        exposures={
            uid: ManifestExposure(
                unique_id=uid,
                name=res.name,
                path=res.path,
                config=res.spec.model_dump(mode="json"),
                depends_on=res.depends_on,
            )
            for uid, res in parsed.exposures.items()
        },
        metrics={name: spec.model_dump(mode="json") for name, spec in parsed.metrics.items()},
        adapter_versions=adapter_versions,
    )
    bus.emit(
        CompileCompleted(
            nodes=len(nodes),
            anchor=anchor_iso,
            elapsed_s=time.monotonic() - started,
        )
    )
    return manifest


# -- rendering ----------------------------------------------------------------


def _build_resolve_context(
    parsed: ParsedProject, profiles: LoadedProfiles, cli_vars: dict[str, Any]
) -> ResolveContext:
    dataset_by_name = {r.name: r.unique_id for r in parsed.datasets.values()}
    model_by_name = {r.name: r.unique_id for r in parsed.models.values()}
    source_by_pair = {(e.group, e.table.name): e.unique_id for e in parsed.sources.values()}

    def ref_resolver(name: str) -> str:
        uid = dataset_by_name.get(name) or model_by_name.get(name)
        if uid is None:
            raise CompilationError(
                f"ref('{name}') does not resolve to a known resource",
                hint="run 'mbt parse' for the full error listing",
            )
        return uid

    def source_resolver(group: str, table: str) -> str:
        uid = source_by_pair.get((group, table))
        if uid is None:
            raise CompilationError(f"source('{group}', '{table}') is not declared")
        return uid

    return ResolveContext(
        target=TargetContext(name=profiles.target_name, threads=profiles.target.threads),
        cli_vars=cli_vars,
        target_vars=profiles.target.vars,
        project_vars=parsed.project.vars,
        ref_resolver=ref_resolver,
        source_resolver=source_resolver,
    )


def _resolve_dataset(
    res: ParsedResource,
    parsed: ParsedProject,
    ctx: ResolveContext,
    anchor: datetime,
    report: ParseReport,
) -> tuple[DatasetSpec, dict[str, Any], dict[str, Any]]:
    rendered = parsed.renderer.resolve(
        res.raw, ctx, resource=res.unique_id, path=parsed.project_dir / res.path
    )
    try:
        spec = DatasetSpec.model_validate(rendered)
    except Exception as exc:
        raise CompilationError(
            f"dataset config invalid after target rendering: {exc}",
            resource=res.unique_id,
            path=res.path,
        ) from exc
    resolved: dict[str, Any] = {}
    if spec.split.strategy is SplitStrategy.TEMPORAL:
        windows: dict[str, list[str]] = {}
        for split_name in ("train", "test", "validation"):
            expression = getattr(spec.split, split_name)
            if expression is None:
                continue
            start, end = parse_window(expression).resolve(anchor)
            windows[split_name] = [format_ts(start), format_ts(end)]
        resolved["windows"] = windows
    return spec, spec.model_dump(mode="json"), resolved


def _resolve_model(
    res: ParsedResource,
    parsed: ParsedProject,
    ctx: ResolveContext,
    anchor: datetime,
    registry: AdapterRegistry,
    report: ParseReport,
) -> tuple[ModelSpec, dict[str, Any], dict[str, Any]]:
    rendered = parsed.renderer.resolve(
        res.raw, ctx, resource=res.unique_id, path=parsed.project_dir / res.path
    )
    try:
        spec = ModelSpec.model_validate(rendered)
    except Exception as exc:
        raise CompilationError(
            f"model config invalid after target rendering: {exc}",
            resource=res.unique_id,
            path=res.path,
        ) from exc
    adapter = registry.training(spec.adapter)
    validate_hyperparameters(
        adapter,
        spec.task,
        spec.hyperparameters,
        resource=res.unique_id,
        rel=res.path,
        report=report,
        phase="compile",
    )
    resolved: dict[str, Any] = {}
    if spec.evaluation.protocol.test_window is not None:
        start, end = parse_window(spec.evaluation.protocol.test_window).resolve(anchor)
        resolved["test_window"] = [format_ts(start), format_ts(end)]
    return spec, spec.model_dump(mode="json"), resolved


# -- snapshots ------------------------------------------------------------------


def _pin_snapshots(
    parsed: ParsedProject,
    profiles: LoadedProfiles,
    registry: AdapterRegistry,
    deep: bool,
) -> dict[str, str | None]:
    """Snapshot every source referenced by at least one dataset (TSD §8.3)."""
    referenced: set[str] = set()
    for dataset in parsed.datasets.values():
        referenced.update(dep for dep in dataset.depends_on if dep.startswith("source."))
    if not referenced:
        return {}
    adapter = build_data_adapter(profiles, parsed.project_dir, registry)

    def snapshot(uid: str) -> tuple[str, str | None]:
        entry = parsed.sources[uid]
        try:
            return uid, adapter.snapshot_id(entry.table, deep=deep)
        except Exception as exc:
            raise CompilationError(
                f"snapshot pinning failed for source '{entry.group}.{entry.table.name}': {exc}",
                resource=uid,
                hint="check the source path and the data adapter config in profiles.yml",
            ) from exc

    with ThreadPoolExecutor(max_workers=min(8, len(referenced))) as pool:
        return dict(pool.map(snapshot, sorted(referenced)))


def _dataset_snapshot(
    res: ParsedResource, parsed: ParsedProject, snapshots: dict[str, str | None]
) -> str | None:
    """One pinned snapshot per dataset; multi-source datasets combine all of
    their tables' snapshots so any input changing flips identity (ADR-4)."""
    return combine_snapshots({dep: snapshots[dep] for dep in res.depends_on if dep in snapshots})


# -- hashing ---------------------------------------------------------------------


def _hash_nodes(parsed: ParsedProject, nodes: dict[str, ManifestNode]) -> None:
    """config_hash per node, then input_hash in topological order (TSD §8.4)."""
    for node in nodes.values():
        hooks_bytes = None
        if node.hooks_path is not None:
            hooks_bytes = (parsed.project_dir / node.hooks_path).read_bytes()
        node.config_hash = config_hash(node.config, hooks_bytes)
    for uid in topological_order(parsed.graph, subset=set(nodes)):
        node = nodes[uid]
        upstream = [nodes[dep].input_hash for dep in node.depends_on if dep in nodes]
        node.input_hash = input_hash(node.config_hash, node.snapshot_id, upstream)


# -- environment ------------------------------------------------------------------


def _preload_target_plugins(profiles: LoadedProfiles, registry: AdapterRegistry) -> None:
    for ref in (
        profiles.target.data,
        profiles.target.tracking,
        profiles.target.registry,
        profiles.target.compute,
    ):
        try:
            registry.get(ref.adapter)
        except ConfigError:
            # Missing execution adapters surface with better context at run
            # time; compile does not require them beyond the data adapter.
            continue


def _adapter_versions(
    parsed: ParsedProject, profiles: LoadedProfiles, registry: AdapterRegistry
) -> dict[str, AdapterVersion]:
    names: set[str] = set()
    for res in parsed.models.values():
        spec = res.spec
        assert isinstance(spec, ModelSpec)
        names.add(spec.adapter)
        if spec.tuning is not None:
            names.add(spec.tuning.engine)
    names.add(profiles.target.data.adapter)
    out: dict[str, AdapterVersion] = {}
    for name in sorted(names):
        try:
            plugin = registry.get(name)
        except ConfigError:
            continue
        package = "mbt-core" if name == "local" else f"mbt-{name}"
        try:
            package_version = version(package)
        except PackageNotFoundError:
            package_version = "unknown"
        out[name] = AdapterVersion(
            package=package, version=package_version, contract=plugin.contract_version
        )
    return out


def _visible_vars(
    parsed: ParsedProject, profiles: LoadedProfiles, cli_vars: dict[str, Any]
) -> dict[str, Any]:
    merged: dict[str, Any] = {}
    merged.update(parsed.project.vars)
    merged.update(profiles.target.vars)
    merged.update(cli_vars)
    return merged
