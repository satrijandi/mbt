"""The parsing pipeline: files -> validated resources -> DAG (TSD §7).

Collects *all* errors in one pass (FR-PARSE-02) and needs neither profiles
nor environment (capture-phase Jinja, TSD §6).
"""

import re
import time
from collections.abc import Mapping
from copy import deepcopy
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any

import networkx as nx
from pydantic import BaseModel

from mbt.adapters.registry import AdapterRegistry, get_registry
from mbt.config.project import ProjectConfig, load_project
from mbt.config.tasks import get_task_schema
from mbt.dag.graph import build_graph, find_cycle
from mbt.exceptions import ConfigError
from mbt.ids import source_unique_id, unique_id
from mbt.jinja.environment import SpecRenderer
from mbt.parsing.errors import ParseReport
from mbt.parsing.loader import (
    TOP_LEVEL_KEYS,
    check_top_level_keys,
    load_yaml_mapping,
    validate_resource,
)
from mbt.parsing.rules import Phase, RuleContext, RuleTarget, run_rules
from mbt.quality.metrics import resolve_metric, resolve_model_metrics
from mbt.quality.python_tests import PythonTestFile, discover_python_tests
from mbt.utils import did_you_mean
from mbt_adapter_base import (
    DatasetSpec,
    ExposureSpec,
    MetricSpec,
    ModelSpec,
    ScoringSpec,
    SourceGroup,
    SourceTable,
)

_REF_RE = re.compile(r"^\s*ref\(\s*['\"](?P<name>[^'\"]+)['\"]\s*\)\s*$")

#: Root-level resource files picked up by convention (plus configured paths).
_ROOT_FILES = (
    "sources.yml",
    "sources.yaml",
    "metrics.yml",
    "metrics.yaml",
    "exposures.yml",
    "exposures.yaml",
)


@dataclass(frozen=True)
class SourceEntry:
    """One source table with its group context."""

    unique_id: str
    group: str
    table: SourceTable
    path: str  # spec file, relative


@dataclass
class ParsedResource:
    """A validated resource plus everything parsing learned about it."""

    unique_id: str
    resource_type: str  # "dataset" | "model" | "scoring" | "exposure"
    name: str
    path: str  # spec file, relative to project dir
    spec: BaseModel
    raw: dict[str, Any]  # original YAML mapping (pre-Jinja), for resolve phase
    refs: list[str] = field(default_factory=list)  # captured ref() names
    sources: list[tuple[str, str]] = field(default_factory=list)
    depends_on: list[str] = field(default_factory=list)  # resolved unique_ids
    hooks_path: str | None = None  # models only, relative to project dir
    #: Models: resolved evaluation metrics. Scoring: resolved ground-truth metrics.
    metric_specs: list[MetricSpec] = field(default_factory=list)

    @property
    def tags(self) -> list[str]:
        return list(getattr(self.spec, "tags", []))


@dataclass
class ParsedProject:
    """The output of ``mbt parse``: resources + DAG (TSD §7)."""

    project: ProjectConfig
    project_dir: Path
    sources: dict[str, SourceEntry]
    datasets: dict[str, ParsedResource]
    models: dict[str, ParsedResource]
    scoring: dict[str, ParsedResource]
    exposures: dict[str, ParsedResource]
    metrics: dict[str, MetricSpec]  # by metric name
    graph: nx.DiGraph
    renderer: SpecRenderer
    python_tests: list[PythonTestFile]
    report: ParseReport
    elapsed_s: float

    @property
    def nodes(self) -> dict[str, ParsedResource]:
        """Compiled DAG nodes: datasets, models, and scoring pipelines."""
        return {**self.datasets, **self.models, **self.scoring}

    def resource(self, name_or_uid: str) -> ParsedResource | SourceEntry | None:
        for pool in (self.datasets, self.models, self.scoring, self.exposures, self.sources):
            if name_or_uid in pool:
                return pool[name_or_uid]
        for pool in (self.datasets, self.models, self.scoring, self.exposures):
            for res in pool.values():
                if res.name == name_or_uid:
                    return res
        for entry in self.sources.values():
            if entry.table.name == name_or_uid:
                return entry
        return None

    def all_names(self) -> list[str]:
        names = [
            r.name
            for r in (
                *self.datasets.values(),
                *self.models.values(),
                *self.scoring.values(),
                *self.exposures.values(),
            )
        ]
        names.extend(e.table.name for e in self.sources.values())
        return names


def _merge_defaults(defaults: dict[str, Any], raw: dict[str, Any]) -> dict[str, Any]:
    """Project model_defaults < spec (TSD §8.1); dicts merge one level deep."""
    merged = deepcopy(defaults)
    for key, value in raw.items():
        if key in merged and isinstance(merged[key], dict) and isinstance(value, dict):
            merged[key] = {**merged[key], **value}
        else:
            merged[key] = value
    return merged


def parse_project(
    project_dir: Path,
    *,
    registry: AdapterRegistry | None = None,
    raise_on_error: bool = True,
    cli_vars: dict[str, Any] | None = None,
) -> ParsedProject:
    """Parse and validate a whole project; collect all errors before failing."""
    started = time.monotonic()
    project_dir = project_dir.resolve()
    project = load_project(project_dir)
    registry = registry or get_registry()
    report = ParseReport()
    cli_vars = dict(cli_vars or {})

    try:
        renderer = SpecRenderer(macro_paths=[project_dir / p for p in project.macro_paths])
    except ConfigError as exc:
        report.error(exc.message, file=exc.path or "", hint=exc.hint)
        renderer = SpecRenderer(macro_paths=None)

    raw_resources = _discover_and_load(project_dir, project, report)

    sources = _parse_sources(raw_resources.get("source", []), project.name, report)
    metrics = _parse_metrics(raw_resources.get("metric", []), report)
    datasets = _parse_datasets(
        raw_resources.get("dataset", []), project, renderer, report, project_dir, cli_vars
    )
    models = _parse_models(
        raw_resources.get("model", []), project, renderer, registry, report, project_dir, cli_vars
    )
    scoring = _parse_scoring(
        raw_resources.get("scoring", []), project, renderer, report, project_dir, cli_vars
    )
    exposures = _parse_exposures(raw_resources.get("exposure", []), project, renderer, report)

    _link(
        project=project,
        sources=sources,
        datasets=datasets,
        models=models,
        scoring=scoring,
        exposures=exposures,
        metrics=metrics,
        registry=registry,
        report=report,
    )

    # Every cross-resource invariant, from one list (A-3). The SAME list runs
    # again at compile time over the target-rendered specs, which is what keeps
    # a target var from moving spec.target or a gate's source past every check
    # (compile/compiler.py).
    run_rules(rule_context(project.name, datasets, models, scoring, registry), report)

    graph = _build_project_graph(sources, datasets, models, scoring, exposures, report)
    python_tests = discover_python_tests(project_dir, project.test_paths, report)
    _check_test_bindings(datasets, python_tests, report)

    if raise_on_error:
        report.raise_if_errors()

    return ParsedProject(
        project=project,
        project_dir=project_dir,
        sources=sources,
        datasets=datasets,
        models=models,
        scoring=scoring,
        exposures=exposures,
        metrics=metrics,
        graph=graph,
        renderer=renderer,
        python_tests=python_tests,
        report=report,
        elapsed_s=time.monotonic() - started,
    )


# -- discovery ---------------------------------------------------------------


def _in_hidden_dir(path: Path, resource_dir: Path) -> bool:
    """True if `path` sits under a dot-directory BELOW `resource_dir`.

    Discovery rglobs the configured resource dirs, and rglob descends into
    dot-directories. Editors and tooling keep copies of the very files we parse
    there - JupyterLab writes `.ipynb_checkpoints/<name>-checkpoint.yml` as soon
    as a DS opens a spec in its editor - so every such copy re-declared its
    resource and the parse died with a `duplicate <kind>` error that blamed the
    REAL file (the checkpoint sorts first, so it registers first).

    Only components below `resource_dir` are tested, so a project that
    deliberately configures a hidden resource dir still works.
    """
    return any(part.startswith(".") for part in path.relative_to(resource_dir).parts[:-1])


def _discover_and_load(
    project_dir: Path, project: ProjectConfig, report: ParseReport
) -> dict[str, list[tuple[str, int, dict[str, Any]]]]:
    """Walk configured paths; returns resource_type -> [(rel, index, raw), ...]."""
    files: list[Path] = []
    for name in _ROOT_FILES:
        candidate = project_dir / name
        if candidate.is_file():
            files.append(candidate)
    for path_list in (project.dataset_paths, project.model_paths, project.scoring_paths):
        for dir_name in path_list:
            resource_dir = project_dir / dir_name
            if not resource_dir.is_dir():
                continue
            for pattern in ("*.yml", "*.yaml"):
                found = resource_dir.rglob(pattern)
                files.extend(sorted(p for p in found if not _in_hidden_dir(p, resource_dir)))

    out: dict[str, list[tuple[str, int, dict[str, Any]]]] = {t: [] for t in TOP_LEVEL_KEYS.values()}
    for path in sorted(set(files)):
        rel = str(path.relative_to(project_dir))
        raw = load_yaml_mapping(path, rel, report)
        if raw is None:
            continue
        check_top_level_keys(raw, rel, report)
        for key, resource_type in TOP_LEVEL_KEYS.items():
            entries = raw.get(key)
            if not isinstance(entries, list):
                continue
            for index, entry in enumerate(entries):
                if not isinstance(entry, dict):
                    report.error(
                        f"entry {index} under '{key}' must be a mapping",
                        file=rel,
                        field_path=f"/{key}/{index}",
                    )
                    continue
                out[resource_type].append((rel, index, entry))
    return out


# -- per-type parsing --------------------------------------------------------


def _parse_sources(
    entries: list[tuple[str, int, dict[str, Any]]], project_name: str, report: ParseReport
) -> dict[str, SourceEntry]:
    sources: dict[str, SourceEntry] = {}
    for rel, index, raw in entries:
        group = validate_resource(
            SourceGroup,
            raw,
            rel=rel,
            resource_name=str(raw.get("name", f"#{index}")),
            base_pointer=f"/sources/{index}",
            report=report,
        )
        if group is None:
            continue
        for table in group.tables:
            uid = source_unique_id(project_name, group.name, table.name)
            if uid in sources:
                report.error(
                    f"duplicate source table '{group.name}.{table.name}'",
                    file=rel,
                    resource=uid,
                )
                continue
            sources[uid] = SourceEntry(unique_id=uid, group=group.name, table=table, path=rel)
    return sources


def _parse_metrics(
    entries: list[tuple[str, int, dict[str, Any]]], report: ParseReport
) -> dict[str, MetricSpec]:
    metrics: dict[str, MetricSpec] = {}
    for rel, index, raw in entries:
        spec = validate_resource(
            MetricSpec,
            raw,
            rel=rel,
            resource_name=str(raw.get("name", f"#{index}")),
            base_pointer=f"/metrics/{index}",
            report=report,
        )
        if spec is None:
            continue
        if spec.name in metrics:
            report.error(f"duplicate metric {spec.name!r}", file=rel, resource=spec.name)
            continue
        metrics[spec.name] = spec
    return metrics


def _parse_datasets(
    entries: list[tuple[str, int, dict[str, Any]]],
    project: ProjectConfig,
    renderer: SpecRenderer,
    report: ParseReport,
    project_dir: Path,
    cli_vars: dict[str, Any],
) -> dict[str, ParsedResource]:
    datasets: dict[str, ParsedResource] = {}
    for rel, index, raw in entries:
        name = str(raw.get("name", f"#{index}"))
        uid = unique_id("dataset", project.name, name) if _valid_name(name) else name
        try:
            captured = renderer.capture(
                raw,
                resource=uid,
                path=project_dir / rel,
                cli_vars=cli_vars,
                project_vars=project.vars,
            )
        except ConfigError as exc:
            report.error(exc.message, file=rel, resource=uid, hint=exc.hint)
            continue
        spec = validate_resource(
            DatasetSpec,
            captured.rendered,
            rel=rel,
            resource_name=name,
            base_pointer=f"/datasets/{index}",
            report=report,
        )
        if spec is None:
            continue
        uid = unique_id("dataset", project.name, spec.name)
        if uid in datasets:
            report.error(f"duplicate dataset {spec.name!r}", file=rel, resource=uid)
            continue

        datasets[uid] = ParsedResource(
            unique_id=uid,
            resource_type="dataset",
            name=spec.name,
            path=rel,
            spec=spec,
            raw=raw,
            refs=captured.refs,
            sources=captured.sources,
        )
    return datasets


def _valid_name(name: str) -> bool:
    return re.fullmatch(r"[a-z][a-z0-9_]*", name) is not None


def _parse_models(
    entries: list[tuple[str, int, dict[str, Any]]],
    project: ProjectConfig,
    renderer: SpecRenderer,
    registry: AdapterRegistry,
    report: ParseReport,
    project_dir: Path,
    cli_vars: dict[str, Any],
) -> dict[str, ParsedResource]:
    models: dict[str, ParsedResource] = {}
    for rel, index, entry in entries:
        raw = _merge_defaults(project.model_defaults, entry)
        name = str(raw.get("name", f"#{index}"))
        uid = unique_id("model", project.name, name) if _valid_name(name) else name
        try:
            captured = renderer.capture(
                raw,
                resource=uid,
                path=project_dir / rel,
                cli_vars=cli_vars,
                project_vars=project.vars,
            )
        except ConfigError as exc:
            report.error(exc.message, file=rel, resource=uid, hint=exc.hint)
            continue
        spec = validate_resource(
            ModelSpec,
            captured.rendered,
            rel=rel,
            resource_name=name,
            base_pointer=f"/models/{index}",
            report=report,
        )
        if spec is None:
            continue
        uid = unique_id("model", project.name, spec.name)
        if uid in models:
            report.error(f"duplicate model {spec.name!r}", file=rel, resource=uid)
            continue

        hooks_path = _detect_hooks(spec, rel, uid, project_dir, report)

        models[uid] = ParsedResource(
            unique_id=uid,
            resource_type="model",
            name=spec.name,
            path=rel,
            spec=spec,
            raw=raw,
            refs=captured.refs,
            sources=captured.sources,
            hooks_path=hooks_path,
        )
    return models


def _detect_hooks(
    spec: ModelSpec, rel: str, uid: str, project_dir: Path, report: ParseReport
) -> str | None:
    if spec.hooks is not None:
        hooks_file = project_dir / spec.hooks
        if not hooks_file.is_file():
            report.error(
                f"hooks file {spec.hooks!r} does not exist",
                file=rel,
                resource=uid,
                field_path="/hooks",
                hint="the path is relative to the project directory",
            )
            return None
        return spec.hooks
    sibling = (project_dir / rel).parent / f"{spec.name}.py"
    if sibling.is_file():
        return str(sibling.relative_to(project_dir))
    return None


def _parse_exposures(
    entries: list[tuple[str, int, dict[str, Any]]],
    project: ProjectConfig,
    renderer: SpecRenderer,
    report: ParseReport,
) -> dict[str, ParsedResource]:
    exposures: dict[str, ParsedResource] = {}
    for rel, index, raw in entries:
        name = str(raw.get("name", f"#{index}"))
        spec = validate_resource(
            ExposureSpec,
            raw,
            rel=rel,
            resource_name=name,
            base_pointer=f"/exposures/{index}",
            report=report,
        )
        if spec is None:
            continue
        uid = unique_id("exposure", project.name, spec.name)
        if uid in exposures:
            report.error(f"duplicate exposure {spec.name!r}", file=rel, resource=uid)
            continue
        refs = []
        for dep in spec.depends_on:
            match = _REF_RE.match(dep)
            if match:
                refs.append(match.group("name"))
            else:
                report.error(
                    f"exposure depends_on entries must be ref() calls, got {dep!r}",
                    file=rel,
                    resource=uid,
                    field_path="/depends_on",
                )
        exposures[uid] = ParsedResource(
            unique_id=uid,
            resource_type="exposure",
            name=spec.name,
            path=rel,
            spec=spec,
            raw=raw,
            refs=refs,
        )
    return exposures


def _parse_scoring(
    entries: list[tuple[str, int, dict[str, Any]]],
    project: ProjectConfig,
    renderer: SpecRenderer,
    report: ParseReport,
    project_dir: Path,
    cli_vars: dict[str, Any],
) -> dict[str, ParsedResource]:
    scoring: dict[str, ParsedResource] = {}
    for rel, index, raw in entries:
        name = str(raw.get("name", f"#{index}"))
        uid = unique_id("scoring", project.name, name) if _valid_name(name) else name
        try:
            captured = renderer.capture(
                raw,
                resource=uid,
                path=project_dir / rel,
                cli_vars=cli_vars,
                project_vars=project.vars,
            )
        except ConfigError as exc:
            report.error(exc.message, file=rel, resource=uid, hint=exc.hint)
            continue
        spec = validate_resource(
            ScoringSpec,
            captured.rendered,
            rel=rel,
            resource_name=name,
            base_pointer=f"/scoring/{index}",
            report=report,
        )
        if spec is None:
            continue
        uid = unique_id("scoring", project.name, spec.name)
        if uid in scoring:
            report.error(f"duplicate scoring pipeline {spec.name!r}", file=rel, resource=uid)
            continue

        scoring[uid] = ParsedResource(
            unique_id=uid,
            resource_type="scoring",
            name=spec.name,
            path=rel,
            spec=spec,
            raw=raw,
            refs=captured.refs,
            sources=captured.sources,
        )
    return scoring


# -- cross-resource checks -----------------------------------------------------


def rule_context(
    project_name: str,
    datasets: Mapping[str, RuleTarget],
    models: Mapping[str, RuleTarget],
    scoring: Mapping[str, RuleTarget],
    registry: AdapterRegistry,
    *,
    phase: Phase = "parse",
    depends_on: dict[str, list[str]] | None = None,
) -> RuleContext:
    """A RuleContext over linked resources.

    At parse time a ``ParsedResource`` IS a ``RuleTarget`` and carries its own
    ``depends_on``; at compile time the targets hold target-rendered specs and
    the links come from the parsed project, so ``depends_on`` is explicit.
    """
    if depends_on is None:
        depends_on = {
            res.unique_id: list(getattr(res, "depends_on", ()))
            for pool in (datasets, models, scoring)
            for res in pool.values()
        }
    return RuleContext(
        project_name=project_name,
        datasets=dict(datasets),
        models=dict(models),
        scoring=dict(scoring),
        depends_on=depends_on,
        registry=registry,
        phase=phase,
    )


def _link(
    *,
    project: ProjectConfig,
    sources: dict[str, SourceEntry],
    datasets: dict[str, ParsedResource],
    models: dict[str, ParsedResource],
    scoring: dict[str, ParsedResource],
    exposures: dict[str, ParsedResource],
    metrics: dict[str, MetricSpec],
    registry: AdapterRegistry,
    report: ParseReport,
) -> None:
    dataset_by_name = {r.name: r for r in datasets.values()}
    model_by_name = {r.name: r for r in models.values()}
    scoring_by_name = {r.name: r for r in scoring.values()}

    for dataset in datasets.values():
        deps: list[str] = []
        for group, table in dataset.sources:
            uid = source_unique_id(project.name, group, table)
            if uid not in sources:
                known = sorted(f"{e.group}.{e.table.name}" for e in sources.values())
                report.error(
                    f"unknown source ('{group}', '{table}')",
                    file=dataset.path,
                    resource=dataset.unique_id,
                    field_path="/source",
                    hint=f"declared sources: {', '.join(known) or '(none)'}",
                )
            else:
                deps.append(uid)
        for ref_name in dataset.refs:
            report.error(
                f"datasets cannot ref() other resources, got ref('{ref_name}')",
                file=dataset.path,
                resource=dataset.unique_id,
                hint="datasets read from source() tables in v0",
            )
        dataset.depends_on = sorted(set(deps))

    for model in models.values():
        spec = model.spec
        assert isinstance(spec, ModelSpec)
        deps = []
        dataset_res = _check_model_dataset_edge(spec, model, dataset_by_name, model_by_name, report)
        if dataset_res is not None:
            deps.append(dataset_res.unique_id)
        for group, table in model.sources:
            report.error(
                f"models cannot use source() directly, got ('{group}', '{table}')",
                file=model.path,
                resource=model.unique_id,
                hint="reference data through a dataset resource",
            )
        model.depends_on = sorted(set(deps))

        _resolve_model_metric_specs(spec, model, metrics, report)

    for sc in scoring.values():
        sc_spec = sc.spec
        assert isinstance(sc_spec, ScoringSpec)
        deps = []
        model_res = _check_scoring_model_edge(
            sc_spec, sc, model_by_name, dataset_by_name, scoring_by_name, report
        )
        if model_res is not None:
            deps.append(model_res.unique_id)
            _resolve_scoring_metric_specs(sc_spec, sc, model_res, metrics, report)
        for group, table in sc.sources:
            source_uid = source_unique_id(project.name, group, table)
            if source_uid not in sources:
                known = sorted(f"{e.group}.{e.table.name}" for e in sources.values())
                report.error(
                    f"unknown source ('{group}', '{table}')",
                    file=sc.path,
                    resource=sc.unique_id,
                    hint=f"declared sources: {', '.join(known) or '(none)'}",
                )
            else:
                deps.append(source_uid)
        sc.depends_on = sorted(set(deps))

    for exposure in exposures.values():
        deps = []
        for ref_name in exposure.refs:
            resource = (
                model_by_name.get(ref_name)
                or dataset_by_name.get(ref_name)
                or scoring_by_name.get(ref_name)
            )
            if resource is None:
                report.error(
                    f"exposure references unknown resource ref('{ref_name}')",
                    file=exposure.path,
                    resource=exposure.unique_id,
                    field_path="/depends_on",
                )
            else:
                deps.append(resource.unique_id)
        exposure.depends_on = sorted(set(deps))


def _check_model_dataset_edge(
    spec: ModelSpec,
    model: ParsedResource,
    dataset_by_name: dict[str, ParsedResource],
    model_by_name: dict[str, ParsedResource],
    report: ParseReport,
) -> ParsedResource | None:
    match = _REF_RE.match(spec.dataset)
    if match is None:
        report.error(
            f"model 'dataset' must be a ref() call, got {spec.dataset!r}",
            file=model.path,
            resource=model.unique_id,
            field_path="/dataset",
            hint="e.g. dataset: ref('churn_training_set')",
        )
        return None
    ref_name = match.group("name")
    if ref_name in model_by_name:
        report.error(
            "model -> model references are not supported in v0",
            file=model.path,
            resource=model.unique_id,
            field_path="/dataset",
            hint="ensembles/stacking arrive in v1 (FR-V1-05)",
        )
        return None
    dataset_res = dataset_by_name.get(ref_name)
    if dataset_res is None:
        suggestion = did_you_mean(ref_name, sorted(dataset_by_name))
        report.error(
            f"model references unknown dataset ref('{ref_name}')",
            file=model.path,
            resource=model.unique_id,
            field_path="/dataset",
            hint=f"did you mean {suggestion!r}?" if suggestion else None,
        )
        return None
    # Extra refs beyond the dataset edge are rejected for clarity.
    for extra in model.refs:
        if extra != ref_name:
            report.error(
                f"unexpected ref('{extra}') in model spec",
                file=model.path,
                resource=model.unique_id,
                hint="v0 models may only ref() their dataset",
            )
    return dataset_res


def _check_scoring_model_edge(
    spec: ScoringSpec,
    sc: ParsedResource,
    model_by_name: dict[str, ParsedResource],
    dataset_by_name: dict[str, ParsedResource],
    scoring_by_name: dict[str, ParsedResource],
    report: ParseReport,
) -> ParsedResource | None:
    match = _REF_RE.match(spec.model)
    if match is None:
        report.error(
            f"scoring 'model' must be a ref() call, got {spec.model!r}",
            file=sc.path,
            resource=sc.unique_id,
            field_path="/model",
            hint="e.g. model: ref('churn_classifier')",
        )
        return None
    ref_name = match.group("name")
    if ref_name in dataset_by_name or ref_name in scoring_by_name:
        report.error(
            f"scoring 'model' must reference a model, got ref('{ref_name}')",
            file=sc.path,
            resource=sc.unique_id,
            field_path="/model",
        )
        return None
    model_res = model_by_name.get(ref_name)
    if model_res is None:
        suggestion = did_you_mean(ref_name, sorted(model_by_name))
        report.error(
            f"scoring references unknown model ref('{ref_name}')",
            file=sc.path,
            resource=sc.unique_id,
            field_path="/model",
            hint=f"did you mean {suggestion!r}?" if suggestion else None,
        )
        return None
    for extra in sc.refs:
        if extra != ref_name:
            report.error(
                f"unexpected ref('{extra}') in scoring spec",
                file=sc.path,
                resource=sc.unique_id,
                hint="a scoring pipeline may only ref() its model",
            )
    return model_res


def _resolve_scoring_metric_specs(
    spec: ScoringSpec,
    sc: ParsedResource,
    model_res: ParsedResource,
    metrics: dict[str, MetricSpec],
    report: ParseReport,
) -> None:
    """Ground-truth metrics must be builtins for the model's task (ADR-21).

    Hook metrics need a training adapter inside a job; ``mbt monitor`` runs
    coordinator-side against realized labels, so only builtins qualify.
    """
    if spec.ground_truth is None:
        return
    model_spec = model_res.spec
    assert isinstance(model_spec, ModelSpec)
    try:
        task_schema = get_task_schema(model_spec.task)
    except ConfigError:
        return  # already reported against the model
    resolved: list[MetricSpec] = []
    for name in spec.ground_truth.metrics:
        outcome = resolve_metric(name, metrics, task_schema, has_hooks=False)
        if isinstance(outcome, str):
            report.error(
                outcome, file=sc.path, resource=sc.unique_id, field_path="/ground_truth/metrics"
            )
        elif outcome.kind != "builtin":
            report.error(
                f"ground-truth metric {name!r} must be a builtin; hook metrics "
                "are computed by training jobs, not by 'mbt monitor'",
                file=sc.path,
                resource=sc.unique_id,
                field_path="/ground_truth/metrics",
            )
        else:
            resolved.append(outcome)
    sc.metric_specs = resolved


def _resolve_model_metric_specs(
    spec: ModelSpec, model: ParsedResource, metrics: dict[str, MetricSpec], report: ParseReport
) -> None:
    try:
        task_schema = get_task_schema(spec.task)
    except ConfigError:
        return  # already reported by _check_adapter
    resolved, errors = resolve_model_metrics(
        spec, metrics, task_schema, has_hooks=model.hooks_path is not None
    )
    model.metric_specs = resolved
    for message in errors:
        report.error(
            message, file=model.path, resource=model.unique_id, field_path="/evaluation/metrics"
        )


# -- graph ---------------------------------------------------------------------


def _build_project_graph(
    sources: dict[str, SourceEntry],
    datasets: dict[str, ParsedResource],
    models: dict[str, ParsedResource],
    scoring: dict[str, ParsedResource],
    exposures: dict[str, ParsedResource],
    report: ParseReport,
) -> nx.DiGraph:
    node_types: dict[str, str] = dict.fromkeys(sources, "source")
    edges: dict[str, list[str]] = {}
    for pool in (datasets, models, scoring, exposures):
        for uid, resource in pool.items():
            node_types[uid] = resource.resource_type
            edges[uid] = resource.depends_on
    graph = build_graph(edges, node_types)
    cycle = find_cycle(graph)
    if cycle is not None:
        report.error(
            "dependency cycle detected: " + " -> ".join(cycle),
            hint="break the cycle by removing one of the ref() edges above",
        )
    return graph


def _check_test_bindings(
    datasets: dict[str, ParsedResource], python_tests: list[PythonTestFile], report: ParseReport
) -> None:
    all_test_names = {name for tf in python_tests for name in tf.test_names}
    for dataset in datasets.values():
        spec = dataset.spec
        assert isinstance(spec, DatasetSpec)
        for test_name in spec.tests:
            if test_name not in all_test_names:
                report.error(
                    f"dataset lists unknown data test {test_name!r}",
                    file=dataset.path,
                    resource=dataset.unique_id,
                    field_path="/tests",
                    hint=f"discovered tests: {', '.join(sorted(all_test_names)) or '(none)'}",
                )
