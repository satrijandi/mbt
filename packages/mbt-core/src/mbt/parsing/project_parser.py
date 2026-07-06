"""The parsing pipeline: files -> validated resources -> DAG (TSD §7).

Collects *all* errors in one pass (FR-PARSE-02) and needs neither profiles
nor environment (capture-phase Jinja, TSD §6).
"""

import re
import time
from copy import deepcopy
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any

import networkx as nx
from pydantic import BaseModel

from mbt.adapters.registry import AdapterRegistry, get_registry
from mbt.compile.windows import VALIDATION_ANCHOR, is_subrange, parse_window
from mbt.config.project import ProjectConfig, load_project
from mbt.config.tasks import get_task_schema
from mbt.contracts import (
    AUTO,
    DatasetSpec,
    ExposureSpec,
    MetricSpec,
    ModelSpec,
    SourceGroup,
    SourceTable,
    SplitStrategy,
    TaskType,
)
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
from mbt.quality.metrics import resolve_model_metrics
from mbt.quality.python_tests import PythonTestFile, discover_python_tests
from mbt.utils import did_you_mean

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

_BUILTIN_CHECKS = {
    "no_future_columns",
    "label_leakage_scan",
    "class_balance_report",
    "schema",
    "not_null",
}


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
    resource_type: str  # "dataset" | "model" | "exposure"
    name: str
    path: str  # spec file, relative to project dir
    spec: BaseModel
    raw: dict[str, Any]  # original YAML mapping (pre-Jinja), for resolve phase
    refs: list[str] = field(default_factory=list)  # captured ref() names
    sources: list[tuple[str, str]] = field(default_factory=list)
    depends_on: list[str] = field(default_factory=list)  # resolved unique_ids
    hooks_path: str | None = None  # models only, relative to project dir
    metric_specs: list[MetricSpec] = field(default_factory=list)  # models only

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
    exposures: dict[str, ParsedResource]
    metrics: dict[str, MetricSpec]  # by metric name
    graph: nx.DiGraph
    renderer: SpecRenderer
    python_tests: list[PythonTestFile]
    report: ParseReport
    elapsed_s: float

    @property
    def nodes(self) -> dict[str, ParsedResource]:
        """Executable DAG nodes: datasets and models."""
        return {**self.datasets, **self.models}

    def resource(self, name_or_uid: str) -> ParsedResource | SourceEntry | None:
        for pool in (self.datasets, self.models, self.exposures, self.sources):
            if name_or_uid in pool:
                return pool[name_or_uid]
        for pool in (self.datasets, self.models, self.exposures):
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
            for r in (*self.datasets.values(), *self.models.values(), *self.exposures.values())
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
    exposures = _parse_exposures(raw_resources.get("exposure", []), project, renderer, report)

    _link_and_check(
        project=project,
        sources=sources,
        datasets=datasets,
        models=models,
        exposures=exposures,
        metrics=metrics,
        registry=registry,
        report=report,
    )

    graph = _build_project_graph(sources, datasets, models, exposures, report)
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
        exposures=exposures,
        metrics=metrics,
        graph=graph,
        renderer=renderer,
        python_tests=python_tests,
        report=report,
        elapsed_s=time.monotonic() - started,
    )


# -- discovery ---------------------------------------------------------------


def _discover_and_load(
    project_dir: Path, project: ProjectConfig, report: ParseReport
) -> dict[str, list[tuple[str, int, dict[str, Any]]]]:
    """Walk configured paths; returns resource_type -> [(rel, index, raw), ...]."""
    files: list[Path] = []
    for name in _ROOT_FILES:
        candidate = project_dir / name
        if candidate.is_file():
            files.append(candidate)
    for path_list in (project.dataset_paths, project.model_paths):
        for dir_name in path_list:
            resource_dir = project_dir / dir_name
            if resource_dir.is_dir():
                files.extend(sorted(resource_dir.rglob("*.yml")))
                files.extend(sorted(resource_dir.rglob("*.yaml")))

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

        _validate_dataset_windows(spec, rel, uid, report)
        _validate_checks(spec, rel, uid, report)

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


def _validate_dataset_windows(spec: DatasetSpec, rel: str, uid: str, report: ParseReport) -> None:
    if spec.split.strategy is not SplitStrategy.TEMPORAL:
        return
    for split_field in ("train", "test", "validation"):
        expression = getattr(spec.split, split_field)
        if expression is None:
            continue
        try:
            parse_window(expression)
        except ConfigError as exc:
            report.error(
                exc.message,
                file=rel,
                resource=uid,
                field_path=f"/split/{split_field}",
                hint=exc.hint,
            )


def _validate_checks(spec: DatasetSpec, rel: str, uid: str, report: ParseReport) -> None:
    for i, check in enumerate(spec.checks):
        check_name = check if isinstance(check, str) else next(iter(check), "")
        if check_name not in _BUILTIN_CHECKS:
            suggestion = did_you_mean(str(check_name), sorted(_BUILTIN_CHECKS))
            report.error(
                f"unknown dataset check {check_name!r}",
                file=rel,
                resource=uid,
                field_path=f"/checks/{i}",
                hint=f"did you mean {suggestion!r}?"
                if suggestion
                else f"built-in checks: {', '.join(sorted(_BUILTIN_CHECKS))}",
            )


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
        _check_adapter(spec, uid, rel, registry, report)

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


def _check_adapter(
    spec: ModelSpec, uid: str, rel: str, registry: AdapterRegistry, report: ParseReport
) -> None:
    """Adapter installed, task supported, static hyperparameters valid (TSD §7)."""
    try:
        plugin = registry.get(spec.adapter)
    except ConfigError as exc:
        report.error(exc.message, file=rel, resource=uid, field_path="/adapter", hint=exc.hint)
        return
    if plugin.training is None:
        report.error(
            f"adapter {spec.adapter!r} provides no training adapter",
            file=rel,
            resource=uid,
            field_path="/adapter",
        )
        return
    adapter = plugin.training({})
    if spec.task not in adapter.supported_tasks:
        supported = ", ".join(sorted(t.value for t in adapter.supported_tasks))
        report.error(
            f"adapter {spec.adapter!r} does not support task {spec.task.value!r}",
            file=rel,
            resource=uid,
            field_path="/task",
            hint=f"supported tasks: {supported}",
        )
        return
    validate_hyperparameters(
        adapter,
        spec.task,
        spec.hyperparameters,
        resource=uid,
        rel=rel,
        report=report,
        phase="parse",
    )
    for issue in adapter.validate(spec):
        add = report.error if issue.severity == "error" else report.warning
        add(issue.message, file=rel, resource=uid, field_path=issue.field_path, hint=issue.hint)

    try:
        task_schema = get_task_schema(spec.task)
    except ConfigError as exc:
        report.error(exc.message, file=rel, resource=uid, field_path="/task", hint=exc.hint)
        return
    for issue in task_schema.validate_spec(spec):
        add = report.error if issue.severity == "error" else report.warning
        add(issue.message, file=rel, resource=uid, field_path=issue.field_path, hint=issue.hint)


def _is_deferred_value(value: Any) -> bool:
    """Values not statically checkable: AUTO sentinels or unresolved Jinja."""
    if value is None:
        return True
    return isinstance(value, str) and (value == AUTO or "{{" in value or "{%" in value)


def validate_hyperparameters(
    adapter: Any,
    task: TaskType,
    hyperparameters: dict[str, Any],
    *,
    resource: str,
    rel: str,
    report: ParseReport,
    phase: str,
) -> None:
    """Two-step param validation: unknown keys always; values when static.

    At parse time, values still holding Jinja or AUTO sentinels are skipped;
    at compile time only AUTO survives (resolved later by the adapter).
    """
    param_model = adapter.param_model(task)
    known = set(param_model.model_fields)
    static: dict[str, Any] = {}
    for key, value in hyperparameters.items():
        if key not in known:
            suggestion = did_you_mean(key, sorted(known))
            report.error(
                f"unknown hyperparameter {key!r} for adapter "
                f"{adapter.name!r} / task {task.value!r}",
                file=rel,
                resource=resource,
                field_path=f"/hyperparameters/{key}",
                hint=f"did you mean {suggestion!r}?"
                if suggestion
                else f"valid: {', '.join(sorted(known))}",
            )
        elif not _is_deferred_value(value):
            static[key] = value
    if not static:
        return
    validate_resource(
        param_model,
        static,
        rel=rel,
        resource_name=resource,
        base_pointer="/hyperparameters",
        report=report,
    )


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


# -- cross-resource checks -----------------------------------------------------


def _link_and_check(
    *,
    project: ProjectConfig,
    sources: dict[str, SourceEntry],
    datasets: dict[str, ParsedResource],
    models: dict[str, ParsedResource],
    exposures: dict[str, ParsedResource],
    metrics: dict[str, MetricSpec],
    registry: AdapterRegistry,
    report: ParseReport,
) -> None:
    dataset_by_name = {r.name: r for r in datasets.values()}
    model_by_name = {r.name: r for r in models.values()}

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
            _check_model_vs_dataset(spec, model, dataset_res, report)
        for group, table in model.sources:
            report.error(
                f"models cannot use source() directly, got ('{group}', '{table}')",
                file=model.path,
                resource=model.unique_id,
                hint="reference data through a dataset resource",
            )
        model.depends_on = sorted(set(deps))

        _resolve_model_metric_specs(spec, model, metrics, report)
        _check_tuning_engine(spec, model, registry, report)

    for exposure in exposures.values():
        deps = []
        for ref_name in exposure.refs:
            resource = model_by_name.get(ref_name) or dataset_by_name.get(ref_name)
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


def _check_model_vs_dataset(
    spec: ModelSpec, model: ParsedResource, dataset_res: ParsedResource, report: ParseReport
) -> None:
    ds_spec = dataset_res.spec
    assert isinstance(ds_spec, DatasetSpec)
    if spec.target != ds_spec.label.column:
        report.error(
            f"model target {spec.target!r} must equal the dataset's label column "
            f"{ds_spec.label.column!r}",
            file=model.path,
            resource=model.unique_id,
            field_path="/target",
            hint="mismatches are an error, not a silent override (TSD §5.6)",
        )
    if spec.evaluation.protocol.split is not ds_spec.split.strategy:
        report.error(
            f"evaluation.protocol.split ({spec.evaluation.protocol.split.value}) must match "
            f"the dataset's split.strategy ({ds_spec.split.strategy.value}) (FR-RES-09)",
            file=model.path,
            resource=model.unique_id,
            field_path="/evaluation/protocol/split",
            hint="the redundancy is deliberate: it keeps the model spec self-describing",
        )
        return
    test_window = spec.evaluation.protocol.test_window
    if test_window is None:
        return
    if ds_spec.split.strategy is not SplitStrategy.TEMPORAL:
        report.error(
            "evaluation.protocol.test_window requires a temporal split",
            file=model.path,
            resource=model.unique_id,
            field_path="/evaluation/protocol/test_window",
        )
        return
    try:
        inner = parse_window(test_window)
        outer = parse_window(ds_spec.split.test)
    except ConfigError as exc:
        report.error(
            exc.message,
            file=model.path,
            resource=model.unique_id,
            field_path="/evaluation/protocol/test_window",
            hint=exc.hint,
        )
        return
    if not is_subrange(inner, outer, VALIDATION_ANCHOR):
        report.error(
            f"test_window {test_window!r} must resolve to a sub-range of the dataset's "
            f"test window {ds_spec.split.test!r}",
            file=model.path,
            resource=model.unique_id,
            field_path="/evaluation/protocol/test_window",
        )


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


def _check_tuning_engine(
    spec: ModelSpec, model: ParsedResource, registry: AdapterRegistry, report: ParseReport
) -> None:
    if spec.tuning is None:
        return
    try:
        plugin = registry.get(spec.tuning.engine)
    except ConfigError as exc:
        report.error(
            exc.message,
            file=model.path,
            resource=model.unique_id,
            field_path="/tuning/engine",
            hint=exc.hint,
        )
        return
    if plugin.tuning is None:
        report.error(
            f"adapter {spec.tuning.engine!r} provides no tuning engine",
            file=model.path,
            resource=model.unique_id,
            field_path="/tuning/engine",
        )


# -- graph ---------------------------------------------------------------------


def _build_project_graph(
    sources: dict[str, SourceEntry],
    datasets: dict[str, ParsedResource],
    models: dict[str, ParsedResource],
    exposures: dict[str, ParsedResource],
    report: ParseReport,
) -> nx.DiGraph:
    node_types: dict[str, str] = dict.fromkeys(sources, "source")
    edges: dict[str, list[str]] = {}
    for pool in (datasets, models, exposures):
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
