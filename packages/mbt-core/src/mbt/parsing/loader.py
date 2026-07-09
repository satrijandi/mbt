"""YAML loading and Pydantic error translation (TSD §7, FR-PARSE-02/04)."""

from pathlib import Path
from typing import Any, TypeVar

import yaml
from pydantic import BaseModel, ValidationError

from mbt.parsing.errors import ParseReport
from mbt.utils import did_you_mean

#: Top-level keys a resource YAML file may carry, mapped to resource type.
TOP_LEVEL_KEYS = {
    "sources": "source",
    "datasets": "dataset",
    "models": "model",
    "metrics": "metric",
    "exposures": "exposure",
    "scoring": "scoring",
}

M = TypeVar("M", bound=BaseModel)


def load_yaml_mapping(path: Path, rel: str, report: ParseReport) -> dict[str, Any] | None:
    """Load one YAML file as a mapping, collecting errors instead of raising."""
    try:
        raw = yaml.safe_load(path.read_text())
    except yaml.YAMLError as exc:
        report.error(f"invalid YAML: {exc}", file=rel)
        return None
    except OSError as exc:
        report.error(f"cannot read file: {exc}", file=rel)
        return None
    if raw is None:
        return {}
    if not isinstance(raw, dict):
        report.error(
            "resource files must be YAML mappings with a top-level resource key",
            file=rel,
            hint=f"expected one of: {', '.join(sorted(TOP_LEVEL_KEYS))}",
        )
        return None
    return raw


def check_top_level_keys(raw: dict[str, Any], rel: str, report: ParseReport) -> None:
    """Reject unknown top-level keys with did-you-mean (FR-PARSE-04)."""
    for key, value in raw.items():
        if key == "version":
            if value != 1:
                report.error(
                    f"unsupported spec file version: {value!r}",
                    file=rel,
                    field_path="/version",
                    hint="this mbt release supports 'version: 1'",
                )
            continue
        if key not in TOP_LEVEL_KEYS:
            suggestion = did_you_mean(key, [*TOP_LEVEL_KEYS, "version"])
            report.error(
                f"unknown top-level key {key!r}",
                file=rel,
                field_path=f"/{key}",
                hint=f"did you mean {suggestion!r}?" if suggestion else None,
            )
            continue
        if not isinstance(value, list):
            report.error(
                f"'{key}' must be a list of resource mappings",
                file=rel,
                field_path=f"/{key}",
            )


def _loc_to_pointer(loc: tuple[Any, ...]) -> str:
    return "/" + "/".join(str(part) for part in loc) if loc else ""


def _fields_at(model_cls: type[BaseModel], loc: tuple[Any, ...]) -> list[str]:
    """Field names of the (possibly nested) model a pydantic loc points into."""
    current: Any = model_cls
    for part in loc:
        if not (isinstance(current, type) and issubclass(current, BaseModel)):
            return []
        if isinstance(part, int):
            continue
        fields = current.model_fields
        if part not in fields:
            return list(fields)
        annotation = fields[part].annotation
        current = _unwrap_model(annotation)
        if current is None:
            return []
    if isinstance(current, type) and issubclass(current, BaseModel):
        return list(current.model_fields)
    return []


def _unwrap_model(annotation: Any) -> type[BaseModel] | None:
    """Dig a BaseModel class out of Optional/list/dict annotations."""
    import typing

    if isinstance(annotation, type) and issubclass(annotation, BaseModel):
        return annotation
    for arg in typing.get_args(annotation):
        found = _unwrap_model(arg)
        if found is not None:
            return found
    return None


def validate_resource(
    model_cls: type[M],
    raw: dict[str, Any],
    *,
    rel: str,
    resource_name: str | None,
    base_pointer: str,
    report: ParseReport,
) -> M | None:
    """Validate one resource mapping, translating errors into ParseIssues.

    When the *only* problems are unknown fields, the resource is salvaged
    (unknown fields stripped) so cross-resource checks can still surface
    further errors in the same pass (FR-PARSE-02); the parse still fails.
    """
    try:
        return model_cls.model_validate(raw)
    except ValidationError as exc:
        extra_locs: list[tuple[Any, ...]] = []
        only_extra = True
        for error in exc.errors():
            loc = tuple(error["loc"])
            pointer = base_pointer + _loc_to_pointer(loc)
            message = error["msg"]
            hint = None
            if error["type"] == "extra_forbidden":
                extra_locs.append(loc)
                unknown = str(loc[-1])
                candidates = _fields_at(model_cls, loc[:-1])
                suggestion = did_you_mean(unknown, candidates)
                message = f"unknown field {unknown!r}"
                if suggestion:
                    hint = f"did you mean {suggestion!r}?"
                else:
                    hint = f"valid fields: {', '.join(sorted(candidates))}"
            else:
                only_extra = False
                if error["type"] == "missing":
                    message = f"required field {str(loc[-1])!r} is missing"
            report.error(
                message,
                file=rel,
                resource=resource_name,
                field_path=pointer,
                hint=hint,
            )
        if only_extra and extra_locs:
            salvaged = _strip_keys(raw, extra_locs)
            try:
                return model_cls.model_validate(salvaged)
            except ValidationError:
                return None
        return None


def _strip_keys(raw: dict[str, Any], locs: list[tuple[Any, ...]]) -> dict[str, Any]:
    """Remove the keys at the given pydantic locations from a deep copy."""
    import copy

    stripped = copy.deepcopy(raw)
    for loc in locs:
        container: Any = stripped
        for part in loc[:-1]:
            try:
                container = container[part]
            except (KeyError, IndexError, TypeError):
                container = None
                break
        if isinstance(container, dict):
            container.pop(loc[-1], None)
    return stripped
