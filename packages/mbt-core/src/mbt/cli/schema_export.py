"""JSON Schema publication for editor autocomplete (FR-PARSE-03, S1-08)."""

import json
from pathlib import Path
from typing import Any

from mbt.config.profiles import ProfilesConfig
from mbt.config.project import ProjectConfig
from mbt.contracts import DatasetSpec, ExposureSpec, MetricSpec, ModelSpec, SourceGroup

#: Matches "{{ ... }}" / "{% ... %}" values users write before rendering.
_JINJA_STRING = {"type": "string", "pattern": "\\{\\{.*\\}\\}|\\{%.*%\\}"}
_SCALAR_TYPES = ("number", "integer", "boolean")


def _jinja_tolerant(schema: Any) -> Any:
    """Editors validate *unrendered* YAML: any scalar may hold a Jinja
    expression string, so widen scalar types with a Jinja alternative."""
    if isinstance(schema, dict):
        out = {k: _jinja_tolerant(v) for k, v in schema.items()}
        if out.get("type") in _SCALAR_TYPES:
            return {"anyOf": [out, _JINJA_STRING]}
        return out
    if isinstance(schema, list):
        return [_jinja_tolerant(v) for v in schema]
    return schema


def _file_schema(key: str, item_schema: dict[str, Any], title: str) -> dict[str, Any]:
    """Wrap a resource schema into the whole-file shape editors validate.

    Pydantic emits ``$ref: '#/$defs/...'`` pointers relative to the document
    root, so nested ``$defs`` must be hoisted when the schema is embedded.
    """
    item_schema = dict(_jinja_tolerant(item_schema))
    defs = item_schema.pop("$defs", {})
    schema: dict[str, Any] = {
        "$schema": "https://json-schema.org/draft/2020-12/schema",
        "title": title,
        "type": "object",
        "properties": {
            "version": {"const": 1},
            key: {"type": "array", "items": item_schema},
        },
        "required": [key],
        "additionalProperties": False,
    }
    if defs:
        schema["$defs"] = defs
    return schema


def write_json_schemas(output_dir: Path) -> list[Path]:
    """Write one JSON Schema per file kind; returns the written paths."""
    output_dir.mkdir(parents=True, exist_ok=True)
    schemas: dict[str, dict[str, Any]] = {
        "sources.schema.json": _file_schema(
            "sources", SourceGroup.model_json_schema(), "mbt sources file"
        ),
        "datasets.schema.json": _file_schema(
            "datasets", DatasetSpec.model_json_schema(), "mbt datasets file"
        ),
        "models.schema.json": _file_schema(
            "models", ModelSpec.model_json_schema(), "mbt models file"
        ),
        "metrics.schema.json": _file_schema(
            "metrics", MetricSpec.model_json_schema(), "mbt metrics file"
        ),
        "exposures.schema.json": _file_schema(
            "exposures", ExposureSpec.model_json_schema(), "mbt exposures file"
        ),
        "mbt_project.schema.json": {
            "$schema": "https://json-schema.org/draft/2020-12/schema",
            "title": "mbt_project.yml",
            **ProjectConfig.model_json_schema(),
        },
        "profiles.schema.json": {
            "$schema": "https://json-schema.org/draft/2020-12/schema",
            "title": "profiles.yml",
            "type": "object",
            "additionalProperties": ProfilesConfig.model_json_schema(),
        },
    }
    written: list[Path] = []
    for name, schema in schemas.items():
        path = output_dir / name
        path.write_text(json.dumps(schema, indent=2, sort_keys=True) + "\n")
        written.append(path)
    return written
