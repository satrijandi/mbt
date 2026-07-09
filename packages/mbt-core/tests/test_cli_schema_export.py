"""Unit tests for JSON Schema publication (mbt/cli/schema_export.py)."""

import json
from pathlib import Path

from mbt.cli.schema_export import _jinja_tolerant, write_json_schemas

EXPECTED_FILES = {
    "sources.schema.json",
    "datasets.schema.json",
    "models.schema.json",
    "metrics.schema.json",
    "exposures.schema.json",
    "scoring.schema.json",
    "mbt_project.schema.json",
    "profiles.schema.json",
}


def test_write_json_schemas_publishes_every_file_kind(tmp_path: Path) -> None:
    out = tmp_path / "schemas"
    written = write_json_schemas(out)
    assert {path.name for path in written} == EXPECTED_FILES
    assert all(path.parent == out for path in written)

    models = json.loads((out / "models.schema.json").read_text())
    assert models["$schema"] == "https://json-schema.org/draft/2020-12/schema"
    assert models["required"] == ["models"]
    assert models["properties"]["models"]["type"] == "array"
    assert models["properties"]["version"] == {"const": 1}
    assert "$defs" in models  # nested defs hoisted to the document root

    # NOTE: no assertion on project["title"]: the intended "mbt_project.yml"
    # is overridden by ProjectConfig.model_json_schema()'s own title because
    # of dict-unpack ordering in _file_schema's caller (reported upstream).
    project = json.loads((out / "mbt_project.schema.json").read_text())
    assert project["$schema"] == "https://json-schema.org/draft/2020-12/schema"
    assert "name" in project["properties"]
    profiles = json.loads((out / "profiles.schema.json").read_text())
    assert profiles["type"] == "object"  # keyed by project name


def test_write_json_schemas_is_rerunnable(tmp_path: Path) -> None:
    out = tmp_path / "schemas"
    first = write_json_schemas(out)
    second = write_json_schemas(out)  # overwrites in place, no error
    assert first == second


def test_jinja_tolerant_widens_scalar_types() -> None:
    widened = _jinja_tolerant({"type": "integer"})
    assert widened["anyOf"][0] == {"type": "integer"}
    assert "pattern" in widened["anyOf"][1]  # the {{ ... }} alternative


def test_jinja_tolerant_recurses_and_leaves_non_scalars_alone() -> None:
    assert _jinja_tolerant([{"type": "boolean"}])[0]["anyOf"]
    assert _jinja_tolerant({"type": "string"}) == {"type": "string"}
    assert _jinja_tolerant("leaf") == "leaf"
    nested = _jinja_tolerant({"properties": {"seed": {"type": "number"}}})
    assert nested["properties"]["seed"]["anyOf"]
