"""Unit tests for mbt.parsing.loader: YAML loading, error translation, salvage."""

from pathlib import Path

from mbt.contracts import DatasetSpec, SourceGroup
from mbt.parsing.errors import ParseReport
from mbt.parsing.loader import (
    _fields_at,
    _strip_keys,
    _unwrap_model,
    check_top_level_keys,
    load_yaml_mapping,
    validate_resource,
)

VALID_SPLIT = {
    "strategy": "temporal",
    "time_column": "snapshot_date",
    "train": "-30d:-7d",
    "test": "-7d:now",
}


# -- load_yaml_mapping ---------------------------------------------------------


def test_invalid_yaml_is_collected(tmp_path: Path) -> None:
    path = tmp_path / "bad.yml"
    path.write_text("datasets: [unclosed\n  - nope: {")
    report = ParseReport()
    assert load_yaml_mapping(path, "bad.yml", report) is None
    assert "invalid YAML" in report.errors[0].message


def test_unreadable_file_is_collected(tmp_path: Path) -> None:
    directory = tmp_path / "actually_a_dir.yml"
    directory.mkdir()
    report = ParseReport()
    assert load_yaml_mapping(directory, "actually_a_dir.yml", report) is None
    assert "cannot read file" in report.errors[0].message


def test_empty_file_loads_as_empty_mapping(tmp_path: Path) -> None:
    path = tmp_path / "empty.yml"
    path.write_text("")
    report = ParseReport()
    assert load_yaml_mapping(path, "empty.yml", report) == {}
    assert not report.errors


def test_non_mapping_file_is_collected(tmp_path: Path) -> None:
    path = tmp_path / "list.yml"
    path.write_text("- a\n- b\n")
    report = ParseReport()
    assert load_yaml_mapping(path, "list.yml", report) is None
    assert "must be YAML mappings" in report.errors[0].message
    assert "expected one of" in (report.errors[0].hint or "")


# -- check_top_level_keys --------------------------------------------------------


def test_version_one_is_accepted() -> None:
    report = ParseReport()
    check_top_level_keys({"version": 1, "models": []}, "f.yml", report)
    assert not report.errors


def test_unsupported_version_is_rejected() -> None:
    report = ParseReport()
    check_top_level_keys({"version": 2}, "f.yml", report)
    assert "unsupported spec file version: 2" in report.errors[0].message
    assert "version: 1" in (report.errors[0].hint or "")


def test_unknown_top_level_key_gets_did_you_mean() -> None:
    report = ParseReport()
    check_top_level_keys({"modles": []}, "f.yml", report)
    assert "unknown top-level key 'modles'" in report.errors[0].message
    assert report.errors[0].hint == "did you mean 'models'?"


def test_unknown_top_level_key_without_suggestion() -> None:
    report = ParseReport()
    check_top_level_keys({"zzzqqqxyz": []}, "f.yml", report)
    assert "unknown top-level key" in report.errors[0].message
    assert report.errors[0].hint is None


def test_resource_key_must_hold_a_list() -> None:
    report = ParseReport()
    check_top_level_keys({"models": {}}, "f.yml", report)
    assert "'models' must be a list of resource mappings" in report.errors[0].message


# -- validate_resource -----------------------------------------------------------


def _validate(model_cls, raw, report):
    return validate_resource(
        model_cls,
        raw,
        rel="f.yml",
        resource_name=str(raw.get("name", "?")),
        base_pointer="/x/0",
        report=report,
    )


def test_missing_required_field_message() -> None:
    report = ParseReport()
    raw = {"name": "ds", "source": "source('a', 'b')", "split": VALID_SPLIT}
    assert _validate(DatasetSpec, raw, report) is None
    assert "required field 'label' is missing" in report.errors[0].message


def test_nested_unknown_field_gets_did_you_mean_and_salvages() -> None:
    report = ParseReport()
    raw = {
        "name": "ds",
        "source": "source('a', 'b')",
        "label": {"column": "churned"},
        "split": {**VALID_SPLIT, "strategi": "temporal"},
    }
    del raw["split"]["strategy"]
    spec = _validate(DatasetSpec, raw, report)
    assert spec is not None  # salvaged: unknown field stripped, defaults apply
    assert "unknown field 'strategi'" in report.errors[0].message
    assert report.errors[0].hint == "did you mean 'strategy'?"
    assert report.errors[0].field_path == "/x/0/split/strategi"


def test_unknown_field_in_list_item_gets_did_you_mean() -> None:
    report = ParseReport()
    raw = {"name": "grp", "tables": [{"name": "t_one", "pth": "x/*.parquet"}]}
    assert _validate(SourceGroup, raw, report) is None  # salvage fails: no path
    assert "unknown field 'pth'" in report.errors[0].message
    assert report.errors[0].hint == "did you mean 'path'?"


def test_unknown_field_without_suggestion_lists_valid_fields() -> None:
    report = ParseReport()
    raw = {
        "name": "ds",
        "source": "source('a', 'b')",
        "label": {"column": "churned"},
        "split": VALID_SPLIT,
        "zzzqqqxyz": 1,
    }
    spec = _validate(DatasetSpec, raw, report)
    assert spec is not None
    assert "unknown field 'zzzqqqxyz'" in report.errors[0].message
    assert (report.errors[0].hint or "").startswith("valid fields: ")


def test_salvage_failure_returns_none() -> None:
    """Stripping the unknown field exposes a model-validator error: no spec."""
    report = ParseReport()
    raw = {  # neither 'source' nor 'inputs': XOR validator fails after salvage
        "name": "ds",
        "label": {"column": "churned"},
        "split": VALID_SPLIT,
        "bogus_field": 1,
    }
    assert _validate(DatasetSpec, raw, report) is None
    assert len(report.errors) == 1
    assert "unknown field 'bogus_field'" in report.errors[0].message


# -- _fields_at / _unwrap_model / _strip_keys --------------------------------------


def test_fields_at_walks_nested_models() -> None:
    assert "strategy" in _fields_at(DatasetSpec, ("split",))
    assert "path" in _fields_at(SourceGroup, ("tables", 0))
    # a loc part that is not a field: the current model's fields are returned
    assert _fields_at(DatasetSpec, ("nonexistent_part", "x")) == list(DatasetSpec.model_fields)
    # a loc pointing through a non-model annotation has no field candidates
    assert _fields_at(DatasetSpec, ("filters", "x")) == []


def test_fields_at_defends_against_non_models() -> None:
    assert _fields_at(str, ("x",)) == []  # type: ignore[arg-type]
    assert _fields_at(str, ()) == []  # type: ignore[arg-type]


def test_unwrap_model_digs_through_annotations() -> None:
    from mbt.contracts import DatasetInputs

    assert _unwrap_model(DatasetInputs) is DatasetInputs
    assert _unwrap_model(DatasetInputs | None) is DatasetInputs
    assert _unwrap_model(list[DatasetInputs]) is DatasetInputs
    assert _unwrap_model(str) is None
    assert _unwrap_model(list[str]) is None


def test_strip_keys_removes_nested_and_tolerates_missing_paths() -> None:
    raw = {"split": {"strategi": 1, "train": "-7d:now"}, "name": "ds"}
    stripped = _strip_keys(raw, [("split", "strategi")])
    assert stripped == {"split": {"train": "-7d:now"}, "name": "ds"}
    assert raw["split"]["strategi"] == 1  # original untouched

    # unresolvable locations are skipped, not fatal
    assert _strip_keys({"a": 1}, [("b", "c", "d")]) == {"a": 1}
    assert _strip_keys({"a": [1]}, [("a", 5, "x")]) == {"a": [1]}
