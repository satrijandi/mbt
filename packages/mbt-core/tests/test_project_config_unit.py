"""Unit tests for mbt.config.project: loading and version requirements."""

from pathlib import Path

import pytest
from core_helpers import write

import mbt
from mbt.config.project import load_project
from mbt.exceptions import ConfigError


def test_missing_project_file(tmp_path: Path) -> None:
    with pytest.raises(ConfigError, match=r"no mbt_project\.yml found") as excinfo:
        load_project(tmp_path)
    assert "mbt init" in (excinfo.value.hint or "")


def test_invalid_yaml(tmp_path: Path) -> None:
    (tmp_path / "mbt_project.yml").write_text("name: [unclosed\n  x: {")
    with pytest.raises(ConfigError, match=r"invalid YAML in mbt_project\.yml"):
        load_project(tmp_path)


def test_non_mapping_project_file(tmp_path: Path) -> None:
    (tmp_path / "mbt_project.yml").write_text("- a\n- b\n")
    with pytest.raises(ConfigError, match=r"mbt_project\.yml must be a YAML mapping"):
        load_project(tmp_path)


def test_non_utf8_project_file_is_a_config_error(tmp_path: Path) -> None:
    # A non-UTF-8 byte used to escape as UnicodeDecodeError and hit the CLI's
    # "Internal error" catch-all; it must surface as a friendly ConfigError.
    (tmp_path / "mbt_project.yml").write_bytes(b"name: demo\n\xff\xfe broken")
    with pytest.raises(ConfigError, match=r"mbt_project\.yml is not valid UTF-8"):
        load_project(tmp_path)


def test_schema_violation(tmp_path: Path) -> None:
    write(tmp_path / "mbt_project.yml", 'name: not-snake\nversion: "1.0"\n')
    with pytest.raises(ConfigError, match=r"invalid mbt_project\.yml"):
        load_project(tmp_path)


def test_an_uppercase_project_name_is_accepted(tmp_path: Path) -> None:
    """The project name is half the tracking experiment name (ADR-28), which is
    an org-facing label like LOAN_APPLY_PROPENSITY__V1_0_0. Forcing it lowercase
    forces a casing the org does not use, and the name is never a path or a
    warehouse identifier - only a unique_id segment and a profiles.yml key."""
    write(
        tmp_path / "mbt_project.yml",
        'name: LOAN_APPLY_PROPENSITY\nversion: "1.0"\n',
    )
    assert load_project(tmp_path).name == "LOAN_APPLY_PROPENSITY"


def test_every_lowercase_project_name_that_used_to_validate_still_does(tmp_path: Path) -> None:
    """The widening to allow uppercase must be additive, never a swap."""
    for name in ("demo", "churn_lake", "a", "a1_b2"):
        write(tmp_path / "mbt_project.yml", f'name: {name}\nversion: "1.0"\n')
        assert load_project(tmp_path).name == name


def test_a_project_name_starting_with_a_digit_is_still_rejected(tmp_path: Path) -> None:
    write(tmp_path / "mbt_project.yml", 'name: 1churn\nversion: "1.0"\n')
    with pytest.raises(ConfigError, match=r"invalid mbt_project\.yml"):
        load_project(tmp_path)


def test_resource_names_stay_lowercase_when_the_project_name_does_not() -> None:
    """The two patterns are deliberately different, and this is the test that
    says so. Widening the project name to admit uppercase is safe because the
    name is only a unique_id SEGMENT; widening resource names too would change
    what a selector matches, so NAME_PATTERN must not follow it."""
    import re

    from mbt.config.project import PROJECT_NAME_PATTERN
    from mbt_adapter_base.specs import NAME_PATTERN

    assert re.match(PROJECT_NAME_PATTERN, "LOAN_APPLY_PROPENSITY")
    assert not re.match(NAME_PATTERN, "LoanApplyXgb")
    # ... and everything legal for a resource is still legal for a project.
    assert re.match(PROJECT_NAME_PATTERN, "loan_apply_xgb")


def test_require_mbt_version_satisfied(tmp_path: Path) -> None:
    write(
        tmp_path / "mbt_project.yml",
        'name: demo\nversion: "1.0"\nrequire_mbt_version: ">=0.0.1"\n',
    )
    assert load_project(tmp_path).require_mbt_version == ">=0.0.1"


def test_require_mbt_version_invalid_specifier(tmp_path: Path) -> None:
    write(
        tmp_path / "mbt_project.yml",
        'name: demo\nversion: "1.0"\nrequire_mbt_version: "not a specifier"\n',
    )
    with pytest.raises(ConfigError, match="invalid require_mbt_version specifier") as excinfo:
        load_project(tmp_path)
    assert "PEP 440" in (excinfo.value.hint or "")


def test_require_mbt_version_unsatisfied(tmp_path: Path) -> None:
    write(
        tmp_path / "mbt_project.yml",
        'name: demo\nversion: "1.0"\nrequire_mbt_version: ">=99.0"\n',
    )
    with pytest.raises(ConfigError, match=r"requires mbt >=99\.0") as excinfo:
        load_project(tmp_path)
    assert mbt.__version__ in excinfo.value.message
