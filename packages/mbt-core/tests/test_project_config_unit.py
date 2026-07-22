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
    write(tmp_path / "mbt_project.yml", 'name: NotSnake\nversion: "1.0"\n')
    with pytest.raises(ConfigError, match=r"invalid mbt_project\.yml"):
        load_project(tmp_path)


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
