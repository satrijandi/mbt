"""Unit tests for mbt.config.profiles: search order, Jinja rendering, validation."""

from pathlib import Path

import pytest
from core_helpers import write

from mbt.config.profiles import find_profiles_path, load_profiles
from mbt.exceptions import ConfigError

MINIMAL = """
demo:
  target: dev
  outputs:
    dev:
      data: {adapter: local}
      tracking: {adapter: fake}
      registry: {adapter: fake}
      artifact_store: {store}
"""


def write_profiles(directory: Path, store: str = "file:///tmp/artifacts") -> Path:
    return write(directory / "profiles.yml", MINIMAL.replace("{store}", store))


@pytest.fixture()
def isolated_home(tmp_path: Path, monkeypatch) -> Path:
    """No profiles.yml reachable via env or the real user home."""
    home = tmp_path / "home"
    home.mkdir()
    monkeypatch.setenv("HOME", str(home))
    monkeypatch.delenv("MBT_PROFILES_DIR", raising=False)
    return home


def test_explicit_profiles_dir_wins(tmp_path: Path, isolated_home: Path) -> None:
    explicit = tmp_path / "elsewhere"
    write_profiles(explicit)
    project_dir = tmp_path / "project"
    project_dir.mkdir()
    assert find_profiles_path(project_dir, explicit) == explicit / "profiles.yml"
    loaded = load_profiles("demo", project_dir, profiles_dir=explicit)
    assert loaded.target_name == "dev"
    assert loaded.raw_target["artifact_store"] == "file:///tmp/artifacts"


def test_env_var_dir_is_searched(tmp_path: Path, isolated_home: Path, monkeypatch) -> None:
    env_dir = tmp_path / "env_profiles"
    write_profiles(env_dir)
    project_dir = tmp_path / "project"
    project_dir.mkdir()
    monkeypatch.setenv("MBT_PROFILES_DIR", str(env_dir))
    assert find_profiles_path(project_dir, None) == env_dir / "profiles.yml"


def test_missing_profiles_lists_search_locations(tmp_path: Path, isolated_home: Path) -> None:
    project_dir = tmp_path / "project"
    project_dir.mkdir()
    with pytest.raises(ConfigError, match=r"no profiles\.yml found") as excinfo:
        find_profiles_path(project_dir, None)
    assert str(project_dir / "profiles.yml") in (excinfo.value.hint or "")
    assert ".mbt" in (excinfo.value.hint or "")


# -- Jinja rendering -------------------------------------------------------------


def test_env_var_rendering_and_required_env(tmp_path: Path, monkeypatch) -> None:
    monkeypatch.delenv("MBT_PROF_STORE_ZZZ", raising=False)
    write_profiles(tmp_path, store="\"{{ env_var('MBT_PROF_STORE_ZZZ') }}\"")
    with pytest.raises(ConfigError, match=r"MBT_PROF_STORE_ZZZ.*is not set"):
        load_profiles("demo", tmp_path)

    write_profiles(tmp_path, store="\"{{ env_var('MBT_PROF_STORE_ZZZ', 'file:///d') }}\"")
    loaded = load_profiles("demo", tmp_path)
    assert loaded.target.artifact_store == "file:///d"
    assert loaded.required_env == ["MBT_PROF_STORE_ZZZ"]

    monkeypatch.setenv("MBT_PROF_STORE_ZZZ", "file:///from-env")
    loaded = load_profiles("demo", tmp_path)
    assert loaded.target.artifact_store == "file:///from-env"


def test_var_rendering_scopes(tmp_path: Path) -> None:
    write_profiles(tmp_path, store="\"{{ var('astore') }}\"")
    loaded = load_profiles("demo", tmp_path, cli_vars={"astore": "file:///cli"})
    assert loaded.target.artifact_store == "file:///cli"

    loaded = load_profiles("demo", tmp_path, project_vars={"astore": "file:///project"})
    assert loaded.target.artifact_store == "file:///project"

    with pytest.raises(ConfigError, match=r"var 'astore' referenced in profiles\.yml has no value"):
        load_profiles("demo", tmp_path)

    write_profiles(tmp_path, store="\"{{ var('astore', 'file:///dflt') }}\"")
    assert load_profiles("demo", tmp_path).target.artifact_store == "file:///dflt"


def test_invalid_jinja_in_profiles(tmp_path: Path) -> None:
    write(tmp_path / "profiles.yml", 'demo:\n  target: "{% if %}"\n')
    with pytest.raises(ConfigError, match=r"invalid Jinja in profiles\.yml"):
        load_profiles("demo", tmp_path)


# -- YAML and schema validation -----------------------------------------------------


def test_invalid_yaml_in_profiles(tmp_path: Path) -> None:
    (tmp_path / "profiles.yml").write_text("demo: [unclosed\n  x: {")
    with pytest.raises(ConfigError, match=r"invalid YAML in unrendered profiles\.yml"):
        load_profiles("demo", tmp_path)


def test_non_mapping_profiles(tmp_path: Path) -> None:
    (tmp_path / "profiles.yml").write_text("- a\n- b\n")
    with pytest.raises(ConfigError, match=r"profiles\.yml must be a YAML mapping"):
        load_profiles("demo", tmp_path)


def test_non_utf8_profiles_is_a_config_error(tmp_path: Path) -> None:
    # A non-UTF-8 byte used to escape as UnicodeDecodeError and hit the CLI's
    # "Internal error" catch-all; it must surface as a friendly ConfigError.
    (tmp_path / "profiles.yml").write_bytes(b"demo:\n  target: dev\n\xff\xfe")
    with pytest.raises(ConfigError, match=r"profiles\.yml is not valid UTF-8"):
        load_profiles("demo", tmp_path)


def test_missing_project_entry(tmp_path: Path) -> None:
    write_profiles(tmp_path)
    with pytest.raises(ConfigError, match="no entry for project 'other'") as excinfo:
        load_profiles("other", tmp_path)
    assert "available: demo" in (excinfo.value.hint or "")


def test_invalid_profiles_schema(tmp_path: Path) -> None:
    write(tmp_path / "profiles.yml", "demo:\n  target: dev\n")
    with pytest.raises(ConfigError, match=r"invalid profiles\.yml"):
        load_profiles("demo", tmp_path)


def test_unknown_target(tmp_path: Path) -> None:
    write_profiles(tmp_path)
    with pytest.raises(ConfigError, match="target 'staging' not defined") as excinfo:
        load_profiles("demo", tmp_path, target_override="staging")
    assert "available targets: dev" in (excinfo.value.hint or "")
