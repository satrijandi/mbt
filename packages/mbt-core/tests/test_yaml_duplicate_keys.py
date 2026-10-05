"""Every mbt config loader refuses a repeated mapping key (FEEDBACK v6 A-4).

PyYAML keeps the last of two equal keys silently, so a reviewer could approve
``version: "3"`` in ``promotions.yml`` while ``version: "5"`` further down the
same entry shipped. One test per loader, so a loader that stops going through
``mbt.yamlio`` fails here; ``tests/test_yaml_loader_guard.py`` stops a new one
from bypassing it in the first place.
"""

from pathlib import Path

import pytest
import yaml

from mbt.adapters.registry import AdapterRegistry
from mbt.cli.common import parse_vars
from mbt.config.profiles import load_profiles
from mbt.config.project import load_project
from mbt.deps import load_packages
from mbt.exceptions import ConfigError
from mbt.parsing import parse_project
from mbt.promote import load_promotions_file
from mbt.yamlio import DuplicateKeyError, safe_load

DUPLICATE = "found duplicate key"


def _append_after(path: Path, anchor: str, extra: str) -> None:
    text = path.read_text()
    assert anchor in text, anchor
    path.write_text(text.replace(anchor, anchor + extra, 1))


def test_a_model_spec_with_a_repeated_hyperparameter_fails_parse(
    demo_project: Path, fake_registry: AdapterRegistry
) -> None:
    _append_after(demo_project / "models/churn_model.yml", "max_depth: 4\n", "      max_depth: 9\n")
    parsed = parse_project(demo_project, registry=fake_registry, raise_on_error=False)
    (issue,) = parsed.report.errors
    assert DUPLICATE in issue.message and "'max_depth'" in issue.message
    assert issue.file == "models/churn_model.yml"
    assert 'in "models/churn_model.yml", line' in issue.message


def test_the_promotions_file_refuses_a_second_version(tmp_path: Path) -> None:
    path = tmp_path / "promotions.yml"
    path.write_text(
        "promotions:\n"
        "  - model: churn_classifier\n"
        '    version: "3"\n'
        "    to: production\n"
        '    version: "5"\n'
    )
    with pytest.raises(ConfigError, match=DUPLICATE) as info:
        load_promotions_file(path)
    assert "line 5" in info.value.message and "'version'" in info.value.message


def test_mbt_project_yml(demo_project: Path) -> None:
    _append_after(demo_project / "mbt_project.yml", "name: demo\n", "name: other\n")
    with pytest.raises(ConfigError, match=DUPLICATE):
        load_project(demo_project)


def test_profiles_yml(demo_project: Path) -> None:
    _append_after(demo_project / "profiles.yml", "  target: dev\n", "  target: prod\n")
    with pytest.raises(ConfigError, match=DUPLICATE):
        load_profiles("demo", demo_project)


def test_packages_yml(tmp_path: Path) -> None:
    (tmp_path / "packages.yml").write_text(
        "packages:\n  - package: mbt-xgboost\n    version: '~=0.1'\n    version: '~=0.2'\n"
    )
    with pytest.raises(ConfigError, match=DUPLICATE):
        load_packages(tmp_path)


def test_cli_vars() -> None:
    with pytest.raises(ConfigError, match=DUPLICATE):
        parse_vars("{sample_fraction: 0.1, sample_fraction: 1.0}")


def test_merge_keys_keep_their_meaning() -> None:
    """An explicit key overriding a merged one is what a merge is for."""
    text = "base: &b {x: 1, y: 2}\nderived:\n  <<: *b\n  x: 3\n"
    assert safe_load(text)["derived"] == {"x": 3, "y": 2}


def test_the_error_is_a_yaml_error_naming_both_lines() -> None:
    with pytest.raises(yaml.YAMLError) as info:
        safe_load("a: 1\nb: 2\na: 3\n", source="f.yml")
    assert isinstance(info.value, DuplicateKeyError)
    text = str(info.value)
    assert 'in "f.yml", line 1' in text and 'in "f.yml", line 3' in text


def test_equal_keys_in_sibling_mappings_are_not_duplicates() -> None:
    assert safe_load("- {a: 1}\n- {a: 2}\n") == [{"a": 1}, {"a": 2}]


def test_an_unhashable_key_keeps_yamls_own_error() -> None:
    with pytest.raises(yaml.YAMLError, match="unhashable key"):
        safe_load("? [1, 2]\n: x\n")
