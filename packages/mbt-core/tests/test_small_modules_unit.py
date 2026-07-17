"""Small mbt-core modules: deps, docsgen edges, errors, git info, ids, runtime,
secrets, utils."""

import subprocess
from pathlib import Path
from types import SimpleNamespace

import pytest
from core_helpers import write
from misc_unit_helpers import RecordingSink, make_manifest

from mbt.adapters.registry import AdapterRegistry
from mbt.artifacts.run_results import NodeResult
from mbt.contracts import AdapterRef
from mbt.deps import PackagePin, install_packages, load_packages
from mbt.docsgen.generator import _metric_table, generate_docs
from mbt.exceptions import AdapterError, ConfigError
from mbt.gitinfo import collect_git_info
from mbt.ids import name_of, resource_type_of, source_unique_id, unique_id
from mbt.runtime import compute_adapter, resolve_artifact_store_uri
from mbt.secrets import taint
from mbt.utils import canonical_json, did_you_mean, levenshtein, slugify

# -- deps ----------------------------------------------------------------------


def test_load_packages_requires_the_file(tmp_path: Path) -> None:
    with pytest.raises(ConfigError, match=r"no packages\.yml"):
        load_packages(tmp_path)


def test_load_packages_parses_pins(tmp_path: Path) -> None:
    write(
        tmp_path / "packages.yml",
        """
        packages:
          - package: mbt-xgboost
            version: "~=0.1"
          - package: mbt-mlflow
        """,
    )
    pins = load_packages(tmp_path)
    assert pins == [
        PackagePin(package="mbt-xgboost", version="~=0.1"),
        PackagePin(package="mbt-mlflow", version=None),
    ]


def test_load_packages_rejects_invalid_yaml_and_schema(tmp_path: Path) -> None:
    (tmp_path / "packages.yml").write_text("packages: [::not yaml")
    with pytest.raises(ConfigError, match=r"invalid packages\.yml"):
        load_packages(tmp_path)
    (tmp_path / "packages.yml").write_text("packages:\n  - pin: wrong-key\n")
    with pytest.raises(ConfigError, match=r"invalid packages\.yml"):
        load_packages(tmp_path)


def test_failed_pip_install_raises_with_the_last_error_line(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    bus = RecordingSink()
    monkeypatch.setattr("mbt.deps.get_bus", lambda: bus)
    monkeypatch.setattr(
        "mbt.deps.subprocess.run",
        lambda command, **kwargs: SimpleNamespace(
            returncode=1, stdout="", stderr="collecting...\nERROR: no matching distribution"
        ),
    )
    with pytest.raises(ConfigError, match=r"pip install failed \(exit 1\)") as excinfo:
        install_packages([PackagePin(package="mbt-ghost")])
    assert excinfo.value.hint == "ERROR: no matching distribution"


def test_failed_pip_install_without_output_hints_at_manual_run(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    bus = RecordingSink()
    monkeypatch.setattr("mbt.deps.get_bus", lambda: bus)
    monkeypatch.setattr(
        "mbt.deps.subprocess.run",
        lambda command, **kwargs: SimpleNamespace(returncode=2, stdout="", stderr=""),
    )
    with pytest.raises(ConfigError, match=r"pip install failed \(exit 2\)") as excinfo:
        install_packages([PackagePin(package="mbt-ghost")])
    assert excinfo.value.hint == "run pip manually to see the full error"


# -- docsgen ---------------------------------------------------------------------


def test_empty_manifest_renders_a_no_nodes_lineage(tmp_path: Path) -> None:
    index = generate_docs(make_manifest(), None, tmp_path / "docs")
    assert "no nodes" in index.read_text()


def test_metric_table_renders_slices() -> None:
    result = NodeResult(
        unique_id="model.demo.churn",
        status="success",
        metrics={"pr_auc": 0.5},
        slices={"plan=basic": {"pr_auc": 0.4125}},
    )
    table = _metric_table(result)
    assert "<h2>Slices</h2>" in table
    assert "plan=basic" in table
    assert "0.4125" in table


# -- exceptions ------------------------------------------------------------------


def test_adapter_error_wrap_carries_adapter_and_resource_context() -> None:
    error = AdapterError.wrap(
        ValueError("bad shape"), adapter="xgboost", resource="model.demo.churn"
    )
    assert error.message == "adapter 'xgboost' failed: bad shape"
    assert error.resource == "model.demo.churn"
    assert "hint: run with --log-format json" in str(error)


# -- gitinfo ---------------------------------------------------------------------


def test_git_info_is_null_when_git_cannot_run(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    def explode(*args: object, **kwargs: object) -> object:
        raise OSError("git binary missing")

    monkeypatch.setattr("mbt.gitinfo.subprocess.run", explode)
    assert collect_git_info(tmp_path) == {"commit": None, "branch": None, "dirty": False}


def test_git_info_is_null_on_timeout(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    def timeout(*args: object, **kwargs: object) -> object:
        raise subprocess.TimeoutExpired(cmd="git", timeout=10)

    monkeypatch.setattr("mbt.gitinfo.subprocess.run", timeout)
    assert collect_git_info(tmp_path) == {"commit": None, "branch": None, "dirty": False}


# -- ids -------------------------------------------------------------------------


def test_unique_id_construction_and_parsing() -> None:
    assert unique_id("model", "demo", "churn") == "model.demo.churn"
    assert source_unique_id("demo", "lakehouse", "rows") == "source.demo.lakehouse.rows"
    assert resource_type_of("model.demo.churn") == "model"
    assert name_of("source.demo.lakehouse.rows") == "rows"


def test_unique_id_rejects_unknown_resource_types() -> None:
    with pytest.raises(ValueError, match="unknown resource type: 'widget'"):
        unique_id("widget", "demo", "churn")


# -- runtime ---------------------------------------------------------------------


def test_compute_adapter_is_wired_from_the_target(fake_registry: AdapterRegistry) -> None:
    profiles = SimpleNamespace(
        target=SimpleNamespace(compute=AdapterRef(adapter="fake", config={"threads": 2}))
    )
    adapter = compute_adapter(profiles, fake_registry)
    assert adapter.name == "fake"
    assert adapter.config == {"threads": 2}


def test_relative_file_store_uris_resolve_against_the_project(tmp_path: Path) -> None:
    resolved = resolve_artifact_store_uri("file://./target/artifacts", tmp_path)
    assert resolved == f"file://{(tmp_path / 'target/artifacts').resolve()}"
    absolute = f"file://{tmp_path}/store"
    assert resolve_artifact_store_uri(absolute, tmp_path) == absolute
    assert resolve_artifact_store_uri("s3://bucket/prefix", tmp_path) == "s3://bucket/prefix"


# -- secrets ---------------------------------------------------------------------


def test_secret_repr_never_shows_the_value() -> None:
    secret = taint("hunter2")
    assert repr(secret) == "'***'"
    assert secret == "hunter2"  # the value itself is intact for adapters


# -- utils -----------------------------------------------------------------------


def test_levenshtein_shortcuts() -> None:
    assert levenshtein("same", "same") == 0
    assert levenshtein("", "abc") == 3
    assert levenshtein("abc", "") == 3
    assert levenshtein("kitten", "sitting") == 3


def test_did_you_mean_respects_the_distance_budget() -> None:
    assert did_you_mean("pr_ac", ["pr_auc", "roc_auc"]) == "pr_auc"
    assert did_you_mean("zzzzzz", ["pr_auc", "roc_auc"]) is None


def test_canonical_json_rejects_non_serializable_values() -> None:
    assert canonical_json({"b": 1, "a": [1.5, "x"]}) == '{"a":[1.5,"x"],"b":1}'
    with pytest.raises(TypeError, match="not canonically serializable: object"):
        canonical_json({"bad": object()})


def test_slugify_normalizes_to_snake_case() -> None:
    assert slugify("Hello, World!") == "hello_world"
    assert slugify("  churn-model v2  ") == "churn_model_v2"


def test_cause_message_uses_only_the_inner_message_for_mbt_errors() -> None:
    from mbt.exceptions import ConfigError, cause_message

    inner = ConfigError("bad thing", resource="x", hint="do y")
    assert "hint:" in str(inner)  # full str is multi-line: would inject a 2nd hint
    assert cause_message(inner) == "bad thing"
    assert cause_message(ValueError("plain")) == "plain"
