"""Compile pipeline tests: determinism, hashing mutations, secrets (S2-03/04/08/09)."""

import os
from datetime import timedelta
from pathlib import Path

from conftest import TEST_ANCHOR, write, write_subscriber_data

from mbt.adapters.registry import AdapterRegistry
from mbt.artifacts.manifest import Manifest
from mbt.compile.compiler import CompileOptions, compile_project
from mbt.config.profiles import load_profiles
from mbt.parsing import parse_project

DS = "dataset.demo.churn_training"
MODEL = "model.demo.churn_model"


def compile_demo(
    project_dir: Path,
    registry: AdapterRegistry,
    target: str | None = None,
    anchor=TEST_ANCHOR,
    cli_vars: dict | None = None,
) -> Manifest:
    parsed = parse_project(project_dir, registry=registry, cli_vars=cli_vars)
    profiles = load_profiles(
        "demo",
        project_dir,
        target_override=target,
        cli_vars=cli_vars or {},
        project_vars=parsed.project.vars,
    )
    return compile_project(
        parsed,
        profiles,
        registry=registry,
        options=CompileOptions(anchor=anchor),
        cli_vars=cli_vars,
    )


def _edit(path: Path, old: str, new: str) -> None:
    text = path.read_text()
    assert old in text, f"{old!r} not found in {path}"
    path.write_text(text.replace(old, new))


def test_compile_is_deterministic_at_same_anchor(
    demo_project: Path, fake_registry: AdapterRegistry
) -> None:
    a = compile_demo(demo_project, fake_registry)
    b = compile_demo(demo_project, fake_registry)
    assert a.to_json() == b.to_json()  # byte-identical (FR-COMP-04)


def test_manifest_shape(demo_project: Path, fake_registry: AdapterRegistry) -> None:
    manifest = compile_demo(demo_project, fake_registry)
    node = manifest.nodes[DS]
    assert node.snapshot_id and node.snapshot_id.startswith("sha256:")
    assert node.config["split"]["train"] == "-180d:-28d"  # expressions, not resolutions
    windows = node.resolved["windows"]
    assert windows["test"] == ["2026-06-03T00:00:00Z", "2026-07-01T00:00:00Z"]
    model = manifest.nodes[MODEL]
    assert model.depends_on == [DS]
    assert model.config["evaluation"]["gates"][0]["threshold"] == 0.4  # var resolved
    assert manifest.metadata.anchor == "2026-07-01T00:00:00Z"
    assert manifest.metadata.generated_at == manifest.metadata.anchor
    assert manifest.metadata.env_digest.startswith("sha256:")


def test_anchor_drift_changes_no_hashes(
    demo_project: Path, fake_registry: AdapterRegistry
) -> None:
    a = compile_demo(demo_project, fake_registry)
    b = compile_demo(demo_project, fake_registry, anchor=TEST_ANCHOR + timedelta(days=7))
    for uid in a.nodes:
        assert a.nodes[uid].config_hash == b.nodes[uid].config_hash
        assert a.nodes[uid].input_hash == b.nodes[uid].input_hash
    assert a.nodes[DS].resolved != b.nodes[DS].resolved  # resolutions do move


def test_hyperparameter_edit_flips_model_only(
    demo_project: Path, fake_registry: AdapterRegistry
) -> None:
    before = compile_demo(demo_project, fake_registry)
    _edit(demo_project / "models/churn_model.yml", "max_depth: 4", "max_depth: 5")
    after = compile_demo(demo_project, fake_registry)
    assert before.nodes[DS].input_hash == after.nodes[DS].input_hash
    assert before.nodes[MODEL].config_hash != after.nodes[MODEL].config_hash
    assert before.nodes[MODEL].input_hash != after.nodes[MODEL].input_hash


def test_dataset_filter_edit_flips_dataset_and_downstream_model(
    demo_project: Path, fake_registry: AdapterRegistry
) -> None:
    before = compile_demo(demo_project, fake_registry)
    _edit(
        demo_project / "datasets/churn_training.yml",
        'filters: ["is_active = true"]',
        'filters: ["is_active = true", "tenure_days >= 30"]',
    )
    after = compile_demo(demo_project, fake_registry)
    assert before.nodes[DS].input_hash != after.nodes[DS].input_hash
    assert before.nodes[MODEL].config_hash == after.nodes[MODEL].config_hash  # spec unchanged
    assert before.nodes[MODEL].input_hash != after.nodes[MODEL].input_hash  # transitive


def test_target_switch_flips_nothing(
    demo_project: Path, fake_registry: AdapterRegistry
) -> None:
    dev = compile_demo(demo_project, fake_registry, target="dev")
    prod = compile_demo(demo_project, fake_registry, target="prod")
    for uid in dev.nodes:
        assert dev.nodes[uid].input_hash == prod.nodes[uid].input_hash  # ADR-5


def test_data_change_flips_input_hash_only(
    demo_project: Path, fake_registry: AdapterRegistry
) -> None:
    before = compile_demo(demo_project, fake_registry)
    write_subscriber_data(demo_project, n_rows=500)  # new snapshot
    after = compile_demo(demo_project, fake_registry)
    assert before.nodes[DS].config_hash == after.nodes[DS].config_hash
    assert before.nodes[DS].input_hash != after.nodes[DS].input_hash
    assert before.nodes[MODEL].input_hash != after.nodes[MODEL].input_hash


def test_description_and_owner_do_not_affect_identity(
    demo_project: Path, fake_registry: AdapterRegistry
) -> None:
    before = compile_demo(demo_project, fake_registry)
    _edit(
        demo_project / "models/churn_model.yml",
        "owner: ds@example.com",
        "owner: other-team@example.com",
    )
    after = compile_demo(demo_project, fake_registry)
    assert before.nodes[MODEL].config_hash == after.nodes[MODEL].config_hash


def test_secrets_never_reach_the_manifest(
    demo_project: Path, fake_registry: AdapterRegistry, monkeypatch
) -> None:
    sentinel = "s3kr3t-value-do-not-leak"
    monkeypatch.setenv("MBT_TEST_SECRET", sentinel)
    write(
        demo_project / "profiles.yml",
        """
        demo:
          target: dev
          outputs:
            dev:
              data: {adapter: local, config: {root: .}}
              tracking: {adapter: fake_tracking, config: {uri: "{{ env_var('MBT_TEST_SECRET') }}"}}
              registry: {adapter: fake_registry}
              artifact_store: file://./target/artifacts
              vars: {sample_fraction: 1.0}
        """,
    )
    manifest = compile_demo(demo_project, fake_registry)
    text = manifest.to_json()
    assert sentinel not in text
    assert "{{ env_var('MBT_TEST_SECRET') }}" in text  # stored unrendered
    assert os.environ["MBT_TEST_SECRET"] == sentinel  # env untouched


def test_manifest_roundtrip_and_hash(demo_project: Path, fake_registry: AdapterRegistry) -> None:
    from mbt.artifacts.manifest import read_manifest

    manifest = compile_demo(demo_project, fake_registry)
    target = demo_project / "target" / "manifest.json"
    manifest.write(target)
    loaded = read_manifest(target)
    assert loaded.to_json() == manifest.to_json()
    drifted = compile_demo(demo_project, fake_registry, anchor=TEST_ANCHOR + timedelta(days=3))
    assert manifest.manifest_hash() != drifted.manifest_hash()  # resolutions differ
