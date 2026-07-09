"""Manifest state index and diff serialization (FR-STATE-01/02, ADR-7)."""

from pathlib import Path

from misc_unit_helpers import make_manifest, make_metadata, make_node

from mbt.state.diff import ManifestStateIndex, NodeDiff, diff_manifests, load_state

MODEL = "model.demo.churn"
OTHER = "model.demo.retired"


def _manifests(*, env_current: str = "sha256:env", env_reference: str = "sha256:env"):
    current = make_manifest(
        make_node(MODEL, input_hash="sha256:new", config_hash="sha256:cfg"),
        metadata=make_metadata(env_digest=env_current),
    )
    reference = make_manifest(
        make_node(MODEL, input_hash="sha256:new", config_hash="sha256:cfg"),
        metadata=make_metadata(env_digest=env_reference),
    )
    return current, reference


def test_load_state_reads_a_manifest_from_a_bare_path(tmp_path: Path) -> None:
    manifest, _ = _manifests()
    path = tmp_path / "state" / "manifest.json"
    manifest.write(path)
    loaded = load_state(str(path))
    assert loaded.to_json() == manifest.to_json()


def test_env_changed_property_tracks_the_env_digest() -> None:
    same = ManifestStateIndex(*_manifests())
    assert not same.env_changed
    changed = ManifestStateIndex(*_manifests(env_reference="sha256:other"))
    assert changed.env_changed


def test_node_absent_from_reference_counts_as_modified() -> None:
    current, reference = _manifests()
    del reference.nodes[MODEL]
    index = ManifestStateIndex(current, reference)
    assert index.is_new(MODEL)
    assert index.is_modified(MODEL)
    assert not index.is_modified("model.demo.ghost")


def test_include_env_marks_unchanged_nodes_modified_on_env_drift() -> None:
    current, reference = _manifests(env_reference="sha256:other")
    default_index = ManifestStateIndex(current, reference)
    assert not default_index.is_modified(MODEL)  # ADR-7: env drift alone never modifies
    opted_in = ManifestStateIndex(current, reference, include_env=True)
    assert opted_in.is_modified(MODEL)


def test_node_diff_to_dict() -> None:
    diff = NodeDiff(unique_id=MODEL, change="modified", components=("config", "snapshot"))
    assert diff.to_dict() == {
        "unique_id": MODEL,
        "change": "modified",
        "components": ["config", "snapshot"],
    }


def test_removed_nodes_are_reported_and_serialized() -> None:
    current, reference = _manifests(env_reference="sha256:other")
    reference.nodes[OTHER] = make_node(OTHER, input_hash="sha256:gone")
    diff = diff_manifests(current, reference)
    assert [d.unique_id for d in diff.removed] == [OTHER]
    assert diff.env_changed
    assert not diff.is_empty

    payload = diff.to_dict()
    assert payload["removed"] == [{"unique_id": OTHER, "change": "removed", "components": []}]
    assert payload["added"] == [] and payload["modified"] == []
    assert payload["env"] == {
        "changed": True,
        "current": "sha256:env",
        "reference": "sha256:other",
        "freeze_current": "",
        "freeze_reference": "",
    }
