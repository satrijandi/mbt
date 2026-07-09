"""Manifest exposure views and reader error paths (TSD §8.5, §19)."""

import json
from pathlib import Path

import pytest
from misc_unit_helpers import make_manifest, make_metadata, make_node

from mbt.artifacts.manifest import MANIFEST_SCHEMA_VERSION, ManifestExposure, read_manifest
from mbt.exceptions import StateError

MODEL = "model.demo.churn"
EXPOSURE = "exposure.demo.dashboard"


def _manifest_with_exposure():
    exposure = ManifestExposure(
        unique_id=EXPOSURE,
        name="dashboard",
        path="exposures.yml",
        config={"type": "dashboard", "tags": ["bi", "weekly"]},
        depends_on=[MODEL],
    )
    return make_manifest(make_node(MODEL), exposures={EXPOSURE: exposure})


def test_graph_includes_exposures_and_their_edges() -> None:
    graph = _manifest_with_exposure().graph()
    assert graph.nodes[EXPOSURE]["resource_type"] == "exposure"
    assert graph.has_edge(MODEL, EXPOSURE)


def test_selectable_nodes_carry_exposure_tags() -> None:
    nodes = _manifest_with_exposure().selectable_nodes()
    exposure = nodes[EXPOSURE]
    assert exposure.resource_type == "exposure"
    assert exposure.name == "dashboard"
    assert exposure.tags == ("bi", "weekly")


def test_selectable_nodes_tolerate_non_list_exposure_tags() -> None:
    manifest = _manifest_with_exposure()
    manifest.exposures[EXPOSURE].config["tags"] = "not-a-list"
    assert manifest.selectable_nodes()[EXPOSURE].tags == ()


def test_read_manifest_missing_file_is_a_state_error(tmp_path: Path) -> None:
    with pytest.raises(StateError, match="manifest not found"):
        read_manifest(tmp_path / "does_not_exist.json")


def test_read_manifest_invalid_json_is_a_state_error() -> None:
    with pytest.raises(StateError, match="invalid JSON in manifest"):
        read_manifest("{ this is not json", source="manifest")


def test_read_manifest_schema_mismatch_is_a_state_error() -> None:
    payload = {"metadata": {"manifest_schema_version": 99}}
    with pytest.raises(StateError, match="manifest_schema_version 99"):
        read_manifest(json.dumps(payload))


def test_read_manifest_validation_failure_is_a_state_error() -> None:
    # right schema version, but metadata is missing its required fields
    payload = {"metadata": {"manifest_schema_version": MANIFEST_SCHEMA_VERSION}}
    with pytest.raises(StateError, match="invalid manifest"):
        read_manifest(json.dumps(payload))


def test_manifest_roundtrips_exposures(tmp_path: Path) -> None:
    manifest = _manifest_with_exposure()
    path = tmp_path / "manifest.json"
    manifest.write(path)
    loaded = read_manifest(path)
    assert loaded.exposures[EXPOSURE].depends_on == [MODEL]
    assert loaded.to_json() == manifest.to_json()
    assert loaded.metadata == make_metadata()
