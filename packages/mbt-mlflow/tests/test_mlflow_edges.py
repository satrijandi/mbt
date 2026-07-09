"""mbt-mlflow edge paths: artifact upload on log(), legacy-mode search
failures, and the plugin descriptor (complements test_mlflow_adapter)."""

from pathlib import Path

import pytest
from mbt_mlflow.adapter import MlflowRegistry, MlflowTracking

from mbt_adapter_base import CONTRACT_VERSION, ArtifactRef, ManifestNode, Stage


@pytest.fixture()
def uri(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> str:
    # mlflow drops its default artifact root (./mlruns) relative to the cwd;
    # chdir keeps it inside tmp instead of littering the repo root.
    monkeypatch.chdir(tmp_path)
    return f"sqlite:///{tmp_path}/mlflow.db"


def _node() -> ManifestNode:
    return ManifestNode(
        unique_id="model.demo.m",
        resource_type="model",
        name="m",
        path="models/m.yml",
        config={},
    )


def test_log_uploads_local_file_artifacts_and_tags_the_rest(uri: str, tmp_path: Path) -> None:
    tracking = MlflowTracking({"uri": uri})
    tracking.prepare()
    run = tracking.start_run(_node(), {})

    local = tmp_path / "model.ubj"
    local.write_bytes(b"weights")
    on_disk = ArtifactRef(
        uri=f"file://{local}", format="xgboost_ubj", content_hash="sha256:a", size_bytes=7
    )
    gone = ArtifactRef(
        uri=f"file://{tmp_path}/missing.bin", format="ghost", content_hash="sha256:b", size_bytes=1
    )
    remote = ArtifactRef(
        uri="s3://bucket/model.onnx", format="onnx", content_hash="sha256:c", size_bytes=1
    )
    tracking.log(run, artifacts=[on_disk, gone, remote])

    from mlflow.tracking import MlflowClient

    client = MlflowClient(tracking_uri=uri)
    stored = client.get_run(run.run_id)
    assert stored.data.tags["mbt.artifact.xgboost_ubj"] == on_disk.uri
    assert stored.data.tags["mbt.artifact.ghost"] == gone.uri
    assert stored.data.tags["mbt.artifact.onnx"] == remote.uri
    # only the existing file:// artifact was uploaded to the run
    uploaded = [a.path for a in client.list_artifacts(run.run_id)]
    assert uploaded == ["model.ubj"]


def test_legacy_champion_lookup_survives_search_failures(
    uri: str, monkeypatch: pytest.MonkeyPatch
) -> None:
    """use_aliases=false + a registry backend that errors on search: the
    champion lookup degrades to 'no champion' instead of crashing."""
    from mlflow.exceptions import MlflowException

    registry = MlflowRegistry({"uri": uri, "use_aliases": False})

    def boom(*args: object, **kwargs: object) -> None:
        raise MlflowException("registry backend unavailable")

    monkeypatch.setattr(registry.client(), "search_model_versions", boom)
    assert registry.get_champion("m", Stage.STAGING) is None


def test_plugin_descriptor_wires_tracking_and_registry() -> None:
    from mbt_mlflow.plugin import PLUGIN

    assert PLUGIN.name == "mlflow"
    assert PLUGIN.contract_version == CONTRACT_VERSION
    assert PLUGIN.tracking is MlflowTracking
    assert PLUGIN.registry is MlflowRegistry
    assert PLUGIN.fingerprint_packages == ["mlflow"]
