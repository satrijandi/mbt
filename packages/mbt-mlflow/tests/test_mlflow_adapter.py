"""mbt-mlflow against a local sqlite backend (S3-04)."""

import subprocess
import sys
from pathlib import Path

import pytest
from mbt_adapter_base import ArtifactRef, ManifestNode, Stage

from mbt_mlflow.adapter import MlflowRegistry, MlflowTracking


@pytest.fixture()
def uri(tmp_path: Path) -> str:
    return f"sqlite:///{tmp_path}/mlflow.db"


def _node() -> ManifestNode:
    return ManifestNode(
        unique_id="model.demo.m",
        resource_type="model",
        name="m",
        path="models/m.yml",
        config={},
    )


def _artifact(tmp_path: Path) -> ArtifactRef:
    path = tmp_path / "model.bin"
    path.write_bytes(b"weights")
    return ArtifactRef(
        uri=f"file://{path}", format="test_bin", content_hash="sha256:abc", size_bytes=7
    )


def test_tracking_round_trip_with_identity_tags(uri: str) -> None:
    tracking = MlflowTracking({"uri": uri})
    tracking.prepare()
    run = tracking.start_run(_node(), {"mbt.config_hash": "sha256:x", "mbt.run_id": "r1"})
    tracking.log(run, params={"max_depth": "4"}, metrics={"pr_auc": 0.41})
    tracking.log(tracking.resume(run.run_id), tags={"mbt.gates_passed": "true"})
    tracking.end_run(run, "FINISHED")

    from mlflow.tracking import MlflowClient

    stored = MlflowClient(tracking_uri=uri).get_run(run.run_id)
    assert stored.data.tags["mbt.config_hash"] == "sha256:x"
    assert stored.data.tags["mbt.gates_passed"] == "true"
    assert stored.data.params["max_depth"] == "4"
    assert stored.data.metrics["pr_auc"] == 0.41
    assert stored.info.status == "FINISHED"


def test_nested_tuning_trials(uri: str) -> None:
    tracking = MlflowTracking({"uri": uri})
    tracking.prepare()
    run = tracking.start_run(_node(), {})
    tracking.log_trial(run, 0, {"max_depth": 3}, 0.39)
    tracking.log_trial(run, 1, {"max_depth": 5}, 0.42)

    from mlflow.tracking import MlflowClient

    client = MlflowClient(tracking_uri=uri)
    experiment = client.get_experiment_by_name("mbt")
    children = client.search_runs(
        [experiment.experiment_id],
        filter_string=f"tags.\"mlflow.parentRunId\" = '{run.run_id}'",
    )
    assert len(children) == 2
    assert {r.data.metrics["objective"] for r in children} == {0.39, 0.42}


def test_registry_stages_and_champion(uri: str, tmp_path: Path) -> None:
    registry = MlflowRegistry({"uri": uri})
    artifact = _artifact(tmp_path)
    metadata = {"mbt.gates_passed": "true", "mbt.tracking_run_id": ""}
    v1 = registry.register(artifact, "m", metadata)
    assert v1.version == "1"

    assert registry.get_champion("m", Stage.STAGING) is None
    registry.transition(v1, Stage.STAGING)
    champion = registry.get_champion("m", Stage.STAGING)
    assert champion is not None and champion.version == "1"
    assert champion.artifact is not None
    assert champion.artifact.uri == artifact.uri  # reconstructed from tags
    assert champion.tags["mbt.gates_passed"] == "true"

    registry.transition(v1, Stage.PRODUCTION)
    assert registry.get_champion("m", Stage.PRODUCTION).version == "1"
    assert registry.get_champion("m", Stage.STAGING) is None
    assert registry.get_version("m", "1") is not None
    assert registry.get_version("m", "99") is None


def test_registry_aliases_mode(uri: str, tmp_path: Path) -> None:
    registry = MlflowRegistry({"uri": uri, "use_aliases": True})
    v1 = registry.register(_artifact(tmp_path), "m", {})
    registry.transition(v1, Stage.PRODUCTION)
    champion = registry.get_champion("m", Stage.PRODUCTION)
    assert champion is not None and champion.version == "1"


def test_plugin_import_hygiene() -> None:
    probe = (
        "import sys\n"
        "import mbt_mlflow.plugin\n"
        "assert 'mlflow' not in sys.modules, 'mlflow imported at plugin load (ADR-14)'\n"
    )
    subprocess.run([sys.executable, "-c", probe], check=True)  # noqa: S603
