"""S3 artifact store round-trip under moto (FR-RUN-08)."""

from pathlib import Path

import pytest
from moto import mock_aws

from mbt.storage import LocalArtifactStore, S3ArtifactStore, artifact_store_for


@pytest.fixture(autouse=True)
def aws_env(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setenv("AWS_DEFAULT_REGION", "us-east-1")
    monkeypatch.setenv("AWS_ACCESS_KEY_ID", "testing")
    monkeypatch.setenv("AWS_SECRET_ACCESS_KEY", "testing")


def test_store_dispatch_by_scheme(tmp_path: Path) -> None:
    assert isinstance(artifact_store_for(f"file://{tmp_path}"), LocalArtifactStore)
    with mock_aws():
        assert isinstance(artifact_store_for("s3://models/mbt"), S3ArtifactStore)


@mock_aws()
def test_s3_put_fetch_round_trip(tmp_path: Path) -> None:
    import boto3

    boto3.client("s3").create_bucket(Bucket="models")
    store = artifact_store_for("s3://models/mbt/prod", run_prefix="runx")
    source = tmp_path / "model.bin"
    source.write_bytes(b"weights")

    assert store.uri == "s3://models/mbt/prod"
    ref = store.put_file(source, "model.bin", format="test_bin")
    assert ref.uri == "s3://models/mbt/prod/runx/model.bin"
    assert ref.content_hash.startswith("sha256:")
    assert ref.size_bytes == 7

    fetched = store.fetch(ref)
    assert fetched.read_bytes() == b"weights"
    assert store.fetch(ref) == fetched  # cached; no second download needed


@mock_aws()
def test_s3_fetch_missing_artifact_is_actionable(tmp_path: Path) -> None:
    import boto3

    from mbt.exceptions import MbtError

    boto3.client("s3").create_bucket(Bucket="models")
    store = artifact_store_for("s3://models/mbt", run_prefix="runx")
    source = tmp_path / "model.bin"
    source.write_bytes(b"weights")
    ref = store.put_file(source, "model.bin", format="test_bin")
    boto3.client("s3").delete_object(Bucket="models", Key="mbt/runx/model.bin")
    with pytest.raises(MbtError, match="artifact not found"):
        store.fetch(ref)


@mock_aws()
def test_s3_fetch_rejects_foreign_uri() -> None:
    from mbt.contracts import ArtifactRef
    from mbt.exceptions import MbtError

    store = artifact_store_for("s3://models/mbt")
    foreign = ArtifactRef(
        uri="file:///tmp/model.bin", format="bin", content_hash="sha256:0", size_bytes=0
    )
    with pytest.raises(MbtError, match="only resolves s3"):
        store.fetch(foreign)


@mock_aws()
def test_read_uri_text_over_s3_round_trip_and_missing() -> None:
    import boto3

    from mbt.exceptions import StateError
    from mbt.storage import read_uri_text

    boto3.client("s3").create_bucket(Bucket="state")
    boto3.client("s3").put_object(Bucket="state", Key="prod/manifest.json", Body=b"{}")
    assert read_uri_text("s3://state/prod/manifest.json") == "{}"

    # A missing key is a hard StateError, mirroring the file:// path.
    with pytest.raises(StateError, match="cannot read"):
        read_uri_text("s3://state/prod/missing.json")
