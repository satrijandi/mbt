"""Local artifact store + URI text reader: pure paths, no network (FR-STATE-01).

Companion to ``test_storage_s3.py`` (moto round-trip): this pins the file://
store contract and the ``read_uri_text`` safety guarantee that an unreadable
``--state`` reference is a hard error, never a silent full retrain.
"""

import hashlib
from pathlib import Path

import pytest

from mbt.contracts import ArtifactRef
from mbt.exceptions import MbtError, StateError
from mbt.storage import (
    LocalArtifactStore,
    S3ArtifactStore,
    artifact_store_for,
    read_uri_text,
)


def _ref(uri: str) -> ArtifactRef:
    return ArtifactRef(uri=uri, format="bin", content_hash="sha256:0", size_bytes=0)


def test_local_store_put_and_fetch_round_trip(tmp_path: Path) -> None:
    store = artifact_store_for(f"file://{tmp_path}/store", run_prefix="run1")
    assert isinstance(store, LocalArtifactStore)
    assert store.uri == f"file://{tmp_path}/store"

    source = tmp_path / "model.bin"
    source.write_bytes(b"weights")
    ref = store.put_file(source, "model.bin", format="bin")

    assert ref.format == "bin"
    assert ref.size_bytes == len(b"weights")
    assert ref.content_hash == "sha256:" + hashlib.sha256(b"weights").hexdigest()
    assert store.fetch(ref).read_bytes() == b"weights"


def test_local_store_rejects_non_file_uri() -> None:
    with pytest.raises(MbtError, match="unsupported artifact store URI"):
        LocalArtifactStore("s3://bucket/key")


def test_local_fetch_rejects_foreign_uri_and_missing_file(tmp_path: Path) -> None:
    store = artifact_store_for(f"file://{tmp_path}")
    with pytest.raises(MbtError, match="only resolves file"):
        store.fetch(_ref("s3://bucket/key"))
    with pytest.raises(MbtError, match="artifact not found"):
        store.fetch(_ref(f"file://{tmp_path}/gone.bin"))


def test_s3_store_rejects_bucketless_uri() -> None:
    # The bucket check precedes any client/network call, so this needs no creds.
    with pytest.raises(MbtError, match="invalid s3 artifact store URI"):
        S3ArtifactStore("s3://")


def test_read_uri_text_reads_file_uri_and_bare_path(tmp_path: Path) -> None:
    state = tmp_path / "state.json"
    state.write_text("baseline")
    assert read_uri_text(f"file://{state}") == "baseline"
    assert read_uri_text(str(state)) == "baseline"  # bare filesystem path


def test_read_uri_text_missing_reference_is_a_hard_error(tmp_path: Path) -> None:
    # FR-STATE-01: an unreadable --state reference must fail loudly, never
    # degrade into a silent full retrain.
    with pytest.raises(StateError, match="file not found"):
        read_uri_text(str(tmp_path / "nope.json"))


def test_artifact_exists_probes_file_uris_and_reports_unknown_schemes(tmp_path: Path) -> None:
    """F12's head probe: True/False for file:// by real presence, None for a
    scheme it cannot probe (the rollback caller then proceeds with a warning)."""
    from mbt.contracts import ArtifactRef
    from mbt.storage import artifact_exists

    def ref(uri: str) -> ArtifactRef:
        return ArtifactRef(uri=uri, format="test_bin", content_hash="sha256:a", size_bytes=1)

    present = tmp_path / "model.bin"
    present.write_bytes(b"weights")
    assert artifact_exists(ref(f"file://{present}")) is True
    assert artifact_exists(ref(f"file://{tmp_path}/gone.bin")) is False
    assert artifact_exists(ref("memory://somewhere/model.bin")) is None
