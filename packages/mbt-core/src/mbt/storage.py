"""Artifact store and URI readers (file:// in v0, s3:// via the s3 extra)."""

import hashlib
import shutil
import uuid
from pathlib import Path

from mbt.contracts import ArtifactRef
from mbt.exceptions import MbtError, StateError


class LocalArtifactStore:
    """A file:// artifact store; each run writes under a unique prefix."""

    def __init__(self, uri: str, run_prefix: str | None = None) -> None:
        if not uri.startswith("file://"):
            raise MbtError(
                f"unsupported artifact store URI: {uri!r}",
                hint="v0 supports file:// stores; s3:// artifact stores arrive with remote compute",
            )
        self._root = Path(uri.removeprefix("file://"))
        self._prefix = run_prefix or uuid.uuid4().hex[:16]
        self._uri = uri

    @property
    def uri(self) -> str:
        return self._uri

    def put_file(self, local_path: Path, name: str, format: str) -> ArtifactRef:
        destination = self._root / self._prefix / name
        destination.parent.mkdir(parents=True, exist_ok=True)
        shutil.copyfile(local_path, destination)
        payload = destination.read_bytes()
        return ArtifactRef(
            uri=f"file://{destination.resolve()}",
            format=format,
            content_hash="sha256:" + hashlib.sha256(payload).hexdigest(),
            size_bytes=len(payload),
        )

    def fetch(self, ref: ArtifactRef) -> Path:
        if not ref.uri.startswith("file://"):
            raise MbtError(
                f"cannot fetch artifact from {ref.uri!r}",
                hint="this store only resolves file:// artifact URIs",
            )
        path = Path(ref.uri.removeprefix("file://"))
        if not path.is_file():
            raise MbtError(
                f"artifact not found: {ref.uri}",
                hint="the artifact store may have been cleaned; re-run the build",
            )
        return path


def artifact_store_for(uri: str, run_prefix: str | None = None) -> LocalArtifactStore:
    return LocalArtifactStore(uri, run_prefix=run_prefix)


def read_uri_text(uri_or_path: str) -> str:
    """Read text from file://, s3://, or a bare filesystem path (FR-STATE-01)."""
    if uri_or_path.startswith("s3://"):
        try:
            import boto3
        except ImportError as exc:  # pragma: no cover - env dependent
            raise StateError(
                "reading s3:// URIs requires the s3 extra",
                hint="pip install 'mbt-core[s3]'",
            ) from exc
        bucket, _, key = uri_or_path.removeprefix("s3://").partition("/")
        try:
            body = boto3.client("s3").get_object(Bucket=bucket, Key=key)["Body"].read()
        except Exception as exc:
            raise StateError(
                f"cannot read {uri_or_path}: {exc}",
                hint="check the bucket/key and AWS credentials",
            ) from exc
        return body.decode("utf-8")  # type: ignore[no-any-return]
    path = Path(uri_or_path.removeprefix("file://"))
    if not path.is_file():
        raise StateError(
            f"cannot read {uri_or_path}: file not found",
            hint="an unreadable --state reference is a hard error, never a silent full retrain",
        )
    return path.read_text()
