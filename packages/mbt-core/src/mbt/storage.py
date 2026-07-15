"""Artifact stores (file://, s3:// via the s3 extra) and URI readers."""

import atexit
import hashlib
import shutil
import tempfile
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
                hint="supported schemes: file://, s3:// (needs the s3 extra)",
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


class S3ArtifactStore:
    """An s3:// artifact store; each run writes under a unique prefix.

    Requires the s3 extra (boto3). ``fetch`` downloads into a per-store
    local cache so adapters keep working with plain local paths. Retention
    for object stores is bucket lifecycle rules, not ``mbt clean``.
    """

    def __init__(self, uri: str, run_prefix: str | None = None) -> None:
        try:
            import boto3
        except ImportError as exc:  # pragma: no cover - env dependent
            raise MbtError(
                "s3:// artifact stores require the s3 extra",
                hint="pip install 'mbt-core[s3]'",
            ) from exc
        bucket, _, base = uri.removeprefix("s3://").partition("/")
        if not bucket:
            raise MbtError(f"invalid s3 artifact store URI: {uri!r}")
        self._bucket = bucket
        self._base = base.strip("/")
        self._prefix = run_prefix or uuid.uuid4().hex[:16]
        self._uri = uri
        self._client = boto3.client("s3")
        # Download cache: lives as long as the process (fetches reuse it),
        # removed at exit so long-lived runners do not accumulate copies.
        self._cache = Path(tempfile.mkdtemp(prefix="mbt-s3-artifacts-"))
        atexit.register(shutil.rmtree, self._cache, ignore_errors=True)

    @property
    def uri(self) -> str:
        return self._uri

    def put_file(self, local_path: Path, name: str, format: str) -> ArtifactRef:
        payload = local_path.read_bytes()
        key = "/".join(part for part in (self._base, self._prefix, name) if part)
        self._client.put_object(Bucket=self._bucket, Key=key, Body=payload)
        return ArtifactRef(
            uri=f"s3://{self._bucket}/{key}",
            format=format,
            content_hash="sha256:" + hashlib.sha256(payload).hexdigest(),
            size_bytes=len(payload),
        )

    def fetch(self, ref: ArtifactRef) -> Path:
        if not ref.uri.startswith("s3://"):
            raise MbtError(
                f"cannot fetch artifact from {ref.uri!r}",
                hint="this store only resolves s3:// artifact URIs",
            )
        bucket, _, key = ref.uri.removeprefix("s3://").partition("/")
        target = self._cache / hashlib.sha256(ref.uri.encode()).hexdigest()[:16] / Path(key).name
        if not target.is_file():
            target.parent.mkdir(parents=True, exist_ok=True)
            try:
                self._client.download_file(bucket, key, str(target))
            except Exception as exc:
                raise MbtError(
                    f"artifact not found: {ref.uri} ({exc})",
                    hint="a lifecycle rule may have removed the object; re-run the build",
                ) from exc
        return target


def artifact_store_for(
    uri: str, run_prefix: str | None = None
) -> "LocalArtifactStore | S3ArtifactStore":
    if uri.startswith("s3://"):
        return S3ArtifactStore(uri, run_prefix=run_prefix)
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
