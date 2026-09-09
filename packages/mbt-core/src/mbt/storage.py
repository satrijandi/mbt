"""Artifact stores (file://, s3:// via the s3 extra) and URI readers."""

import atexit
import hashlib
import shutil
import tempfile
import uuid
from pathlib import Path
from typing import Any

from mbt.contracts import ArtifactRef
from mbt.exceptions import MbtError, StateError

#: Read size for the streaming hashes below. Big enough that a multi-GB model
#: is not millions of syscalls, small enough that nothing large is ever resident.
_HASH_CHUNK = 4 * 1024 * 1024


def _sha256_file(path: Path) -> str:
    """The ``sha256:...`` digest of a file, read in chunks.

    Streaming rather than ``read_bytes()``: Spark model directories, H2O MOJO
    bundles, and large sklearn pipelines are routinely bigger than a runner's
    free memory, and every artifact write and every verified read hashes.
    """
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(_HASH_CHUNK), b""):
            digest.update(chunk)
    return "sha256:" + digest.hexdigest()


def _verify_content_hash(ref: ArtifactRef, path: Path) -> None:
    """Check the fetched bytes against the digest recorded when they were stored.

    Every artifact carries a SHA-256 from ``put_file``, and until this existed
    nothing ever compared it: every consumer of ``fetch()`` deserialized
    immediately, and for sklearn that is joblib, which is pickle, which is
    arbitrary code execution. A digest recorded at write time and never read is
    not a control - it is a record of one. The cases it catches are bucket
    tampering, a truncated multi-GB download, and a lifecycle rule that replaced
    an object under a key that still resolves.

    A ref with no digest is passed through with a warning rather than a failure:
    baseline and inference-config refs reconstructed from an older champion's
    registry tags legitimately carry ``""`` (see ``_baseline_ref``), and refusing
    to score with a champion registered by an older mbt would be a worse outcome
    than the check it skips.
    """
    if not ref.content_hash.startswith("sha256:"):
        from mbt.events import get_bus
        from mbt.events.models import LogMessage

        get_bus().emit(
            LogMessage(
                level="warn",
                message=(
                    f"artifact {ref.uri} carries no content hash (stored by an older "
                    "mbt); its bytes are used without integrity verification"
                ),
            )
        )
        return
    actual = _sha256_file(path)
    if actual != ref.content_hash:
        raise MbtError(
            f"artifact content hash mismatch for {ref.uri}",
            hint=(
                f"expected {ref.content_hash}, got {actual}. The bytes changed since "
                "the artifact was stored - suspect object-store tampering, a "
                "truncated download, or a lifecycle rule that replaced the object. "
                "Do not load it; re-run the build that produced it."
            ),
        )


def _s3_client() -> Any:
    """A boto3 S3 client with bounded retry-with-backoff (R2-2).

    ``standard`` retry mode retries throttling, 5xx, and connection errors with
    exponential backoff plus jitter (botocore maintains the transient-error
    set), up to 5 retries - so a single ``503 SlowDown`` or network blip during
    an artifact upload/download does not fail a node after its GPU-hours.
    """
    import boto3
    from botocore.config import Config

    return boto3.client("s3", config=Config(retries={"max_attempts": 5, "mode": "standard"}))


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
        # Hashed by streaming the copy, not by reading it back into a bytes
        # object: a 2 GB model used to be read twice and held once.
        return ArtifactRef(
            uri=f"file://{destination.resolve()}",
            format=format,
            content_hash=_sha256_file(destination),
            size_bytes=destination.stat().st_size,
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
        _verify_content_hash(ref, path)
        return path


class S3ArtifactStore:
    """An s3:// artifact store; each run writes under a unique prefix.

    Requires the s3 extra (boto3). ``fetch`` downloads into a per-store
    local cache so adapters keep working with plain local paths. Retention
    for object stores is bucket lifecycle rules, not ``mbt clean``.
    """

    def __init__(self, uri: str, run_prefix: str | None = None) -> None:
        try:
            import boto3  # noqa: F401 - probe the s3 extra before building the client
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
        self._client = _s3_client()
        # Download cache: lives as long as the process (fetches reuse it),
        # removed at exit so long-lived runners do not accumulate copies.
        self._cache = Path(tempfile.mkdtemp(prefix="mbt-s3-artifacts-"))
        atexit.register(shutil.rmtree, self._cache, ignore_errors=True)

    @property
    def uri(self) -> str:
        return self._uri

    def put_file(self, local_path: Path, name: str, format: str) -> ArtifactRef:
        key = "/".join(part for part in (self._base, self._prefix, name) if part)
        # upload_file, not put_object: the managed transfer streams from disk and
        # switches to multipart above its threshold. put_object read the whole
        # artifact into memory and is a single PUT, which S3 caps at 5 GiB - and
        # the failure lands after the training hours are already spent, which is
        # the case _s3_client's retry config exists to protect against. fetch()
        # has always used the managed download_file; this is its other half.
        self._client.upload_file(str(local_path), self._bucket, key)
        return ArtifactRef(
            uri=f"s3://{self._bucket}/{key}",
            format=format,
            content_hash=_sha256_file(local_path),
            size_bytes=local_path.stat().st_size,
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
            # Verified on download only. The cache is this process's own temp
            # dir, so a hit re-verifies bytes nobody else could have touched -
            # and re-hashing a multi-GB model on every fetch is not free.
            _verify_content_hash(ref, target)
        return target


def artifact_store_for(
    uri: str, run_prefix: str | None = None
) -> "LocalArtifactStore | S3ArtifactStore":
    if uri.startswith("s3://"):
        return S3ArtifactStore(uri, run_prefix=run_prefix)
    return LocalArtifactStore(uri, run_prefix=run_prefix)


def artifact_exists(ref: ArtifactRef) -> bool | None:
    """Whether the file BEHIND an artifact reference still exists - a cheap
    head probe, no download.

    The registry can outlive the artifact: ``mbt clean`` ages local files out
    and bucket lifecycle rules do the same on S3, so a surviving ``ArtifactRef``
    proves nothing. An incident-response rollback must refuse a destination
    that cannot serve (F12). Returns None when the probe cannot run (an
    unrecognized scheme, s3 without the s3 extra, or an S3 error that is not a
    clean not-found): the caller proceeds with a warning rather than blocking
    an incident on an unprobeable store.
    """
    if ref.uri.startswith("file://"):
        return Path(ref.uri.removeprefix("file://")).is_file()
    if ref.uri.startswith("s3://"):
        try:
            from botocore.exceptions import ClientError
        except ImportError:  # pragma: no cover - env dependent
            return None
        bucket, _, key = ref.uri.removeprefix("s3://").partition("/")
        try:
            _s3_client().head_object(Bucket=bucket, Key=key)
        except ClientError as exc:
            code = str(exc.response.get("Error", {}).get("Code", ""))
            if code in ("404", "NoSuchKey", "NotFound"):
                return False
            return None  # permissions / transient after retries: cannot probe
        return True
    return None


def read_uri_text(uri_or_path: str) -> str:
    """Read text from file://, s3://, or a bare filesystem path (FR-STATE-01)."""
    if uri_or_path.startswith("s3://"):
        try:
            import boto3  # noqa: F401 - probe the s3 extra before building the client
        except ImportError as exc:  # pragma: no cover - env dependent
            raise StateError(
                "reading s3:// URIs requires the s3 extra",
                hint="pip install 'mbt-core[s3]'",
            ) from exc
        bucket, _, key = uri_or_path.removeprefix("s3://").partition("/")
        try:
            body = _s3_client().get_object(Bucket=bucket, Key=key)["Body"].read()
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
