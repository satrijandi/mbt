"""MLflow tracking + registry adapters (TSD §13.3, FR-REG-01/04).

One package, one plugin, two contracts sharing client config (``uri``).
``import mlflow`` happens lazily inside methods (ADR-14).

Canonical mbt stages map to registered-model aliases by default
(``staging``/``production``/``archived``, exclusive per version to keep
mbt's one-stage-at-a-time model); MLflow deprecated stage transitions and
removes them in MLflow 4. ``use_aliases: false`` opts back into the legacy
stage API for registry servers without alias support (MLflow < 2.9).
"""

import functools
import random
import time
from collections.abc import Callable
from pathlib import Path
from typing import TYPE_CHECKING, Any, ParamSpec, TypeVar

from mbt_adapter_base import (
    ArtifactRef,
    ManifestNode,
    ModelVersion,
    RunHandle,
    Stage,
)

if TYPE_CHECKING:
    from mlflow.tracking import MlflowClient

_P = ParamSpec("_P")
_T = TypeVar("_T")

#: A locked SQLite backend - the common local/CI registry+tracking failure the
#: review flagged (R2-2) - surfaces as one of these messages. A locked database
#: means the write never committed, so retrying is safe even for a
#: non-idempotent create_model_version.
_RETRYABLE_MARKERS = ("database is locked", "database table is locked")


def _is_retryable(exc: BaseException) -> bool:
    return any(marker in str(exc).lower() for marker in _RETRYABLE_MARKERS)


#: mlflow's error code for a genuinely absent model / version.
_NOT_FOUND_CODE = "RESOURCE_DOES_NOT_EXIST"


def _is_missing(exc: BaseException) -> bool:
    """True when mlflow reports a genuinely absent alias / stage / version.

    mlflow is inconsistent here: a missing model or version raises
    ``RESOURCE_DOES_NOT_EXIST``, but a missing *alias* raises
    ``INVALID_PARAMETER_VALUE`` with a "... not found" message. Both phrase the
    absence as "not found" / "does not exist", and a transient backend failure
    (a locked DB, a 500/503, a dropped connection) never does - so this
    distinguishes a genuine absence from an error that must not be read as
    'no champion' and silently skip the challenger gate (F9)."""
    if getattr(exc, "error_code", None) == _NOT_FOUND_CODE:
        return True
    text = str(exc).lower()
    return "not found" in text or "does not exist" in text


def _lookup_or_none(fn: Callable[[], _T]) -> _T | None:
    """Run a registry read; map a genuine not-found (see :func:`_is_missing`)
    to ``None`` but let every other ``MlflowException`` (a locked DB, an
    unreachable / 500 backend) propagate - so a merely transient error is never
    swallowed as 'no champion', which would skip the challenger gate on a build
    or hard-fail ``mbt score`` with a misleading message (F9)."""
    from mlflow.exceptions import MlflowException

    try:
        return fn()
    except MlflowException as exc:
        if _is_missing(exc):
            return None
        raise


def _with_retry(fn: Callable[[], _T], *, attempts: int = 6, base_delay: float = 0.05) -> _T:
    """Run ``fn``, retrying a locked-database error with bounded exponential
    backoff; any other error, and the final attempt, propagate (R2-2)."""
    for attempt in range(attempts - 1):
        try:
            return fn()
        except Exception as exc:  # re-raised below unless it is a retryable lock
            if not _is_retryable(exc):
                raise
            # Full jitter over [0, base_delay * 2**attempt]: two jobs colliding
            # on the same locked DB must not re-collide on identical
            # deterministic sleeps (the thundering-herd the retry exists to
            # break), matching the S3 seam's jittered backoff (F10).
            time.sleep(random.uniform(0, base_delay * 2**attempt))
    return fn()


def _retryable(method: Callable[_P, _T]) -> Callable[_P, _T]:
    """Decorate an adapter I/O method to retry a transient locked-DB error."""

    @functools.wraps(method)
    def wrapper(*args: _P.args, **kwargs: _P.kwargs) -> _T:
        return _with_retry(lambda: method(*args, **kwargs))

    return wrapper


_STAGE_MAP = {
    Stage.STAGING: "Staging",
    Stage.PRODUCTION: "Production",
    Stage.ARCHIVED: "Archived",
}
_STAGE_REVERSE = {v: k for k, v in _STAGE_MAP.items()}
#: Alias-derived stage resolution order: a version should hold at most one
#: canonical alias (transition keeps them exclusive), the order is a safety
#: net for aliases set outside mbt.
_ALIAS_STAGES = (Stage.PRODUCTION, Stage.STAGING, Stage.ARCHIVED)

_ARTIFACT_TAGS = (
    "mbt.artifact_uri",
    "mbt.artifact_format",
    "mbt.artifact_content_hash",
    "mbt.artifact_size_bytes",
)


class _MlflowBase:
    def __init__(self, config: dict[str, Any] | None = None) -> None:
        config = config or {}
        self.uri: str = str(config.get("uri", "sqlite:///mlflow.db"))
        self.experiment: str = str(config.get("experiment", "mbt"))
        self.use_aliases: bool = bool(config.get("use_aliases", True))
        self._client: MlflowClient | None = None

    def client(self) -> "MlflowClient":
        if self._client is None:
            from mlflow.tracking import MlflowClient

            self._client = MlflowClient(tracking_uri=self.uri, registry_uri=self.uri)
        return self._client


class MlflowTracking(_MlflowBase):
    """TrackingAdapter over the MLflow client API (no fluent globals)."""

    def prepare(self) -> None:
        """Run store migrations once, before parallel jobs hit the backend.

        Called by the mbt coordinator when present; prevents alembic
        migration races on fresh sqlite databases.
        """
        self._experiment_id()

    def _experiment_id(self) -> str:
        client = self.client()
        experiment = client.get_experiment_by_name(self.experiment)
        if experiment is not None:
            return str(experiment.experiment_id)
        return str(client.create_experiment(self.experiment))

    @_retryable
    def start_run(self, node: ManifestNode, meta: dict[str, str]) -> RunHandle:
        client = self.client()
        tags = {"mlflow.runName": node.name, "mbt.unique_id": node.unique_id, **meta}
        run = client.create_run(self._experiment_id(), tags=tags)
        return RunHandle(run_id=run.info.run_id)

    @_retryable
    def log(
        self,
        run: RunHandle,
        *,
        params: dict[str, Any] | None = None,
        metrics: dict[str, float] | None = None,
        tags: dict[str, str] | None = None,
        artifacts: list[ArtifactRef] | None = None,
    ) -> None:
        client = self.client()
        for key, value in (params or {}).items():
            client.log_param(run.run_id, key, value)
        for key, value in (metrics or {}).items():
            client.log_metric(run.run_id, key, float(value))
        for key, value in (tags or {}).items():
            client.set_tag(run.run_id, key, value)
        for artifact in artifacts or []:
            client.set_tag(run.run_id, f"mbt.artifact.{artifact.format}", artifact.uri)
            if artifact.uri.startswith("file://"):
                path = Path(artifact.uri.removeprefix("file://"))
                if path.is_file():
                    client.log_artifact(run.run_id, str(path))

    @_retryable
    def log_trial(self, run: RunHandle, index: int, params: dict[str, Any], value: float) -> None:
        """Tuning trial history as nested runs (FR-TUNE-03)."""
        client = self.client()
        nested = client.create_run(
            self._experiment_id(),
            tags={
                "mlflow.parentRunId": run.run_id,
                "mlflow.runName": f"trial-{index:03d}",
                "mbt.tuning.trial": str(index),
            },
        )
        for key, val in params.items():
            client.log_param(nested.info.run_id, key, val)
        client.log_metric(nested.info.run_id, "objective", float(value))
        client.set_terminated(nested.info.run_id, "FINISHED")

    @_retryable
    def end_run(self, run: RunHandle, status: str) -> None:
        self.client().set_terminated(run.run_id, status)

    def resume(self, run_id: str) -> RunHandle:
        return RunHandle(run_id=run_id)


class MlflowRegistry(_MlflowBase):
    """RegistryAdapter over the MLflow model registry."""

    def _ensure_registered_model(self, name: str) -> None:
        import contextlib

        from mlflow.exceptions import MlflowException

        with contextlib.suppress(MlflowException):  # already exists
            self.client().create_registered_model(name)

    @_retryable
    def register(self, artifact: ArtifactRef, name: str, metadata: dict[str, str]) -> ModelVersion:
        client = self.client()
        self._ensure_registered_model(name)
        tags = dict(metadata)
        tags.setdefault("mbt.artifact_uri", artifact.uri)
        tags.setdefault("mbt.artifact_format", artifact.format)
        tags.setdefault("mbt.artifact_content_hash", artifact.content_hash)
        tags.setdefault("mbt.artifact_size_bytes", str(artifact.size_bytes))
        version = client.create_model_version(
            name=name,
            source=artifact.uri,
            run_id=metadata.get("mbt.tracking_run_id") or None,
            tags=tags,
        )
        return ModelVersion(name=name, version=str(version.version), artifact=artifact, tags=tags)

    def _to_model_version(self, mv: Any) -> ModelVersion:
        tags = dict(mv.tags or {})
        artifact = None
        if all(tag in tags for tag in _ARTIFACT_TAGS):
            artifact = ArtifactRef(
                uri=tags["mbt.artifact_uri"],
                format=tags["mbt.artifact_format"],
                content_hash=tags["mbt.artifact_content_hash"],
                size_bytes=int(tags["mbt.artifact_size_bytes"]),
            )
        stage = _STAGE_REVERSE.get(getattr(mv, "current_stage", None) or "")
        if stage is None:
            aliases = set(getattr(mv, "aliases", None) or [])
            stage = next((s for s in _ALIAS_STAGES if s.value in aliases), None)
        return ModelVersion(
            name=mv.name,
            version=str(mv.version),
            stage=stage,
            artifact=artifact,
            tags=tags,
        )

    @_retryable
    def get_champion(self, name: str, stage: Stage) -> ModelVersion | None:
        client = self.client()
        if self.use_aliases:
            mv = _lookup_or_none(lambda: client.get_model_version_by_alias(name, stage.value))
            return self._to_model_version(mv) if mv is not None else None
        versions = _lookup_or_none(lambda: client.search_model_versions(f"name = '{name}'"))
        if not versions:
            return None
        in_stage = [v for v in versions if v.current_stage == _STAGE_MAP[stage]]
        if not in_stage:
            return None
        latest = max(in_stage, key=lambda v: int(v.version))
        return self._to_model_version(latest)

    @_retryable
    def get_version(self, name: str, version: str) -> ModelVersion | None:
        mv = _lookup_or_none(lambda: self.client().get_model_version(name, version))
        return self._to_model_version(mv) if mv is not None else None

    def transition(self, version: ModelVersion, stage: Stage) -> None:
        import warnings

        from mlflow.exceptions import MlflowException

        client = self.client()
        if self.use_aliases:
            # Capture the champion this promotion displaces BEFORE the alias
            # moves off it. `transition` is deliberately NOT `@_retryable`:
            # a whole-method retry after the alias had already moved would
            # re-read the production alias as the NEW version, so `outgoing`
            # would become the new version and the displaced champion would
            # never be archived (F8). Instead each write below retries
            # individually, so the sequence runs once top to bottom and
            # `outgoing` stays fixed to the true previous champion.
            outgoing: str | None = None
            if stage is Stage.PRODUCTION:
                try:
                    # mlflow returns .version as an int here; version.version is
                    # a str, so normalize before the identity comparison below.
                    current = _with_retry(
                        lambda: client.get_model_version_by_alias(version.name, stage.value)
                    )
                    outgoing = str(current.version)
                except MlflowException:
                    outgoing = None  # no current champion to displace
            _with_retry(
                lambda: client.set_registered_model_alias(
                    version.name, stage.value, version.version
                )
            )
            # mbt stages are exclusive per version: drop the other canonical
            # aliases this version still holds so promoting from staging to
            # production also vacates the staging slot (stage-API parity).
            mv = _with_retry(lambda: client.get_model_version(version.name, version.version))
            for alias in set(getattr(mv, "aliases", None) or []):
                if alias != stage.value and alias in {s.value for s in _ALIAS_STAGES}:
                    # functools.partial binds `alias` now, not by late reference,
                    # so the retried thunk always deletes the right alias.
                    _with_retry(
                        functools.partial(client.delete_registered_model_alias, version.name, alias)
                    )
            # Archive the displaced champion (mirrors archive_existing_versions),
            # so `archived` is queryable for rollback in both backends.
            if outgoing is not None and outgoing != version.version:
                _with_retry(
                    lambda: client.set_registered_model_alias(
                        version.name, Stage.ARCHIVED.value, outgoing
                    )
                )
            return
        with warnings.catch_warnings():
            # Legacy stage flow for registry servers without alias support
            # (MLflow < 2.9); stages are deprecated upstream, hence the
            # suppression. Aliases are the default (`use_aliases: true`).
            warnings.simplefilter("ignore", FutureWarning)
            _with_retry(
                lambda: client.transition_model_version_stage(
                    name=version.name,
                    version=version.version,
                    stage=_STAGE_MAP[stage],
                    archive_existing_versions=(stage is Stage.PRODUCTION),
                )
            )
