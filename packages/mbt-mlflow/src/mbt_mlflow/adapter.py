"""MLflow tracking + registry adapters (TSD §13.3, FR-REG-01/04).

One package, one plugin, two contracts sharing client config (``uri``).
``import mlflow`` happens lazily inside methods (ADR-14).

Canonical mbt stages map to MLflow stages (``staging -> Staging`` ...);
``use_aliases: true`` switches to registered-model aliases on MLflow >= 2.9.
"""

from pathlib import Path
from typing import TYPE_CHECKING, Any

from mbt_adapter_base import (
    ArtifactRef,
    ManifestNode,
    ModelVersion,
    RunHandle,
    Stage,
)

if TYPE_CHECKING:
    from mlflow.tracking import MlflowClient

_STAGE_MAP = {
    Stage.STAGING: "Staging",
    Stage.PRODUCTION: "Production",
    Stage.ARCHIVED: "Archived",
}
_STAGE_REVERSE = {v: k for k, v in _STAGE_MAP.items()}

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
        self.use_aliases: bool = bool(config.get("use_aliases", False))
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

    def start_run(self, node: ManifestNode, meta: dict[str, str]) -> RunHandle:
        client = self.client()
        tags = {"mlflow.runName": node.name, "mbt.unique_id": node.unique_id, **meta}
        run = client.create_run(self._experiment_id(), tags=tags)
        return RunHandle(run_id=run.info.run_id)

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
        return ModelVersion(
            name=mv.name,
            version=str(mv.version),
            stage=stage,
            artifact=artifact,
            tags=tags,
        )

    def get_champion(self, name: str, stage: Stage) -> ModelVersion | None:
        from mlflow.exceptions import MlflowException

        client = self.client()
        if self.use_aliases:
            try:
                mv = client.get_model_version_by_alias(name, stage.value)
            except MlflowException:
                return None
            return self._to_model_version(mv)
        try:
            versions = client.search_model_versions(f"name = '{name}'")
        except MlflowException:
            return None
        in_stage = [v for v in versions if v.current_stage == _STAGE_MAP[stage]]
        if not in_stage:
            return None
        latest = max(in_stage, key=lambda v: int(v.version))
        return self._to_model_version(latest)

    def get_version(self, name: str, version: str) -> ModelVersion | None:
        from mlflow.exceptions import MlflowException

        try:
            mv = self.client().get_model_version(name, version)
        except MlflowException:
            return None
        return self._to_model_version(mv)

    def transition(self, version: ModelVersion, stage: Stage) -> None:
        import warnings

        client = self.client()
        if self.use_aliases:
            client.set_registered_model_alias(version.name, stage.value, version.version)
            return
        with warnings.catch_warnings():
            # Stages are deprecated upstream but remain the default mapping
            # for canonical mbt stages; opt into aliases with
            # `use_aliases: true` (TSD §13.3).
            warnings.simplefilter("ignore", FutureWarning)
            client.transition_model_version_stage(
                name=version.name,
                version=version.version,
                stage=_STAGE_MAP[stage],
                archive_existing_versions=(stage is Stage.PRODUCTION),
            )
