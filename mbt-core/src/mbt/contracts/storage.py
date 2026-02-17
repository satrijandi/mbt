"""Storage plugin contract - artifact persistence and retrieval.

Storage plugins handle artifact I/O between steps, especially critical
in distributed execution where steps run as separate processes/pods.
"""

from abc import ABC, abstractmethod
from typing import Optional


class StoragePlugin(ABC):
    """Interface for storage backend adapters (local, S3, GCS, etc.)."""

    @abstractmethod
    def put(
        self,
        artifact_name: str,
        data: bytes,
        run_id: str,
        step_name: str,
        metadata: Optional[dict] = None,
    ) -> str:
        """Store an artifact and return its URI.

        Args:
            artifact_name: Name of the artifact (e.g., "model", "scaler", "metrics")
            data: Serialized artifact data
            run_id: Unique run identifier
            step_name: Step that produced this artifact
            metadata: Optional metadata (e.g., {"type": "model", "framework": "sklearn"})

        Returns:
            URI string (e.g., "file:///path/...", "s3://bucket/...")
        """
        ...

    @abstractmethod
    def get(self, artifact_uri: str) -> bytes:
        """Retrieve an artifact by its URI.

        Args:
            artifact_uri: URI returned by put()

        Returns:
            Deserialized artifact data
        """
        ...

    @abstractmethod
    def exists(self, artifact_uri: str) -> bool:
        """Check if an artifact exists.

        Used by retry logic to skip completed idempotent steps.
        """
        ...

    @abstractmethod
    def list_artifacts(self, run_id: str, step_name: str) -> list[str]:
        """List all artifact URIs for a step in a run."""
        ...

    def configure(self, config: dict) -> None:
        """Configure the storage plugin with connection/path settings.

        Called after plugin instantiation to pass environment-specific config
        from profiles.yaml. Default implementation is a no-op.

        Args:
            config: Storage configuration (e.g., {"base_path": "./artifacts"}
                    or {"bucket": "my-bucket", "endpoint_url": "..."})
        """
        pass
