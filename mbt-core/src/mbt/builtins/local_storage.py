"""Local filesystem storage plugin - stores artifacts on disk."""

from pathlib import Path
from typing import Optional

from mbt.contracts.storage import StoragePlugin


class LocalStoragePlugin(StoragePlugin):
    """Stores artifacts in local filesystem under ./local_artifacts/"""

    def __init__(self, base_path: str = "./local_artifacts"):
        self.base_path = Path(base_path)
        self.base_path.mkdir(parents=True, exist_ok=True)

    def put(
        self,
        artifact_name: str,
        data: bytes,
        run_id: str,
        step_name: str,
        metadata: Optional[dict] = None,
    ) -> str:
        """Store artifact to filesystem."""
        # Path: base_path/run_id/step_name/artifact_name
        artifact_dir = self.base_path / run_id / step_name
        artifact_dir.mkdir(parents=True, exist_ok=True)

        artifact_path = artifact_dir / artifact_name
        artifact_path.write_bytes(data)

        # Return file:// URI
        return f"file://{artifact_path.absolute()}"

    def get(self, artifact_uri: str) -> bytes:
        """Load artifact from filesystem."""
        # Extract path from file:// URI
        if artifact_uri.startswith("file://"):
            path = Path(artifact_uri[7:])  # Remove "file://"
        else:
            path = Path(artifact_uri)

        if not path.exists():
            raise FileNotFoundError(f"Artifact not found: {artifact_uri}")

        return path.read_bytes()

    def exists(self, artifact_uri: str) -> bool:
        """Check if artifact exists."""
        if artifact_uri.startswith("file://"):
            path = Path(artifact_uri[7:])
        else:
            path = Path(artifact_uri)

        return path.exists()

    def list_artifacts(self, run_id: str, step_name: str) -> list[str]:
        """List all artifacts for a step."""
        artifact_dir = self.base_path / run_id / step_name
        if not artifact_dir.exists():
            return []

        return [f"file://{p.absolute()}" for p in artifact_dir.iterdir() if p.is_file()]
