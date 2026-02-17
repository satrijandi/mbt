"""S3-compatible storage plugin for MBT.

Stores artifacts in S3-compatible object storage (AWS S3, SeaweedFS, MinIO).
"""

from typing import Optional

from mbt.contracts.storage import StoragePlugin


class S3StoragePlugin(StoragePlugin):
    """S3-compatible artifact storage.

    Configuration (in profiles.yaml):
        storage:
          type: s3
          config:
            bucket: mbt-pipeline-artifacts
            endpoint_url: http://seaweedfs-filer:8333
            access_key: my_access_key
            secret_key: my_secret_key
    """

    def __init__(self):
        self._client = None
        self._bucket: str = "mbt-pipeline-artifacts"

    def configure(self, config: dict) -> None:
        """Configure S3 client from profile config."""
        import boto3

        self._bucket = config.get("bucket", "mbt-pipeline-artifacts")
        endpoint_url = config.get("endpoint_url")
        access_key = config.get("access_key")
        secret_key = config.get("secret_key")
        region = config.get("region", "us-east-1")

        kwargs = {"region_name": region}
        if endpoint_url:
            kwargs["endpoint_url"] = endpoint_url
        if access_key is not None:
            kwargs["aws_access_key_id"] = access_key
        if secret_key is not None:
            kwargs["aws_secret_access_key"] = secret_key

        self._client = boto3.client("s3", **kwargs)

    def _ensure_client(self):
        if self._client is None:
            raise RuntimeError(
                "S3 storage not configured. Call configure() first."
            )

    def put(
        self,
        artifact_name: str,
        data: bytes,
        run_id: str,
        step_name: str,
        metadata: Optional[dict] = None,
    ) -> str:
        """Store artifact to S3."""
        self._ensure_client()

        key = f"{run_id}/{step_name}/{artifact_name}"
        self._client.put_object(Bucket=self._bucket, Key=key, Body=data)

        return f"s3://{self._bucket}/{key}"

    def get(self, artifact_uri: str) -> bytes:
        """Retrieve artifact from S3."""
        self._ensure_client()

        bucket, key = self._parse_uri(artifact_uri)
        response = self._client.get_object(Bucket=bucket, Key=key)
        return response["Body"].read()

    def exists(self, artifact_uri: str) -> bool:
        """Check if artifact exists in S3."""
        self._ensure_client()

        bucket, key = self._parse_uri(artifact_uri)
        try:
            self._client.head_object(Bucket=bucket, Key=key)
            return True
        except self._client.exceptions.ClientError:
            return False
        except Exception:
            return False

    def list_artifacts(self, run_id: str, step_name: str) -> list[str]:
        """List all artifacts for a step in S3."""
        self._ensure_client()

        prefix = f"{run_id}/{step_name}/"
        response = self._client.list_objects_v2(
            Bucket=self._bucket, Prefix=prefix
        )

        return [
            f"s3://{self._bucket}/{obj['Key']}"
            for obj in response.get("Contents", [])
        ]

    @staticmethod
    def _parse_uri(uri: str) -> tuple[str, str]:
        """Parse s3://bucket/key into (bucket, key)."""
        if not uri.startswith("s3://"):
            raise ValueError(f"Invalid S3 URI: {uri}")
        path = uri[5:]  # Remove "s3://"
        parts = path.split("/", 1)
        if len(parts) != 2:
            raise ValueError(f"Invalid S3 URI (no key): {uri}")
        return parts[0], parts[1]
