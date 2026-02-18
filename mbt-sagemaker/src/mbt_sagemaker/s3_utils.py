"""S3 utilities for data upload/download."""

import pandas as pd
import boto3
from io import StringIO
import uuid


def upload_dataframe_to_s3(
    df: pd.DataFrame,
    bucket: str,
    prefix: str,
    session: boto3.Session,
    include_header: bool = False,
) -> str:
    """Upload DataFrame to S3 as CSV.

    SageMaker built-in algorithms expect CSV format with:
    - Target column as first column
    - No header row
    - No index column

    SageMaker Autopilot expects CSV format with:
    - Header row included
    - Target column can be in any position

    Args:
        df: DataFrame to upload (target as first column for built-in algos)
        bucket: S3 bucket name
        prefix: S3 key prefix (e.g., "mbt-models/training-data/")
        session: Boto3 session with AWS credentials
        include_header: If True, include column headers (required for Autopilot)

    Returns:
        S3 URI of uploaded file (s3://bucket/key)

    Raises:
        Exception: If S3 upload fails
    """
    csv_buffer = StringIO()
    df.to_csv(csv_buffer, index=False, header=include_header)

    # Generate unique key
    key = f"{prefix}train-{uuid.uuid4().hex[:8]}.csv"

    # Upload to S3
    s3_client = session.client("s3")
    s3_client.put_object(
        Bucket=bucket,
        Key=key,
        Body=csv_buffer.getvalue().encode("utf-8"),
    )

    return f"s3://{bucket}/{key}"


def download_from_s3(s3_uri: str, local_path: str, session: boto3.Session):
    """Download file from S3 to local path.

    Args:
        s3_uri: S3 URI (s3://bucket/key)
        local_path: Local file path to save to
        session: Boto3 session

    Raises:
        ValueError: If S3 URI is invalid
        Exception: If download fails
    """
    # Parse S3 URI
    if not s3_uri.startswith("s3://"):
        raise ValueError(f"Invalid S3 URI: {s3_uri}")

    parts = s3_uri.replace("s3://", "").split("/", 1)
    if len(parts) != 2:
        raise ValueError(f"Invalid S3 URI format: {s3_uri}")

    bucket = parts[0]
    key = parts[1]

    # Download
    s3_client = session.client("s3")
    s3_client.download_file(bucket, key, local_path)
