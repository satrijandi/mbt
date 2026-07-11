"""Seed the SeaweedFS lake for the showcase (runs inside the runner image).

Creates the buckets and uploads the deterministic demo parquet staged under
/workspace/seed/<table>/ to s3://mbt-lake/<table>/. Idempotent: re-running
overwrites the same keys with the same bytes.

Buckets are created WITHOUT any TTL/retention configuration on purpose:
mbt clean refuses s3:// artifact stores and nothing protects champion
objects server-side, so a retention rule would silently break champion
gates, evaluation, and scoring (DESIGN.md section 9).
"""

import os
import sys
from pathlib import Path

import boto3

LAKE_BUCKET = "mbt-lake"
ARTIFACT_BUCKET = "mbt-artifacts"
SEED_DIR = Path(os.environ.get("SHOWCASE_SEED_DIR", "/workspace/seed"))


def main() -> int:
    s3 = boto3.client("s3")
    existing = {b["Name"] for b in s3.list_buckets().get("Buckets", [])}
    for bucket in (LAKE_BUCKET, ARTIFACT_BUCKET):
        if bucket not in existing:
            s3.create_bucket(Bucket=bucket)
            print(f"created bucket {bucket}")

    if not SEED_DIR.is_dir():
        print(f"seed dir {SEED_DIR} not found", file=sys.stderr)
        return 1

    uploaded = 0
    for table_dir in sorted(p for p in SEED_DIR.iterdir() if p.is_dir()):
        for part in sorted(table_dir.glob("*.parquet")):
            key = f"{table_dir.name}/{part.name}"
            s3.upload_file(str(part), LAKE_BUCKET, key)
            uploaded += 1
            print(f"uploaded s3://{LAKE_BUCKET}/{key} ({part.stat().st_size} bytes)")
    if uploaded == 0:
        print(f"no parquet found under {SEED_DIR}", file=sys.stderr)
        return 1
    print(f"seeded {uploaded} object(s) into s3://{LAKE_BUCKET}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
