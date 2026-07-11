"""Sync the SeaweedFS lake to the local scoring plane (runs in the runner).

mbt score/monitor use the LOCAL data adapter (mbt-spark implements no
contract-1.1 scoring methods), rooted at /workspace/lake_local. This mirrors
s3://mbt-lake there.

Downloaded files get a FIXED mtime: the local adapter's default snapshot
tokens hash (path, size, mtime) listings, and a run_key that forked on every
sync would silently destroy prediction-store idempotency (DESIGN.md section
6). Score/monitor additionally pass --deep-snapshot; this is belt and braces.
"""

import os
from pathlib import Path

import boto3

LAKE_BUCKET = "mbt-lake"
DEST = Path(os.environ.get("SHOWCASE_LAKE_LOCAL", "/workspace/lake_local"))
FIXED_MTIME = 1_750_000_000  # arbitrary constant, deliberately not "now"


def main() -> int:
    s3 = boto3.client("s3")
    count = 0
    paginator = s3.get_paginator("list_objects_v2")
    for page in paginator.paginate(Bucket=LAKE_BUCKET):
        for obj in page.get("Contents", []):
            key = obj["Key"]
            dest = DEST / key
            dest.parent.mkdir(parents=True, exist_ok=True)
            s3.download_file(LAKE_BUCKET, key, str(dest))
            os.utime(dest, (FIXED_MTIME, FIXED_MTIME))
            count += 1
    print(f"synced {count} object(s) from s3://{LAKE_BUCKET} to {DEST}")
    return 0 if count else 1


if __name__ == "__main__":
    raise SystemExit(main())
