#!/usr/bin/env bash
# Build the showcase's Airflow image from the current checkout.
#
# The DAGs import mbt-airflow, which is not on PyPI at this checkout's version,
# so Airflow cannot run upstream's image as it is: images/airflow/Dockerfile
# puts this checkout's mbt-airflow wheels (and the docker SDK) on top of
# apache/airflow. images/airflow/stage.sh stages the build context into a cache
# dir OUTSIDE the repo, as build_image.sh does for the runner.
#
# Staleness: the build is skipped only when the existing image's content label
# matches a hash of everything that shapes it (package sources and pyprojects,
# uv.lock, images/airflow, this script). A bare tag-existence check would let
# week-old wheels pass as "the current build" and every DAG assertion would
# silently test old code.
#
# The image exists only in the local daemon, and the localhost/ name keeps it
# that way: a bare name means docker.io/library/<name>, so a lookup that ever
# reached for the image's origin would fail against Docker Hub instead of
# locally (see build_image.sh). Unlike the runner there is no prebuilt copy to
# pull: the build is a few minutes.
#
# Usage: build_airflow_image.sh [--force | --print-hash]
set -euo pipefail

REPO_ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/../../.." && pwd)"
AIRFLOW_DIR="$REPO_ROOT/examples/showcase/images/airflow"
CACHE_DIR="${MBT_SHOWCASE_CACHE:-$HOME/.cache/mbt-showcase}/build-airflow"
CONTENT_LABEL="mbt.showcase.content"

# The tag is the Airflow release the Dockerfile packages, and compose names it
# the same way (tests/test_showcase_image_pins.py holds the two together).
VERSION="$(sed -nE 's/^ARG VERSION=(.+)$/\1/p' "$AIRFLOW_DIR/Dockerfile" | head -1)"
[ -n "$VERSION" ] || {
    echo "build_airflow_image.sh: images/airflow/Dockerfile declares no ARG VERSION=" >&2
    exit 1
}
IMAGE_TAG="localhost/mbt-showcase-airflow:$VERSION"

# Package tests/READMEs stay out of the hash on purpose: they never reach the
# installed wheels, and hashing them would force pointless rebuilds.
CONTENT_HASH="$(python3 - "$REPO_ROOT" <<'EOF'
import hashlib
import sys
from pathlib import Path

root = Path(sys.argv[1])
paths = [root / "uv.lock"]
paths += sorted((root / "packages").glob("*/pyproject.toml"))
for src in sorted((root / "packages").glob("*/src")):
    paths += sorted(
        p for p in src.rglob("*") if p.is_file() and "__pycache__" not in p.parts
    )
airflow = root / "examples" / "showcase" / "images" / "airflow"
paths += sorted(p for p in airflow.rglob("*") if p.is_file() and "__pycache__" not in p.parts)
paths.append(root / "examples" / "showcase" / "scripts" / "build_airflow_image.sh")
digest = hashlib.sha256()
for path in paths:
    digest.update(str(path.relative_to(root)).encode())
    digest.update(path.read_bytes())
print(digest.hexdigest()[:16])
EOF
)"

if [ "${1:-}" = "--print-hash" ]; then
    echo "$CONTENT_HASH"
    exit 0
fi

content_label() {
    docker image inspect --format '{{json .Config.Labels}}' "$1" 2>/dev/null \
        | python3 -c "import json,sys; print((json.load(sys.stdin) or {}).get('$CONTENT_LABEL',''))" 2>/dev/null \
        || true
}

existing_hash="$(content_label "$IMAGE_TAG")"
if [ "${1:-}" != "--force" ] && [ -n "$existing_hash" ] && [ "$existing_hash" = "$CONTENT_HASH" ]; then
    echo "image $IMAGE_TAG up to date (content $CONTENT_HASH; use --force to rebuild anyway)"
    exit 0
fi

"$AIRFLOW_DIR/stage.sh" "$CACHE_DIR"

echo "==> docker build $IMAGE_TAG (content $CONTENT_HASH)"
docker build -t "$IMAGE_TAG" --label "$CONTENT_LABEL=$CONTENT_HASH" \
    --label "org.opencontainers.image.source=https://github.com/satrijandi/mbt" "$CACHE_DIR"
echo "==> built $IMAGE_TAG"
