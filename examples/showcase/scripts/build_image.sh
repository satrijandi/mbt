#!/usr/bin/env bash
# Build the showcase runner image from the current checkout.
#
# Stages everything into a cache dir OUTSIDE the repo (the repo-root pytest
# session guard forbids new repo-root entries, and dist/ churn in-tree would
# be noise anyway):
#   1. uv build --all-packages          -> workspace wheels
#   2. uv export --frozen               -> third-party pins from uv.lock,
#      with the dev fork's pyspark/h2o lines swapped for the sparkling fork's
#      (pyspark 3.5.x; h2o pinned to the version embedded in h2o-pysparkling,
#      which H2O requires to match exactly).
#   3. docker build
#
# Staleness: the build is skipped only when the existing image's content
# label matches a hash of everything that shapes it (package sources and
# pyprojects, uv.lock, the runner Dockerfile/entrypoint, this script). A
# bare tag-existence check let week-old wheels pass as "the current build"
# and every in-container mbt assertion silently tested old code.
#
# Usage: build_image.sh [--force]
set -euo pipefail

REPO_ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/../../.." && pwd)"
RUNNER_DIR="$REPO_ROOT/examples/showcase/images/runner"
CACHE_DIR="${MBT_SHOWCASE_CACHE:-$HOME/.cache/mbt-showcase}/build"
IMAGE_TAG="${MBT_SHOWCASE_RUNNER_IMAGE:-mbt-showcase-runner:dev}"
CONTENT_LABEL="mbt.showcase.content"

# Pins for the sparkling fork. h2o-pysparkling-3-5 3.46.0.6.post1 embeds the
# H2O 3.46.0.6 backend; the h2o python client version must match it exactly.
PYSPARK_PIN="pyspark==3.5.8"
H2O_PIN="h2o==3.46.0.6"
PYSPARKLING_PIN="h2o-pysparkling-3-5==3.46.0.6.post1"

# Hash of everything that shapes the image. Package tests/READMEs stay out
# on purpose: they never reach the installed wheels, and hashing them would
# force pointless ~10-minute rebuilds.
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
runner = root / "examples" / "showcase" / "images" / "runner"
paths += [
    runner / "Dockerfile",
    runner / "entrypoint.sh",
    root / "examples" / "showcase" / "scripts" / "build_image.sh",
]
digest = hashlib.sha256()
for path in paths:
    digest.update(str(path.relative_to(root)).encode())
    digest.update(path.read_bytes())
print(digest.hexdigest()[:16])
EOF
)"

existing_hash="$(docker image inspect --format '{{json .Config.Labels}}' "$IMAGE_TAG" 2>/dev/null \
    | python3 -c "import json,sys; print((json.load(sys.stdin) or {}).get('$CONTENT_LABEL',''))" 2>/dev/null \
    || true)"
if [ "${1:-}" != "--force" ] && [ -n "$existing_hash" ] && [ "$existing_hash" = "$CONTENT_HASH" ]; then
    echo "image $IMAGE_TAG up to date (content $CONTENT_HASH; use --force to rebuild anyway)"
    exit 0
fi

mkdir -p "$CACHE_DIR"
rm -rf "$CACHE_DIR/dist"

echo "==> building workspace wheels"
(cd "$REPO_ROOT" && uv build --all-packages --out-dir "$CACHE_DIR/dist" >/dev/null)

echo "==> exporting constraints from uv.lock"
(cd "$REPO_ROOT" && uv export --frozen --no-emit-workspace --no-hashes --no-annotate --no-header \
    -o "$CACHE_DIR/constraints-full.txt" >/dev/null)
# Drop the dev-fork JVM pins (pyspark 4.x, h2o 3.46.0.11) and re-pin for the
# sparkling fork; drop torch-style local wheels that pip cannot resolve.
grep -vE '^(pyspark|h2o|h2o-pysparkling-3-5)==' "$CACHE_DIR/constraints-full.txt" \
    > "$CACHE_DIR/constraints.txt"
{
    echo "$PYSPARK_PIN"
    echo "$H2O_PIN"
    echo "$PYSPARKLING_PIN"
} >> "$CACHE_DIR/constraints.txt"

cp "$RUNNER_DIR/Dockerfile" "$RUNNER_DIR/entrypoint.sh" "$CACHE_DIR/"

echo "==> docker build $IMAGE_TAG (content $CONTENT_HASH)"
docker build -t "$IMAGE_TAG" --label "$CONTENT_LABEL=$CONTENT_HASH" "$CACHE_DIR"
echo "==> built $IMAGE_TAG"
