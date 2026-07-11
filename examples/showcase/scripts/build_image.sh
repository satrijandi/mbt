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
# Usage: build_image.sh [--force]
set -euo pipefail

REPO_ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/../../.." && pwd)"
RUNNER_DIR="$REPO_ROOT/examples/showcase/images/runner"
CACHE_DIR="${MBT_SHOWCASE_CACHE:-$HOME/.cache/mbt-showcase}/build"
IMAGE_TAG="${MBT_SHOWCASE_RUNNER_IMAGE:-mbt-showcase-runner:dev}"

# Pins for the sparkling fork. h2o-pysparkling-3-5 3.46.0.6.post1 embeds the
# H2O 3.46.0.6 backend; the h2o python client version must match it exactly.
PYSPARK_PIN="pyspark==3.5.8"
H2O_PIN="h2o==3.46.0.6"
PYSPARKLING_PIN="h2o-pysparkling-3-5==3.46.0.6.post1"

if [ "${1:-}" != "--force" ] && docker image inspect "$IMAGE_TAG" >/dev/null 2>&1; then
    echo "image $IMAGE_TAG already exists (use --force to rebuild)"
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

echo "==> docker build $IMAGE_TAG"
docker build -t "$IMAGE_TAG" "$CACHE_DIR"
echo "==> built $IMAGE_TAG"
