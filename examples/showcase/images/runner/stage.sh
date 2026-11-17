#!/usr/bin/env bash
# Stage the runner image's build context into DIR; scripts/build_images.sh
# runs this, then `docker build DIR`. Everything that shapes the runner lives
# in this directory, so its content hash (build_images.sh --print-hash)
# covers it without covering the rest of the build script.
#   1. uv build --all-packages          -> workspace wheels
#   2. uv export --frozen               -> third-party pins from uv.lock,
#      with the dev fork's pyspark/h2o lines swapped for the sparkling fork's
#      (pyspark 3.5.x; h2o pinned to the version embedded in h2o-pysparkling,
#      which H2O requires to match exactly).
#   3. the Dockerfile, entrypoint and the committed image-extras.txt
#
# Usage: stage.sh DIR
set -euo pipefail

RUNNER_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
REPO_ROOT="$(cd "$RUNNER_DIR/../../../.." && pwd)"
STAGING="${1:?usage: stage.sh DIR}"

# Pins for the sparkling fork. h2o-pysparkling-3-5 3.46.0.6.post1 embeds the
# H2O 3.46.0.6 backend; the h2o python client version must match it exactly.
PYSPARK_PIN="pyspark==3.5.8"
H2O_PIN="h2o==3.46.0.6"
PYSPARKLING_PIN="h2o-pysparkling-3-5==3.46.0.6.post1"

mkdir -p "$STAGING"
rm -rf "$STAGING/dist"

echo "==> building workspace wheels"
(cd "$REPO_ROOT" && uv build --all-packages --out-dir "$STAGING/dist" >/dev/null)

echo "==> exporting constraints from uv.lock"
(cd "$REPO_ROOT" && uv export --frozen --no-emit-workspace --no-hashes --no-annotate --no-header \
    -o "$STAGING/constraints-full.txt" >/dev/null)
# Drop the dev-fork JVM pins (pyspark 4.x, h2o 3.46.0.11) and re-pin for the
# sparkling fork.
grep -vE '^(pyspark|h2o|h2o-pysparkling-3-5)==' "$STAGING/constraints-full.txt" \
    > "$STAGING/constraints.txt"
{
    echo "$PYSPARK_PIN"
    echo "$H2O_PIN"
    echo "$PYSPARKLING_PIN"
} >> "$STAGING/constraints.txt"

# image-extras.txt is committed, not derived here: it pins the closure of the
# non-mbt image deps (jupyterlab, setuptools), which uv.lock cannot cover. Regenerate it
# with scripts/lock_image_extras.sh, never by rebuilding.
cp "$RUNNER_DIR/Dockerfile" "$RUNNER_DIR/entrypoint.sh" "$RUNNER_DIR/image-extras.txt" "$STAGING/"
