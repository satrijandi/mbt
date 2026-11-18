#!/usr/bin/env bash
# Stage the Airflow image's build context into DIR; scripts/build_airflow_image.sh
# runs this, then `docker build DIR`. Everything that shapes the image lives in this
# directory, so its content hash (build_airflow_image.sh --print-hash) covers it.
#   1. uv build                 -> the wheels the DAGs import: mbt-airflow, and the
#      mbt-core / mbt-adapter-base it draws the graph with (ADR-33). The tasks run
#      mbt in the deployable unit, not here, so nothing heavier is needed.
#   2. uv export --frozen       -> third-party pins from uv.lock. The Dockerfile
#      drops the ones Airflow's own constraints already pin, because those win.
#   3. the Dockerfile and Airflow's committed constraints file
#
# Usage: stage.sh DIR
set -euo pipefail

AIRFLOW_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
REPO_ROOT="$(cd "$AIRFLOW_DIR/../../../.." && pwd)"
STAGING="${1:?usage: stage.sh DIR}"

mkdir -p "$STAGING"
rm -rf "$STAGING/dist"

echo "==> building the DAG-side wheels"
# uv build takes one --package per call.
for package in mbt-airflow mbt-core mbt-adapter-base; do
    (cd "$REPO_ROOT" && uv build --package "$package" --out-dir "$STAGING/dist" >/dev/null)
done

echo "==> exporting constraints from uv.lock"
(cd "$REPO_ROOT" && uv export --frozen --no-emit-workspace --no-hashes --no-annotate --no-header \
    -o "$STAGING/constraints.txt" >/dev/null)

cp "$AIRFLOW_DIR/Dockerfile" "$AIRFLOW_DIR/airflow-constraints.txt" "$STAGING/"
