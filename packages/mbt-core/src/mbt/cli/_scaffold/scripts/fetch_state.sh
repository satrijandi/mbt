#!/usr/bin/env bash
# Fetch the latest published prod-state baseline (manifest.json on the
# state branch) into a local path for `--state`. Exits 0 with the file
# written, or 3 when no baseline was ever published (bootstrap), so
# workflows can fall back to a full build.
#
# Usage: bash scripts/fetch_state.sh [out=state/prod/latest.json] [remote=origin]
# Env:   MBT_STATE_BRANCH (default: mbt-state)

set -euo pipefail

out=${1:-state/prod/latest.json}
remote=${2:-origin}
branch=${MBT_STATE_BRANCH:-mbt-state}

if ! git fetch --quiet "$remote" "refs/heads/$branch" 2>/dev/null; then
  echo "fetch_state: no $branch branch on $remote yet (bootstrap - no baseline)" >&2
  exit 3
fi

mkdir -p "$(dirname "$out")"
git show FETCH_HEAD:manifest.json > "$out"
echo "fetch_state: wrote $remote/$branch manifest to $out"
