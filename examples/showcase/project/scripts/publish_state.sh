#!/usr/bin/env bash
# Publish the compiled manifest as the durable prod-state baseline: append a
# commit holding manifest.json to the state branch (created on first
# publish) and push it to the remote.
#
# Ported from the mbt init scaffold (DESIGN.md section 6) with two Woodpecker
# adaptations: the source sha comes from CI_COMMIT_SHA, and the remote is
# passed as a token-credentialed URL (Woodpecker's clone has no push
# credential), so the success message names the branch, never the remote.
#
# Git plumbing only: never touches the working tree, the index, or the
# current branch. Branch protection on main does not apply to the state
# branch, and its history is an audit trail of every published baseline.
#
# Usage: bash scripts/publish_state.sh [manifest=target/manifest.json] [remote=origin]
# Env:   MBT_STATE_BRANCH (default: mbt-state)

set -euo pipefail

manifest=${1:-target/manifest.json}
remote=${2:-origin}
branch=${MBT_STATE_BRANCH:-mbt-state}

if [ ! -f "$manifest" ]; then
  echo "publish_state: $manifest not found - run mbt build/compile first" >&2
  exit 1
fi

export GIT_AUTHOR_NAME=${GIT_AUTHOR_NAME:-mbt-state}
export GIT_AUTHOR_EMAIL=${GIT_AUTHOR_EMAIL:-mbt-state@showcase.local}
export GIT_COMMITTER_NAME=$GIT_AUTHOR_NAME
export GIT_COMMITTER_EMAIL=$GIT_AUTHOR_EMAIL

tmp=$(mktemp -d)
trap 'rm -rf "$tmp"' EXIT
export GIT_INDEX_FILE="$tmp/index"

blob=$(git hash-object -w "$manifest")
git update-index --add --cacheinfo 100644 "$blob" manifest.json
tree=$(git write-tree)
source_sha=${CI_COMMIT_SHA:-$(git rev-parse HEAD 2>/dev/null || echo unknown)}

# One retry: a concurrent publish may advance the branch between fetch and
# push; refetch the parent and re-commit rather than force-pushing history.
for attempt in 1 2; do
  parent=""
  if git fetch --quiet "$remote" "refs/heads/$branch" 2>/dev/null; then
    parent=$(git rev-parse FETCH_HEAD)
  fi
  commit=$(git commit-tree "$tree" ${parent:+-p "$parent"} -m "mbt state: prod manifest from $source_sha")
  if git push --quiet "$remote" "$commit:refs/heads/$branch" 2>/dev/null; then
    echo "publish_state: pushed $manifest to branch $branch ($commit)"
    exit 0
  fi
  [ "$attempt" = 1 ] && echo "publish_state: push raced with another publish, retrying" >&2
done

echo "publish_state: could not push to branch $branch after retry" >&2
exit 1
