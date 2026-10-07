#!/usr/bin/env bash
# Point `origin` at the forge with the CI token, so later git commands in this
# step can read and write a private repo: fetch_state.sh reads the mbt-state
# branch, publish_state.sh pushes it. Woodpecker's clone step authenticates
# with credentials that do not outlive it, and without this a private repo's
# state fetch fails - which fetch_state.sh would report as "no baseline yet"
# and the build would retrain everything.
#
# The address is CI_FORGE_URL/CI_REPO, the forge as the Woodpecker server
# reaches it, so it works wherever the clone did. git never prints a remote
# URL on fetch or push, and Woodpecker masks secret values in step logs.
#
# Env: GITEA_TOKEN (the gitea_token secret), CI_FORGE_URL, CI_REPO.

set -euo pipefail

: "${GITEA_TOKEN:?the gitea_token secret is not wired into this step}"
forge="${CI_FORGE_URL%/}"
git remote set-url origin "${forge%%://*}://ci:${GITEA_TOKEN}@${forge#*://}/${CI_REPO}.git"
echo "ci_git_remote: origin -> ${forge}/${CI_REPO}.git (token auth)"
