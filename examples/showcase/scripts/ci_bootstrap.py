"""Headless bootstrap for the showcase CI tier (Gitea + Woodpecker v3).

Runs on the HOST (the dev venv has `requests` via mlflow) against the
published compose ports; the one flow that must happen in-network - the
OAuth login dance that mints the first Woodpecker API token - executes
inside the gitea container via `docker exec` (the rootless image ships bash
and curl), where gitea:3000 and woodpecker-server:8000 resolve natively.

Two phases, because Woodpecker needs OAuth app credentials at boot:

  pre   - create the Gitea admin (mbtops), an API token, the mbt-showcase
          org and churn repo, push the project source, and create the OAuth2
          app. Prints JSON {gitea_token, client_id, client_secret}; the
          caller re-ups woodpecker-server(+agent) with those credentials.
  post  - scripted OAuth dance (Woodpecker v3 signs the `state` as a JWT, so
          the flow must START at Woodpecker's /authorize; the first login
          always renders Gitea's grant page), then mint the API token, then
          activate the repo (creates the Gitea webhook) and provision the
          gitea_token secret for push + pull_request events. Prints JSON
          {woodpecker_token, repo_id}.

Usage:
  ci_bootstrap.py pre  --gitea-url URL --gitea-container NAME --project-dir DIR
  ci_bootstrap.py post --gitea-url URL --gitea-container NAME \
      --woodpecker-url URL --gitea-token TOKEN
"""

import argparse
import json
import shutil
import subprocess
import sys
import tempfile
import uuid
from pathlib import Path

import requests

ORG = "mbt-showcase"
REPO = "churn"
DEPLOY_REPO = "deploy"
USER = "mbtops"
PASSWORD = "mbtops-showcase-password"
# Second persona for the SHOW-10 authz tests: repo write access, NOT an
# owner - branch protection must stop their direct pushes to main.
DS_USER = "mbtds"
DS_PASSWORD = "mbtds-showcase-password"
REDIRECT_URI = "http://woodpecker-server:8000/authorize"

# The OAuth dance, run inside the gitea container (bash + curl + in-network
# DNS). Woodpecker v3 validates `state` as a signed JWT, so step 2 asks
# Woodpecker to build the authorize URL; the grant POST is session-pinned to
# that exact authorize request (client_id/state/redirect_uri must match).
DANCE = r"""
set -eu
# gitea:3000, not localhost: the authorize URL Woodpecker builds points at
# WOODPECKER_GITEA_URL's host, and curl's cookie jar is host-scoped - a
# localhost login session would never accompany the authorize request.
GITEA=http://gitea:3000
WP=http://woodpecker-server:8000
JAR=$(mktemp)
PAGE=$(mktemp)

CSRF=$(curl -sf -c "$JAR" "$GITEA/user/login" \
  | grep -m1 'name="_csrf"' | sed 's/.*value="\([^"]*\)".*/\1/')
curl -sf -b "$JAR" -c "$JAR" "$GITEA/user/login" \
  --data-urlencode "_csrf=$CSRF" \
  --data-urlencode "user_name=$BOOTSTRAP_USER" \
  --data-urlencode "password=$BOOTSTRAP_PASS" >/dev/null

AUTH_URL=$(curl -sf -o /dev/null -w '%{redirect_url}' -c "$JAR" -b "$JAR" "$WP/authorize")
STATE=$(printf '%s' "$AUTH_URL" | sed 's/.*[?&]state=\([^&]*\).*/\1/')
CLIENT_ID=$(printf '%s' "$AUTH_URL" | sed 's/.*[?&]client_id=\([^&]*\).*/\1/')

REDIR=$(curl -sf -b "$JAR" -c "$JAR" -o "$PAGE" -w '%{redirect_url}' "$AUTH_URL")
if [ -z "$REDIR" ]; then
  GCSRF=$(grep -m1 'name="_csrf"' "$PAGE" | sed 's/.*value="\([^"]*\)".*/\1/')
  REDIR=$(curl -sf -b "$JAR" -c "$JAR" -o /dev/null -w '%{redirect_url}' \
    "$GITEA/login/oauth/grant" \
    --data-urlencode "_csrf=$GCSRF" \
    --data-urlencode "client_id=$CLIENT_ID" \
    --data-urlencode "redirect_uri=$WP/authorize" \
    --data-urlencode "state=$STATE" \
    --data "scope=" --data "nonce=" --data "granted=true")
fi

curl -sfL -b "$JAR" -c "$JAR" "$REDIR" >/dev/null

WPCSRF=$(curl -sf -b "$JAR" "$WP/web-config.js" \
  | grep 'WOODPECKER_CSRF' | sed 's/.*WOODPECKER_CSRF = "\(.*\)".*/\1/')
curl -sf -X POST -b "$JAR" -H "X-CSRF-TOKEN: $WPCSRF" "$WP/api/user/token"
"""


def sh(*cmd: str, check: bool = True, **kwargs: object) -> subprocess.CompletedProcess:
    proc = subprocess.run(cmd, capture_output=True, text=True, timeout=300, check=False, **kwargs)
    if check and proc.returncode != 0:
        raise SystemExit(f"command failed: {cmd}\n{proc.stdout}\n{proc.stderr}")
    return proc


def gitea_api(
    base: str,
    method: str,
    path: str,
    token: str | None = None,
    auth: tuple | None = None,
    payload: dict | None = None,
    ok_statuses: tuple = (),
) -> dict | None:
    headers = {"Authorization": f"token {token}"} if token else {}
    resp = requests.request(
        method, f"{base}/api/v1{path}", json=payload, headers=headers, auth=auth, timeout=30
    )
    if not resp.ok and resp.status_code not in ok_statuses:
        raise SystemExit(f"{method} {path} -> {resp.status_code}: {resp.text[:1000]}")
    return resp.json() if resp.ok and resp.content else None


def _create_user(args: argparse.Namespace, username: str, password: str, admin: bool) -> None:
    create = sh(
        "docker",
        "exec",
        args.gitea_container,
        "gitea",
        "admin",
        "user",
        "create",
        "--username",
        username,
        "--password",
        password,
        "--email",
        f"{username}@example.com",
        *(["--admin"] if admin else []),
        "--must-change-password=false",
        check=False,
    )
    if create.returncode != 0 and "already exists" not in (create.stdout + create.stderr):
        raise SystemExit(
            f"gitea user create ({username}) failed:\n{create.stdout}\n{create.stderr}"
        )


def _user_token(args: argparse.Namespace, username: str) -> str:
    return (
        sh(
            "docker",
            "exec",
            args.gitea_container,
            "gitea",
            "admin",
            "user",
            "generate-access-token",
            "--username",
            username,
            "--token-name",
            f"bootstrap-{uuid.uuid4().hex[:8]}",
            "--scopes",
            "all",
            "--raw",
        )
        .stdout.strip()
        .splitlines()[-1]
    )


def phase_pre(args: argparse.Namespace) -> dict:
    _create_user(args, USER, PASSWORD, admin=True)
    _create_user(args, DS_USER, DS_PASSWORD, admin=False)
    token = _user_token(args, USER)
    ds_token = _user_token(args, DS_USER)

    gitea_api(
        args.gitea_url, "POST", "/orgs", token=token, payload={"username": ORG}, ok_statuses=(422,)
    )
    for repo_name in (REPO, DEPLOY_REPO):
        gitea_api(
            args.gitea_url,
            "POST",
            f"/orgs/{ORG}/repos",
            token=token,
            payload={"name": repo_name, "private": False, "auto_init": False},
            ok_statuses=(409,),
        )
    # The DS persona can push branches and open PRs, but is not an owner.
    gitea_api(
        args.gitea_url,
        "PUT",
        f"/repos/{ORG}/{REPO}/collaborators/{DS_USER}",
        token=token,
        payload={"permission": "write"},
    )

    _seed_repo(args, token, Path(args.project_dir), REPO, "seed churn project")
    _seed_repo(
        args,
        token,
        Path(args.deploy_dir),
        DEPLOY_REPO,
        "seed deploy repo",
        render_images_env={
            "NETWORK": args.network or "mbt-showcase_default",
            "WORKSPACE": args.workspace or "",
        },
    )

    app = gitea_api(
        args.gitea_url,
        "POST",
        "/user/applications/oauth2",
        auth=(USER, PASSWORD),
        payload={
            "name": f"woodpecker-{uuid.uuid4().hex[:8]}",
            "redirect_uris": [REDIRECT_URI],
            "confidential_client": True,
        },
    )
    assert app is not None
    return {
        "gitea_token": token,
        "ds_token": ds_token,
        "client_id": app["client_id"],
        "client_secret": app["client_secret"],
    }


def _seed_repo(
    args: argparse.Namespace,
    token: str,
    source: Path,
    repo_name: str,
    message: str,
    render_images_env: dict | None = None,
) -> None:
    # Idempotent reruns (`make ci` twice): a repo that already has branches
    # was seeded before; a second push would be rejected as non-fast-forward.
    branches = gitea_api(args.gitea_url, "GET", f"/repos/{ORG}/{repo_name}/branches", token=token)
    if branches:
        print(f"repo {ORG}/{repo_name} already seeded; skipping push", file=sys.stderr)
        return

    source = source.resolve()
    host = args.gitea_url.split("://", 1)[1]
    push_url = f"http://{USER}:{token}@{host}/{ORG}/{repo_name}.git"
    staging = Path(tempfile.mkdtemp(prefix="mbt-showcase-seed-"))
    try:
        work = staging / "repo"
        shutil.copytree(
            source, work, ignore=shutil.ignore_patterns("target", ".git", "__pycache__")
        )
        if render_images_env:
            conf_path = work / "images.env"
            lines = []
            for raw in conf_path.read_text().splitlines():
                key = raw.split("=", 1)[0]
                line = raw
                if "=" in raw and not raw.startswith("#") and key in render_images_env:
                    line = f"{key}={render_images_env[key]}"
                lines.append(line)
            conf_path.write_text("\n".join(lines) + "\n")
        git = [
            "git",
            "-c",
            "user.name=mbt-showcase-bootstrap",
            "-c",
            "user.email=bootstrap@showcase.local",
        ]
        sh(*git, "init", "-b", "main", cwd=work)
        sh(*git, "add", "-A", cwd=work)
        sh(*git, "commit", "-m", f"mbt showcase: {message}", cwd=work)
        sh(*git, "push", push_url, "main", cwd=work)
    finally:
        shutil.rmtree(staging, ignore_errors=True)


def phase_post(args: argparse.Namespace) -> dict:
    dance = sh(
        "docker",
        "exec",
        "-i",
        "-e",
        f"BOOTSTRAP_USER={USER}",
        "-e",
        f"BOOTSTRAP_PASS={PASSWORD}",
        args.gitea_container,
        "bash",
        "-s",
        input=DANCE,
    )
    woodpecker_token = dance.stdout.strip().splitlines()[-1]
    if not woodpecker_token:
        raise SystemExit(f"OAuth dance produced no token:\n{dance.stdout}\n{dance.stderr}")

    forge_repo = gitea_api(args.gitea_url, "GET", f"/repos/{ORG}/{REPO}", token=args.gitea_token)
    assert forge_repo is not None
    wp_headers = {"Authorization": f"Bearer {woodpecker_token}"}

    activate = requests.post(
        f"{args.woodpecker_url}/api/repos",
        params={"forge_remote_id": str(forge_repo["id"])},
        headers=wp_headers,
        timeout=60,
    )
    if activate.ok:
        repo_id = activate.json()["id"]
    else:
        lookup = requests.get(
            f"{args.woodpecker_url}/api/repos/lookup/{ORG}/{REPO}", headers=wp_headers, timeout=30
        )
        if not lookup.ok:
            raise SystemExit(
                f"repo activation failed ({activate.status_code}: {activate.text[:500]}) "
                f"and lookup failed ({lookup.status_code})"
            )
        repo_id = lookup.json()["id"]

    # The bake step mounts the docker socket, which needs the repo's
    # trusted.volumes flag - settable by instance admins only (mbtops is in
    # WOODPECKER_ADMIN).
    trust = requests.patch(
        f"{args.woodpecker_url}/api/repos/{repo_id}",
        json={"trusted": {"volumes": True, "network": False, "security": False}},
        headers=wp_headers,
        timeout=30,
    )
    if not trust.ok:
        raise SystemExit(f"trusted patch failed: {trust.status_code} {trust.text[:500]}")

    secrets = {
        "gitea_token": args.gitea_token,
        # Daemon-perspective push repo for the deployable unit (docker
        # treats localhost registries as insecure by default, DESIGN.md
        # open question 1).
        "zot_push_repo": args.zot_ref or "localhost:15000/mbt/churn",
    }
    for name, value in secrets.items():
        secret = requests.post(
            f"{args.woodpecker_url}/api/repos/{repo_id}/secrets",
            json={"name": name, "value": value, "events": ["push", "pull_request"], "images": []},
            headers=wp_headers,
            timeout=30,
        )
        if not secret.ok and secret.status_code != 409:
            raise SystemExit(f"secret {name} failed: {secret.status_code} {secret.text[:500]}")

    return {"woodpecker_token": woodpecker_token, "repo_id": repo_id}


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("phase", choices=["pre", "post"])
    parser.add_argument("--gitea-url", required=True)
    parser.add_argument("--gitea-container", required=True)
    parser.add_argument("--project-dir")
    parser.add_argument(
        "--deploy-dir", default=str(Path(__file__).resolve().parent.parent / "deploy")
    )
    parser.add_argument("--network", help="compose network name for images.env")
    parser.add_argument("--workspace", help="host workspace path for images.env")
    parser.add_argument(
        "--zot-ref", help="daemon-perspective zot repo, e.g. localhost:15000/mbt/churn"
    )
    parser.add_argument("--woodpecker-url")
    parser.add_argument("--gitea-token")
    args = parser.parse_args()

    if args.phase == "pre":
        if not args.project_dir:
            parser.error("pre needs --project-dir")
        result = phase_pre(args)
    else:
        if not (args.woodpecker_url and args.gitea_token):
            parser.error("post needs --woodpecker-url and --gitea-token")
        result = phase_post(args)
    print(json.dumps(result))
    return 0


if __name__ == "__main__":
    sys.exit(main())
