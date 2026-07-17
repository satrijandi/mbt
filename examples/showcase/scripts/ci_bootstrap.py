"""Headless bootstrap for the showcase CI tier (Gitea + Woodpecker v3).

Runs entirely on the HOST (the dev venv has `requests` via mlflow) against
the published compose ports. That includes the OAuth login dance that mints
the first Woodpecker API token: woodpecker-server's split-horizon config
(WOODPECKER_HOST and WOODPECKER_EXPERT_FORGE_OAUTH_HOST are localhost URLs,
see docker-compose.yml) means the dance drives the EXACT flow a human
browser performs - so bootstrapping doubles as proof that the documented
"log in via Gitea OAuth" path works.

Three phases, because Woodpecker needs OAuth app credentials at boot:

  pre   - create the Gitea admin (mbtops), an API token, the mbt-showcase
          org and churn repo, push the project source, and create the OAuth2
          app with the browser-facing redirect URI. Prints JSON
          {gitea_token, client_id, client_secret}; the caller re-ups
          woodpecker-server(+agent) with those credentials.
  post  - OAuth dance as mbtops (Woodpecker v3 signs the `state` as a JWT,
          so the flow must START at Woodpecker's /authorize; the first login
          renders Gitea's grant page), then mint the API token, then
          activate the repo (creates the Gitea webhook) and provision the
          gitea_token secret for push + pull_request events. Prints JSON
          {woodpecker_token, repo_id}.
  login - the dance alone, for any user: prints {woodpecker_token}. The
          test tier uses it to pin the human login path per persona.

Usage:
  ci_bootstrap.py pre   --gitea-url URL --gitea-container NAME \
      --woodpecker-url URL --project-dir DIR
  ci_bootstrap.py post  --gitea-url URL --woodpecker-url URL --gitea-token TOKEN
  ci_bootstrap.py login --gitea-url URL --woodpecker-url URL \
      --user NAME --password PASS
"""

import argparse
import json
import re
import shutil
import subprocess
import sys
import tempfile
import time
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


def _hidden_fields(html: str) -> dict:
    """Hidden inputs of the page's form, posted back exactly like a browser
    submit. Gitea 1.27 dropped the _csrf field from the login and grant
    forms (the session cookie carries the protection); older versions still
    render it - either way, the form itself says what to send."""
    fields = {}
    for tag in re.findall(r'<input[^>]*type="hidden"[^>]*>', html):
        name = re.search(r'name="([^"]+)"', tag)
        value = re.search(r'value="([^"]*)"', tag)
        if name:
            fields[name.group(1)] = value.group(1) if value else ""
    return fields


def _wait_http_ok(url: str, timeout_s: int = 60) -> None:
    deadline = time.time() + timeout_s
    while True:
        try:
            if requests.get(url, timeout=5).ok:
                return
        except requests.RequestException:
            pass
        if time.time() > deadline:
            raise SystemExit(f"{url} not answering within {timeout_s}s")
        time.sleep(1)


def oauth_login(gitea_url: str, woodpecker_url: str, username: str, password: str) -> str:
    """Log into Woodpecker through Gitea OAuth and mint an API token.

    Performs the browser flow verbatim against the host-published ports:
    Gitea form login, Woodpecker /authorize (it builds the authorize URL
    and signs `state` as a JWT), Gitea's grant page on first consent, the
    code callback, then the CSRF-guarded token mint. The grant POST sends
    back exactly the hidden fields of the rendered grant form (client_id,
    state, redirect_uri, ...), which Gitea pins to the authorize request.
    """
    _wait_http_ok(f"{woodpecker_url}/web-config.js")
    sess = requests.Session()

    login_page = sess.get(f"{gitea_url}/user/login", timeout=30)
    login_page.raise_for_status()
    login = sess.post(
        f"{gitea_url}/user/login",
        data={
            **_hidden_fields(login_page.text),
            "user_name": username,
            "password": password,
        },
        timeout=30,
    )
    login.raise_for_status()
    # A failed login re-renders the form with 200; a success redirects away.
    if login.url.rstrip("/").endswith("/user/login"):
        raise SystemExit(f"gitea login failed for {username}")

    start = sess.get(f"{woodpecker_url}/authorize", allow_redirects=False, timeout=30)
    auth_url = start.headers.get("Location", "")
    if not auth_url:
        raise SystemExit(f"woodpecker /authorize did not redirect: {start.status_code}")

    consent = sess.get(auth_url, allow_redirects=False, timeout=30)
    if consent.is_redirect:
        # Re-authorization: Gitea auto-grants and redirects immediately.
        callback = consent.headers["Location"]
    else:
        # First consent for this user+app: Gitea renders the grant page.
        fields = _hidden_fields(consent.text)
        if "client_id" not in fields:
            raise SystemExit(f"expected the grant page, got:\n{consent.text[:1000]}")
        grant = sess.post(
            f"{gitea_url}/login/oauth/grant",
            data={**fields, "granted": "true"},
            allow_redirects=False,
            timeout=30,
        )
        if not grant.is_redirect:
            raise SystemExit(
                f"gitea grant did not redirect: {grant.status_code} {grant.text[:500]}"
            )
        callback = grant.headers["Location"]

    # Woodpecker exchanges the code with Gitea server-side and starts the
    # session; requests follows the final redirect into the logged-in UI.
    done = sess.get(callback, timeout=60)
    done.raise_for_status()

    config = sess.get(f"{woodpecker_url}/web-config.js", timeout=30)
    csrf = re.search(r'WOODPECKER_CSRF = "([^"]+)"', config.text)
    if csrf is None:
        raise SystemExit(f"no WOODPECKER_CSRF in web-config.js:\n{config.text[:500]}")
    token = sess.post(
        f"{woodpecker_url}/api/user/token",
        headers={"X-CSRF-TOKEN": csrf.group(1)},
        timeout=30,
    )
    token.raise_for_status()
    return token.text.strip().strip('"')


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

    # The redirect URI must be the browser-facing Woodpecker URL: Gitea
    # validates it on BOTH the authorize request and the server-side token
    # exchange (woodpecker-server sends WOODPECKER_HOST + /authorize).
    app = gitea_api(
        args.gitea_url,
        "POST",
        "/user/applications/oauth2",
        auth=(USER, PASSWORD),
        payload={
            "name": f"woodpecker-{uuid.uuid4().hex[:8]}",
            "redirect_uris": [f"{args.woodpecker_url.rstrip('/')}/authorize"],
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
    woodpecker_token = oauth_login(args.gitea_url, args.woodpecker_url, USER, PASSWORD)
    if not woodpecker_token:
        raise SystemExit("OAuth dance produced no token")

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
    parser.add_argument("phase", choices=["pre", "post", "login"])
    parser.add_argument("--gitea-url", required=True)
    parser.add_argument("--gitea-container", help="pre only: gitea CLI runs via docker exec")
    parser.add_argument("--woodpecker-url", required=True, help="browser-facing Woodpecker URL")
    parser.add_argument("--project-dir")
    parser.add_argument(
        "--deploy-dir", default=str(Path(__file__).resolve().parent.parent / "deploy")
    )
    parser.add_argument("--network", help="compose network name for images.env")
    parser.add_argument("--workspace", help="host workspace path for images.env")
    parser.add_argument(
        "--zot-ref", help="daemon-perspective zot repo, e.g. localhost:15000/mbt/churn"
    )
    parser.add_argument("--gitea-token")
    parser.add_argument("--user", help="login only: the persona to log in as")
    parser.add_argument("--password", help="login only: the persona's password")
    args = parser.parse_args()

    if args.phase == "pre":
        if not (args.project_dir and args.gitea_container):
            parser.error("pre needs --project-dir and --gitea-container")
        result = phase_pre(args)
    elif args.phase == "post":
        if not args.gitea_token:
            parser.error("post needs --gitea-token")
        result = phase_post(args)
    else:
        if not (args.user and args.password):
            parser.error("login needs --user and --password")
        token = oauth_login(args.gitea_url, args.woodpecker_url, args.user, args.password)
        result = {"woodpecker_token": token}
    print(json.dumps(result))
    return 0


if __name__ == "__main__":
    sys.exit(main())
