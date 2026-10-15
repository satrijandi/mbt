"""Hermetic guards for what drives the showcase: its Makefile and
scripts/lifecycle.py (`make lifecycle`).

The live tiers drive both against a real stack, but only nightly and only in
the orders they script; these need neither docker nor the stack. A stub
`docker` on PATH stands in for the daemon, and a local HTTP server for MLflow.
"""

import importlib.util
import json
import os
import subprocess
import sys
import threading
from http.server import BaseHTTPRequestHandler, HTTPServer
from pathlib import Path

import pytest

REPO_ROOT = Path(__file__).resolve().parent.parent
SHOWCASE = REPO_ROOT / "examples" / "showcase"
PINNED = "localhost:15000/mbt/churn@sha256:" + "a" * 64


@pytest.fixture(scope="module")
def lifecycle():
    path = SHOWCASE / "scripts" / "lifecycle.py"
    spec = importlib.util.spec_from_file_location("showcase_lifecycle_driver", path)
    assert spec is not None and spec.loader is not None
    module = importlib.util.module_from_spec(spec)
    sys.modules[spec.name] = module
    spec.loader.exec_module(module)
    return module


def _stub_docker(bin_dir: Path, script: str) -> None:
    bin_dir.mkdir(exist_ok=True)
    stub = bin_dir / "docker"
    stub.write_text("#!/bin/sh\n" + script)
    stub.chmod(0o755)


def _images_env(image: str) -> str:
    return f"# the deploy repo's pin\nIMAGE={image}\nNETWORK=mbt-showcase_default\n"


def test_compose_up_loads_the_oauth_client_make_ci_wrote_mid_run(tmp_path: Path) -> None:
    """On a fresh stack `make lifecycle` runs `make ci` as a sub-make, which
    writes woodpecker.env after the outer make has read its makefiles. The
    compose up after it must still see that client: with the placeholder
    instead, compose recreated Woodpecker unable to log anyone in, and the
    lifecycle stopped at step 1 with "expected the grant page"."""
    workspace = tmp_path / "workspace"
    workspace.mkdir()
    _stub_docker(
        tmp_path / "bin",
        'echo "client=$SHOWCASE_WOODPECKER_CLIENT secret=$SHOWCASE_WOODPECKER_SECRET"\n',
    )
    # A recipe writing the file stands in for the `make ci` sub-make. An
    # including makefile, not --eval: macOS still ships make 3.81.
    probe = tmp_path / "probe.mk"
    probe.write_text(
        f"include {SHOWCASE / 'Makefile'}\n"
        "probe:\n"
        "\tprintf 'SHOWCASE_WOODPECKER_CLIENT=%s\\nSHOWCASE_WOODPECKER_SECRET=%s\\n'"
        " from-make-ci its-secret > $(WOODPECKER_ENV)\n"
        "\t$(call compose_up,core ci orch)\n"
    )
    env = {
        **os.environ,
        "PATH": f"{tmp_path / 'bin'}{os.pathsep}{os.environ['PATH']}",
        "SHOWCASE_WORKSPACE": str(workspace),
        "DOCKER_SOCK_GID": "0",  # skip the socket probe, which needs docker
    }
    env.pop("SHOWCASE_WOODPECKER_CLIENT", None)
    env.pop("SHOWCASE_WOODPECKER_SECRET", None)
    proc = subprocess.run(
        ["make", "--no-print-directory", "-f", str(probe), "probe"],
        env=env,
        capture_output=True,
        text=True,
        timeout=60,
        check=False,
    )
    assert proc.returncode == 0, proc.stderr
    assert "client=from-make-ci secret=its-secret" in proc.stdout, proc.stdout


def test_wait_served_returns_once_git_sync_hands_airflow_the_pin(
    lifecycle, tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    """Gitea has the pin before git-sync copies it to Airflow. The unit
    phase waits on Airflow's copy, which a DAG task reads, so the retrain is
    never triggered while that copy still pins nothing."""
    calls = tmp_path / "calls"
    unpinned, pinned = tmp_path / "unpinned.env", tmp_path / "pinned.env"
    unpinned.write_text(_images_env(""))
    pinned.write_text(_images_env(PINNED))
    # Airflow's copy is a commit behind for the first two reads.
    _stub_docker(
        tmp_path / "bin",
        f'echo "$@" >> {calls}\n'
        f'if [ "$(wc -l < {calls})" -le 2 ]; then cat {unpinned}; else cat {pinned}; fi\n',
    )
    monkeypatch.setenv("PATH", f"{tmp_path / 'bin'}{os.pathsep}{os.environ['PATH']}")
    monkeypatch.setattr(lifecycle.time, "sleep", lambda _s: None)

    lifecycle.wait_served("mbt-showcase", PINNED)

    reads = calls.read_text().splitlines()
    assert len(reads) == 3, reads
    assert reads[0] == "exec mbt-showcase-airflow-scheduler-1 cat /gitsync/current/images.env"


def test_wait_served_names_git_sync_when_it_never_syncs(
    lifecycle, tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    _stub_docker(tmp_path / "bin", "exit 1\n")  # no current link: git-sync never synced
    monkeypatch.setenv("PATH", f"{tmp_path / 'bin'}{os.pathsep}{os.environ['PATH']}")

    with pytest.raises(SystemExit) as refused:
        lifecycle.wait_served("mbt-showcase", PINNED, timeout_s=0)

    message = str(refused.value)
    assert f"Airflow still runs no unit 0s after the deploy repo pinned {PINNED}" in message
    assert "docker logs mbt-showcase-git-sync-1" in message


def test_image_pin_wants_a_digest(lifecycle) -> None:
    assert lifecycle.image_pin(_images_env(PINNED)) == PINNED
    assert lifecycle.image_pin(_images_env("")) == ""
    assert lifecycle.image_pin(_images_env("localhost:15000/mbt/churn:latest")) == ""


class _Registry(BaseHTTPRequestHandler):
    """MLflow's registered-models/get for churn_automl, holding only staging."""

    def do_GET(self) -> None:
        if self.path.startswith("/api/2.0/mlflow/registered-models/get?name=churn_automl"):
            body = {
                "registered_model": {
                    "name": "churn_automl",
                    "aliases": [{"alias": "staging", "version": "2"}],
                }
            }
            status = 200
        else:
            body = {"error_code": "RESOURCE_DOES_NOT_EXIST"}
            status = 404
        payload = json.dumps(body).encode()
        self.send_response(status)
        self.send_header("Content-Type", "application/json")
        self.send_header("Content-Length", str(len(payload)))
        self.end_headers()
        self.wfile.write(payload)

    def log_message(self, *_args: object) -> None:
        pass


@pytest.fixture
def mlflow_url():
    server = HTTPServer(("127.0.0.1", 0), _Registry)
    thread = threading.Thread(target=server.serve_forever, daemon=True)
    thread.start()
    try:
        yield f"http://127.0.0.1:{server.server_port}"
    finally:
        server.shutdown()
        server.server_close()


def test_alias_version_reads_an_alias_nothing_holds_as_none(lifecycle, mlflow_url) -> None:
    """A stack with no production champion yet (`make lifecycle` without
    `make demo` first): the promote phase asks who holds production and must
    hear "nobody", where looking the alias up directly got MLflow's 400."""
    assert lifecycle.alias_version(mlflow_url, "staging") == "2"
    assert lifecycle.alias_version(mlflow_url, "production") is None


def test_alias_version_of_an_unregistered_model_is_none(
    lifecycle, mlflow_url, monkeypatch: pytest.MonkeyPatch
) -> None:
    monkeypatch.setattr(lifecycle, "MODEL", "never_registered")
    assert lifecycle.alias_version(mlflow_url, "production") is None
