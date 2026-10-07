"""Preflight for the showcase stack: catch the known failure classes up front.

Every check here is a failure that has cost a real debugging session:

- docker unreachable, or too little memory for Spark + H2O + the services
  (the stack's configured ceilings add up past 3GB in use; leave ~10GB);
- the docker VM's disk nearly full: on 2026-10-02 it PANICked airflow-db
  ("No space left on device") and corrupted its WAL, and earlier it made
  SeaweedFS fail every model upload;
- the docker-socket group the Airflow scheduler must join, which differs per
  runtime (Rancher Desktop is not group 0) - scripts/docker_sock_gid.sh;
- a host port already taken by something that is not this stack;
- a leftover network whose endpoint outlived its container, after which even
  `make down && make up` fails with "endpoint ... already exists in network".

Hard failures exit 1 with the fix; soft ones (memory, low-but-usable disk)
warn and exit 0. `--fix` removes a leftover network itself.

Usage: python3 scripts/doctor.py [--fix]   (stdlib only; `make doctor`)
"""

from __future__ import annotations

import argparse
import json
import os
import socket
import subprocess
import sys
from pathlib import Path

SHOWCASE_DIR = Path(__file__).resolve().parent.parent
PROJECT = os.environ.get("SHOWCASE_PROJECT", "mbt-showcase")
NETWORK = os.environ.get("SHOWCASE_NETWORK", f"{PROJECT}_default")
#: The probe image: one the stack already pulls (airflow-db's, the same one
#: docker_sock_gid.sh uses), so the disk check costs no extra download.
PROBE_IMAGE = "postgres:18.6-alpine"
GIB = 1024**3
MEMORY_WANTED = 10 * GIB
#: A runner image build, measured on OrbStack: the new image (~6GB unpacked)
#: beside the old one, ~10GB of layer cache, and the pip cache mount - a build
#: started with 27GB free left 2GB, and the next JVM died with SIGBUS mapping
#: its perf file onto the full disk.
DISK_WARN_BUILD = 25 * GIB
DISK_WARN = 8 * GIB  # image current: room for the lake, artifacts, logs and Airflow's db
DISK_FAIL = 5 * GIB
RUNNER_IMAGE = os.environ.get("SHOWCASE_RUNNER_IMAGE", "mbt-showcase-runner:dev")

#: Host ports the stack publishes, by .env name, with their defaults.
PORTS = {
    "SHOWCASE_S3_PORT": 8333,
    "SHOWCASE_FILER_PORT": 8388,
    "SHOWCASE_MLFLOW_PORT": 5501,
    "SHOWCASE_SPARK_UI_PORT": 8380,
    "SHOWCASE_JUPYTER_PORT": 8899,
    "SHOWCASE_PUSHGW_PORT": 9491,
    "SHOWCASE_ALERTMANAGER_PORT": 9493,
    "SHOWCASE_PROMETHEUS_PORT": 9490,
    "SHOWCASE_GRAFANA_PORT": 3390,
    "SHOWCASE_GITEA_PORT": 3305,
    "SHOWCASE_WOODPECKER_PORT": 8305,
    "SHOWCASE_WEBHOOK_PORT": 9309,
    "SHOWCASE_ZOT_PORT": 15000,
    "SHOWCASE_AIRFLOW_PORT": 8280,
}


class Report:
    def __init__(self) -> None:
        self.failed = False

    def ok(self, text: str) -> None:
        print(f"  ok    {text}")

    def warn(self, text: str, fix: str = "") -> None:
        print(f"  warn  {text}")
        for line in fix.splitlines():
            print(f"        {line}")

    def fail(self, text: str, fix: str) -> None:
        self.failed = True
        print(f"  FAIL  {text}")
        for line in fix.splitlines():
            print(f"        {line}")


def docker(*args: str, timeout: int = 120) -> subprocess.CompletedProcess[str]:
    return subprocess.run(
        ["docker", *args], capture_output=True, text=True, timeout=timeout, check=False
    )


def check_daemon(report: Report) -> bool:
    try:
        info = docker("info", "--format", "{{json .}}", timeout=60)
    except FileNotFoundError:
        report.fail("docker is not on PATH", "install Docker Desktop, OrbStack or Rancher Desktop")
        return False
    if info.returncode != 0:
        report.fail("the docker daemon is not reachable", "start your docker runtime, then retry")
        return False
    data = json.loads(info.stdout)
    memory = int(data.get("MemTotal", 0))
    name = data.get("OperatingSystem", "docker")
    if memory < MEMORY_WANTED:
        report.warn(
            f"{name} gives docker {memory / GIB:.1f}GB of memory; the stack wants ~10GB",
            "raise the VM's memory in your docker runtime's settings (README: Knobs)",
        )
    else:
        report.ok(f"{name}: {memory / GIB:.1f}GB of memory for docker")
    return True


def check_disk(report: Report) -> None:
    probe = docker(
        "run", "--rm", "--network", "none", "--entrypoint", "df", PROBE_IMAGE, "-Pk", "/"
    )
    if probe.returncode != 0:
        report.warn("could not measure the docker VM's free disk", probe.stderr.strip()[:200])
        return
    free = int(probe.stdout.splitlines()[-1].split()[3]) * 1024
    text = f"{free / GIB:.1f}GB free on the docker VM's disk"
    fix = (
        "free space: docker system df, then docker image prune / docker builder prune\n"
        "(a full disk corrupts airflow-db and fails every model upload: docs/troubleshooting.md)"
    )
    building = image_needs_build()
    wanted = DISK_WARN_BUILD if building else DISK_WARN
    cache = build_cache_reclaimable()
    if cache:
        fix = (
            f"{cache} of docker build cache is reclaimable; keep the pip cache mount and drop "
            "the layers with:\n  docker builder prune --filter 'type!=exec.cachemount'\n" + fix
        )
    if free < DISK_FAIL:
        report.fail(text, fix)
    elif free < wanted:
        why = "building the runner image" if building else "the running stack"
        report.warn(f"{text}; {why} wants ~{wanted // GIB}GB", fix)
    else:
        report.ok(text + (" (the runner image will be built)" if building else ""))


def image_needs_build() -> bool:
    """Whether `make up` will build the runner image: none exists, or its
    content label is not this checkout's hash (build_image.sh's own test).
    A prebuilt pull would avoid the build, but whether one is published is
    not known until the pull is tried, so plan for the build."""
    labels = docker("image", "inspect", "--format", "{{json .Config.Labels}}", RUNNER_IMAGE)
    if labels.returncode != 0:
        return True
    current = subprocess.run(
        ["bash", str(SHOWCASE_DIR / "scripts" / "build_image.sh"), "--print-hash"],
        capture_output=True,
        text=True,
        timeout=120,
        check=False,
    ).stdout.strip()
    built = (json.loads(labels.stdout or "null") or {}).get("mbt.showcase.content", "")
    return not current or built != current


def build_cache_reclaimable() -> str:
    """The build cache docker would reclaim, as `docker system df` prints it
    (e.g. "9.9GB"), or "" when it is negligible."""
    usage = docker("system", "df", "--format", "{{.Type}}\t{{.Reclaimable}}")
    for line in usage.stdout.splitlines():
        kind, _, reclaimable = line.partition("\t")
        if kind == "Build Cache" and reclaimable.split(" ")[0].endswith("GB"):
            return reclaimable.split(" ")[0]
    return ""


def check_socket_group(report: Report) -> None:
    probe = subprocess.run(
        [str(SHOWCASE_DIR / "scripts" / "docker_sock_gid.sh")],
        capture_output=True,
        text=True,
        timeout=120,
        check=False,
    )
    gid = probe.stdout.strip()
    if probe.stderr.strip():
        report.fail(
            f"docker socket group {gid}: {probe.stderr.strip()}",
            "Airflow's DAG tasks drive the daemon through this socket; see DOCKER_SOCK_GID "
            "in .env.example",
        )
    else:
        report.ok(f"docker socket group as containers see it: {gid} (the scheduler joins it)")


def stack_containers() -> list[str]:
    listing = docker("ps", "-aq", "--filter", f"label=com.docker.compose.project={PROJECT}")
    return listing.stdout.split()


def check_ports(report: Report, running: bool) -> None:
    if running:
        report.ok(f"stack {PROJECT} has containers, so its ports are its own (not re-checked)")
        return
    taken = []
    for name, default in PORTS.items():
        port = int(os.environ.get(name) or default)
        with socket.socket() as sock:
            sock.setsockopt(socket.SOL_SOCKET, socket.SO_REUSEADDR, 1)
            try:
                sock.bind(("127.0.0.1", port))
            except OSError:
                taken.append(f"{name}={port}")
    if taken:
        report.fail(
            f"host port(s) already in use: {', '.join(taken)}",
            "stop whatever listens there, or move the stack's port in .env (see .env.example)",
        )
    else:
        report.ok(f"all {len(PORTS)} host ports are free")


def check_network(report: Report, running: bool, fix: bool) -> None:
    inspect = docker("network", "inspect", NETWORK, "--format", "{{json .Containers}}")
    if inspect.returncode != 0:
        report.ok(f"no leftover network {NETWORK}")
        return
    endpoints = json.loads(inspect.stdout or "{}") or {}
    ghosts = [
        endpoint["Name"]
        for container_id, endpoint in endpoints.items()
        if docker("inspect", container_id).returncode != 0
    ]
    if not ghosts:
        report.ok(f"network {NETWORK} has no endpoint without a container")
        return
    recipe = "\n".join(
        [
            "make down",
            *(f"docker network disconnect -f {NETWORK} {name}" for name in ghosts),
            f"docker network rm {NETWORK}",
            "(or: make doctor FIX=1)",
        ]
    )
    if not fix:
        report.fail(
            f"network {NETWORK} has endpoint(s) whose container is gone: {', '.join(ghosts)}",
            recipe,
        )
        return
    if running:
        report.fail(
            "run make down before make doctor FIX=1: the stack still has containers", recipe
        )
        return
    for name in ghosts:
        docker("network", "disconnect", "-f", NETWORK, name)
    removed = docker("network", "rm", NETWORK)
    if removed.returncode == 0:
        report.ok(f"removed leftover network {NETWORK}")
    else:
        report.fail(f"could not remove {NETWORK}: {removed.stderr.strip()}", recipe)


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    parser.add_argument("--fix", action="store_true", help="remove a leftover network")
    args = parser.parse_args(argv)

    print(f"mbt showcase doctor ({PROJECT}):")
    report = Report()
    if check_daemon(report):
        check_disk(report)
        check_socket_group(report)
        running = bool(stack_containers())
        check_ports(report, running)
        check_network(report, running, args.fix)
    if report.failed:
        print("doctor found a problem that will stop the stack; fix it and rerun make doctor")
        return 1
    return 0


if __name__ == "__main__":
    sys.exit(main())
