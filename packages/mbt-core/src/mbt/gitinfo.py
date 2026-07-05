"""Git metadata for manifests (TSD §8.5)."""

import subprocess
from pathlib import Path


def collect_git_info(project_dir: Path) -> dict[str, object]:
    """Best-effort git commit/branch/dirty; nulls outside a repository."""

    def run(*args: str) -> str | None:
        try:
            proc = subprocess.run(
                ["git", *args],
                cwd=project_dir,
                capture_output=True,
                text=True,
                timeout=10,
                check=False,
            )
        except (OSError, subprocess.TimeoutExpired):
            return None
        if proc.returncode != 0:
            return None
        return proc.stdout.strip()

    commit = run("rev-parse", "HEAD")
    if commit is None:
        return {"commit": None, "branch": None, "dirty": False}
    branch = run("rev-parse", "--abbrev-ref", "HEAD")
    status = run("status", "--porcelain")
    return {"commit": commit, "branch": branch, "dirty": bool(status)}
