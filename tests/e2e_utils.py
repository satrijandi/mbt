"""Shared helpers for repo-level E2E tests (unique module name for pytest)."""

import subprocess
import sys
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parent.parent
CHURN_DEMO = REPO_ROOT / "examples" / "churn_demo"

#: Anchor matching the committed demo data range (generated around 2026-01..06).
DEMO_ANCHOR = "2026-06-30T00:00:00Z"


def run_mbt(
    args: list[str], cwd: Path, *, expect_exit: int = 0, timeout: int = 300
) -> subprocess.CompletedProcess[str]:
    """Invoke the real CLI in a subprocess (non-interactive, FR-CLI-01)."""
    proc = subprocess.run(
        [sys.executable, "-m", "mbt.cli.main", *args],
        cwd=cwd,
        capture_output=True,
        text=True,
        timeout=timeout,
        stdin=subprocess.DEVNULL,
        check=False,
    )
    assert proc.returncode == expect_exit, (
        f"mbt {' '.join(args)} exited {proc.returncode}, expected {expect_exit}\n"
        f"stdout:\n{proc.stdout}\nstderr:\n{proc.stderr}"
    )
    return proc
