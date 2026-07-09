"""CLI output paths redact tainted secrets (NFR-07 defense in depth).

Text that leaves the process must pass through redact(), like the event /
manifest / run_results sinks. Two vectors are pinned here:

* ``fail()`` renders MbtError text to stderr, and AdapterError.wrap embeds raw
  underlying exceptions that can carry a connection string or token.
* ``mbt show`` dumps compile-rendered spec config, and a spec field may render
  an ``env_var()`` value (the jinja resolve context taints it) - the manifest
  *file* redacts the same config on write, so ``show`` must too.
"""

import os
import subprocess
import sys
from pathlib import Path

from mbt.cli.common import err_console, fail
from mbt.exceptions import AdapterError
from mbt.secrets import REDACTED, taint


def test_fail_redacts_tainted_secret_in_every_field() -> None:
    secret = "sk-DEADBEEF-TOKEN-9999"
    taint(secret)  # exactly what env_var() does while loading profiles
    exc = AdapterError(
        f"connect failed: {secret}",
        resource=f"m.{secret}",
        path=f"/p/{secret}",
        hint=f"token {secret}",
    )
    with err_console.capture() as capture:
        fail(exc)
    out = capture.get()

    assert secret not in out, "raw credential leaked to stderr"
    # message + resource + file + hint each masked exactly once.
    assert out.count(REDACTED) >= 4


def test_show_redacts_env_var_secret_in_output(demo_project: Path) -> None:
    secret = "sk-SHOW-LEAK-4242"
    model = demo_project / "models" / "churn_model.yml"
    model.write_text(
        model.read_text().replace(
            "owner: ds@example.com", "owner: \"{{ env_var('MBT_SHOW_SECRET') }}\""
        )
    )
    proc = subprocess.run(
        [
            sys.executable,
            "-m",
            "mbt.cli.main",
            "show",
            "churn_model",
            "--project-dir",
            str(demo_project),
            "--output",
            "json",
        ],
        capture_output=True,
        text=True,
        check=False,
        stdin=subprocess.DEVNULL,
        env={**os.environ, "MBT_SHOW_SECRET": secret},
    )
    assert proc.returncode == 0, proc.stdout + proc.stderr
    assert secret not in proc.stdout, "raw env_var secret leaked to stdout"
    assert REDACTED in proc.stdout  # the owner field was masked
