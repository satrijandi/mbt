"""Wheel-install smoke test: the built distributions work for a fresh user (NFR-10).

`uv build` every workspace package, install the wheels into a clean venv with
third-party dependencies pinned to uv.lock via constraints, and drive the
quickstart loop with the installed console script. This catches exactly what
the editable workspace hides: data files missing from wheels (the init
scaffold), broken entry points (the `mbt` script, adapter plugins), and
runtime imports that only the dev environment provides.
"""

import json
import os
import shutil
import subprocess
import sys
from pathlib import Path

import pytest
from e2e_utils import REPO_ROOT

import mbt

pytestmark = pytest.mark.e2e

#: The quickstart scaffold trains with xgboost and tracks/registers via
#: mlflow; mbt-adapter-base must arrive transitively from --find-links.
QUICKSTART_WHEELS = ("mbt_core", "mbt_xgboost", "mbt_mlflow")


def _run(
    cmd: list[str],
    cwd: Path,
    env: dict[str, str] | None = None,
    *,
    timeout: int = 600,
    check: bool = True,
) -> subprocess.CompletedProcess[str]:
    proc = subprocess.run(
        cmd,
        cwd=cwd,
        env=env,
        capture_output=True,
        text=True,
        timeout=timeout,
        stdin=subprocess.DEVNULL,
        check=False,
    )
    if check:
        assert proc.returncode == 0, (
            f"{' '.join(cmd)} exited {proc.returncode}\n"
            f"stdout:\n{proc.stdout}\nstderr:\n{proc.stderr}"
        )
    return proc


def test_wheels_install_and_run_the_quickstart(tmp_path: Path) -> None:
    uv = shutil.which("uv")
    assert uv, "uv is required to build the workspace wheels (see README)"
    # Neither var may leak into the clean venv: PYTHONPATH would shadow the
    # wheel installs with workspace sources, VIRTUAL_ENV would misdirect uv.
    clean_env = {k: v for k, v in os.environ.items() if k not in ("VIRTUAL_ENV", "PYTHONPATH")}

    # Build every package (wheels are built from the sdists, so this also
    # verifies sdist completeness) and check nothing is missing or misnamed.
    dist = tmp_path / "dist"
    _run([uv, "build", "--all-packages", "--out-dir", str(dist)], cwd=REPO_ROOT, env=clean_env)
    packages = sorted(p.name for p in (REPO_ROOT / "packages").iterdir() if p.is_dir())
    built = {wheel.name for wheel in dist.glob("*.whl")}
    expected = {f"{p.replace('-', '_')}-{mbt.__version__}-py3-none-any.whl" for p in packages}
    assert built == expected, f"built wheels {built} != workspace packages {expected}"

    # Pin third-party deps to the locked versions so the install is
    # reproducible and (cache warm) network-free; constraints only pin, so
    # the wheels' own declared dependencies still drive what gets installed.
    constraints = tmp_path / "constraints.txt"
    _run(
        [
            uv,
            "export",
            "--frozen",
            "--no-emit-workspace",
            "--no-hashes",
            "--no-annotate",
            "--no-header",
            "-o",
            str(constraints),
        ],
        cwd=REPO_ROOT,
        env=clean_env,
    )

    venv = tmp_path / "venv"
    _run([uv, "venv", str(venv), "--python", sys.executable], cwd=tmp_path, env=clean_env)
    install = [
        uv,
        "pip",
        "install",
        "--python",
        str(venv / "bin" / "python"),
        "--find-links",
        str(dist),
        "--constraint",
        str(constraints),
        *(str(dist / f"{name}-{mbt.__version__}-py3-none-any.whl") for name in QUICKSTART_WHEELS),
    ]
    offline = _run([*install, "--offline"], cwd=tmp_path, env=clean_env, check=False)
    if offline.returncode != 0:  # cold uv cache: fall back to the network
        _run(install, cwd=tmp_path, env=clean_env)

    # The venv resolves its own mbt, not the editable workspace; without this
    # guard, source leakage would make every assertion below vacuous.
    where = _run(
        [str(venv / "bin" / "python"), "-c", "import mbt; print(mbt.__file__)"],
        cwd=tmp_path,
        env=clean_env,
    )
    assert str(venv) in where.stdout, f"mbt leaked from outside the venv: {where.stdout}"

    # The quickstart, exactly as the README tells a new user to run it, via
    # the installed console script. HOME is sandboxed (init writes ~/.mbt).
    home = tmp_path / "home"
    home.mkdir()
    user_env = {**clean_env, "HOME": str(home)}
    mbt_cli = str(venv / "bin" / "mbt")
    _run([mbt_cli, "init", "quickstart"], cwd=tmp_path, env=user_env)
    project = tmp_path / "quickstart"
    assert (home / ".mbt" / "profiles.yml").is_file()  # scaffold shipped in the wheel

    _run(
        [str(venv / "bin" / "python"), str(project / "scripts" / "generate_sample_data.py"), "400"],
        cwd=project,
        env=user_env,
    )
    _run([mbt_cli, "build"], cwd=project, env=user_env)

    # Local data + xgboost + mlflow adapters all loaded via entry points, the
    # model trained, passed its gate, and registered.
    payload = json.loads((project / "target" / "run_results.json").read_text())
    statuses = {result["unique_id"]: result["status"] for result in payload["results"]}
    assert statuses == {
        "dataset.quickstart.churn_training_set": "success",
        "model.quickstart.churn_classifier": "success",
    }
    assert (project / "target" / "manifest.json").is_file()
