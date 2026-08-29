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


#: Per-package runtime-closure probes (F7): what each package's DOCUMENTED
#: user-facing paths import lazily at runtime (ADR-14 lazy imports mean a bare
#: `import <package>` proves nothing). Installing ONE package into its own
#: clean venv and running its probe catches a missing declared dependency that
#: the monorepo workspace masks - the exact way mbt-spark shipped without
#: sklearn/numpy/pandas in its closure. The calibrator probe doubles as the
#: `mbt-adapter-base[metrics]` extra check for every training adapter.
_CALIBRATOR_PROBE = (
    "from mbt_adapter_base.calibration import Calibrator; "
    "Calibrator.fit([0.2, 0.8, 0.4, 0.6], [0, 1, 0, 1], 'isotonic')"
)
CLOSURE_PROBES = {
    "mbt_core": (
        "import duckdb, pyarrow, typer; "
        "from mbt.adapters.local.data import LocalDataAdapter; "
        "from mbt.cli.main import main"
    ),
    "mbt_adapter_base": "import pyarrow, pydantic; import mbt_adapter_base.monitoring",
    "mbt_testing": "import pyarrow; from mbt_testing.adapters import FakeTrainingAdapter",
    "mbt_xgboost": f"import xgboost, numpy; {_CALIBRATOR_PROBE}",
    "mbt_lightgbm": f"import lightgbm, numpy; {_CALIBRATOR_PROBE}",
    "mbt_mlflow": "import mlflow.tracking; from mbt_mlflow.adapter import MlflowTracking",
    "mbt_optuna": "import optuna; from mbt_optuna.engine import OptunaTuningEngine",
    "mbt_snowflake": (
        "import snowflake.connector, pyarrow.parquet; "
        "from mbt_snowflake.adapter import SnowflakeDataAdapter"
    ),
    # the F7 regression: spark's scoring path imports numpy + pandas and its
    # calibration path imports sklearn - none arrive without [metrics]+pandas
    "mbt_spark": f"import pyspark, numpy, pandas; {_CALIBRATOR_PROBE}",
    "mbt_h2o": f"import h2o, numpy; {_CALIBRATOR_PROBE}",
}


def _clean_env() -> dict[str, str]:
    # Neither var may leak into the clean venv: PYTHONPATH would shadow the
    # wheel installs with workspace sources, VIRTUAL_ENV would misdirect uv.
    return {k: v for k, v in os.environ.items() if k not in ("VIRTUAL_ENV", "PYTHONPATH")}


@pytest.fixture(scope="module")
def built_dist(tmp_path_factory: pytest.TempPathFactory) -> tuple[Path, Path]:
    """All workspace wheels + the locked third-party constraints, built once."""
    uv = shutil.which("uv")
    assert uv, "uv is required to build the workspace wheels (see README)"
    root = tmp_path_factory.mktemp("wheelhouse")
    dist = root / "dist"
    env = _clean_env()
    _run([uv, "build", "--all-packages", "--out-dir", str(dist)], cwd=REPO_ROOT, env=env)
    constraints = root / "constraints.txt"
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
        env=env,
    )
    return dist, constraints


@pytest.mark.parametrize("package", sorted(CLOSURE_PROBES))
def test_each_package_installs_standalone_with_a_complete_closure(
    package: str, built_dist: tuple[Path, Path], tmp_path: Path
) -> None:
    """F7: install ONE package's wheel into its own clean venv (deps resolved
    only from the package's declared closure) and exercise its lazy runtime
    imports. The single-venv quickstart test below cannot catch a per-package
    gap - installing everything together masks a missing declaration."""
    dist, constraints = built_dist
    env = _clean_env()
    uv = shutil.which("uv")
    assert uv
    venv = tmp_path / "venv"
    _run([uv, "venv", str(venv), "--python", sys.executable], cwd=tmp_path, env=env)
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
        str(dist / f"{package}-{mbt.__version__}-py3-none-any.whl"),
    ]
    offline = _run([*install, "--offline"], cwd=tmp_path, env=env, check=False)
    if offline.returncode != 0:  # cold uv cache: fall back to the network
        _run(install, cwd=tmp_path, env=env)
    probe = _run(
        [str(venv / "bin" / "python"), "-c", CLOSURE_PROBES[package]],
        cwd=tmp_path,
        env=env,
        check=False,
    )
    assert probe.returncode == 0, (
        f"{package} standalone closure is incomplete: its declared dependencies "
        f"do not cover its runtime imports\nprobe: {CLOSURE_PROBES[package]}\n"
        f"stderr:\n{probe.stderr}"
    )


def test_wheels_install_and_run_the_quickstart(tmp_path: Path) -> None:
    uv = shutil.which("uv")
    assert uv, "uv is required to build the workspace wheels (see README)"
    clean_env = _clean_env()

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


def test_every_built_wheel_carries_its_pep_561_marker(built_dist: tuple[Path, Path]) -> None:
    """tests/test_version_sync.py asserts the marker is in the source tree;
    this asserts the build actually ships it, which is the half that decides
    whether a consumer's type checker sees mbt's types at all.

    Worth proving against a real wheel rather than trusting hatchling's
    default include: a stray `[tool.hatch.build] include`/`exclude` on any one
    package would drop the marker silently, and the failure mode downstream is
    not an error - it is types quietly degrading to Any.
    """
    import zipfile

    dist, _ = built_dist
    wheels = sorted(dist.glob("mbt*-py3-none-any.whl"))
    assert len(wheels) == 10, [w.name for w in wheels]
    for wheel in wheels:
        with zipfile.ZipFile(wheel) as archive:
            markers = [name for name in archive.namelist() if name.endswith("/py.typed")]
        assert markers, f"{wheel.name} ships no py.typed; downstream types degrade to Any"
