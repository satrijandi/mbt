"""Shared helpers for the mbt-airflow tests (unique module name for pytest)."""

import json
import shutil
import stat
import subprocess
import sys
import textwrap
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parents[3]
CHURN_DEMO = REPO_ROOT / "tests" / "fixtures" / "churn_demo"
#: The demo data is fixed-dated, so windows only resolve at a pinned anchor.
ANCHOR = "2026-06-30T00:00:00Z"

#: The CLI entry point, run as the user would run it but without relying on PATH.
MBT = [sys.executable, "-c", "from mbt.cli.main import main; main()"]


def compile_churn_demo(workdir: Path) -> tuple[Path, Path]:
    """Copy the churn_demo fixture project and compile it; return (project, manifest)."""
    project = workdir / "project"
    shutil.copytree(CHURN_DEMO, project)
    target = workdir / "compiled"
    done = subprocess.run(
        [
            *MBT,
            *["compile", "--project-dir", str(project), "--anchor", ANCHOR],
            *["--target-path", str(target), "--quiet"],
        ],
        capture_output=True,
        text=True,
        check=False,
    )
    assert done.returncode == 0, done.stderr
    return project, target / "manifest.json"


#: A stand-in for the ``mbt`` executable, driven by environment variables, so the
#: operators can be tested for every exit code without training anything.
_FAKE_MBT = """\
#!{python}
import json, os, shutil, sys, time
from pathlib import Path

argv = sys.argv[1:]
def option(name):
    return argv[argv.index(name) + 1] if name in argv else None

log = os.environ.get("FAKE_MBT_LOG")
if log:
    with open(log, "a") as handle:
        marker = os.environ.get("FAKE_MBT_MARKER")
        task_env = {{k: v for k, v in os.environ.items() if k.startswith("MBT_AIRFLOW_")}}
        handle.write(json.dumps({{"argv": argv, "marker": marker, "env": task_env}}) + "\\n")

if argv[0] == "anchor":
    print(os.environ.get("FAKE_MBT_ANCHOR", ""))
    print(os.environ.get("FAKE_MBT_STDERR", ""), file=sys.stderr)
    sys.exit(int(os.environ.get("FAKE_MBT_ANCHOR_EXIT", "0")))

started = os.environ.get("FAKE_MBT_STARTED")
if started:
    Path(started).write_text("up")
if os.environ.get("FAKE_MBT_SLEEP"):
    time.sleep(float(os.environ["FAKE_MBT_SLEEP"]))

target = Path(option("--target-path"))
target.mkdir(parents=True, exist_ok=True)
if argv[0] == "compile":
    shutil.copy(os.environ["FAKE_MBT_MANIFEST"], target / "manifest.json")
elif os.environ.get("FAKE_MBT_RESULTS", "yes") == "yes":
    node = {{"unique_id": os.environ.get("FAKE_MBT_NODE") or option("--select"),
            "status": os.environ.get("FAKE_MBT_STATUS", "success"),
            "message": os.environ.get("FAKE_MBT_MESSAGE"),
            "metrics": json.loads(os.environ.get("FAKE_MBT_METRICS", "{{}}"))}}
    (target / "run_results.json").write_text(json.dumps({{"results": [node]}}))

print(os.environ.get("FAKE_MBT_STDERR", ""), file=sys.stderr)
sys.exit(int(os.environ.get("FAKE_MBT_EXIT", "0")))
"""


def write_fake_mbt(directory: Path) -> str:
    """Write the fake ``mbt`` executable and return its path."""
    path = directory / "fake-mbt"
    path.write_text(textwrap.dedent(_FAKE_MBT).format(python=sys.executable))
    path.chmod(path.stat().st_mode | stat.S_IXUSR)
    return str(path)


def logged_calls(log: Path) -> list[dict]:
    return [json.loads(line) for line in log.read_text().splitlines()]
