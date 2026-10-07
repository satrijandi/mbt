"""A weaker challenger meets the champion gate (runs inside the runner image).

The data scientist's story: in a working copy, drop the eight activity
features from churn_automl (logins, sessions, transactions), keep the eight
profile features, and retrain on the shared registry. The model spec's
champion gate - PR-AUC against `production`, judged on the paired-bootstrap
lower bound of the delta (ADR-18) - re-scores the production champion on the
challenger's own test split and refuses the challenger: `mbt build` exits 2,
nothing is staged, and `production` keeps serving the version it served
before. The committed project is never touched; the copy lives under
/workspace/tmp and is removed afterwards.

Exit 0 when the gate refused the challenger and production is unchanged,
1 for anything else (including a challenger the gate let through).
"""

import json
import os
import shutil
import subprocess
import sys
import urllib.error
import urllib.request
from pathlib import Path

PROJECT = Path("/workspace/project")
COPY = Path("/workspace/tmp/challenger")
MODEL = "churn_automl"
MLFLOW = os.environ.get("MLFLOW_TRACKING_URI", "http://mlflow:5000")
DROPPED = (
    "login_days_30d",
    "days_since_login",
    "sessions_30d",
    "avg_session_min",
    "txn_cnt_30d",
    "txn_amt_sum_30d",
    "txn_amt_avg_90d",
    "merchant_diversity",
)


def production_version() -> str | None:
    url = f"{MLFLOW}/api/2.0/mlflow/registered-models/alias?name={MODEL}&alias=production"
    try:
        with urllib.request.urlopen(url, timeout=30) as response:
            return str(json.load(response)["model_version"]["version"])
    except urllib.error.HTTPError as exc:
        if exc.code == 404:
            return None
        raise


def drop_features(spec: Path) -> None:
    lines = spec.read_text().splitlines(keepends=True)
    kept = [line for line in lines if line.strip().removeprefix("- ") not in DROPPED]
    if len(lines) - len(kept) != len(DROPPED):
        raise SystemExit(f"expected to drop {len(DROPPED)} feature lines from {spec}")
    spec.write_text("".join(kept))


def main(anchor: str) -> int:
    before = production_version()
    if before is None:
        print(f"no production version of {MODEL} yet: run `make demo` first, so there is a")
        print("champion for the gate to compare against (without one it passes as a bootstrap)")
        return 1
    print(f"production serves {MODEL} v{before}")

    shutil.rmtree(COPY, ignore_errors=True)
    shutil.copytree(PROJECT, COPY, ignore=shutil.ignore_patterns("target", ".ipynb_checkpoints"))
    try:
        drop_features(COPY / "models" / f"{MODEL}.yml")
        print(f"working copy drops {len(DROPPED)} activity features: {', '.join(DROPPED)}")
        build = subprocess.run(
            ["mbt", "build", "--target", "dev", "--select", f"+{MODEL}", "--anchor", anchor],
            cwd=COPY,
            check=False,
        )
    finally:
        shutil.rmtree(COPY, ignore_errors=True)

    after = production_version()
    print(f"mbt build exited {build.returncode}; production serves {MODEL} v{after}")
    if build.returncode == 2 and after == before:
        print("the champion gate refused the weaker challenger; production is unchanged")
        return 0
    print("UNEXPECTED: the gate should refuse this challenger with exit 2")
    return 1


if __name__ == "__main__":
    sys.exit(main(sys.argv[1]))
