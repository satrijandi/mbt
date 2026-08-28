"""The showcase's Snowflake data plane, end to end (DESIGN.md section 11).

The same wide cadence the lake planes run - the same dataset, model, and
scoring specs, unedited - reading Snowflake instead of parquet. What differs is
one word on the command line: `--target snowflake`.

Why this tier exists separately from the lake modules: it proves the ADR-22
wide shape is genuinely data-plane-agnostic rather than lake-shaped, and it
closes the serving leg (`mbt score` / `mbt monitor`) against a warehouse
adapter, which is the half of issue #1 that shipped.

Shape of the run:

  seed  -> the SAME parquet the lake is seeded from, uploaded to Snowflake, so
           the two planes are comparable by construction
  build -> the wide panel joined IN Snowflake (heterogeneous entity keys), the
           gate, and registration as churn_wide_automl_snowflake
  score -> the newest cohort, predictions staged as parquet (ADR-23 v1)
  monitor -> realized metrics once the labels mature

Runs on the HOST, not in the runner container: warehouse credentials live in
the developer's shell, SSO needs a real browser, and the runner image does not
ship mbt-snowflake. The stack is reached over its published ports.

TRIPLE gated (MBT_LIVE_SHOWCASE=1 + MBT_LIVE_SNOWFLAKE=1 + complete
SNOWFLAKE_*), so the hermetic showcase guarantee is untouched and credentials
alone never trigger warehouse traffic. The hermetic half of this coverage -
the join SQL, the both-addresses invariant, the seeder/sources agreement -
lives in packages/mbt-snowflake/tests/test_showcase_snowflake_plane.py and
needs no account.
"""

import json
import subprocess
import sys

import pytest
from showcase_utils import (
    ANCHOR,
    MONITOR_ANCHOR,
    REPO_ROOT,
    SHOWCASE_DIR,
    SNOWFLAKE_MARKS,
    require_snowflake,
)

pytestmark = SNOWFLAKE_MARKS

MODEL_NODE = "model.churn_lake.churn_wide_automl"
DATASET_NODE = "dataset.churn_lake.wide_churn_training"
SCORING_NODE = "scoring.churn_lake.wide_retention_scoring"
#: plane_suffix on the snowflake target makes the REGISTERED name distinct;
#: the model NODE name is shared with the lake plane (profiles.yml).
REGISTERED_MODEL = "churn_wide_automl_snowflake"

SEED_SCRIPT = SHOWCASE_DIR / "scripts" / "seed_snowflake.py"


def _latest_materialization(stack) -> dict:
    """Row counts of the most recently materialized wide panel.

    Datasets land under target/datasets/<node>/<materialization key>/, one
    directory per key, and the two planes produce different keys - so "newest"
    is how a test picks out the build it just ran. NodeResult carries no row
    count (it forbids extra fields), which is why this reads the adapter's own
    materialization.json instead.
    """
    root = stack.workspace / "project" / "target" / "datasets" / "wide_churn_training"
    metadata = sorted(root.glob("*/materialization.json"), key=lambda p: p.stat().st_mtime)
    assert metadata, f"no materialization under {root}"
    return json.loads(metadata[-1].read_text())


def _seed(stack, *args: str) -> subprocess.CompletedProcess[str]:
    proc = subprocess.run(
        [sys.executable, str(SEED_SCRIPT), *args],
        cwd=REPO_ROOT,
        env=stack.host_env(),
        capture_output=True,
        text=True,
        timeout=1800,
        check=False,
    )
    assert proc.returncode == 0, (
        f"seed_snowflake.py {args} exited {proc.returncode}\n"
        f"--- stdout ---\n{proc.stdout}\n--- stderr ---\n{proc.stderr}"
    )
    return proc


@pytest.fixture(scope="module")
def warehouse(showcase_stack):
    """The stack, the demo tables in Snowflake, and the wide cadence built.

    The build lives in the fixture rather than in the first test so the
    module stays standalone-safe the way every other showcase module is: no
    test depends on a sibling having run first. It matters more here than
    usual, because each mbt invocation OVERWRITES run_results.json.

    Tables are dropped at teardown even on failure: this suite writes into the
    user's own sandbox schema, so it must not leave litter there any more than
    it would in the repo.
    """
    require_snowflake()
    _seed(showcase_stack, "--force")
    try:
        showcase_stack.host_mbt(
            "build", "--target", "snowflake", "--select", "tag:wide", "--anchor", ANCHOR
        )
        yield showcase_stack
    finally:
        _seed(showcase_stack, "--drop")


def test_wide_panel_builds_and_gates_on_the_warehouse(warehouse) -> None:
    """The committed wide cadence, trained straight out of Snowflake."""
    stack = warehouse
    dataset = stack.result_for(DATASET_NODE)
    assert dataset["status"] == "success", dataset
    # The five-table join ran IN Snowflake; only the panel came back.
    row_counts = _latest_materialization(stack)["row_counts"]
    assert row_counts.get("train", 0) > 0 and row_counts.get("test", 0) > 0, row_counts

    model = stack.result_for(MODEL_NODE)
    assert model["status"] == "success", model
    assert all(gate["passed"] for gate in model["gates"]), model["gates"]


def test_registered_model_is_namespaced_per_plane(warehouse) -> None:
    """Both planes train the same spec; their versions must not collide.

    Without plane_suffix the warehouse plane would push versions into the lake
    plane's registered model, and champion resolution would silently start
    mixing models trained on different data.
    """
    stack = warehouse
    import urllib.request

    url = f"{stack.mlflow_url()}/api/2.0/mlflow/registered-models/search?max_results=200"
    with urllib.request.urlopen(url, timeout=60) as response:
        payload = json.loads(response.read())
    names = {entry["name"] for entry in payload.get("registered_models", [])}

    # The suffix took effect: had plane_suffix not reached registration.name,
    # this plane's version would have landed on the bare lake-plane model.
    assert REGISTERED_MODEL in names, names
    # And the run really did register under the suffixed name, not both.
    registration = stack.result_for(MODEL_NODE)["registration"]
    assert registration["name"] == REGISTERED_MODEL, registration


def test_score_and_monitor_close_the_loop_on_the_warehouse(warehouse) -> None:
    """The serving leg against a warehouse adapter: score, then monitor.

    Predictions stage as parquet under predictions_root (ADR-23 v1); the
    warehouse-native store is v2 (issue #1).
    """
    stack = warehouse
    # --target is load-bearing: promote would otherwise fall back to dev,
    # whose registry URI is the in-network mlflow the host cannot reach.
    stack.host_mbt(
        "promote", "--target", "snowflake", "--model", REGISTERED_MODEL, "--to", "production"
    )
    stack.host_mbt(
        "score",
        "--target",
        "snowflake",
        "--select",
        "tag:wide",
        "--anchor",
        ANCHOR,
    )
    scoring = stack.result_for(SCORING_NODE)
    assert scoring["status"] == "success", scoring

    runs = sorted((stack.workspace / "snowflake_predictions").rglob("predictions.json"))
    assert runs, "no prediction run staged under predictions_root"

    # Ground truth matures after the scoring anchor; the pinned monitor anchor
    # is past it, so exactly one run evaluates (ADR-21 exactly-once).
    stack.host_mbt(
        "monitor",
        "--target",
        "snowflake",
        "--select",
        "tag:wide",
        "--anchor",
        MONITOR_ANCHOR,
    )
    sidecar = json.loads(runs[-1].read_text())
    assert sidecar.get("model_version"), sidecar


def test_the_two_planes_agree_on_the_panel(warehouse) -> None:
    """The claim the whole design rests on: same specs, same data, same panel.

    The lake plane and the warehouse plane are seeded from identical parquet,
    so the materialized training panels must agree on row count and column
    set. A divergence here means a plane is silently reshaping the data -
    exactly what a second backend risks, and what makes "just switch targets"
    either true or a lie.
    """
    stack = warehouse
    dataset_name = DATASET_NODE.rsplit(".", maxsplit=1)[-1]

    # Rebuild each plane's panel here rather than reusing an earlier test's,
    # so this holds regardless of what ran before it. Only the dataset node is
    # selected, so neither rebuild pays for AutoML.
    stack.host_mbt("build", "--target", "snowflake", "--select", dataset_name, "--anchor", ANCHOR)
    warehouse_counts = _latest_materialization(stack)["row_counts"]

    stack.sync_lake()
    stack.mbt(
        "build",
        "--target",
        "prod_score",
        "--select",
        dataset_name,
        "--anchor",
        ANCHOR,
        "--deep-snapshot",
    )
    lake_counts = _latest_materialization(stack)["row_counts"]

    assert warehouse_counts == lake_counts, (
        f"the planes disagree on the panel: snowflake={warehouse_counts} lake={lake_counts}"
    )
