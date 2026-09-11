"""The showcase's object-store data plane, end to end (DESIGN.md section 11, P8).

The same wide cadence the lake planes run - the same dataset, model, and
scoring specs, unedited - sourced straight out of SeaweedFS. What differs is
one word on the command line: `--target seaweedfs`.

Why this module exists next to test_showcase_wide.py, which already trains the
wide cadence off s3a: every batch leg in the showcase serves `mbt score` and
`mbt monitor` from `prod_score`, the LOCAL (DuckDB) adapter over the synced
copy that bootstrap/sync_lake.py mirrors out of s3://mbt-lake. So until this
module nothing here ever read a live object store at SCORE time, and
mbt-spark's contract-1.1 methods (`build_scoring_input` / `open_predictions`,
shipped in 6c399c9) had unit coverage only. This closes the serving leg
against an object store, with no copy step anywhere in the loop.

It is the SeaweedFS counterpart of test_showcase_snowflake.py, and the
difference that matters operationally is the gate: this one needs docker and
nothing else - no account, no credentials, no browser - so "one project, many
data planes" is proven on every run of the default showcase tier rather than
only on a machine with a warehouse attached.

Shape of the run:

  build   -> the wide panel read from s3a://mbt-lake, the gate, and
             registration as churn_wide_automl_seaweedfs
  score   -> the newest cohort, read from the same object store, predictions
             staged as parquet under predictions_root (ADR-23 v1)
  monitor -> realized metrics once the labels mature, labels also from s3a

No `--deep-snapshot` anywhere: for a URI source the spark adapter hashes the
table's input-file LISTING, which is already mtime-independent, so the deep
and shallow tokens agree. (ADR-11's fresh-checkout problem is a local-path
problem.) The lake's other planes pass it because they read a downloaded copy
whose mtimes really do move; mixing the two schemes on one pipeline is what
the "one token scheme per pipeline" rule forbids.

Runs IN the stack like every other lake module, so the module is
standalone-safe: it provisions its own champion and never depends on a
sibling having run first.
"""

import json

import pytest
from showcase_utils import ANCHOR, MONITOR_ANCHOR, SHOWCASE_MARKS

pytestmark = SHOWCASE_MARKS

TARGET = "seaweedfs"
DATASET = "wide_churn_training"
DATASET_NODE = "dataset.churn_lake.wide_churn_training"
MODEL_NODE = "model.churn_lake.churn_wide_automl"
SCORING_NODE = "scoring.churn_lake.wide_retention_scoring"
#: plane_suffix on the seaweedfs target makes the REGISTERED name distinct;
#: the model NODE name is shared with the lake plane (profiles.yml).
REGISTERED_MODEL = "churn_wide_automl_seaweedfs"


def _latest_materialization(stack) -> dict:
    """Row counts of the most recently materialized wide panel.

    Datasets land under target/datasets/<node>/<materialization key>/, one
    directory per key, and the planes produce different keys (their source
    snapshots differ), so "newest" is how a test picks out the build it just
    ran. NodeResult carries no row count - it forbids extra fields - which is
    why this reads the adapter's own materialization.json instead.
    """
    root = stack.workspace / "project" / "target" / "datasets" / DATASET
    metadata = sorted(root.glob("*/materialization.json"), key=lambda p: p.stat().st_mtime)
    assert metadata, f"no materialization under {root}"
    return json.loads(metadata[-1].read_text())


def _prediction_runs(stack) -> list:
    """Staged prediction runs, oldest first (ADR-23 v1 parquet staging).

    predictions_root is /workspace/seaweedfs_predictions on this plane, so the
    runs outlive the container the way the lake plane's do under lake_local -
    and never land in the project dir, whose target/ is wiped (F20).
    """
    root = stack.workspace / "seaweedfs_predictions" / "predictions" / "wide_retention_scores"
    if not root.is_dir():
        return []
    return sorted(root.glob("*/predictions.json"), key=lambda p: p.stat().st_mtime)


@pytest.fixture(scope="module")
def object_store(showcase_stack):
    """The stack with the wide cadence built straight off SeaweedFS.

    The build lives in the fixture rather than in the first test so the module
    stays standalone-safe the way every other showcase module is. It matters
    more here than usual, because each mbt invocation OVERWRITES
    run_results.json.
    """
    showcase_stack.mbt(
        "build", "--target", TARGET, "--select", "tag:wide", "--anchor", ANCHOR, timeout=1800
    )
    return showcase_stack


def test_wide_panel_builds_and_gates_off_the_object_store(object_store) -> None:
    """The committed wide cadence, trained with no copy of the lake anywhere."""
    stack = object_store
    dataset = stack.result_for(DATASET_NODE)
    assert dataset["status"] == "success", dataset
    row_counts = _latest_materialization(stack)["row_counts"]
    assert row_counts.get("train", 0) > 0 and row_counts.get("test", 0) > 0, row_counts

    model = stack.result_for(MODEL_NODE)
    assert model["status"] == "success", model
    assert all(gate["passed"] for gate in model["gates"]), model["gates"]


def test_registered_model_is_namespaced_per_plane(object_store) -> None:
    """Both planes train the same spec; their versions must not collide.

    Without plane_suffix this plane would push versions into the lake plane's
    registered model, and champion resolution would silently start mixing
    models trained through different adapters.
    """
    stack = object_store
    url = f"{stack.mlflow_url()}/api/2.0/mlflow/registered-models/search?max_results=200"
    names = {entry["name"] for entry in stack.http_json(url).get("registered_models", [])}

    assert REGISTERED_MODEL in names, names
    registration = stack.result_for(MODEL_NODE)["registration"]
    assert registration["name"] == REGISTERED_MODEL, registration


def test_score_and_monitor_close_the_loop_off_the_object_store(object_store) -> None:
    """The serving leg with the object store as the source, not a copy of it.

    This is the leg `prod_score` cannot prove: it reads /workspace/lake_local,
    so a broken s3a read at score time would look identical to a working one.
    Here `mbt score` resolves the champion, materializes the scoring batch by
    reading s3a://mbt-lake, and stages the run as parquet under
    predictions_root; `mbt monitor` then reads the matured labels from the
    same object store and evaluates exactly once (ADR-21).
    """
    stack = object_store
    # --target is load-bearing on promote: the plane's registered name only
    # exists because of this target's plane_suffix.
    stack.mbt("promote", "--target", TARGET, "--model", REGISTERED_MODEL, "--to", "production")

    before = len(_prediction_runs(stack))
    stack.mbt("score", "--target", TARGET, "--select", "tag:wide", "--anchor", ANCHOR)
    scoring = stack.result_for(SCORING_NODE)
    assert scoring["status"] == "success", scoring
    assert all(m["passed"] for m in scoring["monitors"]), scoring["monitors"]

    runs = _prediction_runs(stack)
    assert len(runs) == before + 1, "scoring staged no new prediction run"
    sidecar = json.loads(runs[-1].read_text())
    assert sidecar["model_version"], sidecar
    assert sidecar["row_count"] > 0, sidecar

    # Ground truth matures after the scoring anchor; the pinned monitor anchor
    # is past it, so the run evaluates and its realized metrics clear the
    # cadence's floor - the same numbers the lake plane gets, since it is the
    # same data read a different way.
    stack.mbt("monitor", "--target", TARGET, "--select", "tag:wide", "--anchor", MONITOR_ANCHOR)
    evaluated = stack.result_for(SCORING_NODE)
    assert evaluated["status"] == "success", evaluated
    assert evaluated["metrics"]["pr_auc"] > 0.2, evaluated["metrics"]
    assert evaluated["metrics"]["roc_auc"] > 0.5, evaluated["metrics"]

    # Exactly-once (ADR-21): re-running the same anchor evaluates nothing. The
    # marker lives in the staged prediction run, which is the piece this plane
    # exercises through the spark adapter rather than the local one.
    proc = stack.mbt(
        "monitor", "--target", TARGET, "--select", "tag:wide", "--anchor", MONITOR_ANCHOR
    )
    assert "0 matured prediction runs" in proc.stdout + proc.stderr, (proc.stdout, proc.stderr)


def test_the_two_planes_agree_on_the_panel(object_store) -> None:
    """The claim the whole design rests on: same specs, same data, same panel.

    The object-store plane reads s3://mbt-lake through Spark; prod_score reads
    the DuckDB copy sync_lake.py mirrors out of the same bucket. The two are
    seeded from identical parquet, so the materialized training panels must
    agree on row counts. A divergence means a plane is silently reshaping the
    data - exactly what a second adapter risks, and what makes "just switch
    targets" either true or a lie.
    """
    stack = object_store

    # Rebuild each plane's panel here rather than reusing an earlier test's, so
    # this holds regardless of what ran before it. Only the dataset node is
    # selected, so neither rebuild pays for AutoML.
    stack.mbt("build", "--target", TARGET, "--select", DATASET, "--anchor", ANCHOR)
    object_store_counts = _latest_materialization(stack)["row_counts"]

    stack.sync_lake()
    stack.mbt(
        "build",
        "--target",
        "prod_score",
        "--select",
        DATASET,
        "--anchor",
        ANCHOR,
        "--deep-snapshot",
    )
    local_counts = _latest_materialization(stack)["row_counts"]

    assert object_store_counts == local_counts, (
        f"the planes disagree on the panel: seaweedfs={object_store_counts} "
        f"prod_score={local_counts}"
    )
