"""The showcase lifecycle, end to end (SHOW-03/04 + the serving differentiators).

One narrative over the compose stack, in deliberate order within this module:

1. DS inner loop: `mbt build --target dev` - Spark local[2] reads the s3a lake
   on SeaweedFS, H2O trains locally, the model registers to MLflow over HTTP
   with the MOJO in the S3 artifact store (integration items A2 + A3, live).
2. Cluster training: `mbt build --target prod` - pushdown on the standalone
   cluster, sparkling H2O inside the executors.
3. Gate-verified GitOps promotion via promotions.yml, pinned-version replay
   idempotency, and the unpinned-replay refusal.
4. Run-time champion resolution + prediction-store idempotency: same anchor
   overwrites (one run_key), a new anchor partitions.
5. Ground-truth monitoring: evaluated exactly once, realized-metric gate
   breach exits 2 (never 1).

Assertions read target/run_results.json from the shared workspace and the
registry through MLflow's HTTP API - the same surfaces a user would look at.
"""

from showcase_utils import ANCHOR, MONITOR_ANCHOR, SHOWCASE_MARKS

pytestmark = SHOWCASE_MARKS

AUTOML = "model.churn_lake.churn_automl"
BASELINE = "model.churn_lake.churn_baseline_xgb"
SCORING = "scoring.churn_lake.retention_scoring"
_state: dict = {}


def _mlflow_client(stack):
    from mlflow.tracking import MlflowClient

    return MlflowClient(tracking_uri=stack.mlflow_url())


def _predictions_root(stack):
    # The prediction store roots at the DATA ADAPTER root (prod_score's local
    # adapter root is /workspace/lake_local), not the project dir.
    return stack.workspace / "lake_local" / "predictions" / "retention_scores"


def _predictions_runs(stack) -> list:
    root = _predictions_root(stack)
    if not root.is_dir():
        return []
    return sorted(p.name for p in root.iterdir() if p.is_dir())


def test_dev_build_lake_to_registry(showcase_stack) -> None:
    """SHOW-03: dev target trains from the object-store lake and registers."""
    stack = showcase_stack
    stack.mbt("parse", timeout=300)
    stack.mbt("build", "--target", "dev", "--anchor", ANCHOR)

    automl = stack.result_for(AUTOML)
    assert automl["status"] == "success", automl
    assert automl["registration"]["stage"] == "staging", automl["registration"]
    _state["dev_version"] = int(automl["registration"]["version"])

    baseline = stack.result_for(BASELINE)
    assert baseline["status"] == "success", baseline

    # Registry state over HTTP: staging alias resolves to the new version and
    # carries the gate stamp that `mbt promote` verifies.
    client = _mlflow_client(stack)
    version = client.get_model_version_by_alias("churn_automl", "staging")
    assert int(version.version) == _state["dev_version"]
    assert version.tags.get("mbt.gates_passed") == "true"

    # The MOJO physically lives in the S3 artifact store (SeaweedFS).
    listing = stack.exec(
        "python",
        "-c",
        "import boto3\n"
        "s3 = boto3.client('s3')\n"
        "objs = s3.list_objects_v2(Bucket='mbt-artifacts', Prefix='churn_lake/')\n"
        "print(len(objs.get('Contents', [])))",
        workdir="/workspace",
    )
    assert int(listing.stdout.strip()) > 0, "no artifacts under s3://mbt-artifacts/churn_lake/"


def test_cluster_sparkling_training(showcase_stack) -> None:
    """SHOW-04: prod target - pushdown AND H2O training on the real cluster."""
    stack = showcase_stack
    stack.mbt(
        "build",
        "--target",
        "prod",
        "--select",
        "+churn_automl",
        "--anchor",
        ANCHOR,
        timeout=1800,
    )

    automl = stack.result_for(AUTOML)
    assert automl["status"] == "success", automl
    prod_version = int(automl["registration"]["version"])
    assert prod_version > _state["dev_version"]
    _state["prod_version"] = prod_version

    # The cluster actually did the work: the master saw our applications.
    master = stack.http_json(f"http://localhost:{stack.ports['SHOWCASE_SPARK_UI_PORT']}/json/")
    seen = len(master.get("completedapps", [])) + len(master.get("activeapps", []))
    assert seen >= 1, f"spark master saw no applications: {master}"

    client = _mlflow_client(stack)
    version = client.get_model_version_by_alias("churn_automl", "staging")
    assert int(version.version) == prod_version


def test_gitops_promotion_pinned_replay_and_refusal(showcase_stack) -> None:
    """SHOW-10: gate-verified promotion; pinned replay idempotent; unpinned refused."""
    stack = showcase_stack
    version = _state["prod_version"]
    promotions = stack.workspace / "project" / "promotions.yml"
    promotions.write_text(
        f"promotions:\n  - model: churn_automl\n    to: production\n    version: '{version}'\n"
    )
    first = stack.mbt("promote", "--from-file", "promotions.yml", timeout=300)
    assert "applied 1 promotion" in first.stdout, first.stdout

    client = _mlflow_client(stack)
    champion = client.get_model_version_by_alias("churn_automl", "production")
    assert int(champion.version) == version

    # Pinned replay is idempotent (GitOps re-runs must be safe).
    stack.mbt("promote", "--from-file", "promotions.yml", timeout=300)
    champion = client.get_model_version_by_alias("churn_automl", "production")
    assert int(champion.version) == version

    # Promotion vacated staging (aliases are exclusive per version), so an
    # UNPINNED entry now has nothing to resolve: hard error, exit 1 - the
    # documented reason promotions.yml always pins version.
    promotions.write_text("promotions:\n  - model: churn_automl\n    to: production\n")
    unpinned = stack.mbt("promote", "--from-file", "promotions.yml", expect_exit=1, timeout=300)
    assert "staging" in (unpinned.stdout + unpinned.stderr)
    promotions.write_text("promotions: []\n")


def test_champion_scoring_and_prediction_idempotency(showcase_stack) -> None:
    """SHOW-11/12: score with the run-time champion; same anchor overwrites."""
    stack = showcase_stack
    stack.sync_lake()

    stack.mbt(
        "score",
        "--target",
        "prod_score",
        "--select",
        "tag:daily",
        "--anchor",
        ANCHOR,
        "--deep-snapshot",
    )
    scoring = stack.result_for(SCORING)
    assert scoring["status"] == "success", scoring
    runs_after_first = _predictions_runs(stack)
    assert len(runs_after_first) == 1, runs_after_first

    # Champion resolved at run time == the promoted version (ADR-20): the
    # prediction sidecar records which model version actually scored.
    import json

    sidecar = json.loads(
        (_predictions_root(stack) / runs_after_first[0] / "predictions.json").read_text()
    )
    assert str(sidecar["model_version"]) == str(_state["prod_version"]), sidecar
    assert sidecar["scored_at"].startswith("2026-06-30"), sidecar

    # Same anchor again: the run_key is anchor-independent by construction
    # (input hash + windows + champion version), so this OVERWRITES.
    stack.mbt(
        "score",
        "--target",
        "prod_score",
        "--select",
        "tag:daily",
        "--anchor",
        ANCHOR,
        "--deep-snapshot",
    )
    assert _predictions_runs(stack) == runs_after_first

    # A different anchor resolves different windows: new partition.
    stack.mbt(
        "score",
        "--target",
        "prod_score",
        "--select",
        "tag:daily",
        "--anchor",
        "2026-07-01T00:00:00Z",
        "--deep-snapshot",
    )
    runs_after_third = _predictions_runs(stack)
    assert len(runs_after_third) == 2, runs_after_third

    for run in runs_after_third:
        assert (_predictions_root(stack) / run / "_SUCCESS").exists(), (
            f"half-written prediction run {run}"
        )


def test_ground_truth_monitoring_exactly_once_and_exit_2(showcase_stack) -> None:
    """SHOW-13: monitor evaluates once; a realized-gate breach exits 2, not 1."""
    stack = showcase_stack

    first = stack.mbt(
        "monitor",
        "--target",
        "prod_score",
        "--select",
        "tag:daily",
        "--anchor",
        MONITOR_ANCHOR,
        "--deep-snapshot",
    )
    scoring = stack.result_for(SCORING)
    assert scoring["status"] == "success", scoring
    assert "evaluated" in (scoring.get("message") or ""), scoring
    assert first.returncode == 0

    # Exactly-once: the same anchor finds nothing left to evaluate.
    second = stack.mbt(
        "monitor",
        "--target",
        "prod_score",
        "--select",
        "tag:daily",
        "--anchor",
        MONITOR_ANCHOR,
        "--deep-snapshot",
    )
    assert "0 matured prediction runs" in (second.stdout + second.stderr), (
        second.stdout + second.stderr
    )

    # A fresh prediction run + an impossible realized-metric floor: the gate
    # verdict is deterministic quality failure (exit 2), never a hard error.
    stack.mbt(
        "score",
        "--target",
        "prod_score",
        "--select",
        "tag:daily",
        "--anchor",
        "2026-07-02T00:00:00Z",
        "--deep-snapshot",
    )
    breached = stack.mbt(
        "monitor",
        "--target",
        "prod_score",
        "--select",
        "tag:daily",
        "--anchor",
        MONITOR_ANCHOR,
        "--deep-snapshot",
        "--vars",
        "pr_auc_floor: 0.99",
        expect_exit=2,
    )
    scoring = stack.result_for(SCORING)
    assert scoring["status"] == "monitor_failed", scoring
    assert breached.returncode == 2
