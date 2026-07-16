"""The monthly batch cadence on the DuckDB plane (SHOW-17).

The `tag:monthly` pipeline lives its whole life on the prod_score target:
train from the synced parquet lake via the LOCAL (DuckDB) adapter - no
cluster - then gate-verified promote, score the newest month-start batch
with the run-time champion, and evaluate its 30-day labels exactly once at
maturity. The pinned showcase anchors drive every step, so this module
composes with the daily/weekly modules on the shared session stack.
"""

import json

import pytest
from showcase_utils import ANCHOR, MONITOR_ANCHOR, SHOWCASE_MARKS

pytestmark = SHOWCASE_MARKS

MODEL = "churn_monthly_xgb"
MODEL_NODE = "model.churn_lake.churn_monthly_xgb"
SCORING_NODE = "scoring.churn_lake.monthly_retention_scoring"


def _sidecars(stack) -> list[dict]:
    root = stack.workspace / "lake_local" / "predictions" / "monthly_retention_scores"
    if not root.is_dir():
        return []
    return [
        json.loads(path.read_text())
        for path in sorted(root.glob("*/predictions.json"), key=lambda p: p.stat().st_mtime)
    ]


@pytest.fixture(scope="module")
def monthly(showcase_stack):
    showcase_stack.sync_lake()
    return showcase_stack


def test_monthly_train_promote_score_on_duckdb_plane(monthly) -> None:
    """Train, promote, and score the monthly batch without a cluster."""
    stack = monthly
    stack.mbt(
        "build",
        "--target",
        "prod_score",
        "--select",
        "tag:monthly",
        "--anchor",
        ANCHOR,
        "--deep-snapshot",
    )
    model = stack.result_for(MODEL_NODE)
    assert model["status"] == "success", model
    assert all(g["passed"] for g in model["gates"]), model["gates"]

    # Gate-verified promotion: the freshest gate-stamped staging version
    # becomes the champion (the CI module may have registered earlier ones
    # from the spark targets; this module's is the newest).
    stack.mbt("promote", "--model", MODEL, "--to", "production", timeout=300)
    from mlflow.tracking import MlflowClient

    client = MlflowClient(tracking_uri=stack.mlflow_url())
    champion = client.get_model_version_by_alias(MODEL, "production")
    assert champion.tags.get("mbt.gates_passed") == "true"

    # The month-start batch scores with the run-time champion (ADR-20) and
    # passes both shift monitors against the champion's training baseline.
    before = len(_sidecars(stack))
    stack.mbt(
        "score",
        "--target",
        "prod_score",
        "--select",
        "tag:monthly",
        "--anchor",
        ANCHOR,
        "--deep-snapshot",
    )
    scoring = stack.result_for(SCORING_NODE)
    assert scoring["status"] == "success", scoring
    sidecars = _sidecars(stack)
    assert len(sidecars) == before + 1
    sidecar = sidecars[-1]
    assert str(sidecar["model_version"]) == str(champion.version), sidecar
    assert sidecar["row_count"] > 0, sidecar


def test_monthly_ground_truth_matures_and_evaluates_once(monthly) -> None:
    """The 30-day labels are mature at the pinned monitor anchor (ADR-21)."""
    stack = monthly
    stack.mbt(
        "monitor",
        "--target",
        "prod_score",
        "--select",
        "tag:monthly",
        "--anchor",
        MONITOR_ANCHOR,
        "--deep-snapshot",
    )
    result = stack.result_for(SCORING_NODE)
    assert result["status"] == "success", result
    assert result["metrics"]["pr_auc"] > 0.15, result["metrics"]
    assert result["metrics"]["roc_auc"] > 0.5, result["metrics"]

    # Exactly-once: a second monitor run finds nothing left to evaluate and
    # succeeds as a no-op.
    proc = stack.mbt(
        "monitor",
        "--target",
        "prod_score",
        "--select",
        "tag:monthly",
        "--anchor",
        MONITOR_ANCHOR,
        "--deep-snapshot",
    )
    assert "0 matured prediction runs" in proc.stdout + proc.stderr, (proc.stdout, proc.stderr)
