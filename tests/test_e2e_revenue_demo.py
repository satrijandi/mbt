"""End-to-end CLI tests over tests/fixtures/revenue_demo - the regression vertical.

The classification path is covered by test_e2e_churn_demo.py; this exercises
what regression changes: the regression metric set, an rmse (lower-is-better)
ceiling gate, and delayed ground-truth monitoring against a continuous target
(R2-4). Real XGBoost regression training in subprocesses, MLflow on sqlite.
"""

import json
from pathlib import Path

import pytest
from e2e_utils import REVENUE_ANCHOR, run_mbt

pytestmark = pytest.mark.e2e

MODEL = "model.revenue_demo.spend_regressor"
DATASET = "dataset.revenue_demo.spend_training_set"
SCORING = "scoring.revenue_demo.spend_scoring"


def _results(project: Path) -> dict[str, dict]:
    payload = json.loads((project / "target" / "run_results.json").read_text())
    return {r["unique_id"]: r for r in payload["results"]}


def test_regression_build_score_and_monitor(revenue_copy: Path) -> None:
    # ---- 1. build: dataset + regressor, regression metrics, rmse gate ----
    run_mbt(["build", "--anchor", REVENUE_ANCHOR], revenue_copy, timeout=600)
    results = _results(revenue_copy)
    assert set(results) == {MODEL, DATASET}
    assert all(r["status"] == "success" for r in results.values())

    model = results[MODEL]
    assert model["registration"]["version"] == "1"
    assert model["registration"]["stage"] == "staging"
    metrics = model["metrics"]
    # the regression metric set is computed (disjoint from the binary set)...
    assert set(metrics) >= {"rmse", "mae", "r2"}
    # ...and the model genuinely fits: a mean predictor scores rmse ~35, so
    # clearing the 12.0 ceiling with room to spare proves real signal capture.
    assert metrics["rmse"] < 12.0
    assert metrics["mae"] < metrics["rmse"]  # MAE <= RMSE always holds
    assert metrics["r2"] > 0.8
    assert "plan_type=enterprise" in model["slices"]  # per-segment error reported
    baseline_rmse = metrics["rmse"]

    # exact reproduction via --manifest (same seed, same data -> same metrics)
    run_mbt(["run", "--manifest", "target/manifest.json"], revenue_copy, timeout=600)
    assert _results(revenue_copy)[MODEL]["metrics"]["rmse"] == baseline_rmse

    # ---- 2. promote + batch score with the production champion ----
    run_mbt(["promote", "--model", "spend_regressor", "--to", "production"], revenue_copy)
    run_mbt(["score", "--anchor", REVENUE_ANCHOR], revenue_copy, timeout=600)
    scored = _results(revenue_copy)[SCORING]
    assert scored["status"] == "success"
    assert scored["metrics"]["rows_scored"] > 500
    assert {m["monitor"] for m in scored["monitors"]} >= {"feature_shift", "prediction_shift"}
    assert all(m["passed"] for m in scored["monitors"])
    run_dirs = sorted(
        p.parent for p in (revenue_copy / "predictions" / "spend_forecasts").glob("*/_SUCCESS")
    )
    assert len(run_dirs) == 1

    # ---- 3. matured ground truth: realized REGRESSION metrics, once (ADR-21) ----
    matured_anchor = "2026-07-20T00:00:00Z"  # scored_at + 14d maturity has passed
    run_mbt(["monitor", "--anchor", matured_anchor], revenue_copy, timeout=600)
    monitored = _results(revenue_copy)[SCORING]
    assert monitored["status"] == "success"
    assert "evaluated 1 of 1" in monitored["message"]
    assert monitored["metrics"]["rmse"] < 12.0  # realized rmse gate holds on fresh spend
    marker = json.loads((run_dirs[0] / "ground_truth.marker.json").read_text())
    assert marker["gates_passed"] is True and marker["matched_rows"] > 500

    run_mbt(["monitor", "--anchor", matured_anchor], revenue_copy, timeout=600)
    assert _results(revenue_copy)[SCORING]["message"] == "0 matured prediction runs to evaluate"


def test_rmse_ceiling_gate_blocks_registration_with_exit_2(revenue_copy: Path) -> None:
    # rmse is lower-is-better: a ceiling below the model's ~7.6 error must fail
    # the gate (exit 2) and block registration - the mirror of a pr_auc floor.
    run_mbt(
        ["build", "--anchor", REVENUE_ANCHOR, "--vars", "rmse_ceiling: 6.0"],
        revenue_copy,
        expect_exit=2,
        timeout=600,
    )
    model = _results(revenue_copy)[MODEL]
    assert model["status"] == "gate_failed"
    assert model["registration"] is None

    from mlflow.tracking import MlflowClient

    client = MlflowClient(tracking_uri=f"sqlite:///{revenue_copy}/mlflow.db")
    assert not client.search_model_versions("name = 'spend_regressor'")
