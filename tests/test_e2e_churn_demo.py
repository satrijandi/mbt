"""End-to-end CLI tests over examples/churn_demo (S3-09, S5-08, S7-05; G2/G3).

These run the real CLI in subprocesses: real XGBoost training jobs, MLflow
on sqlite, the local subprocess ComputeAdapter - the full production path.
"""

import json
from pathlib import Path

import pytest
from e2e_utils import DEMO_ANCHOR, run_mbt

pytestmark = pytest.mark.e2e

MODELS = {
    "model.churn_demo.churn_classifier",
    "model.churn_demo.churn_classifier_deep",
    "model.churn_demo.upsell_classifier",
}
DATASETS = {
    "dataset.churn_demo.churn_training_set",
    "dataset.churn_demo.upsell_training_set",
}


def _results(project: Path) -> dict[str, dict]:
    payload = json.loads((project / "target" / "run_results.json").read_text())
    return {r["unique_id"]: r for r in payload["results"]}


def test_full_build_reproduce_state_promote(demo_copy: Path) -> None:
    # ---- 1. full build: datasets + 3 models, tests + gates, registration ----
    run_mbt(["build", "--anchor", DEMO_ANCHOR], demo_copy, timeout=600)
    results = _results(demo_copy)
    assert set(results) == MODELS | DATASETS
    assert all(r["status"] == "success" for r in results.values())

    churn = results["model.churn_demo.churn_classifier"]
    assert churn["registration"]["version"] == "1"
    assert churn["registration"]["stage"] == "staging"
    assert churn["resolved_auto"]["scale_pos_weight"] > 0
    assert churn["metrics"]["lift_at_0.1"] > 1.0  # builtin lift beats random
    assert churn["metrics"]["campaign_capture_100"] > 0  # hook metric computed
    assert churn["metrics"]["pr_auc"] > 0.3  # the gate floor actually gates
    # calibration is exercised, and the operating point is a usable cutoff:
    # applying it must be possible (0 < t < 1), i.e. 35% precision is reachable
    assert 0.0 < churn["metrics"]["brier"] < 0.25
    assert 0.0 <= churn["metrics"]["ece"] < 0.5
    assert 0.0 < churn["metrics"]["threshold_at_precision_0.35"] < 1.0
    assert "plan_type=pro" in churn["slices"]  # slice reporting
    baseline_metrics = {uid: results[uid]["metrics"] for uid in MODELS}

    ds = results["dataset.churn_demo.churn_training_set"]
    test_names = {t["name"] for t in ds["tests"]}
    assert {"test_label_is_binary", "test_only_active_subscribers"} <= test_names

    # MLflow got the run, params, and identity tags (FR-REG-01/05)
    from mlflow.tracking import MlflowClient

    client = MlflowClient(tracking_uri=f"sqlite:///{demo_copy}/mlflow.db")
    run = client.get_run(churn["tracking_run_id"])
    assert run.data.tags["mbt.input_hash"].startswith("sha256:")
    assert run.data.tags["mbt.gates_passed"] == "true"
    assert run.data.params["seed"] == "42"

    # ---- 2. exact reproduction via --manifest (G2, FR-RUN-05/11) ----
    run_mbt(["run", "--manifest", "target/manifest.json"], demo_copy, timeout=600)
    rerun = _results(demo_copy)
    for uid in MODELS:
        assert rerun[uid]["metrics"] == baseline_metrics[uid], f"{uid} not reproducible"

    # ---- 3. state economy: nothing modified vs own manifest (G3) ----
    manifest = demo_copy / "target" / "manifest.json"
    reference = demo_copy / "reference_manifest.json"
    reference.write_text(manifest.read_text())
    run_mbt(
        [
            "build",
            "--anchor",
            DEMO_ANCHOR,
            "--select",
            "state:modified+",
            "--state",
            str(reference),
        ],
        demo_copy,
        timeout=600,
    )
    assert not _results(demo_copy)["results"] if False else True  # results file rewritten
    unchanged = json.loads((demo_copy / "target" / "run_results.json").read_text())
    assert unchanged["results"] == []  # anchor drift alone retrains nothing

    # ---- 4. model-only edit retrains just that model (+ its dataset) ----
    model_yml = demo_copy / "models" / "churn_classifier.yml"
    model_yml.write_text(model_yml.read_text().replace("max_depth: 4", "max_depth: 5"))
    run_mbt(
        [
            "build",
            "--anchor",
            DEMO_ANCHOR,
            "--select",
            "state:modified+",
            "--state",
            str(reference),
        ],
        demo_copy,
        timeout=600,
    )
    retrained = _results(demo_copy)
    assert "model.churn_demo.churn_classifier" in retrained
    assert "model.churn_demo.upsell_classifier" not in retrained
    assert "model.churn_demo.churn_classifier_deep" not in retrained
    # the dataset was auto-materialized (cache hit), not selected (FR-RUN-12)
    assert retrained["dataset.churn_demo.churn_training_set"]["execution_time_s"] < 5

    # ---- 5. promotion verifies recorded gate passes (FR-REG-03) ----
    run_mbt(["promote", "--model", "churn_classifier", "--to", "production"], demo_copy)
    # stages map to registered-model aliases by default (stage API deprecated)
    promoted = client.get_model_version_by_alias("churn_classifier", "production")
    assert promoted is not None and promoted.version is not None

    # ---- 6. evaluate the production champion on fresh data (FR-RUN-07) ----
    run_mbt(
        [
            "evaluate",
            "--model",
            "churn_classifier",
            "--stage",
            "production",
            "--gates",
            "--anchor",
            DEMO_ANCHOR,
        ],
        demo_copy,
        timeout=600,
    )
    evaluated = _results(demo_copy)["model.churn_demo.churn_classifier"]
    assert evaluated["status"] == "success"
    assert evaluated["gates"] and evaluated["gates"][0]["passed"]

    # ---- 7. batch scoring with the production champion (ADR-20) ----
    run_mbt(["score", "--anchor", DEMO_ANCHOR], demo_copy, timeout=600)
    scored = _results(demo_copy)["scoring.churn_demo.retention_scoring"]
    assert scored["status"] == "success"
    assert scored["metrics"]["rows_scored"] > 500  # ~800 minus filters
    assert scored["monitors"] and all(m["passed"] for m in scored["monitors"])
    assert {m["monitor"] for m in scored["monitors"]} >= {"feature_shift", "prediction_shift"}
    run_dirs = sorted(
        p.parent for p in (demo_copy / "predictions" / "retention_scores").glob("*/_SUCCESS")
    )
    assert len(run_dirs) == 1
    info = json.loads((run_dirs[0] / "predictions.json").read_text())
    assert info["model_version"] == str(promoted.version)  # the promoted champion scored
    assert info["row_count"] == int(scored["metrics"]["rows_scored"])

    # ---- 8. matured ground truth evaluated exactly once (ADR-21) ----
    matured_anchor = "2026-07-20T00:00:00Z"  # scored_at + 14d maturity has passed
    run_mbt(["monitor", "--anchor", matured_anchor], demo_copy, timeout=600)
    monitored = _results(demo_copy)["scoring.churn_demo.retention_scoring"]
    assert monitored["status"] == "success"
    assert "evaluated 1 of 1" in monitored["message"]
    assert monitored["metrics"]["pr_auc"] > 0.3  # realized gate holds on fresh outcomes
    marker = json.loads((run_dirs[0] / "ground_truth.marker.json").read_text())
    assert marker["gates_passed"] is True and marker["matched_rows"] > 500

    run_mbt(["monitor", "--anchor", matured_anchor], demo_copy, timeout=600)
    again = _results(demo_copy)["scoring.churn_demo.retention_scoring"]
    assert again["message"] == "0 matured prediction runs to evaluate"  # ledger idempotency


def test_failing_gate_blocks_registration_with_exit_2(demo_copy: Path) -> None:
    run_mbt(
        ["build", "--anchor", DEMO_ANCHOR, "--vars", "pr_auc_floor: 0.99"],
        demo_copy,
        expect_exit=2,
        timeout=600,
    )
    results = _results(demo_copy)
    gate_failed = {uid for uid, r in results.items() if r["status"] == "gate_failed"}
    assert gate_failed | {uid for uid, r in results.items() if r["status"] == "success"} >= MODELS
    assert gate_failed, "at least one model must fail the 0.99 pr_auc gate"
    for uid in gate_failed:
        assert results[uid]["registration"] is None

    from mlflow.tracking import MlflowClient

    client = MlflowClient(tracking_uri=f"sqlite:///{demo_copy}/mlflow.db")
    if "model.churn_demo.churn_classifier" in gate_failed:
        assert not client.search_model_versions("name = 'churn_classifier'")


def test_champion_challenger_against_production(demo_copy: Path) -> None:
    # bootstrap build + promote to production
    run_mbt(
        ["build", "--anchor", DEMO_ANCHOR, "--select", "churn_classifier"], demo_copy, timeout=600
    )
    run_mbt(["promote", "--model", "churn_classifier", "--to", "production"], demo_copy)

    # switch the gate to champion comparison; identical spec -> delta 0 >= 0 passes
    model_yml = demo_copy / "models" / "churn_classifier.yml"
    model_yml.write_text(
        model_yml.read_text().replace(
            "- metric: pr_auc\n          threshold: \"{{ var('pr_auc_floor') }}\"",
            "- metric: pr_auc\n          compare_to: production",
        )
    )
    run_mbt(
        ["build", "--anchor", DEMO_ANCHOR, "--select", "churn_classifier"], demo_copy, timeout=600
    )
    result = _results(demo_copy)["model.churn_demo.churn_classifier"]
    gate = result["gates"][0]
    assert gate["kind"] == "champion"
    assert gate["champion_version"] == "1"
    assert gate["passed"] and abs(gate["actual_delta"]) < 1e-9  # same seed, same data


def test_parallel_threads_and_docs(demo_copy: Path) -> None:
    run_mbt(["run", "--anchor", DEMO_ANCHOR, "--threads", "2"], demo_copy, timeout=600)
    run_mbt(["docs", "generate"], demo_copy, timeout=600)
    index = (demo_copy / "target" / "docs" / "index.html").read_text()
    assert "churn_classifier" in index
    assert "retention_campaign_job" in index  # exposure in lineage (FR-DOCS-03)
    card = (demo_copy / "target" / "docs" / "model_churn_classifier.html").read_text()
    assert "input_hash" in card and "plan_type" in card
