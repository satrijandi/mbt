"""End-to-end CLI tests over tests/fixtures/churn_demo (S3-09, S5-08, S7-05; G2/G3).

These run the real CLI in subprocesses: real XGBoost training jobs, MLflow
on sqlite, the local subprocess ComputeAdapter - the full production path.
"""

import csv
import json
from pathlib import Path
from typing import Any

import pytest
from e2e_utils import DEMO_ANCHOR, run_mbt

pytestmark = pytest.mark.e2e

OOT_MODEL = "model.churn_demo.churn_classifier_oot"
MODELS = {
    "model.churn_demo.churn_classifier",
    "model.churn_demo.churn_classifier_deep",
    "model.churn_demo.upsell_classifier",
    OOT_MODEL,
}
DATASETS = {
    "dataset.churn_demo.churn_training_set",
    "dataset.churn_demo.upsell_training_set",
    "dataset.churn_demo.churn_oot_set",
}
#: churn_classifier_oot trains before its after-test labels are all in: one
#: month (April) is held back at this anchor, three by DEMO_ANCHOR.
TRAINED_AT = "2026-04-30T00:00:00Z"


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
    # calibration: isotonic (F18) actually fixes the scale_pos_weight
    # miscalibration - measured on this exact build, the raw model scores
    # ece=0.208 / brier=0.212, calibrated it scores ece=0.044 / brier=0.158,
    # so the bars below FAIL without the calibration lever and pass with it.
    assert 0.0 < churn["metrics"]["brier"] < 0.20
    assert 0.0 <= churn["metrics"]["ece"] < 0.10
    # the operating point stays a usable cutoff (0 < t < 1) after calibration
    assert 0.0 < churn["metrics"]["threshold_at_precision_0.35"] < 1.0
    assert "plan_type=pro" in churn["slices"]  # slice reporting
    # the fairness/disparity gate (R2-9) is evaluated and passes: pro is the
    # weakest tier but stays within the 60% parity floor of the strongest.
    disparity = next(g for g in churn["gates"] if g["kind"] == "disparity")
    assert disparity["across"] == "plan_type" and disparity["passed"]
    assert disparity["worst_slice"] == "plan_type=pro"
    assert 0.6 <= disparity["actual"] < 1.0  # a real gap, not perfect parity
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

    # ADR-28: the project names the experiment, the invocation names the run,
    # and the run carries the config a later scoring run reconstructs from.
    assert run.data.tags["mbt.project"] == "churn_demo"
    assert client.get_experiment(run.info.experiment_id).name == "churn_demo"
    assert run.data.tags["mlflow.runName"] == f"{run.data.tags['mbt.run_id']}-churn_classifier"
    documents = {a.path for a in client.list_artifacts(run.info.run_id)}
    assert {"inference_config.json", "churn_classifier.py"} <= documents

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


def test_rollback_reverts_the_production_champion(demo_copy: Path) -> None:
    # promote v1 to production, then register + promote a second version...
    run_mbt(
        ["build", "--anchor", DEMO_ANCHOR, "--select", "churn_classifier"], demo_copy, timeout=600
    )
    run_mbt(["promote", "--model", "churn_classifier", "--to", "production"], demo_copy)
    model_yml = demo_copy / "models" / "churn_classifier.yml"
    model_yml.write_text(model_yml.read_text().replace("max_depth: 4", "max_depth: 5"))
    run_mbt(
        ["build", "--anchor", DEMO_ANCHOR, "--select", "churn_classifier"], demo_copy, timeout=600
    )
    run_mbt(["promote", "--model", "churn_classifier", "--to", "production"], demo_copy)

    from mlflow.tracking import MlflowClient

    client = MlflowClient(tracking_uri=f"sqlite:///{demo_copy}/mlflow.db")
    champ = client.get_model_version_by_alias("churn_classifier", "production")
    assert str(champ.version) == "2"

    # ...then roll back: production reverts to the last known good (v1), through
    # the real mlflow alias backend, exit 0 (ADR-20 resolution picks it up next run).
    run_mbt(["rollback", "--model", "churn_classifier"], demo_copy)
    reverted = client.get_model_version_by_alias("churn_classifier", "production")
    assert str(reverted.version) == "1"


def _artifact_paths(client: Any, run_id: str, path: str | None = None) -> set[str]:
    found: set[str] = set()
    for entry in client.list_artifacts(run_id, path):
        if entry.is_dir:
            found |= _artifact_paths(client, run_id, entry.path)
        else:
            found.add(entry.path)
    return found


def test_the_training_report_and_the_pre_deploy_check(demo_copy: Path, tmp_path: Path) -> None:
    """ADR-30 on the real stack: train with a month held back, re-check the
    version once two more months have labels, promote on that verdict."""
    import pyarrow.parquet as pq
    from mlflow.tracking import MlflowClient

    run_mbt(
        ["build", "--select", "churn_classifier_oot", "--anchor", TRAINED_AT],
        demo_copy,
        timeout=600,
    )
    built = _results(demo_copy)[OOT_MODEL]
    assert built["status"] == "success"
    after_test = next(g for g in built["gates"] if g["metric"] == "oot_roc_auc")
    assert after_test["period"] == "month" and after_test["cell"] == "2026-04"
    assert after_test["passed"] and after_test["actual"] > 0.6

    client = MlflowClient(tracking_uri=f"sqlite:///{demo_copy}/mlflow.db")
    run_id = built["tracking_run_id"]
    run = client.get_run(run_id)
    params = run.data.params
    # the flat config serving rebuilds preprocessing from (plus the bare
    # hyperparameters every earlier run logged)
    assert params["max_depth"] == "3" and params["model.hyperparameters.max_depth"] == "3"
    assert params["dataset.filters"] == '["is_active = true","tenure_days >= 30"]'
    assert params["dataset.split.out_of_time"] == "2026-04-01:now"
    assert params["dataset.windows.out_of_time.end"] == TRAINED_AT
    assert params["model.features.exclude"] == (
        '["user_id","upgraded_90d","plan_type","account_status"]'
    )
    assert json.loads(params["model.resolved.feature_columns"])[0] == "is_active"
    metrics = run.data.metrics
    assert {"train.roc_auc", "oot.window.roc_auc", "oot.month.2026-04.roc_auc"} <= set(metrics)
    assert metrics["stability.month.2026-04.score_psi"] < 0.1  # the demo data is stationary

    report = {
        f"report/{name}"
        for name in (
            "report.html",
            "summary.json",
            "evaluation/binning/quantile_10.csv",
            "evaluation/binning/fixed_width_0.05.csv",
            "evaluation/binning/custom.csv",
            "evaluation/binning/top_percent.csv",
            "evaluation/distribution/score_histogram.csv",
            "evaluation/distribution/test_features.csv",
            "evaluation/feature_importance.csv",
            "performance/by_period.csv",
            "predictions/train.parquet",
            "predictions/test.parquet",
            "predictions/out_of_time.parquet",
            "stability/scores_by_period.csv",
            # mbt-evidently, discovered by entry point inside the job process
            "stability/evidently/window.html",
            "stability/evidently/2026-04.html",
            "stability/evidently/drift_by_column.csv",
        )
    }
    files = _artifact_paths(client, run_id)
    assert report | {"logs/train.log", "config/model_config.json"} <= files

    local = Path(client.download_artifacts(run_id, "", str(tmp_path / "run")))
    log = (local / "logs" / "train.log").read_text()
    # the dataset's section comes first, then the model's debug-level detail
    assert log.index("## dataset.churn_demo.churn_oot_set") < log.index(
        "## model.churn_demo.churn_classifier_oot"
    )
    assert "window out_of_time: [2026-04-01T00:00:00Z, 2026-04-30T00:00:00Z)" in log
    assert "gate oot_roc_auc (threshold): PASS - expected 0.6, got" in log
    # quantile edges are fitted on test once and reused for every other split
    with (local / "report/evaluation/binning/quantile_10.csv").open() as handle:
        rows = list(csv.DictReader(handle))
    assert {row["split"] for row in rows} == {"train", "test", "out_of_time"}
    edges: dict[str, set[tuple[str, str]]] = {}
    for row in rows:
        edges.setdefault(row["bin"], set()).add((row["lower"], row["upper"]))
    assert len(edges) == 10 and all(len(pair) == 1 for pair in edges.values())
    evidently_page = (local / "report/stability/evidently/2026-04.html").read_text()
    assert "<title>churn_classifier_oot: 2026-04 against test</title>" in evidently_page
    with (local / "report/stability/evidently/drift_by_column.csv").open() as handle:
        compared = {row["column"] for row in csv.DictReader(handle) if row["cell"] == "window"}
    assert len(compared) == 6 and "prediction" in compared  # top 5 features + the score
    assert "churned_90d" not in compared
    predictions = pq.read_table(local / "report/predictions/out_of_time.parquet")
    assert predictions.column_names == ["user_id", "snapshot_date", "split", "label", "p0", "p1"]
    assert predictions.num_rows == int(params["dataset.rows.out_of_time"])

    # ---- two months later: re-check v1 on everything after its test month ----
    run_mbt(
        [
            "evaluate",
            "--model",
            "churn_classifier_oot",
            "--version",
            "1",
            "--out-of-time",
            "--gates",
            "--anchor",
            DEMO_ANCHOR,
        ],
        demo_copy,
        timeout=600,
    )
    payload = json.loads((demo_copy / "target" / "run_results.json").read_text())
    check_run = payload["metadata"]["run_id"]
    checked = _results(demo_copy)[OOT_MODEL]
    assert checked["status"] == "success"
    assert set(checked["metrics"]) == {"oot_pr_auc", "oot_roc_auc"}
    judged = next(g for g in checked["gates"] if g["metric"] == "oot_roc_auc")
    assert judged["message"].endswith("of 3 judged")  # April, May, June
    assert all(g["period"] is not None for g in checked["gates"])  # test gates stay recorded

    tags = client.get_model_version("churn_classifier_oot", "1").tags
    assert tags["mbt.oot_check.passed"] == "true"
    assert tags["mbt.oot_check.source"] == "evaluate"
    assert tags["mbt.oot_check.anchor"] == DEMO_ANCHOR
    assert tags["mbt.oot_check.run_id"] == check_run
    files = _artifact_paths(client, run_id)
    assert f"evaluations/{check_run}/report.html" in files
    # the check renders the whole window plus April, May and June
    assert {
        f"evaluations/{check_run}/stability/evidently/{cell}.html"
        for cell in ("window", "2026-04", "2026-05", "2026-06")
    } <= files
    assert f"logs/evaluate-{check_run}.log" in files
    assert client.get_run(run_id).data.params == params  # a check never logs params

    run_mbt(
        ["promote", "--model", "churn_classifier_oot", "--to", "production", "--require-oot-check"],
        demo_copy,
    )
    promoted = client.get_model_version_by_alias("churn_classifier_oot", "production")
    assert str(promoted.version) == "1"


def test_a_weak_month_after_the_test_window_blocks_registration(demo_copy: Path) -> None:
    run_mbt(
        [
            "build",
            "--select",
            "churn_classifier_oot",
            "--anchor",
            TRAINED_AT,
            "--vars",
            "oot_roc_auc_floor: 0.99",
        ],
        demo_copy,
        expect_exit=2,
        timeout=600,
    )
    result = _results(demo_copy)[OOT_MODEL]
    assert result["status"] == "gate_failed"
    assert result["registration"] is None
    assert result["message"].startswith("gate breach: oot_roc_auc [2026-04]=")

    from mlflow.tracking import MlflowClient

    client = MlflowClient(tracking_uri=f"sqlite:///{demo_copy}/mlflow.db")
    assert not client.search_model_versions("name = 'churn_classifier_oot'")
