"""The wide multi-table cadence (SHOW-19, ADR-22).

The realistic churn shape end to end: a population spine with the entity
crosswalk, three feature histories joined by DIFFERENT keys (transactions
only reach the panel through the population's safe_id), the label joined
from one calendar month after each snapshot, LightGBM feature selection as
a committed reviewable diff, sparkling H2O AutoML on the selected columns,
and the population-form scoring input with shift monitors and ground truth.

Sparkling lives here and in the lifecycle module only (flake isolation):
a cluster hiccup fails this module without poisoning the CI/promotion/
idempotency assertions.
"""

import json

import pyarrow.parquet as pq
import pytest
from showcase_utils import ANCHOR, MONITOR_ANCHOR, SHOWCASE_DIR, SHOWCASE_MARKS

pytestmark = SHOWCASE_MARKS

DATASET_NODE = "dataset.churn_lake.wide_churn_training"
PROBE_NODE = "model.churn_lake.churn_wide_probe"
AUTOML = "churn_wide_automl"
AUTOML_NODE = "model.churn_lake.churn_wide_automl"
SCORING_NODE = "scoring.churn_lake.wide_retention_scoring"


@pytest.fixture(scope="module")
def wide(showcase_stack):
    showcase_stack.sync_lake()
    return showcase_stack


def _sidecars(stack) -> list[dict]:
    root = stack.workspace / "lake_local" / "predictions" / "wide_retention_scores"
    if not root.is_dir():
        return []
    return [
        json.loads(path.read_text())
        for path in sorted(root.glob("*/predictions.json"), key=lambda p: p.stat().st_mtime)
    ]


def test_probe_selects_the_committed_feature_list(wide) -> None:
    """LightGBM importance over the pushdown multi-table join reproduces the
    committed include list exactly (exact-determinism tier + ADR-22 joins)."""
    stack = wide
    stack.mbt("build", "--target", "dev", "--select", "churn_wide_probe", "--anchor", ANCHOR)
    probe = stack.result_for(PROBE_NODE)
    assert probe["status"] == "success", probe
    assert all(g["passed"] for g in probe["gates"]), probe["gates"]
    # the two engineered churn drivers dominate the ~65-column importance
    importance = probe["feature_importance"]
    top = sorted(importance, key=importance.get, reverse=True)[:4]
    assert {"login_days_30d", "txn_cnt_30d"} <= set(top), top

    model_file = stack.workspace / "project" / "models" / "churn_wide_automl.yml"
    committed = (SHOWCASE_DIR / "project" / "models" / "churn_wide_automl.yml").read_text()
    stack.exec("python", "scripts/select_features.py")
    assert model_file.read_text() == committed, (
        "select_features.py produced a different top-K than the committed list"
    )


def test_panel_sampling_is_reproducible_and_monotone(wide) -> None:
    """`--vars sample_fraction` hash-samples whole customers via sample_key,
    pushed down into the Spark query (dev target); each fraction partitions
    its own materialization key, and smaller fractions are subsets of
    larger ones. Reads every key dir instead of chasing mtimes: a repeated
    fraction cache-hits and rewrites nothing."""
    stack = wide

    def build_at(fraction: float) -> None:
        stack.mbt(
            "build",
            "--target",
            "dev",
            "--select",
            "wide_churn_training",
            "--anchor",
            ANCHOR,
            "--vars",
            f"sample_fraction: {fraction}",
        )
        assert stack.result_for(DATASET_NODE)["status"] == "success"

    build_at(1.0)
    build_at(0.5)
    build_at(0.2)
    root = stack.workspace / "project" / "target" / "datasets" / "wide_churn_training"
    customer_sets = [
        set(pq.read_table(path, columns=["customer_id"]).column("customer_id").to_pylist())
        for path in root.glob("*/train.parquet")
    ]
    assert len(customer_sets) == 3  # one materialization key per fraction
    fifth, half, full = sorted(customer_sets, key=len)
    assert 0 < len(fifth) < len(half) < len(full)
    assert fifth <= half <= full  # threshold hashing: subsets, not resamples


def test_wide_automl_trains_on_the_cluster(wide) -> None:
    """Sparkling H2O AutoML on the selected top-K of the wide join (prod)."""
    stack = wide
    stack.mbt(
        "build",
        "--target",
        "prod",
        "--select",
        "churn_wide_automl",
        "--anchor",
        ANCHOR,
        timeout=1800,
    )
    model = stack.result_for(AUTOML_NODE)
    assert model["status"] == "success", model
    assert all(g["passed"] for g in model["gates"]), model["gates"]
    assert model["registration"], model

    stack.mbt("promote", "--model", AUTOML, "--to", "production", timeout=300)
    from mlflow.tracking import MlflowClient

    client = MlflowClient(tracking_uri=stack.mlflow_url())
    champion = client.get_model_version_by_alias(AUTOML, "production")
    assert champion.tags.get("mbt.gates_passed") == "true"


def test_population_scoring_input_and_monitors(wide) -> None:
    """The newest cohort scores through the population-form input (per-table
    keys at scoring time) and passes both shift monitors."""
    stack = wide
    before = len(_sidecars(stack))
    stack.mbt(
        "score",
        "--target",
        "prod_score",
        "--select",
        "tag:wide",
        "--anchor",
        ANCHOR,
        "--deep-snapshot",
    )
    scoring = stack.result_for(SCORING_NODE)
    assert scoring["status"] == "success", scoring
    sidecars = _sidecars(stack)
    assert len(sidecars) == before + 1
    assert sidecars[-1]["row_count"] > 0, sidecars[-1]


def test_ground_truth_matures_and_evaluates_once(wide) -> None:
    """The cohort's one-month-later outcomes evaluate exactly once (ADR-21)."""
    stack = wide
    stack.mbt(
        "monitor",
        "--target",
        "prod_score",
        "--select",
        "tag:wide",
        "--anchor",
        MONITOR_ANCHOR,
        "--deep-snapshot",
    )
    result = stack.result_for(SCORING_NODE)
    assert result["status"] == "success", result
    assert result["metrics"]["pr_auc"] > 0.15, result["metrics"]
    assert result["metrics"]["roc_auc"] > 0.5, result["metrics"]

    proc = stack.mbt(
        "monitor",
        "--target",
        "prod_score",
        "--select",
        "tag:wide",
        "--anchor",
        MONITOR_ANCHOR,
        "--deep-snapshot",
    )
    assert "0 matured prediction runs" in proc.stdout + proc.stderr, (proc.stdout, proc.stderr)


def test_evidently_report_renders(wide) -> None:
    """The DS-facing Evidently drift report renders from the same
    materializations mbt's native monitors enforce on."""
    stack = wide
    stack.exec("python", "scripts/evidently_report.py")
    report = stack.workspace / "project" / "drift_report.html"
    assert report.is_file() and report.stat().st_size > 100_000
