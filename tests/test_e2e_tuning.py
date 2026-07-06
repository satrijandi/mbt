"""Optuna tuning E2E: capped, seeded, nested MLflow trials (S8-03)."""

import json
from pathlib import Path

import pytest
from e2e_utils import DEMO_ANCHOR, run_mbt

pytestmark = pytest.mark.e2e

TUNING_BLOCK = """
    tuning:
      engine: optuna
      n_trials: 10
      search_space:
        max_depth: {type: int, low: 2, high: 6}
        learning_rate: {type: loguniform, low: 0.02, high: 0.3}
      objective: {metric: pr_auc, direction: maximize}
"""


def test_optuna_tuning_capped_with_nested_trials(demo_copy: Path) -> None:
    model_yml = demo_copy / "models" / "upsell_classifier.yml"
    model_yml.write_text(
        model_yml.read_text().replace(
            "    evaluation:", TUNING_BLOCK.rstrip() + "\n    evaluation:"
        )
    )
    # dev target caps trials at 3 (max_tuning_trials in profiles.yml)
    run_mbt(
        ["build", "--anchor", DEMO_ANCHOR, "--select", "upsell_classifier"],
        demo_copy,
        timeout=600,
    )
    payload = json.loads((demo_copy / "target" / "run_results.json").read_text())
    result = {r["unique_id"]: r for r in payload["results"]}["model.churn_demo.upsell_classifier"]
    assert result["status"] == "success"

    from mlflow.tracking import MlflowClient

    client = MlflowClient(tracking_uri=f"sqlite:///{demo_copy}/mlflow.db")
    parent = client.get_run(result["tracking_run_id"])
    assert parent.data.tags["mbt.tuning.n_trials"] == "3"  # capped (FR-TUNE-04)
    best_params = json.loads(parent.data.tags["mbt.tuning.best_params"])
    assert set(best_params) == {"max_depth", "learning_rate"}
    # winning params override the spec's static values in the final fit
    assert result["metrics"]["pr_auc"] > 0

    experiment = client.get_experiment_by_name("mbt")
    children = client.search_runs(
        [experiment.experiment_id],
        filter_string=f"tags.\"mlflow.parentRunId\" = '{result['tracking_run_id']}'",
    )
    assert len(children) == 3  # nested trial runs (FR-TUNE-03)
