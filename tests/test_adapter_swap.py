"""G4 extensibility proof: swap xgboost -> lightgbm by editing only the spec
(adapter + hyperparameters); zero mbt-core changes (S8-02)."""

import json
from pathlib import Path

import pytest
from e2e_utils import DEMO_ANCHOR, run_mbt

pytestmark = pytest.mark.e2e

LIGHTGBM_SPEC = """
models:
  - name: churn_classifier
    description: "same model, different adapter (G4 proof)"
    task: binary_classification
    adapter: lightgbm
    owner: growth-ds@example.com
    tags: [churn, weekly]
    dataset: ref('churn_training_set')
    target: churned_90d
    features:
      include: ["*"]
      exclude: [user_id, upgraded_90d, plan_type]
    hyperparameters:
      num_leaves: 15
      learning_rate: 0.1
      n_estimators: 120
      scale_pos_weight: "{{ auto }}"
    evaluation:
      protocol: {split: temporal}
      metrics: [pr_auc, roc_auc]
      gates:
        - metric: pr_auc
          threshold: "{{ var('pr_auc_floor') }}"
    registration:
      name: churn_classifier
      stage_on_pass: staging
    seed: 42
"""


def test_switching_adapter_only_touches_the_spec(demo_copy: Path) -> None:
    (demo_copy / "models" / "churn_classifier.yml").write_text(LIGHTGBM_SPEC)
    (demo_copy / "models" / "churn_classifier.py").unlink()  # hooks were xgboost-demo only

    run_mbt(
        ["build", "--anchor", DEMO_ANCHOR, "--select", "churn_classifier"],
        demo_copy,
        timeout=600,
    )
    payload = json.loads((demo_copy / "target" / "run_results.json").read_text())
    result = {r["unique_id"]: r for r in payload["results"]}["model.churn_demo.churn_classifier"]
    assert result["status"] == "success"
    assert result["artifact"]["format"] == "lightgbm_json"
    assert result["resolved_auto"]["scale_pos_weight"] > 0
    assert result["registration"]["version"] == "1"


SKLEARN_SPEC = """
models:
  - name: churn_classifier
    description: "same model, sklearn adapter (F-1)"
    task: binary_classification
    adapter: sklearn
    owner: growth-ds@example.com
    tags: [churn, weekly]
    dataset: ref('churn_training_set')
    target: churned_90d
    features:
      include: ["*"]
      exclude: [user_id, upgraded_90d, plan_type]
    hyperparameters:
      estimator: hist_gradient_boosting
      max_iter: 60
      learning_rate: 0.1
      class_weight: "{{ auto }}"
    evaluation:
      protocol: {split: temporal}
      metrics: [pr_auc, roc_auc]
      gates:
        - metric: pr_auc
          threshold: "{{ var('pr_auc_floor') }}"
    registration:
      name: churn_classifier
      stage_on_pass: staging
    seed: 42
"""


def test_switching_to_sklearn_only_touches_the_spec(demo_copy: Path) -> None:
    """The same G4 proof for mbt-sklearn: a third adapter, still zero core
    changes, driven through the real CLI rather than the compliance harness."""
    (demo_copy / "models" / "churn_classifier.yml").write_text(SKLEARN_SPEC)
    (demo_copy / "models" / "churn_classifier.py").unlink()  # hooks were xgboost-demo only

    run_mbt(
        ["build", "--anchor", DEMO_ANCHOR, "--select", "churn_classifier"],
        demo_copy,
        timeout=600,
    )
    payload = json.loads((demo_copy / "target" / "run_results.json").read_text())
    result = {r["unique_id"]: r for r in payload["results"]}["model.churn_demo.churn_classifier"]
    assert result["status"] == "success"
    assert result["artifact"]["format"] == "sklearn_joblib"
    assert result["resolved_auto"]["class_weight"] == "balanced"
    assert result["registration"]["version"] == "1"
    assert result["metrics"]["roc_auc"] > 0.5
