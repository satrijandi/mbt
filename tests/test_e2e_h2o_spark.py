"""Full mbt loop with the H2O AutoML and SparkML training adapters (e2e).

Same project, three models, three engines: proves adapter swap by spec edit
extends to JVM-backed, path-access adapters through real training jobs
(subprocess ComputeAdapter -> job -> path materialization -> train ->
gates -> MLflow registration).
"""

import json
import shutil
from pathlib import Path

import pytest
from e2e_utils import run_mbt

pytestmark = [
    pytest.mark.e2e,
    pytest.mark.skipif(shutil.which("java") is None, reason="H2O/Spark need a JVM"),
]

ANCHOR = "2026-06-30T00:00:00Z"


@pytest.fixture()
def jvm_project(tmp_path: Path) -> Path:
    project = tmp_path / "jvm_models"
    (project / "data").mkdir(parents=True)
    (project / "datasets").mkdir()
    (project / "models").mkdir()

    import pyarrow as pa
    import pyarrow.parquet as pq

    n = 1500
    from datetime import datetime, timedelta

    base = datetime(2026, 1, 1)
    signal = [((i * 37) % 100) / 100.0 for i in range(n)]
    noise = [((i * 7919) % 100) / 100.0 for i in range(n)]
    (project / "data" / "events").mkdir()
    pq.write_table(
        pa.table(
            {
                "user_id": list(range(n)),
                "event_ts": [base + timedelta(days=(i * 131) % 180) for i in range(n)],
                "f_signal": signal,
                "f_noise": noise,
                "converted": [
                    1 if s + 0.2 * z > 0.65 else 0 for s, z in zip(signal, noise, strict=True)
                ],
            }
        ),
        project / "data" / "events" / "part-000.parquet",
    )

    (project / "mbt_project.yml").write_text('name: jvm_models\nversion: "0.1.0"\n')
    (project / "profiles.yml").write_text(
        "jvm_models:\n"
        "  target: dev\n"
        "  outputs:\n"
        "    dev:\n"
        "      data: {adapter: local, config: {root: .}}\n"
        "      tracking: {adapter: mlflow, config: {uri: 'sqlite:///mlflow.db'}}\n"
        "      registry: {adapter: mlflow, config: {uri: 'sqlite:///mlflow.db'}}\n"
        "      compute: {adapter: local}\n"
        "      artifact_store: file://./target/artifacts\n"
        "      vars: {sample_fraction: 1.0, spark_master: 'local[2]'}\n"
    )
    (project / "sources.yml").write_text(
        "sources:\n  - name: lake\n    tables:\n"
        "      - name: events\n        path: data/events/*.parquet\n"
    )
    (project / "datasets" / "conversions.yml").write_text(
        "datasets:\n"
        "  - name: conversions\n"
        "    source: source('lake', 'events')\n"
        "    label: {column: converted}\n"
        "    sample_key: [user_id]\n"
        "    split:\n"
        "      strategy: temporal\n"
        "      time_column: event_ts\n"
        '      train: "-180d:-28d"\n'
        '      test: "-28d:now"\n'
    )

    def model(name: str, adapter: str, hyper: str) -> str:
        return (
            "models:\n"
            f"  - name: {name}\n"
            "    task: binary_classification\n"
            f"    adapter: {adapter}\n"
            "    owner: jvm@example.com\n"
            "    dataset: ref('conversions')\n"
            "    target: converted\n"
            "    features: {exclude: [user_id]}\n"
            f"    hyperparameters:\n{hyper}"
            "    evaluation:\n"
            "      protocol: {split: temporal}\n"
            "      metrics: [pr_auc, roc_auc]\n"
            "      gates:\n"
            "        - metric: roc_auc\n"
            "          threshold: 0.7\n"
            f"    registration: {{name: {name}}}\n"
            "    seed: 42\n"
        )

    (project / "models" / "conv_automl.yml").write_text(
        model(
            "conv_automl",
            "h2o_automl",
            "      max_models: 3\n      include_algos: [GLM, GBM]\n      nfolds: 0\n",
        )
    )
    (project / "models" / "conv_sparkml.yml").write_text(
        model("conv_sparkml", "spark", "      max_iter: 10\n      max_depth: 3\n")
    )
    return project


def test_h2o_and_sparkml_through_the_full_loop(jvm_project: Path) -> None:
    run_mbt(["build", "--anchor", ANCHOR], jvm_project, timeout=900)
    results = {
        r["unique_id"].split(".")[-1]: r
        for r in json.loads((jvm_project / "target/run_results.json").read_text())["results"]
    }
    automl = results["conv_automl"]
    sparkml = results["conv_sparkml"]

    assert automl["status"] == "success"
    assert automl["metrics"]["roc_auc"] > 0.7
    assert automl["artifact"]["format"] == "h2o_mojo"
    assert automl["registration"]["version"] == "1"

    assert sparkml["status"] == "success"
    assert sparkml["metrics"]["roc_auc"] > 0.7
    assert sparkml["artifact"]["format"] == "sparkml_zip"
    assert sparkml["registration"]["version"] == "1"

    # evaluate reloads the MOJO champion on fresh data - no retraining
    run_mbt(
        ["evaluate", "--model", "conv_automl", "--stage", "staging", "--gates", "--anchor", ANCHOR],
        jvm_project,
        timeout=600,
    )
    evaluated = {
        r["unique_id"].split(".")[-1]: r
        for r in json.loads((jvm_project / "target/run_results.json").read_text())["results"]
    }["conv_automl"]
    assert evaluated["status"] == "success"
    assert evaluated["gates"][0]["passed"]
