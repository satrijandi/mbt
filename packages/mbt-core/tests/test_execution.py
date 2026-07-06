"""End-to-end execution over the fake adapters (S3-05, S5-01..S5-06)."""

import json
from pathlib import Path

from core_helpers import TEST_ANCHOR, write
from mbt_testing import FakeRegistryAdapter

from mbt.adapters.registry import AdapterRegistry
from mbt.execute.orchestrator import InvocationOptions, run_command

DS = "dataset.demo.churn_training"
MODEL = "model.demo.churn_model"


def invoke(project_dir: Path, registry: AdapterRegistry, command: str = "run", **kwargs):
    opts = InvocationOptions(
        command=command,
        project_dir=project_dir,
        anchor=TEST_ANCHOR,
        **kwargs,
    )
    return run_command(opts, registry=registry)


def test_run_trains_and_registers(demo_project: Path, fake_registry: AdapterRegistry) -> None:
    results = invoke(demo_project, fake_registry)
    assert results.exit_code() == 0
    by_id = {r.unique_id: r for r in results.results}
    assert by_id[DS].status == "success"
    model = by_id[MODEL]
    assert model.status == "success"
    assert model.metrics["pr_auc"] > 0.4
    assert model.gates and model.gates[0].passed
    assert model.registration is not None and model.registration.version == "1"
    assert model.registration.stage == "staging"
    assert model.artifact is not None and model.artifact.uri.startswith("file://")
    assert model.tracking_run_id

    # tracking run persisted with mbt identity tags
    tracking_file = demo_project / "target/fake_tracking" / f"{model.tracking_run_id}.json"
    payload = json.loads(tracking_file.read_text())
    assert payload["tags"]["mbt.input_hash"].startswith("sha256:")
    assert payload["tags"]["mbt.gates_passed"] == "true"
    assert payload["status"] == "FINISHED"

    # run_results.json written and loadable
    stored = json.loads((demo_project / "target/run_results.json").read_text())
    assert stored["metadata"]["command"] == "run"
    assert {r["unique_id"] for r in stored["results"]} == {DS, MODEL}


def test_failing_gate_blocks_registration_exit_2(
    demo_project: Path, fake_registry: AdapterRegistry
) -> None:
    model_yml = demo_project / "models/churn_model.yml"
    model_yml.write_text(
        model_yml.read_text().replace("fake_metric_value: 0.61", "fake_metric_value: 0.30")
    )
    results = invoke(demo_project, fake_registry)
    assert results.exit_code() == 2
    model = {r.unique_id: r for r in results.results}[MODEL]
    assert model.status == "gate_failed"
    assert model.registration is None
    registry_dir = demo_project / "target/fake_registry"
    assert not (registry_dir / "churn_model.json").exists()


def test_champion_challenger_gate(demo_project: Path, fake_registry: AdapterRegistry) -> None:
    # 1) bootstrap: champion gate passes with a warning when none exists
    model_yml = demo_project / "models/churn_model.yml"
    old_gate = (
        "gates:\n        - metric: pr_auc\n          threshold: \"{{ var('default_threshold') }}\""
    )
    new_gate = (
        "gates:\n        - metric: pr_auc\n"
        "          compare_to: production\n          min_delta: 0.005"
    )
    model_yml.write_text(model_yml.read_text().replace(old_gate, new_gate))
    first = invoke(demo_project, fake_registry)
    assert first.exit_code() == 0
    gate = {r.unique_id: r for r in first.results}[MODEL].gates[0]
    assert gate.passed and gate.champion_version is None

    # 2) promote v1 to production, then a *worse* challenger must fail
    registry_adapter = FakeRegistryAdapter({"root": str(demo_project / "target/fake_registry")})
    from mbt.contracts import Stage

    v1 = registry_adapter.get_version("churn_model", "1")
    assert v1 is not None
    registry_adapter.transition(v1, Stage.PRODUCTION)

    model_yml.write_text(
        model_yml.read_text().replace("fake_metric_value: 0.61", "fake_metric_value: 0.55")
    )
    second = invoke(demo_project, fake_registry)
    assert second.exit_code() == 2
    gate = {r.unique_id: r for r in second.results}[MODEL].gates[0]
    assert not gate.passed
    assert gate.champion_version == "1"
    assert gate.champion_value is not None and gate.actual_delta < 0

    # 3) a better challenger passes and registers version 3
    model_yml.write_text(
        model_yml.read_text().replace("fake_metric_value: 0.55", "fake_metric_value: 0.75")
    )
    third = invoke(demo_project, fake_registry)
    assert third.exit_code() == 0
    model = {r.unique_id: r for r in third.results}[MODEL]
    assert model.gates[0].passed and model.gates[0].actual_delta >= 0.005
    assert model.registration is not None


def test_selection_governs_training_not_data(
    demo_project: Path, fake_registry: AdapterRegistry
) -> None:
    """FR-RUN-12: selecting only the model auto-materializes its dataset."""
    results = invoke(demo_project, fake_registry, select=["churn_model"])
    by_id = {r.unique_id: r for r in results.results}
    assert by_id[MODEL].status == "success"
    assert by_id[DS].status == "success"  # auto-materialized, cold cache

    # warm cache: dataset build is a cache hit (still reported success)
    again = invoke(demo_project, fake_registry, select=["churn_model"])
    assert again.exit_code() == 0


def test_dataset_check_failure_fails_downstream(
    demo_project: Path, fake_registry: AdapterRegistry
) -> None:
    ds_yml = demo_project / "datasets/churn_training.yml"
    ds_yml.write_text(
        ds_yml.read_text().replace(
            "checks: [class_balance_report]",
            "checks: [class_balance_report, {not_null: {columns: [nonexistent_column]}}]",
        )
    )
    results = invoke(demo_project, fake_registry, command="build")
    assert results.exit_code() == 2
    by_id = {r.unique_id: r for r in results.results}
    assert by_id[DS].status == "test_failed"
    assert by_id[MODEL].status == "skipped"


def test_python_data_tests_run_in_build(demo_project: Path, fake_registry: AdapterRegistry) -> None:
    write(
        demo_project / "tests/test_no_leakage.py",
        """
        # mbt: select=churn_training
        from mbt.contracts import TestResult

        def test_has_rows(dataset, spec):
            return TestResult(name="test_has_rows", passed=dataset.num_rows > 0)

        def test_label_is_binary(dataset, spec):
            values = set(dataset.column(spec.label.column).to_pylist())
            return TestResult(
                name="test_label_is_binary",
                passed=values <= {0, 1},
                message=f"classes: {sorted(values)}",
            )
        """,
    )
    results = invoke(demo_project, fake_registry, command="build")
    assert results.exit_code() == 0
    ds = {r.unique_id: r for r in results.results}[DS]
    test_names = {t.name for t in ds.tests}
    assert {"test_has_rows", "test_label_is_binary"} <= test_names

    # 'mbt run' does not execute python data tests (checks only)
    run_results = invoke(demo_project, fake_registry, command="run")
    ds_run = {r.unique_id: r for r in run_results.results}[DS]
    assert "test_has_rows" not in {t.name for t in ds_run.tests}


def test_auto_resolution_lands_in_results(
    demo_project: Path, fake_registry: AdapterRegistry
) -> None:
    model_yml = demo_project / "models/churn_model.yml"
    model_yml.write_text(
        model_yml.read_text().replace(
            "max_depth: 4",
            'max_depth: 4\n      scale_pos_weight: "{{ auto }}"',
        )
    )
    results = invoke(demo_project, fake_registry)
    model = {r.unique_id: r for r in results.results}[MODEL]
    assert model.status == "success"
    assert "scale_pos_weight" in model.resolved_auto
    assert float(model.resolved_auto["scale_pos_weight"]) > 0
