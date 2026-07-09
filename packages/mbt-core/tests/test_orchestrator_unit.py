"""Unit tests for command orchestration (mbt/execute/orchestrator.py)."""

import json
import shutil
from pathlib import Path
from types import SimpleNamespace

import pytest
from exec_unit_helpers import DATASET_UID, MODEL_UID, make_options, recording_bus
from test_execution import invoke

from mbt.adapters.registry import AdapterRegistry
from mbt.exceptions import ConfigError, StateError
from mbt.execute.orchestrator import (
    _require_scoring_capability,
    prepare,
    run_evaluate,
)

OLD_GATE = "threshold: \"{{ var('default_threshold') }}\""


def _evaluate(project_dir: Path, registry: AdapterRegistry, *, manifest_path=None, **kwargs):
    opts = make_options(project_dir, command="evaluate", manifest_path=manifest_path)
    return run_evaluate(opts, registry=registry, **kwargs)


# -- --manifest execution (FR-RUN-11, ADR-19) ----------------------------------------


def test_manifest_execution_when_project_no_longer_parses(
    demo_project: Path, fake_registry: AdapterRegistry
) -> None:
    assert invoke(demo_project, fake_registry).exit_code() == 0
    manifest_file = demo_project / "target" / "manifest.json"
    (demo_project / "models" / "churn_model.yml").write_text("models: [\n")
    with recording_bus() as sink:
        results = invoke(demo_project, fake_registry, manifest_path=str(manifest_file))
    assert results.exit_code() == 0
    assert any("no longer parse cleanly" in m for m in sink.messages())


def test_manifest_drift_warns_and_executes_verbatim(
    demo_project: Path, fake_registry: AdapterRegistry
) -> None:
    assert invoke(demo_project, fake_registry).exit_code() == 0
    manifest_file = demo_project / "target" / "manifest.json"
    model_yml = demo_project / "models/churn_model.yml"
    model_yml.write_text(model_yml.read_text().replace(OLD_GATE, "threshold: 0.45"))
    with recording_bus() as sink:
        results = invoke(demo_project, fake_registry, manifest_path=str(manifest_file))
    assert results.exit_code() == 0
    assert any("disagree with the stored manifest" in m for m in sink.messages())


def test_manifest_freshness_check_survives_compile_failure(
    demo_project: Path, fake_registry: AdapterRegistry, monkeypatch: pytest.MonkeyPatch
) -> None:
    assert invoke(demo_project, fake_registry).exit_code() == 0
    manifest_file = demo_project / "target" / "manifest.json"

    def _boom(*args, **kwargs):
        raise ConfigError("unit compile failure")

    monkeypatch.setattr("mbt.execute.orchestrator.compile_project", _boom)
    with recording_bus() as sink:
        results = invoke(demo_project, fake_registry, manifest_path=str(manifest_file))
    assert results.exit_code() == 0
    assert any("could not re-render" in m for m in sink.messages())


# -- --state reference manifests (ADR-7) ---------------------------------------------


def test_state_reference_manifest_and_env_warning(
    demo_project: Path, fake_registry: AdapterRegistry
) -> None:
    assert invoke(demo_project, fake_registry).exit_code() == 0
    state_file = demo_project / "state_reference.json"
    shutil.copyfile(demo_project / "target" / "manifest.json", state_file)

    ok = invoke(demo_project, fake_registry, state=str(state_file))
    assert ok.exit_code() == 0

    payload = json.loads(state_file.read_text())
    payload["metadata"]["env_digest"] = "sha256:" + "0" * 64
    state_file.write_text(json.dumps(payload))
    with recording_bus() as sink:
        again = invoke(demo_project, fake_registry, state=str(state_file))
    assert again.exit_code() == 0
    assert any("environment digest differs" in m for m in sink.messages())


# -- scoring capability gate ----------------------------------------------------------


def test_require_scoring_capability_rejects_old_data_adapter() -> None:
    ctx = SimpleNamespace(
        data_adapter=object(),  # no build_scoring_input / open_predictions
        profiles=SimpleNamespace(target=SimpleNamespace(data=SimpleNamespace(adapter="ancient"))),
    )
    with pytest.raises(ConfigError, match="does not support"):
        _require_scoring_capability(ctx)


# -- mbt evaluate (FR-RUN-07) ---------------------------------------------------------


def test_run_evaluate_reevaluates_registered_version(
    demo_project: Path, fake_registry: AdapterRegistry
) -> None:
    assert invoke(demo_project, fake_registry).exit_code() == 0

    results = _evaluate(demo_project, fake_registry, model_name="churn_model")
    assert results.exit_code() == 0
    assert results.metadata.command == "evaluate"
    assert results.metadata.selector == "churn_model"
    by_id = {r.unique_id: r for r in results.results}
    assert by_id[DATASET_UID].status == "success"  # cache hit
    model = by_id[MODEL_UID]
    assert model.status == "success"
    assert model.metrics["pr_auc"] > 0.5
    assert model.gates == []  # apply_gates not requested
    stored = json.loads((demo_project / "target/run_results.json").read_text())
    assert stored["metadata"]["command"] == "evaluate"

    by_version = _evaluate(demo_project, fake_registry, model_name="churn_model", version="1")
    assert {r.unique_id: r for r in by_version.results}[MODEL_UID].status == "success"

    by_stage = _evaluate(demo_project, fake_registry, model_name="churn_model", stage="staging")
    assert {r.unique_id: r for r in by_stage.results}[MODEL_UID].status == "success"


def test_run_evaluate_unknown_model_errors(
    demo_project: Path, fake_registry: AdapterRegistry
) -> None:
    with pytest.raises(ConfigError, match="unknown model"):
        _evaluate(demo_project, fake_registry, model_name="mystery_model")


def test_run_evaluate_without_registered_version_errors(
    demo_project: Path, fake_registry: AdapterRegistry
) -> None:
    with pytest.raises(StateError, match="no registered version"):
        _evaluate(demo_project, fake_registry, model_name="churn_model")


def test_run_evaluate_apply_gates(demo_project: Path, fake_registry: AdapterRegistry) -> None:
    assert invoke(demo_project, fake_registry).exit_code() == 0

    passed = _evaluate(demo_project, fake_registry, model_name="churn_model", apply_gates=True)
    model = {r.unique_id: r for r in passed.results}[MODEL_UID]
    assert model.status == "success"
    assert model.gates and model.gates[0].passed

    # raise the bar: the registered artifact now fails the gate, no retraining
    model_yml = demo_project / "models/churn_model.yml"
    model_yml.write_text(model_yml.read_text().replace(OLD_GATE, "threshold: 0.99"))
    failed = _evaluate(demo_project, fake_registry, model_name="churn_model", apply_gates=True)
    assert failed.exit_code() == 2
    model = {r.unique_id: r for r in failed.results}[MODEL_UID]
    assert model.status == "gate_failed"
    assert model.gates and not model.gates[0].passed


def test_run_evaluate_job_error_lands_in_results(
    demo_project: Path, fake_registry: AdapterRegistry
) -> None:
    assert invoke(demo_project, fake_registry).exit_code() == 0
    for path in (demo_project / "target" / "artifacts").rglob("fake_model.json"):
        path.write_text("corrupted, not json")
    results = _evaluate(demo_project, fake_registry, model_name="churn_model")
    assert results.exit_code() == 1
    model = {r.unique_id: r for r in results.results}[MODEL_UID]
    assert model.status == "error"


def test_run_evaluate_dataset_error_skips_the_model(
    demo_project: Path, fake_registry: AdapterRegistry
) -> None:
    # compile a manifest without executing anything, then break the dataset
    prepare(make_options(demo_project), registry=fake_registry)
    manifest_file = demo_project / "target" / "manifest.json"
    payload = json.loads(manifest_file.read_text())
    payload["nodes"][DATASET_UID]["depends_on"] = []
    manifest_file.write_text(json.dumps(payload))

    results = _evaluate(
        demo_project, fake_registry, manifest_path=str(manifest_file), model_name="churn_model"
    )
    assert results.exit_code() == 1
    by_id = {r.unique_id: r for r in results.results}
    assert by_id[DATASET_UID].status == "error"
    assert "no source in the manifest" in by_id[DATASET_UID].message
    assert MODEL_UID not in by_id  # the model result is skipped entirely
