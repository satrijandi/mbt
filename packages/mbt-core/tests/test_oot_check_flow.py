"""The pre-deploy check, ``mbt evaluate --out-of-time`` (ADR-30), and the
promotion guard that reads its verdict."""

import json
from datetime import timedelta
from pathlib import Path
from typing import Any

import pytest
from core_helpers import TEST_ANCHOR
from exec_unit_helpers import MODEL_UID, recording_bus
from report_unit_helpers import reporting_project

from mbt.adapters.registry import AdapterRegistry
from mbt.contracts import ArtifactRef, ModelVersion, Stage
from mbt.exceptions import StateError
from mbt.execute.orchestrator import InvocationOptions, run_command, run_evaluate
from mbt.promote import promote_model, rollback_model

TRAINED_AT = TEST_ANCHOR - timedelta(days=20)


def _build(project: Path, registry: AdapterRegistry, anchor: Any = TRAINED_AT) -> None:
    results = run_command(
        InvocationOptions(command="build", project_dir=project, anchor=anchor), registry=registry
    )
    model = next(r for r in results.results if r.unique_id == MODEL_UID)
    assert model.status == "success", model.message


def _check(project: Path, registry: AdapterRegistry, **kwargs: Any) -> Any:
    kwargs.setdefault("apply_gates", True)
    return run_evaluate(
        InvocationOptions(
            command="evaluate", project_dir=project, anchor=kwargs.pop("anchor", TEST_ANCHOR)
        ),
        model_name="churn_model",
        version=kwargs.pop("version", "1"),
        out_of_time=True,
        registry=registry,
        **kwargs,
    )


def _versions(project: Path) -> list[dict[str, Any]]:
    return json.loads((project / "target" / "fake_registry" / "churn_model.json").read_text())


def _run_payload(project: Path) -> dict[str, Any]:
    (path,) = (project / "target" / "fake_tracking").glob("*.json")
    return json.loads(path.read_text())


def test_the_check_reports_on_the_training_run_and_records_its_verdict(
    demo_project: Path, fake_registry: AdapterRegistry
) -> None:
    reporting_project(demo_project)
    _build(demo_project, fake_registry)
    trained = _versions(demo_project)[0]["tags"]
    assert trained["mbt.oot_check.source"] == "build"

    with recording_bus() as sink:
        results = _check(demo_project, fake_registry)
    assert [r.status for r in results.results] == ["success", "success"]
    model = results.results[1]
    assert set(model.metrics) == {"oot_pr_auc", "oot_roc_auc"}  # the window, not test
    assert (
        "pre-deploy check on churn_model v1 passed; recorded as mbt.oot_check.passed=true"
        in sink.messages()
    )
    assert "label balance: no train split to report" in sink.messages()
    assert any(g.metric == "oot_roc_auc" for g in model.gates)
    assert all(g.period is not None for g in model.gates)  # only after-test gates re-run

    tags = _versions(demo_project)[0]["tags"]
    assert tags["mbt.oot_check.passed"] == "true"
    assert tags["mbt.oot_check.source"] == "evaluate"
    assert tags["mbt.oot_check.anchor"] == "2026-07-01T00:00:00Z"
    assert tags["mbt.oot_check.run_id"] == results.metadata.run_id
    assert tags["mbt.oot_check.report_uri"].endswith("/report/report.html")

    run = _run_payload(demo_project)
    directories = run["directories"]
    assert f"evaluations/{results.metadata.run_id}" in directories
    assert f"evaluate-{results.metadata.run_id}.log" in directories["logs"]
    assert any(key.startswith("oot_check.oot.window.") for key in run["metrics"])
    assert run["tags"]["mbt.oot_check.source"] == "evaluate"
    # the check's after-test window starts where the recorded test window ended
    windows = results.results[0]
    assert windows.status == "success"

    outcome = promote_model(
        _registry(demo_project, fake_registry),
        name="churn_model",
        to_stage=Stage.PRODUCTION,
        version="1",
        require_oot_check=True,
    )
    assert outcome.version == "1"


def test_the_check_ranks_features_by_permutation_when_the_model_cannot(
    demo_project: Path, fake_registry: AdapterRegistry, monkeypatch: pytest.MonkeyPatch
) -> None:
    from mbt.execute import job as job_module

    reporting_project(demo_project)
    _build(demo_project, fake_registry)
    monkeypatch.setattr(job_module, "_feature_importance", lambda runtime, model: {})
    with recording_bus() as sink:
        results = _check(demo_project, fake_registry, apply_gates=False)
    assert [r.status for r in results.results] == ["success", "success"]
    assert any("feature importance by permutation" in m for m in sink.messages())


def _registry(project: Path, registry: AdapterRegistry) -> Any:
    from mbt.config.profiles import load_profiles
    from mbt.runtime import registry_adapter

    profiles = load_profiles("demo", project)
    return registry_adapter(profiles, project, registry)


def test_the_check_verdict_can_fail(demo_project: Path, fake_registry: AdapterRegistry) -> None:
    reporting_project(demo_project)
    _build(demo_project, fake_registry)
    versions = _versions(demo_project)
    config_path = Path(versions[0]["tags"]["mbt.inference_config_uri"].removeprefix("file://"))
    document = json.loads(config_path.read_text())
    for gate in document["spec"]["evaluation"]["gates"]:
        if gate.get("source") == "out_of_time":
            gate["threshold"] = 0.99
    config_path.write_text(json.dumps(document))
    _rehash(demo_project, config_path)

    with recording_bus() as sink:
        results = _check(demo_project, fake_registry)
    assert any(m.startswith("pre-deploy check on churn_model v1 FAILED") for m in sink.messages())
    assert results.exit_code() == 2
    assert results.results[1].status == "gate_failed"
    assert "oot_roc_auc" in (results.results[1].message or "")
    assert _versions(demo_project)[0]["tags"]["mbt.oot_check.passed"] == "false"

    registry = _registry(demo_project, fake_registry)
    with pytest.raises(StateError, match=r"its after-test check at .* failed"):
        promote_model(registry, name="churn_model", to_stage=Stage.PRODUCTION, version="1")
    with recording_bus() as sink:
        forced = promote_model(
            registry, name="churn_model", to_stage=Stage.PRODUCTION, version="1", force=True
        )
    assert forced.version == "1"
    assert any("FORCED promotion of churn_model v1 although" in m for m in sink.messages())


def _rehash(project: Path, path: Path) -> None:
    """Keep the fake registry's recorded content hash in step with an edit."""
    import hashlib

    registry_file = project / "target" / "fake_registry" / "churn_model.json"
    versions = json.loads(registry_file.read_text())
    digest = "sha256:" + hashlib.sha256(path.read_bytes()).hexdigest()
    for entry in versions:
        entry["tags"]["mbt.inference_config_content_hash"] = digest
        entry["tags"]["mbt.inference_config_size_bytes"] = str(path.stat().st_size)
    registry_file.write_text(json.dumps(versions))


def test_a_check_without_gates_records_nothing(
    demo_project: Path, fake_registry: AdapterRegistry
) -> None:
    reporting_project(demo_project)
    _build(demo_project, fake_registry)
    results = _check(demo_project, fake_registry, apply_gates=False)
    assert results.exit_code() == 0
    assert results.results[1].gates == []
    assert _versions(demo_project)[0]["tags"]["mbt.oot_check.source"] == "build"


def test_the_check_needs_time_after_the_test_window(
    demo_project: Path, fake_registry: AdapterRegistry
) -> None:
    reporting_project(demo_project)
    _build(demo_project, fake_registry)
    with pytest.raises(StateError, match="nothing to check yet"):
        _check(demo_project, fake_registry, anchor=TRAINED_AT - timedelta(days=40))


def _retag(project: Path, **changes: Any) -> None:
    registry_file = project / "target" / "fake_registry" / "churn_model.json"
    versions = json.loads(registry_file.read_text())
    for key, value in changes.items():
        if value is None:
            versions[0]["tags"].pop(key, None)
        else:
            versions[0]["tags"][key] = value
    registry_file.write_text(json.dumps(versions))


def test_the_check_refuses_what_it_cannot_use(
    demo_project: Path, fake_registry: AdapterRegistry
) -> None:
    reporting_project(demo_project)
    _build(demo_project, fake_registry)
    versions = _versions(demo_project)
    config_path = Path(versions[0]["tags"]["mbt.inference_config_uri"].removeprefix("file://"))
    original = config_path.read_text()

    document = json.loads(original)
    document["dataset"]["split_strategy"] = "random"
    config_path.write_text(json.dumps(document))
    _rehash(demo_project, config_path)
    with pytest.raises(StateError, match="random split"):
        _check(demo_project, fake_registry)

    document = json.loads(original)
    document["dataset"]["unique_id"] = "dataset.demo.gone"
    config_path.write_text(json.dumps(document))
    _rehash(demo_project, config_path)
    with pytest.raises(StateError, match="no longer has"):
        _check(demo_project, fake_registry)

    document = json.loads(original)
    del document["dataset"]
    config_path.write_text(json.dumps(document))
    _rehash(demo_project, config_path)
    with pytest.raises(StateError, match="before mbt recorded its dataset windows"):
        _check(demo_project, fake_registry)

    _retag(demo_project, **{"mbt.inference_config_uri": None})
    with pytest.raises(StateError, match="predates inference-config export"):
        _check(demo_project, fake_registry)


def test_the_check_warns_when_its_reference_moved(
    demo_project: Path, fake_registry: AdapterRegistry
) -> None:
    reporting_project(demo_project)
    _build(demo_project, fake_registry)
    versions = _versions(demo_project)
    config_path = Path(versions[0]["tags"]["mbt.inference_config_uri"].removeprefix("file://"))
    document = json.loads(config_path.read_text())
    document["dataset"]["config_hash"] = "sha256:older"
    document["dataset"]["row_counts"]["test"] = 1
    document["dataset"]["test_window"] = document["dataset"]["windows"]["test"]
    config_path.write_text(json.dumps(document))
    _rehash(demo_project, config_path)
    with recording_bus() as sink:
        results = _check(demo_project, fake_registry)
    assert results.exit_code() == 0
    messages = sink.messages()
    assert any("has changed since the version was trained" in m for m in messages)
    assert any("test window now holds" in m for m in messages)


def test_a_failing_dataset_skips_the_check(
    demo_project: Path, fake_registry: AdapterRegistry
) -> None:
    reporting_project(demo_project)
    _build(demo_project, fake_registry)
    (demo_project / "data" / "subscribers" / "part-000.parquet").write_bytes(b"not parquet")
    results = _check(demo_project, fake_registry, anchor=TEST_ANCHOR + timedelta(days=1))
    assert [r.status for r in results.results] == ["error", "skipped"]


def test_a_failing_job_is_an_error_row(
    demo_project: Path, fake_registry: AdapterRegistry, monkeypatch: pytest.MonkeyPatch
) -> None:
    from mbt.execute import job as job_module

    reporting_project(demo_project)
    _build(demo_project, fake_registry)

    def broken(runtime: Any) -> Any:
        raise RuntimeError("the check blew up")

    monkeypatch.setattr(job_module, "_run_oot_check", broken)
    results = _check(demo_project, fake_registry)
    assert results.results[1].status == "error"
    assert "the check blew up" in (results.results[1].message or "")


class _TaglessRegistry:
    def __init__(self, inner: Any) -> None:
        self._inner = inner

    def __getattr__(self, name: str) -> Any:
        if name == "set_version_tags":
            raise AttributeError(name)
        return getattr(self._inner, name)


def test_a_registry_that_cannot_tag_still_reports(
    demo_project: Path, fake_registry: AdapterRegistry, monkeypatch: pytest.MonkeyPatch
) -> None:
    from mbt_testing.adapters import FakeTrackingAdapter

    from mbt.execute.runners import ExecutionContext

    reporting_project(demo_project)
    _build(demo_project, fake_registry)
    original = ExecutionContext.registry_adapter
    monkeypatch.setattr(
        ExecutionContext, "registry_adapter", lambda self: _TaglessRegistry(original(self))
    )
    real_log = FakeTrackingAdapter.log

    def refuse_verdict_tags(self: Any, run: Any, **kwargs: Any) -> None:
        if "mbt.oot_check.source" in (kwargs.get("tags") or {}):
            raise RuntimeError("tracker unreachable")
        real_log(self, run, **kwargs)

    monkeypatch.setattr(FakeTrackingAdapter, "log", refuse_verdict_tags)
    with recording_bus() as sink:
        results = _check(demo_project, fake_registry)
    assert results.exit_code() == 0
    messages = sink.messages()
    assert any("cannot tag versions" in m for m in messages)
    assert any("could not record the check on the training run" in m for m in messages)


# -- the promotion guard -----------------------------------------------------------------


class _OneVersionRegistry:
    def __init__(self, tags: dict[str, str]) -> None:
        self.version = ModelVersion(
            name="m",
            version="3",
            artifact=ArtifactRef(uri="mem://m/3", format="fake", content_hash="x", size_bytes=1),
            tags={"mbt.gates_passed": "true", **tags},
        )
        self.moved: list[Stage] = []

    def get_version(self, name: str, version: str) -> ModelVersion | None:
        return self.version if version == "3" else None

    def get_champion(self, name: str, stage: Stage) -> ModelVersion | None:
        return self.version

    def transition(self, version: ModelVersion, stage: Stage) -> None:
        self.moved.append(stage)


@pytest.mark.parametrize(
    ("tags", "message"),
    [
        ({}, "no after-test check has judged it"),
        ({"mbt.oot_check.passed": "not_gated"}, "had nothing mature to judge"),
    ],
)
def test_requiring_the_check_refuses_an_unjudged_version(
    tags: dict[str, str], message: str
) -> None:
    registry = _OneVersionRegistry(tags)
    with pytest.raises(StateError, match=message):
        promote_model(
            registry, name="m", to_stage=Stage.PRODUCTION, version="3", require_oot_check=True
        )
    promote_model(registry, name="m", to_stage=Stage.PRODUCTION, version="3")
    assert registry.moved == [Stage.PRODUCTION]


def test_a_rollback_only_warns_about_a_failed_check() -> None:
    class _Rollback(_OneVersionRegistry):
        def get_champion(self, name: str, stage: Stage) -> ModelVersion | None:
            return ModelVersion(name="m", version="4", tags={})

    registry = _Rollback({"mbt.oot_check.passed": "false"})
    with recording_bus() as sink:
        outcome = rollback_model(registry, name="m", to_version="3")
    assert outcome.version == "3"
    assert any("rolling back to m v3 although" in m for m in sink.messages())


def test_a_promotions_file_can_require_the_check(tmp_path: Path) -> None:
    from mbt.promote import load_promotions_file

    path = tmp_path / "promotions.yml"
    path.write_text(
        "promotions:\n  - {model: m, to: production, version: '3', require_oot_check: true}\n"
    )
    (entry,) = load_promotions_file(path)
    assert entry.require_oot_check is True
