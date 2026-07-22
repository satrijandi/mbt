"""Unit tests for node runners and the execution context (mbt/execute/runners.py)."""

import json
from datetime import timedelta
from pathlib import Path
from types import SimpleNamespace

import pyarrow as pa
import pyarrow.parquet as pq
import pytest
from core_helpers import TEST_ANCHOR, write
from exec_unit_helpers import (
    DATASET_UID,
    MODEL_UID,
    SCORING_UID,
    make_execution_context,
    recording_bus,
)
from mbt_testing.adapters import FakeTrackingAdapter
from test_execution import MODEL, invoke
from test_job_unit import SOURCES_WITH_BATCH
from test_scoring_execution import SCORING_YML, _build_and_promote, _write_batch

from mbt.adapters.registry import AdapterRegistry
from mbt.artifacts.run_results import GateResult
from mbt.contracts import ScoringSpec
from mbt.exceptions import ConfigError
from mbt.execute.runners import (
    DatasetRunner,
    ExecutionContext,
    ModelRunner,
    ModelTestRunner,
    ScoringRunner,
    _gate_failure_summary,
)

OLD_GATE = "threshold: \"{{ var('default_threshold') }}\""

MULTI_SOURCES_YML = """
sources:
  - name: lakehouse
    tables:
      - name: subscribers
        path: data/subscribers/*.parquet
      - name: scoring_spine
        path: data/scoring_spine/*.parquet
      - name: scoring_features
        path: data/scoring_features/*.parquet
"""

MULTI_SCORING_YML = """
scoring:
  - name: churn_scoring
    owner: lifecycle-eng@example.com
    model: ref('churn_model')
    input:
      inputs:
        spine: source('lakehouse', 'scoring_spine')
        features:
          - source('lakehouse', 'scoring_features')
        join_key: user_id
      time_column: snapshot_date
      window: "-7d:now"
    output:
      path: predictions/churn_scores
      columns: [user_id]
"""


def _score_setup(project_dir: Path) -> None:
    write(project_dir / "sources.yml", SOURCES_WITH_BATCH)
    write(project_dir / "scoring/churn_scoring.yml", SCORING_YML)
    _write_batch(project_dir)


def _stub_registry_champion_without_artifact(project_dir: Path) -> None:
    registry_dir = project_dir / "target" / "fake_registry"
    registry_dir.mkdir(parents=True, exist_ok=True)
    (registry_dir / "churn_model.json").write_text(
        json.dumps([{"version": "1", "stage": "production", "artifact": None, "tags": {}}])
    )


# -- gate failure summaries -----------------------------------------------------------


def test_gate_failure_summary_metric_only_and_fallback() -> None:
    bare = [GateResult(metric="pr_auc", kind="champion", passed=False)]
    assert _gate_failure_summary(bare) == "gate breach: pr_auc"
    all_passed = [GateResult(metric="pr_auc", kind="threshold", passed=True)]
    assert _gate_failure_summary(all_passed) == "one or more gates failed"


# -- ExecutionContext -----------------------------------------------------------------


def test_execution_context_warms_tracking_backends(
    demo_project: Path, fake_registry: AdapterRegistry, monkeypatch: pytest.MonkeyPatch
) -> None:
    called: list[bool] = []
    monkeypatch.setattr(
        FakeTrackingAdapter, "prepare", lambda self: called.append(True), raising=False
    )
    make_execution_context(demo_project, fake_registry)
    assert called == [True]


def test_execution_context_serializes_shared_state_for_free_threading(
    demo_project: Path, fake_registry: AdapterRegistry
) -> None:
    """next_index and the dataset-handle store go through _state_lock, so the
    coordinator's shared counter and handle map are safe under a free-threaded
    (no-GIL) build (P3). Verified by the lock's mutual exclusion - holding it
    blocks a concurrent mutator - and a concurrent stress run with no lost
    indices."""
    import threading

    ctx = make_execution_context(demo_project, fake_registry)

    # mutual exclusion: while the lock is held, next_index() cannot complete,
    # proving it acquires _state_lock (this is what falsifies a missing lock).
    completed = threading.Event()

    def mutate() -> None:
        ctx.next_index()
        completed.set()

    with ctx._state_lock:
        worker = threading.Thread(target=mutate)
        worker.start()
        assert not completed.wait(timeout=0.25), "next_index did not wait on _state_lock"
    worker.join(timeout=2)
    assert completed.is_set()  # lock released -> it completes

    # correctness under concurrency: 8 threads, every index unique and gap-free.
    seen: list[int] = []
    collect = threading.Lock()

    def hammer() -> None:
        local = [ctx.next_index() for _ in range(500)]
        with collect:
            seen.extend(local)

    threads = [threading.Thread(target=hammer) for _ in range(8)]
    for thread in threads:
        thread.start()
    for thread in threads:
        thread.join()
    assert len(seen) == len(set(seen)) == 8 * 500  # no duplicate / lost updates

    # the dataset-handle map is guarded by the same lock
    ctx.store_dataset_handle("dataset.x", "handle-x")
    assert ctx.dataset_handle("dataset.x") == "handle-x"


def test_raw_adapter_ref_falls_back_to_rendered_profile(
    demo_project: Path, fake_registry: AdapterRegistry
) -> None:
    ctx = make_execution_context(demo_project, fake_registry)
    ctx.manifest.metadata.target_config.pop("tracking", None)
    ref = ctx.raw_adapter_ref("tracking")
    assert ref.adapter == "fake"


# -- DatasetRunner --------------------------------------------------------------------


def test_dataset_spine_missing_from_manifest_is_an_error(
    demo_project: Path, fake_registry: AdapterRegistry
) -> None:
    ctx = make_execution_context(demo_project, fake_registry)
    node = ctx.manifest.nodes[DATASET_UID]
    node.config["source"] = "source.demo.lakehouse.elsewhere"
    result = DatasetRunner(ctx).run(DATASET_UID)
    assert result.status == "error"
    assert "spine source" in (result.message or "")


def test_dataset_declared_tests_bind_and_filter(
    demo_project: Path, fake_registry: AdapterRegistry
) -> None:
    ds_yml = demo_project / "datasets/churn_training.yml"
    ds_yml.write_text(
        ds_yml.read_text().replace("tags: [churn]", "tests: [test_has_rows]\n    tags: [churn]")
    )
    write(
        demo_project / "tests/test_row_checks.py",
        """
        def test_has_rows(dataset, spec):
            return dataset.num_rows > 0

        def test_not_bound(dataset, spec):
            return False
        """,
    )
    results = invoke(demo_project, fake_registry, "build")
    assert results.exit_code() == 0
    ds = {r.unique_id: r for r in results.results}[DATASET_UID]
    names = {t.name for t in ds.tests}
    assert "test_has_rows" in names
    assert "test_not_bound" not in names  # filtered by the dataset's tests list


def test_selectorless_data_test_binds_all_datasets(
    demo_project: Path, fake_registry: AdapterRegistry
) -> None:
    write(
        demo_project / "tests/test_generic_checks.py",
        """
        def test_any_rows(dataset, spec):
            return dataset.num_rows > 0
        """,
    )
    results = invoke(demo_project, fake_registry, "build")
    assert results.exit_code() == 0
    ds = {r.unique_id: r for r in results.results}[DATASET_UID]
    assert "test_any_rows" in {t.name for t in ds.tests}


# -- ModelRunner ----------------------------------------------------------------------


def test_assemble_job_threads_profile_tuning_config(
    demo_project: Path, fake_registry: AdapterRegistry
) -> None:
    """Tuning-engine ops config (sampler/pruner knobs) comes from the target
    profile's `tuning` block, not the spec; absent it the engine gets {}."""
    from mbt_adapter_base import AdapterRef, ModelSpec, TuningSpec

    ctx = make_execution_context(demo_project, fake_registry)
    assert DatasetRunner(ctx).run(DATASET_UID).status == "success"
    node = ctx.manifest.nodes[MODEL_UID]
    spec = ModelSpec.model_validate(node.config).model_copy(
        update={
            "tuning": TuningSpec.model_validate(
                {
                    "engine": "optuna",
                    "n_trials": 2,
                    "search_space": {"max_depth": {"type": "int", "low": 2, "high": 5}},
                    "objective": {"metric": "pr_auc", "direction": "maximize"},
                }
            )
        }
    )
    runner = ModelRunner(ctx)
    metric_specs = runner._metric_specs(spec, node)

    # no profile tuning block: the engine is named but gets no ops config
    job = runner._assemble_job(node, spec, metric_specs, None)
    assert job.tuning_engine is not None
    assert job.tuning_engine.adapter == "optuna" and job.tuning_engine.config == {}

    # a profile tuning block flows its config into the engine (this is what
    # makes the sampler/pruner knobs reachable at all)
    ctx.profiles.target.tuning = AdapterRef(adapter="optuna", config={"multivariate": True})
    job = runner._assemble_job(node, spec, metric_specs, None)
    assert job.tuning_engine is not None and job.tuning_engine.config == {"multivariate": True}


def test_model_metric_resolution_error_is_a_node_error(
    demo_project: Path, fake_registry: AdapterRegistry
) -> None:
    ctx = make_execution_context(demo_project, fake_registry)
    node = ctx.manifest.nodes[MODEL_UID]
    node.config["evaluation"]["gates"] = []
    node.config["evaluation"]["metrics"] = ["mystery_metric"]
    result = ModelRunner(ctx).run(MODEL_UID)
    assert result.status == "error"
    assert "unknown metric" in (result.message or "")


def test_champion_without_loadable_artifact_is_an_error(
    demo_project: Path, fake_registry: AdapterRegistry
) -> None:
    model_yml = demo_project / "models/churn_model.yml"
    model_yml.write_text(model_yml.read_text().replace(OLD_GATE, "compare_to: production"))
    _stub_registry_champion_without_artifact(demo_project)
    results = invoke(demo_project, fake_registry)
    assert results.exit_code() == 1
    model = {r.unique_id: r for r in results.results}[MODEL]
    assert model.status == "error"
    assert "loadable artifact" in (model.message or "")


def test_attach_tracking_tags_skips_without_run_id(
    demo_project: Path, fake_registry: AdapterRegistry
) -> None:
    ctx = make_execution_context(demo_project, fake_registry)
    # would raise if it touched tracking: there is nothing to resume
    ModelRunner(ctx)._attach_tracking_tags(SimpleNamespace(tracking_run_id=None), [], None)


def test_attach_tracking_tags_warns_on_tracking_failure(
    demo_project: Path, fake_registry: AdapterRegistry
) -> None:
    ctx = make_execution_context(demo_project, fake_registry)

    def _boom():
        raise RuntimeError("tracking down")

    ctx.tracking = _boom  # instance attribute shadows the method
    with recording_bus() as sink:
        ModelRunner(ctx)._attach_tracking_tags(SimpleNamespace(tracking_run_id="run-1"), [], None)
    assert any("could not attach tracking tags" in m for m in sink.messages())


def test_failed_training_job_is_a_node_error(
    demo_project: Path, fake_registry: AdapterRegistry
) -> None:
    model_yml = demo_project / "models/churn_model.yml"
    model_yml.write_text(
        model_yml.read_text().replace(
            "fake_metric_value: 0.61", "fake_metric_value: 0.61\n      fail_training: true"
        )
    )
    results = invoke(demo_project, fake_registry)
    assert results.exit_code() == 1
    model = {r.unique_id: r for r in results.results}[MODEL]
    assert model.status == "error"
    assert "fake training failure" in (model.message or "")


# -- ScoringRunner --------------------------------------------------------------------


def test_scoring_champion_without_artifact_is_an_error(
    demo_project: Path, fake_registry: AdapterRegistry
) -> None:
    _score_setup(demo_project)
    _stub_registry_champion_without_artifact(demo_project)
    results = invoke(demo_project, fake_registry, "score")
    assert results.exit_code() == 1
    node = results.results[0]
    assert node.status == "error"
    assert "loadable artifact" in (node.message or "")


def test_multi_table_scoring_input_joins_spine_and_features(
    demo_project: Path, fake_registry: AdapterRegistry
) -> None:
    write(demo_project / "sources.yml", MULTI_SOURCES_YML)
    write(demo_project / "scoring/churn_scoring.yml", MULTI_SCORING_YML)
    n = 40
    base = TEST_ANCHOR.replace(tzinfo=None)
    spread = [(i * 131) % 400 for i in range(n)]
    spine = pa.table(
        {
            "user_id": list(range(n)),
            "snapshot_date": [base - timedelta(days=1 + i % 5) for i in range(n)],
            "is_active": [True] * n,
        }
    )
    features = pa.table(
        {
            "user_id": list(range(n)),
            "tenure_days": [30 + (idx * 7) % 900 for idx in spread],
            "monthly_usage": [round((idx * 13.7) % 500, 2) for idx in spread],
            "plan_type": [("basic", "pro", "enterprise")[idx % 3] for idx in spread],
        }
    )
    for name, table in (("scoring_spine", spine), ("scoring_features", features)):
        out = demo_project / "data" / name
        out.mkdir(parents=True, exist_ok=True)
        pq.write_table(table, out / "part-000.parquet")

    _build_and_promote(demo_project, fake_registry)
    results = invoke(demo_project, fake_registry, "score")
    assert results.exit_code() == 0
    node = results.results[0]
    assert node.status == "success"
    assert node.metrics["rows_scored"] == float(n)


def test_scoring_input_sources_missing_from_manifest(
    demo_project: Path, fake_registry: AdapterRegistry
) -> None:
    _score_setup(demo_project)
    ctx = make_execution_context(demo_project, fake_registry, command="score")
    node = ctx.manifest.nodes[SCORING_UID]
    spec = ScoringSpec.model_validate(node.config)
    uid = next(u for u in ctx.manifest.sources if u.endswith("scoring_batch"))
    del ctx.manifest.sources[uid]
    with pytest.raises(ConfigError, match="missing from the manifest"):
        ScoringRunner(ctx)._materialize_input(node, spec)


def test_scoring_model_node_missing_from_manifest(
    demo_project: Path, fake_registry: AdapterRegistry
) -> None:
    _score_setup(demo_project)
    ctx = make_execution_context(demo_project, fake_registry, command="score")
    del ctx.manifest.nodes[MODEL_UID]
    result = ScoringRunner(ctx).run(SCORING_UID)
    assert result.status == "error"
    assert "missing from the manifest" in (result.message or "")


# -- ModelTestRunner (mbt test, TSD §11.3) --------------------------------------------


def test_test_command_reevaluates_the_registered_version(
    demo_project: Path, fake_registry: AdapterRegistry
) -> None:
    assert invoke(demo_project, fake_registry).exit_code() == 0
    results = invoke(demo_project, fake_registry, "test")
    assert results.exit_code() == 0
    model = {r.unique_id: r for r in results.results}[MODEL]
    assert model.status == "success"
    assert model.gates and model.gates[0].passed
    assert model.metrics["pr_auc"] > 0.5
    assert model.feature_importance


def test_test_command_gate_breach_is_test_failed(
    demo_project: Path, fake_registry: AdapterRegistry
) -> None:
    assert invoke(demo_project, fake_registry).exit_code() == 0
    model_yml = demo_project / "models/churn_model.yml"
    model_yml.write_text(model_yml.read_text().replace(OLD_GATE, "threshold: 0.99"))
    results = invoke(demo_project, fake_registry, "test")
    assert results.exit_code() == 2
    model = {r.unique_id: r for r in results.results}[MODEL]
    assert model.status == "test_failed"
    assert "gate breach" in (model.message or "")


def test_test_command_without_gates_is_skipped(
    demo_project: Path, fake_registry: AdapterRegistry
) -> None:
    model_yml = demo_project / "models/churn_model.yml"
    gate_block = f"gates:\n        - metric: pr_auc\n          {OLD_GATE}"
    model_yml.write_text(model_yml.read_text().replace(gate_block, ""))
    results = invoke(demo_project, fake_registry, "test")
    assert results.exit_code() == 0
    model = {r.unique_id: r for r in results.results}[MODEL]
    assert model.status == "skipped"
    assert model.message == "model declares no gates"


def test_test_command_without_registered_version_is_skipped(
    demo_project: Path, fake_registry: AdapterRegistry
) -> None:
    with recording_bus() as sink:
        results = invoke(demo_project, fake_registry, "test")
    assert results.exit_code() == 0
    model = {r.unique_id: r for r in results.results}[MODEL]
    assert model.status == "skipped"
    assert model.message == "no registered version"
    assert any("mbt test never trains" in m for m in sink.messages())


def test_test_command_mbt_error_is_a_node_error(
    demo_project: Path, fake_registry: AdapterRegistry
) -> None:
    assert invoke(demo_project, fake_registry).exit_code() == 0
    ctx = make_execution_context(demo_project, fake_registry, command="test")
    node = ctx.manifest.nodes[MODEL_UID]
    node.config["evaluation"]["metrics"] = ["mystery_metric"]
    node.config["evaluation"]["gates"] = [{"metric": "mystery_metric", "threshold": 0.5}]
    result = ModelTestRunner(ctx).run(MODEL_UID)
    assert result.status == "error"
    assert "unknown metric" in (result.message or "")


def test_test_command_job_error(demo_project: Path, fake_registry: AdapterRegistry) -> None:
    assert invoke(demo_project, fake_registry).exit_code() == 0
    for path in (demo_project / "target" / "artifacts").rglob("fake_model.json"):
        path.write_text("corrupted, not json")
    results = invoke(demo_project, fake_registry, "test")
    assert results.exit_code() == 1
    model = {r.unique_id: r for r in results.results}[MODEL]
    assert model.status == "error"


def test_cancel_active_jobs_is_a_noop_without_a_terminate_seam() -> None:
    import threading

    ctx = SimpleNamespace(
        compute=object(),  # no terminate(): older/remote compute adapters
        _job_handles_lock=threading.Lock(),
        _active_job_handles=[object()],
    )
    ExecutionContext.cancel_active_jobs(ctx)  # must not raise


def test_cancel_active_jobs_terminates_each_handle_and_survives_races() -> None:
    import threading

    calls: list[tuple[object, str]] = []

    class _Compute:
        def terminate(self, handle: object, reason: str) -> None:
            calls.append((handle, reason))
            if len(calls) == 1:
                raise RuntimeError("job just exited")  # must not stop the sweep

    ctx = SimpleNamespace(
        compute=_Compute(),
        _job_handles_lock=threading.Lock(),
        _active_job_handles=["h1", "h2"],
    )
    ExecutionContext.cancel_active_jobs(ctx)
    assert [reason for _, reason in calls] == ["cancelled by --fail-fast"] * 2


def test_cancel_active_jobs_terminates_handles_concurrently() -> None:
    import threading

    # A 2-party barrier only releases when BOTH terminates are in flight at
    # once; a serial sweep would block the first terminate alone until the
    # barrier times out (BrokenBarrierError, suppressed) and neither records.
    barrier = threading.Barrier(2, timeout=5)
    passed: list[object] = []

    class _Compute:
        def terminate(self, handle: object, reason: str) -> None:
            barrier.wait()
            passed.append(handle)

    ctx = SimpleNamespace(
        compute=_Compute(),
        _job_handles_lock=threading.Lock(),
        _active_job_handles=["h1", "h2"],
    )
    ExecutionContext.cancel_active_jobs(ctx)
    assert sorted(passed) == ["h1", "h2"]  # both ran at the same time
