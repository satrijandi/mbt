"""``mbt score`` end to end over the fake adapters (ADR-20/21)."""

import json
from datetime import timedelta
from pathlib import Path

import pyarrow as pa
import pyarrow.parquet as pq
import pytest
from core_helpers import TEST_ANCHOR, write
from mbt_testing import FakeRegistryAdapter, FakeTrainingAdapter
from test_execution import MODEL, invoke

from mbt.adapters.registry import AdapterRegistry
from mbt.contracts import Stage

SCORING = "scoring.demo.churn_scoring"

SCORING_YML = """
scoring:
  - name: churn_scoring
    owner: lifecycle-eng@example.com
    model: ref('churn_model')
    tags: [daily]
    input:
      source: source('lakehouse', 'scoring_batch')
      time_column: snapshot_date
      window: "-7d:now"
    checks:
      - not_null:
          columns: [user_id]
    monitors:
      feature_shift:
        threshold: 0.5
        include: [tenure_days, monthly_usage, plan_type]
    output:
      path: predictions/churn_scores
      columns: [user_id]
"""


def _write_batch(
    project_dir: Path, *, tenure_offset: int = 0, null_user_ids: bool = False, n: int = 120
) -> None:
    """A fresh batch drawn from the training generator's value space, so an
    unshifted batch stays under the shift thresholds."""
    base = TEST_ANCHOR.replace(tzinfo=None)
    spread = [(i * 131) % 400 for i in range(n)]  # 131 coprime to 400: even coverage
    table = pa.table(
        {
            "user_id": [None if null_user_ids and i % 7 == 0 else i for i in range(n)],
            "snapshot_date": [base - timedelta(days=1 + i % 5) for i in range(n)],
            "is_active": [True] * n,
            "tenure_days": [30 + (idx * 7) % 900 + tenure_offset for idx in spread],
            "monthly_usage": [round((idx * 13.7) % 500, 2) for idx in spread],
            "plan_type": [("basic", "pro", "enterprise")[idx % 3] for idx in spread],
        }
    )
    out = project_dir / "data" / "scoring_batch"
    out.mkdir(parents=True, exist_ok=True)
    pq.write_table(table, out / "part-000.parquet")


@pytest.fixture()
def scoring_project(demo_project: Path) -> Path:
    write(
        demo_project / "sources.yml",
        """
        sources:
          - name: lakehouse
            tables:
              - name: subscribers
                path: data/subscribers/*.parquet
              - name: scoring_batch
                path: data/scoring_batch/*.parquet
        """,
    )
    write(demo_project / "scoring/churn_scoring.yml", SCORING_YML)
    _write_batch(demo_project)
    return demo_project


def _registry_file(project_dir: Path) -> Path:
    return project_dir / "target/fake_registry/churn_model.json"


def _promote(project_dir: Path, version: str = "1") -> None:
    registry = FakeRegistryAdapter({"root": str(project_dir / "target/fake_registry")})
    resolved = registry.get_version("churn_model", version)
    assert resolved is not None
    registry.transition(resolved, Stage.PRODUCTION)


def _edit_tags(project_dir: Path, version: str, **changes: str | None) -> None:
    path = _registry_file(project_dir)
    entries = json.loads(path.read_text())
    for entry in entries:
        if entry["version"] == version:
            for key, value in changes.items():
                if value is None:
                    entry["tags"].pop(key, None)
                else:
                    entry["tags"][key] = value
    path.write_text(json.dumps(entries, indent=2, sort_keys=True))


def _prediction_runs(project_dir: Path) -> list[Path]:
    root = project_dir / "predictions/churn_scores"
    if not root.is_dir():
        return []
    return sorted(p for p in root.iterdir() if (p / "_SUCCESS").is_file())


def _build_and_promote(project_dir: Path, registry: AdapterRegistry) -> None:
    assert invoke(project_dir, registry, "build").exit_code() == 0
    _promote(project_dir)


def test_scoring_emits_decision_column_for_a_configured_operating_point(
    scoring_project: Path, fake_registry: AdapterRegistry
) -> None:
    """A configured operating point emits a 0/1 decision column and records the
    cutoff in the run info, so consumers get a decision rule (R2-5)."""
    write(
        scoring_project / "scoring/churn_scoring.yml",
        SCORING_YML.replace(
            "path: predictions/churn_scores",
            "path: predictions/churn_scores\n      decision_threshold: 0.5",
        ),
    )
    _build_and_promote(scoring_project, fake_registry)
    assert invoke(scoring_project, fake_registry, "score").exit_code() == 0

    run = _prediction_runs(scoring_project)[0]
    table = pq.read_table(run / "predictions.parquet")
    assert "decision" in table.column_names
    pred = table.column("prediction").to_numpy(zero_copy_only=False)
    dec = table.column("decision").to_numpy(zero_copy_only=False)
    assert set(dec.tolist()) <= {0, 1}
    assert all(int(p >= 0.5) == int(d) for p, d in zip(pred, dec, strict=True))  # prob >= threshold
    info = json.loads((run / "predictions.json").read_text())
    assert info["meta"]["decision_threshold"] == "0.5"


def test_scoring_resolves_decision_threshold_from_the_champion_operating_point(
    scoring_project: Path, fake_registry: AdapterRegistry
) -> None:
    """A string decision_threshold names a champion operating-point metric,
    resolved from the registered champion's tags at score time (R2-5), so the
    deployed cutoff tracks the promoted model instead of a hand-copied constant."""
    model_yml = scoring_project / "models/churn_model.yml"
    write(
        model_yml,
        model_yml.read_text().replace(
            "metrics: [pr_auc, roc_auc]",
            "metrics: [pr_auc, roc_auc, threshold_at_precision_0.5]",
        ),
    )
    write(
        scoring_project / "scoring/churn_scoring.yml",
        SCORING_YML.replace(
            "path: predictions/churn_scores",
            "path: predictions/churn_scores\n      decision_threshold: threshold_at_precision_0.5",
        ),
    )
    _build_and_promote(scoring_project, fake_registry)

    # registration recorded the operating point on the champion...
    tags = json.loads(_registry_file(scoring_project).read_text())[0]["tags"]
    recorded = tags["mbt.operating_point.threshold_at_precision_0.5"]

    assert invoke(scoring_project, fake_registry, "score").exit_code() == 0
    run = _prediction_runs(scoring_project)[0]
    info = json.loads((run / "predictions.json").read_text())
    # ...and scoring resolved the string to that concrete cutoff and applied it
    assert info["meta"]["decision_threshold"] == str(float(recorded))
    table = pq.read_table(run / "predictions.parquet")
    pred = table.column("prediction").to_numpy(zero_copy_only=False)
    dec = table.column("decision").to_numpy(zero_copy_only=False)
    thr = float(recorded)
    assert all(int(p >= thr) == int(d) for p, d in zip(pred, dec, strict=True))


def test_resolve_operating_point_requires_a_recorded_champion_tag() -> None:
    """Resolving a string decision_threshold from a champion missing that
    operating point is a loud error, not a silent skip; a numeric threshold and
    a recorded one pass through (R2-5)."""
    from mbt.contracts import ModelVersion, ScoringOutputSpec
    from mbt.exceptions import StateError
    from mbt.execute.runners import ScoringRunner

    op = ScoringOutputSpec(path="predictions/x", decision_threshold="threshold_at_precision_0.9")
    recorded = ModelVersion(
        name="m", version="3", tags={"mbt.operating_point.threshold_at_precision_0.9": "0.42"}
    )
    resolved = ScoringRunner._resolve_operating_point(op, recorded, "scoring.p.s")
    assert resolved.decision_threshold == 0.42  # string -> the champion's recorded float

    missing = ModelVersion(name="m", version="3", tags={})
    with pytest.raises(StateError, match="not a recorded operating point"):
        ScoringRunner._resolve_operating_point(op, missing, "scoring.p.s")

    numeric = ScoringOutputSpec(path="predictions/x", decision_threshold=0.5)
    passthrough = ScoringRunner._resolve_operating_point(numeric, missing, "s")
    assert passthrough.decision_threshold == 0.5  # a numeric cutoff is untouched


def test_scoring_emits_per_prediction_explanation_when_configured(
    scoring_project: Path, fake_registry: AdapterRegistry
) -> None:
    """explain_top_k emits an `explanation` column - each row's top-k feature
    contributors as JSON - so a consumer can answer 'why did THIS row score
    this way' (explainability)."""
    write(
        scoring_project / "scoring/churn_scoring.yml",
        SCORING_YML.replace(
            "path: predictions/churn_scores",
            "path: predictions/churn_scores\n      explain_top_k: 2",
        ),
    )
    _build_and_promote(scoring_project, fake_registry)
    assert invoke(scoring_project, fake_registry, "score").exit_code() == 0

    run = _prediction_runs(scoring_project)[0]
    table = pq.read_table(run / "predictions.parquet")
    assert "explanation" in table.column_names
    top = json.loads(table.column("explanation").to_pylist()[0])
    assert len(top) == 2 and top[0][0] == "fake_signal"  # the fake adapter's stand-in


def test_explain_top_k_without_adapter_support_is_actionable(
    scoring_project: Path, fake_registry: AdapterRegistry, monkeypatch: pytest.MonkeyPatch
) -> None:
    """An adapter that cannot explain fails with mbt's actionable error, not a
    silent skip or an AttributeError."""
    monkeypatch.delattr(FakeTrainingAdapter, "explain", raising=False)  # adapter without support
    write(
        scoring_project / "scoring/churn_scoring.yml",
        SCORING_YML.replace(
            "path: predictions/churn_scores",
            "path: predictions/churn_scores\n      explain_top_k: 2",
        ),
    )
    _build_and_promote(scoring_project, fake_registry)
    result = invoke(scoring_project, fake_registry, "score")
    assert result.exit_code() == 1  # hard error, not a silent skip
    errored = [r for r in result.results if r.status == "error"]
    assert errored and "does not support per-prediction" in (errored[0].message or "")


def test_score_with_manifest_tolerates_a_changed_batch(
    scoring_project: Path, fake_registry: AdapterRegistry
) -> None:
    """--manifest scoring is usable: the scoring input is expected to change
    every run, so a drifted batch scores from the pinned manifest instead of a
    snapshot-mismatch error the way a dataset would (R2-10)."""
    _build_and_promote(scoring_project, fake_registry)
    assert invoke(scoring_project, fake_registry, "score").exit_code() == 0  # pins the manifest
    manifest = scoring_project / "target" / "manifest.json"

    _write_batch(scoring_project, tenure_offset=99)  # fresh batch -> new snapshot
    result = invoke(scoring_project, fake_registry, "score", manifest_path=str(manifest))
    assert result.exit_code() == 0  # not a "source data changed under the pinned manifest" error
    assert {r.unique_id: r for r in result.results}[SCORING].status == "success"


def test_score_with_manifest_rebuilds_the_input_not_a_stale_cache(
    scoring_project: Path, fake_registry: AdapterRegistry
) -> None:
    """F4: under --manifest the materialization key is fully pinned (input_hash
    + resolved windows never move), so the scoring-input cache must not be
    reused - a new batch arriving under the same pinned manifest has to be
    rebuilt and scored, not served stale from the first run's warm cache."""
    _build_and_promote(scoring_project, fake_registry)
    first = invoke(scoring_project, fake_registry, "score")  # pins manifest + warms input cache
    assert first.exit_code() == 0
    assert {r.unique_id: r for r in first.results}[SCORING].metrics["rows_scored"] == 120.0
    manifest = scoring_project / "target" / "manifest.json"

    # A new, larger batch (same value distribution, so shift monitors stay
    # green) arrives under the SAME pinned manifest.
    _write_batch(scoring_project, n=240)
    result = invoke(scoring_project, fake_registry, "score", manifest_path=str(manifest))
    assert result.exit_code() == 0
    # The live 240-row batch is scored, not the stale 120 from the pinned-key cache.
    assert {r.unique_id: r for r in result.results}[SCORING].metrics["rows_scored"] == 240.0


def test_score_full_loop(scoring_project: Path, fake_registry: AdapterRegistry) -> None:
    _build_and_promote(scoring_project, fake_registry)
    results = invoke(scoring_project, fake_registry, "score")
    assert results.exit_code() == 0
    by_id = {r.unique_id: r for r in results.results}
    assert set(by_id) == {SCORING}  # scoring only: no dataset/model auto-joins
    node = by_id[SCORING]
    assert node.status == "success"
    assert node.metrics["rows_scored"] == 120.0
    assert node.tests and all(t.passed for t in node.tests)
    assert node.monitors and all(m.passed for m in node.monitors)
    assert {m.monitor for m in node.monitors} == {"feature_shift"}
    assert node.tracking_run_id

    runs = _prediction_runs(scoring_project)
    assert len(runs) == 1
    info = json.loads((runs[0] / "predictions.json").read_text())
    assert info["model_version"] == "1"
    assert info["row_count"] == 120
    assert info["scored_at"] == "2026-07-01T00:00:00Z"
    assert info["meta"]["input_hash"].startswith("sha256:")
    table = pq.read_table(runs[0] / "predictions.parquet")
    assert table.column_names == ["user_id", "snapshot_date", "prediction"]
    assert table.num_rows == 120

    stored = json.loads((scoring_project / "target/run_results.json").read_text())
    assert stored["metadata"]["command"] == "score"

    # tracking run carries the resolved champion + run identity
    tracking_file = scoring_project / "target/fake_tracking" / f"{node.tracking_run_id}.json"
    payload = json.loads(tracking_file.read_text())
    assert payload["tags"]["mbt.model_version"] == "1"
    assert payload["metrics"]["predictions.rows"] == 120.0


def test_rescore_same_manifest_overwrites_same_run(
    scoring_project: Path, fake_registry: AdapterRegistry
) -> None:
    _build_and_promote(scoring_project, fake_registry)
    assert invoke(scoring_project, fake_registry, "score").exit_code() == 0
    first = _prediction_runs(scoring_project)
    assert invoke(scoring_project, fake_registry, "score").exit_code() == 0
    second = _prediction_runs(scoring_project)
    assert [p.name for p in first] == [p.name for p in second]
    assert len(second) == 1


def test_promotion_is_picked_up_on_next_score(
    scoring_project: Path, fake_registry: AdapterRegistry
) -> None:
    _build_and_promote(scoring_project, fake_registry)
    assert invoke(scoring_project, fake_registry, "score").exit_code() == 0

    # v2: a config change retrains and registers a new version
    model_yml = scoring_project / "models/churn_model.yml"
    model_yml.write_text(
        model_yml.read_text().replace("fake_metric_value: 0.61", "fake_metric_value: 0.63")
    )
    assert invoke(scoring_project, fake_registry, "build").exit_code() == 0
    _promote(scoring_project, version="2")

    assert invoke(scoring_project, fake_registry, "score").exit_code() == 0
    runs = _prediction_runs(scoring_project)
    assert len(runs) == 2  # new champion => new run_key partition
    versions = {json.loads((run / "predictions.json").read_text())["model_version"] for run in runs}
    assert versions == {"1", "2"}


def test_feature_shift_breach_exits_2(
    scoring_project: Path, fake_registry: AdapterRegistry
) -> None:
    _build_and_promote(scoring_project, fake_registry)
    _write_batch(scoring_project, tenure_offset=100_000)
    results = invoke(scoring_project, fake_registry, "score")
    assert results.exit_code() == 2
    node = results.results[0]
    assert node.status == "monitor_failed"
    breached = [m for m in node.monitors if not m.passed]
    assert any(m.monitor == "feature_shift" and m.subject == "tenure_days" for m in breached)
    assert node.message and "tenure_days" in node.message
    # predictions are still written: monitoring flags, it does not censor
    assert len(_prediction_runs(scoring_project)) == 1


def test_missing_baseline_warns_and_passes(
    scoring_project: Path, fake_registry: AdapterRegistry
) -> None:
    _build_and_promote(scoring_project, fake_registry)
    _edit_tags(
        scoring_project,
        "1",
        **{
            "mbt.baseline_uri": None,
            "mbt.baseline_format": None,
            "mbt.baseline_content_hash": None,
            "mbt.baseline_size_bytes": None,
        },
    )
    results = invoke(scoring_project, fake_registry, "score")
    assert results.exit_code() == 0
    node = results.results[0]
    assert node.status == "success"
    assert node.monitors and all(m.passed for m in node.monitors)
    assert all(m.message and "baseline missing" in m.message for m in node.monitors)


def test_missing_champion_is_a_hard_error(
    scoring_project: Path, fake_registry: AdapterRegistry
) -> None:
    assert invoke(scoring_project, fake_registry, "build").exit_code() == 0
    # registered to staging only; scoring wants production
    results = invoke(scoring_project, fake_registry, "score")
    assert results.exit_code() == 1
    node = results.results[0]
    assert node.status == "error"
    assert node.message and "no champion" in node.message


def test_hooks_hash_mismatch_is_a_hard_error(
    scoring_project: Path, fake_registry: AdapterRegistry
) -> None:
    _build_and_promote(scoring_project, fake_registry)
    _edit_tags(scoring_project, "1", **{"mbt.hooks_hash": "sha256:trained-with-other-hooks"})
    results = invoke(scoring_project, fake_registry, "score")
    assert results.exit_code() == 1
    assert results.results[0].message and "hooks" in results.results[0].message


def test_hooks_hash_absent_warns_and_proceeds(
    scoring_project: Path, fake_registry: AdapterRegistry
) -> None:
    _build_and_promote(scoring_project, fake_registry)
    _edit_tags(scoring_project, "1", **{"mbt.hooks_hash": None})
    results = invoke(scoring_project, fake_registry, "score")
    assert results.exit_code() == 0


def test_input_check_failure_skips_scoring(
    scoring_project: Path, fake_registry: AdapterRegistry
) -> None:
    _build_and_promote(scoring_project, fake_registry)
    _write_batch(scoring_project, null_user_ids=True)
    results = invoke(scoring_project, fake_registry, "score")
    assert results.exit_code() == 2
    node = results.results[0]
    assert node.status == "test_failed"
    assert node.message and "scoring skipped" in node.message
    assert _prediction_runs(scoring_project) == []


def test_build_never_executes_scoring_nodes(
    scoring_project: Path, fake_registry: AdapterRegistry
) -> None:
    results = invoke(scoring_project, fake_registry, "build")
    assert {r.unique_id for r in results.results} == {"dataset.demo.churn_training", MODEL}


def test_score_selector_narrows_scoring_nodes(
    scoring_project: Path, fake_registry: AdapterRegistry
) -> None:
    _build_and_promote(scoring_project, fake_registry)
    none_selected = invoke(scoring_project, fake_registry, "score", select=["tag:weekly"])
    assert none_selected.results == []
    selected = invoke(scoring_project, fake_registry, "score", select=["resource_type:scoring"])
    assert {r.unique_id for r in selected.results} == {SCORING}
