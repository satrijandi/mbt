"""The inference config a champion carries, and the scoring run that reads it (ADR-28)."""

import json
from pathlib import Path
from types import SimpleNamespace

from exec_unit_helpers import recording_bus
from test_execution import MODEL, invoke
from test_scoring_execution import _build_and_promote, scoring_project  # noqa: F401 - fixture

from mbt.adapters.registry import AdapterRegistry
from mbt.execute.inference_config import (
    SCHEMA_VERSION,
    build_inference_config,
    operating_points,
)
from mbt.runtime import tracking_adapter_config
from mbt_adapter_base import AdapterRef, ArtifactRef, ManifestNode

# -- the experiment name core composes ----------------------------------------


def _ref(**config: object) -> AdapterRef:
    return AdapterRef(adapter="mlflow", config=dict(config))


def test_project_and_experiment_compose(tmp_path: Path) -> None:
    resolved = tracking_adapter_config(_ref(uri="x", experiment="wide_v2"), tmp_path, "churn_lake")
    assert resolved["experiment"] == "churn_lake_wide_v2"
    assert resolved["uri"] == "x"  # everything else is passed through untouched


def test_the_project_name_stands_alone_when_no_experiment_is_set(tmp_path: Path) -> None:
    """A fresh project's runs land under its own name, not a literal 'mbt'."""
    assert tracking_adapter_config(_ref(uri="x"), tmp_path, "churn_lake")["experiment"] == (
        "churn_lake"
    )


def test_an_empty_experiment_name_is_treated_as_unset(tmp_path: Path) -> None:
    """`experiment: "{{ env('EXPERIMENT_NAME', '') }}"` with the var unset must
    not produce a trailing-underscore experiment named 'churn_lake_'."""
    assert (
        tracking_adapter_config(_ref(uri="x", experiment=""), tmp_path, "churn_lake")["experiment"]
        == "churn_lake"
    )


def test_a_non_name_experiment_is_left_for_the_adapter_to_reject(tmp_path: Path) -> None:
    """Core does not compose what it cannot understand; the adapter raises the
    error that names the shape it wanted."""
    declared = {"model": "a", "scoring": "b"}
    assert (
        tracking_adapter_config(_ref(uri="x", experiment=declared), tmp_path, "p")["experiment"]
        == declared
    )


# -- the document itself -------------------------------------------------------


def _node(**overrides: object) -> ManifestNode:
    base: dict[str, object] = {
        "unique_id": "model.demo.churn_model",
        "resource_type": "model",
        "name": "churn_model",
        "path": "models/churn_model.yml",
        "config": {
            "name": "churn_model",
            "target": "churned",
            "calibration": "isotonic",
            "features": {
                "include": ["*"],
                "categorical": {"contract_code": {"levels": ["a", "b"]}},
                "transforms": {"tenure_days": {"cap": 730, "log": True}},
                "monotonic": {"tenure_days": "increasing"},
            },
        },
        "adapter": "xgboost",
        "task": "binary_classification",
        "seed": 42,
        "hooks_path": "models/churn_model.py",
        "hooks_hash": "sha256:hooks",
        "config_hash": "sha256:cfg",
        "input_hash": "sha256:in",
    }
    base.update(overrides)
    return ManifestNode(**base)  # type: ignore[arg-type]


def _artifact() -> ArtifactRef:
    return ArtifactRef(uri="s3://b/model.ubj", format="ubj", content_hash="sha256:a", size_bytes=7)


def _build(**overrides: object) -> dict:
    kwargs: dict = {
        "node": _node(),
        "project": "demo",
        "run_id": "20260909T101500Z-a1b2c3d4",
        "feature_columns": ["tenure_days", "contract_code"],
        "metrics": {"pr_auc": 0.4, "threshold_at_precision_0.35": 0.37},
        "artifact": _artifact(),
        "baseline": ArtifactRef(
            uri="s3://b/baseline.json", format="json", content_hash="sha256:b", size_bytes=9
        ),
        "meta": {
            "mbt.manifest_hash": "sha256:man",
            "mbt.snapshot_id": "snap",
            "mbt.git_commit": "abc123",
        },
    }
    kwargs.update(overrides)
    return build_inference_config(**kwargs)


def test_the_document_carries_the_spec_verbatim() -> None:
    """`spec` is what ModelSpec.model_validate consumes and what config_hash is
    taken over, so a scoring run reconstructs exactly what was trained."""
    document = _build()
    assert document["spec"] == _node().config
    assert document["schema_version"] == SCHEMA_VERSION
    assert document["project"] == "demo"
    assert document["model"] == "churn_model"
    assert document["unique_id"] == "model.demo.churn_model"
    assert document["trained_at"] == "20260909T101500Z-a1b2c3d4"


def test_the_document_resolves_what_the_manifest_cannot() -> None:
    """features.include: ["*"] does not say which columns the model was fit on."""
    resolved = _build()["resolved"]
    assert resolved["feature_columns"] == ["tenure_days", "contract_code"]
    assert resolved["target"] == "churned"
    assert resolved["task"] == "binary_classification"
    assert resolved["adapter"] == "xgboost"
    assert resolved["seed"] == 42
    assert resolved["calibration"] == "isotonic"
    assert resolved["categorical"] == {"contract_code": {"levels": ["a", "b"]}}
    assert resolved["transforms"] == {"tenure_days": {"cap": 730, "log": True}}
    assert resolved["monotonic"] == {"tenure_days": "increasing"}


def test_only_deployable_cutoffs_are_operating_points() -> None:
    """A threshold_at_* metric names a score cutoff (R2-5); pr_auc is a quality."""
    assert operating_points({"pr_auc": 0.4, "threshold_at_precision_0.35": 0.37}) == {
        "threshold_at_precision_0.35": 0.37
    }
    assert operating_points({"threshold_at_recall_0.8": 0.2}) == {"threshold_at_recall_0.8": 0.2}
    assert operating_points({"roc_auc": 0.9}) == {}


def test_identity_and_pointers_round_trip() -> None:
    document = _build()
    assert document["identity"] == {
        "config_hash": "sha256:cfg",
        "input_hash": "sha256:in",
        "manifest_hash": "sha256:man",
        "snapshot_id": "snap",
        "git_commit": "abc123",
    }
    assert document["artifact"]["uri"] == "s3://b/model.ubj"
    assert document["baseline_uri"] == "s3://b/baseline.json"
    assert document["hooks"] == {"path": "models/churn_model.py", "hash": "sha256:hooks"}


def test_a_model_without_features_or_baseline_still_builds() -> None:
    """A spec may omit `features:` entirely, and a baseline export may be absent."""
    document = _build(node=_node(config={"name": "m", "target": "y"}), baseline=None)
    assert document["baseline_uri"] is None
    assert document["resolved"]["categorical"] is None
    assert document["resolved"]["transforms"] is None


# -- export, registration, and the scoring read-back ---------------------------


def test_training_exports_and_registers_the_inference_config(
    demo_project: Path, fake_registry: AdapterRegistry
) -> None:
    results = invoke(demo_project, fake_registry)
    assert results.exit_code() == 0
    model = next(r for r in results.results if r.unique_id == MODEL)
    assert model.registration is not None

    tags = json.loads((demo_project / "target/fake_registry/churn_model.json").read_text())[0][
        "tags"
    ]
    assert tags["mbt.inference_config_uri"].startswith("file://")
    assert tags["mbt.inference_config_format"] == "json"
    assert tags["mbt.inference_config_content_hash"].startswith("sha256:")
    assert int(tags["mbt.inference_config_size_bytes"]) > 0

    document = json.loads(
        Path(tags["mbt.inference_config_uri"].removeprefix("file://")).read_text()
    )
    assert document["model"] == "churn_model"
    assert document["identity"]["config_hash"] == tags["mbt.config_hash"]
    assert document["resolved"]["feature_columns"]


def test_the_training_run_carries_the_config_as_a_document(
    demo_project: Path, fake_registry: AdapterRegistry
) -> None:
    """The tracker records WHAT was trained, not just that something was."""
    invoke(demo_project, fake_registry)
    payloads = [
        json.loads(p.read_text()) for p in (demo_project / "target/fake_tracking").glob("*.json")
    ]
    assert payloads and all("inference_config.json" in p["documents"] for p in payloads)


def test_the_run_is_named_for_the_invocation_and_the_model(
    demo_project: Path, fake_registry: AdapterRegistry
) -> None:
    invoke(demo_project, fake_registry)
    payload = next(
        json.loads(p.read_text()) for p in (demo_project / "target/fake_tracking").glob("*.json")
    )
    assert payload["run_name"] == f"{payload['tags']['mbt.run_id']}-churn_model"
    assert payload["tags"]["mbt.project"] == "demo"
    assert payload["experiment"] == "demo"  # no experiment: set, so the project name


def _rewrite_champion_config(project: Path, mutate) -> None:
    """Rewrite the champion's exported inference config, keeping the registry's
    recorded digest in step.

    The store verifies `mbt.artifact_content_hash` on fetch, so editing an
    artifact's bytes without updating the tag is corruption, which is exactly
    what the store is there to refuse. A real re-registration writes both, so
    the fixture does too - otherwise these tests would be asserting on the
    integrity check instead of on the spec-divergence behavior they are about.
    """
    import hashlib

    path = project / "target/fake_registry/churn_model.json"
    entries = json.loads(path.read_text())
    tags = entries[0]["tags"]
    document = Path(tags["mbt.inference_config_uri"].removeprefix("file://"))
    payload = json.dumps(mutate(json.loads(document.read_text())))
    document.write_text(payload)
    tags["mbt.inference_config_content_hash"] = (
        "sha256:" + hashlib.sha256(payload.encode()).hexdigest()
    )
    tags["mbt.inference_config_size_bytes"] = str(len(payload.encode()))
    path.write_text(json.dumps(entries))


def test_scoring_reads_the_model_spec_from_the_champion(
    scoring_project: Path,  # noqa: F811
    fake_registry: AdapterRegistry,
) -> None:
    """The champion's spec is authoritative, so a spec edit that has not been
    promoted does not silently change how the deployed model is fed."""
    _build_and_promote(scoring_project, fake_registry)

    def edit(recorded: dict) -> dict:
        recorded["spec"]["description"] = "the champion's own copy"
        return recorded

    _rewrite_champion_config(scoring_project, edit)

    with recording_bus() as sink:
        results = invoke(scoring_project, fake_registry, "score")

    assert results.exit_code() == 0
    assert not any("predates inference-config export" in m for m in sink.messages())


def test_a_champion_without_a_config_falls_back_loudly(
    scoring_project: Path,  # noqa: F811
    fake_registry: AdapterRegistry,
) -> None:
    """A champion registered before ADR-28 must stay scoreable - the same shape
    ADR-21 gave a champion registered before baselines existed."""
    _build_and_promote(scoring_project, fake_registry)
    path = scoring_project / "target/fake_registry/churn_model.json"
    entries = json.loads(path.read_text())
    entries[0]["tags"].pop("mbt.inference_config_uri")
    path.write_text(json.dumps(entries))

    with recording_bus() as sink:
        results = invoke(scoring_project, fake_registry, "score")

    assert results.exit_code() == 0  # scoring proceeds on the project's spec
    assert any("predates inference-config export" in m for m in sink.messages())


def test_a_champion_whose_spec_diverges_warns_and_still_wins(
    scoring_project: Path,  # noqa: F811
    fake_registry: AdapterRegistry,
) -> None:
    """Between merging a spec edit and promoting it, the two legitimately
    differ (ADR-5); scoring says so and uses the champion's."""
    _build_and_promote(scoring_project, fake_registry)

    def edit(recorded: dict) -> dict:
        recorded["identity"]["config_hash"] = "sha256:something_else"
        return recorded

    _rewrite_champion_config(scoring_project, edit)

    with recording_bus() as sink:
        results = invoke(scoring_project, fake_registry, "score")

    assert results.exit_code() == 0
    assert any("whose spec differs from the project's" in m for m in sink.messages())


def test_a_malformed_inference_config_is_an_error(
    scoring_project: Path,  # noqa: F811
    fake_registry: AdapterRegistry,
) -> None:
    """Not a fallback: a config that exists but carries no spec means the
    document is corrupt, and scoring on a guess is worse than stopping."""
    _build_and_promote(scoring_project, fake_registry)
    _rewrite_champion_config(scoring_project, lambda _: {"schema_version": 1})

    results = invoke(scoring_project, fake_registry, "score")

    assert results.exit_code() == 1
    assert "has no model spec" in (results.results[0].message or "")


def test_a_tracker_without_log_document_still_trains() -> None:
    """log_document is probed with hasattr, like prepare and log_trial: a
    tracker that predates it keeps the tags it always had."""
    from mbt.execute.job import _log_run_documents

    class _OldTracker:
        def log(self, *args: object, **kwargs: object) -> None: ...

    runtime = SimpleNamespace(store=None, job=None)
    _log_run_documents(runtime, _OldTracker(), None, None)  # must not raise


def test_a_failing_log_document_does_not_fail_the_training_run() -> None:
    """A tracker hiccup while attaching a document must not lose a model that
    trained, gated and exported successfully."""
    from mbt.execute.job import _log_run_documents

    class _Boom:
        def log_document(self, run: object, path: Path) -> None:
            raise RuntimeError("tracking down")

    runtime = SimpleNamespace(
        store=SimpleNamespace(fetch=lambda ref: Path("/nonexistent")),
        job=SimpleNamespace(node=SimpleNamespace(hooks_path=None), project_dir="."),
    )
    _log_run_documents(runtime, _Boom(), None, None)  # must not raise


def test_scoring_builds_no_tracking_adapter(
    scoring_project: Path,  # noqa: F811
    fake_registry: AdapterRegistry,
) -> None:
    """An invocation with no model nodes must not warm, create, or otherwise
    touch the tracking backend (ADR-28)."""
    _build_and_promote(scoring_project, fake_registry)
    tracked = sorted(p.name for p in (scoring_project / "target/fake_tracking").glob("*.json"))

    assert invoke(scoring_project, fake_registry, "score").exit_code() == 0

    after = sorted(p.name for p in (scoring_project / "target/fake_tracking").glob("*.json"))
    assert after == tracked
