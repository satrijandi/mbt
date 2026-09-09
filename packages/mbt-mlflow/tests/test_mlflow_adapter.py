"""mbt-mlflow against a local sqlite backend (S3-04)."""

import subprocess
import sys
from pathlib import Path

import pytest
from mbt_mlflow.adapter import MlflowRegistry, MlflowTracking

from mbt_adapter_base import ArtifactRef, ManifestNode, Stage


@pytest.fixture()
def uri(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> str:
    # mlflow drops its default artifact root (./mlruns) relative to the cwd;
    # chdir keeps it inside tmp instead of littering the repo root.
    monkeypatch.chdir(tmp_path)
    return f"sqlite:///{tmp_path}/mlflow.db"


def _node() -> ManifestNode:
    return ManifestNode(
        unique_id="model.demo.m",
        resource_type="model",
        name="m",
        path="models/m.yml",
        config={},
    )


def _scoring_node() -> ManifestNode:
    return ManifestNode(
        unique_id="scoring.demo.s",
        resource_type="scoring",
        name="s",
        path="scoring/s.yml",
        config={},
    )


def _experiment_of(uri: str, run_id: str) -> str:
    """The NAME of the experiment a run landed in."""
    from mlflow.tracking import MlflowClient

    client = MlflowClient(tracking_uri=uri)
    return str(client.get_experiment(client.get_run(run_id).info.experiment_id).name)


def _artifact(tmp_path: Path) -> ArtifactRef:
    path = tmp_path / "model.bin"
    path.write_bytes(b"weights")
    return ArtifactRef(
        uri=f"file://{path}", format="test_bin", content_hash="sha256:abc", size_bytes=7
    )


def test_tracking_round_trip_with_identity_tags(uri: str) -> None:
    tracking = MlflowTracking({"uri": uri})
    tracking.prepare()
    run = tracking.start_run(_node(), {"mbt.config_hash": "sha256:x", "mbt.run_id": "r1"})
    tracking.log(run, params={"max_depth": "4"}, metrics={"pr_auc": 0.41})
    tracking.log(tracking.resume(run.run_id), tags={"mbt.gates_passed": "true"})
    tracking.end_run(run, "FINISHED")

    from mlflow.tracking import MlflowClient

    stored = MlflowClient(tracking_uri=uri).get_run(run.run_id)
    assert stored.data.tags["mbt.config_hash"] == "sha256:x"
    assert stored.data.tags["mbt.gates_passed"] == "true"
    assert stored.data.params["max_depth"] == "4"
    assert stored.data.metrics["pr_auc"] == 0.41
    assert stored.info.status == "FINISHED"


def test_nested_tuning_trials(uri: str) -> None:
    tracking = MlflowTracking({"uri": uri})
    tracking.prepare()
    run = tracking.start_run(_node(), {})
    tracking.log_trial(run, 0, {"max_depth": 3}, 0.39)
    tracking.log_trial(run, 1, {"max_depth": 5}, 0.42)

    from mlflow.tracking import MlflowClient

    client = MlflowClient(tracking_uri=uri)
    experiment = client.get_experiment_by_name("mbt")
    children = client.search_runs(
        [experiment.experiment_id],
        filter_string=f"tags.\"mlflow.parentRunId\" = '{run.run_id}'",
    )
    assert len(children) == 2
    assert {r.data.metrics["objective"] for r in children} == {0.39, 0.42}


# -- one experiment, timestamped runs (ADR-28) --------------------------------


def test_every_run_lands_in_the_one_configured_experiment(uri: str) -> None:
    """ADR-26's per-kind split is gone with the serving runs it separated:
    only training opens a tracking run now, so there is one namespace."""
    tracking = MlflowTracking({"uri": uri, "experiment": "churn_demo_wide_v2"})
    tracking.prepare()

    run = tracking.start_run(_node(), {})

    assert _experiment_of(uri, run.run_id) == "churn_demo_wide_v2"


def test_prepare_creates_the_experiment_up_front(uri: str) -> None:
    """Parallel training jobs must never race to create it."""
    from mlflow.tracking import MlflowClient

    MlflowTracking({"uri": uri, "experiment": "churn_demo"}).prepare()

    assert MlflowClient(tracking_uri=uri).get_experiment_by_name("churn_demo") is not None


def test_the_run_name_is_the_one_core_composed(uri: str) -> None:
    """Core composes ``<run_id>-<model>`` and passes it as ``mbt.run_name``
    (ADR-28); this maps it onto MLflow's own tag, so two trainings of one model
    no longer produce two runs with the same name."""
    tracking = MlflowTracking({"uri": uri})
    run = tracking.start_run(_node(), {"mbt.run_name": "20260909T101500Z-a1b2c3d4-m"})

    from mlflow.tracking import MlflowClient

    stored = MlflowClient(tracking_uri=uri).get_run(run.run_id)
    assert stored.data.tags["mlflow.runName"] == "20260909T101500Z-a1b2c3d4-m"


def test_the_node_name_is_the_run_name_when_core_composed_none(uri: str) -> None:
    """A caller driving the contract directly still gets mbt's old behavior
    rather than an unnamed run."""
    tracking = MlflowTracking({"uri": uri})
    run = tracking.start_run(_node(), {})

    from mlflow.tracking import MlflowClient

    assert MlflowClient(tracking_uri=uri).get_run(run.run_id).data.tags["mlflow.runName"] == "m"


def test_log_document_uploads_a_local_file(uri: str, tmp_path: Path) -> None:
    """The seam that puts the inference config on a run even when the model
    artifact store is s3:// and ``log(artifacts=...)`` uploads nothing."""
    from mlflow.tracking import MlflowClient

    tracking = MlflowTracking({"uri": uri})
    run = tracking.start_run(_node(), {})
    document = tmp_path / "inference_config.json"
    document.write_text('{"schema_version": 1}')

    tracking.log_document(run, document)

    listed = MlflowClient(tracking_uri=uri).list_artifacts(run.run_id)
    assert [a.path for a in listed] == ["inference_config.json"]


def test_a_per_node_kind_mapping_is_rejected(uri: str) -> None:
    """The ADR-26 config shape must fail loudly rather than have mbt guess
    which of its two names to honour."""
    with pytest.raises(ValueError, match="no longer takes a mapping keyed by node kind"):
        MlflowTracking({"uri": uri, "experiment": {"model": "a", "scoring": "b"}})


def test_a_non_name_experiment_is_rejected(uri: str) -> None:
    with pytest.raises(ValueError, match="experiment must be a name"):
        MlflowTracking({"uri": uri, "experiment": ["a", "b"]})


def test_a_scoring_node_still_resolves_the_one_experiment(uri: str) -> None:
    """mbt no longer opens scoring runs, but the adapter is a public contract:
    a third-party caller that starts one gets the configured experiment rather
    than an error about node kinds."""
    tracking = MlflowTracking({"uri": uri, "experiment": "churn_demo"})
    run = tracking.start_run(_scoring_node(), {})

    assert _experiment_of(uri, run.run_id) == "churn_demo"


def test_registry_champion_flow_defaults_to_aliases(uri: str, tmp_path: Path) -> None:
    """The default flow uses registered-model aliases (stages are deprecated
    upstream and removed in MLflow 4); mbt's one-stage-per-version semantics
    must hold on the alias backend."""
    registry = MlflowRegistry({"uri": uri})
    assert registry.use_aliases
    artifact = _artifact(tmp_path)
    metadata = {"mbt.gates_passed": "true", "mbt.tracking_run_id": ""}
    v1 = registry.register(artifact, "m", metadata)
    assert v1.version == "1"

    assert registry.get_champion("m", Stage.STAGING) is None
    registry.transition(v1, Stage.STAGING)
    champion = registry.get_champion("m", Stage.STAGING)
    assert champion is not None and champion.version == "1"
    assert champion.artifact is not None
    assert champion.artifact.uri == artifact.uri  # reconstructed from tags
    assert champion.tags["mbt.gates_passed"] == "true"

    # promoting vacates the staging slot: canonical aliases are exclusive
    registry.transition(v1, Stage.PRODUCTION)
    assert registry.get_champion("m", Stage.PRODUCTION).version == "1"
    assert registry.get_champion("m", Stage.STAGING) is None
    assert registry.get_version("m", "1") is not None
    assert registry.get_version("m", "99") is None


def test_registry_alias_exclusivity_and_stage_derivation(uri: str, tmp_path: Path) -> None:
    registry = MlflowRegistry({"uri": uri})
    v1 = registry.register(_artifact(tmp_path), "m", {})
    registry.transition(v1, Stage.STAGING)
    registry.transition(v1, Stage.PRODUCTION)

    from mlflow.tracking import MlflowClient

    mv = MlflowClient(tracking_uri=uri, registry_uri=uri).get_model_version("m", "1")
    assert set(mv.aliases) == {"production"}  # staging alias dropped on promote

    resolved = registry.get_version("m", "1")
    assert resolved is not None and resolved.stage is Stage.PRODUCTION  # from the alias


def test_promoting_a_new_champion_archives_the_outgoing_one(uri: str, tmp_path: Path) -> None:
    """Promoting a new version to production displaces the old champion, which
    becomes discoverable as archived (queryable for rollback) - identically on
    the alias and stage backends (archive_existing_versions parity)."""
    for use_aliases in (True, False):
        registry = MlflowRegistry({"uri": uri, "use_aliases": use_aliases})
        name = f"m_{use_aliases}"
        v1 = registry.register(_artifact(tmp_path), name, {})
        v2 = registry.register(_artifact(tmp_path), name, {})
        registry.transition(v1, Stage.PRODUCTION)
        assert registry.get_champion(name, Stage.ARCHIVED) is None  # nothing displaced yet

        registry.transition(v2, Stage.PRODUCTION)
        assert registry.get_champion(name, Stage.PRODUCTION).version == v2.version
        archived = registry.get_champion(name, Stage.ARCHIVED)
        assert archived is not None and archived.version == v1.version

        # Re-promoting the current champion is idempotent: it does not archive
        # itself, and the previously displaced version stays the archived one.
        registry.transition(v2, Stage.PRODUCTION)
        assert registry.get_champion(name, Stage.PRODUCTION).version == v2.version
        assert registry.get_champion(name, Stage.ARCHIVED).version == v1.version


def test_transition_archives_the_outgoing_champion_even_through_a_lock(
    uri: str, tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    """F8: a transient locked DB mid-transition must not drop the displaced
    champion's archive. The whole-method retry used to re-read the production
    alias AFTER it had already moved to the new version, so `outgoing` became
    the new version and the true previous champion was never archived; retrying
    each write individually keeps `outgoing` fixed so the archive still lands."""
    from mlflow.exceptions import MlflowException

    registry = MlflowRegistry({"uri": uri, "use_aliases": True})
    v1 = registry.register(_artifact(tmp_path), "m", {})
    v2 = registry.register(_artifact(tmp_path), "m", {})
    registry.transition(v1, Stage.PRODUCTION)

    client = registry.client()
    real_set = client.set_registered_model_alias
    archive_attempts = {"n": 0}

    def flaky_set(name: str, alias: str, version: str) -> None:
        # Fail the archive write once, mid-transition, with a transient lock.
        if alias == Stage.ARCHIVED.value:
            archive_attempts["n"] += 1
            if archive_attempts["n"] == 1:
                raise MlflowException("(sqlite3.OperationalError) database is locked")
        return real_set(name, alias, version)

    monkeypatch.setattr(client, "set_registered_model_alias", flaky_set)
    registry.transition(v2, Stage.PRODUCTION)

    assert registry.get_champion("m", Stage.PRODUCTION).version == v2.version
    archived = registry.get_champion("m", Stage.ARCHIVED)
    assert archived is not None and archived.version == v1.version  # displaced champion archived
    assert archive_attempts["n"] == 2  # the archive write was retried, not skipped by a re-read


def test_registry_legacy_stage_mode(uri: str, tmp_path: Path) -> None:
    """``use_aliases: false`` keeps the stage API for MLflow < 2.9 servers."""
    registry = MlflowRegistry({"uri": uri, "use_aliases": False})
    artifact = _artifact(tmp_path)
    v1 = registry.register(artifact, "m", {"mbt.gates_passed": "true"})

    assert registry.get_champion("m", Stage.STAGING) is None
    registry.transition(v1, Stage.STAGING)
    champion = registry.get_champion("m", Stage.STAGING)
    assert champion is not None and champion.version == "1"
    assert champion.stage is Stage.STAGING

    registry.transition(v1, Stage.PRODUCTION)
    assert registry.get_champion("m", Stage.PRODUCTION).version == "1"
    assert registry.get_champion("m", Stage.STAGING) is None


def test_plugin_import_hygiene() -> None:
    probe = (
        "import sys\n"
        "import mbt_mlflow.plugin\n"
        "assert 'mlflow' not in sys.modules, 'mlflow imported at plugin load (ADR-14)'\n"
    )
    subprocess.run([sys.executable, "-c", probe], check=True)
