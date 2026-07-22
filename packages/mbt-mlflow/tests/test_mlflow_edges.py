"""mbt-mlflow edge paths: artifact upload on log(), legacy-mode search
failures, and the plugin descriptor (complements test_mlflow_adapter)."""

from pathlib import Path

import pytest
from mbt_mlflow.adapter import MlflowRegistry, MlflowTracking

from mbt_adapter_base import CONTRACT_VERSION, ArtifactRef, ManifestNode, Stage


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


def test_log_uploads_local_file_artifacts_and_tags_the_rest(uri: str, tmp_path: Path) -> None:
    tracking = MlflowTracking({"uri": uri})
    tracking.prepare()
    run = tracking.start_run(_node(), {})

    local = tmp_path / "model.ubj"
    local.write_bytes(b"weights")
    on_disk = ArtifactRef(
        uri=f"file://{local}", format="xgboost_ubj", content_hash="sha256:a", size_bytes=7
    )
    gone = ArtifactRef(
        uri=f"file://{tmp_path}/missing.bin", format="ghost", content_hash="sha256:b", size_bytes=1
    )
    remote = ArtifactRef(
        uri="s3://bucket/model.onnx", format="onnx", content_hash="sha256:c", size_bytes=1
    )
    tracking.log(run, artifacts=[on_disk, gone, remote])

    from mlflow.tracking import MlflowClient

    client = MlflowClient(tracking_uri=uri)
    stored = client.get_run(run.run_id)
    assert stored.data.tags["mbt.artifact.xgboost_ubj"] == on_disk.uri
    assert stored.data.tags["mbt.artifact.ghost"] == gone.uri
    assert stored.data.tags["mbt.artifact.onnx"] == remote.uri
    # only the existing file:// artifact was uploaded to the run
    uploaded = [a.path for a in client.list_artifacts(run.run_id)]
    assert uploaded == ["model.ubj"]


def test_champion_lookup_propagates_a_backend_error_but_not_found_is_none(
    uri: str, monkeypatch: pytest.MonkeyPatch
) -> None:
    """F9: a real backend failure (an unreachable / 500 registry) must NOT be
    swallowed as 'no champion' - that would silently skip the challenger gate on
    a build, or hard-fail `mbt score` with a misleading message. Only a genuine
    RESOURCE_DOES_NOT_EXIST degrades to None."""
    from mlflow.exceptions import MlflowException
    from mlflow.protos.databricks_pb2 import RESOURCE_DOES_NOT_EXIST

    registry = MlflowRegistry({"uri": uri, "use_aliases": False})

    def boom(*args: object, **kwargs: object) -> None:
        raise MlflowException("registry backend unavailable")  # error_code INTERNAL_ERROR

    monkeypatch.setattr(registry.client(), "search_model_versions", boom)
    with pytest.raises(MlflowException, match="backend unavailable"):
        registry.get_champion("m", Stage.STAGING)

    def missing(*args: object, **kwargs: object) -> None:
        raise MlflowException("no such model", error_code=RESOURCE_DOES_NOT_EXIST)

    monkeypatch.setattr(registry.client(), "search_model_versions", missing)
    assert registry.get_champion("m", Stage.STAGING) is None  # genuine absence -> None


def test_get_version_propagates_a_backend_error(uri: str, monkeypatch: pytest.MonkeyPatch) -> None:
    """F9 (the get_version read): a backend error is not misread as 'no such
    version'."""
    from mlflow.exceptions import MlflowException

    registry = MlflowRegistry({"uri": uri})

    def boom(*args: object, **kwargs: object) -> None:
        raise MlflowException("registry backend unavailable")

    monkeypatch.setattr(registry.client(), "get_model_version", boom)
    with pytest.raises(MlflowException, match="backend unavailable"):
        registry.get_version("m", "1")


def test_champion_read_retries_a_transient_lock(uri: str, monkeypatch: pytest.MonkeyPatch) -> None:
    """F9: champion reads now retry a transient locked DB like the writes do,
    instead of the lock being swallowed as 'no champion'."""
    from mlflow.exceptions import MlflowException
    from mlflow.protos.databricks_pb2 import RESOURCE_DOES_NOT_EXIST

    registry = MlflowRegistry({"uri": uri, "use_aliases": True})
    calls = {"n": 0}

    def flaky(*args: object, **kwargs: object) -> None:
        calls["n"] += 1
        if calls["n"] == 1:
            raise MlflowException("(sqlite3.OperationalError) database is locked")
        raise MlflowException("absent now", error_code=RESOURCE_DOES_NOT_EXIST)

    monkeypatch.setattr(registry.client(), "get_model_version_by_alias", flaky)
    assert registry.get_champion("m", Stage.PRODUCTION) is None  # retried past the lock
    assert calls["n"] == 2  # first attempt locked, second attempt resolved


def test_is_missing_distinguishes_absence_from_a_transient_error() -> None:
    """F9: `_is_missing` recognizes both of mlflow's not-found shapes (the
    RESOURCE_DOES_NOT_EXIST code for a missing version and the "... not found"
    message it uses for a missing alias) and rejects the transient errors that
    must never degrade to 'no champion'."""
    from mbt_mlflow.adapter import _is_missing
    from mlflow.exceptions import MlflowException
    from mlflow.protos.databricks_pb2 import RESOURCE_DOES_NOT_EXIST

    assert _is_missing(MlflowException("gone", error_code=RESOURCE_DOES_NOT_EXIST))
    assert _is_missing(MlflowException("Registered model alias STAGING not found."))
    assert not _is_missing(MlflowException("(sqlite3.OperationalError) database is locked"))
    assert not _is_missing(MlflowException("registry backend unavailable"))


def test_plugin_descriptor_wires_tracking_and_registry() -> None:
    from mbt_mlflow.plugin import PLUGIN

    assert PLUGIN.name == "mlflow"
    assert PLUGIN.contract_version == CONTRACT_VERSION
    assert PLUGIN.tracking is MlflowTracking
    assert PLUGIN.registry is MlflowRegistry
    assert PLUGIN.fingerprint_packages == ["mlflow"]


# -- transient locked-DB retry (R2-2) ------------------------------------------------


def test_is_retryable_recognizes_only_a_locked_database() -> None:
    from mbt_mlflow.adapter import _is_retryable

    assert _is_retryable(RuntimeError("(sqlite3.OperationalError) database is locked"))
    assert _is_retryable(RuntimeError("database table is locked: model_versions"))
    assert not _is_retryable(RuntimeError("no such table: registered_models"))


def test_with_retry_retries_a_lock_then_succeeds(monkeypatch: pytest.MonkeyPatch) -> None:
    from mbt_mlflow import adapter

    monkeypatch.setattr(adapter.time, "sleep", lambda _seconds: None)  # skip the backoff
    calls: list[int] = []

    def flaky() -> str:
        calls.append(1)
        if len(calls) < 3:
            raise RuntimeError("database is locked")
        return "ok"

    assert adapter._with_retry(flaky) == "ok"
    assert len(calls) == 3  # two locked attempts, then success


def test_with_retry_does_not_retry_a_non_lock_error() -> None:
    from mbt_mlflow import adapter

    calls: list[int] = []

    def boom() -> None:
        calls.append(1)
        raise ValueError("not a lock")

    with pytest.raises(ValueError, match="not a lock"):
        adapter._with_retry(boom)
    assert len(calls) == 1  # a non-retryable error propagates on the first try


def test_with_retry_is_bounded(monkeypatch: pytest.MonkeyPatch) -> None:
    from mbt_mlflow import adapter

    monkeypatch.setattr(adapter.time, "sleep", lambda _seconds: None)
    calls: list[int] = []

    def always_locked() -> None:
        calls.append(1)
        raise RuntimeError("database is locked")

    with pytest.raises(RuntimeError, match="database is locked"):
        adapter._with_retry(always_locked, attempts=4)
    assert len(calls) == 4  # bounded: it gives up after `attempts` and re-raises


def test_with_retry_jitters_the_backoff(monkeypatch: pytest.MonkeyPatch) -> None:
    """F10: the locked-DB backoff is full-jittered (uniform in [0, base*2**n]),
    so two jobs colliding on the same lock do not re-collide on identical
    deterministic sleeps (the thundering herd the retry exists to break)."""
    from mbt_mlflow import adapter

    bounds: list[tuple[float, float]] = []

    def rec_uniform(lo: float, hi: float) -> float:
        bounds.append((lo, hi))
        return 0.0  # no real sleep

    monkeypatch.setattr(adapter.random, "uniform", rec_uniform)
    calls: list[int] = []

    def flaky() -> str:
        calls.append(1)
        if len(calls) < 3:
            raise RuntimeError("database is locked")
        return "ok"

    assert adapter._with_retry(flaky, base_delay=0.05) == "ok"
    # each retry jitters over the full [0, base_delay * 2**attempt] window
    assert bounds == [(0.0, 0.05), (0.0, 0.10)]


def test_register_retries_a_transient_registry_lock(
    uri: str, monkeypatch: pytest.MonkeyPatch
) -> None:
    """A locked registry mid-registration is retried, not failed hard after the
    GPU-hours of training are already spent (R2-2)."""
    from mbt_mlflow import adapter

    monkeypatch.setattr(adapter.time, "sleep", lambda _seconds: None)
    registry = MlflowRegistry({"uri": uri})
    client = registry.client()
    real_create = client.create_model_version
    calls: list[int] = []

    def flaky_create(*args: object, **kwargs: object) -> object:
        calls.append(1)
        if len(calls) == 1:
            raise RuntimeError("(sqlite3.OperationalError) database is locked")
        return real_create(*args, **kwargs)

    monkeypatch.setattr(client, "create_model_version", flaky_create)
    artifact = ArtifactRef(uri="file:///x", format="pkl", content_hash="sha256:a", size_bytes=1)
    version = registry.register(artifact, "m", {"mbt.gates_passed": "true"})
    assert version.version == "1"  # registered despite the first lock
    assert len(calls) == 2  # one locked attempt, one success
