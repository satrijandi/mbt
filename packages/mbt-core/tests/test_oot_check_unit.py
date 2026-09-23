"""The pre-deploy check's window pinning, at its own seam (C-3).

All eight tests in ``test_oot_check_flow.py`` go through ``run_command`` then
``run_evaluate``, so there was no seam at which "given a version and a manifest,
what windows get pinned" could be asserted. ``pin_check_windows`` is that seam.
"""

from pathlib import Path
from typing import Any

import pytest

from mbt.contracts import OUT_OF_TIME_SPLIT
from mbt.exceptions import StateError
from mbt.execute.oot_check import pin_check_windows

ANCHOR = "2026-07-01T00:00:00Z"
DATASET_UID = "dataset.demo.panel"
MODEL_UID = "model.demo.clf"


def _ctx(monkeypatch: pytest.MonkeyPatch) -> Any:
    from types import SimpleNamespace

    from mbt.contracts import ManifestNode

    def node(uid: str, kind: str, **extra: Any) -> ManifestNode:
        return ManifestNode(
            unique_id=uid,
            resource_type=kind,  # type: ignore[arg-type]
            name=uid.rsplit(".", 1)[-1],
            path=f"{uid}.yml",
            config={},
            config_hash="sha256:same",
            **extra,
        )

    manifest = SimpleNamespace(
        nodes={
            DATASET_UID: node(DATASET_UID, "dataset"),
            MODEL_UID: node(MODEL_UID, "model", resolved={"test_window": ["old", "older"]}),
        },
        metadata=SimpleNamespace(anchor=ANCHOR),
    )
    return SimpleNamespace(manifest=manifest)


def _record(**overrides: Any) -> dict[str, Any]:
    base = {
        "unique_id": DATASET_UID,
        "config_hash": "sha256:same",
        "windows": {"test": ["2026-05-01T00:00:00Z", "2026-06-01T00:00:00Z"]},
        "test_window": ["2026-05-01T00:00:00Z", "2026-06-01T00:00:00Z"],
    }
    base.update(overrides)
    return base


def test_pinning_puts_the_after_test_window_between_test_end_and_the_anchor(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """The whole point of the check: judge the rows that arrived AFTER the
    window the version was evaluated on (ADR-30)."""
    ctx = _ctx(monkeypatch)
    uid, test = pin_check_windows(ctx, MODEL_UID, _record())
    assert uid == DATASET_UID
    windows = ctx.manifest.nodes[DATASET_UID].resolved["windows"]
    assert windows["test"] == test
    assert windows[OUT_OF_TIME_SPLIT] == [test[1], ANCHOR]


def test_pinning_restores_the_versions_own_test_window_on_the_model_node(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    ctx = _ctx(monkeypatch)
    pin_check_windows(ctx, MODEL_UID, _record())
    assert ctx.manifest.nodes[MODEL_UID].resolved["test_window"] == [
        "2026-05-01T00:00:00Z",
        "2026-06-01T00:00:00Z",
    ]


def test_pinning_drops_a_stale_test_window_when_the_version_recorded_none(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """The manifest's current test_window must not leak into a check of a
    version that recorded none - it would judge a window the model never saw."""
    ctx = _ctx(monkeypatch)
    pin_check_windows(ctx, MODEL_UID, _record(test_window=None))
    assert "test_window" not in ctx.manifest.nodes[MODEL_UID].resolved


def test_pinning_refuses_a_dataset_the_manifest_does_not_have(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    ctx = _ctx(monkeypatch)
    with pytest.raises(StateError):
        pin_check_windows(ctx, MODEL_UID, _record(unique_id="dataset.demo.gone"))


def test_a_version_with_no_artifact_cannot_be_checked(tmp_path: Path) -> None:
    """Surfaced by C-3's typed seam.

    The old path patched ``artifact`` onto the job through an untyped
    ``model_copy``, so a version with no loadable artifact produced a job with
    ``artifact=None`` and failed later, somewhere else. Same stance as the
    scoring path: a version that exists but cannot load is an error (ADR-10).
    """
    from types import SimpleNamespace

    from mbt.contracts import ManifestNode, ModelVersion
    from mbt.execute.oot_check import _check

    node = ManifestNode(
        unique_id=MODEL_UID, resource_type="model", name="m", path="m.yml", config={}
    )
    ctx = SimpleNamespace(manifest=SimpleNamespace(nodes={MODEL_UID: node}))
    version = ModelVersion(name="m", version="3", artifact=None, tags={})
    document = {"dataset": {"unique_id": "dataset.d.panel"}, "spec": {}}

    with pytest.raises(StateError, match="no loadable artifact reference"):
        _check(ctx, MODEL_UID, version, document, apply_gates=False)  # type: ignore[arg-type]
