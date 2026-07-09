"""Artifact-store GC: age pruning with a champion/latest-run keep-set."""

import json
import os
import time
from pathlib import Path
from types import SimpleNamespace

import pytest

from mbt.contracts import Stage
from mbt.exceptions import MbtError
from mbt.gc import (
    apply_gc_plan,
    artifact_gc_plan,
    champion_artifact_uris,
    run_results_artifact_uris,
)

NOW = time.time()
OLD = NOW - 90 * 86400


def _store(tmp_path: Path) -> Path:
    root = tmp_path / "artifacts"
    for prefix, age in (("old_orphan", OLD), ("old_champion", OLD), ("fresh", NOW)):
        artifact = root / prefix / "model.bin"
        artifact.parent.mkdir(parents=True)
        artifact.write_bytes(b"x" * 10)
        os.utime(artifact, (age, age))
    return root


def _cutoff():
    from datetime import UTC, datetime, timedelta

    return datetime.now(tz=UTC) - timedelta(days=30)


def test_gc_prunes_old_unreferenced_prefixes_only(tmp_path: Path) -> None:
    root = _store(tmp_path)
    keep = {f"file://{root / 'old_champion' / 'model.bin'}"}
    plan = artifact_gc_plan(f"file://{root}", cutoff=_cutoff(), keep_uris=keep)

    assert [p.name for p in plan.delete] == ["old_orphan"]
    assert sorted(p.name for p in plan.keep) == ["fresh", "old_champion"]
    assert plan.freed_bytes == 10

    apply_gc_plan(plan)
    assert not (root / "old_orphan").exists()
    assert (root / "old_champion" / "model.bin").is_file()  # champions survive
    assert (root / "fresh" / "model.bin").is_file()  # too new to prune


def test_gc_refuses_object_stores() -> None:
    with pytest.raises(MbtError, match="lifecycle"):
        artifact_gc_plan("s3://models/mbt", cutoff=_cutoff(), keep_uris=set())


def test_clean_artifacts_cli_end_to_end(demo_project: Path, fake_registry) -> None:
    """A real build, an aged orphan prefix, then `mbt clean --artifacts-older-than`:
    the orphan goes, the registered champion's artifact survives."""
    import subprocess
    import sys

    from core_helpers import TEST_ANCHOR

    from mbt.execute.orchestrator import InvocationOptions, run_command

    run_command(
        InvocationOptions(command="run", project_dir=demo_project, anchor=TEST_ANCHOR),
        registry=fake_registry,
    )
    store = demo_project / "target" / "artifacts"
    kept_before = {p for p in store.rglob("*") if p.is_file()}
    assert kept_before  # the build exported an artifact

    orphan = store / "orphan_prefix" / "model.bin"
    orphan.parent.mkdir(parents=True)
    orphan.write_bytes(b"stale")
    os.utime(orphan, (OLD, OLD))

    proc = subprocess.run(
        [
            sys.executable,
            "-m",
            "mbt.cli.main",
            "clean",
            "--project-dir",
            str(demo_project),
            "--artifacts-older-than",
            "30d",
        ],
        capture_output=True,
        text=True,
        check=False,
        stdin=subprocess.DEVNULL,
    )
    assert proc.returncode == 0, proc.stdout + proc.stderr
    assert not orphan.parent.exists()
    assert {p for p in store.rglob("*") if p.is_file()} == kept_before


def test_gc_handles_missing_store_and_reads_run_results(tmp_path: Path) -> None:
    plan = artifact_gc_plan(f"file://{tmp_path}/nope", cutoff=_cutoff(), keep_uris=set())
    assert plan.delete == [] and plan.keep == []

    target = tmp_path / "target"
    target.mkdir()
    (target / "run_results.json").write_text(
        json.dumps(
            {
                "results": [
                    {"unique_id": "model.p.m", "artifact": {"uri": "file:///a/model.bin"}},
                    {"unique_id": "dataset.p.d"},
                ]
            }
        )
    )
    assert run_results_artifact_uris(tmp_path) == {"file:///a/model.bin"}
    assert run_results_artifact_uris(tmp_path / "empty") == set()


# -- champion keep-set (ADR-10: GC must never delete a champion artifact) ---------


def _parsed(*registration_names):
    """A stand-in parsed project: one model per name (None = unregistered)."""
    models = {}
    for i, name in enumerate(registration_names):
        registration = None if name is None else SimpleNamespace(name=name)
        models[f"model.p.m{i}"] = SimpleNamespace(spec=SimpleNamespace(registration=registration))
    return SimpleNamespace(models=models)


class _FakeRegistry:
    """get_champion returns a version for keyed (name, stage) pairs only."""

    def __init__(self, champions):
        self._champions = champions  # {(name, Stage): uri | None}

    def get_champion(self, name, stage):
        if (name, stage) not in self._champions:
            return None  # no champion in this stage
        uri = self._champions[(name, stage)]
        artifact = None if uri is None else SimpleNamespace(uri=uri)
        return SimpleNamespace(artifact=artifact)


def test_champion_artifact_uris_protects_every_stage_champion() -> None:
    parsed = _parsed("m1", None)  # m1 registered; the second model is not
    registry = _FakeRegistry(
        {
            ("m1", Stage.STAGING): "file:///store/staging/model.bin",
            ("m1", Stage.PRODUCTION): "file:///store/prod/model.bin",
        }
    )
    # Both live champions are protected; the unregistered model contributes nothing.
    assert champion_artifact_uris(parsed, registry) == {
        "file:///store/staging/model.bin",
        "file:///store/prod/model.bin",
    }


def test_champion_artifact_uris_skips_missing_and_artifactless() -> None:
    parsed = _parsed("m1")
    # A staging champion exists but carries no artifact; no champion in other stages.
    registry = _FakeRegistry({("m1", Stage.STAGING): None})
    assert champion_artifact_uris(parsed, registry) == set()
