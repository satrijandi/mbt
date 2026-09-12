"""Hermetic guard for the showcase's object-store data plane (DESIGN.md 11, P8).

examples/showcase runs one project over four data planes: the SeaweedFS lake
through Spark (`dev`/`ci`/`prod`, training only), a synced DuckDB copy
(`prod_score`, the batch legs), the object store end to end (`seaweedfs`), and
Snowflake (`snowflake`). The live proof of the third lives in
tests/test_showcase_seaweedfs.py, which needs the docker stack.

These tests need neither docker nor an account, which is the point: the repo
battery cannot otherwise see examples/showcase/project at all. A full
five-phase battery once went green while every showcase spec was broken,
because `-m e2e` excludes the live_showcase tier and `-m "not e2e"` deselects
it. Anything load-bearing about a showcase profile belongs here too.
"""

import os
from pathlib import Path

import pytest

REPO_ROOT = Path(__file__).resolve().parent.parent
PROJECT = REPO_ROOT / "examples" / "showcase" / "project"
LAKE_BUCKET = "s3://mbt-lake"


def _plane(target: str):
    """(project vars, loaded target) for one showcase profile target."""
    from mbt.config.profiles import load_profiles
    from mbt.parsing import parse_project

    parsed = parse_project(PROJECT)
    loaded = load_profiles(
        parsed.project.name, PROJECT, target_override=target, project_vars=parsed.project.vars
    )
    return parsed.project.vars, loaded.target


@pytest.fixture
def s3a_env(monkeypatch):
    """profiles.yml renders WHOLE for whichever target is picked, and the
    shared s3a anchor calls env_var() with no default, so every target needs
    this pair present - even the ones that never touch s3a."""
    monkeypatch.setenv("AWS_ACCESS_KEY_ID", "stub")
    monkeypatch.setenv("AWS_SECRET_ACCESS_KEY", "stub")


def test_the_seaweedfs_target_reads_the_object_store_directly(s3a_env) -> None:
    """No copy step anywhere in this plane's loop.

    `prod_score` serves the batch legs from /workspace/lake_local, which
    bootstrap/sync_lake.py mirrors out of the bucket - so a broken s3a read at
    score time looks exactly like a working one there. This target is what
    makes that leg observable, and the settings below are what make it an
    object-store read: the spark adapter, a bucket root, and a
    predictions_root to stage runs under (ADR-23 v1). Any of them drifting
    back to a local path silently turns this plane into a second prod_score.
    """
    _, target = _plane("seaweedfs")
    config = target.data.config

    assert target.data.adapter == "spark"
    assert config["root"] == LAKE_BUCKET
    # Both addresses are declared on every source, and spark is the one adapter
    # that can read either, so it refuses to guess (mbt_spark's
    # resolve_source_address). Unset here fails the compile, it does not fall
    # back - which is the designed behaviour, and how the lake planes stopped
    # silently reading catalog tables that do not exist.
    assert config["source_address"] == "path"
    assert config["conf"]["spark.hadoop.fs.s3a.endpoint"] == "http://seaweedfs:8333"
    # Staged under the shared bind mount: the runs must outlive the container,
    # and must never land in the project dir, whose target/ is wiped (F20).
    predictions_root = Path(config["predictions_root"])
    assert predictions_root.is_absolute() and predictions_root.parts[1] == "workspace"
    assert "target" not in predictions_root.parts

    # In-network service names: this plane runs IN the stack, unlike the
    # host-run warehouse plane, which reaches published ports on localhost.
    assert target.registry.config["uri"] == "http://mlflow:5000"
    assert target.tracking.config["uri"] == "http://mlflow:5000"
    # Its own experiment, so the planes' histories stay separately comparable
    # rather than interleaved (ADR-28 composes <project>__<experiment>).
    assert target.tracking.config["experiment"] == "seaweedfs"


def test_each_data_plane_registers_and_stores_under_its_own_namespace(s3a_env) -> None:
    """Two planes training one spec must not interleave in the shared registry.

    plane_suffix is what keeps their versions apart, and a suffix reused by two
    planes would be worse than none: champion resolution would silently start
    mixing models trained through different adapters, and it would look fine.
    The artifact prefixes go the same way - one store, one prefix per plane.
    """
    suffixes: dict[str, str] = {}
    stores: dict[str, str] = {}
    for name in ("dev", "ci", "prod", "prod_score", "seaweedfs", "snowflake"):
        project_vars, target = _plane(name)
        suffixes[name] = {**project_vars, **target.vars}["plane_suffix"]
        stores[name] = target.artifact_store

    # The lake targets share the empty suffix on purpose: they are one plane
    # seen from four angles (inner loop, PR check, cluster, batch) and register
    # the same names by design.
    assert {suffixes[name] for name in ("dev", "ci", "prod", "prod_score")} == {""}
    assert suffixes["seaweedfs"] == "_seaweedfs"
    assert suffixes["snowflake"] == "_snowflake"

    assert stores["seaweedfs"] == "s3://mbt-artifacts/churn_seaweedfs"
    # One store, one prefix per plane: sharing a prefix would interleave the
    # planes' champion objects behind identical paths.
    assert len({stores[name] for name in ("dev", "seaweedfs", "snowflake")}) == 3, stores


def test_the_seaweedfs_plane_needs_nothing_but_the_stack(s3a_env, monkeypatch) -> None:
    """It is the second plane the DEFAULT showcase tier can exercise.

    The warehouse plane is triple-gated behind an account, so the showcase's
    "switching planes is just --target" claim used to be provable only on a
    machine with Snowflake credentials. This target must therefore resolve to
    complete, in-network config with no SNOWFLAKE_* or SHOWCASE_* lookup at
    all: docker and nothing else.
    """
    for name in list(os.environ):
        if name.startswith(("SNOWFLAKE_", "SHOWCASE_")):
            monkeypatch.delenv(name, raising=False)

    _, target = _plane("seaweedfs")
    endpoints = [
        target.registry.config["uri"],
        target.tracking.config["uri"],
        target.data.config["conf"]["spark.hadoop.fs.s3a.endpoint"],
    ]
    # An env lookup that silently defaulted to a host port would show up here
    # as localhost, and would be unreachable from the runner container.
    assert all("localhost" not in uri for uri in endpoints), endpoints
    assert all(uri for uri in endpoints), endpoints
