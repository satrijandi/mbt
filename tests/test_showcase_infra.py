"""Showcase stack smoke + environment sanity (SHOW-01, SHOW-02, SHOW-15).

Opt-in: MBT_LIVE_SHOWCASE=1 with docker running boots the compose stack from
examples/showcase (SeaweedFS + MLflow + Spark standalone + JupyterLab, all on
one runner image) and proves the services are real before the lifecycle
module trains anything.
"""

import importlib

import pytest
from showcase_utils import SHOWCASE_MARKS, SKIP_REASON

pytestmark = SHOWCASE_MARKS


def test_services_healthy_and_s3_round_trip(showcase_stack) -> None:
    """SHOW-01: every service answers; boto3 round-trips against SeaweedFS."""
    stack = showcase_stack

    # MLflow over HTTP (integration item A2's surface). /health is the
    # documented liveness endpoint; the REST search APIs are POST-only.
    import urllib.request

    with urllib.request.urlopen(stack.mlflow_url() + "/health", timeout=30) as resp:
        assert resp.status == 200

    # Spark master reports one ALIVE worker with resources.
    master = stack.http_json(f"http://localhost:{stack.ports['SHOWCASE_SPARK_UI_PORT']}/json/")
    alive = [w for w in master.get("workers", []) if w.get("state") == "ALIVE"]
    assert alive, f"no ALIVE spark worker: {master}"

    # Real S3 API round-trip (integration item A3's surface), from inside the
    # runner via boto3 env-chain config - exactly how mbt's S3ArtifactStore
    # will talk to it.
    probe = stack.exec(
        "python",
        "-c",
        "import boto3, uuid\n"
        "s3 = boto3.client('s3')\n"
        "key = f'probe/{uuid.uuid4().hex}'\n"
        "s3.put_object(Bucket='mbt-lake', Key=key, Body=b'ping')\n"
        "assert s3.get_object(Bucket='mbt-lake', Key=key)['Body'].read() == b'ping'\n"
        "s3.delete_object(Bucket='mbt-lake', Key=key)\n"
        "print('s3-round-trip-ok')",
        workdir="/workspace",
    )
    assert "s3-round-trip-ok" in probe.stdout

    # The lake was seeded.
    listing = stack.exec(
        "python",
        "-c",
        "import boto3\n"
        "s3 = boto3.client('s3')\n"
        "keys = [o['Key'] for o in s3.list_objects_v2(Bucket='mbt-lake').get('Contents', [])]\n"
        "print('\\n'.join(sorted(keys)))",
        workdir="/workspace",
    )
    for table in ("subscribers", "scoring_batch", "churn_outcomes"):
        assert any(line.startswith(f"{table}/") for line in listing.stdout.splitlines()), (
            f"lake is missing {table}/: {listing.stdout}"
        )


def test_runner_env_sanity_and_sparkling_version_matrix(showcase_stack) -> None:
    """SHOW-02: mbt runs; the h2o client matches the pysparkling-embedded H2O.

    H2O requires an exact client/server version match, and h2o-pysparkling
    embeds its own H2O backend jars - a mismatch here fails later inside
    H2OContext.getOrCreate with an opaque error, so pin it away up front.
    """
    stack = showcase_stack
    version = stack.mbt("--version", timeout=120)
    assert version.stdout.strip(), "mbt --version printed nothing"

    # The CI tier's state-branch scripts run git inside the runner image; a
    # stale locally cached image without it fails distantly in publish_state
    # (rebuild with scripts/build_image.sh --force after image changes).
    git_probe = stack.exec("git", "--version", workdir="/workspace")
    assert git_probe.stdout.startswith("git version"), git_probe.stdout

    probe = stack.exec(
        "python",
        "-c",
        "import importlib.metadata as im\n"
        "import h2o\n"
        "raw = im.version('h2o-pysparkling-3-5')\n"
        "embedded = raw.split('.post')[0]\n"
        "print(f'h2o-client={h2o.__version__} pysparkling-embedded={embedded}')\n"
        "assert h2o.__version__ == embedded, (h2o.__version__, embedded)",
        workdir="/workspace",
    )
    assert "h2o-client=" in probe.stdout


def test_collection_hygiene_and_double_gate() -> None:
    """SHOW-15: the tier is safe to collect everywhere and loud once opted in."""
    for module_name in (
        "test_showcase_ci",
        "test_showcase_infra",
        "test_showcase_k3d",
        "test_showcase_lifecycle",
        "test_showcase_obs",
        "test_showcase_provenance",
        "test_showcase_scheduling",
    ):
        module = importlib.import_module(module_name)
        reasons = [
            mark.kwargs.get("reason")
            for mark in module.pytestmark
            if getattr(mark, "name", "") == "skipif"
        ]
        assert SKIP_REASON in reasons, f"{module_name} lost the opt-in skipif gate"
        names = [getattr(mark, "name", "") for mark in module.pytestmark]
        assert "live_showcase" in names and "live" in names

    # Gate 2: once opted in, a missing docker binary FAILS, never skips.
    import showcase_utils

    with pytest.MonkeyPatch.context() as mp:
        mp.setattr("shutil.which", lambda _: None)
        with pytest.raises(pytest.fail.Exception, match="docker is not on PATH"):
            showcase_utils.require_docker()
