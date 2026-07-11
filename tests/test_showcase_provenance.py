"""Deployable unit + provenance (SHOW-08/09, DESIGN.md P3).

The prod-build pipeline bakes `project + compiled manifest` into the exact
runner image, pushes it to zot pinned by digest, and oras-pushes
manifest.json + run_results.json as a provenance artifact tagged with the
source sha. This module proves the resulting claims:

- the oras provenance manifest is byte-identical to the mbt-state baseline
  published by the same pipeline run, and carries zero secrets;
- the pulled unit reproduces its manifest: `mbt run --manifest` inside the
  image re-trains with xgboost metrics exactly equal and H2O metrics within
  its documented 0.02 determinism tier (generated_at == anchor, ADR-19);
- the unit is self-checking: a tampered environment refuses to execute the
  manifest (exit 1, env_digest mismatch) and `--allow-env-mismatch`
  downgrades the refusal to a warning (SHOW-09).
"""

import json
import subprocess

import pytest
from showcase_utils import ANCHOR, SHOWCASE_MARKS

pytestmark = SHOWCASE_MARKS

XGB = "model.churn_lake.churn_baseline_xgb"
AUTOML = "model.churn_lake.churn_automl"
H2O_TOLERANCE = 0.02

AWS_ENV = [
    "-e",
    "AWS_ACCESS_KEY_ID=mbtadmin",
    "-e",
    "AWS_SECRET_ACCESS_KEY=mbtsecret",
    "-e",
    "AWS_DEFAULT_REGION=us-east-1",
    "-e",
    "AWS_ENDPOINT_URL_S3=http://seaweedfs:8333",
]

# Forging an mbt-* distribution is the cheapest deterministic tamper: any
# package named mbt-* enters the targeted env_digest (hashing.py).
TAMPER = (
    'sp=$(python3 -c "import site; print(site.getsitepackages()[0])"); '
    'mkdir -p "$sp/mbt_tamper-0.0.1.dist-info"; '
    'printf "Metadata-Version: 2.1\\nName: mbt-tamper\\nVersion: 0.0.1\\n" '
    '> "$sp/mbt_tamper-0.0.1.dist-info/METADATA"; '
    'touch "$sp/mbt_tamper-0.0.1.dist-info/RECORD"; '
)


def _docker_run(ci, image: str, script: str, timeout: int = 900) -> subprocess.CompletedProcess:
    return subprocess.run(
        [
            "docker",
            "run",
            "--rm",
            "--network",
            ci.stack.env["SHOWCASE_NETWORK"],
            *AWS_ENV,
            image,
            "bash",
            "-c",
            script,
        ],
        capture_output=True,
        text=True,
        timeout=timeout,
        check=False,
    )


@pytest.fixture(scope="module")
def unit(showcase_ci) -> dict:
    """The FIRST bake (the bootstrap full build - both models in its
    run_results) plus its provenance artifact and state-branch twin."""
    ci = showcase_ci
    ci.ensure_seeded()

    bumps = [
        c
        for c in reversed(ci.deploy_commits())
        if c["commit"]["message"].startswith("deploy: unit digest from ")
    ]
    assert bumps, "no deploy digest commit - the bake never ran"
    first = bumps[0]
    source_sha = first["commit"]["message"].strip().rsplit(" ", 1)[-1]
    image = ci.images_env(ref=first["sha"])["IMAGE"]
    assert "@sha256:" in image, image

    provenance = ci.provenance_files(source_sha)
    assert set(provenance) == {"target/manifest.json", "target/run_results.json"}, provenance

    state_twin = next((c for c in ci.state_commits() if source_sha in c["commit"]["message"]), None)
    assert state_twin, f"no mbt-state commit for {source_sha}"

    pull = subprocess.run(
        ["docker", "pull", image], capture_output=True, text=True, timeout=600, check=False
    )
    assert pull.returncode == 0, f"pulling the unit from zot failed:\n{pull.stderr}"

    return {
        "ci": ci,
        "image": image,
        "source_sha": source_sha,
        "provenance": provenance,
        "state_manifest_bytes": ci.raw_file("churn", "manifest.json", state_twin["sha"]),
    }


def test_provenance_matches_published_baseline(unit) -> None:
    """SHOW-08: the oras artifact IS the baseline, and it is secret-free."""
    manifest_bytes = unit["provenance"]["target/manifest.json"]
    assert manifest_bytes == unit["state_manifest_bytes"], (
        "provenance manifest diverged from the mbt-state baseline of the same run"
    )

    manifest = json.loads(manifest_bytes)
    assert manifest["metadata"]["generated_at"] == ANCHOR
    assert manifest["metadata"]["git"]["commit"] == unit["source_sha"]

    # Secret-free by construction (taint redaction): the committed demo
    # values and the CI token must not appear anywhere in the manifest.
    text = manifest_bytes.decode()
    assert "mbtsecret" not in text
    assert unit["ci"].gitea_token not in text

    results = json.loads(unit["provenance"]["target/run_results.json"])
    trained = {r["unique_id"] for r in results["results"]}
    assert {XGB, AUTOML} <= trained, trained


def test_unit_reproduces_its_manifest(unit) -> None:
    """SHOW-08: pull the unit, `mbt run --manifest`, metrics reproduce."""
    ci = unit["ci"]

    baked = _docker_run(ci, unit["image"], "cat target/manifest.json", timeout=120)
    assert baked.returncode == 0, baked.stderr
    assert baked.stdout.encode() == unit["provenance"]["target/manifest.json"], (
        "the unit ships a different manifest than its provenance artifact"
    )

    repro = _docker_run(
        ci,
        unit["image"],
        "mbt run --manifest target/manifest.json 1>&2 && cat target/run_results.json",
    )
    assert repro.returncode == 0, f"reproduction failed:\n{repro.stderr[-4000:]}"
    reproduced = {
        r["unique_id"]: r.get("metrics") or {} for r in json.loads(repro.stdout)["results"]
    }
    baseline = {
        r["unique_id"]: r.get("metrics") or {}
        for r in json.loads(unit["provenance"]["target/run_results.json"])["results"]
    }

    # xgboost is bit-exact; H2O's documented determinism tier is 0.02.
    assert reproduced[XGB] == baseline[XGB], (reproduced[XGB], baseline[XGB])
    for metric, value in baseline[AUTOML].items():
        assert abs(reproduced[AUTOML][metric] - value) <= H2O_TOLERANCE, (
            metric,
            reproduced[AUTOML][metric],
            value,
        )


def test_env_drift_refused_then_downgraded(unit) -> None:
    """SHOW-09: the unit is self-checking (ADR-19)."""
    ci = unit["ci"]

    tampered = _docker_run(
        ci,
        unit["image"],
        TAMPER + "mbt run --manifest target/manifest.json --select churn_baseline_xgb",
        timeout=300,
    )
    assert tampered.returncode == 1, (tampered.returncode, tampered.stderr[-2000:])
    assert "does not match the manifest's env_digest" in tampered.stderr, tampered.stderr[-2000:]

    downgraded = _docker_run(
        ci,
        unit["image"],
        TAMPER
        + "mbt run --manifest target/manifest.json --select churn_baseline_xgb "
        + "--allow-env-mismatch",
        timeout=600,
    )
    assert downgraded.returncode == 0, downgraded.stderr[-2000:]
    assert "does not match the manifest's env_digest" in downgraded.stderr, (
        "the downgrade should still WARN about the mismatch"
    )
