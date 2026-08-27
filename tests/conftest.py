"""Repo-level test fixtures: the churn_demo fixture project + showcase stack."""

import os
import shutil
from pathlib import Path

import pytest
from e2e_utils import CHURN_DEMO, REVENUE_DEMO


@pytest.fixture()
def demo_copy(tmp_path: Path) -> Path:
    """A disposable copy of tests/fixtures/churn_demo (keeps the repo clean)."""
    destination = tmp_path / "churn_demo"
    shutil.copytree(
        CHURN_DEMO,
        destination,
        ignore=shutil.ignore_patterns("target", "mlflow.db", "predictions"),
    )
    return destination


@pytest.fixture()
def revenue_copy(tmp_path: Path) -> Path:
    """A disposable copy of tests/fixtures/revenue_demo (the regression fixture)."""
    destination = tmp_path / "revenue_demo"
    shutil.copytree(
        REVENUE_DEMO,
        destination,
        ignore=shutil.ignore_patterns("target", "mlflow.db", "predictions"),
    )
    return destination


@pytest.fixture(scope="session")
def showcase_stack(tmp_path_factory: pytest.TempPathFactory):
    """The docker-compose showcase (examples/showcase), booted once per session.

    Only ever requested by MBT_LIVE_SHOWCASE-gated modules, so the docker
    probing and image build stay out of ordinary runs. Lazy import keeps
    collection light for the fast suite.
    """
    import showcase_utils

    showcase_utils.require_docker()
    showcase_utils.build_runner_image()
    stack = showcase_utils.ComposeStack(tmp_path_factory.mktemp("showcase-ws"))
    try:
        stack.up()
        stack.seed_lake()
        yield stack
    finally:
        # Debug escape hatch: MBT_SHOWCASE_KEEP=1 leaves the stack running
        # for post-mortems (tear down manually: docker compose -p <name> ...
        # down -v --remove-orphans).
        if os.environ.get("MBT_SHOWCASE_KEEP") == "1":
            print(f"\nMBT_SHOWCASE_KEEP=1: stack {stack.project_name} left running")
        else:
            stack.down()


@pytest.fixture(scope="session")
def showcase_ci(showcase_stack, tmp_path_factory: pytest.TempPathFactory):
    """The seeded CI forge on top of the stack (gitea org/repos, OAuth app,
    woodpecker activation, secrets), shared by the CI, provenance, and
    scheduling modules. Lazy: only CI-tier modules pay the bootstrap."""
    import showcase_utils

    return showcase_utils.bootstrap_ci(showcase_stack, tmp_path_factory.mktemp("showcase-ci"))
