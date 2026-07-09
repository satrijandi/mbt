"""Repo-level test fixtures: the churn_demo example project."""

import shutil
from pathlib import Path

import pytest
from e2e_utils import CHURN_DEMO


@pytest.fixture()
def demo_copy(tmp_path: Path) -> Path:
    """A disposable copy of examples/churn_demo (keeps the repo clean)."""
    destination = tmp_path / "churn_demo"
    shutil.copytree(
        CHURN_DEMO,
        destination,
        ignore=shutil.ignore_patterns("target", "mlflow.db", "predictions"),
    )
    return destination
