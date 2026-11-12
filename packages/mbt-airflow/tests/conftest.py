"""Fixtures for the mbt-airflow tests."""

import importlib.util
import os
import tempfile
from pathlib import Path

# Importing Airflow creates ``$AIRFLOW_HOME/logs``, and unset that is ~/airflow:
# a developer's home directory is not the test suite's to touch. This must run
# before anything imports airflow, which is why it sits at module level.
os.environ.setdefault("AIRFLOW_HOME", tempfile.mkdtemp(prefix="mbt-airflow-tests-"))

import pytest
from airflow_helpers import compile_churn_demo

from mbt.artifacts.manifest import Manifest, read_manifest

# The floors job installs the whole workspace except Airflow (scripts/install_floors.py
# EXCLUDED: Airflow only supports installs under its own constraints), so there these
# tests are not collected; CI's airflow-floor job runs them against the Airflow floor.
# Where Airflow is installed nothing is skipped, and the 100% coverage gate would
# notice if it ever silently were not.
collect_ignore_glob = [] if importlib.util.find_spec("airflow") else ["test_*.py"]


@pytest.fixture(scope="session")
def churn_demo(tmp_path_factory: pytest.TempPathFactory) -> tuple[Path, Path]:
    """(project dir, compiled manifest path) for the churn_demo fixture project."""
    return compile_churn_demo(tmp_path_factory.mktemp("churn-demo"))


@pytest.fixture(scope="session")
def manifest(churn_demo: tuple[Path, Path]) -> Manifest:
    return read_manifest(churn_demo[1])
