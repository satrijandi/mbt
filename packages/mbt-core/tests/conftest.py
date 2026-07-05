"""Shared fixtures for mbt-core tests."""

import textwrap
from datetime import UTC, datetime, timedelta
from pathlib import Path

import pyarrow as pa
import pyarrow.parquet as pq
import pytest

from mbt.adapters.registry import AdapterRegistry
from mbt.secrets import clear_taints

#: Fixed anchor used across compile tests (matches the fixture data range).
TEST_ANCHOR = datetime(2026, 7, 1, tzinfo=UTC)


def write_subscriber_data(project_dir: Path, n_rows: int = 400) -> Path:
    """Deterministic sample parquet spanning ~200 days before TEST_ANCHOR."""
    base = TEST_ANCHOR.replace(tzinfo=None) - timedelta(days=200)
    rows = {
        "user_id": list(range(n_rows)),
        "snapshot_date": [base + timedelta(days=(i * 199) % 200) for i in range(n_rows)],
        "is_active": [i % 10 != 0 for i in range(n_rows)],
        "tenure_days": [30 + (i * 7) % 900 for i in range(n_rows)],
        "monthly_usage": [round((i * 13.7) % 500, 2) for i in range(n_rows)],
        "plan_type": [("basic", "pro", "enterprise")[i % 3] for i in range(n_rows)],
        "churned": [1 if (i * 31) % 100 < 22 else 0 for i in range(n_rows)],
    }
    out = project_dir / "data" / "subscribers"
    out.mkdir(parents=True, exist_ok=True)
    path = out / "part-000.parquet"
    pq.write_table(pa.table(rows), path)
    return path


@pytest.fixture(autouse=True)
def _clean_secrets() -> None:
    clear_taints()


@pytest.fixture()
def fake_registry() -> AdapterRegistry:
    # The 'fake' plugin comes from the mbt-testing package via entry points,
    # so it is also discoverable inside training-job subprocesses.
    return AdapterRegistry()


def write(path: Path, content: str) -> Path:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(textwrap.dedent(content).lstrip())
    return path


@pytest.fixture()
def demo_project(tmp_path: Path) -> Path:
    """A small valid project using the fake adapter."""
    write(
        tmp_path / "mbt_project.yml",
        """
        name: demo
        version: "1.0"
        vars:
          default_threshold: 0.4
        """,
    )
    write(
        tmp_path / "sources.yml",
        """
        sources:
          - name: lakehouse
            tables:
              - name: subscribers
                path: data/subscribers/*.parquet
        """,
    )
    write(
        tmp_path / "datasets/churn_training.yml",
        """
        datasets:
          - name: churn_training
            source: source('lakehouse', 'subscribers')
            label:
              column: churned
            filters: ["is_active = true"]
            split:
              strategy: temporal
              time_column: snapshot_date
              train: "-180d:-28d"
              test: "-28d:now"
            checks: [class_balance_report]
            tags: [churn]
        """,
    )
    write(
        tmp_path / "models/churn_model.yml",
        """
        models:
          - name: churn_model
            task: binary_classification
            adapter: fake
            owner: ds@example.com
            tags: [churn, weekly]
            dataset: ref('churn_training')
            target: churned
            hyperparameters:
              max_depth: 4
              fake_metric_value: 0.61
            evaluation:
              protocol: {split: temporal}
              metrics: [pr_auc, roc_auc]
              gates:
                - metric: pr_auc
                  threshold: "{{ var('default_threshold') }}"
            registration:
              name: churn_model
            seed: 42
        """,
    )
    write(
        tmp_path / "profiles.yml",
        """
        demo:
          target: dev
          outputs:
            dev:
              data: {adapter: local, config: {root: .}}
              tracking: {adapter: fake, config: {root: ./target/fake_tracking}}
              registry: {adapter: fake, config: {root: ./target/fake_registry}}
              compute: {adapter: fake}
              artifact_store: file://./target/artifacts
              vars: {sample_fraction: 1.0}
            prod:
              data: {adapter: local, config: {root: .}}
              tracking: {adapter: fake, config: {root: ./target/fake_tracking}}
              registry: {adapter: fake, config: {root: ./target/fake_registry}}
              compute: {adapter: fake}
              artifact_store: file://./target/artifacts
              threads: 4
              vars: {sample_fraction: 1.0}
        """,
    )
    write_subscriber_data(tmp_path)
    return tmp_path
