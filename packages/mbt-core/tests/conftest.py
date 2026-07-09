"""Shared fixtures for mbt-core tests."""

from pathlib import Path

import pytest
from core_helpers import write, write_subscriber_data

from mbt.adapters.registry import AdapterRegistry
from mbt.secrets import clear_taints


@pytest.fixture(autouse=True)
def _clean_secrets() -> None:
    clear_taints()


@pytest.fixture()
def fake_registry() -> AdapterRegistry:
    # The 'fake' plugin comes from the mbt-testing package via entry points,
    # so it is also discoverable inside training-job subprocesses.
    return AdapterRegistry()


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
    # Roots are absolutized into tmp_path: relative fake-adapter roots and
    # file:// stores resolve against the process cwd (the repo root under
    # pytest), which littered ./target (FEEDBACK 2.6).
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
        """.replace("./target", f"{tmp_path}/target"),
    )
    write_subscriber_data(tmp_path)
    return tmp_path
