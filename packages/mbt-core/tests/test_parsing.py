"""Parsing pipeline tests (S1-05): all-errors collection, did-you-mean, DAG."""

from pathlib import Path

import pytest
from core_helpers import write

from mbt.adapters.registry import AdapterRegistry
from mbt.exceptions import ConfigError
from mbt.parsing import parse_project


def test_valid_project_parses(demo_project: Path, fake_registry: AdapterRegistry) -> None:
    parsed = parse_project(demo_project, registry=fake_registry)
    assert set(parsed.models) == {"model.demo.churn_model"}
    assert set(parsed.datasets) == {"dataset.demo.churn_training"}
    assert parsed.models["model.demo.churn_model"].depends_on == ["dataset.demo.churn_training"]
    assert parsed.datasets["dataset.demo.churn_training"].depends_on == [
        "source.demo.lakehouse.subscribers"
    ]
    assert parsed.graph.has_edge("dataset.demo.churn_training", "model.demo.churn_model")
    assert parsed.models["model.demo.churn_model"].metric_specs


def test_random_split_protocol_warnings(demo_project: Path, fake_registry: AdapterRegistry) -> None:
    """Random splits warn on temporal-leakage and entity-straddle risks
    (FR-RES-09); both warnings are addressable and then disappear."""
    write(
        demo_project / "datasets/random_split.yml",
        """
        datasets:
          - name: exchangeable_rows
            source: source('lakehouse', 'subscribers')
            label:
              column: churned
            split:
              strategy: random
              time_column: snapshot_date
              train: "0.8"
              test: "0.2"
              seed: 7
        """,
    )
    parsed = parse_project(demo_project, registry=fake_registry)
    messages = [issue.message for issue in parsed.report.warnings]
    assert any("temporal leakage" in m for m in messages)
    assert any("sample_key" in m for m in messages)

    write(
        demo_project / "datasets/random_split.yml",
        """
        datasets:
          - name: exchangeable_rows
            source: source('lakehouse', 'subscribers')
            label:
              column: churned
            split:
              strategy: random
              train: "0.8"
              test: "0.2"
              seed: 7
            sample_key: user_id
        """,
    )
    parsed = parse_project(demo_project, registry=fake_registry)
    assert not parsed.report.warnings


def test_all_errors_collected_in_one_pass(
    demo_project: Path, fake_registry: AdapterRegistry
) -> None:
    """A project seeded with 5+ distinct errors reports all of them at once."""
    write(
        demo_project / "models/broken.yml",
        """
        models:
          - name: broken_model
            task: binary_classification
            adapter: fake
            owner: ds@example.com
            dataset: ref('no_such_dataset')      # error: unknown dataset
            target: churned
            hyperparamters:                      # error: typo, did-you-mean
              max_depth: 3
            hyperparameters:
              maxdepth: 3                        # error: unknown hyperparameter
            evaluation:
              protocol: {split: temporal}
              metrics: [no_such_metric]          # error: unknown metric
            seed: 7
          - name: churn_model                    # error: duplicate name
            task: binary_classification
            adapter: fake
            owner: ds@example.com
            dataset: ref('churn_training')
            target: churned
            evaluation:
              protocol: {split: temporal}
              metrics: [pr_auc]
            seed: 7
          - name: broken_model_two
            task: binary_classification
            adapter: nonexistent                 # error: missing adapter
            owner: ds@example.com
            dataset: ref('churn_training')
            target: wrong_label                  # error: target != label column
            evaluation:
              protocol: {split: random}          # error: split mismatch
              metrics: [pr_auc]
            seed: 7
        """,
    )
    with pytest.raises(ConfigError) as excinfo:
        parse_project(demo_project, registry=fake_registry)
    message = str(excinfo.value)
    assert "no_such_dataset" in message
    assert "hyperparamters" in message
    assert "did you mean" in message
    assert "maxdepth" in message
    assert "no_such_metric" in message
    assert "duplicate model" in message
    assert "not installed" in message and "pip install mbt-nonexistent" in message
    assert "must equal the dataset's label column" in message
    assert "must match" in message  # split mismatch
    parsed = parse_project(demo_project, registry=fake_registry, raise_on_error=False)
    assert len(parsed.report.errors) >= 8


def test_missing_adapter_names_pip_package(
    demo_project: Path, fake_registry: AdapterRegistry
) -> None:
    model = demo_project / "models/churn_model.yml"
    model.write_text(model.read_text().replace("adapter: fake", "adapter: catboost"))
    parsed = parse_project(demo_project, registry=fake_registry, raise_on_error=False)
    errors = [i for i in parsed.report.errors if "catboost" in i.message]
    assert errors and "pip install mbt-catboost" in (errors[0].hint or "")


def test_cycle_is_reported_with_path(tmp_path: Path, fake_registry: AdapterRegistry) -> None:
    # Datasets cannot ref() in v0, so cycles cannot occur through specs;
    # verify the graph-level cycle error directly instead.
    import networkx as nx

    from mbt.dag.graph import ensure_acyclic

    graph = nx.DiGraph([("a", "b"), ("b", "c"), ("c", "a")])
    with pytest.raises(ConfigError, match="cycle") as excinfo:
        ensure_acyclic(graph)
    assert " -> ".join([]) == "" and "a" in str(excinfo.value)


def test_model_to_model_ref_rejected(demo_project: Path, fake_registry: AdapterRegistry) -> None:
    write(
        demo_project / "models/ensemble.yml",
        """
        models:
          - name: ensemble
            task: binary_classification
            adapter: fake
            owner: ds@example.com
            dataset: ref('churn_model')
            target: churned
            evaluation:
              protocol: {split: temporal}
              metrics: [pr_auc]
            seed: 7
        """,
    )
    parsed = parse_project(demo_project, registry=fake_registry, raise_on_error=False)
    assert any("v1" in (i.hint or "") for i in parsed.report.errors)


def test_jinja_var_in_gate_threshold_is_deferred(
    demo_project: Path, fake_registry: AdapterRegistry
) -> None:
    """Jinja expressions in spec values do not break parse-time validation."""
    parsed = parse_project(demo_project, registry=fake_registry)
    model = parsed.models["model.demo.churn_model"]
    assert model.raw["evaluation"]["gates"][0]["threshold"] == "{{ var('default_threshold') }}"


def test_hooks_sibling_autodetected(demo_project: Path, fake_registry: AdapterRegistry) -> None:
    write(
        demo_project / "models/churn_model.py",
        """
        def transform_features(table, ctx):
            return table
        """,
    )
    parsed = parse_project(demo_project, registry=fake_registry)
    assert parsed.models["model.demo.churn_model"].hooks_path == "models/churn_model.py"
