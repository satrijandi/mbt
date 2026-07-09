"""Scoring resource parsing: linking, checks subset, ground-truth metrics (ADR-20/21)."""

from pathlib import Path

import pytest
from core_helpers import write

from mbt.adapters.registry import AdapterRegistry
from mbt.exceptions import ConfigError
from mbt.parsing import parse_project

SCORING_YML = """
scoring:
  - name: churn_scoring
    owner: lifecycle-eng@example.com
    model: ref('churn_model')
    tags: [daily]
    input:
      source: source('lakehouse', 'scoring_batch')
      time_column: snapshot_date
      window: "-7d:now"
    checks:
      - not_null:
          columns: [user_id]
    monitors:
      feature_shift:
        threshold: 0.2
      prediction_shift:
        threshold: 0.2
    ground_truth:
      label:
        source: source('lakehouse', 'churn_outcomes')
        column: churned
      join_key: user_id
      maturity: "14d"
      metrics: [pr_auc, roc_auc]
      gates:
        - metric: pr_auc
          threshold: 0.3
    output:
      path: predictions/churn_scores
      columns: [user_id]
"""


@pytest.fixture()
def scoring_project(demo_project: Path) -> Path:
    write(
        demo_project / "sources.yml",
        """
        sources:
          - name: lakehouse
            tables:
              - name: subscribers
                path: data/subscribers/*.parquet
              - name: scoring_batch
                path: data/scoring_batch/*.parquet
              - name: churn_outcomes
                path: data/churn_outcomes/*.parquet
        """,
    )
    write(demo_project / "scoring/churn_scoring.yml", SCORING_YML)
    return demo_project


def test_valid_scoring_pipeline_parses(
    scoring_project: Path, fake_registry: AdapterRegistry
) -> None:
    parsed = parse_project(scoring_project, registry=fake_registry)
    assert set(parsed.scoring) == {"scoring.demo.churn_scoring"}
    resource = parsed.scoring["scoring.demo.churn_scoring"]
    assert resource.resource_type == "scoring"
    assert resource.depends_on == [
        "model.demo.churn_model",
        "source.demo.lakehouse.churn_outcomes",
        "source.demo.lakehouse.scoring_batch",
    ]
    assert [m.name for m in resource.metric_specs] == ["pr_auc", "roc_auc"]
    assert all(m.kind == "builtin" for m in resource.metric_specs)
    assert parsed.graph.has_edge("model.demo.churn_model", "scoring.demo.churn_scoring")
    assert parsed.nodes["scoring.demo.churn_scoring"] is resource


def test_unknown_model_ref_is_an_error(
    scoring_project: Path, fake_registry: AdapterRegistry
) -> None:
    write(
        scoring_project / "scoring/churn_scoring.yml",
        SCORING_YML.replace("ref('churn_model')", "ref('churn_modle')"),
    )
    with pytest.raises(ConfigError, match=r"unknown model ref\('churn_modle'\)"):
        parse_project(scoring_project, registry=fake_registry)


def test_dataset_ref_is_rejected(scoring_project: Path, fake_registry: AdapterRegistry) -> None:
    write(
        scoring_project / "scoring/churn_scoring.yml",
        SCORING_YML.replace("ref('churn_model')", "ref('churn_training')"),
    )
    with pytest.raises(ConfigError, match="must reference a model"):
        parse_project(scoring_project, registry=fake_registry)


def test_label_dependent_check_is_rejected(
    scoring_project: Path, fake_registry: AdapterRegistry
) -> None:
    write(
        scoring_project / "scoring/churn_scoring.yml",
        SCORING_YML.replace("- not_null:\n          columns: [user_id]", "- label_leakage_scan"),
    )
    with pytest.raises(ConfigError, match="not available on scoring inputs"):
        parse_project(scoring_project, registry=fake_registry)


def test_not_null_requires_explicit_columns(
    scoring_project: Path, fake_registry: AdapterRegistry
) -> None:
    write(
        scoring_project / "scoring/churn_scoring.yml",
        SCORING_YML.replace("- not_null:\n          columns: [user_id]", "- not_null"),
    )
    with pytest.raises(ConfigError, match="requires explicit 'columns'"):
        parse_project(scoring_project, registry=fake_registry)


def test_ground_truth_hook_metric_is_rejected(
    scoring_project: Path, fake_registry: AdapterRegistry
) -> None:
    write(
        scoring_project / "metrics.yml",
        """
        metrics:
          - name: campaign_capture
            kind: hook
        """,
    )
    write(
        scoring_project / "scoring/churn_scoring.yml",
        SCORING_YML.replace("metrics: [pr_auc, roc_auc]", "metrics: [campaign_capture]").replace(
            "- metric: pr_auc", "- metric: campaign_capture"
        ),
    )
    with pytest.raises(ConfigError, match="must be a builtin"):
        parse_project(scoring_project, registry=fake_registry)


def test_maturity_must_be_a_bare_duration(
    scoring_project: Path, fake_registry: AdapterRegistry
) -> None:
    write(
        scoring_project / "scoring/churn_scoring.yml",
        SCORING_YML.replace('maturity: "14d"', 'maturity: "-14d:now"'),
    )
    with pytest.raises(ConfigError, match="bare duration"):
        parse_project(scoring_project, registry=fake_registry)


def test_invalid_input_window_is_an_error(
    scoring_project: Path, fake_registry: AdapterRegistry
) -> None:
    write(
        scoring_project / "scoring/churn_scoring.yml",
        SCORING_YML.replace('window: "-7d:now"', 'window: "not-a-window"'),
    )
    with pytest.raises(ConfigError):
        parse_project(scoring_project, registry=fake_registry)


def test_unknown_input_source_is_an_error(
    scoring_project: Path, fake_registry: AdapterRegistry
) -> None:
    write(
        scoring_project / "scoring/churn_scoring.yml",
        SCORING_YML.replace(
            "source('lakehouse', 'scoring_batch')", "source('lakehouse', 'missing_batch')"
        ),
    )
    with pytest.raises(ConfigError, match=r"unknown source \('lakehouse', 'missing_batch'\)"):
        parse_project(scoring_project, registry=fake_registry)


def test_exposure_can_depend_on_scoring(
    scoring_project: Path, fake_registry: AdapterRegistry
) -> None:
    write(
        scoring_project / "exposures.yml",
        """
        exposures:
          - name: retention_campaign_job
            type: batch_job
            depends_on: [ref('churn_scoring')]
            owner: lifecycle-eng@example.com
        """,
    )
    parsed = parse_project(scoring_project, registry=fake_registry)
    exposure = parsed.exposures["exposure.demo.retention_campaign_job"]
    assert exposure.depends_on == ["scoring.demo.churn_scoring"]
    assert parsed.graph.has_edge(
        "scoring.demo.churn_scoring", "exposure.demo.retention_campaign_job"
    )
