"""Guard for the committed examples/s3_wide project (R2-16).

Parses the example through the real registry and asserts the wide multi-table
shape - five lake tables (spine + label + three feature tables) feeding one
dataset feeding one model (ADR-22) - so the example, and the ``mbt parse``
command its README advertises, cannot silently rot when a spec schema changes.

This is the parse-only sibling of test_snowflake_wide_example.py: that guard
builds its example through the adapter, but s3_wide's data plane is
Spark-over-s3a and cannot run hermetically, so this guard stops at parse.
Parsing is pure spec validation plus DAG construction - no S3, no Spark, no
JVM - so it belongs in the fast suite.
"""

from pathlib import Path

from mbt.parsing import parse_project

EXAMPLE = Path(__file__).resolve().parents[1] / "examples" / "s3_wide"

SOURCES = {
    "source.s3_wide.lake.customer_population",
    "source.s3_wide.lake.churn_labels",
    "source.s3_wide.lake.demographic_features",
    "source.s3_wide.lake.engagement_features",
    "source.s3_wide.lake.billing_features",
}
DATASET = "dataset.s3_wide.wide_churn_training"
MODEL = "model.s3_wide.churn_wide"


def test_s3_wide_example_parses_to_the_wide_panel_shape() -> None:
    parsed = parse_project(EXAMPLE)

    # No parse errors, and exactly the 7 resources the README advertises
    # ("Parsed 7 resources"): 5 lake sources + 1 dataset + 1 model.
    assert not parsed.report.errors
    assert set(parsed.sources) == SOURCES
    assert set(parsed.datasets) == {DATASET}
    assert set(parsed.models) == {MODEL}
    assert not parsed.scoring and not parsed.exposures
    assert len(parsed.nodes) + len(parsed.sources) + len(parsed.exposures) == 7

    # The wide shape (ADR-22): all five lake tables feed the one dataset, which
    # feeds the one model - the multi-table join the example exists to show.
    assert set(parsed.datasets[DATASET].depends_on) == SOURCES
    assert parsed.models[MODEL].depends_on == [DATASET]
    for source in SOURCES:
        assert parsed.graph.has_edge(source, DATASET)
    assert parsed.graph.has_edge(DATASET, MODEL)
