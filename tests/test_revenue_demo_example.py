"""Guard for the committed examples/revenue_demo project (R2-4).

Parses the regression example through the real registry and asserts the
regression shape - `task: regression`, the regression metric set, and an
`rmse` (lower-is-better) gate - so the repo's one `task: regression` project,
and the `mbt` commands its README advertises, cannot silently rot when a spec
schema changes. This is the fast, parse-only guard; test_e2e_revenue_demo.py
trains it end to end.

Parsing is pure spec validation plus DAG construction (no training, no adapter,
no JVM), so this belongs in the fast suite alongside the other example guards.
"""

from pathlib import Path

from mbt.parsing import parse_project
from mbt_adapter_base import TaskType

EXAMPLE = Path(__file__).resolve().parents[1] / "examples" / "revenue_demo"

SOURCES = {
    "source.revenue_demo.lakehouse.subscribers",
    "source.revenue_demo.lakehouse.scoring_batch",
    "source.revenue_demo.lakehouse.spend_outcomes",
}
DATASET = "dataset.revenue_demo.spend_training_set"
MODEL = "model.revenue_demo.spend_regressor"
SCORING = "scoring.revenue_demo.spend_scoring"
REGRESSION_METRICS = ["rmse", "mae", "r2"]


def test_revenue_demo_parses_to_the_regression_shape() -> None:
    parsed = parse_project(EXAMPLE)

    # No parse errors, and exactly the resources the README advertises:
    # 3 lake sources + 1 dataset + 1 model + 1 scoring pipeline.
    assert not parsed.report.errors
    assert set(parsed.sources) == SOURCES
    assert set(parsed.datasets) == {DATASET}
    assert set(parsed.models) == {MODEL}
    assert set(parsed.scoring) == {SCORING}
    assert not parsed.exposures

    # It is the repo's regression exemplar: task, metric set, and - the point
    # the demo exists to show - a gate on rmse, which is lower-is-better, so
    # the threshold is a ceiling (the mirror image of churn_demo's pr_auc floor).
    model = parsed.models[MODEL].spec
    assert model.task is TaskType.REGRESSION
    assert model.evaluation.metrics == REGRESSION_METRICS
    assert [(g.metric, g.threshold) for g in model.evaluation.gates] == [("rmse", 12.0)]

    # Delayed ground-truth monitoring is wired for the continuous target too:
    # realized regression metrics, gated on the same rmse ceiling (ADR-21).
    scoring = parsed.scoring[SCORING].spec
    assert scoring.ground_truth is not None
    assert scoring.ground_truth.metrics == REGRESSION_METRICS
    assert [(g.metric, g.threshold) for g in scoring.ground_truth.gates] == [("rmse", 12.0)]

    # DAG shape: all three lake tables reach the graph; the dataset feeds the
    # model, and the scoring pipeline depends on the model.
    assert parsed.models[MODEL].depends_on == [DATASET]
    assert parsed.graph.has_edge(DATASET, MODEL)
    assert parsed.graph.has_edge(MODEL, SCORING)
