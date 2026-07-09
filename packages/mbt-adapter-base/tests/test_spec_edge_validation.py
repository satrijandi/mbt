"""Spec-schema validation edge cases (TSD §5.4-§5.8)."""

from typing import Any

import pytest
from pydantic import ValidationError

from mbt_adapter_base.specs import (
    DatasetInputs,
    EvaluationProtocol,
    EvaluationSpec,
    GateSpec,
    GroundTruthSpec,
    ModelSpec,
    ScoringInputs,
    ScoringInputSpec,
    SearchDimension,
    SourceTable,
    SplitSpec,
    TuningObjective,
    TuningSpec,
)
from mbt_adapter_base.types import SplitStrategy, Stage, TaskType


def test_source_table_needs_path_or_identifier() -> None:
    with pytest.raises(ValidationError, match="either 'path' or 'identifier'"):
        SourceTable(name="gold_subscribers")


def test_temporal_split_rejects_stratify_by() -> None:
    with pytest.raises(ValidationError, match="random strategy only"):
        SplitSpec(time_column="ts", train="-180d:-28d", test="-28d:now", stratify_by="plan")


def test_temporal_split_rejects_seed() -> None:
    with pytest.raises(ValidationError, match="deterministic by time"):
        SplitSpec(time_column="ts", train="-180d:-28d", test="-28d:now", seed=7)


def test_random_split_fraction_must_be_in_unit_interval() -> None:
    with pytest.raises(ValidationError, match=r"must be in \(0, 1\)"):
        SplitSpec(strategy=SplitStrategy.RANDOM, train="1.5", test="0.2", seed=7)


def test_dataset_inputs_need_at_least_one_feature_table() -> None:
    with pytest.raises(ValidationError, match="at least one feature table"):
        DatasetInputs(features=[], label="source('a', 'labels')", join_key="user_id")


def test_dataset_inputs_need_nonempty_join_key() -> None:
    with pytest.raises(ValidationError, match="non-empty column"):
        DatasetInputs(features=["source('a', 'f')"], label="source('a', 'labels')", join_key="")


def test_evaluation_metrics_must_be_nonempty() -> None:
    with pytest.raises(ValidationError, match="at least one metric"):
        EvaluationSpec(protocol=EvaluationProtocol(), metrics=[])


def test_categorical_dimension_parses_and_rejects_bounds() -> None:
    dimension = SearchDimension(type="categorical", choices=["gbtree", "dart"])
    assert dimension.choices == ["gbtree", "dart"]
    with pytest.raises(ValidationError, match="not 'low'/'high'"):
        SearchDimension(type="categorical", choices=["gbtree"], low=0.0)


def test_numeric_dimension_rejects_choices() -> None:
    with pytest.raises(ValidationError, match="not 'choices'"):
        SearchDimension(type="int", low=1, high=4, choices=[1, 2])


def test_numeric_dimension_requires_low_below_high() -> None:
    with pytest.raises(ValidationError, match="strictly less than"):
        SearchDimension(type="uniform", low=2.0, high=2.0)


def test_tuning_search_space_must_be_nonempty() -> None:
    with pytest.raises(ValidationError, match="must not be empty"):
        TuningSpec(
            n_trials=5,
            search_space={},
            objective=TuningObjective(metric="roc_auc", direction="maximize"),
        )


def _model_spec(**overrides: Any) -> ModelSpec:
    base: dict[str, Any] = {
        "name": "churn_classifier",
        "task": TaskType.BINARY_CLASSIFICATION,
        "adapter": "xgboost",
        "owner": "ds@company.com",
        "dataset": "ref('churn_training_set')",
        "target": "churned",
        "evaluation": EvaluationSpec(protocol=EvaluationProtocol(), metrics=["roc_auc"]),
        "seed": 42,
    }
    base.update(overrides)
    return ModelSpec(**base)


def test_champion_gates_must_compare_to_one_stage() -> None:
    evaluation = EvaluationSpec(
        protocol=EvaluationProtocol(),
        metrics=["roc_auc"],
        gates=[
            GateSpec(metric="roc_auc", compare_to=Stage.STAGING),
            GateSpec(metric="roc_auc", compare_to=Stage.PRODUCTION),
        ],
    )
    with pytest.raises(ValidationError, match="same stage in v0"):
        _model_spec(evaluation=evaluation)


def test_tuning_objective_metric_must_be_declared() -> None:
    tuning = TuningSpec(
        n_trials=5,
        search_space={"max_depth": SearchDimension(type="int", low=2, high=8)},
        objective=TuningObjective(metric="pr_auc", direction="maximize"),
    )
    with pytest.raises(ValidationError, match=r"must appear in evaluation\.metrics"):
        _model_spec(tuning=tuning)


def test_scoring_inputs_need_nonempty_join_key() -> None:
    with pytest.raises(ValidationError, match="non-empty column"):
        ScoringInputs(spine="source('a', 's')", features=["source('a', 'f')"], join_key="")


def test_scoring_input_sample_key_columns() -> None:
    single = ScoringInputSpec(source="source('a', 'b')", sample_key="user_id")
    assert single.sample_key_columns == ["user_id"]
    multi = ScoringInputSpec(source="source('a', 'b')", sample_key=["user_id", "snapshot_date"])
    assert multi.sample_key_columns == ["user_id", "snapshot_date"]


def test_ground_truth_metrics_must_be_nonempty() -> None:
    with pytest.raises(ValidationError, match="at least one metric"):
        GroundTruthSpec(
            label={"source": "source('a', 'outcomes')", "column": "churned"},
            join_key="user_id",
            maturity="14d",
            metrics=[],
        )


def test_ground_truth_join_key_must_be_nonempty() -> None:
    with pytest.raises(ValidationError, match="non-empty column"):
        GroundTruthSpec(
            label={"source": "source('a', 'outcomes')", "column": "churned"},
            join_key=[],
            maturity="14d",
            metrics=["roc_auc"],
        )
