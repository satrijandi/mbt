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


def test_source_table_rejects_an_unsupported_format() -> None:
    # 'iceberg' is roadmap, not implemented anywhere, so it must not silently
    # parse (and get mis-read as parquet); only the actually-supported formats
    # are accepted - parquet everywhere, delta on spark (F23).
    with pytest.raises(ValidationError, match="format"):
        SourceTable(name="t", path="data/t", format="iceberg")
    with pytest.raises(ValidationError, match="format"):
        SourceTable(name="t", path="data/t", format="csv")
    assert SourceTable(name="t", path="data/t", format="parquet").format == "parquet"
    assert SourceTable(name="t", path="data/t", format="delta").format == "delta"


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


def test_backtest_folds_at_least_two_on_either_strategy() -> None:
    """Cross-validated backtest (R2-7): works on temporal (walk-forward) and
    random (k-fold) splits; >= 2 folds."""
    assert EvaluationProtocol.model_validate({"backtest_folds": 3}).backtest_folds == 3
    assert (
        EvaluationProtocol.model_validate({"split": "random", "backtest_folds": 5}).backtest_folds
        == 5
    )
    with pytest.raises(ValidationError):  # a single fold cannot form a backtest
        EvaluationProtocol.model_validate({"backtest_folds": 1})


def test_backtest_gate_source_requires_threshold_and_backtest_folds() -> None:
    """A gate with source: backtest must be a whole-split threshold gate and
    needs evaluation.protocol.backtest_folds set (R2-7 part 2)."""
    from mbt_adapter_base.specs import GateSpec

    with pytest.raises(ValidationError, match="whole-split threshold gate"):
        GateSpec(metric="pr_auc", compare_to="production", source="backtest")

    def _spec(**protocol: Any) -> ModelSpec:
        return ModelSpec.model_validate(
            {
                "name": "m",
                "task": "binary_classification",
                "adapter": "xgboost",
                "owner": "t@example.com",
                "dataset": "ref('d')",
                "target": "y",
                "evaluation": {
                    "protocol": {"split": "temporal", **protocol},
                    "metrics": ["pr_auc"],
                    "gates": [{"metric": "pr_auc", "threshold": 0.7, "source": "backtest"}],
                },
                "seed": 1,
            }
        )

    with pytest.raises(ValidationError, match="backtest_folds is not set"):
        _spec()
    assert _spec(backtest_folds=4).evaluation.gates[0].source == "backtest"


def test_embargo_is_temporal_only_and_positive() -> None:
    """The split embargo (R2-7) is a positive duration on the temporal strategy."""
    from mbt_adapter_base.specs import SplitSpec

    ok = SplitSpec(time_column="ts", train="-180d:-28d", test="-28d:now", embargo="7d")
    assert ok.embargo == "7d"
    with pytest.raises(ValidationError, match="temporal strategy only"):
        SplitSpec.model_validate(
            {"strategy": "random", "train": "0.8", "test": "0.2", "seed": 1, "embargo": "7d"}
        )
    with pytest.raises(ValidationError, match="positive duration"):
        SplitSpec(time_column="ts", train="-180d:-28d", test="-28d:now", embargo="-7d")


def test_nested_cv_requires_backtest_folds_and_tuning() -> None:
    """Nested CV (R2-7): needs backtest_folds and a tuning block; works on either
    split (temporal walk-forward or random k-fold)."""
    from mbt_adapter_base.specs import EvaluationProtocol

    with pytest.raises(ValidationError, match="needs backtest_folds"):
        EvaluationProtocol.model_validate({"split": "random", "nested_cv": True})
    assert EvaluationProtocol.model_validate(
        {"split": "temporal", "backtest_folds": 3, "nested_cv": True}
    ).nested_cv  # temporal nested CV is now supported (walk-forward outer folds)
    assert EvaluationProtocol.model_validate(
        {"split": "random", "backtest_folds": 3, "nested_cv": True}
    ).nested_cv


def test_calibration_and_backtest_folds_compose() -> None:
    """F5 (real fix): each backtest fold carves its own calibration slice from
    the fold's train, exactly like the production fit (seed+5), so the pairing
    that used to be rejected at parse now parses and trains."""
    backtest_eval = EvaluationSpec(
        protocol=EvaluationProtocol(backtest_folds=3), metrics=["roc_auc"]
    )
    combined = _model_spec(calibration="isotonic", evaluation=backtest_eval)
    assert combined.calibration == "isotonic"
    assert combined.evaluation.protocol.backtest_folds == 3
    assert _model_spec(calibration="isotonic").calibration == "isotonic"
    assert _model_spec(evaluation=backtest_eval).evaluation.protocol.backtest_folds == 3

    def _spec(with_tuning: bool) -> ModelSpec:
        payload: dict[str, Any] = {
            "name": "m",
            "task": "binary_classification",
            "adapter": "xgboost",
            "owner": "t@example.com",
            "dataset": "ref('d')",
            "target": "y",
            "evaluation": {
                "protocol": {"split": "random", "backtest_folds": 3, "nested_cv": True},
                "metrics": ["pr_auc"],
            },
            "seed": 1,
        }
        if with_tuning:
            payload["tuning"] = {
                "engine": "optuna",
                "n_trials": 5,
                "search_space": {"max_depth": {"type": "int", "low": 2, "high": 5}},
                "objective": {"metric": "pr_auc", "direction": "maximize"},
            }
        return ModelSpec.model_validate(payload)

    with pytest.raises(ValidationError, match="needs a 'tuning' block"):
        _spec(with_tuning=False)
    assert _spec(with_tuning=True).evaluation.protocol.nested_cv
