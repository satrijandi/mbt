"""ScoringSpec validation (ADR-20)."""

import pytest
from pydantic import ValidationError

from mbt_adapter_base import ScoringInputSpec, ScoringSpec
from mbt_adapter_base.types import Stage


def _spec(**overrides: object) -> dict[str, object]:
    base: dict[str, object] = {
        "name": "retention_scoring",
        "owner": "growth-ds@company.com",
        "model": "ref('churn_classifier')",
        "input": {"source": "source('lakehouse', 'scoring_batch')"},
        "output": {"path": "predictions/retention", "columns": ["user_id"]},
    }
    base.update(overrides)
    return base


def test_minimal_spec_parses_with_defaults() -> None:
    spec = ScoringSpec.model_validate(_spec())
    assert spec.stage is Stage.PRODUCTION
    assert spec.monitors is None
    assert spec.ground_truth is None
    assert spec.output.format == "parquet"
    assert spec.passthrough_columns == ["user_id"]


def test_input_requires_source_xor_inputs() -> None:
    with pytest.raises(ValidationError, match="exactly one of 'source'"):
        ScoringInputSpec.model_validate({})
    with pytest.raises(ValidationError, match="exactly one of 'source'"):
        ScoringInputSpec.model_validate(
            {
                "source": "source('a', 'b')",
                "inputs": {
                    "spine": "source('a', 'c')",
                    "features": ["source('a', 'd')"],
                    "join_key": "id",
                },
            }
        )


def test_window_requires_time_column() -> None:
    with pytest.raises(ValidationError, match="'window' requires 'time_column'"):
        ScoringInputSpec.model_validate({"source": "source('a', 'b')", "window": "-7d:now"})


def test_multi_table_inputs_need_features_and_join_key() -> None:
    with pytest.raises(ValidationError, match="at least one feature table"):
        ScoringInputSpec.model_validate(
            {"inputs": {"spine": "source('a', 'c')", "features": [], "join_key": "id"}}
        )


def test_feature_entries_need_join_columns_from_somewhere() -> None:
    with pytest.raises(ValidationError, match="has no join columns"):
        ScoringInputSpec.model_validate(
            {"inputs": {"spine": "source('a', 'c')", "features": ["source('a', 'd')"]}}
        )


def test_ground_truth_gate_metric_must_be_declared() -> None:
    with pytest.raises(ValidationError, match=r"must appear in ground_truth\.metrics"):
        ScoringSpec.model_validate(
            _spec(
                ground_truth={
                    "label": {"source": "source('a', 'outcomes')", "column": "churned"},
                    "join_key": "user_id",
                    "maturity": "14d",
                    "metrics": ["roc_auc"],
                    "gates": [{"metric": "pr_auc", "threshold": 0.3}],
                }
            )
        )


def test_passthrough_columns_union_and_order() -> None:
    spec = ScoringSpec.model_validate(
        _spec(
            input={
                "source": "source('lakehouse', 'scoring_batch')",
                "time_column": "snapshot_date",
            },
            ground_truth={
                "label": {"source": "source('a', 'outcomes')", "column": "churned"},
                "join_key": ["user_id", "snapshot_date"],
                "maturity": "14d",
                "metrics": ["roc_auc"],
            },
            output={"path": "p", "columns": ["email", "user_id"]},
        )
    )
    assert spec.passthrough_columns == ["email", "user_id", "snapshot_date"]


def test_empty_passthrough_set_rejected() -> None:
    with pytest.raises(ValidationError, match="at least one identity column"):
        ScoringSpec.model_validate(_spec(output={"path": "predictions/retention"}))


def test_unknown_fields_rejected() -> None:
    with pytest.raises(ValidationError, match="schedule"):
        ScoringSpec.model_validate(_spec(schedule="daily"))


def test_dump_reparse_roundtrip() -> None:
    spec = ScoringSpec.model_validate(
        _spec(
            monitors={
                "feature_shift": {"threshold": 0.2, "exclude": ["email"]},
                "prediction_shift": {"method": "ks", "threshold": 0.15},
            },
            ground_truth={
                "label": {"source": "source('a', 'outcomes')", "column": "churned"},
                "join_key": "user_id",
                "maturity": "14d",
                "metrics": ["roc_auc", "pr_auc"],
                "gates": [{"metric": "pr_auc", "threshold": 0.3}],
            },
        )
    )
    assert ScoringSpec.model_validate(spec.model_dump()) == spec
