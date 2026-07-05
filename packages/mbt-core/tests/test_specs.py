"""Schema validation tests (S1-02)."""

import pytest
from pydantic import ValidationError

from mbt.contracts import (
    DatasetSpec,
    EvaluationProtocol,
    EvaluationSpec,
    GateSpec,
    ModelSpec,
    SearchDimension,
    SplitSpec,
)


def _model_kwargs(**overrides: object) -> dict[str, object]:
    base: dict[str, object] = {
        "name": "m",
        "task": "binary_classification",
        "adapter": "fake",
        "owner": "a@b.c",
        "dataset": "ref('d')",
        "target": "y",
        "evaluation": EvaluationSpec(
            protocol=EvaluationProtocol(), metrics=["pr_auc"]
        ),
        "seed": 1,
    }
    base.update(overrides)
    return base


def test_seed_is_mandatory_with_no_default() -> None:
    kwargs = _model_kwargs()
    del kwargs["seed"]
    with pytest.raises(ValidationError, match="seed"):
        ModelSpec.model_validate(kwargs)


def test_unknown_fields_rejected() -> None:
    with pytest.raises(ValidationError, match="extra_forbidden|Extra inputs"):
        ModelSpec.model_validate({**_model_kwargs(), "hyperparams": {}})


def test_gate_requires_exactly_one_kind() -> None:
    with pytest.raises(ValidationError, match="exactly one"):
        GateSpec(metric="pr_auc")
    with pytest.raises(ValidationError, match="exactly one"):
        GateSpec(metric="pr_auc", threshold=0.4, compare_to="production")
    with pytest.raises(ValidationError, match="min_delta"):
        GateSpec(metric="pr_auc", threshold=0.4, min_delta=0.1)
    assert GateSpec(metric="pr_auc", compare_to="production", min_delta=0.005).min_delta == 0.005


def test_gate_metric_must_be_declared() -> None:
    evaluation = EvaluationSpec(
        protocol=EvaluationProtocol(),
        metrics=["pr_auc"],
        gates=[GateSpec(metric="roc_auc", threshold=0.5)],
    )
    with pytest.raises(ValidationError, match="must appear in evaluation.metrics"):
        ModelSpec.model_validate(_model_kwargs(evaluation=evaluation))


def test_temporal_split_requires_time_column() -> None:
    with pytest.raises(ValidationError, match="time_column"):
        SplitSpec(strategy="temporal", train="-180d:-28d", test="-28d:now")


def test_random_split_requires_seed_and_fractions() -> None:
    with pytest.raises(ValidationError, match="seed"):
        SplitSpec(strategy="random", train="0.8", test="0.2")
    with pytest.raises(ValidationError, match="fraction"):
        SplitSpec(strategy="random", train="-28d:now", test="0.2", seed=7)
    ok = SplitSpec(strategy="random", train="0.8", test="0.2", seed=7)
    assert ok.seed == 7


def test_split_default_is_temporal() -> None:
    spec = DatasetSpec.model_validate(
        {
            "name": "d",
            "source": "source('a', 'b')",
            "label": {"column": "y"},
            "split": {"time_column": "ts", "train": "-180d:-28d", "test": "-28d:now"},
        }
    )
    assert spec.split.strategy.value == "temporal"


def test_search_dimension_shapes() -> None:
    with pytest.raises(ValidationError, match="choices"):
        SearchDimension(type="categorical")
    with pytest.raises(ValidationError, match="low"):
        SearchDimension(type="int", low=3)
    with pytest.raises(ValidationError, match="loguniform"):
        SearchDimension(type="loguniform", low=0, high=1)
    assert SearchDimension(type="int", low=3, high=10).high == 10
