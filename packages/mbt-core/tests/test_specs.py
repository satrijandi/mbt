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
        "evaluation": EvaluationSpec(protocol=EvaluationProtocol(), metrics=["pr_auc"]),
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
    with pytest.raises(ValidationError, match=r"extra_forbidden|Extra inputs"):
        ModelSpec.model_validate({**_model_kwargs(), "hyperparams": {}})


def test_gate_requires_exactly_one_kind() -> None:
    with pytest.raises(ValidationError, match="exactly one"):
        GateSpec(metric="pr_auc")
    with pytest.raises(ValidationError, match="exactly one"):
        GateSpec(metric="pr_auc", threshold=0.4, compare_to="production")
    with pytest.raises(ValidationError, match="min_delta"):
        GateSpec(metric="pr_auc", threshold=0.4, min_delta=0.1)
    assert GateSpec(metric="pr_auc", compare_to="production", min_delta=0.005).min_delta == 0.005


def test_gate_bootstrap_fields_validated() -> None:
    # bootstrap fields are champion-gate-only (ADR-18)
    with pytest.raises(ValidationError, match="confidence"):
        GateSpec(metric="pr_auc", threshold=0.4, confidence=0.9)
    with pytest.raises(ValidationError, match="bootstrap_resamples"):
        GateSpec(metric="pr_auc", threshold=0.4, bootstrap_resamples=500)
    with pytest.raises(ValidationError, match="confidence"):
        GateSpec(metric="pr_auc", compare_to="production", confidence=1.5)
    with pytest.raises(ValidationError, match="at least 100"):
        GateSpec(metric="pr_auc", compare_to="production", bootstrap_resamples=10)

    gate = GateSpec(metric="pr_auc", compare_to="production")
    assert gate.confidence == 0.95 and gate.bootstrap_resamples == 1000
    opted_out = GateSpec(metric="pr_auc", compare_to="production", confidence=None)
    assert opted_out.confidence is None


def test_slice_gate_column_must_be_declared_and_well_formed() -> None:
    def evaluation(slice_key: str, slices: list[str]) -> EvaluationSpec:
        return EvaluationSpec(
            protocol=EvaluationProtocol(),
            metrics=["pr_auc"],
            gates=[GateSpec(metric="pr_auc", threshold=0.4, slice=slice_key)],
            slices=slices,
        )

    spec = ModelSpec.model_validate(
        _model_kwargs(evaluation=evaluation("plan_type=pro", ["plan_type"]))
    )
    assert spec.evaluation.gates[0].slice == "plan_type=pro"

    with pytest.raises(ValidationError, match=r"must appear in evaluation\.slices"):
        ModelSpec.model_validate(_model_kwargs(evaluation=evaluation("region=emea", ["plan_type"])))

    with pytest.raises(ValidationError, match="column=value"):
        ModelSpec.model_validate(_model_kwargs(evaluation=evaluation("plan_type", ["plan_type"])))


def test_gate_metric_must_be_declared() -> None:
    evaluation = EvaluationSpec(
        protocol=EvaluationProtocol(),
        metrics=["pr_auc"],
        gates=[GateSpec(metric="roc_auc", threshold=0.5)],
    )
    with pytest.raises(ValidationError, match=r"must appear in evaluation\.metrics"):
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
