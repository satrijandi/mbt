"""Schema validation tests (S1-02)."""

import pytest
from pydantic import ValidationError

from mbt.contracts import (
    DatasetSpec,
    EvaluationProtocol,
    EvaluationSpec,
    GateSpec,
    ModelSpec,
    ScoringOutputSpec,
    SearchDimension,
    SplitSpec,
)


def test_scoring_decision_threshold_accepts_float_or_operating_point_name() -> None:
    # a float cutoff must be a probability...
    with pytest.raises(ValidationError, match=r"in \[0, 1\]"):
        ScoringOutputSpec(path="p", decision_threshold=1.5)
    # ...and a string must name a champion operating-point metric (R2-5)
    with pytest.raises(ValidationError, match="operating-point metric"):
        ScoringOutputSpec(path="p", decision_threshold="roc_auc")
    assert ScoringOutputSpec(path="p", decision_threshold=0.5).decision_threshold == 0.5
    named = ScoringOutputSpec(path="p", decision_threshold="threshold_at_precision_0.9")
    assert named.decision_threshold == "threshold_at_precision_0.9"
    # per-prediction explanation count must be a positive integer
    with pytest.raises(ValidationError):
        ScoringOutputSpec(path="p", explain_top_k=0)
    assert ScoringOutputSpec(path="p", explain_top_k=3).explain_top_k == 3


def test_shift_significance_requires_ks_and_excludes_a_warn_band() -> None:
    from mbt.contracts import FeatureShiftSpec

    # the n-aware significance is a KS critical value, so it needs method: ks...
    with pytest.raises(ValidationError, match="method: ks"):
        FeatureShiftSpec(method="psi", threshold=0.2, significance=0.05)
    # ...does not combine with an absolute warn band...
    with pytest.raises(ValidationError, match="mutually exclusive"):
        FeatureShiftSpec(method="ks", threshold=0.2, significance=0.05, warn_threshold=0.1)
    # ...and is a p-value in (0, 1)
    with pytest.raises(ValidationError):
        FeatureShiftSpec(method="ks", threshold=0.2, significance=1.5)
    assert FeatureShiftSpec(method="ks", threshold=0.2, significance=0.05).significance == 0.05


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


def test_disparity_gate_is_a_third_kind() -> None:
    # `across` is its own gate kind (disparity), with a default min_ratio.
    gate = GateSpec(metric="pr_auc", across="plan_type")
    assert gate.across == "plan_type" and gate.min_ratio == 0.8
    # ...mutually exclusive with the other two kinds
    with pytest.raises(ValidationError, match="exactly one"):
        GateSpec(metric="pr_auc", across="plan_type", threshold=0.4)
    # min_ratio is across-only, and must be a ratio in (0, 1]
    with pytest.raises(ValidationError, match="min_ratio"):
        GateSpec(metric="pr_auc", threshold=0.4, min_ratio=0.9)
    with pytest.raises(ValidationError, match=r"in \(0, 1\]"):
        GateSpec(metric="pr_auc", across="plan_type", min_ratio=1.5)
    # a disparity gate spans a whole column, so it cannot also fix one slice
    with pytest.raises(ValidationError, match="whole column"):
        GateSpec(metric="pr_auc", across="plan_type", slice="plan_type=basic")
    # r2 is the one signed builtin: its worst/best ratio is ill-defined (two
    # negative slices invert it), so a disparity gate on r2 is rejected at
    # parse, while a non-negative regression metric is allowed (F16).
    with pytest.raises(ValidationError, match="r2"):
        GateSpec(metric="r2", across="segment")
    assert GateSpec(metric="rmse", across="segment").metric == "rmse"


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
