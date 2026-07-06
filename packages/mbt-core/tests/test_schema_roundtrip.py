"""Hypothesis round-trip tests: model -> YAML -> model -> JSON Schema (S1-02)."""

import json

import jsonschema
import yaml
from hypothesis import given, settings
from hypothesis import strategies as st

from mbt.contracts import DatasetSpec, GateSpec, ModelSpec, SourceGroup

_name = st.from_regex(r"[a-z][a-z0-9_]{0,20}", fullmatch=True)
_tags = st.lists(st.sampled_from(["churn", "weekly", "exp"]), max_size=3, unique=True)


@st.composite
def model_specs(draw) -> ModelSpec:
    metrics = draw(
        st.lists(
            st.sampled_from(["pr_auc", "roc_auc", "logloss"]), min_size=1, max_size=3, unique=True
        )
    )
    gates = []
    if draw(st.booleans()):
        gates.append(
            GateSpec(
                metric=metrics[0],
                threshold=draw(st.floats(min_value=0, max_value=1, allow_nan=False)),
            )
        )
    return ModelSpec(
        name=draw(_name),
        description=draw(st.text(max_size=40)),
        task="binary_classification",
        adapter="fake",
        owner="ds@example.com",
        tags=draw(_tags),
        dataset=f"ref('{draw(_name)}')",
        target=draw(_name),
        hyperparameters={"max_depth": draw(st.integers(min_value=1, max_value=12))},
        evaluation={"protocol": {"split": "temporal"}, "metrics": metrics, "gates": gates},
        seed=draw(st.integers(min_value=0, max_value=2**31)),
    )


@st.composite
def dataset_specs(draw) -> DatasetSpec:
    return DatasetSpec(
        name=draw(_name),
        source="source('lakehouse', 'subscribers')",
        label={"column": draw(_name)},
        filters=draw(st.lists(st.sampled_from(["a = 1", "b > 2"]), max_size=2)),
        split={
            "strategy": "temporal",
            "time_column": draw(_name),
            "train": "-180d:-28d",
            "test": "-28d:now",
        },
        tags=draw(_tags),
    )


def _roundtrip(spec, cls) -> None:
    dumped = spec.model_dump(mode="json")
    parsed = cls.model_validate(yaml.safe_load(yaml.safe_dump(dumped)))
    assert parsed == spec
    # the JSON Schema published for editors accepts what the model emits
    jsonschema.validate(json.loads(spec.model_dump_json()), cls.model_json_schema())


@given(model_specs())
@settings(max_examples=40, deadline=None)
def test_model_spec_roundtrip(spec: ModelSpec) -> None:
    _roundtrip(spec, ModelSpec)


@given(dataset_specs())
@settings(max_examples=40, deadline=None)
def test_dataset_spec_roundtrip(spec: DatasetSpec) -> None:
    _roundtrip(spec, DatasetSpec)


def test_source_group_roundtrip() -> None:
    group = SourceGroup(
        name="lakehouse",
        tables=[{"name": "subscribers", "path": "data/*.parquet"}],
    )
    _roundtrip(group, SourceGroup)
