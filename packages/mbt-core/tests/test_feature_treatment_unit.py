"""Unit tests for declarative feature treatment (mbt/execute/feature_treatment.py, ADR-27).

The transform math is exercised here in isolation; test_handles_unit.py covers
that the handle applies it at the right point in the read pipeline.
"""

import math

import pyarrow as pa
import pytest

from mbt.contracts import FeatureSelection
from mbt.exceptions import ConfigError
from mbt.execute.feature_treatment import apply_treatment


def _selection(**raw: object) -> FeatureSelection:
    return FeatureSelection.model_validate(raw)


def _treat(table: pa.Table, split: str = "train", **raw: object) -> pa.Table:
    features = [name for name in table.column_names if name != "y"]
    return apply_treatment(table, _selection(**raw), features, split, resource="m")


# -- cap ------------------------------------------------------------------------


def test_scalar_cap_is_an_upper_plateau_and_keeps_the_integer_type() -> None:
    table = pa.table({"days": pa.array([10, 500, None, 365], type=pa.int64())})
    out = _treat(table, transforms={"days": {"cap": 365}}).column("days")
    assert out.to_pylist() == [10, 365, None, 365]
    assert out.type == pa.int64()


def test_two_sided_cap_clamps_both_ends() -> None:
    table = pa.table({"x": [-5.0, 0.5, 50.0, None]})
    out = _treat(table, transforms={"x": {"cap": {"min": 0, "max": 10}}}).column("x")
    assert out.to_pylist() == [0.0, 0.5, 10.0, None]


# -- log ------------------------------------------------------------------------


def test_log_is_log1p_and_preserves_nulls() -> None:
    table = pa.table({"x": [0.0, 3.0, None]})
    out = _treat(table, transforms={"x": {"log": True}}).column("x").to_pylist()
    assert out[0] == 0.0
    assert out[1] == pytest.approx(math.log(4.0))
    assert out[2] is None


def test_log_below_minus_one_errors_rather_than_emitting_nan() -> None:
    table = pa.table({"x": [-3.0, 1.0]})
    with pytest.raises(ConfigError, match=r"minimum value is -3\.0"):
        _treat(table, transforms={"x": {"log": True}})


def test_a_floor_makes_log_of_a_negative_column_legal() -> None:
    table = pa.table({"x": [-3.0, 1.0]})
    out = _treat(table, transforms={"x": {"cap": {"min": 0}, "log": True}}).column("x")
    assert out.to_pylist() == [0.0, pytest.approx(math.log(2.0))]


# -- percentile -----------------------------------------------------------------


def test_percentile_ranks_within_the_batch_with_averaged_ties() -> None:
    table = pa.table({"x": [1.0, 5.0, 5.0, 5.0, 9.0, None]})
    out = _treat(table, transforms={"x": {"percentile": "batch"}}).column("x")
    # ranks 1, 3, 3, 3, 5 over 5 non-null rows; the tie group takes its mean.
    assert out.to_pylist() == [0.2, 0.6, 0.6, 0.6, 1.0, None]


def test_percentile_is_invariant_under_a_uniform_shift() -> None:
    """The drift-mitigation claim, stated as a test: every subscriber ageing
    by the same amount must leave the feature unchanged."""
    base = pa.table({"x": [1.0, 4.0, 9.0, 16.0]})
    aged = pa.table({"x": [31.0, 34.0, 39.0, 46.0]})
    transform = {"x": {"percentile": "batch"}}
    assert (
        _treat(base, transforms=transform).column("x").to_pylist()
        == _treat(aged, transforms=transform).column("x").to_pylist()
    )


def test_percentile_folds_nan_in_with_null() -> None:
    table = pa.table({"x": [1.0, float("nan"), 3.0, None]})
    out = _treat(table, transforms={"x": {"percentile": "batch"}}).column("x")
    assert out.to_pylist() == [0.5, None, 1.0, None]


def test_percentile_of_an_all_null_column_stays_all_null() -> None:
    table = pa.table({"x": pa.array([None, None], type=pa.float64())})
    out = _treat(table, transforms={"x": {"percentile": "batch"}}).column("x")
    assert out.to_pylist() == [None, None]


def test_transform_order_is_cap_then_log_then_percentile() -> None:
    """A cap collapses its tail into one tie group, so the plateau lands on the
    group's mean rank rather than at 1.0 - which only holds if cap runs first."""
    table = pa.table({"x": [1.0, 400.0, 500.0, 600.0]})
    out = _treat(table, transforms={"x": {"cap": 365, "percentile": "batch"}}).column("x")
    assert out.to_pylist() == [0.25, 0.75, 0.75, 0.75]


# -- categoricals ---------------------------------------------------------------


def test_declared_int_categorical_is_retyped_to_string() -> None:
    table = pa.table({"code": pa.array([0, 1, 2], type=pa.int8())})
    out = _treat(table, categorical=["code"]).column("code")
    assert out.type == pa.string()
    assert out.to_pylist() == ["0", "1", "2"]


def test_absent_categorical_block_leaves_dtype_inference_alone() -> None:
    table = pa.table({"code": pa.array([0, 1], type=pa.int8()), "s": ["a", "b"]})
    out = _treat(table)
    assert out.column("code").type == pa.int8()
    assert out.column("s").type == pa.string()


def test_declared_block_makes_an_undeclared_string_feature_an_error() -> None:
    table = pa.table({"plan": ["basic", "pro"], "n": [1.0, 2.0]})
    with pytest.raises(ConfigError, match=r"not declared in features\.categorical: plan"):
        _treat(table, categorical=[])


def test_empty_list_is_a_valid_assertion_when_no_string_features_exist() -> None:
    table = pa.table({"n": [1.0, 2.0]})
    assert _treat(table, categorical=[]).column("n").to_pylist() == [1.0, 2.0]


def test_pinned_levels_pool_the_rest_and_leave_nulls_null() -> None:
    table = pa.table({"c": ["a", "b", "z", None]})
    out = _treat(table, categorical={"c": {"levels": ["a", "b"]}}).column("c")
    assert out.to_pylist() == ["a", "b", "__other__", None]


def test_null_as_level_runs_after_pooling_so_missing_is_not_pooled() -> None:
    table = pa.table({"c": pa.array([0, 9, None], type=pa.int8())})
    out = _treat(table, categorical={"c": {"levels": [0], "null_as_level": True}}).column("c")
    assert out.to_pylist() == ["0", "__other__", "__missing__"]


def test_max_levels_guards_the_train_split() -> None:
    table = pa.table({"c": [str(i) for i in range(5)]})
    with pytest.raises(ConfigError, match="5 distinct levels in the train split"):
        _treat(table, categorical={"c": {"max_levels": 2}})


def test_max_levels_does_not_fail_a_scoring_batch() -> None:
    """The guard exists to catch an identifier declared categorical, which is a
    modelling mistake visible at train time; a widening batch is the shift
    monitor's job, not a reason to fail a production run."""
    table = pa.table({"c": [str(i) for i in range(5)]})
    out = _treat(table, "score", categorical={"c": {"max_levels": 2}}).column("c")
    assert out.to_pylist() == ["0", "1", "2", "3", "4"]


def test_float_column_declared_categorical_errors() -> None:
    table = pa.table({"c": [1.5, 2.5]})
    with pytest.raises(ConfigError, match="declared categorical but has type double"):
        _treat(table, categorical=["c"])


def test_numeric_transform_on_an_undeclared_string_column_errors() -> None:
    """The schema already refuses cap/log/percentile on a column declared
    categorical; this is the case it cannot see - a string column left to dtype
    inference, caught when the data arrives."""
    table = pa.table({"c": ["a", "b"]})
    with pytest.raises(ConfigError, match="numeric transform but has type string"):
        _treat(table, transforms={"c": {"cap": 1}})


def test_cap_on_a_fractional_bound_widens_an_integer_column() -> None:
    table = pa.table({"n": pa.array([1, 9], type=pa.int64())})
    out = _treat(table, transforms={"n": {"cap": 2.5}}).column("n")
    assert out.type == pa.float64()
    assert out.to_pylist() == [1.0, 2.5]


# -- cross-cutting ---------------------------------------------------------------


def test_treating_a_column_the_model_never_sees_errors() -> None:
    table = pa.table({"a": [1.0], "y": [0]})
    with pytest.raises(ConfigError, match="does not consume: nope"):
        _treat(table, transforms={"nope": {"cap": 1}})


def test_target_and_slice_columns_are_never_treated() -> None:
    table = pa.table({"a": [1.0, 2.0], "y": [3.0, 400.0]})
    out = apply_treatment(
        table, _selection(transforms={"a": {"cap": 1}}), ["a"], "train", resource="m"
    )
    assert out.column("y").to_pylist() == [3.0, 400.0]


def test_a_monotonic_only_entry_leaves_the_data_alone() -> None:
    """`monotonic` constrains the model, not the values, so nothing is rewritten."""
    table = pa.table({"a": [1.0, 900.0]})
    out = _treat(table, transforms={"a": {"monotonic": "decreasing"}}).column("a")
    assert out.to_pylist() == [1.0, 900.0]


def test_column_order_survives_treatment() -> None:
    table = pa.table({"a": [1.0], "code": pa.array([2], type=pa.int8()), "y": [0]})
    out = _treat(table, categorical=["code"], transforms={"a": {"cap": 0.5}})
    assert out.column_names == ["a", "code", "y"]
