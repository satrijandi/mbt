"""Monitoring edge cases: dictionary columns, nulls, and type mismatches."""

import numpy as np
import pyarrow as pa

from mbt_adapter_base.monitoring import MonitoringBaseline, build_baseline, compute_monitor_stats
from mbt_adapter_base.specs import FeatureShiftSpec, MonitorsSpec

_N = 120
_SCORES = np.linspace(0.05, 0.95, _N)


def _baseline_table() -> pa.Table:
    return pa.table(
        {
            "tenure": pa.array([float(i % 40) for i in range(_N)]),
            "plan": pa.array([("basic", "premium", "pro")[i % 3] for i in range(_N)]),
        }
    )


def _baseline() -> MonitoringBaseline:
    return build_baseline(_baseline_table(), ["tenure", "plan"], _SCORES, model_name="m")


def _monitors(**overrides: object) -> MonitorsSpec:
    return MonitorsSpec(feature_shift=FeatureShiftSpec(threshold=0.2, **overrides))  # type: ignore[arg-type]


def test_dictionary_encoded_column_is_categorical() -> None:
    table = pa.table({"plan": pa.array(["basic", "premium", "basic"]).dictionary_encode()})
    baseline = build_baseline(table, ["plan"], _SCORES, model_name="m")
    assert baseline.features["plan"].kind == "categorical"
    assert set(baseline.features["plan"].categories or []) >= {"basic", "premium"}


def test_all_null_columns_are_omitted_from_baseline() -> None:
    table = pa.table(
        {
            "tenure": pa.array([1.0, 2.0, 3.0]),
            "null_num": pa.array([None, None, None], type=pa.float64()),
            "null_cat": pa.array([None, None, None], type=pa.string()),
        }
    )
    baseline = build_baseline(table, ["tenure", "null_num", "null_cat"], _SCORES, model_name="m")
    assert set(baseline.features) == {"tenure"}


def test_feature_column_missing_from_train_split_is_omitted() -> None:
    baseline = build_baseline(_baseline_table(), ["tenure", "ghost"], _SCORES, model_name="m")
    assert "ghost" not in baseline.features
    assert baseline.feature_columns == ["tenure", "ghost"]


def test_type_mismatched_current_columns_are_skipped() -> None:
    baseline = _baseline()
    # Numeric baseline vs string column; categorical baseline vs numeric column.
    current = pa.table(
        {
            "tenure": pa.array(["low", "high", "low"]),
            "plan": pa.array([1.0, 2.0, 3.0]),
        }
    )
    stats = compute_monitor_stats(baseline, current, _SCORES[:3], _monitors())
    assert stats.feature_shift == {}
    assert sorted(stats.skipped_features) == ["plan", "tenure"]


def test_all_null_current_columns_are_skipped() -> None:
    baseline = _baseline()
    current = pa.table(
        {
            "tenure": pa.array([None, None, None], type=pa.float64()),
            "plan": pa.array([None, None, None], type=pa.string()),
        }
    )
    stats = compute_monitor_stats(baseline, current, _SCORES[:3], _monitors())
    assert stats.feature_shift == {}
    assert sorted(stats.skipped_features) == ["plan", "tenure"]


def test_exclude_glob_drops_matching_features() -> None:
    baseline = _baseline()
    stats = compute_monitor_stats(
        baseline, _baseline_table(), _SCORES, _monitors(exclude=["plan*"])
    )
    assert set(stats.feature_shift) == {"tenure"}
    assert stats.skipped_features == []
