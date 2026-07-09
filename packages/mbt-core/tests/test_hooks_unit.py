"""Unit tests for hooks loading and invocation (mbt/quality/hooks.py)."""

import sys
from pathlib import Path

import pyarrow as pa
import pytest
from core_helpers import write

from mbt.exceptions import AdapterError
from mbt.quality.hooks import ModelHooks, load_hooks


def _table() -> pa.Table:
    return pa.table({"a": [1.0, 2.0], "y": [0, 1]})


def test_load_hooks_with_transform_and_metrics(tmp_path: Path) -> None:
    write(
        tmp_path / "hooks" / "full_hooks.py",
        """
        import pyarrow as pa

        def transform_features(table, ctx):
            return table.append_column("derived", pa.array([1.0] * table.num_rows))

        def custom_metrics(predictions, ctx):
            return {"my_metric": 1, "other": 0.25}
        """,
    )
    hooks = load_hooks(tmp_path, "hooks/full_hooks.py")
    assert hooks.has_transform and hooks.has_custom_metrics
    transformed = hooks.transform_features(_table(), None)
    assert "derived" in transformed.column_names
    metrics = hooks.custom_metrics(_table(), None)
    assert metrics == {"my_metric": 1.0, "other": 0.25}
    assert all(isinstance(v, float) for v in metrics.values())


def test_hooks_without_functions_are_noops(tmp_path: Path) -> None:
    write(tmp_path / "hooks" / "empty_hooks.py", "X = 1\n")
    hooks = load_hooks(tmp_path, "hooks/empty_hooks.py")
    assert not hooks.has_transform and not hooks.has_custom_metrics
    table = _table()
    assert hooks.transform_features(table, None) is table
    assert hooks.custom_metrics(table, None) == {}


def test_transform_returning_non_table_errors(tmp_path: Path) -> None:
    write(
        tmp_path / "hooks" / "bad_transform.py",
        """
        def transform_features(table, ctx):
            return [1, 2, 3]
        """,
    )
    hooks = load_hooks(tmp_path, "hooks/bad_transform.py")
    with pytest.raises(AdapterError, match=r"must return a pyarrow\.Table"):
        hooks.transform_features(_table(), None)


def test_custom_metrics_returning_non_dict_errors(tmp_path: Path) -> None:
    write(
        tmp_path / "hooks" / "bad_metrics.py",
        """
        def custom_metrics(predictions, ctx):
            return [0.5]
        """,
    )
    hooks = load_hooks(tmp_path, "hooks/bad_metrics.py")
    with pytest.raises(AdapterError, match="must return a dict"):
        hooks.custom_metrics(_table(), None)


def test_load_hooks_missing_file_errors(tmp_path: Path) -> None:
    with pytest.raises(AdapterError, match="hooks file not found"):
        load_hooks(tmp_path, "hooks/absent.py")


def test_load_hooks_import_failure_cleans_sys_modules(tmp_path: Path) -> None:
    write(
        tmp_path / "hooks" / "explodes.py",
        """
        raise RuntimeError("boom at import")
        """,
    )
    with pytest.raises(AdapterError, match="failed to import"):
        load_hooks(tmp_path, "hooks/explodes.py")
    assert "_mbt_hooks_explodes" not in sys.modules


def test_model_hooks_wraps_plain_module(tmp_path: Path) -> None:
    write(tmp_path / "plain_hooks_module.py", "transform_features = 'not callable'\n")
    hooks = load_hooks(tmp_path, "plain_hooks_module.py")
    assert isinstance(hooks, ModelHooks)
    assert not hooks.has_transform  # attribute exists but is not callable
