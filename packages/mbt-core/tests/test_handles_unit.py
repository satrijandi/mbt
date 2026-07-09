"""Unit tests for the transformed dataset handle (mbt/execute/handles.py)."""

import pyarrow as pa
import pytest
from exec_unit_helpers import minimal_model_spec

from mbt.contracts import DatasetLocator, HookContext
from mbt.events import get_bus
from mbt.exceptions import ConfigError
from mbt.execute.handles import TransformedDatasetHandle, select_feature_columns
from mbt_adapter_base.datasets import InMemoryDatasetHandle


class _LocatableHandle(InMemoryDatasetHandle):
    """InMemory handle that is also locatable (for delegation tests)."""

    def locator(self) -> DatasetLocator:
        return DatasetLocator(adapter="local", uri="file:///unit", snapshot_id=self.snapshot_id)


def _table() -> pa.Table:
    return pa.table({"a": [1.0, 2.0], "b": [3.0, 4.0], "y": [0, 1]})


def _hook_ctx_factory(spec):
    def factory(split: str) -> HookContext:
        return HookContext(spec=spec, profile=None, split=split, logger=get_bus())

    return factory


def test_select_feature_columns_empty_selection_errors() -> None:
    spec = minimal_model_spec(features={"include": ["*"], "exclude": ["a", "b"]})
    with pytest.raises(ConfigError, match="left no columns"):
        select_feature_columns(["a", "b", "y"], spec, None)


def test_read_with_explicit_columns() -> None:
    spec = minimal_model_spec()
    base = _LocatableHandle({"train": _table()}, label_column="y")
    handle = TransformedDatasetHandle(base, spec, None, _hook_ctx_factory(spec), None)
    selected = handle.read("train", columns=["a"])
    assert selected.column_names == ["a"]
    assert handle.feature_columns == ["a", "b"]


def test_hooks_transform_applies_per_split() -> None:
    spec = minimal_model_spec()

    class _Hooks:
        has_transform = True
        has_custom_metrics = False

        def __init__(self) -> None:
            self.splits_seen: list[str] = []

        def transform_features(self, table: pa.Table, ctx: HookContext) -> pa.Table:
            self.splits_seen.append(ctx.split)
            return table.append_column("derived", pa.array([1.0] * table.num_rows))

    hooks = _Hooks()
    base = _LocatableHandle({"train": _table()}, label_column="y")
    handle = TransformedDatasetHandle(base, spec, hooks, _hook_ctx_factory(spec), None)
    table = handle.read("train")
    assert "derived" in table.column_names
    assert hooks.splits_seen == ["train"]
    # the cache means a second read does not re-run the hook
    handle.read("train")
    assert hooks.splits_seen == ["train"]


def test_hooks_dropping_target_errors() -> None:
    spec = minimal_model_spec()

    class _DropTarget:
        has_transform = True
        has_custom_metrics = False

        def transform_features(self, table: pa.Table, ctx: HookContext) -> pa.Table:
            return table.drop_columns(["y"])

    base = _LocatableHandle({"train": _table()}, label_column="y")
    handle = TransformedDatasetHandle(base, spec, _DropTarget(), _hook_ctx_factory(spec), None)
    with pytest.raises(ConfigError, match="target column 'y' missing"):
        handle.read("train")


def test_profile_and_locator_delegate_to_base() -> None:
    spec = minimal_model_spec()
    base = _LocatableHandle({"train": _table()}, label_column="y")
    handle = TransformedDatasetHandle(base, spec, None, _hook_ctx_factory(spec), None)
    assert handle.profile().label_column == "y"
    assert handle.locator().uri == "file:///unit"
    assert handle.snapshot_id == base.snapshot_id
    assert handle.splits() == {"train"}
