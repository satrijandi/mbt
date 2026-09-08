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


def test_feature_treatment_applies_after_selection(monkeypatch) -> None:
    """ADR-27 treatment runs on the projected table, so `include`/`exclude`
    decide membership first and treatment only touches what survived."""
    spec = minimal_model_spec(
        features={
            "include": ["*"],
            "exclude": ["b"],
            "transforms": {"a": {"cap": 1.5}},
        }
    )
    base = _LocatableHandle({"train": _table()}, label_column="y")
    handle = TransformedDatasetHandle(base, spec, None, _hook_ctx_factory(spec), None)
    table = handle.read("train")
    assert table.column_names == ["a", "y"]
    assert table.column("a").to_pylist() == [1.0, 1.5]


def test_feature_treatment_applies_to_an_unlabelled_scoring_split() -> None:
    """Same spec, same code path, no target: train and score cannot diverge."""
    spec = minimal_model_spec(features={"include": ["*"], "transforms": {"a": {"cap": 1.5}}})
    labelled = _LocatableHandle({"train": _table()}, label_column="y")
    unlabelled = _LocatableHandle(
        {"score": pa.table({"a": [1.0, 2.0], "b": [3.0, 4.0]})}, label_column="y"
    )
    trained = TransformedDatasetHandle(labelled, spec, None, _hook_ctx_factory(spec), None)
    scored = TransformedDatasetHandle(
        unlabelled, spec, None, _hook_ctx_factory(spec), None, require_target=False
    )
    assert (
        trained.read("train").column("a").to_pylist()
        == scored.read("score").column("a").to_pylist()
    )


def test_treatment_sees_hook_derived_columns() -> None:
    """Hooks run first, so a column a hook computes can be treated by name."""

    class _Hooks:
        has_transform = True

        def transform_features(self, table: pa.Table, ctx: HookContext) -> pa.Table:
            return table.append_column("derived", pa.array([10.0, 900.0]))

    spec = minimal_model_spec(features={"include": ["*"], "transforms": {"derived": {"cap": 100}}})
    base = _LocatableHandle({"train": _table()}, label_column="y")
    handle = TransformedDatasetHandle(base, spec, _Hooks(), _hook_ctx_factory(spec), None)
    assert handle.read("train").column("derived").to_pylist() == [10.0, 100.0]


def test_slice_columns_are_not_treatable_features() -> None:
    """A declared slice rides along for evaluation but the adapters drop it, so
    it is neither treatable nor caught by the authoritative-categorical rule."""
    spec = minimal_model_spec(
        features={"include": ["*"], "categorical": []},
        evaluation={"protocol": {"split": "temporal"}, "metrics": ["pr_auc"], "slices": ["region"]},
    )
    table = pa.table({"a": [1.0, 2.0], "region": ["north", "south"], "y": [0, 1]})
    base = _LocatableHandle({"train": table}, label_column="y")
    handle = TransformedDatasetHandle(base, spec, None, _hook_ctx_factory(spec), None)
    # `categorical: []` asserts no categoricals; the string slice does not count.
    assert handle.read("train").column("region").to_pylist() == ["north", "south"]

    treated = minimal_model_spec(
        features={"include": ["*"], "categorical": ["region"]},
        evaluation={"protocol": {"split": "temporal"}, "metrics": ["pr_auc"], "slices": ["region"]},
    )
    handle = TransformedDatasetHandle(base, treated, None, _hook_ctx_factory(treated), None)
    with pytest.raises(ConfigError, match="does not consume: region"):
        handle.read("train")
