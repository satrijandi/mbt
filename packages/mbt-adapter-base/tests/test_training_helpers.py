"""The shared training helpers: evaluate body, AUTO resolution, staged splits."""

from pathlib import Path

import pyarrow.parquet as pq
import pytest

from mbt_adapter_base.compliance import tiny_binary_dataset
from mbt_adapter_base.events import EarlyStoppingWithoutValidation
from mbt_adapter_base.interchange import DatasetProfile
from mbt_adapter_base.specs import MetricSpec
from mbt_adapter_base.training_helpers import (
    evaluate_split,
    note_early_stopping_without_validation,
    positive_rate,
    resolve_scale_pos_weight,
    staged_split_path,
)


def _profile(balance: dict[str, float] | None) -> DatasetProfile:
    return DatasetProfile(
        n_rows={"train": 10, "test": 5},
        columns={"f": "double", "label": "int64"},
        label_column="label",
        label_balance=balance,
    )


def test_evaluate_split_computes_metrics_and_slices() -> None:
    import pyarrow as pa

    table = pa.table(
        {
            "label": [0, 1, 0, 1],
            "plan": ["a", "a", "b", "b"],  # both classes per slice
        }
    )
    scores = [0.1, 0.9, 0.2, 0.8]
    results = evaluate_split(table, "label", scores, [MetricSpec(name="roc_auc")], slices=["plan"])
    assert results.metrics["roc_auc"] == 1.0
    assert set(results.slices) == {"plan=a", "plan=b"}
    # a declared slice column absent from the table is simply skipped
    no_slice = evaluate_split(
        table, "label", scores, [MetricSpec(name="roc_auc")], slices=["missing"]
    )
    assert no_slice.slices == {}


def test_resolve_scale_pos_weight_is_six_decimals_or_a_hard_error() -> None:
    assert resolve_scale_pos_weight(_profile({"1": 0.22, "0": 0.78})) == round(0.78 / 0.22, 6)
    assert positive_rate(_profile({"true": 0.4})) == 0.4
    assert positive_rate(_profile(None)) is None
    with pytest.raises(ValueError, match="no positive-class balance"):
        resolve_scale_pos_weight(_profile({"0": 1.0}))


def test_staged_split_path_prefers_the_handles_own_file(tmp_path: Path) -> None:
    class _PathHandle:
        def split_path(self, split: str) -> Path:
            return tmp_path / f"{split}.parquet"

    out = staged_split_path(_PathHandle(), "train", prefix="mbt-test-stage-")
    assert out == tmp_path / "train.parquet"


def test_staged_split_path_stages_handles_without_disk_backing() -> None:
    handle = tiny_binary_dataset()
    out = staged_split_path(handle, "train", prefix="mbt-test-stage-")
    assert out.is_file() and out.name == "train.parquet"
    assert pq.read_table(out).num_rows == handle.read("train").num_rows


def test_pooling_leaves_a_level_set_alone_when_nothing_is_rare() -> None:
    """No level below the threshold means no `__other__` bucket - a pooled
    level would shift every other level's code for no reason, and codes are a
    persisted part of the artifact (ADR-27)."""
    import pyarrow as pa

    from mbt_adapter_base.encoding import train_categories
    from mbt_adapter_base.specs import CategoricalPolicy

    table = pa.table({"r": ["north"] * 50 + ["south"] * 50})
    levels = train_categories(table, ["r"], {"r": CategoricalPolicy(min_frequency=0.1)})
    assert levels["r"] == ["north", "south"]


def test_pooling_of_an_empty_column_is_a_no_op() -> None:
    import pyarrow as pa

    from mbt_adapter_base.encoding import train_categories
    from mbt_adapter_base.specs import CategoricalPolicy

    table = pa.table({"r": pa.array([None, None], type=pa.string())})
    assert train_categories(table, ["r"], {"r": CategoricalPolicy(min_frequency=0.1)})["r"] == []


def test_early_stopping_without_a_validation_split_is_said_out_loud() -> None:
    from mbt_adapter_base.datasets import InMemoryDatasetHandle

    class _Sink:
        def __init__(self) -> None:
            self.messages: list[object] = []

        def emit(self, event: object) -> None:
            self.messages.append(event)

    base = tiny_binary_dataset()
    no_validation = InMemoryDatasetHandle(
        {"train": base.read("train"), "test": base.read("test")}, label_column="label"
    )
    with_validation = InMemoryDatasetHandle(
        {"train": base.read("train"), "validation": base.read("test")}, label_column="label"
    )

    sink = _Sink()
    assert note_early_stopping_without_validation(no_validation, 30, sink, adapter="xgboost")
    # A typed WARN, not a bare string at the bus's default level (B-4), and
    # worded as the CONSEQUENCE rather than as a missing declaration (D-2).
    [said] = sink.messages
    assert isinstance(said, EarlyStoppingWithoutValidation)
    assert (said.adapter, said.rounds, said.level) == ("xgboost", 30, "warn")
    assert "regularized differently from the one the search scored" in said.human()
    # nothing to say when early stopping is off, or has a split to watch
    quiet = _Sink()
    assert not note_early_stopping_without_validation(no_validation, None, quiet, adapter="x")
    assert not note_early_stopping_without_validation(with_validation, 30, quiet, adapter="x")
    assert quiet.messages == []
