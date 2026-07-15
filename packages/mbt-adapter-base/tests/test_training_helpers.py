"""The shared training helpers: evaluate body, AUTO resolution, staged splits."""

from pathlib import Path

import pyarrow.parquet as pq
import pytest

from mbt_adapter_base.compliance import tiny_binary_dataset
from mbt_adapter_base.interchange import DatasetProfile
from mbt_adapter_base.specs import MetricSpec
from mbt_adapter_base.training_helpers import (
    evaluate_binary_split,
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


def test_evaluate_binary_split_computes_metrics_and_slices() -> None:
    import pyarrow as pa

    table = pa.table(
        {
            "label": [0, 1, 0, 1],
            "plan": ["a", "a", "b", "b"],  # both classes per slice
        }
    )
    scores = [0.1, 0.9, 0.2, 0.8]
    results = evaluate_binary_split(
        table, "label", scores, [MetricSpec(name="roc_auc")], slices=["plan"]
    )
    assert results.metrics["roc_auc"] == 1.0
    assert set(results.slices) == {"plan=a", "plan=b"}
    # a declared slice column absent from the table is simply skipped
    no_slice = evaluate_binary_split(
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
