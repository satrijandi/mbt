"""InMemoryDatasetHandle and MaterializedDatasetHandle edge behavior."""

from pathlib import Path

import pyarrow as pa
import pyarrow.parquet as pq
import pytest

from mbt_adapter_base.datasets import InMemoryDatasetHandle
from mbt_adapter_base.materialization import (
    SUCCESS_FILE,
    MaterializationError,
    MaterializedDatasetHandle,
    combine_snapshots,
    write_materialization_metadata,
)


def _train_table() -> pa.Table:
    return pa.table(
        {
            "user_id": [1, 2, 3, 4],
            "ts": ["2026-01-01", "2026-01-02", "2026-01-03", "2026-01-04"],
            "label": [0, 1, 0, 1],
        }
    )


# -- InMemoryDatasetHandle ---------------------------------------------------------


def _in_memory_handle() -> InMemoryDatasetHandle:
    train = _train_table()
    return InMemoryDatasetHandle(
        {"train": train, "empty": train.slice(0, 0)},
        snapshot_id="sha256:test",
        label_column="label",
        time_column="ts",
    )


def test_in_memory_metadata_properties() -> None:
    handle = _in_memory_handle()
    assert handle.snapshot_id == "sha256:test"
    assert handle.label_column == "label"
    assert handle.time_column == "ts"
    assert handle.splits() == {"train", "empty"}


def test_in_memory_read_projects_columns() -> None:
    handle = _in_memory_handle()
    assert handle.read("train", columns=["user_id"]).column_names == ["user_id"]


def test_in_memory_with_split_preserves_metadata() -> None:
    handle = _in_memory_handle()
    validation = _train_table().slice(0, 2)
    carved = handle.with_split("validation", validation)
    assert carved.splits() == {"train", "empty", "validation"}
    assert carved.read("validation").num_rows == 2
    assert carved.snapshot_id == handle.snapshot_id
    assert carved.label_column == handle.label_column
    assert carved.time_column == handle.time_column


def test_in_memory_profile_skips_empty_splits_for_time_range() -> None:
    profile = _in_memory_handle().profile()
    assert profile.n_rows == {"train": 4, "empty": 0}
    assert profile.time_range == ("2026-01-01", "2026-01-04")
    assert profile.label_balance == {"0": 0.5, "1": 0.5}


def test_in_memory_handle_is_not_locatable() -> None:
    with pytest.raises(NotImplementedError, match="not locatable"):
        _in_memory_handle().locator()


# -- combine_snapshots -------------------------------------------------------------


def test_combine_snapshots_without_any_snapshot_is_none() -> None:
    assert combine_snapshots({}) is None
    assert combine_snapshots({"source.a.t": None}) is None


# -- MaterializedDatasetHandle -----------------------------------------------------


def _materialization_dir(tmp_path: Path) -> Path:
    directory = tmp_path / "dataset"
    directory.mkdir()
    train = _train_table()
    pq.write_table(train, directory / "train.parquet")
    pq.write_table(train.slice(0, 0), directory / "empty.parquet")
    write_materialization_metadata(
        directory,
        snapshot_id="sha256:abc",
        dataset="churn_training_set",
        label_column="label",
        time_column="ts",
        windows={"train": ("-180d", "-28d")},
        sample_fraction=1.0,
        row_counts={"train": 4, "empty": 0},
    )
    return directory


def test_missing_materialization_raises(tmp_path: Path) -> None:
    with pytest.raises(MaterializationError, match="no complete dataset materialization"):
        MaterializedDatasetHandle(tmp_path / "nowhere")


def test_materialization_without_success_marker_raises(tmp_path: Path) -> None:
    directory = _materialization_dir(tmp_path)
    (directory / SUCCESS_FILE).unlink()
    with pytest.raises(MaterializationError, match="no complete dataset materialization"):
        MaterializedDatasetHandle(directory)


def test_unmaterialized_split_raises(tmp_path: Path) -> None:
    handle = MaterializedDatasetHandle(_materialization_dir(tmp_path))
    with pytest.raises(MaterializationError, match="'validation' is not materialized"):
        handle.split_path("validation")


def test_profile_skips_empty_splits_and_caches(tmp_path: Path) -> None:
    handle = MaterializedDatasetHandle(_materialization_dir(tmp_path))
    profile = handle.profile()
    assert profile.n_rows == {"empty": 0, "train": 4}
    assert profile.time_range == ("2026-01-01", "2026-01-04")
    assert profile.label_balance == {"0": 0.5, "1": 0.5}
    assert handle.profile() is profile
