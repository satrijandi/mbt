"""The shared dataset-materialization format any DataAdapter can produce.

A materialization is a directory holding one parquet file per split plus
``materialization.json`` (metadata) and a ``_SUCCESS`` marker. The local
DuckDB adapter, the Snowflake adapter, and any future warehouse adapter all
write this layout, so the training job reopens datasets the same way no
matter where the rows came from (``DataAdapter.from_locator``).
"""

import hashlib
import json
from collections.abc import Mapping
from pathlib import Path
from typing import Any

import pyarrow as pa
import pyarrow.compute as pc
import pyarrow.parquet as pq

from mbt_adapter_base.interchange import DatasetLocator, DatasetProfile

METADATA_FILE = "materialization.json"
PROFILE_FILE = "profile.json"
SUCCESS_FILE = "_SUCCESS"

#: Modulus for deterministic hash sampling; thresholds are ``fraction * MOD``.
SAMPLE_MODULUS = 1_000_000


class MaterializationError(RuntimeError):
    """A materialization directory is missing or incomplete."""


def combine_snapshots(snapshots: Mapping[str, str | None]) -> str | None:
    """One snapshot id for a dataset built from one or more sources.

    A single source keeps its id verbatim; multiple sources combine into a
    stable digest over the sorted ``uid=snapshot`` pairs, so any source
    changing flips the dataset's identity (ADR-4).
    """
    present = {uid: snap for uid, snap in snapshots.items() if snap}
    if not present:
        return None
    if len(present) == 1:
        return next(iter(present.values()))
    digest = hashlib.sha256()
    for uid, snap in sorted(present.items()):
        digest.update(f"{uid}={snap}\n".encode())
    return "sha256:" + digest.hexdigest()


def write_materialization_metadata(
    directory: Path,
    *,
    snapshot_id: str | None,
    dataset: str,
    label_column: str,
    time_column: str | None,
    windows: Mapping[str, Any],
    sample_fraction: float,
    row_counts: Mapping[str, int],
    label_join_coverage: Mapping[str, int] | None = None,
) -> None:
    """Write ``materialization.json`` and the ``_SUCCESS`` marker.

    ``label_join_coverage`` (population-spine datasets, F21) records
    ``{"spine_rows": N, "matched_rows": M}`` - how many spine rows survived the
    inner label join, measured before filters/sampling/windows so the ratio
    isolates the join drop. The ``label_join_coverage`` check enforces it.
    """
    metadata = {
        "snapshot_id": snapshot_id,
        "dataset": dataset,
        "label_column": label_column,
        "time_column": time_column,
        "windows": {k: list(v) for k, v in windows.items()},
        "sample_fraction": sample_fraction,
        "row_counts": dict(row_counts),
    }
    if label_join_coverage is not None:
        metadata["label_join_coverage"] = dict(label_join_coverage)
    (directory / METADATA_FILE).write_text(json.dumps(metadata, indent=2, sort_keys=True))
    (directory / SUCCESS_FILE).write_text("")


class MaterializedDatasetHandle:
    """DatasetHandle over a materialization directory (pyarrow-backed)."""

    def __init__(self, directory: Path, *, adapter: str = "local") -> None:
        self.directory = directory
        self._adapter = adapter
        metadata_path = directory / METADATA_FILE
        if not metadata_path.is_file() or not (directory / SUCCESS_FILE).is_file():
            raise MaterializationError(
                f"no complete dataset materialization at {directory}; "
                "the dataset build may have failed - re-run without a warm cache"
            )
        self._metadata: dict[str, Any] = json.loads(metadata_path.read_text())
        self._profile: DatasetProfile | None = None

    @property
    def snapshot_id(self) -> str:
        return str(self._metadata["snapshot_id"])

    @property
    def label_column(self) -> str:
        return str(self._metadata["label_column"])

    @property
    def time_column(self) -> str | None:
        value = self._metadata.get("time_column")
        return str(value) if value is not None else None

    @property
    def label_join_coverage(self) -> dict[str, int] | None:
        """``{"spine_rows": N, "matched_rows": M}`` for a population-spine
        dataset (F21), or None when the build recorded no coverage (single-table
        datasets, or a materialization from an older mbt)."""
        value = self._metadata.get("label_join_coverage")
        return {k: int(v) for k, v in value.items()} if value is not None else None

    def splits(self) -> set[str]:
        return {p.stem for p in self.directory.glob("*.parquet")}

    def split_path(self, split: str) -> Path:
        path = self.directory / f"{split}.parquet"
        if not path.is_file():
            raise MaterializationError(
                f"split {split!r} is not materialized at {self.directory} "
                f"(available: {', '.join(sorted(self.splits()))})"
            )
        return path

    def read(self, split: str, columns: list[str] | None = None) -> pa.Table:
        return pq.read_table(self.split_path(split), columns=columns)

    def profile(self) -> DatasetProfile:
        if self._profile is not None:
            return self._profile
        profile_path = self.directory / PROFILE_FILE
        if profile_path.is_file():
            self._profile = DatasetProfile.model_validate_json(profile_path.read_text())
            return self._profile
        self._profile = self._compute_profile()
        profile_path.write_text(self._profile.model_dump_json(indent=2))
        return self._profile

    def _compute_profile(self) -> DatasetProfile:
        n_rows: dict[str, int] = {}
        for split in sorted(self.splits()):
            n_rows[split] = pq.ParquetFile(self.split_path(split)).metadata.num_rows

        # Schema from the train split when present; scoring materializations
        # hold a single "score" split (ADR-20).
        schema_split = "train" if "train" in n_rows else min(sorted(n_rows))
        schema = pq.read_schema(self.split_path(schema_split))
        columns = {field.name: str(field.type) for field in schema}

        label_balance: dict[str, float] | None = None
        label = self.label_column
        if label in columns and n_rows.get("train", 0) > 0:
            values = pq.read_table(self.split_path("train"), columns=[label]).column(label)
            counts = pc.value_counts(values)
            total = len(values)
            label_balance = {
                str(entry["values"].as_py()): entry["counts"].as_py() / total for entry in counts
            }

        time_range: tuple[str, str] | None = None
        time_column = self.time_column
        if time_column and time_column in columns:
            lows: list[Any] = []
            highs: list[Any] = []
            for split in sorted(self.splits()):
                column = pq.read_table(self.split_path(split), columns=[time_column]).column(
                    time_column
                )
                if len(column) == 0:
                    continue
                bounds = pc.min_max(column)
                lows.append(bounds["min"].as_py())
                highs.append(bounds["max"].as_py())
            if lows:
                time_range = (str(min(lows)), str(max(highs)))

        return DatasetProfile(
            n_rows=n_rows,
            columns=columns,
            label_column=label,
            label_balance=label_balance,
            time_range=time_range,
        )

    def locator(self) -> DatasetLocator:
        return DatasetLocator(
            adapter=self._adapter,
            uri=f"file://{self.directory.resolve()}",
            snapshot_id=self.snapshot_id,
        )
