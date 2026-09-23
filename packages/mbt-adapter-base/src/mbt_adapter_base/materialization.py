"""The shared dataset materialization: its format, and the recipe that builds it.

A materialization is a directory holding one parquet file per split plus
``materialization.json`` (metadata) and a ``_SUCCESS`` marker. The local
DuckDB adapter, the Snowflake adapter, and any future warehouse adapter all
write this layout, so the training job reopens datasets the same way no
matter where the rows came from (``DataAdapter.from_locator``).

**The build recipe lives here too** (A-1). ``build_dataset`` is one fixed
eleven-step sequence, and all three data adapters used to write it out: the
same output-dir prologue, the same sample-fraction check, the same zero-row
loop with its ``out_of_time`` exemption, the same row-count event, the same
metadata write. Two of the three said so in a comment - "mirrors the local
adapter" - which is a contract held in prose, and the prose did not hold: the
local adapter emitted the empty-after-test warning as a typed WARN while the
other two emitted a bare string the bus logged at info (v5 live defect 1).

So the policy is ``build_dataset_materialization`` and the per-engine half is
``DatasetBuildEngine``: read the relation, apply filters, express the sampling
predicate, write the splits. Four methods, and nothing about ordering.

The bucket-edge arithmetic (``bucket_ranges``) matters most of all, because it
decides which rows train. It was written three times - once per adapter, each
pinned by its own test to a reference the local adapter's docstring described
in prose. It is written once here, and ``compliance.DataAdapterCompliance``
pins every engine to the same digest.
"""

import hashlib
import json
from collections.abc import Mapping, Sequence
from pathlib import Path
from typing import TYPE_CHECKING, Any, Protocol

import pyarrow as pa
import pyarrow.compute as pc
import pyarrow.parquet as pq

from mbt_adapter_base.events import (
    DatasetMaterialized,
    EmptyAfterTestSplit,
    ScoringInputMaterialized,
)
from mbt_adapter_base.interchange import DatasetLocator, DatasetProfile
from mbt_adapter_base.types import OUT_OF_TIME_SPLIT

if TYPE_CHECKING:
    from mbt_adapter_base.protocols import DataBuildContext
    from mbt_adapter_base.specs import DatasetSpec, ScoringInputSpec, SplitSpec

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
) -> None:
    """Write ``materialization.json`` and the ``_SUCCESS`` marker."""
    metadata = {
        "snapshot_id": snapshot_id,
        "dataset": dataset,
        "label_column": label_column,
        "time_column": time_column,
        "windows": {k: list(v) for k, v in windows.items()},
        "sample_fraction": sample_fraction,
        "row_counts": dict(row_counts),
    }
    (directory / METADATA_FILE).write_text(json.dumps(metadata, indent=2, sort_keys=True))
    (directory / SUCCESS_FILE).write_text("")


# -- the build recipe (A-1) -------------------------------------------------


class DatasetBuildEngine(Protocol):
    """The genuinely per-engine half of a dataset build.

    Everything else - the output-dir prologue, the sample-fraction check, the
    zero-row loop, the events, the metadata write, the handle - is policy and
    lives in ``build_dataset_materialization`` / ``build_scoring_materialization``.

    An implementation reads one relation, applies the spec's filters, expresses
    the sampling predicate in its own dialect, and writes one parquet file per
    split. It decides nothing about severity, ordering, or wording.
    """

    @property
    def name(self) -> str:
        """The adapter name recorded on the materialization's locator."""
        ...

    def build_failure(
        self, message: str, *, ctx: "DataBuildContext", hint: str | None = None
    ) -> Exception:
        """This engine's exception type, so callers keep catching what they did.

        The recipe raises what this returns rather than a shared class, because
        ``AdapterError``, ``SparkAdapterError`` and ``SnowflakeAdapterError``
        have different constructors and existing callers catch the specific one.
        """
        ...

    def verify_snapshot(self, ctx: "DataBuildContext") -> None:
        """Fail if the source data no longer matches the manifest's pin.

        A no-op when ``ctx.node.snapshot_id`` is None. Scoring inputs are
        expected to change every run and are never verified (R2-10), which is
        why this is on the dataset path only.
        """
        ...

    def write_dataset_splits(
        self, spec: "DatasetSpec", ctx: "DataBuildContext", output_dir: Path
    ) -> dict[str, int]:
        """Write ``<split>.parquet`` for every split; return the row counts.

        Splits come from ``ctx.resolved_windows`` (temporal) or from
        ``bucket_ranges(split_fractions(spec.split))`` (random) - use the shared
        arithmetic for the latter, never a local re-derivation.
        """
        ...

    def write_scoring_batch(
        self, spec: "ScoringInputSpec", ctx: "DataBuildContext", out: Path
    ) -> int:
        """Write one unlabeled batch to ``out``; return its row count."""
        ...


def prepare_output_dir(output_dir: Path) -> Path:
    """Empty the materialization directory, creating it if needed.

    A build always writes a complete materialization or none, so leftovers from
    a previous (possibly differently-split) build cannot survive into this one.
    """
    output_dir.mkdir(parents=True, exist_ok=True)
    for stale in output_dir.glob("*"):
        stale.unlink()
    return output_dir


def check_sample_fraction(engine: DatasetBuildEngine, ctx: "DataBuildContext") -> float:
    """Validate ``ctx.sample_fraction`` and return it."""
    fraction = ctx.sample_fraction
    if not 0.0 < fraction <= 1.0:
        raise engine.build_failure(
            f"sample_fraction must be in (0, 1], got {fraction}",
            ctx=ctx,
            hint="set the 'sample_fraction' var in the target's vars",
        )
    return fraction


def split_fractions(split: "SplitSpec") -> dict[str, float]:
    """The random strategy's per-split fractions, in bucket order.

    Order is load-bearing: it fixes which hash-bucket range each split gets, so
    train/validation/test must be built in this order by every engine.
    """
    fractions: dict[str, float] = {"train": float(split.train)}
    if split.validation is not None:
        fractions["validation"] = float(split.validation)
    fractions["test"] = float(split.test)
    return fractions


def bucket_ranges(fractions: Mapping[str, float]) -> list[tuple[str, int, int]]:
    """``(split, lo, hi)`` half-open hash-bucket edges over ``SAMPLE_MODULUS``.

    This is the arithmetic that decides which rows train, so it exists exactly
    once (A-1) and every engine's SQL is pinned to it by
    ``DataAdapterCompliance``. The final split's upper bound is the modulus
    itself, so no bucket falls through a float-accumulation gap (F19).
    """
    ranges: list[tuple[str, int, int]] = []
    low = 0.0
    entries = list(fractions.items())
    for index, (split, fraction) in enumerate(entries):
        lo = int(low * SAMPLE_MODULUS)
        last = index == len(entries) - 1
        hi = SAMPLE_MODULUS if last else int((low + fraction) * SAMPLE_MODULUS)
        ranges.append((split, lo, hi))
        low += fraction
    return ranges


def reference_bucket(key_values: "Sequence[str]", salt: str = "") -> int:
    """THE canonical cross-adapter row bucket (F19), in pure Python.

    The unsigned lower 64 bits of the md5 of the ``'|'``-joined preimage (the
    salt first when present, then each key column as text), modulo
    ``SAMPLE_MODULUS``. Snowflake's ``MD5_NUMBER_LOWER64``, Spark's
    ``conv(substring(md5(...), 17, 16), 16, 10)`` and DuckDB's hex-slice cast
    all compute this value, which is why a row trains on every backend or on
    none.

    It lives here because it was previously re-derived in three separate test
    files, each pinning one adapter's SQL to a local copy of the same six
    lines (A-1). ``DataAdapterCompliance`` now pins every engine to this one.

    md5 is a bucket function, not a security primitive: the preimage is a row
    key and the output is a split assignment.
    """
    parts = [salt, *key_values] if salt else list(key_values)
    digest = hashlib.md5("|".join(parts).encode(), usedforsecurity=False).hexdigest()
    return int(digest[16:32], 16) % SAMPLE_MODULUS


def reference_split(key_values: "Sequence[str]", fractions: Mapping[str, float], seed: int) -> str:
    """Which split a row lands in under the random strategy, by the reference."""
    bucket = reference_bucket(key_values, salt=str(seed))
    for split, lo, hi in bucket_ranges(fractions):
        if lo <= bucket < hi:
            return split
    raise AssertionError(  # pragma: no cover - bucket_ranges covers [0, MOD)
        f"bucket {bucket} fell outside every split range"
    )


def _report_split_counts(
    engine: DatasetBuildEngine, ctx: "DataBuildContext", written: Mapping[str, int]
) -> None:
    """Zero-row policy, then the row-count line - once, for every engine.

    An empty ``out_of_time`` split is the one empty split that is not an error
    (ADR-30): a window ending at the anchor is routinely empty until newer rows
    land upstream. Every other empty split means the windows or fractions do not
    match the data, which is a build failure.
    """
    for split, count in sorted(written.items()):
        if count != 0:
            continue
        if split == OUT_OF_TIME_SPLIT:
            ctx.events.emit(
                EmptyAfterTestSplit(
                    unique_id=ctx.node.unique_id,
                    window=ctx.resolved_windows.get(OUT_OF_TIME_SPLIT),
                )
            )
            continue
        raise engine.build_failure(
            f"split {split!r} materialized 0 rows",
            ctx=ctx,
            hint="check the split windows/fractions against the data's time range",
        )


def build_dataset_materialization(
    engine: DatasetBuildEngine, spec: "DatasetSpec", ctx: "DataBuildContext"
) -> "MaterializedDatasetHandle":
    """Build one dataset materialization: the whole recipe, written once (A-1)."""
    engine.verify_snapshot(ctx)
    output_dir = prepare_output_dir(ctx.output_dir)
    check_sample_fraction(engine, ctx)

    written = engine.write_dataset_splits(spec, ctx, output_dir)

    _report_split_counts(engine, ctx, written)
    ctx.events.emit(
        DatasetMaterialized(
            unique_id=ctx.node.unique_id, dataset=spec.name, row_counts=dict(written)
        )
    )
    write_materialization_metadata(
        output_dir,
        snapshot_id=ctx.node.snapshot_id,
        dataset=spec.name,
        label_column=spec.label.column,
        time_column=spec.split.time_column,
        windows=ctx.resolved_windows,
        sample_fraction=ctx.sample_fraction,
        row_counts=written,
    )
    return MaterializedDatasetHandle(output_dir, adapter=engine.name)


def build_scoring_materialization(
    engine: DatasetBuildEngine, spec: "ScoringInputSpec", ctx: "DataBuildContext"
) -> "MaterializedDatasetHandle":
    """Build one unlabeled scoring batch as a single ``score`` split (ADR-20).

    No snapshot verification, deliberately: a scoring input - and the arriving
    labels ``mbt monitor`` reads through this same path - is expected to change
    every run, so a pinned manifest scores the live data rather than hard-failing
    on drift the way a dataset does (R2-10). The node's ``snapshot_id`` is still
    recorded for provenance.

    Zero rows warns rather than failing: an empty nightly batch is legitimate.
    """
    output_dir = prepare_output_dir(ctx.output_dir)
    check_sample_fraction(engine, ctx)

    count = engine.write_scoring_batch(spec, ctx, output_dir / "score.parquet")

    ctx.events.emit(ScoringInputMaterialized(unique_id=ctx.node.unique_id, rows=count))
    write_materialization_metadata(
        output_dir,
        snapshot_id=ctx.node.snapshot_id,
        dataset=ctx.node.name,
        label_column="",  # unlabeled by design (ADR-20)
        time_column=spec.time_column,
        windows=ctx.resolved_windows,
        sample_fraction=ctx.sample_fraction,
        row_counts={"score": count},
    )
    return MaterializedDatasetHandle(output_dir, adapter=engine.name)


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
    def metadata(self) -> dict[str, Any]:
        """A copy of ``materialization.json``: windows, sample fraction, row counts."""
        return dict(self._metadata)

    @property
    def time_column(self) -> str | None:
        value = self._metadata.get("time_column")
        return str(value) if value is not None else None

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
