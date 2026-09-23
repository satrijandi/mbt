"""An in-memory DataAdapter: the data seam's second adapter (A-1).

``mbt-testing`` shipped fakes for the training, tracking, registry, tuning,
reporting and compute seams and none for the data seam, so every dataset test
in the repo ran real DuckDB - slow, and a seam whose only implementation is the
production one is not really being tested as a seam.

That was expensive to fix while the eleven-step build recipe lived inside each
adapter: a fake would have had to reimplement the prologue, the zero-row policy,
the events and the metadata write to behave like the real thing. With the recipe
in ``materialization.build_dataset_materialization`` the fake is just the four
per-engine steps, which is the point of the seam.

It is pyarrow-only: tables are held in memory, filters are evaluated by a small
expression subset, and splits are written with the shared bucket arithmetic, so
membership matches every other backend by construction.
"""

from collections.abc import Sequence
from pathlib import Path
from typing import Any

import pyarrow as pa
import pyarrow.compute as pc
import pyarrow.parquet as pq

from mbt_adapter_base.errors import AdapterFailure
from mbt_adapter_base.interchange import DatasetLocator
from mbt_adapter_base.materialization import (
    SAMPLE_MODULUS,
    MaterializationError,
    MaterializedDatasetHandle,
    bucket_ranges,
    build_dataset_materialization,
    build_scoring_materialization,
    combine_snapshots,
    reference_bucket,
    split_fractions,
)
from mbt_adapter_base.predictions import LocalPredictionStore
from mbt_adapter_base.protocols import DataBuildContext, SourceTableLike
from mbt_adapter_base.specs import DatasetSpec, ScoringInputSpec, ScoringOutputSpec
from mbt_adapter_base.types import SplitStrategy


class InMemoryDataError(AdapterFailure):
    """The in-memory data adapter could not do what it was asked."""


class InMemoryDataAdapter:
    """A DataAdapter over tables handed to it, keyed by source unique_id.

    ``tables`` maps a source's ``unique_id`` to the Arrow table it stands for.
    ``predictions_root`` is where ``open_predictions`` stages runs; unset, it
    is a directory under the adapter's own root.
    """

    name = "inmemory"
    supported_source_formats = frozenset({"parquet", "arrow"})

    def __init__(self, config: dict[str, Any] | None = None) -> None:
        config = config or {}
        self.tables: dict[str, pa.Table] = dict(config.get("tables", {}))
        self.root = Path(config.get("root", "."))
        self.predictions_root = Path(config.get("predictions_root", self.root / "predictions"))

    def add_table(self, unique_id: str, table: pa.Table) -> None:
        self.tables[unique_id] = table

    # -- source resolution --------------------------------------------------

    def _table(self, source: SourceTableLike) -> pa.Table:
        for key in (source.identifier, source.path, source.name):
            if key is not None and key in self.tables:
                return self.tables[key]
        raise InMemoryDataError(
            f"no in-memory table registered for source {source.name!r}",
            hint="pass tables={'<source uid or name>': pa.table(...)} in the adapter config",
        )

    def snapshot_id(self, source: SourceTableLike, deep: bool = False) -> str:
        """Content digest of the registered table; ``deep`` changes nothing
        because an in-memory table has no mtime to be fooled by (ADR-11)."""
        import hashlib

        table = self._table(source)
        digest = hashlib.sha256()
        digest.update(str(table.schema).encode())
        for batch in table.to_batches():
            for column in batch.columns:
                digest.update(str(column.to_pylist()).encode())
        return "sha256:" + digest.hexdigest()

    # -- DatasetBuildEngine (A-1) -------------------------------------------

    def build_failure(
        self, message: str, *, ctx: DataBuildContext, hint: str | None = None
    ) -> Exception:
        return InMemoryDataError(message, hint, resource=ctx.node.unique_id)

    def verify_snapshot(self, ctx: DataBuildContext) -> None:
        if ctx.node.snapshot_id is None:
            return
        current = combine_snapshots(
            {uid: self.snapshot_id(table) for uid, table in ctx.source_tables.items()}
        )
        if current != ctx.node.snapshot_id:
            raise InMemoryDataError(
                f"source data changed under the pinned manifest: snapshot "
                f"{current} != pinned {ctx.node.snapshot_id}",
                hint="recompile to pin the new snapshot",
                resource=ctx.node.unique_id,
            )

    def write_dataset_splits(
        self, spec: DatasetSpec, ctx: DataBuildContext, output_dir: Path
    ) -> dict[str, int]:
        base = self._base_table(
            ctx.source_tables[spec.source], spec.filters, spec.sample_key_columns, ctx
        )
        written: dict[str, int] = {}
        if spec.split.strategy is SplitStrategy.TEMPORAL:
            assert spec.split.time_column is not None
            for split, (start, end) in sorted(ctx.resolved_windows.items()):
                rows = _within_window(base, spec.split.time_column, start, end)
                written[split] = _write(rows, output_dir / f"{split}.parquet")
            return written

        buckets = _buckets(base, spec.sample_key_columns, salt=str(spec.split.seed or 0))
        for split, lo, hi in bucket_ranges(split_fractions(spec.split)):
            mask = [lo <= bucket < hi for bucket in buckets]
            written[split] = _write(base.filter(pa.array(mask)), output_dir / f"{split}.parquet")
        return written

    def write_scoring_batch(self, spec: ScoringInputSpec, ctx: DataBuildContext, out: Path) -> int:
        base = self._base_table(
            ctx.source_tables[spec.source], spec.filters, spec.sample_key_columns, ctx
        )
        window = ctx.resolved_windows.get("score")
        if spec.time_column is not None and window is not None:
            base = _within_window(base, spec.time_column, *window)
        return _write(base, out)

    def _base_table(
        self,
        source: SourceTableLike,
        filters: Sequence[str],
        sample_keys: Sequence[str],
        ctx: DataBuildContext,
    ) -> pa.Table:
        table = self._table(source)
        for clause in filters:
            table = table.filter(_predicate(table, clause))
        if ctx.sample_fraction < 1.0:
            if not sample_keys:
                raise self.build_failure(
                    "sampling needs a stable row identity",
                    ctx=ctx,
                    hint="declare sample_key (the entity id column(s))",
                )
            threshold = int(ctx.sample_fraction * SAMPLE_MODULUS)
            keep = [bucket < threshold for bucket in _buckets(table, sample_keys)]
            table = table.filter(pa.array(keep))
        return table

    # -- the rest of the DataAdapter protocol --------------------------------

    def build_dataset(self, spec: DatasetSpec, ctx: DataBuildContext) -> MaterializedDatasetHandle:
        return build_dataset_materialization(self, spec, ctx)

    def build_scoring_input(
        self, spec: ScoringInputSpec, ctx: DataBuildContext
    ) -> MaterializedDatasetHandle:
        return build_scoring_materialization(self, spec, ctx)

    def from_locator(self, locator: DatasetLocator) -> MaterializedDatasetHandle:
        try:
            handle = MaterializedDatasetHandle(
                Path(locator.uri.removeprefix("file://")), adapter=self.name
            )
        except MaterializationError as exc:
            raise InMemoryDataError(str(exc)) from exc
        if handle.snapshot_id != locator.snapshot_id:
            raise InMemoryDataError(
                "dataset materialization snapshot mismatch: "
                f"{handle.snapshot_id} != {locator.snapshot_id}"
            )
        return handle

    def open_predictions(self, output: ScoringOutputSpec) -> LocalPredictionStore:
        return LocalPredictionStore(self.predictions_root / output.path)

    # -- source-level checks (declared on the protocol since v5, A-1) --------

    def count_source_duplicates(self, source: SourceTableLike, columns: list[str]) -> int:
        table = self._table(source)
        seen: dict[tuple[Any, ...], int] = {}
        for row in table.select(list(columns)).to_pylist():
            key = tuple(row[c] for c in columns)
            if any(value is None for value in key):
                continue  # null keys are ignored, as in dbt
            seen[key] = seen.get(key, 0) + 1
        return sum(1 for count in seen.values() if count > 1)

    def read_source_distinct(self, source: SourceTableLike, column: str) -> pa.Table:
        values = self._table(source).column(column).drop_null().unique()
        return pa.table({"value": values})


def _write(table: pa.Table, out: Path) -> int:
    pq.write_table(table, out)
    return int(table.num_rows)


def _buckets(table: pa.Table, key_columns: Sequence[str], salt: str = "") -> list[int]:
    """One canonical bucket per row, from the shared reference (F19)."""
    columns = [table.column(name).to_pylist() for name in key_columns]
    return [
        reference_bucket(["" if value is None else str(value) for value in values], salt=salt)
        for values in zip(*columns, strict=True)
    ]


def _within_window(table: pa.Table, time_column: str, start: str, end: str) -> pa.Table:
    """Half-open ``[start, end)`` on ``time_column``, as every engine applies it."""
    from datetime import datetime

    def _naive(iso: str) -> datetime:
        return datetime.fromisoformat(iso.replace("Z", "+00:00")).replace(tzinfo=None)

    column = table.column(time_column)
    low = pa.scalar(_naive(start)).cast(column.type)
    high = pa.scalar(_naive(end)).cast(column.type)
    return table.filter(pc.and_(pc.greater_equal(column, low), pc.less(column, high)))


def _predicate(table: pa.Table, clause: str) -> pa.Array:
    """Evaluate the small filter subset this fake supports.

    Deliberately narrow: ``<col> <op> <literal>`` with ``= != < <= > >=``, plus
    ``is true``/``is false``. A dataset test needing more SQL than that wants a
    real engine, and pretending otherwise would make the fake lie.
    """
    import re

    match = re.fullmatch(r"\s*(\w+)\s*(=|!=|<>|<=|>=|<|>)\s*('[^']*'|[-\w.]+)\s*", clause)
    if match is None:
        raise InMemoryDataError(
            f"the in-memory data adapter cannot evaluate filter {clause!r}",
            hint="it supports '<column> <op> <literal>' only; use a real engine for more",
        )
    name, operator, literal = match.groups()
    column = table.column(name)
    value: Any
    if literal.startswith("'"):
        value = literal[1:-1]
    elif literal.lower() in ("true", "false"):
        value = literal.lower() == "true"
    else:
        value = float(literal) if "." in literal else int(literal)
    scalar = pa.scalar(value).cast(column.type) if not isinstance(value, bool) else pa.scalar(value)
    return {
        "=": pc.equal,
        "!=": pc.not_equal,
        "<>": pc.not_equal,
        "<": pc.less,
        "<=": pc.less_equal,
        ">": pc.greater,
        ">=": pc.greater_equal,
    }[operator](column, scalar)


__all__ = ["InMemoryDataAdapter", "InMemoryDataError"]
