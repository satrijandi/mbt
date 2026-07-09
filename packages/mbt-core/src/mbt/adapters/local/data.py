"""Local Parquet DataAdapter via DuckDB (TSD §13.2, FR-ADPT-04).

Sources resolve to Parquet globs under ``config.root``. ``build_dataset``
runs one DuckDB query: read the source table(s) - joining feature tables
onto the label table for multi-table ``inputs`` datasets - then filters,
deterministic hash sampling, and split assignment (resolved temporal
windows, or seeded hash split), writing one Parquet file per split.

Sampling reproducibility: rows are kept when
``md5_number(key) % 1_000_000 < fraction * 1_000_000`` over the dataset's
``sample_key`` (or join key). The same fraction always keeps the same rows,
and smaller fractions are subsets of larger ones. Without a key the digest
falls back to hashing every column - correct, but slow on wide tables.
"""

import glob as globlib
import hashlib
from dataclasses import dataclass
from datetime import datetime
from pathlib import Path
from typing import Any

import duckdb

from mbt.contracts import (
    DataBuildContext,
    DatasetLocator,
    DatasetSpec,
    ScoringInputSpec,
    ScoringOutputSpec,
    SourceTableLike,
    SplitStrategy,
)
from mbt.events.models import LogMessage
from mbt.exceptions import AdapterError
from mbt_adapter_base.materialization import (
    SAMPLE_MODULUS,
    MaterializationError,
    MaterializedDatasetHandle,
    combine_snapshots,
    write_materialization_metadata,
)
from mbt_adapter_base.predictions import LocalPredictionStore


def _uri_to_path(uri: str) -> Path:
    if uri.startswith("file://"):
        return Path(uri.removeprefix("file://"))
    return Path(uri)


def _quote(identifier: str) -> str:
    return '"' + identifier.replace('"', '""') + '"'


def _sql_str(value: str) -> str:
    return "'" + value.replace("'", "''") + "'"


class LocalDatasetHandle(MaterializedDatasetHandle):
    """The shared materialization handle, tagged with the local adapter."""

    def __init__(self, directory: Path) -> None:
        super().__init__(directory, adapter="local")


@dataclass(frozen=True)
class _RelationSpec:
    """The FROM-clause shape shared by datasets and scoring inputs."""

    spine: str  # uid of the single source, or of the spine/label table
    features: list[str]  # feature table uids; empty for single-source
    join_columns: list[str]
    join: str  # "left" | "inner"


def _dataset_relation(spec: DatasetSpec) -> _RelationSpec:
    if spec.inputs is None:
        assert spec.source is not None
        return _RelationSpec(spine=spec.source, features=[], join_columns=[], join="left")
    return _RelationSpec(
        spine=spec.inputs.label,
        features=list(spec.inputs.features),
        join_columns=spec.inputs.join_columns,
        join=spec.inputs.join,
    )


def _scoring_relation(spec: ScoringInputSpec) -> _RelationSpec:
    if spec.inputs is None:
        assert spec.source is not None
        return _RelationSpec(spine=spec.source, features=[], join_columns=[], join="left")
    return _RelationSpec(
        spine=spec.inputs.spine,
        features=list(spec.inputs.features),
        join_columns=spec.inputs.join_columns,
        join=spec.inputs.join,
    )


class LocalDataAdapter:
    """Parquet-under-a-root DataAdapter (TSD §13.2)."""

    name = "local"

    def __init__(self, config: dict[str, Any] | None = None) -> None:
        config = config or {}
        self.root = Path(config.get("root", "."))

    # -- snapshots (TSD §8.3, ADR-11) ---------------------------------------

    def _matching_files(self, source: SourceTableLike) -> list[Path]:
        if source.path is None:
            raise AdapterError(
                f"source table {source.name!r} has no 'path'",
                hint="the local data adapter needs path-based sources (parquet globs)",
            )
        pattern = str(self.root / source.path)
        files = sorted(Path(p) for p in globlib.glob(pattern, recursive=True))
        files = [f for f in files if f.is_file()]
        if not files:
            raise AdapterError(
                f"no files match source {source.name!r} pattern {pattern!r}",
                hint="check profiles.yml data.config.root and the source path",
            )
        return files

    def snapshot_id(self, source: SourceTableLike, deep: bool = False) -> str:
        files = self._matching_files(source)
        digest = hashlib.sha256()
        for file in files:
            rel = file.relative_to(self.root) if file.is_relative_to(self.root) else file
            if deep:
                digest.update(str(rel).encode())
                digest.update(file.read_bytes())
            else:
                stat = file.stat()
                digest.update(f"{rel}|{stat.st_size}|{stat.st_mtime_ns}\n".encode())
        return "sha256:" + digest.hexdigest()

    def _verify_snapshot(self, ctx: DataBuildContext) -> None:
        """The data must still match the manifest pin: a drifted source under
        a pinned manifest is an error, not a silent rebuild (TSD §10.4)."""
        if ctx.node.snapshot_id is None:
            return
        current = combine_snapshots(
            {
                uid: self.snapshot_id(table, deep=ctx.deep_snapshot)
                for uid, table in ctx.source_tables.items()
            }
        )
        if current != ctx.node.snapshot_id:
            raise AdapterError(
                f"source data changed under the pinned manifest: snapshot "
                f"{current} != pinned {ctx.node.snapshot_id}",
                resource=ctx.node.unique_id,
                hint="recompile to pin the new snapshot, or restore the data",
            )

    # -- materialization (TSD §13.2, §10.4) ----------------------------------

    def build_dataset(self, spec: DatasetSpec, ctx: DataBuildContext) -> LocalDatasetHandle:
        self._verify_snapshot(ctx)
        output_dir = ctx.output_dir
        output_dir.mkdir(parents=True, exist_ok=True)
        for stale in output_dir.glob("*"):
            stale.unlink()

        con = duckdb.connect()
        try:
            self._create_base_view(
                con, _dataset_relation(spec), ctx, spec.filters, spec.sample_key_columns
            )
            if spec.split.strategy is SplitStrategy.TEMPORAL:
                written = self._write_temporal_splits(con, spec, ctx, output_dir)
            else:
                written = self._write_random_splits(con, spec, output_dir)
        except duckdb.Error as exc:
            raise AdapterError(
                f"dataset build failed in DuckDB: {exc}",
                resource=ctx.node.unique_id,
                hint=(
                    "check the dataset's filters, join keys, and split configuration; "
                    "column names must be unique across joined tables"
                ),
            ) from exc
        finally:
            con.close()

        for split, count in written.items():
            if count == 0:
                raise AdapterError(
                    f"split {split!r} materialized 0 rows",
                    resource=ctx.node.unique_id,
                    hint="check the split windows/fractions against the data's time range",
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
        return LocalDatasetHandle(output_dir)

    # -- SQL assembly ---------------------------------------------------------

    def _table_relation(self, ctx: DataBuildContext, uid: str) -> str:
        table = ctx.source_tables.get(uid)
        if table is None:
            raise AdapterError(
                f"dataset references source {uid!r} that is not in the manifest",
                resource=ctx.node.unique_id,
            )
        files = ", ".join(_sql_str(str(f)) for f in self._matching_files(table))
        return f"read_parquet([{files}])"

    def _base_relation(self, rel: _RelationSpec, ctx: DataBuildContext) -> str:
        """FROM clause: the single source, or spine + feature joins."""
        if not rel.features:
            return self._table_relation(ctx, rel.spine)
        using = ", ".join(_quote(c) for c in rel.join_columns)
        join_kind = "LEFT JOIN" if rel.join == "left" else "JOIN"
        sql = f"{self._table_relation(ctx, rel.spine)} AS mbt_spine"
        for i, feature_uid in enumerate(rel.features):
            sql += (
                f" {join_kind} {self._table_relation(ctx, feature_uid)} AS mbt_f{i} USING ({using})"
            )
        return sql

    def _digest_columns(
        self, con: "duckdb.DuckDBPyConnection", sample_keys: list[str], relation: str
    ) -> list[str]:
        """Columns hashed for sampling/splitting: the declared key, else all."""
        if sample_keys:
            return sample_keys
        described = con.execute(f"DESCRIBE SELECT * FROM {relation}").fetchall()
        return [row[0] for row in described]

    def _digest_sql(self, columns: list[str], salt: str = "") -> str:
        parts = ", ".join(f"COALESCE(CAST({_quote(c)} AS VARCHAR), '')" for c in columns)
        prefix = f"{_sql_str(salt + '|')}, " if salt else ""
        return f"md5_number(concat_ws('|', {prefix}{parts}))"

    def _create_base_view(
        self,
        con: "duckdb.DuckDBPyConnection",
        rel: _RelationSpec,
        ctx: DataBuildContext,
        filters: list[str],
        sample_keys: list[str],
    ) -> None:
        relation = self._base_relation(rel, ctx)
        where: list[str] = [f"({f})" for f in filters]
        sample_fraction = ctx.sample_fraction
        if not 0.0 < sample_fraction <= 1.0:
            raise AdapterError(
                f"sample_fraction must be in (0, 1], got {sample_fraction}",
                hint="set the 'sample_fraction' var in the target's vars",
            )
        if sample_fraction < 1.0:
            digest = self._digest_sql(self._digest_columns(con, sample_keys, relation))
            threshold = int(sample_fraction * SAMPLE_MODULUS)
            where.append(f"({digest} % {SAMPLE_MODULUS}) < {threshold}")
        where_sql = f" WHERE {' AND '.join(where)}" if where else ""
        con.execute(f"CREATE TEMP VIEW mbt_base AS SELECT * FROM {relation}{where_sql}")

    def _write_temporal_splits(
        self,
        con: "duckdb.DuckDBPyConnection",
        spec: DatasetSpec,
        ctx: DataBuildContext,
        output_dir: Path,
    ) -> dict[str, int]:
        assert spec.split.time_column is not None
        time_sql = f"CAST({_quote(spec.split.time_column)} AS TIMESTAMP)"
        written: dict[str, int] = {}
        for split, (start, end) in sorted(ctx.resolved_windows.items()):
            start_ts = _iso_to_sql_ts(start)
            end_ts = _iso_to_sql_ts(end)
            out = output_dir / f"{split}.parquet"
            con.execute(
                f"COPY (SELECT * FROM mbt_base WHERE {time_sql} >= TIMESTAMP '{start_ts}' "
                f"AND {time_sql} < TIMESTAMP '{end_ts}') TO '{out}' (FORMAT PARQUET)"
            )
            row = con.execute("SELECT count(*) FROM read_parquet(?)", [str(out)]).fetchone()
            written[split] = int(row[0]) if row else 0
        return written

    def _write_random_splits(
        self,
        con: "duckdb.DuckDBPyConnection",
        spec: DatasetSpec,
        output_dir: Path,
    ) -> dict[str, int]:
        fractions: dict[str, float] = {"train": float(spec.split.train)}
        if spec.split.validation is not None:
            fractions["validation"] = float(spec.split.validation)
        fractions["test"] = float(spec.split.test)

        seed = spec.split.seed or 0
        columns = self._digest_columns(con, spec.sample_key_columns, "mbt_base")
        rank_key = self._digest_sql(columns, salt=str(seed))
        partition = (
            f"PARTITION BY {_quote(spec.split.stratify_by)} " if spec.split.stratify_by else ""
        )
        rank = f"percent_rank() OVER ({partition}ORDER BY {rank_key})"

        bounds: list[tuple[str, float, float]] = []
        low = 0.0
        for split, fraction in fractions.items():
            bounds.append((split, low, low + fraction))
            low += fraction

        con.execute(f"CREATE TEMP VIEW mbt_ranked AS SELECT *, {rank} AS __mbt_rank FROM mbt_base")
        written: dict[str, int] = {}
        for split, lo, hi in bounds:
            out = output_dir / f"{split}.parquet"
            upper = f"__mbt_rank < {hi}" if hi < 1.0 else f"__mbt_rank <= {hi}"
            con.execute(
                f"COPY (SELECT * EXCLUDE (__mbt_rank) FROM mbt_ranked "
                f"WHERE __mbt_rank >= {lo} AND {upper}) TO '{out}' (FORMAT PARQUET)"
            )
            row = con.execute("SELECT count(*) FROM read_parquet(?)", [str(out)]).fetchone()
            written[split] = int(row[0]) if row else 0
        return written

    # -- scoring (contract 1.1, ADR-20/21) -------------------------------------

    def build_scoring_input(
        self, spec: ScoringInputSpec, ctx: DataBuildContext
    ) -> LocalDatasetHandle:
        """Materialize one unlabeled batch as a single ``score`` split.

        Zero rows is a warning, not an error: an empty nightly batch is
        legitimate (unlike an empty training split).
        """
        self._verify_snapshot(ctx)
        output_dir = ctx.output_dir
        output_dir.mkdir(parents=True, exist_ok=True)
        for stale in output_dir.glob("*"):
            stale.unlink()

        con = duckdb.connect()
        try:
            self._create_base_view(
                con, _scoring_relation(spec), ctx, spec.filters, spec.sample_key_columns
            )
            where = ""
            if spec.time_column is not None and "score" in ctx.resolved_windows:
                start, end = ctx.resolved_windows["score"]
                time_sql = f"CAST({_quote(spec.time_column)} AS TIMESTAMP)"
                where = (
                    f" WHERE {time_sql} >= TIMESTAMP '{_iso_to_sql_ts(start)}' "
                    f"AND {time_sql} < TIMESTAMP '{_iso_to_sql_ts(end)}'"
                )
            out = output_dir / "score.parquet"
            con.execute(f"COPY (SELECT * FROM mbt_base{where}) TO '{out}' (FORMAT PARQUET)")
            row = con.execute("SELECT count(*) FROM read_parquet(?)", [str(out)]).fetchone()
            count = int(row[0]) if row else 0
        except duckdb.Error as exc:
            raise AdapterError(
                f"scoring input build failed in DuckDB: {exc}",
                resource=ctx.node.unique_id,
                hint="check the input's filters, join keys, and window configuration",
            ) from exc
        finally:
            con.close()

        if count == 0:
            ctx.events.emit(
                LogMessage(
                    level="warn",
                    unique_id=ctx.node.unique_id,
                    message="scoring input materialized 0 rows; nothing to score",
                )
            )
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
        return LocalDatasetHandle(output_dir)

    def open_predictions(self, output: ScoringOutputSpec) -> LocalPredictionStore:
        return LocalPredictionStore(self.root / output.path)

    # -- reopening -----------------------------------------------------------

    def from_locator(self, locator: DatasetLocator) -> LocalDatasetHandle:
        try:
            handle = LocalDatasetHandle(_uri_to_path(locator.uri))
        except MaterializationError as exc:
            raise AdapterError(str(exc)) from exc
        if handle.snapshot_id != locator.snapshot_id:
            raise AdapterError(
                "dataset materialization snapshot mismatch: "
                f"{handle.snapshot_id} != {locator.snapshot_id}",
                hint="the data moved under a pinned manifest; recompile or restore the data",
            )
        return handle


def _iso_to_sql_ts(iso: str) -> str:
    """ISO-8601 (Z-suffixed) -> DuckDB TIMESTAMP literal text (UTC, naive)."""
    ts = datetime.fromisoformat(iso.replace("Z", "+00:00"))
    return ts.replace(tzinfo=None).isoformat(sep=" ")
