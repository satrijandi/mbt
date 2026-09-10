"""Local Parquet DataAdapter via DuckDB (TSD §13.2, FR-ADPT-04).

Sources resolve to Parquet globs under ``config.root``. ``build_dataset``
runs one DuckDB query: read the source table(s) - joining feature tables
onto the label table for multi-table ``inputs`` datasets - then filters,
deterministic hash sampling, and split assignment (resolved temporal
windows, or seeded hash split), writing one Parquet file per split.

Sampling and random-split reproducibility: rows hash to a bucket via the
canonical cross-adapter digest - the unsigned lower 64 bits of
``md5('|'-joined key)`` modulo 1_000_000 - over the dataset's ``sample_key``
(or join key). The same fraction always keeps the same rows, smaller
fractions are subsets of larger ones, and the same key lands in the same
bucket on Snowflake and Spark too (F19). Without a key the digest falls
back to hashing every column - correct, but slow on wide tables.
"""

import glob as globlib
import hashlib
import os
from datetime import datetime
from pathlib import Path
from typing import TYPE_CHECKING, Any

import duckdb

if TYPE_CHECKING:
    import pyarrow as pa

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


def _connect_duckdb(output_dir: Path, parallelism: int = 1) -> "duckdb.DuckDBPyConnection":
    """A DuckDB connection scoped to mbt's own build budget (F22).

    ``temp_directory`` is the build's own (absolute) output dir, not DuckDB's
    default relative ``.tmp``: because the coordinator has chdir'd to the project
    dir, that default resolves to ``<project>/.tmp``, so a build that spills under
    memory pressure litters the project root and can fill a constrained CI disk.
    Both callers close the connection in a ``finally``, so DuckDB reclaims the
    spill files on close.

    When ``parallelism > 1`` (concurrent in-process builds under ``--threads``),
    cores and DuckDB's 80%-of-RAM default are DIVIDED by it, so N parallel builds
    do not each claim all cores and 80% of RAM and oversubscribe the box. A lone
    build keeps DuckDB's full-machine defaults. (The RAM budget needs POSIX
    ``sysconf``; on a non-POSIX host only the thread budget is applied.)
    """
    config: dict[str, Any] = {"temp_directory": str(output_dir.resolve())}
    if parallelism > 1:
        cores = os.cpu_count() or 1
        config["threads"] = str(max(1, cores // parallelism))
        if hasattr(os, "sysconf") and "SC_PHYS_PAGES" in os.sysconf_names:
            total_ram = os.sysconf("SC_PHYS_PAGES") * os.sysconf("SC_PAGE_SIZE")
            budget_mib = max(64, int(0.8 * total_ram / parallelism / 2**20))
            config["memory_limit"] = f"{budget_mib}MiB"
    return duckdb.connect(config=config)


class LocalDataAdapter:
    """Parquet-under-a-root DataAdapter (TSD §13.2)."""

    name = "local"
    #: Source formats this adapter can read; the compiler rejects a referenced
    #: source declaring any other format before anything runs (F23).
    supported_source_formats = frozenset({"parquet"})

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

        con = _connect_duckdb(output_dir, ctx.build_parallelism)
        try:
            self._create_base_view(con, spec.source, ctx, spec.filters, spec.sample_key_columns)
            if spec.split.strategy is SplitStrategy.TEMPORAL:
                written = self._write_temporal_splits(con, spec, ctx, output_dir)
            else:
                written = self._write_random_splits(con, spec, ctx, output_dir)
        except duckdb.Error as exc:
            raise AdapterError(
                f"dataset build failed in DuckDB: {exc}",
                resource=ctx.node.unique_id,
                hint=(
                    "check the dataset's filters and split configuration against "
                    "the relation's columns"
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

        ctx.events.emit(
            LogMessage(
                unique_id=ctx.node.unique_id,
                message=(
                    f"materialized {sum(written.values())} rows: "
                    + ", ".join(f"{split}={count}" for split, count in sorted(written.items()))
                ),
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
        return LocalDatasetHandle(output_dir)

    # -- SQL assembly ---------------------------------------------------------

    def _table_relation(self, ctx: DataBuildContext, uid: str) -> str:
        table = ctx.source_tables.get(uid)
        if table is None:
            raise AdapterError(
                f"dataset references source {uid!r} that is not in the manifest",
                resource=ctx.node.unique_id,
            )
        return self._source_relation(table)

    # -- source-level checks (F2/F21) ------------------------------------------

    def _source_relation(self, source: SourceTableLike) -> str:
        files = ", ".join(_sql_str(str(f)) for f in self._matching_files(source))
        return f"read_parquet([{files}])"

    def count_source_duplicates(self, source: SourceTableLike, columns: list[str]) -> int:
        """Distinct COMPOSITE keys appearing more than once in the raw source
        (pre-join): the 1:1 join-cardinality contract behind the ``unique``
        check's ``source:`` mode (F2). Null keys are ignored, as in dbt."""
        cols = ", ".join(_quote(c) for c in columns)
        not_null = " AND ".join(f"{_quote(c)} IS NOT NULL" for c in columns)
        con = duckdb.connect()
        try:
            row = con.execute(
                f"SELECT count(*) FROM (SELECT 1 FROM {self._source_relation(source)} "
                f"WHERE {not_null} GROUP BY {cols} HAVING count(*) > 1)"
            ).fetchone()
            return int(row[0]) if row else 0
        finally:
            con.close()

    def read_source_distinct(self, source: SourceTableLike, column: str) -> "pa.Table":
        """DISTINCT non-null values of one raw source column, as a
        single-column table named ``value`` - the parent side of the
        ``relationships`` check (F2/F21)."""
        con = duckdb.connect()
        try:
            return con.sql(
                f"SELECT DISTINCT {_quote(column)} AS value FROM "
                f"{self._source_relation(source)} WHERE {_quote(column)} IS NOT NULL"
            ).to_arrow_table()
        finally:
            con.close()

    def _digest_columns(
        self,
        con: "duckdb.DuckDBPyConnection",
        sample_keys: list[str],
        relation: str,
        *,
        ctx: DataBuildContext,
        purpose: str,
    ) -> list[str]:
        """Columns hashed for sampling/splitting: the declared key. No fallback.

        ADR-16 framed the all-columns fallback as the slow path. It was also the
        UNSTABLE one, and that is the more important half: the column list is
        the hash preimage, so adding one column to a source re-buckets every
        row. Measured against DuckDB with this module's own digest expression -
        ten rows, one column added, an 80/20 boundary - three of ten rows
        changed side; with a sample_key, none did (test_local_data_unit).
        Metric history in the tracker stops being comparable across any schema
        evolution, and rows previously held out silently enter training. (Within
        one run the champion is re-evaluated on the challenger's split per
        ADR-9, so the promotion decision itself stays fair; it is the
        longitudinal record that degrades.)

        It was also local-only: Snowflake and Spark have always raised on a
        keyless sample or random split rather than inventing a row identity, so
        a spec that worked here failed there. The parser now requires
        ``sample_key`` outright (ADR-29) and this is the runtime backstop, so
        all three planes agree. ``ctx`` stays in the signature for the resource
        name in the error.
        """
        if sample_keys:
            return sample_keys
        raise AdapterError(
            f"no 'sample_key' declared, so {purpose} would hash every column - "
            "adding or removing any column then re-buckets every row and moves "
            "rows across the train/test boundary",
            resource=ctx.node.unique_id,
            hint="declare 'sample_key: <id column(s)>' on the dataset for a "
            "split that survives schema evolution and ports across backends",
        )

    def _digest_sql(self, columns: list[str], salt: str = "") -> str:
        """The canonical cross-adapter row hash (F19): the unsigned LOWER 64
        BITS of the md5 of a '|'-joined preimage (the salt first when present,
        then each column COALESCEd to ''). Snowflake computes the identical
        value natively (``MD5_NUMBER_LOWER64``) and Spark via
        ``conv(substring(md5(...), 17, 16), 16, 10)``, so the same key lands in
        the same sample/split bucket on every backend. DuckDB's own
        ``md5_number*`` functions use a different byte interpretation, hence
        the explicit hex-slice cast here."""
        parts = ", ".join(f"COALESCE(CAST({_quote(c)} AS VARCHAR), '')" for c in columns)
        if salt:
            parts = f"{_sql_str(salt)}, {parts}"
        return f"('0x' || substring(md5(concat_ws('|', {parts})), 17, 16))::UBIGINT"

    def _create_base_view(
        self,
        con: "duckdb.DuckDBPyConnection",
        source_uid: str,
        ctx: DataBuildContext,
        filters: list[str],
        sample_keys: list[str],
    ) -> None:
        relation = self._table_relation(ctx, source_uid)
        where: list[str] = [f"({f})" for f in filters]
        sample_fraction = ctx.sample_fraction
        if not 0.0 < sample_fraction <= 1.0:
            raise AdapterError(
                f"sample_fraction must be in (0, 1], got {sample_fraction}",
                hint="set the 'sample_fraction' var in the target's vars",
            )
        if sample_fraction < 1.0:
            digest = self._digest_sql(
                self._digest_columns(con, sample_keys, relation, ctx=ctx, purpose="sampling")
            )
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
        ctx: DataBuildContext,
        output_dir: Path,
    ) -> dict[str, int]:
        """Random splits as stable hash-bucket ranges, exactly as the warehouse
        adapters compute them (F19): membership is a pure function of the key,
        so it neither shifts as the dataset grows nor differs across backends.
        ``stratify_by`` is the one exception - exact per-stratum fractions need
        ranking, which stays size-dependent (documented in spec-reference)."""
        fractions: dict[str, float] = {"train": float(spec.split.train)}
        if spec.split.validation is not None:
            fractions["validation"] = float(spec.split.validation)
        fractions["test"] = float(spec.split.test)

        seed = spec.split.seed or 0
        columns = self._digest_columns(
            con, spec.sample_key_columns, "mbt_base", ctx=ctx, purpose="the random split"
        )
        written: dict[str, int] = {}

        if spec.split.stratify_by:
            rank_key = self._digest_sql(columns, salt=str(seed))
            partition = f"PARTITION BY {_quote(spec.split.stratify_by)} "
            rank = f"percent_rank() OVER ({partition}ORDER BY {rank_key})"
            bounds: list[tuple[str, float, float]] = []
            low = 0.0
            for split, fraction in fractions.items():
                bounds.append((split, low, low + fraction))
                low += fraction
            con.execute(
                f"CREATE TEMP VIEW mbt_ranked AS SELECT *, {rank} AS __mbt_rank FROM mbt_base"
            )
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

        # The boundary arithmetic mirrors snowflake's split_queries verbatim so
        # the two backends compute identical bucket edges; the final split's
        # upper bound is pinned to the modulus so no bucket can fall through a
        # float-accumulation gap.
        bucket = f"({self._digest_sql(columns, salt=str(seed))} % {SAMPLE_MODULUS})"
        low = 0.0
        entries = list(fractions.items())
        for index, (split, fraction) in enumerate(entries):
            lo = int(low * SAMPLE_MODULUS)
            hi = (
                SAMPLE_MODULUS
                if index == len(entries) - 1
                else int((low + fraction) * SAMPLE_MODULUS)
            )
            out = output_dir / f"{split}.parquet"
            con.execute(
                f"COPY (SELECT * FROM mbt_base WHERE {bucket} >= {lo} AND {bucket} < {hi}) "
                f"TO '{out}' (FORMAT PARQUET)"
            )
            row = con.execute("SELECT count(*) FROM read_parquet(?)", [str(out)]).fetchone()
            written[split] = int(row[0]) if row else 0
            low += fraction
        return written

    # -- scoring (contract 1.1, ADR-20/21) -------------------------------------

    def build_scoring_input(
        self, spec: ScoringInputSpec, ctx: DataBuildContext
    ) -> LocalDatasetHandle:
        """Materialize one unlabeled batch as a single ``score`` split.

        Zero rows is a warning, not an error: an empty nightly batch is
        legitimate (unlike an empty training split).

        No snapshot verification: a scoring input (and the arriving labels
        ``mbt monitor`` reads through this same path) is expected to change
        every run, so a pinned manifest must score the live data, not hard-fail
        on drift the way a dataset does (R2-10). The node's ``snapshot_id`` is
        still recorded in the materialization metadata for provenance.
        """
        output_dir = ctx.output_dir
        output_dir.mkdir(parents=True, exist_ok=True)
        for stale in output_dir.glob("*"):
            stale.unlink()

        con = _connect_duckdb(output_dir, ctx.build_parallelism)
        try:
            self._create_base_view(con, spec.source, ctx, spec.filters, spec.sample_key_columns)
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
        else:
            ctx.events.emit(
                LogMessage(
                    unique_id=ctx.node.unique_id,
                    message=f"scoring input materialized {count} rows to score",
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
