"""Spark DataAdapter: lakehouse dataset construction (parquet/Delta/catalog).

Sources resolve either by ``path`` (parquet or Delta directories - local or
object-store URIs) or by ``identifier`` (Spark catalog tables, e.g. Unity
Catalog / Hive metastore). Joins, filters, deterministic key sampling, and
split assignment all push down as Spark SQL; each split lands as one
parquet file in the shared mbt materialization, so training jobs reopen
datasets without a Spark session.

Sampling and random splits use the canonical cross-adapter digest (F19): the
unsigned LOWER 64 BITS of the md5 of the '|'-joined key -
``conv(substring(md5(key), 17, 16), 16, 10)`` here, ``MD5_NUMBER_LOWER64`` on
Snowflake, and the same hex-slice cast on local DuckDB - bucketed with
``% SAMPLE_MODULUS``. The same fraction keeps the same rows, smaller fractions
are subsets of larger ones, and the same key lands in the same sample/split
bucket on every backend, so a model validated locally trains on the same
partition in the warehouse.
"""

import glob as globlib
import hashlib
import shutil
import tempfile
from datetime import datetime
from pathlib import Path
from typing import TYPE_CHECKING, Any

from mbt_adapter_base import (
    DatasetLocator,
    DatasetSpec,
    ScoringInputSpec,
    ScoringOutputSpec,
    parse_time_offset,
)
from mbt_adapter_base.materialization import (
    SAMPLE_MODULUS,
    MaterializationError,
    MaterializedDatasetHandle,
    combine_snapshots,
    write_materialization_metadata,
)
from mbt_adapter_base.protocols import DataBuildContext, SourceTableLike

if TYPE_CHECKING:
    from pyspark.sql import DataFrame, SparkSession

    from mbt_adapter_base.predictions import LocalPredictionStore


class SparkAdapterError(RuntimeError):
    """Spark adapter failures with an actionable message."""

    def __init__(self, message: str, hint: str | None = None) -> None:
        if hint:
            message = f"{message}\n  hint: {hint}"
        super().__init__(message)


def _quote(column: str) -> str:
    return "`" + column.replace("`", "``") + "`"


#: time_offset units -> SQL interval keywords (calendar month included).
_INTERVAL_UNITS = {"mo": "MONTH", "d": "DAY", "w": "WEEK", "h": "HOUR"}


def _interval_sql(count: int, unit: str) -> str:
    """``(1, "mo")`` -> ``+ INTERVAL 1 MONTH`` (sign as the operator)."""
    operator = "-" if count < 0 else "+"
    return f"{operator} INTERVAL {abs(count)} {_INTERVAL_UNITS[unit]}"


def key_hash_sql(key_columns: list[str], salt: str = "") -> str:
    """Deterministic 0..SAMPLE_MODULUS-1 bucket from a stable row key
    (Spark SQL; md5 is stable across Spark versions, unlike hash()).

    The digest is the canonical cross-adapter one (F19): the unsigned lower 64
    bits of the md5 - the last 16 hex chars, which ``conv`` parses as an
    unsigned 64-bit value (its decimal string exceeds BIGINT for high bits, so
    it goes through DECIMAL(20,0)) - matching Snowflake's
    ``MD5_NUMBER_LOWER64`` and the local DuckDB hex-slice cast exactly.
    """
    parts = ", ".join(f"COALESCE(CAST({_quote(c)} AS STRING), '')" for c in key_columns)
    if salt:
        safe = salt.replace("'", "''")
        parts = f"'{safe}', {parts}"
    digest = f"conv(substring(md5(concat_ws('|', {parts})), 17, 16), 16, 10)"
    return f"CAST(pmod(CAST({digest} AS DECIMAL(20, 0)), {SAMPLE_MODULUS}) AS BIGINT)"


class SparkDataAdapter:
    """DataAdapter over Spark-readable tables."""

    name = "spark"
    #: Source formats this adapter can read; the compiler rejects a referenced
    #: source declaring any other format before anything runs (F23). Spark is
    #: the one adapter that also reads Delta tables (`_read`).
    supported_source_formats = frozenset({"parquet", "delta"})

    def __init__(self, config: dict[str, Any] | None = None) -> None:
        config = config or {}
        self.master: str = str(config.get("master", "local[*]"))
        self.conf: dict[str, Any] = dict(config.get("conf", {}))
        self.root: str | None = config.get("root")  # optional prefix for path sources
        from mbt_adapter_base.predictions import resolve_predictions_root

        #: Where staged prediction runs land (contract 1.1); joined with the
        #: scoring node's output.path. Unset, defaults to an ephemeral
        #: <tmpdir>/mbt-predictions (never the project dir), like Snowflake (F20).
        self.predictions_root: str = str(resolve_predictions_root(config.get("predictions_root")))
        self._session: SparkSession | None = None

    def _spark(self) -> "SparkSession":
        if self._session is None:
            from mbt_spark.session import get_session

            self._session = get_session(self.master, self.conf, app_name="mbt-spark-data")
        return self._session

    # -- source resolution -------------------------------------------------------

    def _resolve_path(self, source: SourceTableLike) -> str:
        assert source.path is not None
        if self.root and "://" not in source.path and not source.path.startswith("/"):
            return f"{self.root.rstrip('/')}/{source.path}"
        return source.path

    def _read(self, source: SourceTableLike) -> "DataFrame":
        spark = self._spark()
        try:
            if source.identifier is not None:
                return spark.table(source.identifier)
            if source.path is not None:
                fmt = "delta" if source.format == "delta" else "parquet"
                return spark.read.format(fmt).load(self._resolve_path(source))
        except Exception as exc:
            raise SparkAdapterError(
                f"cannot read source {source.name!r}: {exc}",
                hint="check the path/identifier and, for Delta, that "
                "delta-spark is installed and configured",
            ) from exc
        raise SparkAdapterError(f"source table {source.name!r} needs 'path' or 'identifier'")

    # -- snapshots ------------------------------------------------------------------

    def snapshot_id(self, source: SourceTableLike, deep: bool = False) -> str:
        """Local paths: ``(path, size, mtime)`` listing - cheap, no Spark
        session - or, with ``deep``, a content hash of each file so a fresh
        checkout (which rewrites mtimes) does not flag everything as modified
        (ADR-11), mirroring the local adapter's deep snapshot.
        URIs/catalog tables: hash of the table's input file listing, which is
        already mtime-independent for immutable (Delta/Iceberg/committed
        parquet) files, so deep and shallow agree there.

        The branch is decided on the RESOLVED path: a relative table path
        under a URI root (e.g. root s3://lake + path t/*.parquet) is a URI
        source, and locally globbing it would always find nothing."""
        digest = hashlib.sha256()
        resolved = self._resolve_path(source) if source.path is not None else None
        if resolved is not None and "://" not in resolved:
            pattern = resolved
            root = Path(pattern.split("*", 1)[0]).parent if "*" in pattern else Path(pattern)
            files = sorted(
                p
                for p in (
                    Path(f)
                    for f in globlib.glob(
                        pattern + ("/**" if root.is_dir() else ""), recursive=True
                    )
                )
                if p.is_file()
            ) or sorted(p for p in root.rglob("*") if p.is_file())
            if not files:
                raise SparkAdapterError(f"no files under source path {pattern!r}")
            for file in files:
                if deep:
                    digest.update(f"{file}\n".encode())
                    digest.update(file.read_bytes())  # content, not mtime
                else:
                    stat = file.stat()
                    digest.update(f"{file}|{stat.st_size}|{stat.st_mtime_ns}\n".encode())
        else:
            for uri in sorted(self._read(source).inputFiles()):
                digest.update(uri.encode())
                digest.update(b"\n")
        return "sha256:" + digest.hexdigest()

    def _verify_snapshot(self, ctx: DataBuildContext) -> None:
        if ctx.node.snapshot_id is None:
            return
        current = combine_snapshots(
            {uid: self.snapshot_id(table) for uid, table in ctx.source_tables.items()}
        )
        if current != ctx.node.snapshot_id:
            raise SparkAdapterError(
                f"source data changed under the pinned manifest: snapshot "
                f"{current} != pinned {ctx.node.snapshot_id}",
                hint="recompile to pin the new snapshot",
            )

    # -- materialization ----------------------------------------------------------------

    def build_dataset(self, spec: DatasetSpec, ctx: DataBuildContext) -> MaterializedDatasetHandle:
        self._verify_snapshot(ctx)
        spark = self._spark()
        output_dir = ctx.output_dir
        output_dir.mkdir(parents=True, exist_ok=True)
        for stale in output_dir.glob("*"):
            stale.unlink()

        base = self._base_frame(spec, ctx)
        for clause in spec.filters:
            base = base.filter(clause)
        if not 0.0 < ctx.sample_fraction <= 1.0:
            raise SparkAdapterError(f"sample_fraction must be in (0, 1], got {ctx.sample_fraction}")
        if ctx.sample_fraction < 1.0:
            keys = spec.sample_key_columns
            if not keys:
                raise SparkAdapterError(
                    "sampling on Spark needs a stable row identity",
                    hint="declare sample_key: [<id columns>] on the dataset "
                    "(or use the multi-table inputs form)",
                )
            threshold = int(ctx.sample_fraction * SAMPLE_MODULUS)
            base = base.filter(f"{key_hash_sql(keys)} < {threshold}")

        written = self._write_splits(base, spec, ctx, output_dir)
        for split, count in written.items():
            if count == 0:
                raise SparkAdapterError(
                    f"split {split!r} materialized 0 rows",
                    hint="check the split windows/fractions and filters",
                )
        # Positive-path row counts on the bus (a plain string the EventSink
        # wraps in a LogMessage); mirrors the local and snowflake adapters.
        ctx.events.emit(
            f"dataset {ctx.node.unique_id}: materialized {sum(written.values())} rows: "
            + ", ".join(f"{split}={count}" for split, count in sorted(written.items()))
        )
        coverage: dict[str, int] | None = None
        if spec.inputs is not None and spec.inputs.population is not None:
            # Label-join coverage (F21): spine rows vs rows surviving the inner
            # label join, counted before filters/sampling/windows.
            tables = {uid: self._read(table) for uid, table in ctx.source_tables.items()}
            coverage = {
                "spine_rows": self._spine_frame(spec, tables).count(),
                "matched_rows": self._base_frame(spec, ctx).count(),
            }
            if coverage["spine_rows"] > 0:
                fraction = coverage["matched_rows"] / coverage["spine_rows"]
                ctx.events.emit(
                    f"dataset {ctx.node.unique_id}: label join matched "
                    f"{coverage['matched_rows']} of {coverage['spine_rows']} "
                    f"spine rows ({fraction:.1%})"
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
            label_join_coverage=coverage,
        )
        _ = spark  # session kept alive for the adapter's lifetime
        return MaterializedDatasetHandle(output_dir, adapter=self.name)

    def _spine_frame(self, spec: DatasetSpec, tables: dict[str, "DataFrame"]) -> "DataFrame":
        """The spine + feature joins, before any label join (shared by
        ``_base_frame`` and the label-join coverage counts, F21)."""
        assert spec.inputs is not None
        frame = tables[spec.inputs.spine]
        how = "left" if spec.inputs.join == "left" else "inner"
        for feature_uid, using in spec.inputs.feature_entries:
            frame = frame.join(tables[feature_uid], on=using, how=how)
        return frame

    def _base_frame(self, spec: DatasetSpec, ctx: DataBuildContext) -> "DataFrame":
        tables = {uid: self._read(table) for uid, table in ctx.source_tables.items()}
        if spec.inputs is None:
            assert spec.source is not None
            return tables[spec.source]
        frame = self._spine_frame(spec, tables)
        if spec.inputs.population is None:
            return frame
        # The label joins the population spine last (always inner - an example
        # without an observed outcome is not a training example, ADR-22): its
        # join columns are renamed, matched with an expression join so the
        # time_offset can shift the spine's time column, then dropped.
        from pyspark.sql import functions as F

        label = tables[spec.inputs.label_source]
        renames = {c: f"__mbt_lbl{i}" for i, c in enumerate(spec.inputs.label_join_columns)}
        for column, alias in renames.items():
            label = label.withColumnRenamed(column, alias)
        offset = spec.inputs.label_time_offset
        conditions = []
        for column, alias in renames.items():
            if offset is not None and column == spec.split.time_column:
                count, unit = parse_time_offset(offset)
                conditions.append(
                    f"CAST({_quote(alias)} AS TIMESTAMP) = "
                    f"CAST({_quote(column)} AS TIMESTAMP) {_interval_sql(count, unit)}"
                )
            else:
                conditions.append(f"{_quote(alias)} = {_quote(column)}")
        frame = frame.join(label, on=F.expr(" AND ".join(conditions)), how="inner")
        return frame.drop(*renames.values())

    # -- source-level checks (F2/F21) ----------------------------------------------------

    def count_source_duplicates(self, source: SourceTableLike, columns: list[str]) -> int:
        """Distinct COMPOSITE keys appearing more than once in the raw source
        (pre-join): the 1:1 join-cardinality contract behind the ``unique``
        check's ``source:`` mode (F2). Null keys are ignored, as in dbt."""
        frame = self._read(source).dropna(subset=list(columns))
        return frame.groupBy(*columns).count().filter("count > 1").count()

    def read_source_distinct(self, source: SourceTableLike, column: str) -> Any:
        """DISTINCT non-null values of one raw source column as a
        single-column ``value`` arrow table - the parent side of the
        ``relationships`` check (F2/F21)."""
        import pyarrow as pa

        rows = (
            self._read(source)
            .select(column)
            .dropna()
            .distinct()
            .withColumnRenamed(column, "value")
            .collect()
        )
        return pa.table({"value": [row["value"] for row in rows]})

    def _write_splits(
        self,
        base: "DataFrame",
        spec: DatasetSpec,
        ctx: DataBuildContext,
        output_dir: Path,
    ) -> dict[str, int]:
        written: dict[str, int] = {}
        if spec.split.strategy.value == "temporal":
            assert spec.split.time_column is not None
            time_sql = f"CAST({_quote(spec.split.time_column)} AS TIMESTAMP)"
            for split, (start, end) in sorted(ctx.resolved_windows.items()):
                frame = base.filter(
                    f"{time_sql} >= to_timestamp('{_iso_to_ts(start)}') AND "
                    f"{time_sql} < to_timestamp('{_iso_to_ts(end)}')"
                )
                written[split] = self._write_one(frame, output_dir / f"{split}.parquet")
            return written

        keys = spec.sample_key_columns
        if not keys:
            raise SparkAdapterError(
                "a random split on Spark needs 'sample_key' (or inputs.join_key)"
            )
        bucket = key_hash_sql(keys, salt=str(spec.split.seed or 0))
        fractions: dict[str, float] = {"train": float(spec.split.train)}
        if spec.split.validation is not None:
            fractions["validation"] = float(spec.split.validation)
        fractions["test"] = float(spec.split.test)
        low = 0.0
        entries = list(fractions.items())
        for index, (split, fraction) in enumerate(entries):
            lo = int(low * SAMPLE_MODULUS)
            # the final split's upper bound is pinned to the modulus, mirroring
            # the local and snowflake adapters, so no bucket can fall through a
            # float-accumulation gap (F19)
            hi = (
                SAMPLE_MODULUS
                if index == len(entries) - 1
                else int((low + fraction) * SAMPLE_MODULUS)
            )
            frame = base.filter(f"{bucket} >= {lo} AND {bucket} < {hi}")
            written[split] = self._write_one(frame, output_dir / f"{split}.parquet")
            low += fraction
        return written

    def _write_one(self, frame: "DataFrame", out: Path) -> int:
        """Write one split as a single parquet file (materializations are
        sampled/windowed slices sized for single-node training)."""
        staging = Path(tempfile.mkdtemp(prefix="mbt-spark-split-"))
        try:
            frame.coalesce(1).write.mode("overwrite").parquet(str(staging / "data"))
            parts = list((staging / "data").glob("part-*.parquet"))
            if not parts:  # empty result set still writes metadata-only output
                import pyarrow.parquet as pq

                spark_schema = frame.schema
                import pyarrow as pa

                empty = pa.table({f.name: pa.array([], type=pa.string()) for f in spark_schema})
                pq.write_table(empty, out)
                return 0
            shutil.move(str(parts[0]), out)
            import pyarrow.parquet as pq

            return int(pq.ParquetFile(out).metadata.num_rows)
        finally:
            shutil.rmtree(staging, ignore_errors=True)

    # -- reopening -------------------------------------------------------------------------

    def from_locator(self, locator: DatasetLocator) -> MaterializedDatasetHandle:
        """Reopen a materialization; needs no Spark session."""
        path = Path(locator.uri.removeprefix("file://"))
        try:
            handle = MaterializedDatasetHandle(path, adapter=self.name)
        except MaterializationError as exc:
            raise SparkAdapterError(str(exc)) from exc
        if handle.snapshot_id != locator.snapshot_id:
            raise SparkAdapterError(
                "dataset materialization snapshot mismatch: "
                f"{handle.snapshot_id} != {locator.snapshot_id}"
            )
        return handle

    # -- batch scoring (contract 1.1, ADR-20/21) -------------------------------------

    def _scoring_frame(self, spec: ScoringInputSpec, ctx: DataBuildContext) -> "DataFrame":
        """Spine + feature joins for a scoring batch - the training relation
        (``_base_frame``) minus the label (scoring inputs are unlabeled)."""
        tables = {uid: self._read(table) for uid, table in ctx.source_tables.items()}
        if spec.inputs is None:
            assert spec.source is not None
            return tables[spec.source]
        frame = tables[spec.inputs.spine]
        how = "left" if spec.inputs.join == "left" else "inner"
        for feature_uid, using in spec.inputs.feature_entries:
            frame = frame.join(tables[feature_uid], on=using, how=how)
        return frame

    def build_scoring_input(
        self, spec: ScoringInputSpec, ctx: DataBuildContext
    ) -> MaterializedDatasetHandle:
        """Materialize one unlabeled Spark batch as a single ``score`` split.

        Mirrors ``build_dataset`` (spine + feature joins, filters, key sampling,
        the ``score`` window) but writes one ``score.parquet`` with no label.
        Zero rows is a warning, not an error - an empty nightly batch is
        legitimate (unlike a training split; ADR-20). No snapshot verification:
        the scoring input (and the monitor's arriving labels, read through this
        same path) is expected to change every run, so a pinned manifest scores
        the live data instead of hard-failing on drift the way a dataset does
        (R2-10)."""
        self._spark()  # session kept alive for the adapter's lifetime
        output_dir = ctx.output_dir
        output_dir.mkdir(parents=True, exist_ok=True)
        for stale in output_dir.glob("*"):
            stale.unlink()

        base = self._scoring_frame(spec, ctx)
        for clause in spec.filters:
            base = base.filter(clause)
        if not 0.0 < ctx.sample_fraction <= 1.0:
            raise SparkAdapterError(f"sample_fraction must be in (0, 1], got {ctx.sample_fraction}")
        if ctx.sample_fraction < 1.0:
            keys = spec.sample_key_columns
            if not keys:
                raise SparkAdapterError(
                    "sampling a Spark scoring input needs a stable row identity",
                    hint="declare sample_key on the input (or use the inputs form, "
                    "whose join_key is used)",
                )
            threshold = int(ctx.sample_fraction * SAMPLE_MODULUS)
            base = base.filter(f"{key_hash_sql(keys)} < {threshold}")

        if spec.time_column is not None:
            window = ctx.resolved_windows.get("score")
            if window is not None:
                start, end = window
                time_sql = f"CAST({_quote(spec.time_column)} AS TIMESTAMP)"
                base = base.filter(
                    f"{time_sql} >= to_timestamp('{_iso_to_ts(start)}') AND "
                    f"{time_sql} < to_timestamp('{_iso_to_ts(end)}')"
                )

        count = self._write_one(base, output_dir / "score.parquet")
        # A plain string the EventSink wraps in a LogMessage (adapters cannot
        # import core event models); mirrors the local and snowflake adapters.
        if count == 0:
            ctx.events.emit(
                f"scoring input {ctx.node.unique_id}: materialized 0 rows; nothing to score"
            )
        else:
            ctx.events.emit(
                f"scoring input {ctx.node.unique_id}: materialized {count} rows to score"
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
        return MaterializedDatasetHandle(output_dir, adapter=self.name)

    def open_predictions(self, output: ScoringOutputSpec) -> "LocalPredictionStore":
        """Prediction store for a Spark scoring pipeline.

        v1 stages prediction runs as parquet under ``predictions_root`` using the
        shared local layout (ADR-21's sanctioned reuse: "warehouse adapters can
        reuse it for staged exports"), the same stance as the Snowflake adapter;
        a lakehouse-table-backed store is the ADR-23 v2 design, gated on live
        verification. ``predictions_root`` (adapter config; unset, an ephemeral
        ``<tmpdir>/mbt-predictions``) is joined with the scoring node's
        ``output.path``."""
        from mbt_adapter_base.predictions import LocalPredictionStore

        return LocalPredictionStore(Path(self.predictions_root) / output.path)


def _iso_to_ts(iso: str) -> str:
    ts = datetime.fromisoformat(iso.replace("Z", "+00:00"))
    return ts.replace(tzinfo=None).isoformat(sep=" ")
