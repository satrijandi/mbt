"""Spark DataAdapter: lakehouse dataset construction (parquet/Delta/catalog).

Sources resolve either by ``path`` (parquet or Delta directories - local or
object-store URIs) or by ``identifier`` (Spark catalog tables, e.g. Unity
Catalog / Hive metastore). Joins, filters, deterministic key sampling, and
split assignment all push down as Spark SQL; each split lands as one
parquet file in the shared mbt materialization, so training jobs reopen
datasets without a Spark session.

Sampling uses the same md5-threshold formula as the local and Snowflake
adapters (``conv(substring(md5(key),1,15),16,10) % 1e6 < fraction*1e6``):
same fraction -> same rows; smaller fractions are subsets of larger ones.
"""

import glob as globlib
import hashlib
import shutil
import tempfile
from datetime import datetime
from pathlib import Path
from typing import TYPE_CHECKING, Any

from mbt_adapter_base import DatasetLocator, DatasetSpec
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


class SparkAdapterError(RuntimeError):
    """Spark adapter failures with an actionable message."""

    def __init__(self, message: str, hint: str | None = None) -> None:
        if hint:
            message = f"{message}\n  hint: {hint}"
        super().__init__(message)


def _quote(column: str) -> str:
    return "`" + column.replace("`", "``") + "`"


def key_hash_sql(key_columns: list[str], salt: str = "") -> str:
    """Deterministic 0..SAMPLE_MODULUS-1 bucket from a stable row key
    (Spark SQL; md5 is stable across Spark versions, unlike hash())."""
    parts = ", ".join(f"COALESCE(CAST({_quote(c)} AS STRING), '')" for c in key_columns)
    if salt:
        safe = salt.replace("'", "''")
        parts = f"'{safe}', {parts}"
    digest = f"conv(substring(md5(concat_ws('|', {parts})), 1, 15), 16, 10)"
    return f"pmod(CAST({digest} AS BIGINT), {SAMPLE_MODULUS})"


class SparkDataAdapter:
    """DataAdapter over Spark-readable tables."""

    name = "spark"

    def __init__(self, config: dict[str, Any] | None = None) -> None:
        config = config or {}
        self.master: str = str(config.get("master", "local[*]"))
        self.conf: dict[str, Any] = dict(config.get("conf", {}))
        self.root: str | None = config.get("root")  # optional prefix for path sources
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
        """Local paths: (path, size, mtime) listing - cheap, no Spark session.
        URIs/catalog tables: hash of the table's input file listing."""
        if deep:
            raise SparkAdapterError(
                "--deep-snapshot is not supported by the spark adapter yet",
                hint="rely on Delta/Iceberg immutable files, or file listings",
            )
        digest = hashlib.sha256()
        if source.path is not None and "://" not in source.path:
            pattern = self._resolve_path(source)
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
        _ = spark  # session kept alive for the adapter's lifetime
        return MaterializedDatasetHandle(output_dir, adapter=self.name)

    def _base_frame(self, spec: DatasetSpec, ctx: DataBuildContext) -> "DataFrame":
        tables = {uid: self._read(table) for uid, table in ctx.source_tables.items()}
        if spec.inputs is None:
            assert spec.source is not None
            return tables[spec.source]
        frame = tables[spec.inputs.label]
        how = "left" if spec.inputs.join == "left" else "inner"
        for feature_uid in spec.inputs.features:
            frame = frame.join(tables[feature_uid], on=spec.inputs.join_columns, how=how)
        return frame

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
        for split, fraction in fractions.items():
            lo, hi = int(low * SAMPLE_MODULUS), int((low + fraction) * SAMPLE_MODULUS)
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


def _iso_to_ts(iso: str) -> str:
    ts = datetime.fromisoformat(iso.replace("Z", "+00:00"))
    return ts.replace(tzinfo=None).isoformat(sep=" ")
