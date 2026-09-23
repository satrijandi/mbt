"""Spark DataAdapter: lakehouse dataset construction (parquet/Delta/catalog).

Sources resolve either by ``path`` (parquet or Delta directories - local or
object-store URIs) or by ``identifier`` (Spark catalog tables, e.g. Unity
Catalog / Hive metastore). A dataset reads one relation (ADR-29); filters,
deterministic key sampling, and split assignment push down as Spark SQL, and
each split lands as one parquet file in the shared mbt materialization, so
training jobs reopen datasets without a Spark session.

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
import threading
from datetime import datetime
from pathlib import Path
from typing import TYPE_CHECKING, Any, Literal

from mbt_adapter_base import (
    DatasetLocator,
    DatasetSpec,
    ScoringInputSpec,
    ScoringOutputSpec,
)
from mbt_adapter_base.errors import AdapterFailure
from mbt_adapter_base.materialization import (
    SAMPLE_MODULUS,
    MaterializationError,
    MaterializedDatasetHandle,
    bucket_ranges,
    build_dataset_materialization,
    build_scoring_materialization,
    combine_snapshots,
    split_fractions,
)
from mbt_adapter_base.protocols import DataBuildContext, SourceTableLike

if TYPE_CHECKING:
    from pyspark.sql import DataFrame, SparkSession

    from mbt_adapter_base.predictions import LocalPredictionStore


class SparkAdapterError(AdapterFailure):
    """Spark adapter failures with an actionable message.

    ``hint`` is a FIELD on the shared base rather than text flattened into the
    message, so core can render it the way it renders ``MbtError.hint`` (A-4).
    ``__str__`` still puts it on its own ``hint:`` line, so the wording a user
    or a test sees is unchanged.
    """


#: Which of a source table's two addresses this adapter reads it by.
SourceAddress = Literal["path", "identifier"]


def resolve_source_address(
    source: SourceTableLike, configured: SourceAddress | None = None
) -> SourceAddress:
    """Decide whether to read ``source`` by ``path`` or by ``identifier``.

    Spark is the only adapter that understands both (object-store/local
    directories AND catalog tables), so it is the only one that can be handed
    an ambiguous table. Declaring both is a legitimate thing to do - it is how
    one project serves a file plane and a warehouse plane off a single
    ``sources.yml`` - but which one *this* target reads is then a property of
    the target, not of the table, and only the operator knows it.

    So: a table declaring one address is read by that address, and a table
    declaring both is an error unless the adapter's ``source_address`` config
    says which. Guessing is not an option here. Until this function existed
    ``_read`` silently preferred ``identifier`` while ``snapshot_id`` hashed
    the ``path``, so an ambiguous table pinned one dataset and trained on
    another - and adding a warehouse identifier to a shared ``sources.yml``
    redirected the file planes at catalog tables that did not exist.
    """
    has_path = source.path is not None
    has_identifier = source.identifier is not None
    if has_path and has_identifier:
        if configured is None:
            raise SparkAdapterError(
                f"source table {source.name!r} declares both 'path' and 'identifier', "
                f"and the Spark adapter can read either",
                hint="set source_address: path (or: identifier) in this target's "
                "spark adapter config in profiles.yml to say which address it reads",
            )
        return configured
    if has_path:
        return "path"
    if has_identifier:
        return "identifier"
    raise SparkAdapterError(f"source table {source.name!r} needs 'path' or 'identifier'")


def _quote(column: str) -> str:
    return "`" + column.replace("`", "``") + "`"


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
        #: Tie-breaker for source tables that declare BOTH ``path`` and
        #: ``identifier`` (see ``resolve_source_address``). Unset is the right
        #: default: it means "no table may be ambiguous", which is only a
        #: constraint on projects that address the same table two ways.
        address: Any = config.get("source_address")
        if address is not None and address not in ("path", "identifier"):
            raise SparkAdapterError(
                f"source_address must be 'path' or 'identifier', got {address!r}"
            )
        self.source_address: SourceAddress | None = address
        from mbt_adapter_base.predictions import resolve_predictions_root

        #: Where staged prediction runs land (contract 1.1); joined with the
        #: scoring node's output.path. Unset, defaults to an ephemeral
        #: <tmpdir>/mbt-predictions (never the project dir), like Snowflake (F20).
        self.predictions_root: str = str(resolve_predictions_root(config.get("predictions_root")))
        self._session: SparkSession | None = None
        #: Serializes lazy session setup. The compiler's snapshot-pinning pool
        #: shares one adapter across threads (compile/compiler.py), so this is
        #: the same check-then-act shape that made the Snowflake adapter open
        #: one connection - and one SSO browser window - per source table.
        #: Here it is defensive rather than a live bug: pyspark's
        #: ``getOrCreate`` takes its own lock and returns the single active
        #: session, so a lost race only repeats builder work. Guarding it
        #: keeps that a pyspark implementation detail rather than something
        #: mbt's correctness leans on.
        self._session_lock = threading.Lock()

    def _spark(self) -> "SparkSession":
        # Read through a local so mypy cannot narrow the attribute to None and
        # declare the under-lock re-check unreachable (see _connect in the
        # Snowflake adapter for the same shape).
        session = self._session
        if session is not None:
            return session
        with self._session_lock:
            session = self._session
            if session is not None:
                return session
            from mbt_spark.session import get_session

            self._session = get_session(self.master, self.conf, app_name="mbt-spark-data")
            return self._session

    # -- source resolution -------------------------------------------------------

    def _resolve_path(self, source: SourceTableLike) -> str:
        assert source.path is not None
        if self.root and "://" not in source.path and not source.path.startswith("/"):
            return f"{self.root.rstrip('/')}/{source.path}"
        return source.path

    def _address(self, source: SourceTableLike) -> SourceAddress:
        return resolve_source_address(source, self.source_address)

    def _read(self, source: SourceTableLike) -> "DataFrame":
        # Resolve the address OUTSIDE the try: an ambiguous or address-less
        # table is a config error, and wrapping it as "cannot read source"
        # would bury the hint that names the fix.
        address = self._address(source)
        spark = self._spark()
        try:
            if address == "identifier":
                assert source.identifier is not None
                return spark.table(source.identifier)
            fmt = "delta" if source.format == "delta" else "parquet"
            return spark.read.format(fmt).load(self._resolve_path(source))
        except Exception as exc:
            raise SparkAdapterError(
                f"cannot read source {source.name!r}: {exc}",
                hint="check the path/identifier and, for Delta, that "
                "delta-spark is installed and configured",
            ) from exc

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
        source, and locally globbing it would always find nothing.

        It is also decided on the address this adapter actually READS the
        table by, not merely on ``path`` being present: pinning the local
        parquet files of a table that ``_read`` then serves from the catalog
        would pin a snapshot of data the run never touches."""
        digest = hashlib.sha256()
        resolved = self._resolve_path(source) if self._address(source) == "path" else None
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

    def verify_snapshot(self, ctx: DataBuildContext) -> None:
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
        return build_dataset_materialization(self, spec, ctx)

    # -- DatasetBuildEngine (A-1) --------------------------------------------

    def build_failure(
        self, message: str, *, ctx: DataBuildContext, hint: str | None = None
    ) -> Exception:
        return SparkAdapterError(message, hint, resource=ctx.node.unique_id)

    def write_dataset_splits(
        self, spec: DatasetSpec, ctx: DataBuildContext, output_dir: Path
    ) -> dict[str, int]:
        """Read the relation, filter, sample, and write one parquet per split."""
        self._spark()  # session kept alive for the adapter's lifetime
        base = self._base_frame(spec, ctx)
        for clause in spec.filters:
            base = base.filter(clause)
        if ctx.sample_fraction < 1.0:
            threshold = int(ctx.sample_fraction * SAMPLE_MODULUS)
            base = base.filter(f"{key_hash_sql(spec.sample_key_columns)} < {threshold}")
        return self._write_splits(base, spec, ctx, output_dir)

    def _base_frame(self, spec: DatasetSpec, ctx: DataBuildContext) -> "DataFrame":
        """The dataset's one relation, whatever built it (ADR-29)."""
        return self._read(ctx.source_tables[spec.source])

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

        # sample_key is required and validated non-empty on the spec (ADR-29).
        bucket = key_hash_sql(spec.sample_key_columns, salt=str(spec.split.seed or 0))
        # Shared bucket edges (A-1): the three backends agree by construction
        # rather than by three implementations mirroring each other (F19).
        for split, lo, hi in bucket_ranges(split_fractions(spec.split)):
            frame = base.filter(f"{bucket} >= {lo} AND {bucket} < {hi}")
            written[split] = self._write_one(frame, output_dir / f"{split}.parquet")
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

                # Typed like a non-empty split, so a reader concatenating or
                # scoring it sees the panel's real schema rather than all-string
                # columns (an empty after-test split is routine, ADR-30).
                pq.write_table(_empty_table(frame.schema), out)
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
        """The scoring batch's one relation: the serving twin of the training
        panel, unlabeled by design (ADR-20/29)."""
        return self._read(ctx.source_tables[spec.source])

    def build_scoring_input(
        self, spec: ScoringInputSpec, ctx: DataBuildContext
    ) -> MaterializedDatasetHandle:
        return build_scoring_materialization(self, spec, ctx)

    def write_scoring_batch(self, spec: ScoringInputSpec, ctx: DataBuildContext, out: Path) -> int:
        """One relation, filters, key sampling, the ``score`` window - no label."""
        self._spark()  # session kept alive for the adapter's lifetime
        base = self._scoring_frame(spec, ctx)
        for clause in spec.filters:
            base = base.filter(clause)
        if ctx.sample_fraction < 1.0:
            keys = spec.sample_key_columns
            if not keys:
                raise self.build_failure(
                    "sampling a Spark scoring input needs a stable row identity",
                    ctx=ctx,
                    hint="declare input.sample_key (the entity id column(s)) in the "
                    "scoring spec, or set sample_fraction: 1.0 for this target",
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
        return self._write_one(base, out)

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


def _empty_table(spark_schema: Any) -> Any:
    """A zero-row arrow table carrying the Spark frame's real column types.

    Falls back to string per column only for a Spark type arrow has no
    mapping for, so the column set is always preserved.
    """
    import pyarrow as pa
    from pyspark.sql.pandas.types import to_arrow_type

    fields = []
    for field in spark_schema:
        try:
            arrow_type = to_arrow_type(field.dataType)
        except Exception:  # an unmappable Spark type: keep the column, as string
            arrow_type = pa.string()
        fields.append(pa.field(field.name, arrow_type))
    schema = pa.schema(fields)
    return pa.Table.from_pylist([], schema=schema)
