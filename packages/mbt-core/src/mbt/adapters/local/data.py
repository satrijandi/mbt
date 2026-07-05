"""Local Parquet DataAdapter via DuckDB (TSD §13.2, FR-ADPT-04).

Sources resolve to Parquet globs under ``config.root``. ``build_dataset``
runs one DuckDB query: read source -> filters -> deterministic hash sampling
-> split assignment (resolved temporal windows, or seeded hash split) ->
one Parquet file per split under the materialization directory.
"""

import glob as globlib
import hashlib
import json
from datetime import datetime
from pathlib import Path
from typing import Any

import duckdb
import pyarrow as pa

from mbt.contracts import (
    DataBuildContext,
    DatasetLocator,
    DatasetProfile,
    DatasetSpec,
    SourceTableLike,
    SplitStrategy,
)
from mbt.exceptions import AdapterError

_METADATA_FILE = "materialization.json"
_PROFILE_FILE = "profile.json"
_SUCCESS_FILE = "_SUCCESS"
_SAMPLE_MODULUS = 1_000_000


def _uri_to_path(uri: str) -> Path:
    if uri.startswith("file://"):
        return Path(uri.removeprefix("file://"))
    return Path(uri)


def _path_to_uri(path: Path) -> str:
    return f"file://{path.resolve()}"


def _quote(identifier: str) -> str:
    return '"' + identifier.replace('"', '""') + '"'


class LocalDatasetHandle:
    """A materialized dataset directory: one parquet file per split."""

    def __init__(self, directory: Path) -> None:
        self.directory = directory
        metadata_path = directory / _METADATA_FILE
        if not metadata_path.is_file() or not (directory / _SUCCESS_FILE).is_file():
            raise AdapterError(
                f"no complete dataset materialization at {directory}",
                hint="the dataset build may have failed; re-run without a warm cache",
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
        return self._metadata.get("time_column")

    def splits(self) -> set[str]:
        return {p.stem for p in self.directory.glob("*.parquet")}

    def split_path(self, split: str) -> Path:
        path = self.directory / f"{split}.parquet"
        if not path.is_file():
            raise AdapterError(
                f"split {split!r} is not materialized at {self.directory}",
                hint=f"available splits: {', '.join(sorted(self.splits()))}",
            )
        return path

    def read(self, split: str, columns: list[str] | None = None) -> pa.Table:
        path = self.split_path(split)
        con = duckdb.connect()
        try:
            if columns:
                column_sql = ", ".join(_quote(c) for c in columns)
            else:
                column_sql = "*"
            return con.execute(
                f"SELECT {column_sql} FROM read_parquet(?)", [str(path)]
            ).to_arrow_table()
        finally:
            con.close()

    def profile(self) -> DatasetProfile:
        if self._profile is not None:
            return self._profile
        profile_path = self.directory / _PROFILE_FILE
        if profile_path.is_file():
            self._profile = DatasetProfile.model_validate_json(profile_path.read_text())
            return self._profile
        self._profile = self._compute_profile()
        profile_path.write_text(self._profile.model_dump_json(indent=2))
        return self._profile

    def _compute_profile(self) -> DatasetProfile:
        con = duckdb.connect()
        try:
            n_rows: dict[str, int] = {}
            for split in sorted(self.splits()):
                (count,) = con.execute(
                    "SELECT count(*) FROM read_parquet(?)", [str(self.split_path(split))]
                ).fetchone()
                n_rows[split] = int(count)
            schema = con.execute(
                "SELECT * FROM read_parquet(?) LIMIT 0", [str(self.split_path("train"))]
            ).to_arrow_table().schema
            columns = {field.name: str(field.type) for field in schema}

            label_balance: dict[str, float] | None = None
            label = self.label_column
            if label in columns and n_rows.get("train", 0) > 0:
                rows = con.execute(
                    f"SELECT CAST({_quote(label)} AS VARCHAR) AS cls, count(*) AS n "
                    f"FROM read_parquet(?) GROUP BY 1 ORDER BY 1",
                    [str(self.split_path("train"))],
                ).fetchall()
                total = sum(int(n) for _, n in rows)
                if total:
                    label_balance = {str(cls): int(n) / total for cls, n in rows}

            time_range: tuple[str, str] | None = None
            time_column = self.time_column
            if time_column and time_column in columns:
                paths = [str(self.split_path(s)) for s in sorted(self.splits())]
                low, high = con.execute(
                    f"SELECT CAST(min({_quote(time_column)}) AS VARCHAR), "
                    f"CAST(max({_quote(time_column)}) AS VARCHAR) FROM read_parquet(?)",
                    [paths],
                ).fetchone()
                if low is not None:
                    time_range = (str(low), str(high))
            return DatasetProfile(
                n_rows=n_rows,
                columns=columns,
                label_column=label,
                label_balance=label_balance,
                time_range=time_range,
            )
        finally:
            con.close()

    def locator(self) -> DatasetLocator:
        return DatasetLocator(
            adapter="local", uri=_path_to_uri(self.directory), snapshot_id=self.snapshot_id
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

    # -- materialization (TSD §13.2, §10.4) ----------------------------------

    def build_dataset(self, spec: DatasetSpec, ctx: DataBuildContext) -> LocalDatasetHandle:
        # The data must still match the manifest pin: a drifted source under a
        # pinned manifest is an error, not a silent rebuild (TSD §10.4 step 3).
        if ctx.node.snapshot_id is not None:
            current = self.snapshot_id(ctx.source, deep=ctx.deep_snapshot)
            if current != ctx.node.snapshot_id:
                raise AdapterError(
                    f"source data changed under the pinned manifest: snapshot "
                    f"{current} != pinned {ctx.node.snapshot_id}",
                    resource=ctx.node.unique_id,
                    hint="recompile to pin the new snapshot, or restore the data",
                )
        files = [str(f) for f in self._matching_files(ctx.source)]
        output_dir = ctx.output_dir
        output_dir.mkdir(parents=True, exist_ok=True)
        for stale in output_dir.glob("*"):
            stale.unlink()

        con = duckdb.connect()
        try:
            self._create_base_view(con, files, spec, ctx.sample_fraction)
            if spec.split.strategy is SplitStrategy.TEMPORAL:
                written = self._write_temporal_splits(con, spec, ctx, output_dir)
            else:
                written = self._write_random_splits(con, spec, output_dir)
        except duckdb.Error as exc:
            raise AdapterError(
                f"dataset build failed in DuckDB: {exc}",
                resource=ctx.node.unique_id,
                hint="check the dataset's filters and split configuration",
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

        metadata = {
            "snapshot_id": ctx.node.snapshot_id,
            "dataset": spec.name,
            "label_column": spec.label.column,
            "time_column": spec.split.time_column,
            "windows": {k: list(v) for k, v in ctx.resolved_windows.items()},
            "sample_fraction": ctx.sample_fraction,
            "row_counts": written,
        }
        (output_dir / _METADATA_FILE).write_text(json.dumps(metadata, indent=2, sort_keys=True))
        (output_dir / _SUCCESS_FILE).write_text("")
        return LocalDatasetHandle(output_dir)

    def _create_base_view(
        self,
        con: "duckdb.DuckDBPyConnection",
        files: list[str],
        spec: DatasetSpec,
        sample_fraction: float,
    ) -> None:
        file_list = ", ".join("'" + f.replace("'", "''") + "'" for f in files)
        where: list[str] = [f"({f})" for f in spec.filters]
        if not 0.0 < sample_fraction <= 1.0:
            raise AdapterError(
                f"sample_fraction must be in (0, 1], got {sample_fraction}",
                hint="set the 'sample_fraction' var in the target's vars",
            )
        if sample_fraction < 1.0:
            digest = self._row_digest_sql(con, file_list)
            threshold = int(sample_fraction * _SAMPLE_MODULUS)
            where.append(f"(md5_number({digest}) % {_SAMPLE_MODULUS}) < {threshold}")
        where_sql = f" WHERE {' AND '.join(where)}" if where else ""
        con.execute(
            f"CREATE TEMP VIEW mbt_base AS SELECT * FROM read_parquet([{file_list}]){where_sql}"
        )

    def _row_digest_sql(self, con: "duckdb.DuckDBPyConnection", file_list: str) -> str:
        """A stable per-row digest over all columns (deterministic sampling)."""
        described = con.execute(
            f"DESCRIBE SELECT * FROM read_parquet([{file_list}])"
        ).fetchall()
        parts = ", ".join(
            f"COALESCE(CAST({_quote(row[0])} AS VARCHAR), '')" for row in described
        )
        return f"concat_ws('|', {parts})"

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
            (count,) = con.execute(
                "SELECT count(*) FROM read_parquet(?)", [str(out)]
            ).fetchone()
            written[split] = int(count)
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

        described = con.execute("DESCRIBE SELECT * FROM mbt_base").fetchall()
        parts = ", ".join(
            f"COALESCE(CAST({_quote(row[0])} AS VARCHAR), '')" for row in described
        )
        digest = f"concat_ws('|', {parts})"
        seed = spec.split.seed or 0
        rank_key = f"md5_number(concat('{seed}|', {digest}))"
        partition = (
            f"PARTITION BY {_quote(spec.split.stratify_by)} " if spec.split.stratify_by else ""
        )
        rank = f"percent_rank() OVER ({partition}ORDER BY {rank_key})"

        bounds: list[tuple[str, float, float]] = []
        low = 0.0
        for split, fraction in fractions.items():
            bounds.append((split, low, low + fraction))
            low += fraction

        con.execute(
            f"CREATE TEMP VIEW mbt_ranked AS SELECT *, {rank} AS __mbt_rank FROM mbt_base"
        )
        written: dict[str, int] = {}
        for split, lo, hi in bounds:
            out = output_dir / f"{split}.parquet"
            upper = f"__mbt_rank < {hi}" if hi < 1.0 else f"__mbt_rank <= {hi}"
            con.execute(
                f"COPY (SELECT * EXCLUDE (__mbt_rank) FROM mbt_ranked "
                f"WHERE __mbt_rank >= {lo} AND {upper}) TO '{out}' (FORMAT PARQUET)"
            )
            (count,) = con.execute(
                "SELECT count(*) FROM read_parquet(?)", [str(out)]
            ).fetchone()
            written[split] = int(count)
        return written

    # -- reopening -----------------------------------------------------------

    def from_locator(self, locator: DatasetLocator) -> LocalDatasetHandle:
        handle = LocalDatasetHandle(_uri_to_path(locator.uri))
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
