"""Local DuckDB data adapter: random splits, error paths, locator reopening."""

from datetime import datetime, timedelta
from pathlib import Path

import pyarrow as pa
import pyarrow.parquet as pq
import pytest
from misc_unit_helpers import RecordingSink, make_node

from mbt.adapters.local.data import LocalDataAdapter, _uri_to_path
from mbt.contracts import DatasetLocator, DatasetSpec, ManifestNode, ScoringInputSpec, SourceTable
from mbt.exceptions import AdapterError
from mbt.execute.runners import _BuildContext

ANCHOR = datetime(2026, 7, 1)
ROWS_UID = "source.demo.lakehouse.rows"


def _write_rows(root: Path, n: int = 60) -> None:
    table = pa.table(
        {
            "user_id": list(range(n)),
            "snapshot_date": [ANCHOR - timedelta(days=1 + i % 30) for i in range(n)],
            "is_active": [i % 5 != 0 for i in range(n)],
            "plan": [("basic", "pro")[i % 2] for i in range(n)],
            "churned": [i % 3 == 0 for i in range(n)],
        }
    )
    out = root / "data" / "rows"
    out.mkdir(parents=True, exist_ok=True)
    pq.write_table(table, out / "part-000.parquet")


def _tables() -> dict[str, SourceTable]:
    return {ROWS_UID: SourceTable(name="rows", path="data/rows/*.parquet")}


def _ctx(
    adapter: LocalDataAdapter,
    tables: dict[str, SourceTable],
    output_dir: Path,
    *,
    node: ManifestNode | None = None,
    windows: dict[str, tuple[str, str]] | None = None,
    sample_fraction: float = 1.0,
) -> _BuildContext:
    return _BuildContext(
        node=node or make_node("dataset.demo.churn_random"),
        source=next(iter(tables.values()), SourceTable(name="dummy", path="unused")),
        source_tables=tables,
        resolved_windows=windows or {},
        sample_fraction=sample_fraction,
        deep_snapshot=False,
        output_dir=output_dir,
        events=RecordingSink(),
    )


def _random_spec(**overrides: object) -> DatasetSpec:
    payload: dict = {
        "name": "churn_random",
        "source": ROWS_UID,
        "label": {"column": "churned"},
        "split": {
            "strategy": "random",
            "train": "0.6",
            "validation": "0.2",
            "test": "0.2",
            "seed": 7,
        },
    }
    payload.update(overrides)
    return DatasetSpec.model_validate(payload)


def test_uri_to_path_accepts_bare_paths_and_file_uris() -> None:
    assert _uri_to_path("/plain/dir") == Path("/plain/dir")
    assert _uri_to_path("file:///plain/dir") == Path("/plain/dir")


def test_source_without_a_path_is_rejected(tmp_path: Path) -> None:
    adapter = LocalDataAdapter({"root": str(tmp_path)})
    warehouse_table = SourceTable(name="rows", identifier="db.schema.rows")
    with pytest.raises(AdapterError, match="has no 'path'"):
        adapter.snapshot_id(warehouse_table)


def test_empty_glob_is_rejected(tmp_path: Path) -> None:
    adapter = LocalDataAdapter({"root": str(tmp_path)})
    table = SourceTable(name="rows", path="data/nothing/*.parquet")
    with pytest.raises(AdapterError, match="no files match source"):
        adapter.snapshot_id(table)


def test_random_split_build_partitions_all_rows(tmp_path: Path) -> None:
    _write_rows(tmp_path)
    adapter = LocalDataAdapter({"root": str(tmp_path)})
    output_dir = tmp_path / "target" / "datasets" / "churn_random" / "k1"
    output_dir.mkdir(parents=True)
    stale = output_dir / "stale.parquet"
    stale.write_text("junk from a previous build")

    # snapshot_id=None on the node: unpinned build skips snapshot verification
    handle = adapter.build_dataset(_random_spec(), _ctx(adapter, _tables(), output_dir))

    assert not stale.exists()  # stale files from a previous build are cleared
    assert handle.splits() == {"train", "validation", "test"}
    split_rows = {split: handle.read(split) for split in ("train", "validation", "test")}
    assert sum(t.num_rows for t in split_rows.values()) == 60
    all_ids = [i for t in split_rows.values() for i in t.column("user_id").to_pylist()]
    assert sorted(all_ids) == list(range(60))  # disjoint and exhaustive
    assert all(t.num_rows > 0 for t in split_rows.values())
    assert "__mbt_rank" not in split_rows["train"].column_names


def test_random_split_is_deterministic_for_a_seed(tmp_path: Path) -> None:
    _write_rows(tmp_path)
    adapter = LocalDataAdapter({"root": str(tmp_path)})
    memberships = []
    for key in ("k1", "k2"):
        output_dir = tmp_path / "target" / key
        handle = adapter.build_dataset(_random_spec(), _ctx(adapter, _tables(), output_dir))
        memberships.append(sorted(handle.read("train").column("user_id").to_pylist()))
    assert memberships[0] == memberships[1]


def test_duckdb_failure_becomes_an_adapter_error(tmp_path: Path) -> None:
    _write_rows(tmp_path)
    adapter = LocalDataAdapter({"root": str(tmp_path)})
    spec = _random_spec(filters=["no_such_column = 1"])
    with pytest.raises(AdapterError, match="dataset build failed in DuckDB"):
        adapter.build_dataset(spec, _ctx(adapter, _tables(), tmp_path / "target" / "bad"))


def test_empty_temporal_split_is_an_error(tmp_path: Path) -> None:
    _write_rows(tmp_path)
    adapter = LocalDataAdapter({"root": str(tmp_path)})
    spec = DatasetSpec.model_validate(
        {
            "name": "churn_temporal",
            "source": ROWS_UID,
            "label": {"column": "churned"},
            "split": {
                "strategy": "temporal",
                "time_column": "snapshot_date",
                "train": "-30d:-5d",
                "test": "-5d:now",
            },
        }
    )
    windows = {
        "train": ("2026-06-01T00:00:00Z", "2026-07-01T00:00:00Z"),
        "test": ("2030-01-01T00:00:00Z", "2030-02-01T00:00:00Z"),  # future: no rows
    }
    ctx = _ctx(adapter, _tables(), tmp_path / "target" / "empty", windows=windows)
    with pytest.raises(AdapterError, match="'test' materialized 0 rows"):
        adapter.build_dataset(spec, ctx)


def test_sample_fraction_must_be_a_valid_fraction(tmp_path: Path) -> None:
    _write_rows(tmp_path)
    adapter = LocalDataAdapter({"root": str(tmp_path)})
    ctx = _ctx(adapter, _tables(), tmp_path / "target" / "frac", sample_fraction=0.0)
    with pytest.raises(AdapterError, match=r"sample_fraction must be in \(0, 1\]"):
        adapter.build_dataset(_random_spec(), ctx)


def test_unknown_source_reference_is_an_adapter_error(tmp_path: Path) -> None:
    adapter = LocalDataAdapter({"root": str(tmp_path)})
    ctx = _ctx(adapter, {}, tmp_path / "target" / "missing")
    with pytest.raises(AdapterError, match="not in the manifest"):
        adapter._table_relation(ctx, "source.demo.lakehouse.ghost")


def test_scoring_input_duckdb_failure_becomes_an_adapter_error(tmp_path: Path) -> None:
    _write_rows(tmp_path)
    adapter = LocalDataAdapter({"root": str(tmp_path)})
    spec = ScoringInputSpec.model_validate({"source": ROWS_UID, "filters": ["no_such_column = 1"]})
    node = make_node("scoring.demo.batch")
    ctx = _ctx(adapter, _tables(), tmp_path / "target" / "score_bad", node=node)
    with pytest.raises(AdapterError, match="scoring input build failed in DuckDB"):
        adapter.build_scoring_input(spec, ctx)


def test_from_locator_rejects_an_incomplete_materialization(tmp_path: Path) -> None:
    empty = tmp_path / "not_a_materialization"
    empty.mkdir()
    adapter = LocalDataAdapter({"root": str(tmp_path)})
    locator = DatasetLocator(adapter="local", uri=f"file://{empty}", snapshot_id="sha256:x")
    with pytest.raises(AdapterError, match="no complete dataset materialization"):
        adapter.from_locator(locator)


def test_from_locator_verifies_the_pinned_snapshot(tmp_path: Path) -> None:
    _write_rows(tmp_path)
    adapter = LocalDataAdapter({"root": str(tmp_path)})
    tables = _tables()
    snapshot = adapter.snapshot_id(tables[ROWS_UID])
    node = make_node("dataset.demo.churn_random", snapshot_id=snapshot)
    output_dir = tmp_path / "target" / "pinned"
    handle = adapter.build_dataset(_random_spec(), _ctx(adapter, tables, output_dir, node=node))

    reopened = adapter.from_locator(handle.locator())
    assert reopened.snapshot_id == snapshot

    drifted = DatasetLocator(adapter="local", uri=f"file://{output_dir}", snapshot_id="sha256:old")
    with pytest.raises(AdapterError, match="snapshot mismatch"):
        adapter.from_locator(drifted)
