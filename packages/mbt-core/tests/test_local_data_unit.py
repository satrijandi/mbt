"""Local DuckDB data adapter: random splits, error paths, locator reopening."""

from datetime import datetime, timedelta
from pathlib import Path

import pyarrow as pa
import pyarrow.parquet as pq
import pytest
from misc_unit_helpers import RecordingSink, make_node

from mbt.adapters.local.data import LocalDataAdapter, _connect_duckdb, _uri_to_path
from mbt.contracts import DatasetLocator, DatasetSpec, ManifestNode, ScoringInputSpec, SourceTable
from mbt.exceptions import AdapterError
from mbt.execute.runners import BuildContext

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
    events: RecordingSink | None = None,
) -> BuildContext:
    return BuildContext(
        node=node or make_node("dataset.demo.churn_random"),
        source=next(iter(tables.values()), SourceTable(name="dummy", path="unused")),
        source_tables=tables,
        resolved_windows=windows or {},
        sample_fraction=sample_fraction,
        deep_snapshot=False,
        output_dir=output_dir,
        events=events or RecordingSink(),
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


def test_build_emits_materialized_row_counts(tmp_path: Path) -> None:
    """A successful dataset build reports per-split row counts on the bus, so
    the positive path is no longer silent (only 0-row raised before)."""
    _write_rows(tmp_path)
    adapter = LocalDataAdapter({"root": str(tmp_path)})
    sink = RecordingSink()
    output_dir = tmp_path / "target" / "datasets" / "churn_random" / "k1"
    adapter.build_dataset(_random_spec(), _ctx(adapter, _tables(), output_dir, events=sink))
    materialized = [
        m for e in sink.events if (m := getattr(e, "message", "")).startswith("materialized ")
    ]
    assert len(materialized) == 1
    assert "train=" in materialized[0] and "test=" in materialized[0]


def test_random_split_is_deterministic_for_a_seed(tmp_path: Path) -> None:
    _write_rows(tmp_path)
    adapter = LocalDataAdapter({"root": str(tmp_path)})
    memberships = []
    for key in ("k1", "k2"):
        output_dir = tmp_path / "target" / key
        handle = adapter.build_dataset(_random_spec(), _ctx(adapter, _tables(), output_dir))
        memberships.append(sorted(handle.read("train").column("user_id").to_pylist()))
    assert memberships[0] == memberships[1]


def _reference_bucket(preimage: str) -> int:
    """The canonical cross-adapter bucket (F19): unsigned lower 64 bits of the
    md5 of the '|'-joined preimage, mod SAMPLE_MODULUS. Snowflake's
    MD5_NUMBER_LOWER64 and Spark's conv(substring(md5, 17, 16)) compute the
    same value; each adapter's tests pin their SQL to this one reference."""
    import hashlib

    return int(hashlib.md5(preimage.encode()).hexdigest()[16:32], 16) % 1_000_000


def test_random_split_membership_matches_the_python_reference(tmp_path: Path) -> None:
    """Stable hash buckets (F19): each row's split is a pure function of its
    key - int(md5 low-64 of 'seed|key') % 1e6 against the fraction edges - so
    membership neither shifts as data grows nor differs across backends."""
    _write_rows(tmp_path)
    adapter = LocalDataAdapter({"root": str(tmp_path)})
    output_dir = tmp_path / "target" / "parity"
    spec = _random_spec(sample_key=["user_id"])
    handle = adapter.build_dataset(spec, _ctx(adapter, _tables(), output_dir))

    def expected_split(user_id: int) -> str:
        bucket = _reference_bucket(f"7|{user_id}")  # salt = split.seed
        if bucket < 600_000:
            return "train"
        if bucket < 800_000:
            return "validation"
        return "test"

    for split in ("train", "validation", "test"):
        for user_id in handle.read(split).column("user_id").to_pylist():
            assert expected_split(user_id) == split


def test_random_split_membership_is_stable_as_the_dataset_grows(tmp_path: Path) -> None:
    """The percent_rank ranking this replaced was relative to the whole
    dataset, so near-boundary rows flipped splits between retrains as data
    grew; hash buckets must not (F19)."""
    adapter = LocalDataAdapter({"root": str(tmp_path)})
    spec = _random_spec(sample_key=["user_id"])
    memberships: list[dict[str, str]] = []
    for key, n in (("small", 60), ("grown", 120)):
        _write_rows(tmp_path, n=n)
        handle = adapter.build_dataset(spec, _ctx(adapter, _tables(), tmp_path / "target" / key))
        assignment = {
            str(uid): split
            for split in ("train", "validation", "test")
            for uid in handle.read(split).column("user_id").to_pylist()
        }
        memberships.append(assignment)
    small, grown = memberships
    assert all(grown[uid] == split for uid, split in small.items())


def test_sampling_matches_the_python_reference_and_nests(tmp_path: Path) -> None:
    """sample_fraction keeps rows whose unsalted bucket clears the threshold
    (F19): reference-checked, and smaller fractions are subsets of larger."""
    _write_rows(tmp_path)
    adapter = LocalDataAdapter({"root": str(tmp_path)})
    spec = _random_spec(sample_key=["user_id"])
    kept: dict[float, set[int]] = {}
    for fraction in (0.5, 0.25):
        output_dir = tmp_path / "target" / f"sample-{fraction}"
        ctx = _ctx(adapter, _tables(), output_dir, sample_fraction=fraction)
        handle = adapter.build_dataset(spec, ctx)
        kept[fraction] = {
            uid
            for split in ("train", "validation", "test")
            for uid in handle.read(split).column("user_id").to_pylist()
        }
        expected = {
            uid for uid in range(60) if _reference_bucket(str(uid)) < int(fraction * 1_000_000)
        }
        assert kept[fraction] == expected
    assert kept[0.25] <= kept[0.5]


def test_stratified_random_split_keeps_exact_per_stratum_fractions(tmp_path: Path) -> None:
    """stratify_by keeps the ranking path: exact fractions per stratum (the
    documented size-dependent exception to stable hash buckets, F19)."""
    _write_rows(tmp_path)
    adapter = LocalDataAdapter({"root": str(tmp_path)})
    spec = _random_spec(
        sample_key=["user_id"],
        split={
            "strategy": "random",
            "train": "0.6",
            "validation": "0.2",
            "test": "0.2",
            "seed": 7,
            "stratify_by": "plan",
        },
    )
    handle = adapter.build_dataset(spec, _ctx(adapter, _tables(), tmp_path / "target" / "strat"))
    train = handle.read("train")
    assert "__mbt_rank" not in train.column_names
    plans = train.column("plan").to_pylist()
    # 30 rows per plan; 0.6 of each stratum = 18 exactly (percent_rank is exact)
    assert plans.count("basic") == 18 and plans.count("pro") == 18


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


def test_build_dataset_rejects_source_drift_under_a_pinned_manifest(tmp_path: Path) -> None:
    """A dataset (training data) must NOT drift under a pinned manifest -
    reproducibility. R2-10 relaxes this only for expected-mutable scoring
    inputs; datasets stay strict."""
    _write_rows(tmp_path)
    adapter = LocalDataAdapter({"root": str(tmp_path)})
    tables = _tables()
    node = make_node("dataset.demo.churn_random", snapshot_id="sha256:pinned-elsewhere")
    ctx = _ctx(adapter, tables, tmp_path / "target" / "drifted", node=node)
    with pytest.raises(AdapterError, match="source data changed under the pinned manifest"):
        adapter.build_dataset(_random_spec(), ctx)


def test_duckdb_spills_to_the_absolute_build_dir_not_a_stray_cwd_tmp(tmp_path: Path) -> None:
    """F22: a build's DuckDB connection must spill to its own absolute output dir,
    not DuckDB's default relative './.tmp' - which, under the coordinator's chdir
    into the project dir, litters <project>/.tmp and can fill a constrained CI
    disk. Rooting temp_directory at the resolved output dir keeps spills inside
    the managed build artifact."""
    out = tmp_path / "target" / "datasets" / "node" / "key"
    out.mkdir(parents=True)
    con = _connect_duckdb(out)
    try:
        setting = con.execute("SELECT current_setting('temp_directory')").fetchone()[0]
    finally:
        con.close()
    assert Path(setting).is_absolute()  # NOT the default relative '.tmp'
    assert Path(setting) == out.resolve()


def test_duckdb_divides_cores_and_memory_across_parallel_builds(tmp_path: Path) -> None:
    """F22: with N concurrent in-process builds (--threads=N), each DuckDB
    connection gets cores // N and ~80%/N of RAM, so N parallel large builds do
    not each grab all cores and 80% of RAM and oversubscribe the box; a lone
    build keeps the full-machine defaults."""
    import os

    out = tmp_path / "target" / "node"
    out.mkdir(parents=True)
    solo = _connect_duckdb(out, parallelism=1)
    shared = _connect_duckdb(out, parallelism=4)
    try:
        solo_threads = int(solo.execute("SELECT current_setting('threads')").fetchone()[0])
        shared_threads = int(shared.execute("SELECT current_setting('threads')").fetchone()[0])
        solo_mem = solo.execute("SELECT current_setting('memory_limit')").fetchone()[0]
        shared_mem = shared.execute("SELECT current_setting('memory_limit')").fetchone()[0]
    finally:
        solo.close()
        shared.close()
    # 4 concurrent builds each get a quarter of the cores (at least 1), never more
    # than a lone build's full-machine default
    assert shared_threads == max(1, (os.cpu_count() or 1) // 4)
    assert shared_threads <= solo_threads
    assert shared_mem != solo_mem  # the RAM budget is divided, not the 80% default
