"""Population-spine datasets (ADR-22): per-table join keys, the inner label
join with a calendar time offset, and the local adapter's SQL assembly."""

from datetime import datetime
from pathlib import Path

import pyarrow as pa
import pyarrow.parquet as pq
import pytest
from core_helpers import TEST_ANCHOR, write
from test_execution import invoke

from mbt.adapters.registry import AdapterRegistry
from mbt.contracts import DatasetSpec, parse_time_offset

DS = "dataset.demo.wide_churn"
MODEL = "model.demo.wide_churn_model"

#: Month starts covering the year before the test anchor.
MONTHS = [datetime(TEST_ANCHOR.year - 1, m, 1) for m in range(TEST_ANCHOR.month, 13)] + [
    datetime(TEST_ANCHOR.year, m, 1) for m in range(1, TEST_ANCHOR.month + 1)
]


def _next_month(when: datetime) -> datetime:
    return datetime(when.year + when.month // 12, when.month % 12 + 1, 1)


def _write_tables(project_dir: Path, customers: int = 40) -> None:
    population = {"customer_id": [], "safe_id": [], "snapshot_date": []}
    labels = {"customer_id": [], "snapshot_date": [], "churned": []}
    demo = {"customer_id": [], "snapshot_date": [], "age_years": []}
    txn = {"safe_id": [], "snapshot_date": [], "txn_total": []}
    for month_idx, when in enumerate(MONTHS):
        for cid in range(customers):
            population["customer_id"].append(cid)
            population["safe_id"].append(f"sf-{cid:04d}")
            population["snapshot_date"].append(when)
            demo["customer_id"].append(cid)
            demo["snapshot_date"].append(when)
            demo["age_years"].append(20 + (cid * 7) % 50)
            txn["safe_id"].append(f"sf-{cid:04d}")
            txn["snapshot_date"].append(when)
            txn["txn_total"].append(round((cid * 13.7 + month_idx) % 500, 2))
            # The outcome for snapshot m lives at m+1; its value encodes BOTH
            # sides of the join so a mis-aligned offset is detectable.
            # Labels for the newest month are deliberately not written yet
            # (immature outcomes) - those population rows must drop (inner).
            if month_idx < len(MONTHS) - 1:
                labels["customer_id"].append(cid)
                labels["snapshot_date"].append(_next_month(when))
                labels["churned"].append((cid + month_idx) % 2)
    for name, columns in (
        ("population", population),
        ("labels", labels),
        ("demo_features", demo),
        ("txn_features", txn),
    ):
        out = project_dir / "data" / name
        out.mkdir(parents=True, exist_ok=True)
        pq.write_table(pa.table(columns), out / "part-000.parquet")


@pytest.fixture()
def population_project(demo_project: Path) -> Path:
    _write_tables(demo_project)
    write(
        demo_project / "sources.yml",
        """
        sources:
          - name: lakehouse
            tables:
              - name: subscribers
                path: data/subscribers/*.parquet
              - name: population
                path: data/population/*.parquet
              - name: churn_labels
                path: data/labels/*.parquet
              - name: demo_features
                path: data/demo_features/*.parquet
              - name: txn_features
                path: data/txn_features/*.parquet
        """,
    )
    write(
        demo_project / "datasets/wide_churn.yml",
        """
        datasets:
          - name: wide_churn
            inputs:
              population: source('lakehouse', 'population')
              label:
                source: source('lakehouse', 'churn_labels')
                using: [customer_id, snapshot_date]
                time_offset: "1mo"
              features:
                - source: source('lakehouse', 'demo_features')
                  using: [customer_id, snapshot_date]
                - source: source('lakehouse', 'txn_features')
                  using: [safe_id, snapshot_date]
            sample_key: customer_id
            label:
              column: churned
            split:
              strategy: temporal
              time_column: snapshot_date
              # explicit ISO date ranges (DS-defined train/test start+end);
              # the test window deliberately extends PAST the newest month so
              # only the inner label join can be what drops it.
              train: "2025-07-01:2026-04-01"
              test: "2026-04-01:2026-07-02"
        """,
    )
    write(
        demo_project / "models/wide_churn_model.yml",
        """
        models:
          - name: wide_churn_model
            task: binary_classification
            adapter: fake
            owner: ds@example.com
            dataset: ref('wide_churn')
            target: churned
            features:
              include: ["*"]
              exclude: [customer_id, safe_id]
            hyperparameters: {fake_metric_value: 0.7}
            evaluation:
              protocol: {split: temporal}
              metrics: [pr_auc]
            seed: 3
        """,
    )
    return demo_project


def test_population_spine_builds_and_trains(
    population_project: Path, fake_registry: AdapterRegistry
) -> None:
    results = invoke(population_project, fake_registry, select=["wide_churn_model"])
    assert results.exit_code() == 0
    by_id = {r.unique_id: r for r in results.results}
    assert by_id[DS].status == "success"
    assert by_id[MODEL].status == "success"

    key = next((population_project / "target/datasets/wide_churn").iterdir())
    train = pq.read_table(key / "train.parquet")
    # Spine + both feature tables + label, with the label's join columns
    # projected away (no duplicates, no __mbt_lbl aliases).
    assert set(train.column_names) == {
        "customer_id",
        "safe_id",
        "snapshot_date",
        "age_years",
        "txn_total",
        "churned",
    }

    # The offset join matched label rows at snapshot + 1 month: the label
    # value encodes (customer_id + month_index) parity of the SPINE month.
    months = {when: idx for idx, when in enumerate(MONTHS)}
    for batch in train.to_batches():
        rows = batch.to_pylist()
        for row in rows:
            expected = (row["customer_id"] + months[row["snapshot_date"]]) % 2
            assert row["churned"] == expected


def test_per_table_projection_prunes_at_the_source(
    population_project: Path, fake_registry: AdapterRegistry
) -> None:
    """ADR-25 through the local adapter: a keep-list and a drop-list on the
    two feature tables prune inside DuckDB's scan, so the materialized panel
    never contains the pruned columns - unlike the model's features.exclude,
    which selects after materialization."""
    # Widen both feature tables with bookkeeping columns the spec prunes.
    for name, extra in (("demo_features", "ingest_batch"), ("txn_features", "etl_loaded_at")):
        path = population_project / "data" / name / "part-000.parquet"
        table = pq.read_table(path)
        widened = table.append_column(extra, pa.array([f"b-{i}" for i in range(table.num_rows)]))
        pq.write_table(widened, path)
    write(
        population_project / "datasets/wide_churn.yml",
        """
        datasets:
          - name: wide_churn
            inputs:
              population: source('lakehouse', 'population')
              label:
                source: source('lakehouse', 'churn_labels')
                using: [customer_id, snapshot_date]
                time_offset: "1mo"
              features:
                - source: source('lakehouse', 'demo_features')
                  using: [customer_id, snapshot_date]
                  columns: [age_years]
                - source: source('lakehouse', 'txn_features')
                  using: [safe_id, snapshot_date]
                  exclude: [etl_loaded_at]
            sample_key: customer_id
            label:
              column: churned
            split:
              strategy: temporal
              time_column: snapshot_date
              train: "2025-07-01:2026-04-01"
              test: "2026-04-01:2026-07-02"
        """,
    )
    results = invoke(population_project, fake_registry, select=["wide_churn_model"])
    assert results.exit_code() == 0

    key = next((population_project / "target/datasets/wide_churn").iterdir())
    for split in ("train", "test"):
        table = pq.read_table(key / f"{split}.parquet")
        assert set(table.column_names) == {
            "customer_id",
            "safe_id",
            "snapshot_date",
            "age_years",
            "txn_total",
            "churned",
        }


def test_immature_outcomes_drop_via_inner_label_join(
    population_project: Path, fake_registry: AdapterRegistry
) -> None:
    results = invoke(population_project, fake_registry, select=["wide_churn_model"])
    assert results.exit_code() == 0
    key = next((population_project / "target/datasets/wide_churn").iterdir())
    dates: set[datetime] = set()
    for split in ("train", "test"):
        table = pq.read_table(key / f"{split}.parquet", columns=["snapshot_date"])
        dates |= set(table.column("snapshot_date").to_pylist())
    # The newest month has no labels yet; the inner label join drops it even
    # though the test window extends past it. Its neighbors are all present.
    assert MONTHS[-1] not in dates
    assert MONTHS[-2] in dates
    assert MONTHS[0] in dates


def test_dataset_depends_on_all_four_sources(
    population_project: Path, fake_registry: AdapterRegistry
) -> None:
    from test_compile import compile_demo

    manifest = compile_demo(population_project, fake_registry)
    node = manifest.nodes[DS]
    assert set(node.depends_on) == {
        "source.demo.lakehouse.population",
        "source.demo.lakehouse.churn_labels",
        "source.demo.lakehouse.demo_features",
        "source.demo.lakehouse.txn_features",
    }
    assert node.snapshot_id and node.snapshot_id.startswith("sha256:")


def test_panel_sampling_keeps_whole_customers(
    population_project: Path, fake_registry: AdapterRegistry
) -> None:
    """sample_key: customer_id keeps or drops every snapshot of a customer."""
    results = invoke(
        population_project,
        fake_registry,
        select=["wide_churn_model"],
        cli_vars={"sample_fraction": 0.5},
    )
    assert results.exit_code() == 0
    key = next((population_project / "target/datasets/wide_churn").iterdir())
    kept: set[int] = set()
    counts: dict[int, int] = {}
    for split in ("train", "test"):
        table = pq.read_table(key / f"{split}.parquet", columns=["customer_id"])
        for cid in table.column("customer_id").to_pylist():
            kept.add(cid)
            counts[cid] = counts.get(cid, 0) + 1
    assert 0 < len(kept) < 40
    # every kept customer carries all 12 labeled snapshots (panel sampling)
    assert all(count == len(MONTHS) - 1 for count in counts.values())


def test_label_join_coverage_is_recorded_and_reported(
    population_project: Path, fake_registry: AdapterRegistry
) -> None:
    """The build records spine-vs-matched counts (F21): 13 months x 40
    customers = 520 spine rows, of which the newest month's 40 have no labels
    yet, so the inner label join keeps 480 (92.3%)."""
    from mbt_adapter_base.materialization import MaterializedDatasetHandle

    results = invoke(population_project, fake_registry, select=["wide_churn_model"])
    assert results.exit_code() == 0
    key = next((population_project / "target/datasets/wide_churn").iterdir())
    handle = MaterializedDatasetHandle(key)
    assert handle.label_join_coverage == {"spine_rows": 520, "matched_rows": 480}


def test_label_join_coverage_check_gates_the_build(
    population_project: Path, fake_registry: AdapterRegistry
) -> None:
    """A declared floor above the real 92.3% coverage turns the silent drop
    into a loud quality failure (exit 2), and a floor below it passes."""
    dataset = population_project / "datasets/wide_churn.yml"
    original = dataset.read_text()
    dataset.write_text(
        original.replace(
            "sample_key: customer_id",
            "sample_key: customer_id\n"
            "    checks:\n"
            "      - label_join_coverage: {min_fraction: 0.99}\n"
            "      - label_leakage_scan: {enabled: false}\n",
        )
    )
    results = invoke(population_project, fake_registry, select=["wide_churn_model"])
    assert results.exit_code() == 2
    by_id = {r.unique_id: r for r in results.results}
    assert by_id[DS].status == "test_failed"
    coverage = next(t for t in by_id[DS].tests if t.name == "label_join_coverage")
    assert "matched 480 of 520 spine rows (92.3%)" in coverage.message

    dataset.write_text(original.replace("min_fraction: 0.99", "min_fraction: 0.9"))
    dataset.write_text(
        original.replace(
            "sample_key: customer_id",
            "sample_key: customer_id\n"
            "    checks:\n"
            "      - label_join_coverage: {min_fraction: 0.9}\n"
            "      - label_leakage_scan: {enabled: false}\n",
        )
    )
    results = invoke(population_project, fake_registry, select=["wide_churn_model"])
    assert results.exit_code() == 0


def test_pre_join_unique_source_check_catches_a_fanned_feature_table(
    population_project: Path, fake_registry: AdapterRegistry
) -> None:
    """unique with source: fails on the RAW feature table whose duplicated
    join key would fan out the spine (F2) - blame lands on the table, not on
    a mysterious row-count inflation post-join."""
    # duplicate one txn_features row -> (safe_id, snapshot_date) is not unique
    txn_dir = population_project / "data" / "txn_features"
    table = pq.read_table(txn_dir / "part-000.parquet")
    pq.write_table(pa.concat_tables([table, table.slice(0, 1)]), txn_dir / "part-000.parquet")
    dataset = population_project / "datasets/wide_churn.yml"
    dataset.write_text(
        dataset.read_text().replace(
            "sample_key: customer_id",
            "sample_key: customer_id\n"
            "    checks:\n"
            "      - unique: {source: lakehouse.txn_features, columns: [safe_id, snapshot_date]}\n"
            "      - label_leakage_scan: {enabled: false}\n",
        )
    )
    results = invoke(population_project, fake_registry, select=["wide_churn_model"])
    assert results.exit_code() == 2
    by_id = {r.unique_id: r for r in results.results}
    unique = next(t for t in by_id[DS].tests if t.name == "unique")
    assert not unique.passed
    assert "source lakehouse.txn_features" in unique.message
    assert "composite key (safe_id, snapshot_date): 1 duplicated key(s)" in unique.message


def test_relationships_check_runs_against_a_raw_source(
    population_project: Path, fake_registry: AdapterRegistry
) -> None:
    """relationships pulls the parent side from the RAW source through the
    local adapter (F2/F21): spine safe_ids all exist in the population table
    (passes); customer_ids checked against safe_id are all orphans (fails)."""
    dataset = population_project / "datasets/wide_churn.yml"
    original = dataset.read_text()
    dataset.write_text(
        original.replace(
            "sample_key: customer_id",
            "sample_key: customer_id\n"
            "    checks:\n"
            "      - relationships: {column: safe_id, to: lakehouse.population, field: safe_id}\n"
            "      - label_leakage_scan: {enabled: false}\n",
        )
    )
    results = invoke(population_project, fake_registry, select=["wide_churn_model"])
    assert results.exit_code() == 0

    dataset.write_text(
        original.replace(
            "sample_key: customer_id",
            "sample_key: customer_id\n"
            "    checks:\n"
            "      - relationships: "
            "{column: customer_id, to: lakehouse.txn_features, field: safe_id}\n"
            "      - label_leakage_scan: {enabled: false}\n",
        )
    )
    results = invoke(population_project, fake_registry, select=["wide_churn_model"])
    assert results.exit_code() == 2
    by_id = {r.unique_id: r for r in results.results}
    rel = next(t for t in by_id[DS].tests if t.name == "relationships")
    assert "not in lakehouse.txn_features.safe_id" in rel.message


# -- schema validation ------------------------------------------------------


def _base(inputs: dict) -> dict:
    return {
        "name": "bad",
        "inputs": inputs,
        "label": {"column": "y"},
        "split": {"time_column": "ts", "train": "-30d:-7d", "test": "7d"},
    }


def test_label_mapping_requires_population() -> None:
    with pytest.raises(Exception, match="requires a 'population' spine"):
        DatasetSpec.model_validate(
            _base(
                {
                    "label": {"source": "source('a', 'l')", "using": ["id"]},
                    "features": ["source('a', 'f')"],
                    "join_key": "id",
                }
            )
        )


def test_feature_entry_without_join_columns_is_rejected() -> None:
    with pytest.raises(Exception, match="has no join columns"):
        DatasetSpec.model_validate(
            _base(
                {
                    "label": "source('a', 'l')",
                    "features": ["source('a', 'f')"],
                }
            )
        )


def test_population_label_needs_join_columns() -> None:
    with pytest.raises(Exception, match="label needs join columns"):
        DatasetSpec.model_validate(
            _base(
                {
                    "population": "source('a', 'p')",
                    "label": "source('a', 'l')",
                    "features": [{"source": "source('a', 'f')", "using": ["id"]}],
                }
            )
        )


def test_bad_time_offset_grammar_is_rejected() -> None:
    with pytest.raises(Exception, match="invalid time_offset"):
        DatasetSpec.model_validate(
            _base(
                {
                    "population": "source('a', 'p')",
                    "label": {
                        "source": "source('a', 'l')",
                        "using": ["id", "ts"],
                        "time_offset": "1month",
                    },
                    "features": [{"source": "source('a', 'f')", "using": ["id"]}],
                }
            )
        )


def test_offset_must_shift_the_split_time_column() -> None:
    with pytest.raises(Exception, match="must be one of the label's join columns"):
        DatasetSpec.model_validate(
            _base(
                {
                    "population": "source('a', 'p')",
                    "label": {
                        "source": "source('a', 'l')",
                        "using": ["id"],
                        "time_offset": "1mo",
                    },
                    "features": [{"source": "source('a', 'f')", "using": ["id"]}],
                }
            )
        )


def test_single_string_using_and_accessors() -> None:
    spec = DatasetSpec.model_validate(
        _base(
            {
                "population": "source('a', 'p')",
                "label": {
                    "source": "source('a', 'l')",
                    "using": "id",
                },
                "features": [{"source": "source('a', 'f')", "using": "id"}],
            }
        )
    )
    assert spec.inputs is not None
    assert spec.inputs.label_join_columns == ["id"]
    assert spec.inputs.feature_entries == [("source('a', 'f')", ["id"], None, None)]
    assert spec.inputs.feature_sources == ["source('a', 'f')"]
    assert spec.inputs.spine == "source('a', 'p')"
    # sample_key fallback: no join_key, so the label's join columns
    assert spec.sample_key_columns == ["id"]


def test_per_table_projection_normalizes_and_dedupes() -> None:
    """ADR-25: `columns` (keep) and `exclude` (drop) ride each feature entry;
    keep_columns prepends the join columns and dedupes redundant mentions."""
    spec = DatasetSpec.model_validate(
        _base(
            {
                "population": "source('a', 'p')",
                "label": {"source": "source('a', 'l')", "using": "id"},
                "features": [
                    {"source": "source('a', 'f')", "using": "id", "columns": ["id", "x"]},
                    {"source": "source('a', 'g')", "using": "id", "exclude": "loaded_at"},
                    "source('a', 'h')",
                ],
                "join_key": "id",
            }
        )
    )
    assert spec.inputs is not None
    keep, drop, bare = spec.inputs.feature_entries
    assert keep.keep_columns == ["id", "x"]  # join column deduped, order kept
    assert keep.exclude is None
    assert drop.exclude == ["loaded_at"] and drop.keep_columns is None
    assert bare.columns is None and bare.exclude is None


def test_per_table_projection_rejects_bad_shapes() -> None:
    def feature(**overrides: object) -> dict:
        base = _base(
            {
                "label": "source('a', 'l')",
                "features": [{"source": "source('a', 'f')", **overrides}],
                "join_key": "id",
            }
        )
        return base

    with pytest.raises(Exception, match="mutually exclusive"):
        DatasetSpec.model_validate(feature(columns=["x"], exclude=["y"]))
    with pytest.raises(Exception, match="at least one non-empty column"):
        DatasetSpec.model_validate(feature(columns=[]))
    with pytest.raises(Exception, match="at least one non-empty column"):
        DatasetSpec.model_validate(feature(exclude=[""]))
    with pytest.raises(Exception, match="join column cannot be dropped"):
        DatasetSpec.model_validate(feature(exclude=["id"]))


def test_mapping_entries_without_using_fall_back_to_join_key() -> None:
    spec = DatasetSpec.model_validate(
        _base(
            {
                "population": "source('a', 'p')",
                "label": {"source": "source('a', 'l')", "time_offset": "1mo"},
                "features": [{"source": "source('a', 'f')"}],
                "join_key": ["id", "ts"],
            }
        )
    )
    assert spec.inputs is not None
    assert spec.inputs.label_join_columns == ["id", "ts"]
    assert spec.inputs.feature_entries == [("source('a', 'f')", ["id", "ts"], None, None)]


def test_empty_using_is_rejected() -> None:
    with pytest.raises(Exception, match="'using' must name at least one"):
        DatasetSpec.model_validate(
            _base(
                {
                    "label": "source('a', 'l')",
                    "features": [{"source": "source('a', 'f')", "using": []}],
                    "join_key": "id",
                }
            )
        )
    with pytest.raises(Exception, match="'using' must name at least one"):
        DatasetSpec.model_validate(
            _base(
                {
                    "population": "source('a', 'p')",
                    "label": {"source": "source('a', 'l')", "using": []},
                    "features": [{"source": "source('a', 'f')", "using": "id"}],
                }
            )
        )


def test_offset_requires_a_split_time_column() -> None:
    with pytest.raises(Exception, match="split declares none"):
        DatasetSpec.model_validate(
            {
                "name": "bad",
                "inputs": {
                    "population": "source('a', 'p')",
                    "label": {
                        "source": "source('a', 'l')",
                        "using": ["id"],
                        "time_offset": "1mo",
                    },
                    "features": [{"source": "source('a', 'f')", "using": "id"}],
                },
                "label": {"column": "y"},
                "split": {"strategy": "random", "train": "0.8", "test": "0.2", "seed": 7},
            }
        )


def test_parse_time_offset_units_and_signs() -> None:
    assert parse_time_offset("1mo") == (1, "mo")
    assert parse_time_offset("-28d") == (-28, "d")
    assert parse_time_offset("+2w") == (2, "w")
    assert parse_time_offset("12h") == (12, "h")
    with pytest.raises(ValueError, match="invalid time_offset"):
        parse_time_offset("mo")
