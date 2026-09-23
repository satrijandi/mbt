"""Hermetic guard for the showcase's one lake table (examples/showcase).

The showcase's whole data model is a single table, churn_panel, that
bootstrap/churn_panel.py generates into the lake. The dataset, the scoring
input, and the ground-truth labels all read it; they differ only in which rows
they take. The live tier proves that on the docker stack. These tests need
neither docker nor the stack, which is the point: `-m e2e` excludes the
live_showcase tier and `-m "not e2e"` deselects it, so without this module the
repo battery could not see the showcase project or its generator at all.
"""

import importlib.util
import os
import sys
from datetime import datetime
from pathlib import Path

import pyarrow.parquet as pq
import pytest

REPO_ROOT = Path(__file__).resolve().parent.parent
SHOWCASE = REPO_ROOT / "examples" / "showcase"
PROJECT = SHOWCASE / "project"
GENERATOR = SHOWCASE / "bootstrap" / "churn_panel.py"
SMALL = ["--customers", "60", "--noise-columns", "3"]


def _generator():
    spec = importlib.util.spec_from_file_location("showcase_churn_panel", GENERATOR)
    assert spec is not None and spec.loader is not None
    module = importlib.util.module_from_spec(spec)
    # dataclasses resolve their module through sys.modules.
    sys.modules[spec.name] = module
    spec.loader.exec_module(module)
    return module


@pytest.fixture(scope="module")
def panel():
    return _generator()


def _seed(panel, lake: Path, *extra: str) -> Path:
    assert panel.main(["--lake", str(lake), "seed", *SMALL, *extra]) == 0
    return lake


def _files(lake: Path) -> dict[str, bytes]:
    return {path.name: path.read_bytes() for path in sorted(lake.glob("*.parquet"))}


def _rows(lake: Path, pattern: str = "*.parquet"):
    import pyarrow as pa

    return pa.concat_tables(pq.read_table(p) for p in sorted(lake.glob(pattern))).to_pylist()


# -- the generator ----------------------------------------------------------------


def test_seeding_is_deterministic(panel, tmp_path: Path) -> None:
    first = _files(_seed(panel, tmp_path / "a"))
    second = _files(_seed(panel, tmp_path / "b"))
    assert first and first == second


def test_one_table_holds_population_features_and_label(panel, tmp_path: Path) -> None:
    lake = _seed(panel, tmp_path / "lake")
    rows = _rows(lake)
    cohorts = sorted({row["inference_date"] for row in rows})
    assert cohorts == panel.MONTHS

    # The label is NULL exactly where no outcome is known: inactive rows, and
    # the newest cohort while its outcome window is open.
    for row in rows:
        known = row["is_active"] and row["inference_date"] != panel.NEWEST
        assert (row["is_churn"] is not None) == known, row

    labelled = [row["is_churn"] for row in rows if row["is_churn"] is not None]
    assert set(labelled) == {0, 1}
    assert any(not row["is_active"] for row in rows), "no inactive rows: the filter is untested"
    assert len({(row["customer_id"], row["inference_date"]) for row in rows}) == len(rows)
    assert {f"f{i:04d}" for i in range(3)} <= set(rows[0])


def test_outcomes_land_and_reset_restores_the_seeded_bytes(panel, tmp_path: Path) -> None:
    lake = _seed(panel, tmp_path / "lake")
    seeded = _files(lake)
    newest = f"{panel.NEWEST:%Y-%m-%d}"

    assert panel.main(["--lake", str(lake), "land-outcomes"]) == 0
    landed = _files(lake)
    # New file names, never an in-place rewrite: Spark pins an object-store
    # source by its file listing, so this is what makes the change visible.
    changed = set(landed) ^ set(seeded)
    assert changed == {f"{newest}-open-000.parquet", f"{newest}-matured-000.parquet"}
    newest_rows = _rows(lake, f"{newest}-*.parquet")
    assert all((row["is_churn"] is not None) == row["is_active"] for row in newest_rows)
    assert {row["is_churn"] for row in newest_rows if row["is_active"]} == {0, 1}

    assert panel.main(["--lake", str(lake), "reset"]) == 0
    assert _files(lake) == seeded


def test_drift_shifts_features_but_not_keys_or_labels(panel, tmp_path: Path) -> None:
    lake = _seed(panel, tmp_path / "lake")
    newest = f"{panel.NEWEST:%Y-%m-%d}"
    before = _rows(lake, f"{newest}-*.parquet")

    assert panel.main(["--lake", str(lake), "inject-drift"]) == 0
    assert [p.name for p in lake.glob(f"{newest}-*")] == [f"{newest}-drifted-000.parquet"]
    after = _rows(lake, f"{newest}-*.parquet")
    for old, new in zip(before, after, strict=True):
        for column in panel.UNSHIFTED:
            assert new[column] == old[column], column
        assert new["age_years"] == 3 * old["age_years"]
        assert new["f0000"] == pytest.approx(3 * old["f0000"])


def test_chunks_bound_file_size_and_survive_a_rewrite(panel, tmp_path: Path) -> None:
    lake = _seed(panel, tmp_path / "lake", "--rows-per-file", "25")
    newest = f"{panel.NEWEST:%Y-%m-%d}"
    chunks = sorted(p.name for p in lake.glob(f"{newest}-open-*.parquet"))
    assert len(chunks) > 1
    assert all(pq.read_metadata(lake / name).num_rows <= 25 for name in chunks)

    # The rewrite reads the seeded parameters back from a file footer, so it
    # regenerates the same rows in the same chunks.
    assert panel.main(["--lake", str(lake), "land-outcomes"]) == 0
    matured = sorted(p.name for p in lake.glob(f"{newest}-*.parquet"))
    assert matured == [name.replace("-open-", "-matured-") for name in chunks]


def test_rewriting_an_unseeded_lake_fails_loudly(panel, tmp_path: Path) -> None:
    with pytest.raises(SystemExit, match="run `seed` first"):
        panel.main(["--lake", str(tmp_path / "empty"), "land-outcomes"])


# -- the project reads it ---------------------------------------------------------


@pytest.fixture(scope="module")
def parsed():
    from mbt.parsing import parse_project

    # profiles.yml is not read by parse, but keep the s3a env_var() calls
    # satisfied for anything that renders it.
    os.environ.setdefault("AWS_ACCESS_KEY_ID", "stub")
    os.environ.setdefault("AWS_SECRET_ACCESS_KEY", "stub")
    return parse_project(PROJECT)


def test_every_node_reads_the_one_table(parsed) -> None:
    (source,) = parsed.sources
    assert source.endswith(".lake.churn_panel"), source
    (dataset,) = parsed.datasets.values()
    (scoring,) = parsed.scoring.values()
    for text in (
        dataset.spec.source,
        scoring.spec.input.source,
        scoring.spec.ground_truth.label.source,
    ):
        assert text == "source('lake', 'churn_panel')", text
    # Every source reference the parser recorded, from any node, is that table.
    for node in (*parsed.datasets.values(), *parsed.scoring.values(), *parsed.models.values()):
        assert set(node.sources) <= {("lake", "churn_panel")}, (node.name, node.sources)


def test_ground_truth_joins_on_key_and_cohort(parsed) -> None:
    """customer_id alone matches every cohort a customer was ever in, grading
    this month's prediction against old outcomes. The join must carry the
    cohort's time column too."""
    (dataset,) = parsed.datasets.values()
    (scoring,) = parsed.scoring.values()
    time_column = dataset.spec.split.time_column
    assert scoring.spec.input.time_column == time_column
    assert set(scoring.spec.ground_truth.join_columns) == {"customer_id", time_column}
    assert scoring.spec.ground_truth.label.column == dataset.spec.label.column


def test_training_never_reaches_the_open_cohort(parsed, panel) -> None:
    (dataset,) = parsed.datasets.values()
    for window in (dataset.spec.split.train, dataset.spec.split.test):
        end = datetime.fromisoformat(str(window).split(":")[1])
        assert end <= panel.NEWEST, f"{window} reaches the unlabelled cohort"


def test_models_read_named_columns_and_declare_every_categorical(parsed, panel) -> None:
    """`features.categorical` is authoritative once present (ADR-27): a string
    feature it does not declare is a hard error at train time, which only the
    live tier would otherwise see."""
    schema = panel.schema(0)
    strings = {f.name for f in schema if str(f.type) == "string"}
    for model in parsed.models.values():
        include = set(model.spec.features.include)
        assert include == set(panel.FEATURES), model.name
        assert include <= set(schema.names), model.name
        categorical = set(model.spec.features.categorical)
        assert strings & include <= categorical, (model.name, sorted(strings - categorical))
        assert "contract_code" in categorical, model.name
