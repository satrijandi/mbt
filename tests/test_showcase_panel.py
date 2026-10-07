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
import re
import sys
from datetime import UTC, datetime, timedelta
from itertools import pairwise
from pathlib import Path

import pyarrow.parquet as pq
import pytest
from showcase_utils import ANCHOR, MONITOR_ANCHOR

REPO_ROOT = Path(__file__).resolve().parent.parent
SHOWCASE = REPO_ROOT / "examples" / "showcase"
PROJECT = SHOWCASE / "project"
GENERATOR = SHOWCASE / "bootstrap" / "churn_panel.py"
LAKE_ANCHOR = PROJECT / "scripts" / "lake_anchor.py"
SMALL = ["--customers", "60", "--noise-columns", "3"]
#: A real date after the seeded as-of date, for the advance tests: they must
#: not depend on the calendar the suite happens to run on.
TODAY = ["--today", "2026-10-07"]


def _load(path: Path, name: str):
    spec = importlib.util.spec_from_file_location(name, path)
    assert spec is not None and spec.loader is not None
    module = importlib.util.module_from_spec(spec)
    # dataclasses resolve their module through sys.modules.
    sys.modules[spec.name] = module
    spec.loader.exec_module(module)
    return module


@pytest.fixture(scope="module")
def panel():
    return _load(GENERATOR, "showcase_churn_panel")


@pytest.fixture(scope="module")
def lake_anchor():
    return _load(LAKE_ANCHOR, "showcase_lake_anchor")


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
    assert cohorts == panel.COHORTS

    # The label is NULL exactly where no outcome is known: inactive rows, and
    # the newest cohort while its outcome window is open.
    for row in rows:
        known = row["is_active"] and panel.outcome_known(row["inference_date"])
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


# -- time moves: advance ----------------------------------------------------------


def _advance(panel, lake: Path, *extra: str) -> None:
    assert panel.main(["--lake", str(lake), "advance", *extra]) == 0


def test_a_week_passing_labels_the_newest_cohort_and_lands_the_next(panel, tmp_path) -> None:
    lake = _seed(panel, tmp_path / "lake")
    seeded = _files(lake)
    _advance(panel, lake, *TODAY)

    after = _files(lake)
    newest = panel.NEWEST + timedelta(weeks=1)
    assert set(after) - set(seeded) == {
        f"{panel.NEWEST:%Y-%m-%d}-matured-000.parquet",
        f"{newest:%Y-%m-%d}-open-000.parquet",
    }
    assert set(seeded) - set(after) == {f"{panel.NEWEST:%Y-%m-%d}-open-000.parquet"}
    # Every cohort the advance did not touch keeps its exact bytes: moving the
    # clock never rewrites history.
    untouched = set(seeded) - {f"{panel.NEWEST:%Y-%m-%d}-open-000.parquet"}
    assert all(after[name] == seeded[name] for name in untouched)

    as_of = newest
    for row in _rows(lake):
        known = row["is_active"] and panel.outcome_known(row["inference_date"], as_of)
        assert (row["is_churn"] is not None) == known, row
    # The new cohort is the same one a full replay to that date generates.
    replay = tmp_path / "replay"
    _seed(panel, replay)
    _advance(panel, replay, *TODAY)
    assert _files(replay) == after


def test_advancing_twice_equals_advancing_two_weeks(panel, tmp_path: Path) -> None:
    one = _seed(panel, tmp_path / "one")
    _advance(panel, one, "--simulate-future", *TODAY)
    _advance(panel, one, "--simulate-future", *TODAY)
    two = _seed(panel, tmp_path / "two")
    _advance(panel, two, "--weeks", "2", "--simulate-future", *TODAY)
    assert _files(one) == _files(two)


def test_the_lake_never_passes_today_unless_told_to_simulate(panel, tmp_path: Path) -> None:
    """A cohort dated after today describes a population that does not exist
    yet - the same honesty that keeps an open cohort's label NULL."""
    lake = _seed(panel, tmp_path / "lake")
    before = _files(lake)
    with pytest.raises(SystemExit, match="has not happened yet"):
        panel.main(["--lake", str(lake), "advance", "--today", "2026-10-04"])
    assert _files(lake) == before
    _advance(panel, lake, "--today", "2026-10-05")  # the day the cohort lands
    with pytest.raises(SystemExit, match="--simulate-future"):
        panel.main(["--lake", str(lake), "advance", "--today", "2026-10-11"])


def test_the_simulated_book_outlives_the_seeded_customer_pool(panel, tmp_path: Path) -> None:
    """Joiners exhaust the seeded pool after about 16 weeks; past that the
    book keeps growing from a stream of its own, with unique keys and churn
    still in it."""
    lake = _seed(panel, tmp_path / "lake")
    _advance(panel, lake, "--weeks", "14", "--simulate-future", *TODAY)
    newest = panel.NEWEST + timedelta(weeks=14)
    rows = _rows(lake, f"{newest:%Y-%m-%d}-*.parquet")
    previous = _rows(lake, f"{newest - timedelta(weeks=1):%Y-%m-%d}-*.parquet")
    assert len(rows) > len(previous) > 0
    assert len({row["customer_id"] for row in rows}) == len(rows)
    assert {row["is_churn"] for row in previous if row["is_active"]} == {0, 1}


def test_seed_takes_an_advanced_lake_back_to_the_seeded_week(panel, tmp_path: Path) -> None:
    lake = _seed(panel, tmp_path / "lake")
    seeded = _files(lake)
    _advance(panel, lake, "--weeks", "3", "--simulate-future", *TODAY)
    _seed(panel, lake)
    assert _files(lake) == seeded


# -- anchors come from the data ---------------------------------------------------


def test_the_lake_anchor_is_the_day_after_its_newest_cohort(panel, lake_anchor, tmp_path) -> None:
    lake = _seed(panel, tmp_path / "lake")
    assert lake_anchor.anchor(str(lake)) == ANCHOR
    assert lake_anchor.anchor(str(lake), monitor=True) == MONITOR_ANCHOR
    _advance(panel, lake, *TODAY)
    assert lake_anchor.anchor(str(lake)) == "2026-10-06T00:00:00Z"
    assert lake_anchor.anchor(str(lake), monitor=True) == "2026-10-16T00:00:00Z"
    # Outcomes landing and drift change the newest cohort's files, not its date.
    assert panel.main(["--lake", str(lake), "inject-drift"]) == 0
    assert lake_anchor.anchor(str(lake)) == "2026-10-06T00:00:00Z"


def test_an_unseeded_lake_has_no_anchor(lake_anchor, tmp_path: Path) -> None:
    with pytest.raises(SystemExit, match="seed the lake first"):
        lake_anchor.anchor(str(tmp_path / "empty"))


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
    this week's prediction against old outcomes. The join must carry the
    cohort's time column too."""
    (dataset,) = parsed.datasets.values()
    (scoring,) = parsed.scoring.values()
    time_column = dataset.spec.split.time_column
    assert scoring.spec.input.time_column == time_column
    assert set(scoring.spec.ground_truth.join_columns) == {"customer_id", time_column}
    assert scoring.spec.ground_truth.label.column == dataset.spec.label.column


def test_cohorts_are_weekly_from_august_3_to_the_as_of_date(panel) -> None:
    first, *_, newest = panel.COHORTS
    assert (first, newest) == (datetime(2026, 8, 3), datetime(2026, 9, 28))
    assert newest == panel.NEWEST
    assert all(c.weekday() == 0 for c in panel.COHORTS), "every cohort is a Monday"
    gaps = {b - a for a, b in pairwise(panel.COHORTS)}
    assert gaps == {timedelta(weeks=1)}
    # No cohort from a population that does not exist yet.
    assert max(panel.COHORTS) <= panel.AS_OF


def test_only_cohorts_whose_outcome_week_closed_are_labelled(parsed, panel) -> None:
    """On the as-of date, a cohort's label exists only once its whole outcome
    week has passed. 2026-09-21's closed on 2026-09-28; 2026-09-28's runs to
    2026-10-05, so it is the one open cohort - and the only one land-outcomes,
    inject-drift and reset rewrite."""
    (dataset,) = parsed.datasets.values()
    # The generator's outcome week IS the dataset's declared label horizon.
    assert dataset.spec.label.horizon == f"{panel.LABEL_HORIZON.days}d" == "7d"
    assert [c for c in panel.COHORTS if not panel.outcome_known(c)] == [panel.NEWEST]
    assert panel.outcome_known(datetime(2026, 9, 21))


def _cohorts_in(cohorts: list[datetime], start: datetime, end: datetime) -> list[datetime]:
    return [c for c in cohorts if start <= c.replace(tzinfo=UTC) < end]


def _anchor_after(weeks: int) -> datetime:
    """The lake's anchor once it has advanced ``weeks`` weeks past the seed."""
    return datetime.fromisoformat(ANCHOR) + timedelta(weeks=weeks)


def test_a_month_of_training_and_the_labelled_september_weeks_of_testing(parsed, panel) -> None:
    """The weekly model's contract at the seeded anchor: a 7d label, embargoed
    by 7d, trained on August's four Monday cohorts and tested on September's
    three labelled ones (09-28 is still open). The windows resolve the way the
    compiler resolves them (the embargo trimming the train window's tail)."""
    from mbt.compile.windows import parse_window, subtract_duration

    (dataset,) = parsed.datasets.values()
    (scoring,) = parsed.scoring.values()
    split = dataset.spec.split
    assert dataset.spec.label.horizon == "7d"
    assert split.embargo == "7d"
    assert scoring.spec.ground_truth.maturity == "7d"

    anchor = datetime.fromisoformat(ANCHOR)
    start, end = parse_window(str(split.train)).resolve(anchor)
    train = _cohorts_in(panel.COHORTS, start, subtract_duration(end, split.embargo))
    test = _cohorts_in(panel.COHORTS, *parse_window(str(split.test)).resolve(anchor))
    assert train == [datetime(2026, 8, d) for d in (3, 10, 17, 24)]
    assert test == [datetime(2026, 9, d) for d in (7, 14, 21)]
    assert all(panel.outcome_known(c) for c in train + test)


@pytest.mark.parametrize("weeks", [0, 1, 2, 5])
def test_every_week_trains_a_month_tests_three_weeks_and_scores_the_newest(
    parsed, panel, weeks: int
) -> None:
    """Time moves (make advance-week) and the windows move with it: at every
    lake anchor the dataset trains on four labelled cohorts, tests on the
    three before the newest, never reaches the open cohort, and scoring reads
    exactly the newest one - which monitoring then waits out."""
    from mbt.compile.windows import parse_window, subtract_duration

    (dataset,) = parsed.datasets.values()
    (scoring,) = parsed.scoring.values()
    split = dataset.spec.split
    as_of = panel.NEWEST + timedelta(weeks=weeks)
    cohorts = panel.cohorts_until(as_of)
    anchor = _anchor_after(weeks)

    start, end = parse_window(str(split.train)).resolve(anchor)
    train = _cohorts_in(cohorts, start, subtract_duration(end, split.embargo))
    test = _cohorts_in(cohorts, *parse_window(str(split.test)).resolve(anchor))
    assert len(train) == 4 and len(test) == 3, (train, test)
    assert max(train) + timedelta(weeks=2) == min(test), "one embargoed cohort between"
    assert max(test) + timedelta(weeks=1) == as_of
    assert all(panel.outcome_known(c, as_of) for c in train + test)
    assert not panel.outcome_known(as_of, as_of)

    window = parse_window(str(scoring.spec.input.window)).resolve(anchor)
    assert _cohorts_in(cohorts, *window) == [as_of]
    maturity = timedelta(days=int(scoring.spec.ground_truth.maturity.removesuffix("d")))
    monitor_anchor = datetime.fromisoformat(MONITOR_ANCHOR) + timedelta(weeks=weeks)
    assert anchor + maturity <= monitor_anchor


@pytest.mark.parametrize("days_late", [0, 1, 2])
def test_scoring_reads_exactly_the_newest_cohort(parsed, panel, days_late: int) -> None:
    """At the lake's anchor and the later anchors the live tier scores at, the
    input window is the newest (still unlabelled) cohort and nothing else."""
    from mbt.compile.windows import parse_window

    (scoring,) = parsed.scoring.values()
    anchor = datetime.fromisoformat(ANCHOR) + timedelta(days=days_late)
    window = parse_window(str(scoring.spec.input.window)).resolve(anchor)
    assert _cohorts_in(panel.COHORTS, *window) == [panel.NEWEST]
    # Monitoring waits out the maturity for every one of those runs.
    maturity = timedelta(days=int(scoring.spec.ground_truth.maturity.removesuffix("d")))
    assert anchor + maturity <= datetime.fromisoformat(MONITOR_ANCHOR)


def test_no_showcase_file_pins_an_anchor() -> None:
    """Every anchor derives from the lake (scripts/lake_anchor.py): a date
    literal in the Makefile, a DAG, a pipeline, the CronJob or the notebook
    would score an empty window the first time the lake advances."""
    pinned = {
        path.relative_to(SHOWCASE): sorted(set(re.findall(r"\d{4}-\d\d-\d\dT\d\d:\d\d", text)))
        for path in SHOWCASE.rglob("*")
        if path.is_file()
        and path.suffix in {".py", ".yml", ".yaml", ".ipynb", ".sh", ""}
        and (text := path.read_text(errors="replace"))
    }
    assert {path: found for path, found in pinned.items() if found} == {}


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
