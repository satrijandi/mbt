"""The after-test split (ADR-30): windows, identity, checks, adapters, and the
training view that keeps it - and the undocumented test_window gap - in line."""

from datetime import date, timedelta
from pathlib import Path

import pyarrow as pa
import pyarrow.parquet as pq
import pytest
from core_helpers import TEST_ANCHOR
from exec_unit_helpers import DATASET_UID, make_training_job, recording_bus
from mbt_testing.adapters import FakeTrainingAdapter
from misc_unit_helpers import RecordingSink, make_node
from parse_unit_helpers import error_messages
from report_unit_helpers import MODEL_FILE, compile_demo, edit, model_evaluation, with_out_of_time

from mbt.adapters.local.data import LocalDataAdapter
from mbt.adapters.registry import AdapterRegistry
from mbt.compile.hashing import config_hash
from mbt.contracts import (
    CONTRACT_VERSION,
    OUT_OF_TIME_SPLIT,
    AdapterPlugin,
    DatasetSpec,
    SourceTable,
)
from mbt.events.models import EmptyAfterTestSplit
from mbt.exceptions import CompilationError, ConfigError
from mbt.execute.handles import TrainingSplitView
from mbt.execute.job import _materialize_for_path_adapter, _prepare, run_job
from mbt.execute.runners import BuildContext
from mbt.parsing import parse_project
from mbt.quality.checks import labeled_splits, run_checks
from mbt_adapter_base.datasets import InMemoryDatasetHandle

MODEL = "model.demo.churn_model"


# -- compile and parse --------------------------------------------------------------


def test_compile_resolves_the_out_of_time_window(
    demo_project: Path, fake_registry: AdapterRegistry
) -> None:
    with_out_of_time(demo_project)
    windows = compile_demo(demo_project, fake_registry).nodes[DATASET_UID].resolved["windows"]
    assert windows["test"] == ["2026-05-02T00:00:00Z", "2026-06-01T00:00:00Z"]
    assert windows[OUT_OF_TIME_SPLIT] == ["2026-06-01T00:00:00Z", "2026-07-01T00:00:00Z"]


def test_relative_overlap_with_the_test_window_is_a_parse_error(
    demo_project: Path, fake_registry: AdapterRegistry
) -> None:
    with_out_of_time(demo_project, after="-45d:now")
    parsed = parse_project(demo_project, registry=fake_registry, raise_on_error=False)
    assert any("starts before the test window" in m for m in error_messages(parsed))


def test_mixed_kind_overlap_is_caught_at_compile(
    demo_project: Path, fake_registry: AdapterRegistry
) -> None:
    # An absolute start orders against a relative test end only at an anchor.
    with_out_of_time(demo_project, after="2026-05-15:now")
    with pytest.raises(CompilationError, match="before the test window ends at 2026-06-01"):
        compile_demo(demo_project, fake_registry)


def test_unparseable_out_of_time_window_is_a_parse_error(
    demo_project: Path, fake_registry: AdapterRegistry
) -> None:
    with_out_of_time(demo_project, after="soon")
    parsed = parse_project(demo_project, registry=fake_registry, raise_on_error=False)
    assert any(
        "soon" in i.message and i.field_path == "/split/out_of_time" for i in parsed.report.errors
    )


def test_after_test_gates_and_stability_need_the_window(
    demo_project: Path, fake_registry: AdapterRegistry
) -> None:
    model_evaluation(
        demo_project,
        "      stability:\n        prediction_shift: {threshold: 0.25}\n",
    )
    edit(
        demo_project / "models/churn_model.yml",
        "          threshold: \"{{ var('default_threshold') }}\"\n",
        "          threshold: \"{{ var('default_threshold') }}\"\n"
        "        - metric: roc_auc\n"
        "          threshold: 0.5\n"
        "          source: out_of_time\n",
    )
    parsed = parse_project(demo_project, registry=fake_registry, raise_on_error=False)
    paths = {i.field_path for i in parsed.report.errors}
    assert {"/evaluation/gates/1/source", "/evaluation/stability"} <= paths

    with_out_of_time(demo_project)
    parsed = parse_project(demo_project, registry=fake_registry, raise_on_error=False)
    assert not parsed.report.errors


def test_non_native_report_engine_must_be_an_installed_plugin(
    demo_project: Path, fake_registry: AdapterRegistry
) -> None:
    model_evaluation(demo_project, "      report: {stability: {engine: drifty}}\n")
    parsed = parse_project(demo_project, registry=fake_registry, raise_on_error=False)
    engine = [
        i for i in parsed.report.errors if i.field_path == "/evaluation/report/stability/engine"
    ]
    assert engine and "adapter 'drifty' is not installed" in engine[0].message
    assert "install mbt-drifty, or use engine: native" in (engine[0].hint or "")

    fake_registry.register(AdapterPlugin(name="drifty", contract_version=CONTRACT_VERSION))
    parsed = parse_project(demo_project, registry=fake_registry, raise_on_error=False)
    assert any("provides no report engine" in m for m in error_messages(parsed))

    # mbt-testing's fake plugin carries one, as mbt-evidently does
    edit(demo_project / MODEL_FILE, "engine: drifty", "engine: fake")
    parsed = parse_project(demo_project, registry=fake_registry, raise_on_error=False)
    assert not parsed.report.errors


# -- identity ------------------------------------------------------------------------


def test_report_block_is_presentation_not_identity(
    demo_project: Path, fake_registry: AdapterRegistry
) -> None:
    before = compile_demo(demo_project, fake_registry).nodes[MODEL]
    model_evaluation(demo_project, "      report: {binning: all, importance: {top_n: 5}}\n")
    after = compile_demo(demo_project, fake_registry).nodes[MODEL]
    assert after.config["evaluation"]["report"]["binning"] == "all"
    assert before.config_hash == after.config_hash


def test_stability_thresholds_are_identity(
    demo_project: Path, fake_registry: AdapterRegistry
) -> None:
    with_out_of_time(demo_project)
    before = compile_demo(demo_project, fake_registry).nodes[MODEL]
    model_evaluation(demo_project, "      stability: {prediction_shift: {threshold: 0.25}}\n")
    after = compile_demo(demo_project, fake_registry).nodes[MODEL]
    assert before.config_hash != after.config_hash


def test_hash_exclusion_tolerates_configs_without_the_path() -> None:
    dataset_like = {"name": "d", "split": {"test": "-7d:now"}}
    assert config_hash(dataset_like) == config_hash(dict(dataset_like))
    odd = {"name": "m", "evaluation": None}
    assert config_hash(odd) != config_hash({"name": "m"})


# -- checks --------------------------------------------------------------------------


def _checks_spec(checks: list) -> DatasetSpec:
    return DatasetSpec.model_validate(
        {
            "name": "unit_ds",
            "source": "source('a', 'b')",
            "sample_key": "user_id",
            "label": {"column": "y"},
            "split": {
                "strategy": "temporal",
                "time_column": "t",
                "train": "-60d:-30d",
                "test": "-30d:-10d",
                "out_of_time": "-10d:now",
            },
            "checks": checks,
        }
    )


def _checks_handle() -> InMemoryDatasetHandle:
    labelled = pa.table({"user_id": [1, 2, 3, 4], "x": [1.0] * 4, "y": [0, 1, 0, 1]})
    immature = pa.table({"user_id": [5, 6], "x": [None, 2.0], "y": pa.array([None, 1], pa.int64())})
    return InMemoryDatasetHandle(
        {"train": labelled, "test": labelled, OUT_OF_TIME_SPLIT: immature}, label_column="y"
    )


def test_labeled_splits_leave_out_the_after_test_split() -> None:
    assert labeled_splits(_checks_handle()) == ["test", "train"]


def test_not_null_accepts_immature_labels_but_still_checks_features() -> None:
    spec = _checks_spec(["not_null", {"not_null": {"columns": ["x"]}}])
    results = run_checks(spec, _checks_handle(), {}, resource="dataset.unit")
    label_check, feature_check = (r for r in results if r.name == "not_null")
    assert label_check.passed, label_check.message
    assert not feature_check.passed
    assert feature_check.message == f"{OUT_OF_TIME_SPLIT}.x: 1 null(s)"


def test_row_count_ignores_the_after_test_split() -> None:
    spec = _checks_spec([{"row_count": {"max": 8}}, {"row_count": {"min": 9}}])
    results = run_checks(spec, _checks_handle(), {}, resource="dataset.unit")
    ceiling, floor = (r for r in results if r.name == "row_count")
    assert ceiling.passed
    assert floor.message == "8 rows below the minimum 9"


# -- the local data adapter -----------------------------------------------------------

ROWS_UID = "source.demo.lakehouse.rows"


def _local_build(tmp_path: Path, windows: dict[str, tuple[str, str]], sink: RecordingSink) -> dict:
    anchor = TEST_ANCHOR.replace(tzinfo=None)
    table = pa.table(
        {
            "user_id": list(range(40)),
            "snapshot_date": [anchor - timedelta(days=1 + i) for i in range(40)],
            "churned": [i % 2 for i in range(40)],
        }
    )
    (tmp_path / "data" / "rows").mkdir(parents=True)
    pq.write_table(table, tmp_path / "data" / "rows" / "part-000.parquet")
    spec = DatasetSpec.model_validate(
        {
            "name": "rows_ds",
            "source": ROWS_UID,
            "sample_key": "user_id",
            "label": {"column": "churned"},
            "split": {
                "strategy": "temporal",
                "time_column": "snapshot_date",
                "train": "-40d:-20d",
                "test": "-20d:-10d",
                "out_of_time": "-10d:now",
            },
        }
    )
    tables = {ROWS_UID: SourceTable(name="rows", path="data/rows/*.parquet")}
    adapter = LocalDataAdapter({"root": str(tmp_path)})
    ctx = BuildContext(
        node=make_node("dataset.demo.rows_ds"),
        source=tables[ROWS_UID],
        source_tables=tables,
        resolved_windows=windows,
        sample_fraction=1.0,
        deep_snapshot=False,
        output_dir=tmp_path / "out",
        events=sink,
    )
    handle = adapter.build_dataset(spec, ctx)
    return {split: handle.read(split).num_rows for split in sorted(handle.splits())}


def test_local_adapter_materializes_the_after_test_split(tmp_path: Path) -> None:
    windows = {
        "train": ("2026-05-22T00:00:00Z", "2026-06-11T00:00:00Z"),
        "test": ("2026-06-11T00:00:00Z", "2026-06-21T00:00:00Z"),
        OUT_OF_TIME_SPLIT: ("2026-06-21T00:00:00Z", "2026-07-01T00:00:00Z"),
    }
    counts = _local_build(tmp_path, windows, RecordingSink())
    assert counts == {"out_of_time": 10, "test": 10, "train": 20}


def test_an_empty_after_test_split_warns_instead_of_failing(tmp_path: Path) -> None:
    windows = {
        "train": ("2026-05-22T00:00:00Z", "2026-06-11T00:00:00Z"),
        "test": ("2026-06-11T00:00:00Z", "2026-07-01T00:00:00Z"),
        OUT_OF_TIME_SPLIT: ("2026-07-01T00:00:00Z", "2026-08-01T00:00:00Z"),
    }
    sink = RecordingSink()
    counts = _local_build(tmp_path, windows, sink)
    assert counts[OUT_OF_TIME_SPLIT] == 0
    empty = [e for e in sink.events if isinstance(e, EmptyAfterTestSplit)]
    assert len(empty) == 1
    # The severity is the EVENT's, not the adapter's. It used to be the
    # adapter's, which is how the same condition was a WARN on local and an
    # info line on Spark and Snowflake (v5 live defect 1); a test that asserted
    # only `level == "warn"` on the local adapter could not see that.
    assert empty[0].level == "warn"
    assert empty[0].window == windows[OUT_OF_TIME_SPLIT]
    assert "'out_of_time' materialized 0 rows [2026-07-01T00:00:00Z" in empty[0].human()


# -- the training view ------------------------------------------------------------------


def _view_base() -> InMemoryDatasetHandle:
    days = [date(2026, 6, 1) + timedelta(days=i) for i in range(10)]
    test = pa.table({"t": days, "x": list(range(10)), "y": [i % 2 for i in range(10)]})
    return InMemoryDatasetHandle(
        {"train": test, "test": test, OUT_OF_TIME_SPLIT: test},
        label_column="y",
        time_column="t",
    )


def test_training_view_hides_the_after_test_split() -> None:
    view = TrainingSplitView(_view_base(), time_column="t")
    assert view.splits() == {"train", "test"}
    assert OUT_OF_TIME_SPLIT not in view.profile().n_rows
    assert view.profile().n_rows == {"train": 10, "test": 10}
    assert view.read("test").num_rows == 10
    assert (view.snapshot_id, view.time_column, view.label_column) == (
        "sha256:inmemory",
        "t",
        "y",
    )
    with pytest.raises(ConfigError, match="reserved for the training report"):
        view.read(OUT_OF_TIME_SPLIT)
    with pytest.raises(NotImplementedError):
        view.locator()


def test_training_view_applies_test_window() -> None:
    view = TrainingSplitView(
        _view_base(),
        time_column="t",
        test_window=("2026-06-03T00:00:00Z", "2026-06-06T00:00:00Z"),
    )
    assert view.read("test", columns=["x"]).column("x").to_pylist() == [2, 3, 4]
    assert view.read("train").num_rows == 10  # only test narrows


def test_test_window_without_a_time_column_is_an_error() -> None:
    view = TrainingSplitView(
        _view_base(), time_column=None, test_window=("2026-06-03T00:00:00Z", "2026-06-06T00:00:00Z")
    )
    with pytest.raises(ConfigError, match="needs the split time column"):
        view.read("test")


# -- the training job ---------------------------------------------------------------------


def test_training_never_reads_the_after_test_split(
    demo_project: Path, fake_registry: AdapterRegistry, monkeypatch: pytest.MonkeyPatch
) -> None:
    with_out_of_time(demo_project)
    _, job = make_training_job(demo_project, fake_registry)
    runtime = _prepare(job)
    assert OUT_OF_TIME_SPLIT in runtime.materialization.splits()
    assert OUT_OF_TIME_SPLIT not in runtime.base_handle.splits()
    staged = _materialize_for_path_adapter(runtime.transformed, runtime.spec)
    assert OUT_OF_TIME_SPLIT not in staged.splits()

    seen: list[set[str]] = []
    original = FakeTrainingAdapter.train

    def spy(self, spec, data, ctx):  # type: ignore[no-untyped-def]
        seen.append(set(data.splits()))
        return original(self, spec, data, ctx)

    monkeypatch.setattr(FakeTrainingAdapter, "train", spy)
    with recording_bus():
        result = run_job(job)
    assert result.status == "success", result.error
    assert seen and all(OUT_OF_TIME_SPLIT not in splits for splits in seen)


def test_job_scores_only_the_declared_test_window(
    demo_project: Path, fake_registry: AdapterRegistry, monkeypatch: pytest.MonkeyPatch
) -> None:
    edit(
        demo_project / "models/churn_model.yml",
        "protocol: {split: temporal}",
        'protocol: {split: temporal, test_window: "-7d:now"}',
    )
    _, job = make_training_job(demo_project, fake_registry)
    full = _prepare(job).materialization.read("test").num_rows
    narrowed = _prepare(job).base_handle.read("test")
    assert 0 < narrowed.num_rows < full
    oldest = min(narrowed.column("snapshot_date").to_pylist())
    assert oldest >= (TEST_ANCHOR - timedelta(days=7)).replace(tzinfo=None)
