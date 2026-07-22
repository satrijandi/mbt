"""Unit tests for built-in dataset checks (mbt/quality/checks.py)."""

import pyarrow as pa
from exec_unit_helpers import recording_bus

from mbt.contracts import DatasetSpec
from mbt.events.models import CheckEvaluated
from mbt.quality.checks import SourceAccess, run_checks
from mbt_adapter_base.datasets import InMemoryDatasetHandle


def _spec(checks: list) -> DatasetSpec:
    return DatasetSpec.model_validate(
        {
            "name": "unit_ds",
            "source": "source('a', 'b')",
            "label": {"column": "y"},
            "split": {
                "strategy": "temporal",
                "time_column": "t",
                "train": "-30d:-7d",
                "test": "-7d:now",
            },
            "checks": checks,
        }
    )


def _handle() -> InMemoryDatasetHandle:
    # 'x' is constant so the auto-appended leakage scan sees a NULL correlation
    table = pa.table({"x": [1.0, 1.0, 1.0, 1.0], "y": [0, 1, 0, 1]})
    return InMemoryDatasetHandle({"train": table}, label_column="y")


def test_schema_check_reports_missing_typed_column() -> None:
    spec = _spec([{"schema": {"columns": {"absent": "int64", "y": "int64"}}}])
    results = {r.name: r for r in run_checks(spec, _handle(), {}, resource="dataset.unit")}
    schema = results["schema"]
    assert not schema.passed
    assert "missing column 'absent'" in schema.message


def test_no_future_columns_without_windows_passes() -> None:
    spec = _spec(["no_future_columns"])
    results = {r.name: r for r in run_checks(spec, _handle(), {}, resource="dataset.unit")}
    check = results["no_future_columns"]
    assert check.passed
    assert "no temporal windows" in check.message


def test_no_future_columns_skips_absent_splits() -> None:
    spec = _spec(["no_future_columns"])
    windows = {"windows": {"validation": ["2026-01-01T00:00:00Z", "2026-02-01T00:00:00Z"]}}
    results = {r.name: r for r in run_checks(spec, _handle(), windows, resource="dataset.unit")}
    check = results["no_future_columns"]
    assert check.passed and check.message == ""


def test_unique_check_flags_a_duplicated_key() -> None:
    # a multi-table join that fans out the population spine leaves a duplicated
    # key; the unique check catches it and fails the build (F2).
    table = pa.table({"user_id": [1, 2, 2, 3], "y": [0, 1, 0, 1]})
    handle = InMemoryDatasetHandle({"train": table}, label_column="y")
    spec = _spec([{"unique": {"columns": ["user_id"]}}, {"label_leakage_scan": {"enabled": False}}])
    results = {r.name: r for r in run_checks(spec, handle, {}, resource="dataset.unit")}
    assert not results["unique"].passed
    assert "user_id: 1 duplicated value" in results["unique"].message  # value 2 appears twice


def test_unique_check_passes_a_distinct_key() -> None:
    table = pa.table({"user_id": [1, 2, 3, 4], "y": [0, 1, 0, 1]})
    handle = InMemoryDatasetHandle({"train": table}, label_column="y")
    spec = _spec([{"unique": {"columns": ["user_id"]}}, {"label_leakage_scan": {"enabled": False}}])
    results = {r.name: r for r in run_checks(spec, handle, {}, resource="dataset.unit")}
    assert results["unique"].passed and results["unique"].message == ""


def test_unique_check_requires_columns() -> None:
    spec = _spec([{"unique": {}}, {"label_leakage_scan": {"enabled": False}}])
    results = {r.name: r for r in run_checks(spec, _handle(), {}, resource="dataset.unit")}
    assert not results["unique"].passed
    assert "requires explicit 'columns'" in results["unique"].message


def test_accepted_values_flags_a_value_outside_the_set() -> None:
    # a categorical feature that drifted to an unexpected level ('enterprise') is
    # caught rather than silently trained as a bogus category (F21).
    table = pa.table({"plan": ["basic", "pro", "enterprise", "basic"], "y": [0, 1, 0, 1]})
    handle = InMemoryDatasetHandle({"train": table}, label_column="y")
    spec = _spec(
        [
            {"accepted_values": {"column": "plan", "values": ["basic", "pro"]}},
            {"label_leakage_scan": {"enabled": False}},
        ]
    )
    results = {r.name: r for r in run_checks(spec, handle, {}, resource="dataset.unit")}
    assert not results["accepted_values"].passed
    assert "plan: 1 unexpected value" in results["accepted_values"].message  # 'enterprise'


def test_accepted_values_passes_all_in_set_and_ignores_nulls() -> None:
    # nulls are ignored, as in dbt - pair with not_null when presence matters.
    table = pa.table({"plan": ["basic", "pro", None, "basic"], "y": [0, 1, 0, 1]})
    handle = InMemoryDatasetHandle({"train": table}, label_column="y")
    spec = _spec(
        [
            {"accepted_values": {"column": "plan", "values": ["basic", "pro"]}},
            {"label_leakage_scan": {"enabled": False}},
        ]
    )
    results = {r.name: r for r in run_checks(spec, handle, {}, resource="dataset.unit")}
    assert results["accepted_values"].passed and results["accepted_values"].message == ""


def test_accepted_values_requires_column_and_values() -> None:
    spec = _spec(
        [{"accepted_values": {"column": "plan"}}, {"label_leakage_scan": {"enabled": False}}]
    )
    results = {r.name: r for r in run_checks(spec, _handle(), {}, resource="dataset.unit")}
    assert not results["accepted_values"].passed
    assert "requires a 'column' and non-empty 'values'" in results["accepted_values"].message


def test_row_count_fails_below_the_minimum() -> None:
    # a silent volume drop (labels fell off the offset grid) becomes a loud fail.
    handle = InMemoryDatasetHandle({"train": pa.table({"y": [0, 1, 0]})}, label_column="y")
    spec = _spec([{"row_count": {"min": 100}}, {"label_leakage_scan": {"enabled": False}}])
    results = {r.name: r for r in run_checks(spec, handle, {}, resource="dataset.unit")}
    assert not results["row_count"].passed
    assert "3 rows below the minimum 100" in results["row_count"].message


def test_row_count_fails_above_the_maximum() -> None:
    handle = InMemoryDatasetHandle({"train": pa.table({"y": [0, 1, 0, 1, 0]})}, label_column="y")
    spec = _spec([{"row_count": {"max": 2}}, {"label_leakage_scan": {"enabled": False}}])
    results = {r.name: r for r in run_checks(spec, handle, {}, resource="dataset.unit")}
    assert not results["row_count"].passed
    assert "5 rows above the maximum 2" in results["row_count"].message


def test_row_count_passes_within_bounds_across_all_splits() -> None:
    # every split counts (they are proportional slices of one build): 3 + 1 = 4
    handle = InMemoryDatasetHandle(
        {"train": pa.table({"y": [0, 1, 0]}), "test": pa.table({"y": [1]})},
        label_column="y",
    )
    spec = _spec([{"row_count": {"min": 2, "max": 10}}, {"label_leakage_scan": {"enabled": False}}])
    results = {r.name: r for r in run_checks(spec, handle, {}, resource="dataset.unit")}
    assert results["row_count"].passed and results["row_count"].message == ""


def test_row_count_requires_a_bound() -> None:
    spec = _spec([{"row_count": {}}, {"label_leakage_scan": {"enabled": False}}])
    results = {r.name: r for r in run_checks(spec, _handle(), {}, resource="dataset.unit")}
    assert not results["row_count"].passed
    assert "requires 'min' and/or 'max'" in results["row_count"].message


class _FakeSourceAdapter:
    """Data-adapter stand-in exposing the optional source-check methods."""

    name = "fake_src"

    def __init__(self, duplicates: int = 0, parents: tuple = ("basic", "pro")) -> None:
        self._duplicates = duplicates
        self._parents = parents
        self.calls: list = []

    def count_source_duplicates(self, table, columns) -> int:
        self.calls.append(("duplicates", tuple(columns)))
        return self._duplicates

    def read_source_distinct(self, table, column):
        self.calls.append(("distinct", column))
        return pa.table({"value": list(self._parents)})


def _sources(adapter=None) -> SourceAccess:
    return SourceAccess(
        tables={"lakehouse.plans": object()}, adapter=adapter or _FakeSourceAdapter()
    )


def test_unique_source_mode_checks_the_composite_key_pre_join() -> None:
    """unique with source: runs against the RAW table before the join can fan
    anything out - the 1:1 join-cardinality contract (F2)."""
    spec = _spec(
        [
            {"unique": {"source": "lakehouse.plans", "columns": ["id", "snapshot_date"]}},
            {"label_leakage_scan": {"enabled": False}},
        ]
    )
    fanned = _FakeSourceAdapter(duplicates=3)
    results = {
        r.name: r for r in run_checks(spec, _handle(), {}, resource="d", sources=_sources(fanned))
    }
    assert not results["unique"].passed
    assert "composite key (id, snapshot_date): 3 duplicated key(s)" in results["unique"].message
    assert ("duplicates", ("id", "snapshot_date")) in fanned.calls

    clean = {r.name: r for r in run_checks(spec, _handle(), {}, resource="d", sources=_sources())}
    assert clean["unique"].passed
    assert "unique" in clean["unique"].message  # reports what it verified


def test_unique_source_mode_resolves_bare_names_and_rejects_unknown() -> None:
    spec = _spec(
        [
            {"unique": {"source": "plans", "columns": ["id"]}},  # bare name, unambiguous
            {"label_leakage_scan": {"enabled": False}},
        ]
    )
    results = {r.name: r for r in run_checks(spec, _handle(), {}, resource="d", sources=_sources())}
    assert results["unique"].passed

    unknown = _spec(
        [
            {"unique": {"source": "ghosts", "columns": ["id"]}},
            {"label_leakage_scan": {"enabled": False}},
        ]
    )
    results = {
        r.name: r for r in run_checks(unknown, _handle(), {}, resource="d", sources=_sources())
    }
    assert not results["unique"].passed
    assert "unknown source 'ghosts'" in results["unique"].message
    assert "lakehouse.plans" in results["unique"].message  # lists what exists


def test_unique_source_mode_needs_a_capable_adapter_and_source_access() -> None:
    spec = _spec(
        [
            {"unique": {"source": "plans", "columns": ["id"]}},
            {"label_leakage_scan": {"enabled": False}},
        ]
    )
    results = {r.name: r for r in run_checks(spec, _handle(), {}, resource="d")}
    assert not results["unique"].passed and "source access" in results["unique"].message

    incapable = SourceAccess(tables={"lakehouse.plans": object()}, adapter=object())
    results = {r.name: r for r in run_checks(spec, _handle(), {}, resource="d", sources=incapable)}
    assert not results["unique"].passed
    assert "does not support source-level checks" in results["unique"].message


def test_relationships_flags_orphaned_values_with_samples() -> None:
    """dbt-parity relationships (F2/F21): child values must exist in the
    referenced source column; orphans fail with examples."""
    table = pa.table({"plan": ["basic", "pro", "legacy", "legacy"], "y": [0, 1, 0, 1]})
    handle = InMemoryDatasetHandle({"train": table}, label_column="y")
    spec = _spec(
        [
            {"relationships": {"column": "plan", "to": "lakehouse.plans", "field": "id"}},
            {"label_leakage_scan": {"enabled": False}},
        ]
    )
    results = {r.name: r for r in run_checks(spec, handle, {}, resource="d", sources=_sources())}
    assert not results["relationships"].passed
    assert "train.plan: 1 value(s) not in lakehouse.plans.id" in results["relationships"].message
    assert "legacy" in results["relationships"].message

    ok = pa.table({"plan": ["basic", "pro", None], "y": [0, 1, 0]})
    handle = InMemoryDatasetHandle({"train": ok}, label_column="y")
    results = {r.name: r for r in run_checks(spec, handle, {}, resource="d", sources=_sources())}
    assert results["relationships"].passed  # nulls ignored, as in dbt


def test_relationships_requires_params_and_source_access() -> None:
    spec = _spec([{"relationships": {}}, {"label_leakage_scan": {"enabled": False}}])
    results = {r.name: r for r in run_checks(spec, _handle(), {}, resource="d", sources=_sources())}
    assert not results["relationships"].passed
    assert "requires 'column', 'to', and 'field'" in results["relationships"].message

    full = _spec(
        [
            {"relationships": {"column": "x", "to": "lakehouse.plans", "field": "id"}},
            {"label_leakage_scan": {"enabled": False}},
        ]
    )
    results = {r.name: r for r in run_checks(full, _handle(), {}, resource="d")}
    assert not results["relationships"].passed
    assert "source access" in results["relationships"].message

    incapable = SourceAccess(tables={"lakehouse.plans": object()}, adapter=object())
    results = {r.name: r for r in run_checks(full, _handle(), {}, resource="d", sources=incapable)}
    assert not results["relationships"].passed
    assert "does not support source-level checks" in results["relationships"].message


class _CoverageHandle(InMemoryDatasetHandle):
    """A handle carrying the label-join coverage a population-spine build records."""

    def __init__(self, coverage: dict | None) -> None:
        table = pa.table({"x": [1.0], "y": [1]})
        super().__init__({"train": table}, label_column="y")
        self._coverage = coverage

    @property
    def label_join_coverage(self) -> dict | None:
        return self._coverage


def test_label_join_coverage_enforces_the_floor() -> None:
    spec = _spec(
        [
            {"label_join_coverage": {"min_fraction": 0.95}},
            {"label_leakage_scan": {"enabled": False}},
        ]
    )
    low = _CoverageHandle({"spine_rows": 520, "matched_rows": 480})
    results = {r.name: r for r in run_checks(spec, low, {}, resource="d")}
    assert not results["label_join_coverage"].passed
    assert "matched 480 of 520 spine rows (92.3%)" in results["label_join_coverage"].message

    lenient = _spec(
        [
            {"label_join_coverage": {"min_fraction": 0.9}},
            {"label_leakage_scan": {"enabled": False}},
        ]
    )
    results = {r.name: r for r in run_checks(lenient, low, {}, resource="d")}
    assert results["label_join_coverage"].passed
    assert "92.3%" in results["label_join_coverage"].message  # reports even on pass


def test_label_join_coverage_degenerate_inputs_fail_clearly() -> None:
    checks = [
        {"label_join_coverage": {"min_fraction": 0.9}},
        {"label_leakage_scan": {"enabled": False}},
    ]
    none = _CoverageHandle(None)
    results = {r.name: r for r in run_checks(_spec(checks), none, {}, resource="d")}
    assert not results["label_join_coverage"].passed
    assert "no label-join coverage recorded" in results["label_join_coverage"].message

    empty = _CoverageHandle({"spine_rows": 0, "matched_rows": 0})
    results = {r.name: r for r in run_checks(_spec(checks), empty, {}, resource="d")}
    assert not results["label_join_coverage"].passed
    assert "0 rows" in results["label_join_coverage"].message

    bad = _spec([{"label_join_coverage": {}}, {"label_leakage_scan": {"enabled": False}}])
    results = {r.name: r for r in run_checks(bad, _CoverageHandle(None), {}, resource="d")}
    assert not results["label_join_coverage"].passed
    assert "requires min_fraction in (0, 1]" in results["label_join_coverage"].message


def test_run_checks_emits_check_evaluated_per_check() -> None:
    """Every check evaluated puts a CheckEvaluated on the bus - a failing one,
    a passing one, and an explicitly disabled one - carrying the dataset uid."""
    spec = _spec(
        [
            {"schema": {"columns": ["absent"]}},  # fails: missing column
            "not_null",  # passes: label 'y' has no nulls
            {"label_leakage_scan": {"enabled": False}},  # disabled
        ]
    )
    with recording_bus() as sink:
        results = run_checks(spec, _handle(), {}, resource="dataset.unit")
    evaluated = [e for e in sink.events if isinstance(e, CheckEvaluated)]
    # One event per result (the leakage scan is not auto-appended a second time
    # because it is declared here).
    assert len(evaluated) == len(results)
    assert all(e.unique_id == "dataset.unit" for e in evaluated)
    outcomes = {e.check: e.passed for e in evaluated}
    assert outcomes["schema"] is False
    assert outcomes["not_null"] is True
    disabled = next(e for e in evaluated if e.check == "label_leakage_scan")
    assert disabled.passed and disabled.message == "check disabled"


def test_check_names_match_dispatch_table() -> None:
    """The name registry and the impl dispatch table are one source of truth:
    every declared name has an implementation and vice versa, so a check added
    to only one side fails loudly here instead of becoming a confusing
    'unknown dataset check' at parse time."""
    from mbt.quality.check_names import BUILTIN_CHECK_NAMES, SCORING_CHECK_NAMES
    from mbt.quality.checks import _CHECKS

    assert set(_CHECKS) == set(BUILTIN_CHECK_NAMES)
    assert SCORING_CHECK_NAMES <= BUILTIN_CHECK_NAMES


def test_parser_validates_against_shared_check_names() -> None:
    """The parser derives its valid-check sets from the authoritative module
    rather than re-listing them (the drift the shared source removes)."""
    from mbt.parsing import project_parser
    from mbt.quality.check_names import BUILTIN_CHECK_NAMES, SCORING_CHECK_NAMES

    assert project_parser._BUILTIN_CHECKS is BUILTIN_CHECK_NAMES
    assert project_parser._SCORING_CHECKS is SCORING_CHECK_NAMES
