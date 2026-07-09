"""Built-in dataset check tests, positive and negative (S4-06)."""

from datetime import datetime

import pyarrow as pa

from mbt.contracts import DatasetSpec
from mbt.quality.checks import run_checks
from mbt_adapter_base.datasets import InMemoryDatasetHandle


def _handle(**overrides) -> InMemoryDatasetHandle:
    base = {
        "user_id": [1, 2, 3, 4],
        "snapshot_date": [datetime(2026, 1, d) for d in (1, 2, 3, 4)],
        "feature": [1.0, 2.0, 3.0, 4.0],
        "churned": [0, 1, 0, 1],
    }
    base.update(overrides)
    table = pa.table(base)
    return InMemoryDatasetHandle(
        {"train": table, "test": table}, label_column="churned", time_column="snapshot_date"
    )


def _spec(checks) -> DatasetSpec:
    return DatasetSpec.model_validate(
        {
            "name": "d",
            "source": "source('a', 'b')",
            "label": {"column": "churned"},
            "split": {"time_column": "snapshot_date", "train": "-30d:-7d", "test": "-7d:now"},
            "checks": checks,
        }
    )


WINDOWS = {
    "windows": {
        "train": ["2026-01-01T00:00:00Z", "2026-01-03T00:00:00Z"],
        "test": ["2026-01-03T00:00:00Z", "2026-01-05T00:00:00Z"],
    }
}


def _run(checks, handle):
    return {r.name: r for r in run_checks(_spec(checks), handle, WINDOWS, resource="d")}


def test_not_null_passes_and_fails() -> None:
    ok = _run([{"not_null": {"columns": ["churned"]}}], _handle())
    assert ok["not_null"].passed
    bad = _run(
        [{"not_null": {"columns": ["feature"]}}],
        _handle(feature=[1.0, None, 3.0, None]),
    )
    assert not bad["not_null"].passed
    assert "2 null" in bad["not_null"].message


def test_schema_check() -> None:
    ok = _run([{"schema": {"columns": ["churned", "feature"]}}], _handle())
    assert ok["schema"].passed
    missing = _run([{"schema": {"columns": ["nonexistent"]}}], _handle())
    assert not missing["schema"].passed
    typed = _run([{"schema": {"columns": {"feature": "double"}}}], _handle())
    assert typed["schema"].passed
    wrong_type = _run([{"schema": {"columns": {"feature": "int64"}}}], _handle())
    assert not wrong_type["schema"].passed


def _split_handle(
    train_event_at: list[datetime], test_event_at: list[datetime]
) -> InMemoryDatasetHandle:
    """Window-realistic splits: train inside Jan 1-3, test inside Jan 3-5."""

    def table(dates: list[datetime], events: list[datetime]) -> pa.Table:
        n = len(dates)
        return pa.table(
            {
                "user_id": list(range(n)),
                "snapshot_date": dates,
                "last_event_at": events,
                "feature": [float(i) for i in range(n)],
                "churned": [i % 2 for i in range(n)],
            }
        )

    train = table([datetime(2026, 1, 1), datetime(2026, 1, 2)], train_event_at)
    test = table([datetime(2026, 1, 3), datetime(2026, 1, 4)], test_event_at)
    return InMemoryDatasetHandle(
        {"train": train, "test": test}, label_column="churned", time_column="snapshot_date"
    )


def test_no_future_columns() -> None:
    within = _split_handle(
        [datetime(2026, 1, 1), datetime(2026, 1, 2)],
        [datetime(2026, 1, 3), datetime(2026, 1, 4)],
    )
    assert _run(["no_future_columns"], within)["no_future_columns"].passed

    # absolutely-future values (beyond every window) still fail
    future = _split_handle(
        [datetime(2026, 1, 1), datetime(2026, 1, 2)],
        [datetime(2026, 1, 3), datetime(2026, 9, 9)],
    )
    bad = _run(["no_future_columns"], future)
    assert not bad["no_future_columns"].passed
    assert "last_event_at" in bad["no_future_columns"].message


def test_no_future_columns_catches_train_test_overlap() -> None:
    """A train row carrying a timestamp inside the TEST window is temporal
    leakage; the check compares each split against its OWN window end
    (section 3.6: it previously caught only absolute-future values)."""
    overlap = _split_handle(
        [datetime(2026, 1, 1), datetime(2026, 1, 4)],  # Jan 4 is in the test window
        [datetime(2026, 1, 3), datetime(2026, 1, 4)],
    )
    bad = _run(["no_future_columns"], overlap)
    assert not bad["no_future_columns"].passed
    assert "train.last_event_at" in bad["no_future_columns"].message
    assert "temporal leakage" in bad["no_future_columns"].message


def test_label_leakage_scan() -> None:
    ok = _run(["label_leakage_scan"], _handle())
    assert ok["label_leakage_scan"].passed
    # a feature perfectly correlated with the label must be flagged
    leaky = _handle(feature=[0.0, 1.0, 0.0, 1.0])
    bad = _run(["label_leakage_scan"], leaky)
    assert not bad["label_leakage_scan"].passed
    assert "feature" in bad["label_leakage_scan"].message


def test_label_leakage_scan_runs_by_default() -> None:
    """The scan needs no declaration: leakage guards are on by default."""
    assert _run([], _handle())["label_leakage_scan"].passed
    leaky = _handle(feature=[0.0, 1.0, 0.0, 1.0])
    assert not _run([], leaky)["label_leakage_scan"].passed


def test_label_leakage_scan_warn_band_exclude_and_opt_out() -> None:
    # warn band: recorded in the message and warned, but passing
    tuned = _run(
        [{"label_leakage_scan": {"warn_abs_correlation": 0.3, "max_abs_correlation": 0.9}}],
        _handle(),
    )
    result = tuned["label_leakage_scan"]
    assert result.passed
    assert "warn band" in result.message and "feature" in result.message

    # reviewed columns can be excluded
    leaky = _handle(feature=[0.0, 1.0, 0.0, 1.0])
    excluded = _run([{"label_leakage_scan": {"exclude": ["feature"]}}], leaky)
    assert excluded["label_leakage_scan"].passed

    # opting out is explicit and visible, never silent
    disabled = _run([{"label_leakage_scan": {"enabled": False}}], leaky)
    assert disabled["label_leakage_scan"].passed
    assert disabled["label_leakage_scan"].message == "check disabled"


def test_label_leakage_scan_catches_categorical_leaks() -> None:
    """String columns are screened with Cramér's V on the same bar as
    numeric |corr| (section 3.6: the scan was numeric-only, so a status
    string literally encoding the label sailed through)."""
    # a categorical that perfectly encodes the label must fail (V = 1)
    leaky = _handle(status=["active", "churned", "active", "churned"])
    bad = _run([], leaky)["label_leakage_scan"]
    assert not bad.passed
    assert "status (V=1.000)" in bad.message

    # an independent categorical passes (V = 0 for this arrangement)
    ok = _run([], _handle(plan=["basic", "basic", "pro", "pro"]))
    assert ok["label_leakage_scan"].passed

    # warn band: strong-but-not-failing association is recorded and passes
    labels = [0] * 100 + [1] * 100
    status = ["no" if y == 0 else "yes" for y in labels]
    for i in (0, 20, 40, 60, 80, 100, 120, 140, 160, 180):  # flip 10 of 200 -> V = 0.9
        status[i] = "yes" if status[i] == "no" else "no"
    warn = _run(
        [],
        _handle(
            user_id=list(range(200)),
            snapshot_date=[datetime(2026, 1, 1 + i % 4) for i in range(200)],
            feature=[float(i % 7) for i in range(200)],
            churned=labels,
            status=status,
        ),
    )["label_leakage_scan"]
    assert warn.passed
    assert "warn band" in warn.message and "status (V=0.9" in warn.message

    # reviewed categorical columns can be excluded like numeric ones
    excluded = _run([{"label_leakage_scan": {"exclude": ["status"]}}], leaky)
    assert excluded["label_leakage_scan"].passed


def test_label_leakage_scan_skips_degenerate_and_id_like_strings() -> None:
    # a single-level column carries no association signal
    ok = _run([], _handle(constant=["same", "same", "same", "same"]))
    assert ok["label_leakage_scan"].passed
    # unique-per-row strings (IDs) would saturate V spuriously; skipped
    ok = _run([], _handle(email=[f"user{i}@x.com" for i in range(4)]))
    assert ok["label_leakage_scan"].passed


def test_class_balance_report_never_fails() -> None:
    result = _run(["class_balance_report"], _handle())
    assert result["class_balance_report"].passed
    assert "label balance" in result["class_balance_report"].message
