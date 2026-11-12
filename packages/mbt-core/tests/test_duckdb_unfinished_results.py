"""No check may leave a DuckDB result unfinished on a connection it reuses (issue #32).

The nightly run against newest versions found this the day ``duckdb`` reached 2.0:
``no_future_columns`` and ``freshness`` query once per split on one connection, and
2.0 refuses new work while a ``fetchone()`` result is open. These run every
connection-reusing check under the 2.0 rule, whatever DuckDB is installed.
"""

from datetime import datetime

import duckdb
import pyarrow as pa
import pytest
from duckdb_strict import StrictConnection, strict_connect

from mbt.contracts import DatasetSpec
from mbt.quality.checks import run_checks
from mbt.utils import first_row
from mbt_adapter_base.datasets import InMemoryDatasetHandle

WINDOWS = {
    "windows": {
        "train": ["2026-01-01T00:00:00Z", "2026-01-03T00:00:00Z"],
        "test": ["2026-01-03T00:00:00Z", "2026-01-05T00:00:00Z"],
    }
}


def handle() -> InMemoryDatasetHandle:
    """Two splits, so a check that queries per split queries on its connection twice."""

    def table(days: tuple[int, ...], first_id: int) -> pa.Table:
        return pa.table(
            {
                "user_id": list(range(first_id, first_id + 4)),
                "snapshot_date": [datetime(2026, 1, d) for d in days],
                "plan": ["basic", "basic", "pro", "pro"],  # independent of the label
                "churned": [0, 1, 0, 1],
            }
        )

    return InMemoryDatasetHandle(
        {"train": table((1, 1, 2, 2), 1), "test": table((3, 3, 4, 4), 5)},
        label_column="churned",
        time_column="snapshot_date",
    )


def spec(checks: list) -> DatasetSpec:
    return DatasetSpec.model_validate(
        {
            "name": "d",
            "source": "source('a', 'b')",
            "sample_key": "user_id",
            "label": {"column": "churned"},
            "split": {"time_column": "snapshot_date", "train": "-30d:-7d", "test": "-7d:now"},
            "checks": checks,
        }
    )


@pytest.mark.parametrize(
    "check",
    [
        {"not_null": {"columns": ["plan", "churned"]}},
        {"unique": {"columns": ["user_id"]}},
        {"accepted_values": {"column": "plan", "values": ["basic", "pro"]}},
        {"freshness": {"max_lag": "2d"}},
        "no_future_columns",
        "label_leakage_scan",
    ],
    ids=lambda check: check if isinstance(check, str) else next(iter(check)),
)
def test_a_check_that_reuses_a_connection_leaves_no_result_open(
    check: object, monkeypatch: pytest.MonkeyPatch
) -> None:
    monkeypatch.setattr(duckdb, "connect", strict_connect)
    name = check if isinstance(check, str) else next(iter(check))
    results = {r.name: r for r in run_checks(spec([check]), handle(), WINDOWS, resource="d")}
    # the always-on scans run beside it, so every connection-reusing check is covered
    for result in results.values():
        assert "Resource In Use" not in result.message, result.message
    assert results[name].passed, results[name].message


def test_the_strict_connection_really_refuses_work_on_an_unfinished_result() -> None:
    """The guard above is only worth something if it would have caught the bug."""
    con = StrictConnection(duckdb.connect())
    con.register("t", pa.table({"x": [1, 2, 3]}))
    con.execute("SELECT max(x) FROM t").fetchone()  # the old pattern: result left open
    with pytest.raises(RuntimeError, match="unfinished result"):
        con.execute("SELECT 1")
    with pytest.raises(RuntimeError, match="unfinished result"):
        con.register("u", pa.table({"x": [1]}))


def test_first_row_fully_consumes_the_result_it_reads() -> None:
    con = StrictConnection(duckdb.connect())
    assert first_row(con.execute("SELECT 7, 8")) == (7, 8)
    assert con.unfinished is None
    assert con.execute("SELECT 1").fetchall() == [(1,)]  # the connection is free again


def test_first_row_of_an_empty_result_is_none() -> None:
    con = duckdb.connect()
    assert first_row(con.execute("SELECT 1 WHERE false")) is None


def test_strict_results_fetched_whole_by_other_means_are_finished() -> None:
    con = StrictConnection(duckdb.connect())
    whole = con.execute("SELECT 5")
    # renamed to_arrow_table in DuckDB 2.0; either way it consumes the whole result
    fetch = getattr(whole, "to_arrow_table", None) or whole.fetch_arrow_table
    assert fetch().num_rows == 1
    assert con.unfinished is None
    result = con.execute("SELECT 6")
    assert result.fetchone() == (6,)
    assert con.unfinished is result  # a row came back, so the result is still open
    result.fetchall()  # fetchall finishes what fetchone opened
    assert con.unfinished is None
    assert con.execute("SELECT 7").fetchall() == [(7,)]
