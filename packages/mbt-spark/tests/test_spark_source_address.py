"""Which address the Spark adapter reads a source table by.

Deliberately NOT under the e2e marker: this is pure decision logic and needs
no JVM, and the bug it pins escaped precisely because every mbt-spark test
was JVM-gated and therefore invisible to the fast tier. The showcase added a
warehouse ``identifier:`` to a shared ``sources.yml``, which silently
redirected the Spark lake plane at catalog tables that did not exist, and
only the nightly live run noticed.
"""

from pathlib import Path
from typing import Any

import pytest
import yaml
from mbt_spark.data import SparkAdapterError, SparkDataAdapter, resolve_source_address


class FakeSource:
    """Minimal SourceTableLike. The real SourceTable forbids declaring
    neither address, so the protocol is what lets us cover that branch."""

    def __init__(
        self, name: str = "t", path: str | None = None, identifier: str | None = None
    ) -> None:
        self.name = name
        self.path = path
        self.identifier = identifier
        self.format = "parquet"


def test_a_single_declared_address_is_used_whatever_the_config_says() -> None:
    # The config is a tie-breaker, not an override: a mixed project where some
    # tables live in the catalog and others in files must still resolve both.
    path_only = FakeSource(path="t/*.parquet")
    identifier_only = FakeSource(identifier="db.schema.T")

    for configured in (None, "path", "identifier"):
        assert resolve_source_address(path_only, configured) == "path"  # type: ignore[arg-type]
        assert resolve_source_address(identifier_only, configured) == "identifier"  # type: ignore[arg-type]


def test_declaring_both_without_config_is_a_config_error_not_a_guess() -> None:
    both = FakeSource(name="churn_outcomes", path="churn/*.parquet", identifier="MBT_CHURN")

    with pytest.raises(SparkAdapterError) as excinfo:
        resolve_source_address(both, None)

    message = str(excinfo.value)
    assert "churn_outcomes" in message
    # The hint must name the knob; an operator hitting this mid-build should
    # not have to read the adapter source to find out how to resolve it.
    assert "source_address" in message


def test_config_picks_the_address_when_both_are_declared() -> None:
    both = FakeSource(path="churn/*.parquet", identifier="MBT_CHURN")

    assert resolve_source_address(both, "path") == "path"
    assert resolve_source_address(both, "identifier") == "identifier"


def test_a_table_declaring_no_address_is_rejected() -> None:
    with pytest.raises(SparkAdapterError, match="needs 'path' or 'identifier'"):
        resolve_source_address(FakeSource(), None)


def test_adapter_rejects_a_nonsense_source_address_at_construction() -> None:
    # Fail on the profile, not later on the first read of an ambiguous table.
    with pytest.raises(SparkAdapterError, match="source_address"):
        SparkDataAdapter({"source_address": "catalog"})


@pytest.mark.parametrize("configured", [None, "path", "identifier"])
def test_adapter_threads_its_config_into_the_decision(configured: Any) -> None:
    adapter = SparkDataAdapter({"source_address": configured} if configured else {})
    assert adapter.source_address == configured

    both = FakeSource(path="churn/*.parquet", identifier="MBT_CHURN")
    if configured is None:
        with pytest.raises(SparkAdapterError):
            adapter._address(both)
    else:
        assert adapter._address(both) == configured


class FakeFrame:
    def __init__(self, uris: list[str]) -> None:
        self._uris = uris

    def inputFiles(self) -> list[str]:  # pyspark's spelling
        return self._uris


def test_snapshot_pinning_follows_the_address_the_data_is_read_by(
    monkeypatch: pytest.MonkeyPatch, tmp_path: Any
) -> None:
    """Pinning and reading must never diverge: hashing the parquet files of a
    table that is then served from the catalog pins a snapshot of data the run
    never touches, so the pin cannot detect the data changing under it.

    ``_read`` is stubbed rather than exercised - the point is which branch
    ``snapshot_id`` takes, and a real session would need a JVM.
    """
    (tmp_path / "part-000.parquet").write_bytes(b"local bytes")
    both = FakeSource(path=str(tmp_path / "*.parquet"), identifier="db.T")

    reads: list[str] = []

    def fake_read(source: Any) -> FakeFrame:
        reads.append(source.name)
        return FakeFrame(["s3://catalog/db/T/part-0.parquet"])

    by_identifier = SparkDataAdapter({"source_address": "identifier"})
    monkeypatch.setattr(by_identifier, "_read", fake_read)
    catalog_pin = by_identifier.snapshot_id(both)
    assert reads == ["t"], "identifier addressing must pin the catalog listing"

    by_path = SparkDataAdapter({"source_address": "path"})
    monkeypatch.setattr(by_path, "_read", fake_read)
    path_pin = by_path.snapshot_id(both)
    assert reads == ["t"], "path addressing must not need a session at all"

    assert catalog_pin != path_pin


def test_ambiguity_surfaces_at_pinning_time_which_is_compile_time() -> None:
    # Snapshot pinning runs during compile, so an ambiguous table fails the
    # compile rather than a job several minutes in.
    adapter = SparkDataAdapter({})
    both = FakeSource(path="churn/*.parquet", identifier="MBT_CHURN")

    with pytest.raises(SparkAdapterError, match="source_address"):
        adapter.snapshot_id(both)


# -- the showcase project, which is what actually broke -------------------------
#
# examples/showcase declares BOTH addresses on all 12 tables so one project can
# serve a lake plane and a warehouse plane. Nothing held the spark half of that
# bargain: `identifier:` was added for the Snowflake target, every spark target
# silently started resolving catalog tables that do not exist, push CI stayed
# green, and the nightly live tier was red for a day. These are static reads of
# the committed project, so the next time it happens the fast suite says so.

SHOWCASE = Path(__file__).resolve().parents[3] / "examples" / "showcase" / "project"


def _showcase_sources() -> list[dict[str, Any]]:
    doc = yaml.safe_load((SHOWCASE / "sources.yml").read_text())
    tables = doc["sources"][0]["tables"]
    assert isinstance(tables, list) and tables
    return tables


def _showcase_spark_targets() -> dict[str, dict[str, Any]]:
    # Quoted `{{ env_var(...) }}` scalars and `<<:` merge keys both survive
    # safe_load, so the profile parses without rendering jinja.
    doc = yaml.safe_load((SHOWCASE / "profiles.yml").read_text())
    (profile,) = doc.values()
    return {
        name: output["data"].get("config", {})
        for name, output in profile["outputs"].items()
        if output.get("data", {}).get("adapter") == "spark"
    }


def test_every_showcase_spark_target_resolves_every_source() -> None:
    targets = _showcase_spark_targets()
    assert targets, "the showcase has no spark targets left; this guard is stale"

    for target, config in targets.items():
        adapter = SparkDataAdapter(config)
        for table in _showcase_sources():
            source = FakeSource(
                name=table["name"], path=table.get("path"), identifier=table.get("identifier")
            )
            # Raises on an ambiguous table, which is the whole point.
            address = adapter._address(source)
            assert address == "path", (
                f"target {target!r} would read source {table['name']!r} by "
                f"{address}; the showcase's lake planes read the lake"
            )
