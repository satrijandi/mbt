"""The in-memory DataAdapter's own paths (A-1's second adapter for the seam).

``DataAdapterCompliance`` covers what every engine must agree on; these cover
what is this engine's own - its filter subset, its snapshot digest, and the
errors it raises.
"""

from pathlib import Path
from typing import Any

import pyarrow as pa
import pytest
from mbt_testing import InMemoryDataAdapter, InMemoryDataError

from mbt_adapter_base.compliance import RecordingEvents, tiny_source_rows
from mbt_adapter_base.compliance.suite import ComplianceBuildContext, _compliance_node
from mbt_adapter_base.interchange import DatasetLocator
from mbt_adapter_base.specs import ScoringInputSpec, ScoringOutputSpec, SourceTable

UID = "source.compliance.rows"


def _adapter(root: Path, rows: pa.Table | None = None) -> InMemoryDataAdapter:
    return InMemoryDataAdapter(
        {"root": str(root), "tables": {"rows": rows if rows is not None else tiny_source_rows(50)}}
    )


def _source() -> SourceTable:
    return SourceTable(name="rows", identifier="rows")


def _ctx(root: Path, *, windows: dict[str, Any] | None = None, fraction: float = 1.0) -> Any:
    source = _source()
    return ComplianceBuildContext(
        node=_compliance_node("scoring.compliance.batch", "scoring"),
        source=source,
        source_tables={UID: source},
        resolved_windows=windows or {},
        sample_fraction=fraction,
        deep_snapshot=False,
        output_dir=root / "out",
        events=RecordingEvents(),
    )


def test_add_table_registers_a_relation_after_construction(tmp_path: Path) -> None:
    adapter = InMemoryDataAdapter({"root": str(tmp_path)})
    adapter.add_table("late", tiny_source_rows(5))
    assert adapter.count_source_duplicates(SourceTable(name="late", identifier="late"), ["ts"]) >= 0


def test_an_unregistered_source_says_how_to_register_one(tmp_path: Path) -> None:
    adapter = InMemoryDataAdapter({"root": str(tmp_path)})
    with pytest.raises(InMemoryDataError, match="no in-memory table registered"):
        adapter.snapshot_id(SourceTable(name="absent", identifier="absent"))


def test_the_snapshot_digest_follows_the_content(tmp_path: Path) -> None:
    """An in-memory table has no mtime to be fooled by, so deep and shallow agree."""
    adapter = _adapter(tmp_path)
    first = adapter.snapshot_id(_source())
    assert first == adapter.snapshot_id(_source(), deep=True)
    adapter.add_table("rows", tiny_source_rows(51))
    assert adapter.snapshot_id(_source()) != first


def test_a_drifted_source_under_a_pinned_manifest_is_an_error(tmp_path: Path) -> None:
    adapter = _adapter(tmp_path)
    ctx = _ctx(tmp_path)
    ctx.node.snapshot_id = "sha256:something-else"
    ctx.source_tables = {UID: _source()}
    with pytest.raises(InMemoryDataError, match="source data changed under the pinned manifest"):
        adapter.verify_snapshot(ctx)


def test_no_pin_means_no_snapshot_check(tmp_path: Path) -> None:
    adapter = _adapter(tmp_path)
    assert adapter.verify_snapshot(_ctx(tmp_path)) is None


def test_a_scoring_batch_honours_its_window(tmp_path: Path) -> None:
    adapter = _adapter(tmp_path)
    spec = ScoringInputSpec.model_validate(
        {"source": UID, "time_column": "ts", "sample_key": ["row_id"]}
    )
    ctx = _ctx(tmp_path, windows={"score": ("2026-01-01T00:00:00Z", "2026-01-10T00:00:00Z")})
    handle = adapter.build_scoring_input(spec, ctx)
    scored = handle.read("score")
    assert 0 < scored.num_rows < 50


def test_sampling_a_batch_without_a_key_is_refused(tmp_path: Path) -> None:
    adapter = _adapter(tmp_path)
    spec = ScoringInputSpec.model_validate({"source": UID})
    with pytest.raises(InMemoryDataError, match="stable row identity"):
        adapter.build_scoring_input(spec, _ctx(tmp_path, fraction=0.5))


def test_reopening_verifies_the_snapshot(tmp_path: Path) -> None:
    adapter = _adapter(tmp_path)
    spec = ScoringInputSpec.model_validate({"source": UID, "sample_key": ["row_id"]})
    handle = adapter.build_scoring_input(spec, _ctx(tmp_path))
    reopened = adapter.from_locator(handle.locator())
    assert reopened.read("score").num_rows == handle.read("score").num_rows

    drifted = DatasetLocator(
        adapter="inmemory", uri=handle.locator().uri, snapshot_id="sha256:moved"
    )
    with pytest.raises(InMemoryDataError, match="snapshot mismatch"):
        adapter.from_locator(drifted)


def test_reopening_a_missing_materialization_says_so(tmp_path: Path) -> None:
    adapter = _adapter(tmp_path)
    locator = DatasetLocator(adapter="inmemory", uri=f"file://{tmp_path / 'gone'}", snapshot_id="x")
    with pytest.raises(InMemoryDataError, match="no complete dataset materialization"):
        adapter.from_locator(locator)


def test_open_predictions_roots_under_the_predictions_root(tmp_path: Path) -> None:
    adapter = _adapter(tmp_path)
    store = adapter.open_predictions(ScoringOutputSpec.model_validate({"path": "scores"}))
    assert "scores" in str(store.root)


@pytest.mark.parametrize(
    ("clause", "kept"),
    [
        ("label = 1", 25),
        ("label != 1", 25),
        ("label <> 1", 25),
        ("row_id < 10", 10),
        ("row_id <= 9", 10),
        ("row_id > 39", 10),
        ("row_id >= 40", 10),
        ("feature > 8.0", 23),
    ],
)
def test_the_supported_filter_subset(tmp_path: Path, clause: str, kept: int) -> None:
    adapter = _adapter(tmp_path)
    spec = ScoringInputSpec.model_validate({"source": UID, "filters": [clause]})
    handle = adapter.build_scoring_input(spec, _ctx(tmp_path))
    assert handle.read("score").num_rows == kept


def test_a_filter_beyond_the_subset_says_to_use_a_real_engine(tmp_path: Path) -> None:
    """Deliberately narrow: pretending to evaluate arbitrary SQL would make the
    fake lie about what a real engine would do."""
    adapter = _adapter(tmp_path)
    spec = ScoringInputSpec.model_validate({"source": UID, "filters": ["row_id % 2 = 0"]})
    with pytest.raises(InMemoryDataError, match="cannot evaluate filter"):
        adapter.build_scoring_input(spec, _ctx(tmp_path))


def test_a_string_literal_filter_works(tmp_path: Path) -> None:
    rows = pa.table({"row_id": [1, 2, 3], "plan": ["a", "b", "a"], "label": [0, 1, 0]})
    adapter = _adapter(tmp_path, rows)
    spec = ScoringInputSpec.model_validate({"source": UID, "filters": ["plan = 'a'"]})
    assert adapter.build_scoring_input(spec, _ctx(tmp_path)).read("score").num_rows == 2


def test_a_boolean_literal_filter_works(tmp_path: Path) -> None:
    rows = pa.table({"row_id": [1, 2, 3], "active": [True, False, True], "label": [0, 1, 0]})
    adapter = _adapter(tmp_path, rows)
    spec = ScoringInputSpec.model_validate({"source": UID, "filters": ["active = true"]})
    assert adapter.build_scoring_input(spec, _ctx(tmp_path)).read("score").num_rows == 2


def test_duplicate_composite_keys_are_counted_ignoring_nulls(tmp_path: Path) -> None:
    rows = pa.table({"row_id": [1, 1, 2, None], "label": [0, 1, 0, 1]})
    adapter = _adapter(tmp_path, rows)
    assert adapter.count_source_duplicates(_source(), ["row_id"]) == 1
