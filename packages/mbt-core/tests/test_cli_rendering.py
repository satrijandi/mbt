"""CLI results-table rendering helpers (human-facing output polish)."""

from mbt.cli.common import _format_metric


def test_count_metrics_render_as_integers() -> None:
    # rows_scored and other whole-number metrics must not show spurious
    # decimals (a count is not "460.0000").
    assert _format_metric(460.0) == "460"
    assert _format_metric(0.0) == "0"
    assert _format_metric(1.0) == "1"


def test_fractional_metrics_keep_four_decimals() -> None:
    assert _format_metric(0.31613495) == "0.3161"
    assert _format_metric(0.6817777) == "0.6818"  # rounds at the 4th decimal
    assert _format_metric(0.5) == "0.5000"


def test_errored_node_detail_is_first_line_only() -> None:
    """An errored node's message is str(MbtError) (multi-line: message, then
    resource and hint). The table detail keeps only the first line - resource
    is already the node column, and the hint is in the event log."""
    from types import SimpleNamespace

    from mbt.artifacts.run_results import NodeResult, RunResults, RunResultsMetadata
    from mbt.cli.common import out_console, render_results_table

    results = RunResults(
        metadata=RunResultsMetadata(
            run_id="r",
            mbt_version="0",
            target="prod",
            manifest_hash="h",
            anchor="a",
            started_at="s",
            command="score",
        ),
        results=[
            NodeResult(
                unique_id="scoring.p.s",
                status="error",
                message="no champion available\n  resource: scoring.p.s\n  hint: promote it",
            )
        ],
    )
    ctx = SimpleNamespace(quiet=False, log_format="text")
    with out_console.capture() as capture:
        render_results_table(results, ctx)  # type: ignore[arg-type]
    out = capture.get()
    assert "champion" in out  # the error text survives
    assert "resource:" not in out and "hint:" not in out  # redundant tail dropped
