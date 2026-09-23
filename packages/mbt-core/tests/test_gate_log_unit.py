"""Counting how often one test window has arbitrated a gate (D-3).

ADR-30's after-test window is the control and is already present; this counts,
warns, and carries no policy - the way the backtest std is reported next to the
backtest mean, so a reader knows how much to trust a number that otherwise
looks unconditional.
"""

from pathlib import Path

from mbt.state.gate_log import (
    GATE_LOG_FILE,
    REUSE_WARN_THRESHOLD,
    read_log,
    record_evaluation,
    reuse_warning,
)

DATASET = "dataset.demo.panel"
WINDOW = ("2026-05-01T00:00:00Z", "2026-06-01T00:00:00Z")


def test_each_evaluation_increments_the_pairs_count(tmp_path: Path) -> None:
    assert record_evaluation(tmp_path, DATASET, WINDOW).count == 1
    assert record_evaluation(tmp_path, DATASET, WINDOW).count == 2
    assert (tmp_path / "target" / GATE_LOG_FILE).is_file()


def test_a_different_window_counts_separately(tmp_path: Path) -> None:
    """Moving the test window forward starts a fresh count, which is the point:
    a NEW window has not been selected against."""
    record_evaluation(tmp_path, DATASET, WINDOW)
    record_evaluation(tmp_path, DATASET, WINDOW)
    moved = ("2026-06-01T00:00:00Z", "2026-07-01T00:00:00Z")
    assert record_evaluation(tmp_path, DATASET, moved).count == 1
    assert record_evaluation(tmp_path, DATASET, WINDOW).count == 3


def test_a_different_dataset_counts_separately(tmp_path: Path) -> None:
    record_evaluation(tmp_path, DATASET, WINDOW)
    assert record_evaluation(tmp_path, "dataset.demo.other", WINDOW).count == 1


def test_no_resolved_window_records_nothing(tmp_path: Path) -> None:
    """A random split has no test window, so there is none to have reused."""
    assert record_evaluation(tmp_path, DATASET, None) is None
    assert read_log(tmp_path) == {}


def test_the_threshold_is_what_turns_a_count_into_a_warning(tmp_path: Path) -> None:
    use = None
    for _ in range(REUSE_WARN_THRESHOLD):
        use = record_evaluation(tmp_path, DATASET, WINDOW)
    assert use is not None and use.overused
    message = reuse_warning(use)
    assert f"judged {REUSE_WARN_THRESHOLD} candidates" in message
    assert "no longer fully held out" in message

    # one below the bar stays quiet
    fresh = tmp_path / "fresh"
    for _ in range(REUSE_WARN_THRESHOLD - 1):
        below = record_evaluation(fresh, DATASET, WINDOW)
    assert below is not None and not below.overused


def test_an_unreadable_log_reads_as_empty_rather_than_failing(tmp_path: Path) -> None:
    """The count is a signal, not a contract: failing a build over its
    bookkeeping would be worse than losing the count."""
    (tmp_path / "target").mkdir()
    (tmp_path / "target" / GATE_LOG_FILE).write_text("{not json")
    assert read_log(tmp_path) == {}
    assert record_evaluation(tmp_path, DATASET, WINDOW).count == 1


def test_a_non_mapping_log_reads_as_empty(tmp_path: Path) -> None:
    (tmp_path / "target").mkdir()
    (tmp_path / "target" / GATE_LOG_FILE).write_text("[1, 2, 3]")
    assert read_log(tmp_path) == {}


def test_an_unwritable_log_never_fails_the_run(tmp_path: Path, monkeypatch) -> None:
    """The count is a signal, not a contract."""
    import pathlib

    def boom(self, *args, **kwargs):
        raise OSError("read-only filesystem")

    monkeypatch.setattr(pathlib.Path, "write_text", boom)
    use = record_evaluation(tmp_path, DATASET, WINDOW)
    assert use is not None and use.count == 1  # counted in memory, not persisted
