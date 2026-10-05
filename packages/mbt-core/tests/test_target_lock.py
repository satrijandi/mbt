"""One writer per project target/ (FEEDBACK v6 B-1)."""

import json
import multiprocessing
from pathlib import Path

import pytest

from mbt.exceptions import StateError
from mbt.state.lock import LOCK_FILE, holds_target_lock, target_lock


def _hold_until_released(project: str, ready, release) -> None:  # pragma: no cover - child
    with target_lock(Path(project), "build"):
        ready.set()
        release.wait(30)


def test_a_second_writer_is_refused_and_told_who_holds_the_lock(tmp_path: Path) -> None:
    """Two builds in one project used to race: the loser died on a DuckDB file
    lock reported as a filter problem, or both ran and the last writer's
    manifest won. Now the second one is refused, naming the first."""
    context = multiprocessing.get_context("spawn")
    ready, release = context.Event(), context.Event()
    holder = context.Process(target=_hold_until_released, args=(str(tmp_path), ready, release))
    holder.start()
    try:
        assert ready.wait(30)
        with (
            pytest.raises(StateError, match=rf"pid {holder.pid} \(mbt build, started ") as info,
            target_lock(tmp_path, "score"),
        ):
            pass  # pragma: no cover - never entered
        assert "released the moment that process exits" in (info.value.hint or "")
    finally:
        release.set()
        holder.join(30)
    # ...and the lock went with the process: no stale lock to clear by hand
    with target_lock(tmp_path, "score"):
        record = json.loads((tmp_path / "target" / LOCK_FILE).read_text())
        assert record["command"] == "score"


def test_the_lock_is_reentrant_within_a_process(tmp_path: Path) -> None:
    with target_lock(tmp_path, "evaluate"), target_lock(tmp_path, "build"):
        pass
    with target_lock(tmp_path, "build"):  # fully released after the nesting
        pass


def test_an_unreadable_record_still_refuses(tmp_path: Path) -> None:
    import fcntl
    import os

    (tmp_path / "target").mkdir()
    path = tmp_path / "target" / LOCK_FILE
    path.write_text("not json")
    fd = os.open(path, os.O_RDWR)
    fcntl.flock(fd, fcntl.LOCK_EX)  # held by a separate open file description
    try:
        with (
            pytest.raises(StateError, match=r"writing .*: another process"),
            target_lock(tmp_path, "build"),
        ):
            pass  # pragma: no cover - never entered
    finally:
        os.close(fd)


def test_entry_points_hold_the_lock(tmp_path: Path) -> None:
    class Opts:
        project_dir = tmp_path
        command = "build"

    @holds_target_lock
    def entry(opts: Opts) -> str:
        return json.loads((tmp_path / "target" / LOCK_FILE).read_text())["command"]

    assert entry(Opts()) == "build"
