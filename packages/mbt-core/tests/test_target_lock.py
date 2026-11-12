"""One writer per target directory (FEEDBACK v6 B-1), one builder per dataset key (ADR-32)."""

import json
import multiprocessing
from pathlib import Path

import pytest

from mbt.exceptions import StateError
from mbt.state.layout import TargetLayout
from mbt.state.lock import LOCK_FILE, holds_target_lock, materialization_lock, target_lock


def _hold_until_released(target: str, ready, release) -> None:  # pragma: no cover - child
    with target_lock(Path(target), "build"):
        ready.set()
        release.wait(30)


def test_a_second_writer_is_refused_and_told_who_holds_the_lock(tmp_path: Path) -> None:
    """Two builds in one project used to race: the loser died on a DuckDB file
    lock reported as a filter problem, or both ran and the last writer's
    manifest won. Now the second one is refused, naming the first."""
    context = multiprocessing.get_context("spawn")
    ready, release = context.Event(), context.Event()
    holder = context.Process(
        target=_hold_until_released, args=(str(tmp_path / "target"), ready, release)
    )
    holder.start()
    try:
        assert ready.wait(30)
        with (
            pytest.raises(StateError, match=rf"pid {holder.pid} \(mbt build, started ") as info,
            target_lock(tmp_path / "target", "score"),
        ):
            pass  # pragma: no cover - never entered
        assert "released the moment that process exits" in (info.value.hint or "")
        assert "--target-path" in (info.value.hint or "")
    finally:
        release.set()
        holder.join(30)
    # ...and the lock went with the process: no stale lock to clear by hand
    with target_lock(tmp_path / "target", "score"):
        record = json.loads((tmp_path / "target" / LOCK_FILE).read_text())
        assert record["command"] == "score"


def test_the_lock_is_reentrant_within_a_process(tmp_path: Path) -> None:
    target = tmp_path / "target"
    with target_lock(target, "evaluate"), target_lock(target, "build"):
        pass
    with target_lock(target, "build"):  # fully released after the nesting
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
            target_lock(tmp_path / "target", "build"),
        ):
            pass  # pragma: no cover - never entered
    finally:
        os.close(fd)


def test_entry_points_hold_the_lock(tmp_path: Path) -> None:
    class Opts:
        layout = TargetLayout.for_project(tmp_path)
        command = "build"

    @holds_target_lock
    def entry(opts: Opts) -> str:
        return json.loads((tmp_path / "target" / LOCK_FILE).read_text())["command"]

    assert entry(Opts()) == "build"


def test_a_target_path_elsewhere_has_its_own_lock(tmp_path: Path) -> None:
    """ADR-32: two tasks that each own a target directory do not contend."""
    first, second = tmp_path / "task-a", tmp_path / "task-b"
    with target_lock(first, "build"), target_lock(second, "build"):
        assert (first / LOCK_FILE).is_file() and (second / LOCK_FILE).is_file()


def test_a_second_builder_of_one_key_waits_then_proceeds(tmp_path: Path) -> None:
    """The cache can be shared between invocations, so the process that loses
    the race for a key waits (and says so) instead of emptying the directory
    the winner is filling."""
    import threading

    key_dir = tmp_path / "datasets" / "panel" / "abc123"
    holding, may_release = threading.Event(), threading.Event()
    order: list[str] = []

    def winner() -> None:
        with materialization_lock(key_dir):
            holding.set()
            may_release.wait(30)
            order.append("winner done")

    thread = threading.Thread(target=winner)
    thread.start()
    try:
        assert holding.wait(30)
        waited: list[bool] = []

        def on_wait() -> None:
            waited.append(True)
            may_release.set()  # the winner finishes only once we are known to be waiting

        with materialization_lock(key_dir, on_wait=on_wait):
            order.append("loser entered")
    finally:
        may_release.set()
        thread.join(30)
    assert waited == [True]
    assert order == ["winner done", "loser entered"]


def test_an_uncontended_key_does_not_call_on_wait(tmp_path: Path) -> None:
    def never() -> None:  # pragma: no cover - failing is the point
        raise AssertionError("nobody held the lock")

    with materialization_lock(tmp_path / "datasets" / "panel" / "k", on_wait=never):
        pass
    # the lock file sits BESIDE the key directory, which a build empties first
    assert (tmp_path / "datasets" / "panel" / "k.lock").is_file()
    assert not (tmp_path / "datasets" / "panel" / "k").exists()
