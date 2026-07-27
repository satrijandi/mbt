"""Atomic control-file writes: never leave a truncated file (R2-3)."""

import os
import stat
from pathlib import Path

import pytest

from mbt.artifacts import atomic

posix_only = pytest.mark.skipif(os.name != "posix", reason="POSIX permission bits")


def test_writes_text_and_leaves_no_temp_file(tmp_path: Path) -> None:
    target = tmp_path / "manifest.json"
    atomic.atomic_write_text(target, '{"ok": true}\n')
    assert target.read_text() == '{"ok": true}\n'
    # the sibling temp file was renamed into place, not left behind
    assert list(tmp_path.iterdir()) == [target]


def test_creates_missing_parent_directories(tmp_path: Path) -> None:
    target = tmp_path / "target" / "run_results.json"
    atomic.atomic_write_text(target, "payload")
    assert target.read_text() == "payload"


def test_overwrites_an_existing_file(tmp_path: Path) -> None:
    target = tmp_path / "manifest.json"
    target.write_text("old")
    atomic.atomic_write_text(target, "new")
    assert target.read_text() == "new"
    assert list(tmp_path.iterdir()) == [target]


def test_failed_replace_cleans_up_and_leaves_original_intact(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    target = tmp_path / "manifest.json"
    target.write_text("original complete json")

    def boom(src: object, dst: object) -> None:
        raise OSError("replace failed mid-swap")

    monkeypatch.setattr(atomic.os, "replace", boom)

    with pytest.raises(OSError, match="replace failed"):
        atomic.atomic_write_text(target, "half-written")

    # the reader never sees a truncated file: the original survives untouched,
    # and the abandoned temp file was cleaned up
    assert target.read_text() == "original complete json"
    assert list(tmp_path.iterdir()) == [target]


def test_fsyncs_data_before_replace_and_dir_after(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    """F13: a power loss cannot leave a present-but-empty control file - the
    temp's data is fsynced before the rename, and the parent directory after."""
    events: list[str] = []
    real_fsync = atomic.os.fsync
    real_replace = atomic.os.replace

    def rec_fsync(fd: int) -> None:
        events.append("fsync")
        real_fsync(fd)

    def rec_replace(src: object, dst: object) -> None:
        events.append("replace")
        real_replace(src, dst)

    monkeypatch.setattr(atomic.os, "fsync", rec_fsync)
    monkeypatch.setattr(atomic.os, "replace", rec_replace)

    target = tmp_path / "sub" / "manifest.json"
    atomic.atomic_write_text(target, '{"ok": true}\n')

    assert target.read_text() == '{"ok": true}\n'  # still writes correctly
    # data fsync, then the atomic rename, then the parent-directory fsync
    assert events == ["fsync", "replace", "fsync"]


@posix_only
@pytest.mark.parametrize("umask, expected", [(0o022, 0o644), (0o002, 0o664), (0o077, 0o600)])
def test_permissions_follow_the_umask_not_the_temp_file(
    tmp_path: Path, umask: int, expected: int
) -> None:
    """The control files are routinely written by one uid and read by another -
    a container writing a workspace its host user inspects, or one CI step
    handing artifacts to the next. `tempfile.mkstemp` hardcodes 0600 and
    `os.replace` carries that onto the destination, which made every
    manifest.json/run_results.json private to whoever wrote it.
    """
    previous = os.umask(umask)
    try:
        target = tmp_path / "run_results.json"
        atomic.atomic_write_text(target, '{"results": []}\n')
        assert stat.S_IMODE(target.stat().st_mode) == expected
    finally:
        os.umask(previous)


@posix_only
def test_rewriting_an_existing_file_repairs_private_permissions(tmp_path: Path) -> None:
    """A target left 0600 by the old behavior must not stay 0600 forever: the
    rename replaces the destination inode, so the next write heals it."""
    target = tmp_path / "manifest.json"
    target.write_text("old")
    target.chmod(0o600)

    previous = os.umask(0o022)
    try:
        atomic.atomic_write_text(target, "new")
    finally:
        os.umask(previous)

    assert stat.S_IMODE(target.stat().st_mode) == 0o644
