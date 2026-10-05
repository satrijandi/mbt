"""One writer per project ``target/`` at a time (FEEDBACK v6 B-1).

Every command that writes ``target/`` - the manifest, ``run_results.json``,
dataset materializations, run logs - holds an advisory lock on
``target/.mbt.lock`` for its whole lifetime. Without it two ``mbt build``s in
one project directory (cron and a developer, two Airflow tasks on one worker)
raced: if they compiled in the same second they picked the same dataset
directory and one died on a DuckDB file lock, reported as a filter problem; a
second apart, both ran and the last writer's manifest and run results won.

The lock is an OS ``flock``, so it frees itself when the holding process exits,
however it exits - a crashed or killed run never leaves a stale lock behind.
The file carries the holder's pid and command purely so the error can name
them. It is reentrant within a process, because ``mbt evaluate`` and friends
reach one locked entry point through another.
"""

import json
import os
import sys
from collections.abc import Callable, Iterator
from contextlib import contextmanager
from datetime import UTC, datetime
from functools import wraps
from pathlib import Path
from typing import Any, ParamSpec, TypeVar

from mbt.exceptions import StateError

LOCK_FILE = ".mbt.lock"

#: target dir -> (fd, depth) for the locks this process holds.
_HELD: dict[Path, tuple[int, int]] = {}

P = ParamSpec("P")
R = TypeVar("R")


def _try_lock(fd: int) -> bool:
    if sys.platform == "win32":  # pragma: no cover - CI and the docs target POSIX
        import msvcrt

        try:
            msvcrt.locking(fd, msvcrt.LK_NBLCK, 1)
        except OSError:
            return False
        return True
    import fcntl

    try:
        fcntl.flock(fd, fcntl.LOCK_EX | fcntl.LOCK_NB)
    except BlockingIOError:
        return False
    return True


def _holder(path: Path) -> str:
    """Who holds the lock, as the holder recorded it.

    The holder writes its record just AFTER taking the lock, so a process that
    loses the race by microseconds can find the file empty; it looks again for
    a moment before settling for "another process".
    """
    import time

    for _ in range(25):
        try:
            info = json.loads(path.read_text())
            return f"pid {info['pid']} (mbt {info['command']}, started {info['started_at']})"
        except (OSError, ValueError, KeyError, TypeError):
            time.sleep(0.02)
    return "another process"


@contextmanager
def target_lock(project_dir: Path, command: str) -> Iterator[None]:
    """Hold ``<project>/target/.mbt.lock`` for the duration of the block."""
    target = (project_dir / "target").resolve()
    held = _HELD.get(target)
    if held is not None:
        fd, depth = held
        _HELD[target] = (fd, depth + 1)
        try:
            yield
        finally:
            _HELD[target] = (fd, depth)
        return

    target.mkdir(parents=True, exist_ok=True)
    path = target / LOCK_FILE
    fd = os.open(path, os.O_RDWR | os.O_CREAT, 0o644)
    try:
        if not _try_lock(fd):
            raise StateError(
                f"another mbt command is already writing {target}: {_holder(path)}",
                hint="wait for it to finish, or run this one from a separate checkout; "
                "the lock is released the moment that process exits, so a crashed run "
                "never leaves it behind",
            )
        record = {
            "pid": os.getpid(),
            "command": command,
            "started_at": datetime.now(tz=UTC).isoformat(timespec="seconds"),
        }
        os.ftruncate(fd, 0)
        os.lseek(fd, 0, os.SEEK_SET)
        os.write(fd, json.dumps(record).encode())
        _HELD[target] = (fd, 1)
        try:
            yield
        finally:
            del _HELD[target]
    finally:
        os.close(fd)  # closing the descriptor releases the lock


def holds_target_lock(fn: Callable[P, R]) -> Callable[P, R]:
    """Run an orchestrator entry point under its project's target lock.

    The wrapped function's first argument is the ``InvocationOptions``.
    """

    @wraps(fn)
    def wrapper(*args: P.args, **kwargs: P.kwargs) -> R:
        opts: Any = args[0]
        with target_lock(opts.project_dir, opts.command):
            return fn(*args, **kwargs)

    return wrapper


__all__ = ["LOCK_FILE", "holds_target_lock", "target_lock"]
