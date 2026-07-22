"""Atomic single-file writes for the control files (manifest, run_results).

A killed ``mbt run``, or a ``--state``/``--manifest``/``docs`` read racing an
in-progress write, must never observe a truncated file: the reader would then
hard-fail with ``invalid JSON in manifest`` until the file is regenerated - a
real CI flake vector where these commands run back-to-back (R2-3).

Writing to a sibling temp file and ``os.replace``-ing it into place makes the
swap atomic on POSIX and Windows: a concurrent reader sees either the old
complete file or the new complete file, never a partial one. The temp file
lives in the target's own directory so ``os.replace`` is a rename within one
filesystem; a cross-filesystem temp would degrade to a non-atomic copy and
defeat the purpose.
"""

import os
import tempfile
from pathlib import Path


def _fsync_dir(directory: Path) -> None:
    """fsync a directory so a rename into it survives a power loss - the file
    fsync only persists the file's own data blocks, not the parent's record of
    the new name. Directory fsync is a POSIX facility; on other platforms the OS
    handles rename durability and this is a no-op."""
    if os.name != "posix":  # pragma: no cover - directory fsync is POSIX-only
        return
    dir_fd = os.open(directory, os.O_RDONLY)
    try:
        os.fsync(dir_fd)
    finally:
        os.close(dir_fd)


def atomic_write_text(path: Path, text: str) -> None:
    """Write ``text`` to ``path`` atomically via a same-directory temp file.

    The temp file's data is fsynced before the rename and the parent directory
    after, so a power loss or kernel panic cannot leave the control file present
    but zero-length (the rename persisting while the data blocks do not), which
    would brick every subsequent ``--state``/``--manifest``/``docs`` read (F13).
    """
    path.parent.mkdir(parents=True, exist_ok=True)
    fd, tmp_name = tempfile.mkstemp(dir=path.parent, prefix=f".{path.name}.", suffix=".tmp")
    tmp = Path(tmp_name)
    try:
        with os.fdopen(fd, "w") as handle:
            handle.write(text)
            handle.flush()
            os.fsync(handle.fileno())  # data durable before the rename
        os.replace(tmp, path)
        _fsync_dir(path.parent)  # the rename itself durable
    except BaseException:
        tmp.unlink(missing_ok=True)
        raise
