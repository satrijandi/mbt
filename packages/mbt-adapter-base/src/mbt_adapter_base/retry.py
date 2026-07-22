"""Retry-with-jitter for transient data-plane failures (R2-2).

A small generic helper: each I/O seam supplies its own predicate for what
counts as a *transient* error (a warehouse blip, a dropped request, a locked
database) versus a deterministic one (a SQL error, a missing object) that no
retry can fix. Keeping the predicate a parameter is the whole point - a
warehouse adapter and a file store share the backoff but not the error
taxonomy.
"""

from __future__ import annotations

import random
import time
from collections.abc import Callable
from typing import TypeVar

_T = TypeVar("_T")


def retry_with_jitter(
    fn: Callable[[], _T],
    *,
    is_transient: Callable[[BaseException], bool],
    attempts: int = 6,
    base_delay: float = 0.05,
) -> _T:
    """Call ``fn``; on a *transient* error (per ``is_transient``) sleep and retry,
    up to ``attempts`` total tries. A non-transient error, or the last attempt,
    propagates unchanged.

    The backoff is full jitter over ``[0, base_delay * 2**attempt)``: two jobs
    colliding on the same warehouse must not re-collide on identical
    deterministic sleeps (the thundering herd the retry exists to break),
    matching the S3 and mlflow seams' jittered backoff.
    """
    for attempt in range(attempts - 1):
        try:
            return fn()
        except Exception as exc:  # re-raised below unless the seam calls it transient
            if not is_transient(exc):
                raise
            time.sleep(random.uniform(0, base_delay * 2**attempt))
    return fn()  # the last attempt is un-caught: its error is the caller's to see
