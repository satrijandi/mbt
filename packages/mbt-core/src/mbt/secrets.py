"""Secret tainting and redaction (TSD §18, FR-PROJ-05, NFR-07).

Values entering through ``env_var()`` are wrapped in :class:`Secret` and
registered as tainted. Every serialization path (events, run results,
manifests) passes its text through :func:`redact` as defense in depth; the
manifest additionally stores profile configs *unrendered* so secrets never
reach it in the first place.
"""

import threading

REDACTED = "***"

_lock = threading.Lock()
_tainted: set[str] = set()


class Secret(str):
    """A string whose repr never shows its value.

    It still *is* the value (adapters need real URIs/credentials), which is
    why tainting + redaction exists for anything that leaves the process.
    """

    __slots__ = ()

    def __repr__(self) -> str:
        return f"'{REDACTED}'"


def taint(value: str) -> Secret:
    """Mark a value as secret and return it wrapped."""
    if value:
        with _lock:
            _tainted.add(str(value))
    return Secret(value)


def clear_taints() -> None:
    """Testing hook: forget all tainted values."""
    with _lock:
        _tainted.clear()


def redact(text: str) -> str:
    """Replace every tainted value occurring in ``text`` with ``***``."""
    with _lock:
        tainted = sorted(_tainted, key=len, reverse=True)
    for value in tainted:
        if value in text:
            text = text.replace(value, REDACTED)
    return text
