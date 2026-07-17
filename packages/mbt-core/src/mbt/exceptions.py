"""mbt error taxonomy (TSD §17).

Every user-facing error carries: what happened, which resource/file, and
what to do next (``hint``). Exit codes: hard errors exit 1; quality failures
(gates/tests) exit 2 via run status, not exceptions.
"""

from pathlib import Path


class MbtError(Exception):
    """Base for all mbt errors. Exit code 1."""

    exit_code = 1

    def __init__(
        self,
        message: str,
        *,
        resource: str | None = None,
        path: str | Path | None = None,
        hint: str | None = None,
    ) -> None:
        self.message = message
        self.resource = resource
        self.path = str(path) if path is not None else None
        self.hint = hint
        super().__init__(message)

    def __str__(self) -> str:
        parts = [self.message]
        if self.resource:
            parts.append(f"resource: {self.resource}")
        if self.path:
            parts.append(f"file: {self.path}")
        if self.hint:
            parts.append(f"hint: {self.hint}")
        return "\n  ".join(parts)


def cause_message(exc: BaseException) -> str:
    """The one-line text to interpolate when wrapping ``exc`` in another error.

    For an ``MbtError``, use only its ``message``: its ``__str__`` is multi-line
    (message + ``resource:``/``file:``/``hint:``), so interpolating the full str
    into an outer error that also carries a hint renders two out-of-order
    ``hint:`` lines. For any other exception, ``str(exc)`` is already one-line.
    """
    return exc.message if isinstance(exc, MbtError) else str(exc)


class ConfigError(MbtError):
    """Invalid project, profile, or resource configuration (parse/validation)."""


class CompilationError(MbtError):
    """Failures while rendering, anchoring, pinning, or hashing a manifest."""


class AdapterError(MbtError):
    """An adapter failed; wraps the original exception with node context."""

    @classmethod
    def wrap(cls, exc: Exception, *, adapter: str, resource: str | None = None) -> "AdapterError":
        return cls(
            f"adapter '{adapter}' failed: {cause_message(exc)}",
            resource=resource,
            hint="run with --log-format json for the full event stream",
        )


class GateFailure(MbtError):
    """A quality gate failed hard (used only where an exception is required)."""

    exit_code = 2


class StateError(MbtError):
    """Unreadable or incompatible --state / --manifest references."""
