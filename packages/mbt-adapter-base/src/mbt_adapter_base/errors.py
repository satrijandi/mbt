"""The error shape adapters raise, so core can render it (A-4, "Related").

``SparkAdapterError`` and ``SnowflakeAdapterError`` were identical classes in
two packages, neither deriving from a shared base. Both flattened ``hint`` into
the message string (``f"{message}\\n  hint: {hint}"``), so core could not render
it as a hint field the way it does for ``MbtError.hint`` - the hint was there,
but only as text that happened to look like one.

Keeping ``hint`` a field costs nothing: ``__str__`` still renders it the same
way, so existing messages and the ``match=`` in existing tests are unchanged,
while a caller that wants the hint alone can now read it.
"""


class AdapterFailure(RuntimeError):
    """An adapter could not do what it was asked.

    Not ``mbt.exceptions.AdapterError``: that lives in core and an adapter
    package cannot import it. Core wraps this on the way out.
    """

    def __init__(self, message: str, hint: str | None = None, *, resource: str | None = None):
        self.message = message
        self.hint = hint
        self.resource = resource
        super().__init__(message)

    def __str__(self) -> str:
        parts = [self.message]
        if self.resource:
            parts.append(f"resource: {self.resource}")
        if self.hint:
            parts.append(f"hint: {self.hint}")
        return "\n  ".join(parts)


__all__ = ["AdapterFailure"]
