"""DuckDB 2.0's connection rule, enforceable on any DuckDB (unique module name for pytest).

DuckDB 2.0 raises ``ResourceInUseException`` when a connection is asked to do
anything while it still holds an unfinished result, and ``execute(...).fetchone()``
leaves one unfinished until a fetch returns nothing. DuckDB 1.x tolerated that, so
code that queried once per split on a shared connection passed for years here and
broke the night ``duckdb`` moved to 2.0 (issue #32) - found only by the nightly
run against newest versions, because the lock pins 1.x.

``StrictConnection`` enforces the 2.0 rule on whatever is installed, so the
regression is caught by the ordinary suite.
"""

from typing import Any

import duckdb


class StrictResult:
    """A result that stays 'unfinished' on its owner until it is fully fetched."""

    def __init__(self, owner: "StrictConnection", result: Any) -> None:
        self._owner = owner
        self._result = result
        owner.unfinished = self

    def fetchone(self) -> Any:
        row = self._result.fetchone()
        if row is None:
            self._owner.unfinished = None
        return row

    def fetchall(self) -> Any:
        rows = self._result.fetchall()
        self._owner.unfinished = None
        return rows

    def __getattr__(self, name: str) -> Any:
        # arrow/df/numpy fetches consume the whole result
        attribute = getattr(self._result, name)
        self._owner.unfinished = None
        return attribute


class StrictConnection:
    def __init__(self, real: Any) -> None:
        self._real = real
        self.unfinished: StrictResult | None = None

    def _guard(self) -> None:
        if self.unfinished is not None:
            raise RuntimeError(
                "Resource In Use Error: connection has an unfinished result from execute(); "
                "call complete() or abort(), or fetch until a fetch returns no rows"
            )

    def execute(self, *args: Any, **kwargs: Any) -> StrictResult:
        self._guard()
        return StrictResult(self, self._real.execute(*args, **kwargs))

    def register(self, *args: Any, **kwargs: Any) -> Any:
        self._guard()
        return self._real.register(*args, **kwargs)

    def __getattr__(self, name: str) -> Any:
        return getattr(self._real, name)


def strict_connect(*args: Any, **kwargs: Any) -> StrictConnection:
    return StrictConnection(_REAL_CONNECT(*args, **kwargs))


_REAL_CONNECT = duckdb.connect
