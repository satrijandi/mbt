"""Small shared helpers."""

import json
import re
from typing import Any


def levenshtein(a: str, b: str) -> int:
    """Edit distance, iterative two-row implementation."""
    if a == b:
        return 0
    if not a or not b:
        return len(a) + len(b)
    previous = list(range(len(b) + 1))
    for i, ca in enumerate(a, start=1):
        current = [i]
        for j, cb in enumerate(b, start=1):
            current.append(min(previous[j] + 1, current[j - 1] + 1, previous[j - 1] + (ca != cb)))
        previous = current
    return previous[-1]


def did_you_mean(unknown: str, candidates: list[str], max_distance: int = 3) -> str | None:
    """The closest candidate within an edit-distance budget, if any."""
    scored = sorted((levenshtein(unknown, c), c) for c in candidates)
    if scored and scored[0][0] <= max_distance:
        return scored[0][1]
    return None


def canonical_json(value: Any) -> str:
    """Canonical JSON for hashing: UTF-8, sorted keys, no whitespace (TSD §8.4).

    CPython's json module already serializes floats via ``repr`` (shortest
    round-trip form), which is exactly the canonical float text we want.
    """

    def default(obj: Any) -> Any:
        raise TypeError(f"not canonically serializable: {type(obj).__name__}")

    return json.dumps(
        value,
        sort_keys=True,
        separators=(",", ":"),
        ensure_ascii=False,
        default=default,
    )


_IDENT_RE = re.compile(r"[^a-z0-9_]+")


def slugify(text: str) -> str:
    """Lowercase snake_case identifier from arbitrary text."""
    return _IDENT_RE.sub("_", text.lower()).strip("_")
