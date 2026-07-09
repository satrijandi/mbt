"""Repo-root pytest guard: the suite must leave the working tree clean.

Tests write only under tmp dirs; anything that lands in the repo root
(./target, ./mlruns, stray dbs) pollutes `git status`, can cross-contaminate
golden/e2e runs, and grows without bound (FEEDBACK section 2.6).
"""

from collections.abc import Iterator
from pathlib import Path

import pytest

REPO_ROOT = Path(__file__).resolve().parent

#: Tooling caches the test/coverage machinery itself writes at session end;
#: everything else that appears in the repo root is litter.
_TOOLING = (
    "__pycache__",
    ".pytest_cache",
    ".hypothesis",
    ".coverage",
    ".mypy_cache",
    ".ruff_cache",
)


def _snapshot() -> set[str]:
    entries = {p.name for p in REPO_ROOT.iterdir() if not p.name.startswith(_TOOLING)}
    # new writes into a pre-existing ./target must be caught too
    target = REPO_ROOT / "target"
    if target.is_dir():
        entries |= {f"target/{p.name}" for p in target.iterdir()}
    return entries


@pytest.fixture(scope="session", autouse=True)
def repo_root_stays_clean() -> Iterator[None]:
    before = _snapshot()
    yield
    litter = sorted(_snapshot() - before)
    assert not litter, (
        f"this test run littered the repo root: {litter} - write under tmp_path "
        "(and use absolute roots in generated profiles; cwd-relative paths "
        "resolve against the pytest cwd, not the project dir)"
    )
