"""Repo-root pytest guard: the suite must leave the working tree clean.

Tests write only under tmp dirs; anything that lands in the repo root
(./target, ./mlruns, stray dbs) pollutes `git status`, can cross-contaminate
golden/e2e runs, and grows without bound (FEEDBACK section 2.6).

Two checks, because a before/after diff alone has a blind spot: anything
already present when the session starts is in `before` forever, so a leftover
from an earlier run is invisible to every run after the one that made it
(FEEDBACK v3 G-2). `_KNOWN_LITTER` names the entries that are never legitimate
and fails on them whether or not this session created them.
"""

import fnmatch
from collections.abc import Iterator
from pathlib import Path

import pytest

REPO_ROOT = Path(__file__).resolve().parent

#: Tooling caches the test/coverage machinery itself writes at session end;
#: everything else that appears in the repo root is litter. Matched exactly:
#: `startswith` on a tuple is a PREFIX test, which also excused anything merely
#: beginning with one of these names.
_TOOLING = frozenset(
    {
        "__pycache__",
        ".pytest_cache",
        ".hypothesis",
        ".coverage",
        ".mypy_cache",
        ".ruff_cache",
    }
)

#: Root entries that are always litter, whenever they appeared. Generated
#: project state (`target/`, `mlruns/`, stray dbs) plus the JVM leftovers the
#: Spark and H2O tiers can drop - `.gitignore` lists those five, which keeps
#: `git status` clean and therefore keeps them invisible. The e2e tier does not
#: currently produce any of them; this is what notices if that changes.
_KNOWN_LITTER = (
    "target",
    "mlruns",
    "mlartifacts",
    "mlflow.db",
    "*.duckdb",
    "*.duckdb.wal",
    "derby.log",
    "metastore_db",
    "spark-warehouse",
    "h2ologs",
    "hs_err_pid*",
)


def _snapshot() -> set[str]:
    entries = {p.name for p in REPO_ROOT.iterdir() if p.name not in _TOOLING}
    # new writes into a pre-existing ./target must be caught too
    target = REPO_ROOT / "target"
    if target.is_dir():
        entries |= {f"target/{p.name}" for p in target.iterdir()}
    return entries


def _known_litter(entries: set[str]) -> list[str]:
    return sorted(
        entry
        for entry in entries
        if "/" not in entry and any(fnmatch.fnmatch(entry, pat) for pat in _KNOWN_LITTER)
    )


@pytest.fixture(scope="session", autouse=True)
def repo_root_stays_clean() -> Iterator[None]:
    before = _snapshot()
    stale = _known_litter(before)
    assert not stale, (
        f"the repo root already held generated state before this run: {stale} - "
        "delete it. A before/after diff cannot see pre-existing litter, so it "
        "would stay invisible to every future run."
    )
    yield
    after = _snapshot()
    litter = sorted(after - before) or _known_litter(after)
    assert not litter, (
        f"this test run littered the repo root: {litter} - write under tmp_path "
        "(and use absolute roots in generated profiles; cwd-relative paths "
        "resolve against the pytest cwd, not the project dir)"
    )
