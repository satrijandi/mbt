# Lessons behind the rules in CLAUDE.md

`CLAUDE.md` is read at the start of every agent session, so it states each rule and the one-line reason, and nothing more.
This file keeps the incident behind a rule, for the person who wants to know whether the rule can be relaxed.
Like the rest of this directory it is provenance: it says what happened, not how mbt works today.

## Never run two `--cov` suites at once

Both write the same `.coverage` file, so the second clobbers the first, and the report reads as a coverage failure, not as a conflict.
It sent two sessions hunting a regression that did not exist: a fast-suite run reporting 98.8% and an e2e run reporting 20.2%, both green on tests.
CI is immune because its jobs are separate containers; local runs are not.

## The mutation tier is one CI job per module

Run as one job, `mutation.yml` hit its two-hour limit on most nights, and the failure notice called each timeout a regression.
The split into `--plan`, one job per module and `--combine` (which refuses a partial set) is proven exact: 464 + 86 survivors, the same as the single run, with the slowest module at 59 minutes of a 120-minute limit.

## `upstream.yml` is the only tier that ignores `uv.lock`

The lock hides upstream breakage from every other check.
ci.yml's `floors` job also re-resolves, but runs only `-m "not e2e"`, so it could not see h2o 3.46.0.12 move MOJO export behind a paid tier, which nothing except the e2e H2O compliance module calls.
A red run there is usually upstream's change, so the resolution diff in the job summary is the first thing to read.

## `released.yml` installs the newest release tag

Nothing tested the published tag after it shipped.
v0.2.0 declared `duckdb>=1.2`; DuckDB 2.0 arrived later and every fresh install failed the quickstart's first `mbt build`, while `main` (fixed on 2026-11-12) and every other tier stayed green.
v0.1.0 had failed the same way for a different reason in v6 (A-1).
The tier installs the tag the way `docs/installation.md` says, with no lock, and walks the quickstart to the end (FEEDBACK v7 A-1, F-1).

## `uv lock --upgrade-package X` is not surgical

With a wide subtree it re-resolves and can move unrelated packages: it once silently took h2o across a breaking release.
uv's "Updated ..." lines do not list everything that moved, so read the lock diff.

## Dependency floors are checked, not declared

CI's `floors` job used `uv sync --resolution lowest-direct`, which in a virtual workspace root resolves everything to newest, because the members' requirements are not "direct".
The job was therefore a duplicate of the `test` job until 2026-08-27.
That is why `duckdb>=1.0` (could not parse the local adapter's own SQL) and `click>=8.1` (could not run the CLI tests) survived as declared floors, and why about 70 advisories sat unnoticed against floor versions.
`scripts/install_floors.py` replaced it, and re-asserts with `--verify` that the environment really is at the floors, so the regression fails loudly.

## A check that raises is not a failed check

`quality/checks.py` caught every exception from a built-in check and recorded it as a failed check, which exits 2, the code that means "the data said no".
On a fresh v0.2.0 install the DuckDB 2.0 error therefore read as a data-quality problem, and anything that routes exit 2 to the data owners would have paged them for a broken environment (FEEDBACK v7 A-3).
A check's own verdict (a returned failure, or `CheckFailure`) is exit 2; anything else is `CheckError`, exit 1.
