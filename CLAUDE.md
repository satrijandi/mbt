# CLAUDE.md - working guide for the mbt repo

mbt ("dbt for ML models") is a uv workspace monorepo: `packages/{mbt-core, mbt-adapter-base, mbt-xgboost, mbt-lightgbm, mbt-sklearn, mbt-mlflow, mbt-optuna, mbt-snowflake, mbt-spark, mbt-h2o, mbt-testing}`, plus `examples/showcase`, repo-root `tests/` (E2E, golden, perf, live, plus `tests/fixtures/{churn_demo, revenue_demo}` - whole mbt projects the suite copies to tmp and drives through the real CLI, excluded from collection via `norecursedirs`), and `docs/` (mkdocs + ADRs).
Design history lives in `docs/adr/`; read the relevant ADR before "fixing" anything that looks odd.
The review cycle in flight lives at the repo root (`FEEDBACK_v3.md`): findings plus a progress log, one appended entry per completed item (symptom, fix, verification, docs).
Closed cycles move to `design-history/reviews/`; code comments cite them by section (`FEEDBACK 2.6`, `R2-7`, `F17`), so do not delete them.

## Verify (run all of these before calling work done)

```bash
uv run pytest -q -m "not e2e" --cov  # fast suite + CI's 100% coverage gate (~2 min)
uv run pytest -q -m e2e --timeout 1800   # e2e tier incl. JVM; needs java 17
uv run ruff check . && uv run ruff format --check .
uv run mypy packages/mbt-core/src packages/mbt-adapter-base/src \
  packages/mbt-xgboost/src packages/mbt-mlflow/src packages/mbt-optuna/src \
  packages/mbt-lightgbm/src packages/mbt-sklearn/src packages/mbt-testing/src \
  packages/mbt-snowflake/src packages/mbt-spark/src \
  packages/mbt-h2o/src   # strict, all 11 packages, must be clean
uv run pre-commit run --all-files
uv run mkdocs build --strict         # docs changes; site/ is gitignored output
uv run yamllint -d "{extends: relaxed, rules: {line-length: {max: 140}}}" packages examples tests/fixtures .github
uv run python scripts/audit_dependencies.py   # dependency advisories; needs network
```

- Run the JVM e2e tier via `uv run` (so `.venv/bin/spark-submit` is on PATH) with `JAVA_HOME=/opt/homebrew/opt/openjdk@17` locally.
- Live external-system tests (`-m live`) are opt-in and NOT part of the battery above; both tiers run nightly in CI via `.github/workflows/live.yml`. `-m live_snowflake` skips unless `MBT_LIVE_SNOWFLAKE=1`, then fails loudly if `SNOWFLAKE_*` env vars are incomplete (setup in `packages/mbt-snowflake/README.md`). `-m live_showcase` skips unless `MBT_LIVE_SHOWCASE=1`, then fails loudly if docker is unusable; it boots the `examples/showcase` compose stack (see its README).
- Do not pipe test commands through `tail` and trust the exit code; the pipeline returns tail's status, not pytest's. Piping through `tail` also throws away the traceback you will need; write the run to a file instead.
- Never run two `--cov` suites at once (e.g. the fast suite while the e2e tier runs in the background). Both write the same `.coverage` file, so the second clobbers the first and the report is nonsense - it reads as a coverage FAILURE, not as a conflict, which sends you hunting a regression that does not exist. Seen twice: a fast-suite run reporting 98.8% and an e2e run reporting 20.2%, both green on tests. CI is immune (separate jobs, separate containers); local runs are not. Serialize them, or pass `COVERAGE_FILE=.coverage.<name>`.
- `.github/workflows/upstream.yml` is the only tier that ignores `uv.lock`: it re-resolves to the newest versions our constraints allow and runs the fast + e2e tiers against them, nightly. It exists because the lock hides upstream breakage from every other check - ci.yml's `floors` job also re-resolves but runs only `-m "not e2e"`, so it could not see h2o moving MOJO export behind a paid tier. A red run there is usually upstream's change, not ours; read the resolution diff in the job summary first, then either adapt mbt or cap the package in its pyproject with the reason as a comment. It never commits the re-resolved lock.
- `scripts/audit_dependencies.py` wraps pip-audit so an accepted advisory cannot rot: it fails on an unaccepted finding AND on an acceptance that no longer fires. Every entry in its `ACCEPTED` map states why the fix is unreachable, why the code is not in mbt's execute path, and what ends the acceptance. Adding an entry is a security-posture decision, not a build fix.
- `uv lock --upgrade-package X` is NOT surgical when X has a wide subtree - it re-resolves and can move unrelated packages (it silently took h2o across a breaking release once). Always read the lock diff; never trust uv's "Updated ..." summary lines.
- Workflow YAML under `.github/` (repo and scaffold) must also parse with PyYAML; unquoted scalars containing `: ` are syntax errors that yamllint's relaxed profile is the only local check for.

## Load-bearing decisions (do not "clean up")

- Lazy imports everywhere are intentional (plugin import hygiene, ADR-14); ruff PLC0415 is disabled for this reason.
- Events go to stderr; stdout is command data. Job subprocesses emit JSON events on stdout and the coordinator forwards them.
- `main()` in `mbt/cli/main.py` catches BOTH real-click and typer-vendored-click exceptions; typer >= 0.20 vendors click, so the duplicate-looking except tuples are required.
- `uv.lock` contains TWO pyspark versions on purpose: a `[tool.uv] conflicts` fork keeps the dev lock on Spark 4.x while `mbt-h2o[sparkling]` pins 3.5. Never hand-edit the lock; use `uv lock`.
- Dependency floors are load-bearing metadata: CI's `floors` job installs every direct dependency at its declared lower bound via `scripts/install_floors.py`, then runs the fast suite and the advisory audit against it.
  When adding or bumping a dependency, keep the floor honest (test it or raise it, with the reason as a pyproject comment).
  Do NOT go back to `uv sync --resolution lowest-direct`: in a virtual workspace root the members' requirements are not "direct", so it resolves everything to newest and the job silently becomes a duplicate of the `test` job.
  That is not hypothetical - it is what the job did until 2026-08-27, which is why `duckdb>=1.0` (could not parse the local adapter's own SQL) and `click>=8.1` (could not run the CLI tests) survived as declared floors, and why ~70 advisories sat unnoticed against floor versions.
  The job re-asserts the environment really is at the floors (`--verify`) so that regression fails loudly instead of quietly.
  Every step in that job must pass `uv run --no-project`; without it uv re-syncs from `uv.lock` and replaces the floors with the locked versions.
  Reproduce locally with a throwaway venv, never the repo's own: `uv venv /tmp/floors && VIRTUAL_ENV=/tmp/floors uv run --no-project python scripts/install_floors.py`.
- CI matrixes the fast suite over CPython 3.11-3.14; the JVM e2e tier stays on 3.11 deliberately.
- Snapshots: one token scheme per pipeline. The scaffold CI workflows pass `--deep-snapshot` on every compiling step because fresh checkouts rewrite mtimes (ADR-11); a deep baseline diffed with the default mtime scheme flags everything.
- Champion gates use a paired bootstrap lower bound (ADR-18); the seed ladder is `spec.seed` train, `+1` tuning, `+2` validation carve, `+3` bootstrap, `+4` random k-fold, `+5` calibration carve - a new seeded stage takes the next rung.
- Path semantics: the CLI coordinator chdirs to `--project-dir` in `make_ctx` (jobs already run with cwd=project), so config-relative paths are project-relative; paths typed on the command line are absolutized against the invocation cwd via `ctx.resolve_cli_path` BEFORE use. New CLI path options must go through `resolve_cli_path`.
- Manifests verify `env_digest` on `--manifest` execution (ADR-19); `generated_at == anchor` keeps same-anchor compiles byte-identical.

## Test-suite conventions

- Test module basenames must be unique across all tests dirs; shared helpers live in uniquely named modules (`core_helpers.py`, `e2e_utils.py`), never imported from conftest.
- Tests write ONLY under pytest tmp dirs. A session guard in the repo-root `conftest.py` fails any run that leaves new entries in the repo root.
  Relative roots in generated profiles (`root: ./target/...`, `file://./...`, mlflow's `./mlruns`, the fake adapters' default roots) resolve against the pytest cwd, not the project dir - always write absolute tmp paths into generated profiles, or chdir.
- Golden manifests are byte-compared and regenerated deliberately: `UPDATE_GOLDEN=1 uv run pytest tests/test_golden_manifest.py`. Spec/gate/hooks edits flip config hashes by design (ADR-6/ADR-7), so golden churn after those edits is expected - regenerate and say so.
- The e2e scaffold state-loop test uses a local bare git repo as origin; no network or GitHub is required anywhere in the suite.
- Exit-code semantics everywhere: 0 success, 1 hard error, 2 quality failure (gates/checks/tests said no). Tests assert on these; do not conflate 1 and 2.

## Docs

- mkdocs nav lives in `mkdocs.yml`; new pages must be added there or `--strict` fails the build.
- `docs/troubleshooting.md` is the operator runbook: every entry's symptom text was captured from a real reproduction; if you change an error message, update the runbook (and vice versa).
- `docs/v0.1-status.md` markets the project's rigor; keep its claims exactly true (the review dinged it for one overclaim).
- The scaffold under `packages/mbt-core/src/mbt/cli/_scaffold/` is stamped into user projects by `mbt init`; its README, workflows, and pins are user-facing docs and are asserted by `tests/test_cli_basics.py`.
- `docs/cli-reference.md` is guarded by `tests/test_cli_reference_sync.py`: every CLI command and every non-boilerplate `--flag` must appear in it, so adding a command/flag fails the suite until it is documented.
