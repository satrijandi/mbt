# CLAUDE.md - working guide for the mbt repo

mbt ("dbt for ML models") is a uv workspace monorepo: `packages/{mbt-core, mbt-adapter-base, mbt-xgboost, mbt-lightgbm, mbt-mlflow, mbt-optuna, mbt-snowflake, mbt-spark, mbt-h2o, mbt-testing}`, plus `examples/churn_demo`, repo-root `tests/` (E2E, golden, perf), and `docs/` (mkdocs + ADRs).
Design history lives in `docs/adr/`; read the relevant ADR before "fixing" anything that looks odd.
`FEEDBACK.md` carries an external review and a progress log; when working through it, append a log entry per completed item (symptom, fix, verification, docs).

## Verify (run all of these before calling work done)

```bash
uv run pytest -q -m "not e2e"        # fast suite (~35 s)
uv run pytest -q -m e2e --timeout 1800   # e2e tier incl. JVM; needs java 17
uv run ruff check . && uv run ruff format --check .
uv run mypy packages/mbt-core/src packages/mbt-adapter-base/src \
  packages/mbt-xgboost/src packages/mbt-mlflow/src packages/mbt-optuna/src \
  packages/mbt-lightgbm/src packages/mbt-testing/src   # strict, must be clean
uv run pre-commit run --all-files
uv run mkdocs build --strict         # docs changes; site/ is gitignored output
uv run yamllint -d "{extends: relaxed, rules: {line-length: {max: 140}}}" packages examples .github
```

- Run the JVM e2e tier via `uv run` (so `.venv/bin/spark-submit` is on PATH) with `JAVA_HOME=/opt/homebrew/opt/openjdk@17` locally.
- Live external-system tests (`-m live_snowflake`) are opt-in and NOT part of the battery above: they skip unless `MBT_LIVE_SNOWFLAKE=1`, then fail loudly if `SNOWFLAKE_*` env vars are incomplete (setup in `packages/mbt-snowflake/README.md`; nightly in CI via `.github/workflows/live.yml`).
- Do not pipe test commands through `tail` and trust the exit code; the pipeline returns tail's status, not pytest's.
- Workflow YAML under `.github/` (repo and scaffold) must also parse with PyYAML; unquoted scalars containing `: ` are syntax errors that yamllint's relaxed profile is the only local check for.

## Load-bearing decisions (do not "clean up")

- Lazy imports everywhere are intentional (plugin import hygiene, ADR-14); ruff PLC0415 is disabled for this reason.
- Events go to stderr; stdout is command data. Job subprocesses emit JSON events on stdout and the coordinator forwards them.
- `main()` in `mbt/cli/main.py` catches BOTH real-click and typer-vendored-click exceptions; typer >= 0.20 vendors click, so the duplicate-looking except tuples are required.
- `uv.lock` contains TWO pyspark versions on purpose: a `[tool.uv] conflicts` fork keeps the dev lock on Spark 4.x while `mbt-h2o[sparkling]` pins 3.5. Never hand-edit the lock; use `uv lock`.
- Dependency floors are load-bearing metadata: CI's `floors` job installs every direct dependency at its declared lower bound (`uv sync --resolution lowest-direct`) and runs the fast suite. When adding or bumping a dependency, keep the floor honest (test it or raise it, with the reason as a pyproject comment).
- CI matrixes the fast suite over CPython 3.11-3.14; the JVM e2e tier stays on 3.11 deliberately.
- Snapshots: one token scheme per pipeline. The scaffold CI workflows pass `--deep-snapshot` on every compiling step because fresh checkouts rewrite mtimes (ADR-11); a deep baseline diffed with the default mtime scheme flags everything.
- Champion gates use a paired bootstrap lower bound (ADR-18); seed derivation is `spec.seed + 3` (documented; +2 is taken).
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
