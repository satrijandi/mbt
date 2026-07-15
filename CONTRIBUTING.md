# Contributing to mbt

Thanks for your interest in mbt.
This page covers the mechanics: setup, the verification battery, test conventions, and what a good PR looks like.

## Setup

```bash
git clone https://github.com/satrijandi/mbt && cd mbt
uv sync                      # whole workspace, all extras
uv run pre-commit install    # hooks: ruff, yamllint, mypy
```

Python 3.11+ is required (CI tests 3.11 through 3.14).
The JVM end-to-end tier (Spark, H2O) additionally needs Java 17 on PATH (`JAVA_HOME=/opt/homebrew/opt/openjdk@17` on macOS with Homebrew).

## Verify before you push

CI runs all of these; running them locally first saves a round trip.

```bash
uv run pytest -q -m "not e2e"            # fast suite (~1 min)
uv run pytest -q -m e2e --timeout 1800   # e2e tier incl. JVM; needs Java 17
uv run ruff check . && uv run ruff format --check .
uv run mypy packages/mbt-core/src packages/mbt-adapter-base/src \
  packages/mbt-xgboost/src packages/mbt-mlflow/src packages/mbt-optuna/src \
  packages/mbt-lightgbm/src packages/mbt-testing/src packages/mbt-snowflake/src \
  packages/mbt-spark/src packages/mbt-h2o/src
uv run pre-commit run --all-files
uv run mkdocs build --strict             # for docs changes
```

The fast suite enforces 100% line coverage on the coordinator-process packages; a PR that adds uncovered lines fails CI.

## Conventions that will save you time

- **Read the ADRs first.** Design history lives in `docs/adr/`.
  Several things that look wrong (lazy imports, duplicate except tuples, two pyspark versions in the lock) are deliberate and documented; a PR "cleaning them up" will be declined with a pointer to the ADR.
- **Tests write only under pytest tmp dirs.**
  A session guard fails any run that leaves new entries in the repo root.
  Always write absolute tmp paths into generated profiles, or chdir.
- **Test module basenames must be unique** across all tests directories.
- **Golden manifests are byte-compared.**
  Spec/gate/hook edits flip config hashes by design; regenerate deliberately with `UPDATE_GOLDEN=1 uv run pytest tests/test_golden_manifest.py` and say so in the PR.
- **Exit codes are load-bearing:** 0 success, 1 hard error, 2 quality failure.
  Tests assert on them; do not conflate 1 and 2.
- **Docs stay in sync by test.**
  Every CLI command and non-boilerplate flag must appear in `docs/cli-reference.md` (`tests/test_cli_reference_sync.py` enforces it), and new docs pages must be added to `mkdocs.yml` nav or the strict build fails.
- **Error messages are documented.**
  `docs/troubleshooting.md` entries carry reproduced symptom text; if you change an error message, update the runbook.
- **Dependency floors are tested.**
  CI installs every direct dependency at its declared lower bound and runs the fast suite; when you add or bump a dependency, keep the floor honest (test it or raise it, with the reason as a pyproject comment).

## Writing an adapter

`docs/adapter-authoring.md` is the guide.
The ship bar is the compliance suite in `mbt-adapter-base`: subclass `TrainingAdapterCompliance` (and `PredictionStoreCompliance` where relevant) and keep `test_no_core_imports` green - adapters build against `mbt-adapter-base` only, never `mbt-core` internals.

## License

By contributing, you agree that your contributions are licensed under the [Apache License 2.0](LICENSE) that covers the project.
