<!--
CONTRIBUTING.md has the full verification battery and the conventions that
will save you time. The short version is below.
-->

## What and why

<!-- What changes, and what problem it solves. If it fixes an issue, "Fixes #N". -->

## Verification

<!--
Say what you ran and what it said - "tests pass" is not evidence, and CI
running them later is not a substitute for you having read the output.
For a bug fix, the useful thing is the reproduction: what failed before,
and what that same thing does now.
-->

- [ ] `uv run pytest -q -m "not e2e" --cov` (the 100% coverage gate is enforced)
- [ ] `uv run ruff check . && uv run ruff format --check .`
- [ ] `uv run mypy` over the packages you touched (CI runs all ten, strict)
- [ ] `uv run pytest -q -m e2e` if you touched an adapter, the CLI, or the scaffold
- [ ] `uv run mkdocs build --strict` if you touched docs

## Things that fail CI if missed

- [ ] A new CLI command or flag is documented in `docs/cli-reference.md` (a test enforces this)
- [ ] A changed error message is reflected in `docs/troubleshooting.md`, and vice versa
- [ ] Golden manifests regenerated *deliberately* if spec/gate/hook changes moved config hashes
      (`UPDATE_GOLDEN=1 uv run pytest tests/test_golden_manifest.py`) - say so in the description
- [ ] A new or bumped dependency has an honest floor (CI installs every declared lower bound)
- [ ] Anything that looks like a "cleanup" of a documented decision has an ADR argument behind it
