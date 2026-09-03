# FEEDBACK v3: a whole-repo review of mbt

Review date: 2026-09-01, against `main` (`591f5de`).
Perspective: a senior MLOps engineer, a senior data scientist, and a release/supply-chain reviewer, each reading the tree from their own workflow.
Method: full verification battery run locally first, then a code and docs read, then the product driven end to end as a new user through the real CLI (`mbt init` -> `build` -> `promote` -> `score` -> `monitor` -> `docs generate`) in a throwaway directory.
Every finding carries file evidence, and every one of them was reproduced or proven by experiment rather than inferred.
Findings already closed in the two earlier cycles are not re-litigated (they were `FEEDBACK.md` and `FEEDBACK_v2.md` at the repo root when this review was written; G-1 below moved them to `feedback-v1.md` and `feedback-v2.md` beside this file).

## The tree is green

Measured on this machine at review time, not quoted from CI:

| Check | Result |
|---|---|
| `pytest -q -m "not e2e" --cov` | **1247 passed, 57 skipped**, coverage **100.00%** of 9249 statements, 188s |
| `pytest -q -m e2e --timeout 1800` (Java 17) | **78 passed, 6 skipped**, 488s |
| `pytest -q -m "not e2e" -W error::DeprecationWarning` | **1247 passed** (no deprecation debt in mbt's own path) |
| `ruff check .` / `ruff format --check .` | clean, 362 files |
| `mypy --strict` (all 10 packages) | clean, 122 source files |
| `yamllint` (packages, examples, fixtures, .github) | clean |
| `mkdocs build --strict` | clean |
| `scripts/audit_dependencies.py` | clean, 6 still-earned acceptances |
| `mbt init` -> `build` -> `promote` -> `score` -> `monitor` -> `docs generate` | all exit 0, first build 5.6s from a cold scaffold |

So this is a review of gaps and risk, not of a broken tree.

## The honest top line

mbt's engineering discipline is genuinely unusual, and the evidence is in the places most projects leave empty.
The advisory wrapper in `scripts/audit_dependencies.py` fails on acceptances that stop firing, so a suppression cannot outlive its reason.
The floors job installs every declared lower bound and then re-asserts that it really did.
Three separate workflows open and close their own tracking issue so a red tier cannot go unread.
The CI comments explain not just what a step does but which past incident it exists because of.
This is a repo that has learned specific lessons and encoded them.

The gap this review keeps finding is narrower and more uncomfortable: **the discipline is applied unevenly, and the places it has not reached are the places a user actually stands.**

Three patterns:

1. **One real defect is known, worked around in a config comment, and never fixed.** `env_var()` taints every value it returns, so a non-secret environment value corrupts every serialized surface downstream, including `run_results.json` - the file `docs/v0.1-status.md` names as mbt's integration contract. The maintainer already hit this, and the fix that shipped was a warning comment in one showcase profile (A-1).
2. **The claim-enforcement machinery that guards dependencies and docs does not guard the claims about itself.** Coverage on two of ten packages is asserted and never measured. The perf budgets have 50x headroom, measured. `CONTRIBUTING.md` hands a contributor the exact command that will make them fail CI. Three published documents state things that are no longer true (B-1..B-5, C-1..C-3).
3. **The user-facing surfaces got the least attention.** The first thing a new user reads is `--help`, which leaks 23 internal requirement IDs pointing at a directory the repo itself labels unmaintained. The first artifact they generate is a model card that silently says "no run results yet - run `mbt build`" after they ran `mbt build` (A-2, E-1..E-7).

None of these are architectural. All are small. That is the point: the architecture is done, and what remains is finishing.

---

## Priority action items

Ordered by (impact / effort). Every row is a concrete change, not a direction.

| # | Action | Impact | Effort |
|---|---|---|---|
| **A-1** | Stop `env_var()` tainting non-secret values; add a non-tainting accessor and a regression test on `run_results.json` | High | S |
| **B-1** | Fix `CONTRIBUTING.md:22` to pass `--cov`, matching `CLAUDE.md` | High | XS |
| **B-2** | Delete `attr-defined` from `pyproject.toml:167`; mypy strict is already clean without it | Medium | XS |
| **C-1** | Fix `README.md:145-146`, which documents the known-broken floors mechanism | Medium | XS |
| **A-2** | Make `run_results.json` survive across commands, or fix the docs generator to say what is actually wrong | Medium | S |
| **C-2** | Correct ADR-23 and the Snowflake README: the credentialed `live_snowflake` run happened 2026-08-28 | Medium | XS |
| **D-1** | Attest released wheels with `actions/attest-build-provenance` | Medium | S |
| **B-3** | Measure `mbt-spark` / `mbt-h2o` coverage in the e2e job with an enforced floor | Medium | M |
| **B-4** | Close the `FLOOR_ONLY` rot hole `audit_dependencies.py:95-98` names | Medium | S |
| **E-1** | Strip `FR-*` / `TSD §*` IDs from user-visible `--help` text | Medium | S |
| **C-3** | Correct the two stale claims in `docs/v0.1-status.md` | Medium | XS |
| **D-2** | Declare and cap `mkdocs>=1.6,<2` before MkDocs 2.0 lands | Medium | XS |
| **F-1** | Ship `mbt-sklearn`; scikit-learn is already a dependency | High | M |
| **E-2** | Default `ConsoleSink` to stderr so the invariant holds by construction | Low | XS |
| **B-5** | Add a ratcheted perf baseline alongside the loose NFR-03 contract | Low | S |
| **D-3** | Add a generated `CHANGELOG.md` | Medium | S |
| **E-3..E-7** | CLI and model-card polish (timestamps, double print, wrapping, dark mode, demo data) | Low | S each |
| **G-1** | Move `FEEDBACK*.md` (447 KB) out of the repo root | Low | XS |
| **G-2** | Make the repo-root guard catch pre-existing litter | Low | S |

---

## A. Correctness defects

### A-1. `env_var()` taints every value, so redaction corrupts unrelated output

**Evidence:** `packages/mbt-core/src/mbt/jinja/environment.py:174-178` (capture) and `:203-213` (resolve) both call `taint(value)` on whatever the environment holds.
`packages/mbt-core/src/mbt/secrets.py:46-53` then replaces every occurrence of every tainted string, anywhere, in every serialized surface.

`env_var()` is the *only* channel for environment values in specs and profiles (`SECURITY.md`, `docs/spec-reference.md:49`), so it is used for secrets and for ordinary configuration alike.
The repo's own documented examples do exactly this: `SNOWFLAKE_ACCOUNT`, `SNOWFLAKE_USER`, `SNOWFLAKE_SCHEMA`, `SNOWFLAKE_WAREHOUSE`, `MBT_DATA_ROOT`, `SPARK_DRIVER_HOST` (`docs/spec-reference.md:282-284`, `docs/tutorial.md:46`, `examples/showcase/project/profiles.yml:101,148-157`, and the scaffold's own `profiles.yml:18-20`).

**Reproduced:**

```python
taint("dev")
taint("public")
redact("materialized /data/dev/churn.parquet into public.churn_features (dev target)")
# -> 'materialized /data/***/churn.parquet into ***.churn_features (*** target)'

taint("1")
redact("trained 1000 rows in 1.5s, 1 model")
# -> 'trained ***000 rows in ***.5s, *** model'
```

**This is already known.** `examples/showcase/project/profiles.yml:166-171` carries the workaround as a comment:

> A WHOLE URI, never a bare port. `env_var()` taints its value, and `redact()` replaces every tainted string anywhere in serialized output - so a 4-digit port becomes a "secret" that gets censored out of the middle of unrelated floats, corrupting the job-result JSON (`"value":0.1***234`).

**Why it matters:** the corrupted surface is not cosmetic.
`docs/v0.1-status.md:23` (NFR-06) names `run_results.json` as mbt's observability integration contract, and the scaffold's PR bot parses it.
A tainted short value silently produces malformed numbers in a machine-read file, and the current mitigation is that every user must independently discover the rule and encode it in prose in their own profile.
It also degrades every operator surface: taint `MBT_DATA_ROOT` and every log line about your data directory reads `***`, which is precisely the information an operator needs during an incident.

**Action:**
1. Add a non-tainting accessor for non-secret environment values (for example `env_var(name, default, secret=false)`, or a separate `env()` function), keeping `taint` as the default so the safe path stays the default.
2. Document in `SECURITY.md` that `env_var()` marks its value secret and must not be used for configuration that appears in output.
3. Add a regression test asserting that a run whose profile resolves a short `env_var` still writes numerically valid `run_results.json`.
4. Update `docs/spec-reference.md`, `docs/tutorial.md`, the scaffold profile, and the showcase profiles to use the non-tainting form for the values that are not secrets, and delete the two duplicated warning comments in `examples/showcase/project/profiles.yml`.

**Impact:** High (DS and MLOps both). **Effort:** S.

### A-2. `run_results.json` is latest-write-wins, so model cards silently lose their metrics

**Evidence:** `packages/mbt-core/src/mbt/execute/orchestrator.py:369` and `:556`, and `packages/mbt-core/src/mbt/execute/monitor.py:113`, all write the same `target/run_results.json`.
Each command writes only its own nodes.
`packages/mbt-core/src/mbt/docsgen/generator.py:302-309` reads that one file for model metrics, and `:115` is the fallback text.

**Reproduced** in a fresh `mbt init` project, following the operationally natural order:

```
mbt build      # ok, model registered, metrics computed
mbt promote --model churn_classifier --to production
mbt score      # rewrites run_results.json with only the scoring node
mbt monitor    # rewrites it again
mbt docs generate
```

The generated model card's Metrics section reads:

> no run results yet - run `mbt build`

Running `mbt build && mbt docs generate` immediately afterwards produces the full metrics and slice tables, confirming the cause.

**Why it matters:** the message is actively misleading - it tells the user to do the thing they just did.
The README's quickstart happens to order `docs generate` right after `build`, which is why this has not surfaced; any other order loses the model card content.
The scaffold's CI workflows keep build and score in separate jobs, so CI is not affected today, but nothing enforces that separation.

A secondary consequence: `packages/mbt-core/src/mbt/gc.py:26-34` derives part of the artifact keep-set from the same file, so after a `score` run the keep-set loses its "latest run" contribution.
Severity there is low, because `champion_artifact_uris` protects every stage champion independently (ADR-10) and the `--artifacts-older-than` cutoff protects recent prefixes, but it is the same root cause.

**Action:** write per-command results alongside the latest (`target/run_results.<command>.json`), have `docsgen` and `gc` read the most recent file that contains the node type they need, and change `generator.py:115` to name the real condition ("no metrics for this model in `target/run_results.json`, last written by `mbt monitor`").
Document the overwrite semantics in `docs/architecture.md`.

**Impact:** Medium-High. **Effort:** S.

---

## B. Claims that nothing enforces

This repo's strongest idea is that a claim nobody checks silently becomes false - it is the stated reason the floors job was rewritten and the reason `audit_dependencies.py` exists.
These are the claims that idea has not been applied to yet.

### B-1. `CONTRIBUTING.md` hands contributors the command that fails CI

**Evidence:** `CONTRIBUTING.md:22` gives the fast suite as `uv run pytest -q -m "not e2e"`, with no `--cov`.
`CONTRIBUTING.md:33` then states: "The fast suite enforces 100% line coverage on the coordinator-process packages; a PR that adds uncovered lines fails CI."

Without `--cov` the `fail_under = 100` gate does not run.
A contributor follows the documented command, sees green, pushes, and fails CI.
`CLAUDE.md` gets this right (`uv run pytest -q -m "not e2e" --cov`); the contributor-facing document does not.

**Action:** add `--cov` to `CONTRIBUTING.md:22`.
While there, align the verify list with `CLAUDE.md`: it currently omits `yamllint` and `scripts/audit_dependencies.py`, and its "~1 min" estimate is 188s in practice.

**Impact:** High (this is the single highest-friction thing a new contributor will hit). **Effort:** XS.

### B-2. The `attr-defined` mypy relaxation is dead and can be deleted today

**Evidence:** `pyproject.toml:167` disables `["no-untyped-call", "attr-defined"]` for nine modules.
The comment above it (`:143-154`) explains `no-untyped-call` carefully - pyarrow's `py.typed` flips between releases - and says nothing at all about `attr-defined`.

**Proven by experiment:** removing `attr-defined` and re-running the full strict command gives:

```
Success: no issues found in 122 source files
```

**Why it matters:** this is exactly the failure mode `scripts/audit_dependencies.py` was written to prevent, one layer up: a suppression that outlived its reason, reading to the next person as "recently reviewed".
`attr-defined` is a broad error code, and it is currently switched off for `mbt.execute.job`, the largest module in the execution path.

**Action:** delete `"attr-defined"` from `pyproject.toml:167` and re-run the strict command.
Consider a periodic check (or a note in `CLAUDE.md`) that per-module suppressions are re-tested when the pins move.

**Impact:** Medium. **Effort:** XS.

### B-3. Coverage of `mbt-spark` and `mbt-h2o` is asserted, never measured

**Evidence:** `pyproject.toml` `[tool.coverage.run] source_pkgs` lists eight packages; `mbt_spark` and `mbt_h2o` are absent.
`docs/v0.1-status.md:24` explains this as "JVM-bound Spark/H2O adapters are covered by the e2e tier instead", and `[tool.coverage.report]` repeats it.

Nothing measures it.
The e2e job runs without `--cov`, so the coverage of 1701 source lines across two published packages (`mbt-spark` 1233, `mbt-h2o` 468) is unknown to CI and to this review.

**Why it matters:** this is the floors-job lesson verbatim.
That job was a no-op for months precisely because it asserted something nobody verified, and the cost was two broken floors and roughly seventy unnoticed advisories.
The same shape is present here, with the difference that nobody has looked yet.

**Action:** run the e2e tier with `--cov` over `mbt_spark` and `mbt_h2o` in `.github/workflows/ci.yml`, publish the number, and set `fail_under` to whatever it actually is today so it can only ratchet up.
If the real number is low, that is the finding and it belongs in `docs/v0.1-status.md` in place of the current claim.

**Impact:** Medium. **Effort:** M.

### B-4. `audit_dependencies.py` names a rot hole it does not close

**Evidence:** `scripts/audit_dependencies.py:95-98`, in the repo's own words:

> The one rot risk this cannot close: if the floor later rises past them, nothing here forces the entry out, because the environment that would prove it (the floors job) deliberately runs with `--no-check-stale`. Raising an h2o floor is the moment to re-read this set.

Three of six acceptances are `FLOOR_ONLY`.
Their whole justification is "these fire at the declared floor and not at the lock", and the floors job is the only environment that can test the first half of that sentence - and it is told not to.
The mitigation is a comment asking a human to remember.

**Action:** add a `--require-floor-only` flag that asserts every `FLOOR_ONLY` id *does* fire in the current resolution, and pass it in the floors job's audit step alongside `--no-check-stale`.
The two flags are complementary: one says "do not demand that ordinary acceptances fire here", the other says "do demand that the floor-only ones do".
An h2o floor that rises past the advisory then fails the floors job with a message telling the operator to delete the entry, which is the behaviour every other acceptance already gets.

**Impact:** Medium (security posture). **Effort:** S.

### B-5. The NFR-03 perf budgets have 50x headroom, measured

**Evidence:** `uv run pytest -q -m perf --durations=10` on this machine:

| Test | Duration (warmup + 3 runs) | Budget | Approx. headroom |
|---|---|---|---|
| `test_parse_under_2s_at_50_resources` | 0.17s | 2s | ~47x |
| `test_compile_under_10s_at_50_resources` | 0.76s | 10s | ~52x |
| `test_per_node_overhead_under_2s` | 0.82s | 2s | - |

The docstring is honest that these are "deliberately LOOSE - a catastrophic-regression guard", and the loose thresholds are the right choice for a shared CI runner.
But at 50x headroom, an accidental O(n²) that makes compile ten times slower still passes, so NFR-03 currently proves almost nothing about compile performance.

**Action:** keep the NFR-03 thresholds as the hard contract, and add a second assertion against a committed baseline measurement with a generous multiplier (3-5x).
Store the baseline in the repo and regenerate it deliberately, exactly the way golden manifests already work, so the regeneration is a visible decision in a diff rather than a silent drift.

**Impact:** Low-Medium. **Effort:** S.

---

## C. Documentation drift

In a repo where `CLAUDE.md` instructs readers to "read the relevant ADR before 'fixing' anything that looks odd", a stale ADR is not a cosmetic problem.
The memory of issue #1 records the lesson already: a stale ADR propagated into the repo's only public gap statement.

### C-1. `README.md` documents the known-broken floors mechanism

**Evidence:** `README.md:145-146`:

> a `floors` job re-resolves every direct dependency at its declared lower bound (`uv sync --resolution lowest-direct`) and runs the suite against that

`CLAUDE.md:39` says the opposite, in bold terms: "Do NOT go back to `uv sync --resolution lowest-direct`: in a virtual workspace root the members' requirements are not 'direct', so it resolves everything to newest and the job silently becomes a duplicate of the `test` job."
`.github/workflows/ci.yml:86-95` and `scripts/install_floors.py` implement the corrected mechanism.

The README is the public face of the repo and currently advertises, as the thing that protects users, the exact command that did not.

**Action:** rewrite `README.md:145-146` to describe `scripts/install_floors.py` and the `--verify` re-assertion.

**Impact:** Medium. **Effort:** XS.

### C-2. ADR-23 and the Snowflake README say the live tier has never run

**Evidence:** `docs/adr/0023-warehouse-batch-scoring.md:44`:

> the `live_snowflake` tier has never had a credentialed run

`packages/mbt-snowflake/README.md:96` repeats the framing ("gated on the first credentialed `live_snowflake` run").

Issue #1, re-scoped 2026-08-29, records the opposite:

> `live_snowflake` had its first credentialed run on 2026-08-28 and the warehouse **read** path is proven end to end. `promote` / `score` / `monitor` are not.

**Action:** amend ADR-23's Verification section to record what the 2026-08-28 run settled and what it did not, and update `packages/mbt-snowflake/README.md:93-97` to the same wording.
The v2 gate is still real; it is just a narrower gate than the ADR currently describes.

**Impact:** Medium. **Effort:** XS.

### C-3. Two stale claims in `docs/v0.1-status.md`

`CLAUDE.md` says this file "markets the project's rigor; keep its claims exactly true".

1. **`:20` (NFR-04)** attributes the per-module mypy relaxations to "the JVM/warehouse adapters".
   The list at `pyproject.toml:155-166` also includes `mbt.execute.job` (mbt-core's largest execution module), `mbt.cli._scaffold.scripts.generate_sample_data`, and three `mbt_adapter_base` modules.
   Six of the nine entries are not JVM or warehouse adapters.
2. **`:24` (NFR-08)** says "750+ tests".
   The actual count is 1247 in the fast suite plus 78 in the e2e tier.
   This one understates rather than overstates, but the file's job is to be exactly true.

**Action:** correct both.
For NFR-04, the honest phrasing names the two error codes and the module list, and after B-2 lands only `no-untyped-call` remains, which the existing pyarrow comment already justifies well.

**Impact:** Medium. **Effort:** XS.

---

## D. Supply chain and release

### D-1. Released wheels carry no build provenance

**Evidence:** `.github/workflows/release.yml` runs `uv build --all-packages` and attaches the artifacts to a GitHub release; there is no `actions/attest-build-provenance` step and no `attestations: write` permission.
A repo-wide grep for `attest|sigstore|cosign` in `.github/workflows/` returns nothing.

**Why it matters:** this is out of step with the rest of the repo's posture.
CI already runs pip-audit, CodeQL, and gitleaks over full history; the showcase goes as far as publishing oras provenance artifacts for its digest-pinned deployable unit and refusing tampered environments.
The artifacts that real users will install are the only ones with no attestation.

**Action:** add to the `release` job, after the build:

```yaml
- uses: actions/attest-build-provenance@v2
  with: {subject-path: 'dist/*.whl,dist/*.tar.gz'}
```

plus `attestations: write` in the job permissions, and note the verification command (`gh attestation verify`) in `CONTRIBUTING.md`'s Releasing section.
`tests/test_release_workflow.py` already pins the permissions envelope, so extend it to cover the new permission.

**Impact:** Medium. **Effort:** S.

### D-2. `mkdocs` is undeclared and unbounded, and MkDocs 2.0 has no migration path

**Evidence:** the dev group declares `mkdocs-material>=9.7` and `mkdocstrings[python]>=0.26` but never `mkdocs` itself; it arrives transitively (`uv.lock` pins 1.6.1).
Every `mkdocs build` prints an upstream warning that MkDocs 2.0 will remove the plugin system, rewrite theming, and ship with "No migration path exists - existing projects cannot be upgraded".

The docs toolchain is therefore held at 1.x purely by luck of resolution.
Renovate's `lockFileMaintenance` is enabled, and `.github/workflows/upstream.yml` deliberately re-resolves to newest - though notably it does not run `mkdocs build`, so the nightly tier that exists to find upstream breakage first would not see this one.

**Action:** declare the cap explicitly, with the reason as a comment, the way `mbt-h2o` already caps h2o:

```toml
# MkDocs 2.0 removes the plugin system and ships no migration path
# (squidfunk.github.io/mkdocs-material/blog/2026/02/18/mkdocs-2.0/).
# The docs site is built on plugins, so 1.x is a deliberate ceiling.
"mkdocs>=1.6,<2",
```

Optionally add `uv run mkdocs build --strict` to `upstream.yml` so the nightly tier covers the docs toolchain too.

**Impact:** Medium. **Effort:** XS.

### D-3. Ten publishable packages and no changelog

**Evidence:** no `CHANGELOG.md` anywhere in the tree.
`.github/workflows/release.yml` relies on `generate_release_notes: true`, which derives notes from merged pull requests - and this repo has none (0 open, and history is direct-to-main).
So a v0.2 release would ship ten packages with effectively empty notes.

**Why it matters:** mbt has an upgrade hazard that specifically requires a changelog.
`docs/v0.1-status.md` and ADR-7 record it: a release that adds spec fields flips config hashes under full-dump hashing, so upgrading users should expect a one-time full-retrain signal.
That is exactly the kind of thing a consumer must read before upgrading, and there is nowhere for it to live.

**Action:** add a generated `CHANGELOG.md` (git-cliff or towncrier), wire it into the release job, and adopt a commit convention it can read.
Reserve a "Retraining impact" section per release for the ADR-7 signal.
Generate rather than hand-maintain, to match the rest of the repo's automation.

**Impact:** Medium. **Effort:** S.

---

## E. CLI and generated-artifact polish

All of these were observed while driving the real CLI through the documented quickstart.
Individually each is small; together they are the entire first impression.

### E-1. `--help` leaks internal requirement IDs pointing at an unmaintained directory

**Evidence:** `mbt --help` renders every command's summary with its spec ID:

```
init     Scaffold a golden-path project (FR-PROJ-01).
run      Build datasets and train models in DAG order (FR-RUN-01).
build    run + test interleaved in DAG order - the CI workhorse (FR-RUN-01).
```

`mbt build --help` adds `--manifest ... (FR-RUN-11)`.
`packages/mbt-core/src/mbt/cli/main.py` carries 23 such references.

`FR-*` and `TSD §*` are defined in `design-history/`, whose own README states: "They describe the project as it was envisioned before v0.1 shipped and are **not maintained**."
So the most-read documentation surface in the product points users at a document the repo tells them not to trust.

**Action:** remove `FR-*` and `TSD §*` from typer docstrings and `--help` strings, keeping them in code comments where they serve maintainers.
Keep the ADR references (`ADR-7`, `ADR-19`, `ADR-20`, `ADR-21`): ADRs are maintained and authoritative, and pointing a user at one is useful.

**Impact:** Medium. **Effort:** S.

### E-2. `ConsoleSink` defaults to stdout, contradicting a load-bearing invariant

**Evidence:** `packages/mbt-core/src/mbt/events/sinks.py:35`:

```python
self.console = console or Console(stderr=False, highlight=False)
```

`CLAUDE.md` lists "Events go to stderr; stdout is command data" among the load-bearing decisions, and `docs/architecture.md` repeats it.
The invariant holds in practice only because `packages/mbt-core/src/mbt/cli/common.py:125` passes `err_console` explicitly - verified by capturing the two streams separately, which cleanly separates the event log from the results table.

The default is a latent trap for anyone embedding `ConsoleSink()` without an explicit console, and it makes the code read as if the invariant were optional.

**Action:** change the default to `Console(stderr=True, highlight=False)` so the invariant holds by construction.

**Impact:** Low. **Effort:** XS.

### E-3. Two clocks in one log stream

**Evidence:** `packages/mbt-core/src/mbt/events/sinks.py:42` formats `event.ts.strftime("%H:%M:%S")`.
`event.ts` is UTC-aware, so `strftime` prints the UTC wall clock with no marker.
Third-party libraries in the same stream print local time.
Observed in a single `mbt build` on a UTC+7 machine:

```
07:09:24  Compiling against target 'dev'
2026/09/01 14:09:26 INFO mlflow.store.db.utils: Creating initial MLflow database tables...
07:09:26  [1/2] START dataset dataset.my_models.churn_training_set
```

Seven hours apart, interleaved, with no indication that either is deliberate.

**Action:** either render local time (`event.ts.astimezone().strftime(...)`) or mark the zone (`%H:%M:%SZ`).
Local time is the better default for a console sink; the JSON-lines sink already carries the full ISO timestamp for machines.

**Impact:** Low. **Effort:** XS.

### E-4. `mbt promote` prints its outcome twice

**Evidence:** `packages/mbt-core/src/mbt/promote.py:101-105` emits a `PromotionApplied` event whose `human()` (`packages/mbt-core/src/mbt/events/models.py:188-190`) renders `promoted {name} v{version} -> {stage}`, and `packages/mbt-core/src/mbt/cli/main.py:751-753` prints the same sentence to `out_console`.
In a terminal:

```
07:10:41  promoted churn_classifier v3 -> production
promoted churn_classifier v3 -> production
```

Both are individually correct under the stdout/stderr split, but the user sees a duplicated line.

**Action:** demote the event to debug level, or give it detail the stdout line does not carry (the resolved version and the gate records consulted), so the two lines say different things.

**Impact:** Low. **Effort:** XS.

### E-5. Console output wrapping is unpolished

Three observations from the same session:

1. **Long paths break mid-token**, so they cannot be copied:
   ```
   07:09:25  Compiled 3 nodes in 0.18s (anchor 2026-09-01T07:09:24Z) ->
   /private/tmp/claude-501/-Users-bb8-code-mbt/bba118a8-1de2-4a58-8d0b-d55babd4218d
   /scratchpad/ux/my_models/target/manifest.json
   ```
2. **Wrapped event messages have no hanging indent**, so continuation lines land in the timestamp gutter and break the visual column:
   ```
   07:10:46  feature_shift most shifted: monthly_usage=0.0568, tenure_days=0.0385,
   support_tickets=0.0274 (top 3 of 5)
   ```
3. **The results table truncates the identifier column** (`dataset.my_models.churn_tra…`) while the `detail` column has slack.

**Action:** pass `overflow="fold"` or `soft_wrap` for path rendering, give `ConsoleSink` a hanging indent equal to the timestamp gutter, and let the Rich table give the node column `ratio` priority over `detail`.

**Impact:** Low. **Effort:** S.

### E-6. The generated docs site has no dark mode

**Evidence:** `target/docs/index.html` and the model card ship hand-written CSS with `--bg:#ffffff` and no `prefers-color-scheme` query anywhere (grep count: 0).
The site is otherwise good - self-contained, an SVG lineage graph, model card with identity, data window, features, hyperparameters, metrics, and slice tables.

**Action:** add a `@media (prefers-color-scheme: dark)` block overriding the six CSS custom properties already declared in `:root`.
The variables are all there; this is a dozen lines.

**Impact:** Low. **Effort:** XS.

### E-7. The scaffold's demo model is weak

**Evidence:** `mbt init` -> `generate_sample_data.py` -> `mbt build` trains a model with `roc_auc=0.6634`, `pr_auc=0.2983`, passing its threshold gate of 0.25.
The README sells this path as "Five minutes to a trained, registered model", and it is the first model any evaluator will see.

A near-random ROC AUC undercuts the demonstration, and it also makes the gate example less instructive: a gate that passes at 0.30 teaches nothing about what a real gate does.

**Action:** strengthen the synthetic signal in `packages/mbt-core/src/mbt/cli/_scaffold/scripts/generate_sample_data.py` to land around 0.85 ROC AUC, and raise the scaffold gate threshold to match.
Keep it honest (the data is clearly synthetic), but make the demo show the tool working well.

**Impact:** Low, but it is the first impression. **Effort:** S.

---

## F. Product direction

### F-1. There is no sklearn adapter, and scikit-learn is already a dependency

**Evidence:** the training adapters are XGBoost, LightGBM, H2O AutoML, and SparkML.
`docs/roadmap.md` places "sklearn / PyTorch adapters" in v1 candidates.

Meanwhile `packages/mbt-adapter-base/pyproject.toml:41` already declares `scikit-learn>=1.5` for the `metrics` extra, and `packages/mbt-adapter-base/src/mbt_adapter_base/metrics.py` imports from `sklearn.metrics` throughout.
So scikit-learn is installed in essentially every real mbt environment already.

**Why it matters:** the current portfolio is optimized for the enterprise JVM path.
That is a defensible strategic bet and the Spark and H2O work is genuinely hard, but it means the single most common tabular modelling stack in the world is the one thing a new team cannot use.
`mbt-lightgbm` is 378 lines and exists explicitly as the extensibility proof, built against public contracts only.
An `mbt-sklearn` of similar size, against a dependency already present, is the cheapest adoption unlock available and would let `mbt init` produce a working model with no extra install at all.

**Action:** ship `mbt-sklearn` covering `LogisticRegression`, `RandomForestClassifier`/`Regressor`, and `HistGradientBoosting*`, against `mbt-adapter-base` only, passing the compliance suite with an exact determinism tier.
Move it from "v1 candidates" to the next release in `docs/roadmap.md`.

**Impact:** High (adoption). **Effort:** M.

---

## G. Repo hygiene

### G-1. 447 KB of review logs in the repo root

**Evidence:** `FEEDBACK.md` is 284 KB and `FEEDBACK_v2.md` is 163 KB.
Together they are roughly fifteen times the size of every other root document combined, and they dominate the file listing a first-time visitor sees.

They are valuable and should be kept: they carry the reasoning behind a large share of the current design.
They are just not root-level documents.

**Action:** move them (and this file, once its log is closed) to `docs/reviews/` or into `design-history/`, which already exists for exactly this kind of frozen provenance and already has a README explaining the convention.
Leave a one-line pointer in `CONTRIBUTING.md`.

**Impact:** Low. **Effort:** XS.

### G-2. The repo-root guard cannot see pre-existing litter

**Evidence:** `conftest.py:36-45` snapshots the root before the session and diffs after, so anything already present is invisible forever.
The snapshot itself is `conftest.py:27-34`.
A `spark-warehouse/` directory dated 2026-08-28 sits in the root right now, and `.gitignore:33-38` lists five JVM leftover patterns (`derby.log`, `metastore_db/`, `spark-warehouse/`, `h2ologs/`, `hs_err_pid*`) under "JVM adapter leftovers (Spark/H2O e2e runs)".

To be fair to the current state: the full e2e tier was run during this review and created **none** of them, so the Spark and H2O tests are writing where they should.
The stale directory is from something else, and the gitignore entries are belt and braces.
The gap is that the guard could not tell you either way.

A second, smaller issue: `conftest.py:28` uses `p.name.startswith(_TOOLING)`, which is prefix matching against a tuple, so a stray entry whose name merely begins with `.coverage` or `__pycache__` is also excluded.

**Action:** add an explicit deny-list of known-bad root entries (the five JVM patterns, `target`, `mlruns`, `mlflow.db`) that fails whenever they are present, not only when they appear mid-session.
Switch `_TOOLING` to exact matching.
Delete the stale `spark-warehouse/`.

**Impact:** Low. **Effort:** S.

---

## What this review checked and found healthy

Stating this explicitly, because a list of findings is not a picture of the repo.

- **Architecture.** The coordinator/job split, the two-phase Jinja capture/resolve design, the adapter contract with its versioned compliance suite, and the manifest-as-pinned-truth model all hold up under reading. `packages/mbt-core/src/mbt/execute/scheduler.py` is 106 lines and correct, including skip propagation and fail-fast cancellation.
- **Test quality, not just quantity.** 1259 test functions, 3065 assertions (2.4 per test), and only 18 with neither an assertion nor a `pytest.raises` - and most of those delegate to helpers that assert internally. This is not coverage chasing.
- **The advisory wrapper.** `scripts/audit_dependencies.py` is the best piece of security engineering in the repo. The bidirectional staleness assertion, the alias-matching fix for pip-audit's unstable primary ids, and the `FLOOR_ONLY` / `misfiled_floor_only` pair are all correct and well tested (16 tests). B-4 is the one hole it names itself.
- **The troubleshooting runbook.** Spot-checked ten distinctive symptom strings from `docs/troubleshooting.md` against package sources; all ten resolve to real emitted messages (several are f-strings, which is why a naive substring search under-reports). The claim that every entry came from a real reproduction survives scrutiny.
- **CI design.** Concurrency groups that never cancel main, an e2e diagnostics upload, the `workflow_call` gate so a release runs the same CI that guards main, the `permissions` envelope comment explaining a real startup failure, and three independent tiers that each open and close their own tracking issue. The comments explaining *which incident* each guard exists for are unusually valuable.
- **No deprecation debt.** The fast suite passes under `-W error::DeprecationWarning`.
- **The loop actually works.** `init` -> `build` -> `promote` -> `score` -> `monitor` -> `docs generate` all exit 0 from a cold scaffold, with correct exit-code semantics, gate evaluation, shift monitoring, and a real prediction store. That is the thing being claimed, and it is true.

---

## Progress log

**Sweep closed 2026-09-01.** Every finding above is implemented. One entry per item: symptom, fix, verification, docs.

Tree state at close, measured on this machine:

| Check | Result |
|---|---|
| `pytest -q -m "not e2e" --cov` | **1349 passed, 66 skipped**, coverage **100.00%** of 9618 statements, 191s |
| `pytest -q -m e2e` with `--cov-config=tests/coverage-jvm.cfg` | **79 passed, 6 skipped**, 451s; JVM adapters **84.09%** against the new enforced floor of 83 |
| `ruff check .` / `ruff format --check .` | clean, 371 files |
| `mypy --strict` (all **11** packages) | clean, 126 source files |
| `yamllint`, `mkdocs build --strict`, `pre-commit run --all-files` | clean |
| `scripts/audit_dependencies.py` | clean, 6 still-earned acceptances |
| `scripts/generate_changelog.py --check` | up to date |
| `mbt init` -> `build` -> `promote` -> `score` -> `monitor` -> `docs generate` | all exit 0; the model card carries its metrics (A-2) and renders in dark mode (E-6) |

The two tiers were run **serially**. Running them concurrently is what produced a spurious 98.8% and then a spurious 20.2% during this sweep: both write the same `.coverage` file, and the clobbered report reads as a coverage failure rather than as a conflict. That gotcha is now recorded in `CLAUDE.md`, because the first instinct on seeing it is to hunt a regression that does not exist.

---

### A-1. `env_var()` tainted every value, corrupting unrelated output

**Symptom.** Redaction is exact-substring, and `env_var()` was the only way to read the environment, so a non-secret value poisoned every serialized surface. Reproduced: `redact("trained 1000 rows in 1.5s, 1 model")` -> `trained ***000 rows in ***.5s, *** model`. The showcase profile carried the workaround as a comment rather than a fix.

**Fix.** Added `env()` as a non-tainting sibling of `env_var()` in all four render contexts (`jinja/environment.py` capture + resolve, `config/profiles.py`, `execute/job.py`). `env_var()` still taints, so the safe path stays the default; `env()` tracks `required_env` identically and is re-resolved in the job subprocess identically. Switched the identifiers, hosts, roots, and URIs in the scaffold profile, the showcase profiles, `docs/spec-reference.md`, `docs/tutorial.md`, and the Snowflake README to `env()`, and deleted the two duplicated warning comments in `examples/showcase/project/profiles.yml` that the fix makes obsolete.

**Verification.** `test_env_does_not_taint_but_env_var_does` pins both directions on the same payload (`0.1234` survives `env()`, becomes `0.***234` under `env_var()`); `test_env_rendering_tracks_required_env_without_tainting` pins the profiles path including `required_env`; `test_render_adapter_ref_env_resolves_without_tainting` pins the job subprocess, where a re-taint would corrupt the result JSON on its way back to the coordinator.

**Docs.** `SECURITY.md` gains a "Choosing between `env_var()` and `env()`" section stating the failure mode in both directions and the tie-break (when in doubt, `env_var()`); `docs/spec-reference.md` gains the comparison table; `docs/concepts.md`, `docs/mlops-alignment.md`, and the scaffold profile header explain the split at the point of use.

### A-2. `run_results.json` was latest-write-wins, so model cards lost their metrics

**Symptom.** `mbt score` and `mbt monitor` rewrote the shared results file with only their own nodes, so `mbt docs generate` afterwards rendered "no run results yet - run `mbt build`" at a user who had just built. Reproduced through the real CLI in the operationally natural order.

**Fix.** Every command now writes `run_results.<command>.json` alongside `run_results.json` with identical content. Added `read_latest_results(path, commands=...)`, which picks the newest matching sibling and falls back to the shared file. `mbt docs generate` and `mbt clean --artifacts-older-than` both ask for the training commands. The docsgen fallback text now names the file it actually looked in instead of telling the user to repeat what they just did. `gc.py` keeps parsing raw JSON rather than the pydantic model, deliberately: `mbt clean` must not hard-fail on a results file another mbt version wrote.

**Verification.** `test_model_card_keeps_metrics_after_a_scoring_run` drives the real orchestrator and asserts the card survives a later command; `test_gc_keep_set_survives_a_scoring_run`, `test_gc_prefers_the_newest_training_command`, and `test_gc_tolerates_an_unreadable_results_file` cover the GC half; re-ran the original CLI reproduction end to end and the metrics table is now present.

**Docs.** `docs/architecture.md` gains "`run_results.json` is latest-write-wins; the siblings are not", stating which consumer should read which.

### B-1. `CONTRIBUTING.md` handed contributors the command that fails CI

**Symptom.** The documented fast-suite command omitted `--cov`, and the page then claimed the suite enforces 100% coverage. A contributor followed it, saw green, and failed CI.

**Fix.** Added `--cov`, plus the `yamllint` and `audit_dependencies.py` steps the list was missing, and corrected the runtime estimate. Added an explicit callout explaining that `fail_under` only evaluates under `--cov`, citing the 99.7% run that once reached main.

**Verification.** The documented command is the one used throughout this sweep; it reports the gate.

### B-2. The `attr-defined` mypy relaxation was dead

**Symptom.** `pyproject.toml` disabled `no-untyped-call` **and** `attr-defined` for nine modules; the comment justified only the first. A broad error code was switched off for `mbt.execute.job`, the largest module on the execution path.

**Fix.** Deleted `attr-defined`. Documented in the same comment block why it went and what would bring it back, so the next reader does not re-add it speculatively.

**Verification.** `mypy --strict` over all 11 packages: clean, 126 source files.

### B-3. Spark/H2O coverage was asserted and never measured

**Symptom.** `docs/v0.1-status.md` claimed the e2e tier covers `mbt-spark`/`mbt-h2o` "instead". The e2e job ran without `--cov`, so the real number was unknown - the same shape as the floors job that was a no-op for months.

**Fix.** Measured it: **84.1%** (792 statements, 126 missed) on the full e2e tier. Added `tests/coverage-jvm.cfg` carrying that first measurement and a `fail_under = 83` floor, and wired `--cov --cov-config=tests/coverage-jvm.cfg` into the e2e job. The header records the per-module numbers and explains why the three subprocess-only modules read 0% (spark-submit is not traced, the same reason `mbt/execute/job.py` is held at 100% by in-process unit tests).

**Verification.** The e2e tier runs green under the floor. `test_the_jvm_coverage_floor_is_real_and_enforced` guards both halves that could decouple - the config declaring those packages, and the workflow actually passing it - and `test_the_jvm_packages_are_outside_the_fast_suite_gate` pins the non-overlap that makes the 100% gate reachable.

**Docs.** `docs/v0.1-status.md` NFR-08 now says the substitution is measured, not asserted.

### B-4. The `FLOOR_ONLY` rot hole the audit script named itself

**Symptom.** `scripts/audit_dependencies.py` documented that nothing could force a floor-only acceptance out, because the only environment that could judge it (the floors job) runs `--no-check-stale`. Three of six acceptances were floor-only, mitigated by a comment asking a human to remember.

**Fix.** Added `unearned_floor_only()` and a `--require-floor-only` flag that fails when a floor-only entry does **not** fire, and passed it in the floors job alongside `--no-check-stale`. The two flags are complementary: one says "do not demand ordinary acceptances fire here", the other says "do demand the floor-only ones do". A risen floor now fails that job with an instruction to delete the entry, which is the treatment every other acceptance already got.

**Verification.** `test_a_floor_only_entry_that_stopped_firing_at_the_floors_fails` and `test_require_floor_only_is_opt_in` cover both directions, including that the same report is judged differently by the locked and floors environments. The real locked audit still passes with all six acceptances earned.

### B-5. The perf budgets had 50x headroom

**Symptom.** Measured: parse 0.17s against a 2s budget, compile 0.76s against 10s. A test whose stated purpose is catching an accidental O(n²) could not see a 10x regression.

**Fix.** Kept the NFR-03 thresholds (they are the published contract, and a tight wall-clock budget on a shared runner flakes) and added `test_parse_scales_linearly` / `test_compile_scales_linearly`. They measure the same operation at 20 and 80 models on the same machine in the same run and assert growth stays under 8x for a 4x size increase. Machine speed cancels out of a ratio, so there is no baseline to commit and nothing to recalibrate per runner - a rejected wall-clock baseline precisely because it bakes in one machine's speed.

**Verification.** Observed growth 2.84x (parse) and 1.05x (compile) against the 8x limit, with quadratic landing at ~16x: real margin against flakes in one direction and against regressions in the other.

### C-1, C-2, C-3. Documentation drift

- **C-1.** `README.md` advertised `uv sync --resolution lowest-direct` as the floors mechanism - the exact command `CLAUDE.md` forbids because it silently does nothing in a virtual workspace root. Rewritten to describe `scripts/install_floors.py` and its `--verify` re-assertion. Same correction applied to `docs/v0.1-status.md` NFR-05.
- **C-2.** ADR-23 and the Snowflake README said the `live_snowflake` tier "has never had a credentialed run"; issue #1 records the first one on 2026-08-28. Both now state what that run settled (the warehouse **read** path, end to end) and what it did not (`promote`/`score`/`monitor`), so the v2 gate is recorded as narrower rather than gone.
- **C-3.** `docs/v0.1-status.md` NFR-04 attributed the mypy relaxations to "the JVM/warehouse adapters" when six of nine modules are neither; it now names the single remaining error code and its reason. NFR-08's "750+ tests" is now the real count. Package counts across `CONTRIBUTING.md`, `test_version_sync.py`, and `test_release_workflow.py` moved 10 -> 11 with the new adapter.

  Worth recording how those were found: all three, plus `test_wheel_install.py`, **failed on their own** when the eleventh package landed. That is the repo's drift guards working exactly as designed, and it is why the counts were stale nowhere. `test_wheel_install.py` now derives the expected wheel count from the workspace instead of hard-coding it, since the invariant it tests is "every package ships its PEP 561 marker", not "there are ten packages".

### D-1. Released wheels carried no provenance

**Symptom.** CI scanned dependencies, sources, and git history, and the showcase published oras provenance for its deployable unit - but the artifacts that actually leave the repo had none.

**Fix.** Added `actions/attest-build-provenance@v2` over `dist/*.whl` and `dist/*.tar.gz`, ordered before the GitHub release so a failure cannot ship unattested wheels, plus the `attestations: write` permission it needs.

**Verification.** `test_every_published_artifact_carries_build_provenance` pins the subject set and the step ordering; `test_the_attestation_permission_is_actually_granted` pins the permission, which is not implied by `contents: write` and would otherwise fail at run time - the same class of error that made the first v0.1.0 tag fail at startup.

**Docs.** `CONTRIBUTING.md` Releasing gains the `gh attestation verify` command.

### D-2. `mkdocs` was undeclared and unbounded

**Symptom.** `mkdocs` arrived only transitively through `mkdocs-material`, so the announced MkDocs 2.0 - which removes the plugin system and ships with "no migration path exists" - had nothing to stop it.

**Fix.** Declared `mkdocs>=1.6,<2` with the reason and the upstream link as a comment, in the same style as the h2o cap.

### D-3. Ten packages, no changelog

**Symptom.** No `CHANGELOG.md`, and `generate_release_notes: true` derives notes from merged pull requests - of which this repo has none, so a release's notes came out empty. The upgrade hazard that most needs writing down (ADR-7: a release adding a spec field flips every config hash and signals a full retrain) had nowhere to live.

**Fix.** Added `scripts/generate_changelog.py`, generating `CHANGELOG.md` from git tags and commit subjects - generated, not hand-written, matching the repo's automation-first stance and CONTRIBUTING's existing position. Every release section carries a **Retraining impact** line from a maintainer-curated map, defaulting to the conservative answer when unrecorded. `release.yml` verifies it with `--check` at tag time rather than on every push: the "Unreleased" section legitimately moves with each commit, and only at a tag is it empty and the check stable.

**Verification.** 13 tests covering the generator's invariants (every tag gets a section, every section states impact, curated keys reference real tags, `--check` agrees with what it just wrote, the entrypoint works as a subprocess).

**Docs.** `CONTRIBUTING.md` gains "The changelog" with the two pre-tag steps.

### E-1..E-7, G-2. CLI and generated-artifact polish

- **E-1.** `--help` leaked 23 internal `FR-*` / `TSD §*` IDs pointing at `design-history/`, which its own README declares unmaintained. Removed from every user-visible docstring and help string; ADR references stay, because ADRs are authoritative and pointing a user at one is useful.
- **E-2.** `ConsoleSink` defaulted to stdout, contradicting the "events go to stderr" invariant; it held only because the CLI passed an stderr console at the one call site. The default is now stderr, so the invariant holds by construction.
- **E-3.** Console timestamps were UTC with no marker, interleaved with third-party lines in local time (observed 7 hours apart in one `mbt build`). Now rendered via `astimezone()`; machines still read the full ISO timestamp from the JSON sink.
- **E-4.** `mbt promote` printed its outcome twice, once as an event and once as command data. `PromotionApplied` is now a debug-level event, so the JSON stream and `-v` still carry it while the terminal shows one line. A FORCED promotion keeps its own warn-level message.
- **E-5.** Long paths wrapped mid-token (uncopyable) and continuation lines restarted in the timestamp gutter. Added explicit wrapping that breaks at word boundaries and hangs continuations under the message, with a narrow-terminal escape.
- **E-6.** The generated docs site had no dark mode. Every colour was already a CSS variable except four literals; those are now variables too, and a `prefers-color-scheme: dark` block overrides the palette. The lineage SVG arrowhead follows the edge colour.
- **E-7.** The quickstart's demo model scored 0.66 ROC AUC / 0.30 PR AUC - the first model anyone sees, and it looked like the tool could not learn. The sample data now uses a logistic generative model over tenure, usage, tickets, and plan tier (plan tier previously carried no signal at all, yet the card slices on it). Measured after the change: **0.81 ROC AUC, 0.50 PR AUC** at a 20% base rate. The example gate moved 0.25 -> 0.40 so it teaches something, and `test_cli_basics.py` now reads the threshold from the scaffold instead of asserting a literal.
- **G-2.** The repo-root guard was a before/after diff, so any leftover already present was invisible forever - and a `spark-warehouse/` from 2026-08-28 was sitting there. Added an explicit deny-list of never-legitimate root entries checked at session start as well as end, and switched `_TOOLING` from prefix to exact matching. The new check found and reported that exact directory on its first run; it has been deleted.

### F-1. No sklearn adapter

**Symptom.** The adapter portfolio covered XGBoost, LightGBM, SparkML, and H2O AutoML, and put scikit-learn in "v1 candidates" - while `mbt-adapter-base[metrics]` already depends on scikit-learn to compute PR-AUC, so every install that evaluates a model already had it. The single most common tabular stack was the one a new team could not declare.

**Fix.** Shipped `packages/mbt-sklearn`, against `mbt-adapter-base` only. Four estimators over two tasks: `logistic`/`linear` (the interpretable baseline), `random_forest`, and `hist_gradient_boosting`. Exact determinism tier, `n_jobs=1` by default with a warning when raised. Two decisions worth recording:

- **Encoding is a property of the model family, not the data.** Trees take ordinal codes; the linear estimators one-hot, because a linear model reads an ordinal code as a magnitude. The compliance suite's mixed fixture proves this is not theoretical - its levels sort to east/north/south with positive rates .5/.92/.08, so ordinal codes there produce a model that cannot learn. `feature_importance` folds one-hot columns back onto their source feature.
- **`HistGradientBoosting*` reports `{}` rather than zeros.** It exposes neither `coef_` nor `feature_importances_`, and the contract has a documented escape hatch for exactly that; a row of zeros would be a fabricated ranking.

`param_model(task)` returns one model per task, so the per-estimator models sit behind a union that validates the supplied keys against the *concrete* estimator - naming a `random_forest` knob on a `logistic` model is a parse-time error with a field path, not a `TypeError` an hour into a build. `penalty` is forwarded only when explicitly set, so the default path does not trip scikit-learn 1.8's deprecation.

**One upstream change was required**, and it is a genuine generalization: `TrainingAdapterCompliance` gained `regression_hyperparameters`, defaulting to `valid_hyperparameters`. Every existing adapter takes the same knobs for both tasks; an adapter that selects an estimator in the spec cannot, and would otherwise have been untestable for one of its two declared tasks.

**Verification.** The compliance suite runs three times, once per estimator, because a linear model and a tree ensemble differ in what they can express - 61 tests, `mbt-sklearn` at **100%** line coverage. `test_no_core_imports` holds (zero mbt-core imports). `tests/test_adapter_swap.py::test_switching_to_sklearn_only_touches_the_spec` proves G4 a second time through the real CLI: a spec edit, no core changes, `sklearn_joblib` artifact, `class_weight` auto-resolved to `balanced`.

**Docs.** New package README; `README.md` layout table and the dbt comparison; `docs/roadmap.md` moves sklearn from "v1 candidates" to shipped and leaves PyTorch; `docs/v0.1-status.md` G4 reads "Met, twice"; `CLAUDE.md`, `CONTRIBUTING.md`, `.pre-commit-config.yaml`, and `ci.yml` all carry the 11th package.

### Bonus: `mbt init` crashed on any `__pycache__` in the scaffold tree

Not in the review - found while verifying E-7's new sample-data generator, which is exactly the kind of accident worth writing down.

**Symptom.** Importing `scripts/generate_sample_data.py` (to check the Bayes-optimal ceiling of the new signal) left a `.pyc` beside it in the scaffold template. The next `mbt init` died with:

```
Internal error: UnicodeDecodeError: 'utf-8' codec can't decode byte 0xa7 in position 0
  hint: this is a bug in mbt; please report it with the command you ran.
```

**Why it matters.** The template ships as *source*, so in an editable or checked-out install anything that imports that script leaves bytecode behind - and `mbt init` is the first command a new user ever runs. The scaffold walk read every file as text, so one stray `.pyc` turned the golden path into "this is a bug in mbt".

**Fix.** `_walk` in `cli/scaffold.py` now skips `__pycache__`/cache directories and `.pyc`/`.pyo` files, with the incident recorded at the constant.

**Verification.** `test_init_ignores_bytecode_left_beside_the_template` plants a non-UTF-8 `.pyc` in the real template and asserts `mbt init` succeeds and does not copy it. Confirmed it fails without the fix (`UnicodeDecodeError`) and passes with it.

### Post-commit: two defects the sweep itself shipped, found on the first CI run

Both were invisible locally and turned main red on the commit that closed this review. Recording them because each is an instance of a pattern this review is about: a check that passes in the environment it was written in and nowhere else.

**D-3 asserted the thing its own design says not to assert.** `test_the_script_runs_as_a_subprocess` ran `generate_changelog.py --check` and demanded exit 0. But `--check` compares the whole file, and the "Unreleased" section gains a line on every commit - which is exactly why `release.yml` runs it at tag time, as its own comment explains. So the fast suite was guaranteed to go red on the first commit after any regeneration, and did: the sweep commit. The test now asserts the entrypoint *reaches a verdict* rather than which verdict (a bad flag still exits 2, and a crash exits 1 with no verdict printed, so both are still caught). Whole-file agreement stays where it is stable: `release.yml` at the tag, and `test_the_committed_changelog_is_in_sync_for_released_tags` for the part below the newest tag.

**The changelog guards read git history that CI does not check out.** `actions/checkout` clones depth-1 with no tags, so `released_tags()` returned `[]` and five more tests failed - three of them with `IndexError`, which reads as a generator bug rather than as a checkout that fetched nothing. The changelog is generated *from* git, so these guards have no other source of truth: the fix is to give CI the history, not to weaken them. `fetch-depth: 0` on the two jobs that run the fast suite (`test`, `floors`), the shared `_NEEDS_TAGS` message so a tagless clone says so, and `test_every_job_running_this_suite_checks_out_tags` pinning the workflow half - the same two-halves-could-decouple shape as B-3's coverage floor.

**Also fixed: the guard hardened in G-2 was blind to the file `CLAUDE.md` tells you to create.** `COVERAGE_FILE=.coverage.<name>` is the documented way to run two tiers without clobbering one `.coverage`. That name is not in `.gitignore` (which lists `.coverage` exactly) and not in the guard's now-exact `_TOOLING` set, so it dirtied `git status` and the session guard never fired - pytest-cov writes the file after the fixture's teardown. Added `.coverage.*` to both, as an explicit pattern rather than by loosening `_TOOLING` back to prefix matching. Coverage's own parallel mode writes `.coverage.<host>.<pid>.<random>`, which the same pattern covers.

### G-1. 447 KB of review logs in the repo root

**Symptom.** `FEEDBACK.md` (284 KB) and `FEEDBACK_v2.md` (163 KB) dominated the root file listing at roughly fifteen times every other root document combined.

**Fix.** Moved to `design-history/reviews/{feedback-v1,feedback-v2}.md` - outside `docs/`, so 447 KB of review log does not land on the published site, and inside the directory that already exists for frozen provenance. Kept rather than deleted because code comments cite them by section (`FEEDBACK 2.6`, `R2-7`, `F17`). `design-history/README.md` gained a "Closed review cycles" section explaining the convention; `CLAUDE.md` now points at the in-flight review at the root and says closed ones move here.

**This file moved there too**, on 2026-09-03, once its own progress log closed and the sweep was on `main` and green - it is `design-history/reviews/feedback-v3.md`, which is what you are reading. Its findings are cited in code as `FEEDBACK v3 <id>`.

---

## Residual, and why

Two things in this document were deliberately **not** closed, because neither is a code change this repo can make:

- **The `live_snowflake` serving leg** (C-2, issue #1). `promote`/`score`/`monitor` against a real Snowflake account still needs a credentialed run; the ADR-23 v2 native prediction store stays gated on it, which is the right call for a stateful table store whose correctness rests on real `MERGE`/transaction semantics.
- **PyPI publication** (`docs/v0.1-status.md` NFR-10). Still an organizational decision. The release workflow is inert-but-green until the repo variable is flipped, and now attests every artifact it builds.
