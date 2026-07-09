# FEEDBACK: senior MLOps and data science review of mbt

Review date: 2026-07-06, against `main` (fb2a954).
Perspective: a senior MLOps engineer and a senior data scientist evaluating the repo for production readiness.
Framework: findings are mapped to the MLOps principles published at [ml-ops.org](https://ml-ops.org/content/mlops-principles) (Automation, Continuous X, Versioning, Testing, Reproducibility, Monitoring, Loosely Coupled Architecture).
Every finding carries file evidence and a priority: P1 (must fix), P2 (should fix), P3 (nice to have).

## Review status: CLOSED (2026-07-07)

Every finding in this review is now closed or explicitly deferred with a reason.
A section-by-section completeness audit (final loop iteration) confirmed the map:

- Sections 1.1-1.5, 2.1-2.2, 2.6-2.8, 3.2-3.9 and all twelve action items: closed - see the progress log below for each item's fix, tests, and docs.
- Closed by automation rather than one-time bumps: GitHub Action major versions (1.2) are Renovate's job (`renovate.json`); bumping majors blind with no way to execute the workflows locally was judged riskier than stale-but-working pins. Prerequisite: install the Renovate app on the hosting repo.
- The audit itself surfaced and fixed the last two unclosed section findings: `mbt deps` loose installs (2.2 - now prefers the project's pinned requirements.txt, warns on the unpinned fallback, and verifies packages.yml against the installed environment either way) and the quickstart's "drift check" mislabel (2.4 - now honestly "champion decay check on fresh data").
- Deferred by scope, deliberately (each is roadmap-scale, not a gap to patch): regression/multiclass verticals (3.1), cross-validation (3.2/3.5), post-hoc calibration step (3.8), OTel export (2.4, documented in v0.1-status), first-class rollback tooling (2.3), transformed-secret redaction (2.5), release/PyPI pipeline (2.1, organizational). JVM flaky-rerun tooling (2.7) waits for observed CI flakes; adding reruns beforehand hides real failures.
- Reopened and closed after the review (2026-07-07): the two biggest deliberate scope cuts - "no serving, batch scoring, or inference surface at all" (2.3) and "monitoring, the sixth ml-ops.org principle, is essentially absent ... no statistical drift detection exists anywhere" (2.4, top-5 item 5) - are now shipped as batch scoring pipelines with distribution-shift monitors and delayed ground-truth evaluation (ADR-20/21); see the progress-log entry below. Online (request/response) serving and OTel export remain the documented non-goals/deferrals.

## Progress log

- 2026-07-08: VERIFIED CLEAN (no change) - `mbt-spark` adapter code review (the largest previously-unreviewed package, 721 LOC).
  Data materialization writes distributed (`frame.coalesce(1).write.parquet`) rather than collecting to the driver, moves the single part file into place, and cleans the staging dir in a `finally`; `from_locator` reopens without a Spark session; snapshot mismatch is a hard error. The one `toPandas()` (`training.py` `_scores`) collects a single prediction column on a BOUNDED train/test split for the shared metric engine - the necessary collect-for-metrics, not an unbounded batch. Session is a per-subprocess `getOrCreate` singleton torn down on subprocess exit (no leak). No defect found.
  Verified: full fast suite green (exit 0, 312 tests); no code change.
- 2026-07-08: VERIFIED SAFE (no change) - concurrency / `--threads` thread-safety.
  New lens: the scheduler runs node runners in a `ThreadPoolExecutor`. Its own state (`results`, `remaining`, `stop_scheduling`) is mutated only under its lock, and training runs in subprocesses (ADR-3), so the parallel work touches no shared Python state. The one shared-mutable spot reachable from the pool threads is `ExecutionContext.next_index` (`self._counter[0] += 1`), a non-atomic read-modify-write that feeds only the cosmetic `[i/N]` progress index.
  Reproduction attempt (bug-repro-first): 16 threads x 5000 increments of the exact `list[0] += 1` pattern, even under a forced `sys.setswitchinterval(1e-9)`, produced zero duplicates and zero lost updates (final count exact) - CPython does not interleave this pattern in practice, so the theoretical race is unreachable and its impact would be cosmetic regardless. Deliberately left as-is rather than churn working, tested code for a race that cannot be reproduced.
  Verified: full fast suite green (exit 0, 312 tests); no code change.
- 2026-07-08: DONE - `mbt --version` was missing (errored `No such option: --version`); added it, and profiled startup.
  Found by opening a new angle - CLI startup profiling: `mbt --version`, the affordance users reach for first (bug reports, CI, "what am I on"), errored with `No such option: --version` (exit 1). Only `evaluate`/`promote` carried a (registry) `--version`; the root never did.
  Fix: a root `@app.callback()` with an eager `--version` option prints `mbt {__version__}` and exits 0; `no_args_is_help` and every existing command are unaffected.
  Verified: `mbt --version` -> `mbt 0.1.0` (exit 0); `mbt` with no args still shows help; a regression test proves it works with NO project (the eager callback fires before any project load); ruff check + format clean; mypy --strict clean; full fast suite green (exit 0, 312 tests); mkdocs --strict green; the CLI-doc drift guard stays green.
  Observation (measured, deliberately not acted on): startup is ~0.34s and `cli.common` eagerly imports `artifacts.manifest` -> pyarrow/networkx/numpy, so even `--version`/`--help` pay ~250ms of import. ADR-14's rule keeps the ML FRAMEWORKS (xgboost/mlflow) off the path - confirmed absent from `-X importtime` - but the data/graph libs load eagerly. 0.34s is fine for a CLI; deferring those imports in `cli.common` would shave it but is a nontrivial, risky refactor - logged as a low-priority option, not a defect.
  Docs: cli-reference documents `mbt --version`; this entry.
- 2026-07-08: VERIFIED CLEAN (no change) - `mbt ls`, `run-operation`, and `state diff` outputs and error paths are correct; user-journey verification is now broad enough to call the surface validated.
  `ls` renders a clean table and well-formed JSON; `run-operation recent_window '{days: 7}'` yields `-7d:now` (exit 0) and an unknown macro errors with an available-macros hint (exit 1); `state diff` shows "no node changes" when unchanged and, after a `max_depth 4->6` edit, correctly flags `model...churn_classifier | config` plus the transitive `scoring...churn_scoring | upstream`.
  Session journey-verification map (all driven live end-to-end, output/exit-code inspected): init -> generate-data -> parse -> build -> show -> docs generate -> promote (direct and GitOps `--from-file`) -> score -> monitor; error paths: gate breach (exit 2), missing champion / bad version / bad model / unknown macro (exit 1); all clean after the fixes logged above (vacuous scaffold gate, float count display, generic gate message, `__mbt_auto__` sentinel in docs, redundant error-detail line, quickstart data mismatch).
  The product is polished; the user-journey vein has effectively reached zero new findings. Verified: full fast suite still green (exit 0, 311 tests); mkdocs --strict green.
- 2026-07-08: DONE - quickstart understated the sample-data generator (doc/behavior mismatch, caught by driving the tool).
  Verified the quickstart against the actual init -> build -> promote -> score -> monitor flow: the steps match reality (the `~/.mbt/profiles.yml` install claim is real - `scaffold.py` writes it; promote/score/monitor messages and exit codes match what I observed). One inaccuracy: the "Get data" step said the generator "writes deterministic sample subscriber data to `data/subscribers/`", but it also writes `data/scoring_batch/` and `data/churn_outcomes/` - the very data the score (step 9) and monitor steps consume. A reader following the quickstart top-to-bottom would reach `mbt score` without knowing where the batch came from.
  Fix: the "Get data" step now names all three datasets and ties them to the later score/monitor steps.
  Verified: `mkdocs build --strict` green; the generator's three output paths (subscribers/scoring_batch/churn_outcomes) confirmed in the scaffold script.
  Docs: quickstart.md.
- 2026-07-08: DONE - errored-node results-table detail repeated a truncated `resource:` line; the promote/score error paths are otherwise solid.
  Drove the operational error paths as a user (score with no production champion; promote a non-existent version/model; GitOps `--from-file` promotion): all have clear messages, actionable hints, and correct exit codes (missing champion / bad promote = exit 1; `--from-file` applies real promotions, the empty scaffold file no-ops cleanly, then score succeeds). One self-caught false alarm worth noting: an apparent "promote exits 0 on error" was `tail`'s exit read through a pipe (the exact CLAUDE.md trap) - the real exit is 1.
  The one real wart: an errored node's `NodeResult.message` is `str(MbtError)` (message, then `resource:` and `hint:` lines), and the table rendered `message[:100]`, so the detail column repeated a truncated `resource: scoring.p.churn_s` - redundant with the node column and cut mid-word, while the event log already prints the fully-formatted error.
  Fix: the table detail shows the message's first line only; resource stays in the node column and the hint stays in the log and the JSON `message`.
  Verified live: a fresh missing-champion score now shows a clean `no champion of 'churn_classifier' in stage 'production' to score with` detail; a regression test renders an errored node and asserts the resource/hint tail is dropped (targeted falsification: reverting only this line fails it at the assertion); ruff check + format clean; mypy --strict clean over mbt-core; full fast suite green (exit 0, 311 tests).
  Docs: this entry.
- 2026-07-08: DONE - the published model card leaked the internal `__mbt_auto__` sentinel instead of `auto`.
  Found by rendering the generated docs and actually reading the HTML (not the code): the Hyperparameters table showed `scale_pos_weight __mbt_auto__ 3.622865` - the raw AUTO sentinel the manifest keeps verbatim (ADR-12) leaked into the human-facing card that deploys to GitHub Pages. (The manifest and `mbt show` keeping the sentinel is correct - those are raw config dumps; the docs card is a presentation layer and should show the keyword the user wrote.)
  Fix: the card's hyperparameter value column renders the AUTO sentinel as `auto`; the separate "resolved auto" column still shows the concrete resolved number (e.g. 3.622865).
  Verified live: the card now reads `scale_pos_weight auto 3.622865` with zero `__mbt_auto__` remaining; a regression test builds a model with `scale_pos_weight: "{{ auto }}"` and asserts the card shows `<code>auto</code>` and no sentinel (falsification-proven).
  While there, inspected the rest of the rendered site and found it clean: identity hashes, data-window splits, feature-importance percentages, metrics, per-slice metrics, gate history, and the index lineage SVG + model table all render with real values and no `None`/`nan` leakage.
  Verified: ruff check + format clean; mypy --strict clean over mbt-core; full fast suite green (exit 0, 310 tests).
  Docs: this entry.
- 2026-07-08: DONE - gate-failure messages were generic ("one or more gates failed") in the results table, JSON, and PR comment, unlike the monitor path.
  Found by driving the gate-breach error path as a user (`build` with an unachievable floor): the scrollback log named the gate (`gate pr_auc (threshold): FAIL - expected 0.99, got 0.3161`), but the results-table detail and `NodeResult.message` - the field the GitOps PR-comment script surfaces to reviewers - said only "one or more gates failed". The monitor path already reports `gate breach: ...` specifically; model-build and test gates did not.
  Fix: `_gate_failure_summary` in `runners.py` renders the failing gate(s) - threshold (`pr_auc=0.3161 failed threshold 0.99`), champion (`challenger delta lower bound ... < required ...`), with the slice named for slice gates - and falls back to the generic message only if a failing gate has no comparable fields; both the model-build and test call sites use it.
  Verified live: a gate breach now shows `gate breach: pr_auc=0.3161 failed threshold 0.99` in the detail column; exit code stays 2 (bad-flag stays 1); unit tests pin threshold/champion/slice/fallback; ruff check + format clean; mypy --strict clean over mbt-core; full fast suite green (exit 0, 309 tests).
  Docs: this entry.
- 2026-07-08: DONE - results-table polish: integer count metrics (`rows_scored`) rendered as `460.0000`.
  Found by driving the new score/monitor flow as a user (init -> build -> promote -> score -> monitor, all green): `mbt score` printed `rows_scored=460.0000`. The shared results-table detail column formats every metric with `.4f`, but `rows_scored` (`runners.py`) is an integer count stored as a float metric, so a count showed spurious sub-integer precision.
  Fix: `_format_metric` in `cli/common.py` renders whole-number metric values as integers and keeps four decimals for genuine fractional metrics (pr_auc etc. unchanged); the JSON `run_results` is untouched (metrics stay floats for machine consumers).
  Verified live: a re-score now shows `rows_scored=460`, and monitor's realized `pr_auc=0.3988 roc_auc=0.7145` still render with four decimals; unit tests pin both cases (count-as-int, fraction-4dp).
  Bonus from the same journey: confirmed the whole new-capability loop works end to end - promote staging->production, `mbt score` (460 rows), and `mbt monitor` at a matured anchor evaluates 2 of 2 runs with a passing realized-metric gate (now meaningful at 0.25 thanks to the scaffold fix below).
  Verified: ruff check + format clean; mypy --strict clean over mbt-core; full fast suite green (exit 0, 305 tests).
  Docs: this entry.
- 2026-07-08: DONE - the `mbt init` scaffold shipped the same vacuous gate the demo fix removed: `pr_auc_floor: 0.05`, below the base rate, can never fail.
  Found by driving the product as a brand-new user (init -> generate_sample_data -> build), not by reading code: the build printed `gate pr_auc (threshold): PASS - expected 0.05, got 0.3161` with a ~22% positive label balance - so the 0.05 floor sits below the random-baseline PR-AUC (~0.22) and gates nothing. FEEDBACK #160 raised the DEMO's floor 0.05->0.3 for exactly this reason, but the scaffold - what EVERY `mbt init` stamps into a user project - was missed, so every new project started with gate theater and taught the anti-pattern.
  Fix: scaffold `pr_auc_floor` 0.05 -> 0.25 with a teaching comment (sample data ~22% positive, model scores ~0.32, so 0.25 is meaningfully above chance with headroom to spare); `test_cli_basics` show-config assertion updated 0.05 -> 0.25.
  Verified end-to-end: a fresh `mbt init` + `mbt build --target prod` now prints `PASS - expected 0.25, got 0.3161` (meaningful AND green); full fast suite green (exit 0, 303 tests); the scaffold e2e state-loop (real XGBoost prod build under the new gate) passes; yamllint clean; no stale doc references to 0.05.
  Docs: scaffold `mbt_project.yml` comment (teaches why a gate must clear the base rate); this entry.
- 2026-07-08: VERIFIED ROBUST (no change) - three core-correctness invariants audited, all sound; recorded so future sessions skip the re-audit.
  (1) Determinism / byte-reproducibility: `canonical_json` sorts keys, strips whitespace, and its `default` RAISES on any non-JSON type - so a stray `set` in a config fails loudly at hash time rather than hashing non-deterministically; `input_hash` sorts upstream hashes, `env_digest`/`env_freeze_digest` sort, `manifest_hash` blanks volatile metadata. The `set()` usages in compile/specs are internal dedup/validation, never serialized.
  (2) CLI path resolution (CLAUDE.md invariant "every path option must resolve against the invocation cwd before the chdir"): airtight - `make_ctx` absolutizes `project_dir` and `profiles_dir`; the `_register_execution_command` factory resolves `state`+`manifest` for run/build/test/score; and monitor/evaluate/state-diff/docs/promote each resolve theirs. The new (uncommitted) score/monitor commands correctly inherit/apply it - the class of gap that produced the duckdb floor bug, but here clean.
  (3) Paired-bootstrap champion gate (ADR-18): statistically correct - genuinely paired (both models scored on the same resampled indices so sampling noise cancels), direction-aware delta (challenger-improvement always positive), one-sided lower bound = the (1-confidence) quantile (percentile method), single-class resamples skipped with a documented point-delta fallback, seeded for reproducibility.
  Verified: full fast suite green (exit 0, 303 tests); no code change.
- 2026-07-08: VERIFIED SAFE (no fix, and deliberately not the obvious one) - the MLflow/tracking backend, the 4th candidate leak vector, does not carry env_var() secrets in practice.
  Closing the security audit: checked whether a spec `env_var()` value reaches the tracking backend (a stored, shareable surface). It does not, for two reasons proven by reproduction and code:
  (1) the only spec data logged as params is `spec.hyperparameters`, which is schema-validated to typed fields per adapter - injecting an arbitrary `api_token: "{{ env_var(...) }}"` fails parsing outright (`hint: valid: fail_training, fake_metric_value, learning_rate, max_depth, scale_pos_weight`), so free-string secrets can't ride a hyperparameter; and
  (2) the free-string spec fields that CAN hold env_var (owner, description) are never logged - `tracking_meta` tags are hashes/IDs only (`mbt.config_hash`, `mbt.input_hash`, `mbt.run_id`, `mbt.git_commit`), and metrics are computed floats.
  Crucially I did NOT add the "obvious" `redact()` at the `tracking.log(params=...)` call: training runs in a subprocess that re-taints only ADAPTER configs (`_render_adapter_ref`), NOT spec fields, and the job payload is written unredacted (`compute.py`), so a redact there would find an empty taint set - false security. The real guarantee is (1)+(2), not a redact call.
  Guard-rail for future work: do not start logging a free-string spec field (owner/description/tags) to the tracking backend without first solving the job-side taint (the job cannot redact a spec secret it never tainted). Coordinator-side surfaces (manifest, `mbt show`, docs) are safe because compile taints there; the job subprocess is the exception.
  Verified: reproduction confirmed the parse rejection; full fast suite green (exit 0, 303 tests) after removing the scratch repro. No code or doc change beyond this entry.
- 2026-07-08: DONE - SECURITY (follow-on #2): `mbt docs generate` leaked env_var() secrets into the published docs site (P2) - the highest-blast-radius vector.
  Third path in the same audit: `docsgen/generator.py` renders spec config (description, owner, hyperparameters, features) into HTML model cards with `html.escape` (XSS) but no `redact`. The default `mbt docs generate` path (no `--manifest`) calls `compile_project` fresh (`cli/main.py`), so `generate_docs` receives the RAW in-memory manifest (redaction happens only on the manifest *file* write) - and the site deploys to GitHub Pages, so a leak here is world-readable. (The `--manifest` path reads the already-redacted file, so it was safe.)
  Fix: redact the two assembled-page write choke points (`_model_card` and the index `_page`) rather than each field, so every current and future config-derived value is covered, mirroring how the event sinks redact the final line.
  Regression test (`test_docsgen.py`): injects `env_var('MBT_DOCS_SECRET')` into the demo model's `owner`, compiles fresh via `compile_demo` (the in-memory-raw path the CLI default uses), generates docs, and asserts the token never reaches the card or index; proven non-vacuous by falsification.
  Verified: docs redaction test passes and fails-without-fix; ruff check + format clean; mypy --strict clean over mbt-core; full fast suite green (exit 0, 303 tests).
  Docs: `secrets.py` docstring now lists the docs site among the redacted paths; this entry.
- 2026-07-08: DONE - SECURITY (follow-on): `mbt show` was the second unredacted output path - it dumped compile-rendered spec config to stdout raw (P2).
  Continuing the previous entry's audit of user-facing output: `env_var()` is available in the spec/source jinja resolve context and returns a tainted value (`jinja/environment.py:213`), so a spec field like `owner: "{{ env_var('X') }}"` renders the secret into the node config. The manifest FILE redacts that config on write (`manifest.py:147`) - which is itself proof the maintainers know rendered configs carry secrets - but `cli/main.py:show` re-dumped the same in-memory config via `typer.echo` WITHOUT redact, so `cat target/manifest.json` was safe while `mbt show X` leaked.
  Audited the sibling data-output commands: `state diff` emits only node ids + changed component *names* (no config values) and `ls` emits names/file-paths, so neither leaks; `show` was the sole vector.
  Fix: `show` now routes both its JSON and YAML output through `redact()`, matching the manifest write and the `fail()` fix.
  Regression test (added to `test_error_redaction.py`, now a two-vector module): injects `env_var('MBT_SHOW_SECRET')` into the demo model's `owner`, runs the real `mbt show ... --output json` via subprocess, and asserts the token never appears and the field is masked; proven non-vacuous by falsification (reverting the fix fails on the leak).
  Verified: both redaction tests pass and fail-without-fix; ruff check + format clean; mypy --strict clean over mbt-core; full fast suite green (exit 0, 302 tests).
  Docs: `secrets.py` docstring now names `mbt show` among the redacted paths; this entry.
- 2026-07-08: DONE - SECURITY: the CLI error path leaked tainted credentials to stderr (P1) - the one serialization path that skipped redaction.
  `secrets.py` promises "every serialization path (events, run results, manifests) passes its text through redact"; `cli/common.py:fail()` - the `guard` decorator's uniform `MbtError` renderer for every command - printed `exc.message`/`resource`/`path`/`hint` to stderr raw. `AdapterError.wrap` embeds the raw underlying exception (`f"adapter '{a}' failed: {exc}"`), so a warehouse connection failure carrying `snowflake://user:TOKEN@acct` prints the token to the terminal and CI logs.
  Reproduced before fixing (bug-repro-first): tainted a token, raised an `AdapterError` embedding it, captured `fail()` output - the raw secret appeared in both the message and the hint, while `redact()` proved the value was tainted and would mask it to `***`.
  Fix: `fail()` now routes all four fields through `redact()`, closing the gap the event/manifest/run_results sinks already covered. One-line, no behavior change for secret-free errors (redact is a no-op when nothing is tainted).
  Regression test: `test_error_redaction.py` taints a token, drives `fail()` with it in every field, and asserts the raw value never appears and each field is masked; proven non-vacuous by falsification (reverting the fix fails the test on the leak). This is also the first test to exercise the redaction *output* at all - `test_compile.py` only covered the primary "manifest stored unrendered" defense.
  Verified: regression test passes and fails-without-fix; ruff check + format clean; mypy --strict clean over mbt-core; full fast suite green (exit 0, 301 tests).
  Docs: `secrets.py` module docstring now lists CLI error output among the redacted serialization paths; this entry.
- 2026-07-08: DONE - crash-safety coverage: `list_runs` ignoring incomplete prediction writes was untested (the last uncovered line in `predictions.py`), found during a correctness review of the ground-truth maturity/ledger path.
  Reviewed `execute/monitor.py`'s "evaluate each matured run exactly once" logic (ADR-21) and found it correct: the maturity arithmetic (`scored_at - negative_delta > anchor` = "not yet `scored_at + |maturity|`"), the ledger-marker skip, the write-marker-only-after-metrics-compute ordering (a non-joinable or single-class run returns None and is NOT marked, so it retries), and `newest_metrics = last-of-scored_at-ascending` - whose ordering assumption is itself pinned by `PredictionStoreCompliance.test_list_runs_ordered_by_scored_at`.
  The one untested branch: `LocalPredictionStore.list_runs` skips a run directory that has an info sidecar but no `_SUCCESS` marker (a crash after the sidecar, before completion), so `mbt monitor` never evaluates a half-written run - a real production crash-safety property that rested on no test.
  Fix: `test_list_runs_ignores_incomplete_writes` writes one complete run plus one info-only (crash-simulated) directory and asserts `list_runs` returns only the complete run.
  Verified: `predictions.py` coverage 98.1% -> 100.0%; `test_local_scoring_data.py` +1 test; ruff check + format clean; full fast suite green (exit 0, 300 tests).
  Docs: this entry.
- 2026-07-07: DONE - doc-drift guard: `docs/cli-reference.md` had no automated tie to the actual CLI surface.
  Audited it by hand first (introspected the Typer app): the reference was fully accurate today - all 17 commands and all 28 non-boilerplate flags documented, zero phantom flags - but nothing stopped a future `--flag` from silently going undocumented (the exact rot the review flagged for other docs).
  Fix: `tests/test_cli_reference_sync.py` - two tests, in the spirit of the NFR-04 import guard and the repo-litter conftest guard, asserting every command (`mbt <name>`) and every non-boilerplate `--flag` appears in `cli-reference.md` (`--help`/completion flags allowlisted).
  Proven non-vacuous: the guard checks 17 commands + 28 flags and correctly reports a bogus command/flag as missing (falsification run), so it is not a no-op.
  Verified: both tests pass; ruff check + format clean; full fast suite green (exit 0, 299 tests).
  Docs: CLAUDE.md Docs section records the guard (adding a command/flag now fails the suite until documented); this entry.
- 2026-07-07: DONE - DS rigor: property-based invariants for the PSI/KS shift statistics.
  `test_monitoring_shift.py` had 9 example-based tests but no property tests, despite hypothesis being a repo dependency - the mathematical invariants of the shift measures were asserted only on hand-picked distributions.
  Fix: a hypothesis property test (60 examples) generates arbitrary baseline and current numeric arrays, builds a real baseline, and asserts PSI >= 0 (it is a divergence: every bin term `(q-p)*ln(q/p)` is non-negative) and KS in [0, 1] (a max abs ECDF gap) for BOTH the feature-shift and prediction-shift paths, through the actual quantile-grid binning - so a bin-indexing or proportion-normalization bug would surface as a negative PSI or an out-of-range KS rather than hiding behind curated fixtures.
  Verified: 60 generated examples pass; `test_monitoring_shift.py` 9 -> 10 tests; ruff check + format clean; full fast suite green (exit 0).
  Docs: this entry.
- 2026-07-07: DONE - test-coverage gap #3: the local artifact store and the `--state` URI reader were only S3-round-trip-tested.
  `mbt/storage.py`'s `LocalArtifactStore` (put/fetch/error guards) and `read_uri_text` (file://, bare path, s3, and the file-not-found hard error) had no direct unit tests - only `test_storage_s3.py`'s moto round-trip existed - so the file:// store contract and the FR-STATE-01 safety guarantee (an unreadable `--state` reference is a hard error, never a silent full retrain) rested entirely on subprocess/e2e coverage.
  Fix: new `test_storage_local.py` (non-network: local put/fetch round-trip with content-hash + size assertions, unsupported-scheme / foreign-ref / missing-file guards, the bucketless-s3-URI guard, and `read_uri_text` over file:// / bare path / missing) plus two moto-backed tests in `test_storage_s3.py` (the S3 fetch foreign-URI guard and `read_uri_text` over s3: round-trip and a missing key -> StateError).
  Verified: `mbt/storage.py` coverage 75.3% -> 100.0%; storage tests 3 -> 11; ruff check + format clean; full fast suite green (296 passed).
  Docs: this entry.
- 2026-07-07: DONE - test-coverage gap #2: the GC champion keep-set (`champion_artifact_uris`) was untested, the highest-consequence branch in `gc.py`.
  `mbt/gc.py`'s `champion_artifact_uris` builds the keep-set that stops `mbt clean` from deleting a registered stage champion's artifact - and deleting one hard-fails champion re-evaluation (ADR-10), i.e. a production-model-loss hazard.
  It had no direct test: the existing GC tests pass a keep-set in directly, and in the CLI e2e the champion's artifact coincides with the latest run's (so `run_results_artifact_uris` already covers it) - the function's distinct contribution, protecting a champion whose artifact is NOT in the latest run, was never exercised, and it runs only in the untraced `mbt clean` subprocess (coverage read it as dead).
  Fix: two unit tests in `test_artifact_gc.py` (lightweight `SimpleNamespace` fakes, ML-free) pinning that every stage champion's artifact is protected, unregistered models contribute nothing, and missing / artifact-less champions are skipped without error.
  Verified: `mbt/gc.py` coverage 78.8% -> 100.0%; `test_artifact_gc.py` 4 -> 6 tests; ruff check + format clean; full fast suite green (288 passed).
  Docs: this entry.
- 2026-07-07: DONE - test-coverage gap: the coordinator-side monitor evaluation had no direct unit tests, unlike its siblings.
  Found via CI-style coverage (`pytest -m "not e2e" --cov`) on the uncommitted scoring/monitoring stack: `mbt/quality/monitors.py` sat at 83.3%, with the prediction-shift branch, the skipped-features path, the baseline-missing "pass loudly" path, and - most importantly - the misconfig branch of `evaluate_ground_truth_gates` (a gate naming a metric the run never computed) exercised only by integration/e2e, never at the fast tier.
  That is the "core compares" half of monitoring (ADR-3), yet its pure comparators had no `test_gates.py`/`test_checks.py`-style sibling.
  Fix: new `packages/mbt-core/tests/test_monitor_evaluation.py` - 8 fast, ML-free unit tests pinning direction-aware feature/prediction shift thresholds, the skipped-features warning (skipped != a passing result), baseline-missing pass-loudly, direction-aware ground-truth gates (greater-is-better and lower-is-better), and the exit-2 guarantee that a gate on an uncomputed metric fails rather than silently passing.
  Verified: `mbt/quality/monitors.py` coverage 83.3% -> 100.0% (0 missing lines); the 8 tests pass; ruff check + format clean; full fast suite green (286 passed).
  Tooling note for future sessions: measure coverage with bare `--cov` (it reads `[tool.coverage.run] source_pkgs`, as CI does); passing `--cov=<dotted.module>` trips numpy 2.x's "cannot load module more than once per process" guard during collection.
  Docs: this entry.
- 2026-07-07: DONE - duckdb Arrow idiom: a latent dependency-floor bug in the new monitoring code plus the `fetch_arrow_table()` deprecation warning the fast suite was emitting.
  Symptom: the fast suite printed `DeprecationWarning: fetch_arrow_table() is deprecated, use to_arrow_table() instead.` from the snowflake stub, and `mbt/execute/monitor.py` (the ADR-20/21 ground-truth join) called `con.execute(...).to_arrow_table()`.
  Root cause, verified against the actual duckdb 1.0.0 manylinux wheel stubs (the `mbt-core` floor is `duckdb>=1.0`): `to_arrow_table` exists on `DuckDBPyRelation` at 1.0 but NOT on `DuckDBPyConnection` (added later), while `DuckDBPyConnection.fetch_arrow_table` exists at 1.0 but is deprecated on the locked 1.5.4.
  So `monitor.py`'s `con.execute(...).to_arrow_table()` would `AttributeError` under `uv sync --resolution lowest-direct` - a real floors-job failure masked only because the monitoring code is still uncommitted (CI's floors job runs the committed tree) and the local lock carries 1.5.4 where `Connection.to_arrow_table` happens to exist.
  Fix, one idiom at both call sites: `con.sql(query).to_arrow_table()` (the Relation API), which is present and non-deprecated across the whole floor-to-lock range (1.0 through 1.5.4); the stale "to_arrow_table is a newer alias" comment (progress entry of 2026-07-07 on the floors work) is corrected in place at both sites.
  Verified: the two affected modules pass under `-W error::DeprecationWarning` (warning is gone, not just hidden); a live probe proves `con.sql(...).to_arrow_table()` is warning-free while the old connection form warns; fast suite green and now emits no warnings-summary block at all; mypy --strict clean over mbt-core; ruff clean; and the churn_demo e2e (real XGBoost build -> promote -> score -> monitor, i.e. the ground-truth join's actual runtime path) passes 4/4.
  Docs: this entry; the two in-code comments now state the real floor/deprecation reason rather than the incorrect alias explanation.
- 2026-07-07: DONE - sections 2.3/2.4 scope cuts (top-5 item 5): batch serving + monitoring ("Monitoring, the sixth ml-ops.org principle, is essentially absent: no OTel, no drift detection, no alerting, no serving-side signal of any kind"; "the tool cannot yet close the loop it draws in its own GitOps diagram").
  One `scoring` YAML = one batch serving pipeline (new resource kind, ADR-20), executed only by `mbt score`: it resolves the model's registered champion by stage alias at RUN time (promotions apply on the next scheduled run; registry state stays outside node identity per ADR-5), materializes an unlabeled input under the same snapshot/window/identity rules as datasets, applies the champion's own hooks + feature selection (parity enforced via a new `mbt.hooks_hash` registration tag; mismatch is a hard error), predicts through the existing `TrainingAdapter.predict` contract, and writes predictions through a new prediction-store seam (DataAdapter contract 1.0 -> 1.1, local parquet implementation, capability-probed before any job runs).
  Monitoring lives in the same config, closing the review's distribution-monitoring gap with distinct vocabulary ("shift", since this repo reserves "drift" for snapshot drift): label-free input checks, per-feature and score-distribution PSI/KS monitors against a training-time baseline that every training job now exports and registers (quantile grids, ADR-21), and a `ground_truth` block that `mbt monitor` evaluates on its own schedule - matured prediction runs join arrived labels in-process, realized metrics come from the shared metric engine, threshold gates apply, each run is evaluated exactly once via ledger markers, and results log to tracking so MLflow accumulates the production-performance series.
  Breaches are quality failures: new `monitor_failed` status maps to exit code 2 exactly like gates/tests, and the scaffold ships `scheduled_score.yml` (daily) + `scheduled_monitor.yml` (weekly) with the same pinned installs and `MBT_ALERT_WEBHOOK` alerting as the retrain workflow.
  Tested: ~60 new tests across the stack - spec validation and PSI/KS determinism units (incl. duplicate-bin-edge and unseen-category edges), parsing/link rules, compile identity (arriving labels never re-flag the scorer; model edits flip it transitively; anchor drift flips nothing), baseline capture + registration tags, local prediction store incl. a new `PredictionStoreCompliance` suite and a label-free-predict compliance case for training adapters, `mbt score` integration over the fake stack (promotion pickup, idempotent same-key re-runs, shift breach exit 2, hooks mismatch, missing champion/baseline, input-check failure skips the job), `mbt monitor` integration (maturity arithmetic, ledger idempotency, partial coverage, no-labels retry, gate breach), and the churn_demo E2E extended with real XGBoost score -> monitor -> ledger-no-op steps.
  Verified: fast suite 274/274; full e2e tier green (exit 0, incl. the JVM adapters); manifest schema v2 (still reads v1) with the golden regenerated; mypy --strict, ruff check + format, pre-commit, mkdocs --strict, yamllint, and PyYAML workflow parses all green.
  Docs: ADR-20 (scoring resource, run-time champion, shift vocabulary, monitor_failed) and ADR-21 (prediction store + run_key idempotency, baseline artifact, ground-truth ledger); spec-reference scoring section; cli-reference `score`/`monitor` rows + exit-code note; concepts resource table + "Batch scoring and monitoring" section; gitops loop steps 5-6; roadmap `mbt score` marked shipped; v0.1-status updated (scoring moved out of the v1 list, claims kept exact); five new troubleshooting entries with reproduced verbatim messages; scaffold README serving section.
  Still out of scope, deliberately: online (request/response) serving (PRD non-goal), warehouse prediction sinks (Snowflake/Spark implement the store seam later), OTel export.
- 2026-07-07: DONE - completeness audit + the two stragglers it found (sections 2.2 and 2.4; final loop iteration).
  `mbt deps` (2.2, "loose PEP 440 specifiers with no lock or hash verification"): now installs from the project's pinned `requirements.txt` when present (pip honors hashes in the file; scaffolded projects ship exact pins since action item 1), falls back to the loose specifiers with a loud unpinned-install warning otherwise, and in BOTH paths verifies the installed environment satisfies every packages.yml pin afterward - a drifted requirements.txt now fails at install time with the offending package named, not at first import.
  Quickstart (2.4): `mbt evaluate ... --gates` was labeled "drift check", the exact data-versioning-vs-distribution-monitoring conflation the review called out; relabeled "champion decay check on fresh data".
  Tested: five deps unit tests (pinned file preferred with `-r`, loose fallback command + warning, post-install drift detection raises with the mismatch named, missing-package verification, dry-run never invokes pip); fast suite 209/209; full e2e tier 32/32; mypy --strict, ruff, pre-commit, mkdocs --strict green.
  Docs: cli-reference deps row rewritten; the closure summary above records the full audit map.
- 2026-07-07: DONE - section 3.5 P3 (item 12), Optuna pruning ("no pruner and no trial.report() integration, so every trial trains to completion").
  `tuning.pruner: median` is a validated spec field: a seeded MedianPruner stops trials whose per-round validation value falls below the median of prior trials at the same step; pruner knobs (n_startup_trials, n_warmup_steps) live in the ENGINE's profile config, not the spec - ops tuning is not model identity, so config hashes stay clean.
  The report channel is a framework-neutral optional adapter method, consistent with the log_trial/feature_importance pattern: `train_with_report(spec, data, ctx, report)` calls `report(step, value)` per iteration with a HIGHER-IS-BETTER validation value; the engine flips the sign for minimize objectives (the pruner compares in study direction - the subtle bug this contract prevents), and pruning is a raise out of the training loop, exactly optuna's own integration pattern. Adapters without the method fall back to full trials with a one-time warning; ADR-8 intact (intermediate values come from the validation split, never test).
  Implemented in xgboost (TrainingCallback, AUC series merged into any user eval_metric) and lightgbm (callback on valid_sets); the fake adapter reports a deterministic 10-step ramp so the whole path tests without frameworks; pruned counts land in the `mbt.tuning.n_pruned` tracking tag.
  Tested at every layer: engine units (pruning fires and saves training work, legacy objective signature untouched, deterministic across runs, minimize-direction sign flip cuts weak-not-strong trials, no-completed-trials guard, spec literal validation), adapter units (per-round AUC streams; a raising report aborts xgb/lgb training promptly - 3 of 50 rounds trained), job plumbing (pruner spec survives to tracking tags via the fake engine), and the optuna+xgboost e2e now runs with the pruner active through the real report path.
  Verified: fast suite 204/204 (10 new tests); full e2e tier 32/32; golden manifests untouched (no demo model declares tuning; the new TuningSpec field only enters hashes of models that do); mypy --strict, ruff, pre-commit, mkdocs --strict green.
  Docs: spec-reference tuning block documents the pruner and its adapter requirement; adapter-authoring's optional-method table specifies the train_with_report contract (higher-is-better values, let raises propagate).
  With this, every actionable item in the review is closed; what remains is deferred by design (post-hoc calibration step, cross-validation, release pipeline) or blocked on evidence (JVM flaky reruns).
- 2026-07-07: DONE - sections 3.3/3.8 (item 12), operating-point selection and calibration metrics in the demo ("accuracy uses a fixed 0.5 threshold with no operating-point selection"; "brier/ece are never exercised by the demo, despite churn probabilities being what interventions consume").
  New parameterized builtins in the shared metric layer (identical across adapters by construction): `threshold_at_precision_<p>` reports the smallest score cutoff meeting the precision target (maximal coverage at the required precision), `threshold_at_recall_<r>` the largest cutoff meeting the recall target (best precision at the required coverage) - the deployable decision rule an outreach campaign actually consumes.
  Defined edges: an unattainable precision target reports the 1.0 sentinel ("predict nothing" is the only rule honoring the target), degenerate labels report sentinels instead of raising, and the sklearn PR-curve's dominated-point truncation means the reported cutoff is the best of equal-coverage options.
  Tested: hand-computed thresholds against a 6-row curve (including the non-monotone-precision case), sugar parsing and rejection, sentinels, and the semantic guarantee itself - a property test applies the returned cutoff back to the scores and asserts the requested precision/recall is met.
  Demo: churn_classifier now measures `brier`, `ece`, and `threshold_at_precision_0.35` (the cutoff for a 35%-precision campaign, chosen above the ~0.2 base rate); the churn e2e asserts calibration values are sane and the operating point is usable (0 < t < 1, i.e. the target is genuinely attainable on the demo data); golden manifests regenerated (metrics list enters config_hash).
  Verified: fast suite 194/194 (5 new tests); full e2e tier 32/32 incl. byte-repro; mypy --strict, ruff, pre-commit, mkdocs --strict green.
  Docs: spec-reference builtin list documents both operating-point metrics and the sentinel; demo spec comment explains the campaign framing.
  Deliberately deferred from 3.8: a post-hoc calibration STEP (Platt/isotonic recalibration artifacts) - that is a v1-scale feature touching artifacts, champion parity, and export formats, not a metric-layer addition.
- 2026-07-07: DONE - P2 action item 6, docs-publish and CodeQL clauses (sections 2.1; the last automatable pieces of the item).
  Docs publication: the strict-built mkdocs site now deploys to GitHub Pages on pushes to main via a `docs-publish` job (actions/upload-pages-artifact + deploy-pages, minimal job-scoped `pages: write`/`id-token: write` permissions, `github-pages` environment); PRs keep the strict build as a gate without deploying. One-time repo setup documented (Settings -> Pages -> Source: GitHub Actions).
  CodeQL: a new workflow scans package sources (python, default queries) on push/PR plus a weekly off-peak schedule, complementing the existing pip-audit dependency scan; scaffold templates are excluded (user-project code, not shipped source). Requires code scanning availability on the hosting repo - noted in the workflow header.
  Verification is necessarily CI-shaped for deploy machinery; locally: every workflow PyYAML-parses (the earlier broken-scalar lesson) and passes yamllint, `mkdocs build --strict` proves the published artifact, fast suite 189/189 and hooks green. The deploy/scan jobs themselves can only prove out on the hosted repo - flagged for the first push.
  Docs: README development section (site URL mechanics + security-scanning summary), v0.1-status NFR-09.
  Item 6 is now closed except the release/PyPI pipeline, which stays deferred as organizational (name check/release decision) per `docs/v0.1-status.md` NFR-10 - deliberately not scaffolded until that decision exists.
- 2026-07-07: DONE - section 3.9 P3, richer demo data as a teaching asset (plus the 3.6 P3 finding that `tests/test_no_leakage.py` "writes a check the code does not cash").
  The generator now produces 6000 rows (was 2400) with two genuine noise features the models must cope with - `weekly_logins` (numeric) and `signup_channel` (categorical, exercising native categoricals in the churn models too) - and a deliberately leaky `account_status` string column: generated post-outcome, it encodes the label exactly, showcasing the categorical leakage scan shipped earlier this session.
  The leak is quarantined visibly at every layer: the dataset declares the reviewed `label_leakage_scan` exclusion with a comment explaining the teaching intent, all three models exclude it from features (post-outcome columns are leakage for ANY 90-day prediction, upsell included), and `test_no_leakage.py` finally cashes its name - `test_planted_leak_is_quarantined` pins both that the leak stays leaky (the asset cannot silently rot) and that the exclusion stays declared.
  Negative proof, live: stripping the exclusion in a scratch copy blocks the build at exit 2 with TWO independent guards firing - the always-on scan (`account_status (V=1.000)`) and the failing quarantine data test. Defense in depth, demonstrated.
  Verified: golden manifests deliberately regenerated (committed data bytes + spec changes flip deep-snapshot hashes); fast suite 189/189; full e2e tier 32/32 including the churn byte-reproducibility rerun and all three models clearing their gates on the richer data (the pr_auc floor holds with noise features present); pre-commit, mypy --strict, ruff green.
  Docs: the teaching surface IS the demo (generator comments, dataset spec comment, the data test docstring); no external doc claims cited row counts.
- 2026-07-07: DONE - section 3.6, final leftover: `no_future_columns` caught only absolute-future timestamps, not train/test overlap.
  Gap proven live before fixing (bug-repro-first): a train row carrying a `last_event_at` inside the TEST window - textbook temporal leakage - passed the old check, because it examined only the train split against the GLOBAL max window end (so anything below the test window's end looked fine).
  The check now validates every split's timestamp columns against that split's OWN resolved window end: train values reaching into the test period are flagged as `train.<column> reaches <ts> beyond the train window end <ts> (temporal leakage)`; absolute-future detection is preserved (a value beyond the final window still exceeds its split's end); the no-windows case (random splits) keeps its explicit pass message.
  Tested: unit tests for the previously-missed overlap case, the absolute-future case, and clean window-realistic splits (the old shared-table fixture would trip the per-split check by construction, so the fixture now builds properly filtered splits - itself evidence the old check was too weak to notice unrealistic fixtures); an execution test declares the check on the demo project, plants snapshot+30d event timestamps so late-train rows land in the test window, and proves the build blocks (exit 2, dataset test_failed, model skipped, message names `train.last_event_at`).
  No false positives: fast suite 189/189 and full e2e tier 32/32 - both churn_demo and the scaffold declare `no_future_columns` on real window-filtered data and pass clean; mypy --strict, ruff, pre-commit, mkdocs --strict green.
  Docs: spec-reference check comment rewritten (per-split semantics, overlap coverage).
  Section 3.6 is now fully closed: always-on scan (P2), categorical association, and train/test-overlap detection are all shipped; the remaining teaching-asset idea (a deliberately leaky demo column) is covered by the runbook's live reproduction instead.
- 2026-07-07: DONE - section 3.6 leftover, categorical association in the leakage scan (the scan was numeric-only; native-categorical support from item 9 had widened the blind spot - a status string literally encoding the label sailed straight through the always-on guard).
  String/categorical columns are now screened with Cramér's V from the contingency table vs the label - pure duckdb-SQL + arithmetic, no scipy (mbt-core carries none). For two binary variables V equals |phi|, the Pearson correlation of the indicators, so the existing two-tier bar transfers unchanged: >= 0.95 fails (reported as `V=...`), 0.85-0.95 warns, `exclude`/`enabled: false` behave identically across both column kinds.
  Guards against spurious saturation: single-level columns skip (no signal), and quasi-identifiers (distinct levels > half the non-null rows) skip - a unique-per-row email/ID string would otherwise score V ~= 1 without indicating leakage.
  Tested: unit tests for a perfect label-encoding categorical (fails at V=1.000), an independent categorical, a constructed warn-band case (200 rows, 10 flips -> V=0.9 exactly), exclude, and both skip guards; an execution test proves an undeclared scan blocks a real build with a leaky string column (exit 2, model skipped, message names `subscription_status (V=1.000)`).
  No false positives: full fast suite 187/187 and the full e2e tier 32/32 with the always-on scan screening every string column in every fixture - the demo's `plan_type` and the adapters' categorical compliance fixtures all pass clean; mypy --strict, ruff, format, pre-commit, mkdocs --strict green.
  Docs: spec-reference checks section documents the dual statistic and the skip rules; the troubleshooting runbook's leakage entry now covers the `V=...` form.
  Still open from 3.6: `no_future_columns` train/test-overlap detection (it only catches absolute-future timestamps).
- 2026-07-07: DONE - the path-resolution product wart logged in the 2.6 entry (config-relative paths resolved against the process cwd, the root cause behind the reviewer's own `target/fake_tracking` litter evidence).
  Diagnosis: job subprocesses always ran with `cwd=project_dir` (local and spark compute both), so the JOB side was already project-relative; only the COORDINATOR kept the user's cwd - `mbt promote`/`evaluate`/`clean`/`docs` from a foreign cwd opened relative sqlite URIs and `file://` stores against the wrong directory (promote would silently create an empty registry db in the invocation cwd and report "model not found").
  Fix, one boundary: `make_ctx` chdirs the coordinator to the resolved `--project-dir` (matching what jobs always did), with a clear ConfigError for a nonexistent dir and `chdir=False` for `mbt init` (its project-dir is the parent to scaffold into).
  Convention made explicit: paths TYPED on the command line are shell-relative to the invocation cwd - `--state`, `--manifest`, `--from-file` are absolutized via a new `ctx.resolve_cli_path` BEFORE the chdir takes effect (URIs pass through); paths in CONFIG are project-relative.
  Properly tested: a new e2e runs the real CLI from an unrelated cwd with `--project-dir` - build confines target/, the artifact store, and the relative-sqlite mlflow.db to the project with the invocation cwd left empty; promote finds the project's registry (the previously-broken case); `state diff --state ./baseline.json` proves shell-relative typed paths. Falsification run: with the chdir commented out the test fails exactly at the promote-leak assertion, then the fix was restored - the test is not vacuous.
  Verified: fast suite 184/184; FULL e2e tier 32/32 (churn, scaffold state loop, JVM Spark/H2O, the new confinement test) with the repo-litter guard silent; mypy --strict, ruff (incl. a drive-by E501 fix in the new root conftest), format, pre-commit --all-files, mkdocs --strict all green.
  Docs: cli-reference gained a Path-semantics section; CLAUDE.md records the rule (new CLI path options must route through `resolve_cli_path`); FEEDBACK 2.6 wart closed.
- 2026-07-07: DONE - section 2.7 P3 / action item 12, perf-budget tolerances (bare wall-clock asserts flake under noisy CI neighbors).
  The three NFR-03 tests now measure achievable speed instead of a single sample: one untimed warmup (cold imports, first materialization), then best-of-3 for parse and compile, and a per-node minimum across two measured `run_command` invocations for the overhead budget - min filters scheduler noise because a budget bounds what the code CAN do, not what a contended runner happened to do.
  The budget numbers themselves (parse < 2 s, compile < 10 s, node overhead < 2 s at 50 resources) are untouched: the NFR-03 contract is unchanged, only the measurement is noise-robust.
  Verified: 10 consecutive perf runs green with exit codes checked individually (not piped); total perf runtime stays ~2 s because the measured operations are sub-second; fast suite 184/184; pre-commit --all-files green.
  Docs: module docstring states the methodology and why; v0.1-status NFR-03 row updated.
  Housekeeping this iteration (user request): the repo now has a `CLAUDE.md` capturing verify commands, load-bearing decisions (dual-pyspark lock fork, floors job, typer-vendored-click handling, deep-snapshot CI), test conventions (tmp-only writes + root conftest guard, unique test basenames, golden regeneration), and docs rules, so future sessions inherit the loop's accumulated knowledge.
  Still open from section 2.7 (P3): flaky-test rerun tooling for the JVM e2e tier - deliberately not bundled here; adding rerun machinery without observed flakes hides real failures, so it should wait for evidence from CI.
- 2026-07-07: DONE - section 2.6 P3 / action item 12, tmp-dir test artifacts (tests littered the repo root; flagged "still open" since the first progress entry).
  Root cause was subtler than the review's one-liner: relative paths in generated test profiles (`root: ./target/fake_tracking`, `artifact_store: file://...`, and the fake tracking adapter's own `./target/fake_tracking` DEFAULT) resolve against the pytest process cwd - the repo root - not the tmp project dir; mlflow's default artifact root (`./mlruns`) does the same.
  Fixed: mbt-core's `demo_project` conftest profiles absolutize every root into tmp_path; the perf-budget project declares explicit tmp roots (bare `adapter: fake` falls back to the cwd-relative default); the mlflow `uri` fixture chdirs to tmp so `./mlruns` lands there.
  Enforcement, not just cleanup: a repo-root `conftest.py` session guard snapshots the repo root (and `target/`'s children) before the run and fails the session naming any new entries - it proved itself immediately by catching the perf test's default-root fallback that the grep sweep missed. Tooling caches (`.pytest_cache`, `.coverage`, `.hypothesis`, ...) are allowlisted so fresh clones and CI's `--cov` run don't false-positive.
  Verified: from a clean tree, fast suite 184/184 and the FULL e2e tier 31/31 (churn + scaffold-state-loop + Spark/H2O under JVM) leave zero new entries at the repo root (guard green, `git status` unchanged, no `target/` or `mlruns/` exists afterward); pre-commit --all-files green. Local leftover dirt (456K of fake-adapter JSON, empty `mlruns/`) deleted.
  Docs: README development section documents the invariant and the guard; the guard's failure message teaches the fix (tmp_path + absolute roots in generated profiles).
  Observed for a future item (product wart, out of scope for a test-hygiene fix): adapter-config relative roots and `file://` store paths resolve against the process cwd while the local data adapter's `root` resolves against the project dir - an inconsistency that could bite real users running `mbt --project-dir X` from elsewhere; fixing it properly means threading project context through adapter construction.
- 2026-07-07: DONE - section 2.2 P3 / action item 12, deep-snapshots-in-CI clause (ADR-11's "mtimes lie" admission was load-bearing, not cosmetic).
  Every CI run is a fresh checkout with rewritten mtimes, so the scaffold's mtime-token baselines could never match the next run's compile: every dataset flagged `modified` on every run, every model retrained via `upstream` - the G3 economy loop silently degraded to a full retrain for local-file data.
  The scaffolded workflows now pass `--deep-snapshot` on every compiling step (pr_check compile/diff/build, prod_build both build branches), so published baselines and their consumers share one content-hash token scheme; the scaffold README's day-to-day snippet gained the flag too (a deep baseline diffed with the default scheme flags everything - the schemes differ by construction).
  Enabled by the previous iteration's fix adding `--deep-snapshot` to `state diff`; without it this configuration was impossible.
  Properly tested: the state-branch loop e2e now runs the whole loop deep, and gained a fresh-checkout simulation - `shutil.copytree(..., copy_function=shutil.copy)` (fresh mtimes, same bytes, like actions/checkout) - proving the copied checkout diffs EMPTY against the published baseline while a real spec edit still flags exactly `churn_classifier`; the mtime-flags-touch contrast is pinned by the companion `test_state_diff_deep_snapshot_ignores_mtime_churn`.
  Verified: loop e2e green (6 s); fast suite 184/184; workflows PyYAML-parse + yamllint clean; pre-commit --all-files, mkdocs --strict green.
  Docs: ADR-11 records the CI default and the one-scheme-per-pipeline rule; gitops.md loop step 1; troubleshooting.md mtime entry notes the scaffold already does this; scaffold README day-to-day snippet.
  Cost note: content-hashing data files is the tradeoff; fine at demo scale, and warehouse adapters (Snowflake) are unaffected - their snapshots are server-side and never mtime-based.
- 2026-07-07: DONE - section 2.8 runbook finding (P2 in-section; "runbooks" clause of action item 12): operators had design docs but no playbook.
  `docs/troubleshooting.md` now maps every deliberate failure mode to symptom -> why it is intentional (with the ADR) -> fix: mtime-snapshot false positives, gate-edit retrains, `state:modified` without `--state`, env-digest mismatch and its `--allow-env-mismatch` downgrade, gate failure exit-2 semantics, leakage-scan blocks, missing-champion bootstrap vs unloadable-champion hard error, degenerate slice gates, snapshot-mismatch pins, and mistyped flags; an exit-code triage table (1 = page someone, 2 = review the model) opens the page.
  Every symptom is verbatim from a live reproduction against a scratch churn_demo this session - none are imagined; the runbook clarifies where a generic hint misleads (an unloadable champion's "re-run the build" cannot restore the champion's file; the remedies are store restore or deliberate re-promotion).
  Reproducing the entries surfaced two real bugs, both fixed:
  (1) any mistyped CLI flag dumped a 40-line traceback because typer >= 0.20 vendors click, so `main()`'s `except click.UsageError` never matched the vendored exception types (`typer._click.exceptions`) - untested and broken since before the review; main() now catches both hierarchies and a regression test asserts clean output, exit 1, no traceback;
  (2) `state diff` lacked `--deep-snapshot`, so the documented mtime-lies remedy (deep snapshots in CI, ADR-11) could not actually be used for diffing - deep-published baselines always flagged every dataset; the flag now exists and a test proves deep-vs-deep diffs survive a `touch` while the default scheme flags it.
  Verified: 9 live CLI reproductions (build/promote/tamper/delete flows on the scratch demo); fast suite 184/184 (two new tests); churn e2e 4/4; mypy --strict, ruff, format, pre-commit --all-files, mkdocs --strict (new page in nav) all green.
  Docs: `docs/troubleshooting.md` (new, in mkdocs nav), gitops.md cross-link, cli-reference `state diff` row, v0.1-status NFR-09.
- 2026-07-07: DONE - P2 action item 6, durable prod-state clause (section 2.1, no durable prod state baseline out of the box).
  The scaffold's durable baseline moved from bot-commits-to-main (added in an earlier pass, but broken under branch protection and polluting main history with `[skip ci]` hacks) to a dedicated append-only `mbt-state` branch driven by two new scaffold scripts: `publish_state.sh` (git plumbing only - hash-object/commit-tree against a temp index, never touches the working tree, current branch, or main's protection rules; one refetch-and-retry on push races) and `fetch_state.sh` (restores the baseline for `--state`; distinct exit code 3 when no baseline exists so workflows bootstrap with a full build).
  prod_build now fetches the baseline before building (in the reviewed version its `[ -f ... ]` check was dead code on a fresh runner; the interim main-commit approach fixed that but only where main is unprotected), builds `state:modified+` against it, and publishes the new manifest; pr_check fetches the same way; the fetched path is gitignored in scaffolded projects.
  Properly tested, not string-asserted: a new e2e test runs the loop for real against a local bare origin - bootstrap fetch exits 3, publish without a manifest fails loudly, a real `mbt build --target prod` (xgboost + mlflow sqlite on generated data) then publish, a clean-worktree assertion (the plumbing contract), fetch + `state diff --output json` empty on an unchanged project, a spec edit flags exactly `churn_classifier`, and a second publish appends (audit trail depth 2) with fetched bytes identical to the published manifest. Runs in 7 s under the e2e marker.
  Scaffold completeness and guardrails tests updated (ship the scripts; prod publishes and both workflows fetch); all four scaffold workflows PyYAML-parse and yamllint clean.
  Verified: the new e2e test green; fast suite 182/182 (31 deselected now includes the new test); ruff, format, pre-commit --all-files, mkdocs --strict green.
  Docs: `docs/gitops.md` loop step 2 and the operations section rewritten around the state branch; scaffold README documents the scripts and the day-to-day fetch command.
  Still open from action item 6 (P2): a release/PyPI pipeline (deferred as organizational in `docs/v0.1-status.md`), docs site publication (built strict in CI, not yet deployed), and CodeQL/bandit on top of the existing pip-audit job.
- 2026-07-07: DONE - P2 action item 11 (section 1.1, PySpark ceiling isolation and Spark 4.x validation).
  `mbt-spark` now declares `pyspark>=3.5,<4.2` - the `<3.6` pin lives only in `mbt-h2o[sparkling]`, where `h2o-pysparkling-3.5` genuinely requires it.
  The non-obvious part was the workspace lock: uv unifies one pyspark across every member's extras, so the optional sparkling extra silently held the whole dev environment at 3.5 even though nothing installs it; a `[tool.uv] conflicts` fork between the dev group and the sparkling extra lets the lock carry 4.1.2 for dev/CI and 3.5.8 for sparkling side by side (published metadata is unaffected - end users installing both packages still intersect to Spark 3.5, which is the documented tradeoff).
  Zero adapter-code changes were needed for Spark 4: the full JVM suite (data push-down, SparkML GBT compliance incl. export -> load round-trip, spark-submit compute seam, H2O AutoML) passes 23/23 on pyspark 4.1.2 under Java 17, after first re-proving the same 23 on 3.5.8 as baseline - both ends of the declared range are validated on this machine, and CI now exercises 4.1.2 in the locked resolution.
  Real bug found by the E2E-first discipline: `get_session` never pinned the executor interpreter, so any local run without the venv activated died with PYTHON_VERSION_MISMATCH (driver = venv 3.11, workers = whatever `python3` is on PATH; CI only passed by PATH luck). Local masters now pin `spark.pyspark.python` to `sys.executable` unless the caller or `PYSPARK_PYTHON` already chose one; remote masters keep the image contract. Proven by re-running the exact failing bare invocation.
  Verified: JVM e2e 23/23 on both Spark majors; fast suite 182/182 on the new lock; the floors job's lowest-direct resolution still solves with the conflict fork (pyspark 3.5.0 at the floor); ruff, format, yamllint, mypy --strict (now also clean over mbt-spark src), pre-commit --all-files, mkdocs --strict all green; `uv lock --check` clean.
  Docs: ADR-17 gained decisions 5 (version policy + sparkling pin-back tradeoff, as the review asked) and 6 (executor interpreter pinning); ci.yml e2e comment rewritten; v0.1-status mbt-spark line now names the validated majors.
  Still open (P3, section 1.1 adjacent): the e2e JVM tier could run a newer interpreter now that Spark 4 supports 3.13; left on 3.11 deliberately since Python coverage belongs to the test matrix.
- 2026-07-07: DONE - P2 action item 4 (sections 1.1 and 1.4, untested Python versions and dependency floors).
  The CI `test` job now matrixes the fast suite over CPython 3.11/3.12/3.13/3.14 (fail-fast off) - everything `requires-python = ">=3.11"` advertises - and a new `floors` job installs every direct dependency at its declared lower bound (`uv sync --resolution lowest-direct`) and runs the fast suite against it, so an unsupportable floor now fails CI instead of a user install.
  The e2e JVM tier stays on 3.11 with the PySpark-3.5 reason documented in the workflow (revisited by action item 11).
  The floors job immediately earned its keep: `typer>=0.12` fails on PEP 604 unions and on current click (make_metavar signature; first working release is 0.16), `mlflow>=2.9` imports the long-gone `pkg_resources` in setuptools-free venvs (first clean release is 2.16), and `lightgbm>=4.0` has no macOS arm64 wheel before 4.4; all three floors raised with the reason commented in the owning pyproject.
  `types-PyYAML` gained its missing floor, and lint-stack dev floors now track what the repo actually enforces (mypy>=2.0, ruff>=0.15, yamllint>=1.38).
  Two floor-compat code fixes: `ProjectConfig` sets `protected_namespaces=()` (pydantic < 2.10 warned on `model_defaults`/`model_paths`, polluting CLI stderr at the floor), and the snowflake DuckDB stub uses `fetch_arrow_table` (exists at the duckdb>=1.0 floor; `to_arrow_table` is a newer alias).
  Drive-by fixes per house rule: `ci.yml`'s yamllint step was an outright YAML syntax error (unquoted plain scalar containing `: `), invisible because nothing lints `.github/`; quoted it and folded the over-long mypy line. Three files carried ruff violations from earlier iterations (E501 in `encoding.py`, C408 in `test_paired_bootstrap.py`, format drift in `gc.py`) - cleaned.
  Verified: fast suite green (182 passed) on 3.11/3.12/3.13/3.14 in fresh per-version venvs synced from the universal lock, and green against the lowest-direct environment (typer 0.16.0, click 8.4.2, rich 13.7.0, mlflow 2.16.0, lightgbm 4.4.0, xgboost 2.0.0, optuna 3.6.0, pydantic 2.7.0, pyarrow 15.0.0, numpy 1.26.0, duckdb 1.0.0, snowflake-connector 3.7.0, pytest 8.0.0); uv.lock re-locked (specifier-only diff); ruff, ruff format, mypy --strict, yamllint (now also covering `.github`), pre-commit --all-files, and mkdocs build --strict all green; churn e2e re-run.
  Docs: README development section (matrix + floors), quickstart Python line, v0.1-status NFR-05.
  Still open from section 1.1 (P2): the PySpark `<3.6` ceiling isolation is action item 11; keeping the matrix current as new CPythons release is Renovate's job.
- 2026-07-07: DONE - P2 action item 7 (section 2.6, file://-only artifact store and no retention).
  `S3ArtifactStore` ships behind the existing `mbt-core[s3]` extra: uploads under the per-run prefix, downloads to a local cache on `fetch` so adapters keep working with plain paths, actionable errors for a missing extra or a lifecycle-removed object; `artifact_store_for` dispatches by scheme.
  Retention: `mbt clean --artifacts-older-than <duration> [--dry-run]` prunes file:// store run prefixes older than the cutoff, always preserving the latest run's artifacts and every stage champion of every registered model (deleting a champion would trip ADR-10's hard error by design); object stores are pointed at bucket lifecycle rules instead of a client-side GC.
  Verified: moto-backed S3 round-trip, scheme dispatch, and missing-object tests; GC unit tests for age pruning, keep-set, dry-run, s3 refusal, and missing store; a subprocess CLI test runs a real build, ages an orphan prefix, and proves `mbt clean --artifacts-older-than 30d` removes only the orphan; fast suite, hooks, strict mypy green. moto added as a dev dependency.
  Docs: `docs/gitops.md` artifact-storage and retention section, `docs/cli-reference.md` clean row, `docs/spec-reference.md` profiles sample, `docs/v0.1-status.md` limitation updated (s3 stores no longer deferred).
  Still open from section 2.6 (P3): tests writing into repo-root ./target.
- 2026-07-07: DONE - P2 action item 9, feature-importance clause (section 3.7, no explainability output) - action item 9 is now fully closed.
  Training adapters can expose an optional `feature_importance(model)` (hasattr-based, like trackers' `log_trial`); XGBoost and LightGBM implement it as gain importance normalized to fractions over the exact feature list (unused features report 0.0), and the fake adapter returns a deterministic stand-in so the plumbing is testable.
  Importance flows challenger-side through `JobResult` into `run_results.json` (`NodeResult.feature_importance`) for both train and evaluate modes, and `mbt docs generate` model cards render a top-15 importance table between Features and Hyperparameters.
  Verified: adapter tests assert keys match features, fractions sum to 1, and the categorical column dominates in the categorical fixtures; an execution test asserts importance reaches run_results; a new `test_docsgen.py` (the docs generator's first test) asserts the card renders the table; churn e2e, fast suite, hooks, and strict mypy green (mypy caught xgboost's multi-class list-typed `get_score` union, handled explicitly).
  Docs: adapter-authoring contract table documents the optional method; cli-reference updated.
- 2026-07-07: DONE - P2 action item 9, native-categorical clause (section 3.7, no native categorical support).
  String feature columns now train as native categoricals in both tree adapters: XGBoost via integer codes + `feature_types`/`enable_categorical` on the DMatrix, LightGBM via codes + `categorical_feature`.
  Shared derivation and encoding live in the new `mbt_adapter_base.encoding` module so both adapters map values to codes identically (champion/challenger parity): levels are the sorted unique train-split values (deterministic), persisted with the artifact (XGBoost booster attribute `mbt_categories`, LightGBM JSON envelope), and unseen or missing levels become NaN (the frameworks' missing branch).
  Other non-numeric types (timestamps, nested) keep an actionable error; ONNX export errors clearly when categoricals are present.
  The demo now showcases it: `upsell_classifier` stopped throwing away `plan_type` (genuinely predictive of upsell in the generator); `churn_classifier` keeps `plan_type` as its slice column (a column cannot be both feature and slice under the derivation contract).
  Verified: per-adapter tests prove the categorical carries the signal (roc_auc > 0.85 with signal only in the category), categories survive export -> load with identical scores (the champion path), and unseen levels predict without crashing; compliance suites green; the churn e2e passes end to end including the byte-reproducibility rerun (categorical determinism in the real flow); golden manifests regenerated; fast suite, hooks, and strict mypy green.
  Docs: adapter docstrings, `docs/adapter-authoring.md` feature contract rewritten, `docs/spec-reference.md` features note.
- 2026-07-06: DONE - P2 action item 9, lift-metrics and demo-gate clauses (sections 3.3 and 3.9).
  `lift` and `gain` are first-class binary builtins with `lift_at_<fraction>` / `gain_at_<fraction>` sugar (`lift_at_0.1` = decile lift; `gain_at_0.25` = share of positives captured in the top quartile), computed identically across adapters via the shared metric layer, with stable deterministic tie-breaking and zero-positive guards.
  The churn demo migrated from its hand-rolled `lift_at_decile` hook to builtin `lift_at_0.1`; the hooks escape hatch now demonstrates a genuinely custom metric (`campaign_capture_100`, churners captured within a fixed 100-contact budget) so FR-RES-07 keeps real e2e coverage.
  The vacuous demo gate is fixed: `pr_auc_floor` raised 0.05 -> 0.3, meaningfully above the ~0.20 random-baseline PR-AUC, with a comment explaining why a floor below the base rate never gates.
  Verified: hand-computed unit tests for lift/gain plus sugar parsing, tie determinism, and degenerate-label guards; the churn e2e proves all three demo models clear the new floor and both metric paths compute (`lift_at_0.1 > 1.0`, `campaign_capture_100 > 0`); golden manifests regenerated (demo spec and hooks changed); full fast suite and hooks green.
  Docs: `docs/spec-reference.md` builtin metric list and the metrics.yml/hooks examples updated to match the demo.
- 2026-07-06: DONE - P2 action item 9, split-protocol clause (section 3.2, grouped-split control and random-on-temporal warning).
  Two parse-time guardrail warnings for `strategy: random`: combining it with a declared `time_column` warns about temporal leakage (with the hint to use the temporal strategy), and omitting `sample_key` warns that rows split independently so repeated entities can straddle train and test.
  Warnings, not errors: random splits over truly exchangeable rows are legitimate; both warnings disappear when addressed (temporal strategy, or `sample_key` set).
  The grouped-split control is `sample_key` itself (hash-based ranking keeps an entity's rows on one side); it is now documented as such rather than left implicit.
  Verified: a parser test asserts both warnings fire on the risky configuration and that the addressed configuration parses warning-free; full fast suite and hooks green.
  Docs: `docs/spec-reference.md` random-split section documents both guardrails and the sample_key grouping semantics.
  Still open from section 3.2: cross-validation (no CV anywhere) - a larger feature deferred with the tuning-protocol findings in 3.5.
- 2026-07-06: DONE - P2 action item 9, leakage-scan clause (section 3.6, leakage prevention is user discipline).
  `label_leakage_scan` now runs on every dataset build whether declared or not; declaring it is only for tuning (`max_abs_correlation`, new `warn_abs_correlation`, new `exclude`) or opting out (`enabled: false`, recorded as "check disabled" rather than silently skipped).
  The bar is two-tier: |corr| >= 0.95 fails the build, the 0.85-0.95 warn band emits a warning and is recorded in the check message without failing.
  The churn demo (and every mbt project) is now covered automatically, answering the "demo does not even enable it" finding.
  Verified: unit tests for default-run, warn band, exclude, and opt-out; an execution test proves an undeclared scan blocks a build with a leaky column (exit 2, downstream model skipped); full fast suite and full e2e suite green, confirming no false positives across all fixtures and the realistic demo data.
  Docs: `docs/spec-reference.md` checks section rewritten, `docs/quickstart.md` mention.
  Still open from section 3.6: categorical association (scan is numeric-only) and `no_future_columns` train/test-overlap detection; from 3.9: a deliberately leaky demo column as a teaching asset.
- 2026-07-06: DONE - P2 action item 5 (section 2.2, environment digest gaps).
  Manifests now record `env_freeze_digest`: a sha256 over every installed distribution (pip-freeze-like), so transitive drift (numpy/scipy) the targeted `env_digest` cannot see is pinned in the manifest.
  `run/build/test --manifest` and `evaluate --manifest` verify the environment before executing: `env_digest` mismatch is a hard error with a new `--allow-env-mismatch` downgrade flag; freeze-only mismatch warns; N-1 manifests without the field skip the freeze check.
  Verification preloads the manifest's node adapters so `fingerprint_packages()` matches what compile saw even when project files no longer parse; `current_env_digests()` in the compiler is the single shared computation.
  `mbt state diff` reports both digests; ADR-7 semantics unchanged (the freeze digest never triggers `state:modified`, by design - see ADR-19).
  Bonus fix: the ADR-18 bootstrap seed collided with the implicit-validation-carve seed (both `spec.seed + 2`); bootstrap moved to `seed + 3` and the derivation scheme is now documented consistently in code, ADR-18, and concepts.md.
  Verified: compile tests assert the freeze digest; a new orchestrator test covers tampered-manifest error, `--allow-env-mismatch` override, warn-only freeze drift, and N-1 skip; the churn e2e's `run --manifest` right after build proves no same-environment false positive; golden manifest regenerated (new metadata field); full fast suite and hooks green.
  Docs: ADR-19, `docs/gitops.md` environment section, `docs/cli-reference.md` flags, `docs/concepts.md` reproducibility contract, ADR count 19.
- 2026-07-06: DONE - P2 action item 8 (section 1.5, deprecated MLflow stage API).
  The MLflow registry now maps mbt stages to registered-model aliases by default (`use_aliases` default flipped to true); `use_aliases: false` keeps the legacy stage API for MLflow servers without alias support (< 2.9), and the FutureWarning suppression is scoped to that legacy path only.
  Alias transitions keep mbt's one-stage-per-version semantics: promoting a version drops its other canonical aliases, so moving staging -> production vacates the staging slot exactly like the stage API did.
  `ModelVersion.stage` is derived from aliases on the alias flow, so `mbt promote` and champion resolution report stages correctly.
  Side effect vs the old stage flow: promotion no longer auto-archives the incumbent (it simply loses the production alias), which improves the rollback story flagged in section 2.3 P3.
  Verified: adapter tests for the default alias flow, alias exclusivity plus stage derivation, and legacy stage mode; the churn_demo e2e (build -> promote -> evaluate champion by stage) runs the alias flow end to end after fixing its stale `current_stage` assertion; full fast suite, hooks, and mypy green.
  Docs: adapter module docstring rewritten, `docs/spec-reference.md` profiles sample documents the `use_aliases: false` escape hatch.
- 2026-07-06: DONE - P2 action item 10 (sections 3.4 and 2.7, claims vs reality).
  Slice gates are now a validated, tested capability instead of shipping untested: parse-time validation requires `gate.slice` to be `column=value` with the column declared under `evaluation.slices`; the runtime missing-slice hint now names the real causes (degenerate or absent slice value).
  `GateResult` records the gate's `slice` so run artifacts distinguish slice gates from whole-split gates on the same metric.
  Tests: gate-engine units for slice threshold gates, slice champion gates (point criterion; the whole-split ADR-18 bound must never leak into a slice decision), missing-slice hard error, parse validation, and an execution test proving a failing slice gate blocks registration with exit 2.
  The unbacked "mutation" testing claim was removed from the NFR-08 line in `docs/v0.1-status.md` (correcting the doc, per the review's either/or).
  Docs reconciled: `specs.py` comment, `v0.1-status.md` limitation line (now states the true caveat: champion slice gates compare point deltas), `spec-reference.md` slice-gate syntax, `concepts.md` mention.
  Verified: full fast suite and all hooks green; golden manifest byte-identical (no config-hash surface change).
- 2026-07-06: DONE - P1 action item 2 (section 3.4, point-estimate champion gates).
  Champion gates now pass only when the one-sided lower confidence bound (default 95%, 1000 resamples) of a seeded paired bootstrap over per-example predictions clears `min_delta`; both models score the same resampled rows of the pinned test split, so a challenger ahead on noise alone is blocked.
  New `GateSpec.confidence` / `GateSpec.bootstrap_resamples` fields (parse-validated; `confidence: null` opts out); bootstrap runs in the job (`_champion_delta_bounds`) via `adapter.predict`, travels in `JobResult.champion_delta_bounds`, and the pure gate engine compares (`gates.py`).
  Point comparisons remain only for slice gates, hook metrics, degenerate resamples, and explicit opt-out; the gate result records `delta_lower`, `confidence`, and a criterion message.
  Bootstrap seed = `spec.seed + 3` for byte-reproducible bounds (`+2` was already taken by the implicit validation carve; corrected after the collision was spotted in a later iteration).
  The fake training adapter's `predict` now emits ranking-quality scores consistent in direction with `fake_metric_value` so gate scenarios stay scriptable.
  Verified: new unit tests (`test_paired_bootstrap.py`, incl. the noise-promotion scenario now blocked), gate engine and spec validator tests, golden manifests deliberately regenerated (gate fields enter config_hash per ADR-6), full fast suite plus e2e suite (XGBoost/MLflow/JVM) green.
  Docs: ADR-0018, `docs/concepts.md`, `docs/spec-reference.md`, stale "15 ADRs" count in `docs/v0.1-status.md` corrected to 18 (partially addresses the P3 doc-drift item).
  Note: the mutation-testing claim in `docs/v0.1-status.md:24` (NFR-08) is still unreconciled (P2 action item 10).
- 2026-07-06: DONE - P1 action item 1 (section 2.1, unpinned scaffold workflow installs).
  All four scaffold workflows now run `pip install -r requirements.txt` instead of bare `pip install mbt-core mbt-xgboost mbt-mlflow`.
  The scaffold ships `requirements.in` plus `requirements.txt` with exact `==` pins, stamped at init time with the generating mbt version via a new `__MBT_VERSION__` template token in `cli/scaffold.py`.
  Full hash-verified transitive pinning is documented in the file headers and scaffold README (`uv pip compile --generate-hashes`); pre-shipping real hashes is impossible until the packages are on an index (publication is deferred per `docs/v0.1-status.md`).
  Verified: new `test_scaffold_ci_installs_are_pinned` asserts no workflow installs unpinned and pins match `mbt.__version__`; scaffold completeness test extended; fast suite and all pre-commit hooks green.
  Docs: scaffold README layout section and `docs/quickstart.md` updated.
- 2026-07-06: DONE - P1 action item 3 (section 1.3, pre-commit drift).
  `ruff-pre-commit` synced to `v0.15.20` (matches `uv.lock`) and yamllint bumped to `v1.38.0` in both `.pre-commit-config.yaml` and the `mbt init` scaffold.
  Renovate configs added at repo root and in the scaffold (pre-commit hooks, GitHub Actions, Python deps; ruff hook grouped with the locked ruff package).
  The new hook rev immediately caught a real drift artifact: `packages/mbt-h2o/tests/test_h2o_compliance.py` was committed with `v0.5.0`-era formatting that CI's `ruff format --check` rejects; it was reformatted.
  Verified: `pre-commit run --all-files` clean, `pytest -m "not e2e"` green, scaffold completeness test now asserts `renovate.json` ships.
  Docs: README development section and `docs/cli-reference.md` updated.
  Note: the Renovate GitHub App still needs to be installed on the hosting repo for the config to take effect.

## Executive summary

mbt is an unusually well-engineered v0.1 for its stated scope: the compile/hash/manifest core, adapter contracts, secrets handling, and test layering are genuinely strong.
The biggest problems are not in what exists but in the gap between what the project promises and what it enforces.

The five findings I would fix first:

1. **The shipped reference workflows install the toolchain unpinned** (`pip install mbt-core mbt-xgboost mbt-mlflow`), which contradicts the reproducibility promise the whole tool is built on (P1).
2. **Champion/challenger promotion decisions are single point-estimate deltas with no statistical uncertainty**, so a challenger can be promoted on test-set noise (P1).
3. **The pre-commit ruff rev (`v0.5.0`, June 2024) is two years behind the ruff CI actually runs (`0.15.20`)**, so local hooks and CI enforce different lint/format rules (P1).
4. **Toolchain versions lag latest stable across the board**: Python 3.11 only (3.14 is current), PySpark 3.5 (4.1 is current), GitHub Actions several majors behind, and MLflow registry stages used via a deprecated API (P2, detailed below).
5. **Monitoring, the sixth ml-ops.org principle, is essentially absent**: no OTel, no drift detection, no alerting, no serving-side signal of any kind (P2, partially acknowledged as deferred).

## Scorecard against ml-ops.org principles

| Principle (ml-ops.org) | Rating | Summary |
|---|---|---|
| Iterative-incremental development | Strong | PRD/TSD/PLAN plus 17 ADRs document the design loop end to end. |
| Automation | Partial (level 2 of 3) | Declarative pipeline automation and scheduled retraining exist; CI/CD automation has gaps (no release, no docs build, no coverage). |
| Continuous Integration | Good | Three-tier CI with an architectural import guard; but single Python version, no caching, no security scanning. |
| Continuous Delivery | Partial | Training pipelines and registry promotion are delivered; there is no prediction service or `mbt score`, so delivery stops at the registry. |
| Continuous Training | Partial | `scheduled_retrain.yml` is cron-driven only; ml-ops.org ties retraining frequency to model decay metrics, and no decay signal exists. |
| Continuous Monitoring | Missing | No production monitoring, no drift detection, no alerting; "drift" in this repo means snapshot drift (data versioning), not distribution drift. |
| Versioning | Strong | Two-hash identity, data snapshot pins, committed `uv.lock`, MLflow model versions; weakened by an env digest that ignores transitive deps. |
| Testing | Good | 170+ tests across unit/property/golden/compliance/E2E; but data and model tests (leakage, fairness, staleness) are opt-in or absent, and a documented capability (mutation testing) does not exist. |
| Reproducibility | Strong design, weak edges | Manifest pins everything the compiler sees, but the environment story leaks (unpinned reference workflows, partial env digest, mtime snapshots by default). |
| Loosely coupled architecture | Strong | Adapter contracts in a separate package, plugin import hygiene enforced in CI, LightGBM adapter proves the extension seam. |

Against the ml-ops.org ML Test Score rubric (features/data, model development, ML infrastructure), the infrastructure axis scores well but the features/data and model-development axes are thin: no automated leakage tests by default, no staleness tests, no fairness tests, no baseline-model comparison requirement.

## 1. Technology currency (the "not latest stable" audit)

Snapshot of declared vs locked vs latest stable, as of 2026-07-06.
Credit where due: most of `uv.lock` is current (pyarrow 24.0.0, duckdb 1.5.4, optuna 4.9.0, scikit-learn 1.9.0, lightgbm 4.6.0, snowflake-connector-python 4.6.0, mlflow 3.14.0, typer 0.26.8, pydantic 2.13.4, ruff 0.15.20, mypy 2.1.0, pytest 9.1.1).
The staleness is concentrated in five places.

### 1.1 Runtime and language

| Component | Repo | Latest stable | Evidence |
|---|---|---|---|
| Python | 3.11 pinned; CI tests 3.11 only | 3.14.6 (3.13.14, 3.12.13 also current) | `.python-version:1`, `ci.yml:15,46,58` |
| PySpark | `>=3.5,<3.6`, locks 3.5.8 | 4.1.2 | `packages/mbt-spark/pyproject.toml:15`, `uv.lock:3499` |
| pandas (transitive) | locks 2.3.3 | 3.0.3 | `uv.lock:2808` |
| rich | locks 14.3.4 | 15.0.0 | `uv.lock:3697` |
| xgboost | locks 3.2.0 | 3.3.0 | `uv.lock:4439` |

- P2: Python 3.11 is two majors behind current stable and reaches EOL in October 2027; `requires-python = ">=3.11"` advertises 3.12/3.13/3.14 support that is never tested (`pyproject.toml:5`, `ci.yml`).
- P2: the PySpark `<3.6` ceiling exists for `h2o-pysparkling-3.5` compatibility (`packages/mbt-h2o/pyproject.toml:20`), which is a defensible reason, but it silently holds every mbt-spark user a full major behind Spark 4.x; the constraint should be isolated to the `sparkling` extra rather than imposed on `mbt-spark` itself, and the tradeoff should be documented in ADR-17.

### 1.2 CI actions (all majors behind)

| Action | Used | Latest | Evidence |
|---|---|---|---|
| actions/checkout | v4 | v7.0.0 | `ci.yml:12,43,55`, all scaffold workflows |
| astral-sh/setup-uv | v5 | v8.3.0 | `ci.yml:13,44,56` |
| actions/setup-python | v5 | v6.3.0 | scaffold workflows |
| actions/setup-java | v4 | v5.4.0 | `ci.yml:59` |
| actions/upload-artifact | v4 | v7.0.1 | `_scaffold/.../prod_build.yml:42` |
| actions/github-script | v7 | v9.0.0 | `_scaffold/.../pr_check.yml:57` |

- P2: old action majors eventually stop running when GitHub retires their Node runtimes, and this bites the scaffold especially hard because `mbt init` stamps these stale versions into every new user project.
- P3: no Dependabot/Renovate config exists to keep any of this current automatically (`.github/` contains only `ci.yml`).

### 1.3 Pre-commit hooks (the worst offender)

- P1: `.pre-commit-config.yaml:3` pins `ruff-pre-commit` at `v0.5.0` while the workspace locks ruff `0.15.20` (`uv.lock:3833`) and CI runs it via `uv run ruff` (`ci.yml:19-21`).
  Ruff's formatter and rule set changed substantially across those ~2 years of releases, so the local hook can rewrite files in a style CI then rejects, and vice versa.
- P2: the same stale revs are baked into the scaffold (`_scaffold/.pre-commit-config.yaml:3,8`), so every `mbt init` project inherits the drift.
- P3: `yamllint` rev `v1.35.1` vs latest `1.38.0` (`.pre-commit-config.yaml:9`).

### 1.4 Declared floors vs tested reality

- P2: dependency floors span years of untested majors: `xgboost>=2.0` (locked 3.2), `mlflow>=2.9` (locked 3.14, and MLflow 3.x broke 2.x APIs), `optuna>=3.6` (locked 4.9), `snowflake-connector-python>=3.7` (locked 4.6), `mypy>=1.10` (locked 2.1), `ruff>=0.5`, `pytest>=8.0` (locked 9.1).
  Only the locked resolution is ever tested in CI, so the metadata promises compatibility ranges nobody verifies; either test the floors (a minimum-versions CI job with `uv sync --resolution lowest-direct`) or raise them to what is actually supported.
- P3: `types-PyYAML` is completely unbounded (`pyproject.toml:45`).

### 1.5 Deprecated API usage

- P2: the MLflow registry adapter defaults to the stage-based API and suppresses its `FutureWarning` (`packages/mbt-mlflow/src/mbt_mlflow/adapter.py:207-217`); MLflow deprecated stages in favor of aliases years ago and mbt already implements aliases behind `use_aliases`, so aliases should be the default before MLflow 4 removes stages.

## 2. MLOps engineering findings

### 2.1 CI/CD (ml-ops.org: Automation, CI/CD)

Strengths worth keeping:

- The NFR-04 guard that asserts no ML framework imports on mbt-core's import path is exactly the kind of architectural test most teams never write (`ci.yml:26-38`).
- The three-tier split (lint/type, unit, JVM-provisioned e2e) keeps the fast lane fast (`ci.yml:9-64`).

Gaps:

- P1: all four scaffold workflows install unpinned (`pip install mbt-core mbt-xgboost mbt-mlflow` at `pr_check.yml:28`, `prod_build.yml:25`, `promote.yml:32`, `scheduled_retrain.yml:20`).
  A tool whose README promises "the compiled manifest pins everything" (`README.md:84-88`) ships reference CI where the training environment floats on every run, which invalidates the env digest across runs and directly undermines G2/NFR-01.
  Fix: generate a `requirements.txt` or `uv.lock` in the scaffold and install with hashes.
- P2: no test coverage measurement at all (no `pytest-cov` in any `pyproject.toml`, bare `pytest -q` at `ci.yml:50`); 170+ tests with zero visibility into what they exercise.
- P2: no security scanning of any kind: no `pip-audit`, no CodeQL/bandit, no `gitleaks`, no Dependabot, on a codebase that resolves warehouse credentials (`config/profiles.py`).
- P2: the mkdocs site is never built in CI and never published, and there is no release/PyPI pipeline (acknowledged as deferred in `docs/v0.1-status.md:26`), so docs rot silently and the 10 "publishable" packages have no publish path.
- P2: no failure alerting in any workflow; a failed Monday 05:00 scheduled retrain (`scheduled_retrain.yml:7`) fails silently, which violates the Continuous Training principle's operational side.
- P2: `prod_build.yml:34-45` writes the "published" manifest into the ephemeral workspace and only uploads a per-run artifact, so out of the box there is no durable prod state baseline and the `state:modified` economy loop (G3) is a no-op until someone hand-wires S3.
- P3: no uv cache in CI (`setup-uv@v5` without `enable-cache`), so every job re-resolves the workspace.

### 2.2 Reproducibility and versioning (ml-ops.org: Reproducibility, Versioning)

Strengths worth keeping:

- The two-hash identity model (config hash + transitive input hash, profiles excluded) is sound and golden-tested (`compile/hashing.py:31-44`, `tests/test_golden_manifest.py`, ADR-4/ADR-5).
- Snowflake snapshot pinning with hard failure on drift, and deterministic push-down sampling, are the right design (`mbt_snowflake/adapter.py:151-164`, `mbt_snowflake/sql.py:63-77`).

Gaps:

- P2: `env_digest` hashes only Python micro, `mbt-*` versions, and adapter `fingerprint_packages` (`compile/hashing.py:47-60`); a numpy or scipy bump that changes XGBoost numerics is invisible to it.
  Include a hash of the resolved environment (e.g., `uv.lock` hash or `pip freeze` digest) in the manifest.
- P2: `mbt build --manifest` never verifies the env digest at rebuild time; it is only a `state diff` signal (`state/diff.py:30,118`), so "reproduce this manifest" silently proceeds in a mismatched environment.
- P2: `mbt deps` installs adapters via `pip install` against loose PEP 440 specifiers from `packages.yml` with no lock or hash verification (`deps.py:44-59`).
- P3: the default local snapshot is mtime-based, which ADR-11 itself admits "lies" in CI and fresh checkouts; `--deep-snapshot` (content hashing) should be the default, or at least the default in CI.

### 2.3 Registry, promotion, deployment (ml-ops.org: Deployment, CD)

Strengths worth keeping:

- Promotion refuses versions whose gates were not recorded as passed, and `--force` is auditable (`promote.py:84-108`).
- Champion re-evaluation in-job on the identical pinned split (ADR-9) and the missing-vs-unloadable champion distinction (ADR-10) are principled.

Gaps:

- P2: there is no serving, batch scoring, or inference surface at all; `mbt score` is v1 roadmap (`docs/roadmap.md:24`), and exposures are descriptive metadata only (`artifacts/manifest.py:65-73`).
  ml-ops.org's deployment principle expects the model to reach a prediction service; today the pipeline ends at the registry, which is a legitimate v0.1 scope cut but means the tool cannot yet close the loop it draws in its own GitOps diagram.
- P3: no first-class rollback; promotion to Production auto-archives the incumbent (`mbt_mlflow/adapter.py:216`), so recovery from a bad promotion is a manual re-promote, and MTTR (an ml-ops.org delivery metric) has no tooling.

### 2.4 Monitoring and observability (ml-ops.org: Monitoring, Continuous Monitoring)

- Strength: typed Pydantic events with console and JSON-lines sinks, redaction on every write, machine-readable `run_results.json` (`events/models.py`, `events/sinks.py:31-58`).
- P2: OTel is documented as deferred (`docs/v0.1-status.md:22,41`) and nothing exports to standard observability backends.
- P2: no statistical drift detection exists anywhere; all quality checks run at build time against training data (`quality/checks.py:210-216`), and `docs/quickstart.md:112` mislabels champion re-evaluation as a "drift check", which conflates data versioning with distribution monitoring.
- P2: no alerting integration (no slack/webhook/pagerduty anywhere in `_scaffold` or `.github`).

### 2.5 Secrets

- Strength: env_var-only ingress, taint-and-redact on all sinks, and the manifest stores target config unrendered so secrets never reach the primary artifact (`config/profiles.py:90-116`, `secrets.py:18-52`, `artifacts/manifest.py:47-49,143`).
  This is better than most production ML platforms.
- P3: `redact()` is exact-substring replacement of tainted values only (`secrets.py:45-52`); secrets read directly from `os.environ` in hooks/adapters, or transformed (base64, URL-encoded), escape redaction.

### 2.6 Storage and artifact lifecycle

- P2: the artifact store is `file://` only (`storage.py:16-20`); trained models cannot land in object storage, which blocks any multi-runner or production topology.
- P2: there is no retention, GC, or pruning of `target/` or the artifact store; every run writes a new uuid prefix forever (`storage.py:23,29-39`), and the repo's own `target/fake_tracking/` full of leftover test JSON illustrates the unbounded growth.
- P3: tests write into the repo-root `./target` instead of a tmp dir (`tests/test_perf_budgets.py:36`), polluting the working tree.

### 2.7 Testing infrastructure (ml-ops.org: Testing)

Strengths worth keeping:

- The per-adapter compliance suite (determinism tiers, import hygiene, export/load round-trip, "actually learns") is a reusable conformance asset (`mbt_adapter_base/compliance/suite.py:161-256`).
- Golden manifests, perf budgets, property tests, and spec-only adapter swap are a layered strategy most tools this age lack.

Gaps:

- P2: `docs/v0.1-status.md:24` claims "mutation" testing under NFR-08, but no mutation tool (`mutmut`, `cosmic-ray`) exists in any dependency set; the only "mutation" is a test-name string in `test_compile.py`.
  Either add mutation testing or correct the status doc; a status document that overstates rigor is worse than the gap itself.
- P3: no flaky-test mitigation (no `pytest-rerunfailures`, no `pytest-xdist`, no `pytest-randomly`) even though the JVM e2e tier (H2O/Spark) is flake-prone by nature.
- P3: perf budgets are bare wall-clock asserts on shared CI runners with no warmup or tolerance (`tests/test_perf_budgets.py:87,98,117`), which will flake under noisy neighbors.

### 2.8 Documentation and operability

- Strength: 17 ADRs, a spec reference, quickstart, gitops and adapter-authoring guides; unusually complete for v0.1.
- P2: no troubleshooting guide or runbook exists for the failure modes the code deliberately produces (snapshot mismatch, unloadable champion, gate-edit retrain signals); operators have design docs but no playbook.
- P3: doc drift: `docs/v0.1-status.md:25` says "15 ADRs" while 17 exist; small, but this is the document that markets the project's rigor.

## 3. Data science findings

### 3.1 Task coverage

- P2: only binary classification works end to end; the task registry hardcodes a single entry (`config/tasks/__init__.py:11-13`) and every adapter pins `supported_tasks = {BINARY_CLASSIFICATION}` (`mbt_xgboost/adapter.py:67`, `mbt_lightgbm/adapter.py:61`).
  The extension seam (`register_task_schema`) is clean, so regression should be the next vertical; a "dbt for ML" that cannot fit a regression is a hard sell to most DS teams.
- P2: the metric layer is binary-only (`mbt_adapter_base/metrics.py:21-32,140-167`); no RMSE/MAE/R2 or multiclass path exists.

### 3.2 Splitting and validation protocol

Strengths worth keeping:

- Temporal split as the default, resolved as disjoint SQL time windows against a single anchor, is the right call for the churn-style use case (`adapters/local/data.py:228-248`, ADR-12).
- Random splits require an explicit seed, and hash-based ranking over `sample_key` gives implicit entity-grouped splits when `sample_key` is the entity id (`specs.py:93-96`, `data.py:195-208`).

Gaps:

- P2: there is no explicit grouped-split control and no warning when it is absent; without `sample_key`, the row digest falls back to all columns (`data.py:200-203`), so the same customer can straddle train and test with no signal to the user.
- P2: `strategy: random` is allowed on datasets with a time column and nothing detects or warns about the temporal leakage this invites (`specs.py:72,92-109`); the framework should at minimum warn when a random split coexists with a time column.
- P2: no cross-validation exists anywhere; tuning selects on a single validation carve (see 3.5) and H2O's `nfolds` defaults to 0 (`mbt_h2o/params.py:28`).

### 3.3 Metrics

- Strength: the binary set is genuinely good: `roc_auc, pr_auc, logloss, brier, ece, recall_at_precision, precision_at_recall` with parameterized sugar, computed identically across adapters on the held-out test split (`metrics.py:21-53`, `execute/job.py:456-460`).
- P2: no lift/gain/decile or top-k precision as first-class metrics; the churn demo hand-rolls `lift_at_decile` as a hook (`examples/churn_demo/models/churn_classifier.py:4-14`), yet lift tables are the lingua franca of churn/marketing DS.
- P2: slice metrics silently drop degenerate slices (`metrics.py:160-161`) and slicing a continuous column explodes into one slice per unique value with no binning.
- P3: `accuracy` uses a fixed 0.5 threshold with no operating-point selection (`metrics.py:129`).

### 3.4 Evaluation gates

- Strength: the gate engine is pure and direction-aware, tolerance widening applies only in the model's favor on absolute gates, and missing-champion bootstrap passes loudly (`quality/gates.py:35-116`, ADR-10).
- P1: champion comparison is a bare point-estimate: `passed = delta >= gate.min_delta` (`gates.py:92-97`).
  There is no bootstrap CI, no paired test, and no accounting for test-set size, so with the small test windows typical of temporal splits a challenger can be promoted on noise; `min_delta` is the only, blunt, mitigation.
  A paired bootstrap over per-example scores is cheap (both models' predictions on the identical pinned split are already in hand thanks to ADR-9) and would make the promotion decision statistically defensible.
- P2: slice gates are contradictory: `gates.py:16-25` fully evaluates them, but `specs.py:204` and `docs/v0.1-status.md:31` call them report-only/deferred; either the docs understate a shipped capability or untested behavior is shipping.

### 3.5 Hyperparameter tuning

- Strength: seeded TPE with clean search-space validation, tuning seed derived separately from the training seed, and ADR-8 ("tuning never sees test") verified in code: trials score on validation only, and an implicit carve is reabsorbed into the final fit while an explicit one stays held out (`mbt_optuna/engine.py:16-59`, `execute/job.py:387-423`).
- P2: best-trial selection is max-over-n-trials on one small validation set (often an implicit 20% carve), a classic optimistic-bias setup with no CV, repeated folds, or re-validation of the winner (`engine.py:60-63`, `job.py:295-322`).
- P3: no Optuna pruner and no `trial.report()` integration, so every trial trains to completion (`engine.py:51-59`), which is wasteful at scale.

### 3.6 Leakage protections

- Strength: the split time column is always stripped from features on every materialization path (`execute/handles.py:30-50`, `job.py:160`), and window-expression hashing means anchor drift cannot silently shift train into test (ADR-12).
- P2: beyond that, leakage prevention is user discipline: `label_leakage_scan` is opt-in, numeric-only, and fires only at `|corr| >= 0.95` (`quality/checks.py:148-185`); the churn demo does not even enable it, and sibling-label exclusion is manual (`churn_classifier.yml:14`).
  A framework whose README markets "explicit leakage guards" should enable the scan by default and lower the bar.
- P2: `no_future_columns` only catches values later than the newest window end (`checks.py:110-145`), i.e. absolute-future timestamps, not train/test overlap.
- P3: the demo's `test_no_leakage.py` tests label binarity and an activity filter, not leakage (`examples/churn_demo/tests/test_no_leakage.py:7-22`); the filename writes a check the code does not cash.

### 3.7 Feature handling and explainability

- P2: no native categorical support: both tree adapters densify to float and hard-error on any non-numeric column (`mbt_xgboost/adapter.py:133-166`, `mbt_lightgbm/adapter.py:122-146`), even though both frameworks support categoricals natively (`enable_categorical`, `categorical_feature`).
  The demo consequently throws away `plan_type`, its only categorical and a plausibly predictive feature (`churn_classifier.yml:14`).
- P2: no feature importance, SHAP, or any explainability output anywhere; model cards render hyperparameters and metrics only.
  For a tool that positions model specs as reviewable governance artifacts, a per-model importance table in `mbt docs generate` is table stakes.
- P2: no fairness tooling: slices exist, but there is no disparity metric or fairness gate, which ml-ops.org lists under model-development testing.

### 3.8 Class imbalance

- Strength: `{{ auto }}` `scale_pos_weight` computed from the observed class balance in both tree adapters, with a parse-time warning path for extreme imbalance (`mbt_xgboost/adapter.py:105-123`, `config/tasks/binary.py:75-86`).
- P2: no threshold tuning or calibration step exists; probability reweighting is the only lever, and the calibration metrics that do exist (brier, ece) are never exercised by the demo, despite churn probabilities being what interventions consume.

### 3.9 The churn demo as a DS exemplar

- Strength: the scaffolding is methodologically correct: temporal split, id and sibling-label exclusion, PR-AUC primary, champion/challenger pair, slice reporting.
- P2: the gate is vacuous: `pr_auc_floor: 0.05` (`examples/churn_demo/mbt_project.yml:5`) against a ~20% positive rate whose random baseline PR-AUC is roughly 0.20, so any non-degenerate model passes and the demo's "quality gate" never actually gates.
  The first thing a prospective user copies from the example is a threshold that teaches the wrong lesson; set the floor above the base rate.
- P3: 2,400 synthetic rows with a logistic label generated from the same three features the model uses (`scripts/generate_data.py:16,38`) cannot surface collinearity, drift, or leakage; a slightly richer generator (irrelevant features, a deliberately leaky column for the scan to catch, more rows) would make the demo a teaching asset.

## 4. Prioritized action list

P1 (fix before wider adoption):

1. Pin the scaffold workflows' installs (locked requirements with hashes) so reference CI is reproducible (`_scaffold/.github/workflows/*.yml`).
2. Add statistical uncertainty to champion gates (paired bootstrap CI on the pinned test split) (`quality/gates.py:92-97`).
3. Sync `ruff-pre-commit` to the locked ruff version and add automation (Renovate/Dependabot) to keep hooks, actions, and floors current (`.pre-commit-config.yaml:3`).

P2 (next quarter):

4. Test what you claim: CI matrix over Python 3.11-3.14, and either a lowest-resolution CI job for the dependency floors or raised floors.
5. Fold the resolved-environment hash into `env_digest` and verify it on `mbt build --manifest`.
6. Close the CI gaps: coverage reporting, pip-audit/CodeQL, docs build+publish, release pipeline, failure alerting, durable prod-state publication.
7. Ship the S3 artifact store and a retention/GC story.
8. Default the MLflow registry to aliases instead of deprecated stages.
9. DS methodology: grouped-split control plus random-on-temporal warning, leakage scan on by default, native categorical support, feature importance in model cards, lift/gain metrics, and a non-vacuous demo gate.
10. Reconcile slice-gate code with docs, and remove or implement the mutation-testing claim in `docs/v0.1-status.md`.
11. Isolate the PySpark `<3.6` ceiling to the `sparkling` extra and validate Spark 4.x in the main adapter.

P3 (opportunistic):

12. Deep snapshots by default in CI, tmp-dir test artifacts, uv cache, flaky-test reruns, perf-budget tolerances, Optuna pruning, richer demo data, threshold/calibration tooling, runbooks, and the stale "15 ADRs" count.

## 5. What is genuinely good (keep it)

- The two-hash reproducibility model with golden-manifest byte-determinism testing.
- The NFR-04 CI guard keeping ML frameworks off mbt-core's import path, and the adapter compliance suite as a conformance contract.
- ADR-8/9/10 discipline: tuning never sees test, champions re-evaluated on the identical pinned split, missing-vs-unloadable champion distinction.
- Secrets: env_var-only ingress, taint-based redaction on every sink, unrendered target config in the manifest.
- Warehouse-native snapshot pinning with hard drift failures and deterministic push-down sampling.
- Seventeen ADRs that record why, not just what; most teams cannot answer "why is this lazy import here" two years later, and this repo can.
