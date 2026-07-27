# Troubleshooting runbook

The failure modes on this page are deliberate: each one is mbt refusing to do something unreproducible, unsafe, or statistically indefensible.
Every symptom below was reproduced against the real CLI; the messages are verbatim.

Exit codes (TSD §17): `0` success, `1` hard error (something is broken), `2` quality failure (the pipeline ran; a gate, check, or test said no).
CI should treat `1` as "page someone" and `2` as "review the model, not the plumbing".

## `git checkout -q v0.1.0 did not run successfully` installing a scaffolded project

**Symptom:** `pip install -r requirements.txt` in a fresh `mbt init` project (locally or in its first CI run) fails:

```text
  error: subprocess-exited-with-error

  × git checkout -q v0.1.0 did not run successfully.
  │ exit code: 1
  ╰─> No available output.

  note: This error originates from a subprocess, and is likely not a problem with pip.
ERROR: Failed to build 'mbt-core' when git checkout -q v0.1.0
```

**Why:** the scaffold pins mbt to the immutable release tag matching the mbt version that generated it (`mbt-core @ git+https://github.com/satrijandi/mbt@v0.1.0#...`), and that tag has not been cut on the mbt repo yet.
The pin is deliberate - a floating ref would invalidate the manifest's env digest - but it only resolves once the release exists.

**Fix:** until the tag is cut, replace `@v0.1.0` in all three refs in `requirements.txt` (and `requirements.in`) with the commit SHA you installed mbt from - a SHA is just as immutable, so reproducibility is preserved - and re-run the install.
Once mbt is released (tag on the repo, or packages on PyPI), restore the tag pin or switch to plain version pins per the file's header.

## Everything is `state:modified` but nobody changed anything

**Symptom:** `mbt state diff` flags datasets as `modified` with component `snapshot`, and every downstream model as `modified` with component `upstream`:

```json
{"unique_id": "dataset.churn_demo.churn_training_set", "change": "modified", "components": ["snapshot"]}
{"unique_id": "model.churn_demo.churn_classifier", "change": "modified", "components": ["upstream"]}
```

**Why:** the default local snapshot is an mtime listing (fast, but it lies - ADR-11).
A `touch`, a fresh CI checkout, or a copied workspace changes mtimes without changing bytes, and the snapshot token moves.

**Fix:** use content-hash snapshots on both sides of the comparison: publish prod manifests from `mbt compile --deep-snapshot` (or `mbt build --deep-snapshot`) and diff with `mbt state diff --deep-snapshot --state ...`.
Deep-vs-deep comparisons ignore mtime churn entirely; a deep baseline diffed against a default (mtime) compile - or vice versa - flags every dataset, because the token schemes differ.
Pick one scheme per pipeline and use it everywhere.
The workflows shipped by `mbt init` already pass `--deep-snapshot` on every compiling step; keep the flag if you edit them, and use it locally whenever you diff against a CI-published baseline.

## A one-line gate edit retrained every model

**Symptom:** after editing a gate threshold or a shared var, `mbt state diff` marks models `modified` with component `config`.

**Why:** gates and vars are part of the model's config hash, so editing them re-opens the model's evidence (ADR-6: a model whose bar moved must prove it clears the new bar).
A var referenced by many models (like the demo's `pr_auc_floor`) moves all of their hashes at once.

**Fix:** nothing is broken - the retrain is the feature.
The PR comment and `mbt state diff --output json` show exactly which nodes retrain and why (`components: ["config"]`); review that list before merging.

## `selector 'state:modified' requires --state`

**Symptom:**

```text
Error: selector 'state:modified' requires --state <path-or-URI> pointing at a
reference manifest
  hint: e.g. --state s3://bucket/mbt/proj/prod/manifests/latest.json
```

**Why:** `state:modified` is a comparison; without a baseline manifest there is nothing to compare against, and guessing (e.g. "treat everything as modified") would silently retrain the world.

**Fix:** fetch the baseline first.
Scaffolded projects: `bash scripts/fetch_state.sh` restores the prod baseline from the `mbt-state` branch to `state/prod/latest.json` (see [GitOps & CI](gitops.md)).

## `the current environment does not match the manifest's env_digest`

**Symptom (hard error, exit 1):**

```text
Error: the current environment does not match the manifest's env_digest
(manifest sha256:0000..., current sha256:08c4...)
  hint: reinstall the environment the manifest was compiled in, or pass
--allow-env-mismatch to execute anyway (breaks reproducibility)
```

**Why:** `--manifest` promises "reproduce this run"; executing it in a different environment (other adapter versions, other Python) silently breaks that promise, so mbt refuses up front (ADR-19).
A related warning-only variant fires when only `env_freeze_digest` (transitive packages) drifted.

**Fix:** reinstall the pinned environment the manifest was built in (the scaffold's `requirements.txt` pins it).
`--allow-env-mismatch` downgrades the error to `WARN ... proceeding (--allow-env-mismatch)` - use it only when you accept that metrics may not reproduce.

## `gate pr_auc (threshold): FAIL` and the build exits 2

**Symptom:**

```text
gate pr_auc (threshold): FAIL - expected 0.9, got 0.3191...
[2/2] GATE_FAILED model model.churn_demo.churn_classifier - one or more gates failed
build finished [quality_failure]: 1 ok, 1 failed, 0 skipped
```

**Why:** exit code 2 means quality failure: the pipeline is healthy, the model did not clear its declared bar, and registration was withheld (nothing reached the registry).

**Fix:** read the gate table in the output or `target/run_results.json` (`gates` per node).
Either improve the model or - deliberately, in a reviewed PR - move the bar; remember that gate edits retrain (see above).

## `label_leakage_scan: suspiciously label-associated features`

**Symptom (exit 2):**

```text
TEST_FAILED dataset dataset.churn_demo.churn_training_set - 1 check/test
failure(s): label_leakage_scan: suspiciously label-associated features:
days_since_cancellation (|corr|=1.000)
```

**Why:** the scan runs on every dataset build, declared or not: numeric columns are screened with |corr| against the label, string/categorical columns with Cramér's V (reported as `V=...`, same 0-1 bar) - ≥ 0.95 fails the build, 0.85-0.95 warns.
A feature this associated with the label is almost always leakage (a post-outcome column, a status string that encodes the outcome, a sibling label).

**Fix:** if it is leakage, exclude the column in the dataset/model spec - do not widen the threshold.
If it is a genuine, available-at-prediction-time signal, declare the check with `exclude: [column]` (the exception is then visible in the spec diff) or tune `max_abs_correlation`; `enabled: false` opts out loudly (recorded as "check disabled").

## A build exits 2 with `check <name>: FAIL` or `test <name>: FAIL`

**Symptom (exit 2):**

```text
check not_null: FAIL - train.churned: 3 null(s)
test test_row_count: FAIL - raised AssertionError()
TEST_FAILED dataset dataset.churn_demo.churn_training_set - 1 check/test failure(s): not_null: train.churned: 3 null(s)
```

**Why:** every built-in check and every Python data test emits its own PASS/FAIL event as it runs, so a failing build names the exact check or test on its own line instead of only in the aggregated node status.
The `TEST_FAILED` line that follows is the node summary; the per-check and per-test lines above it pinpoint which one failed and why.
`PASS` lines are normal and confirm each check and test actually ran.

**Fix:** read the message on the `FAIL` line - it carries the specific reason (a null column, a raised assertion, a schema mismatch).
The same failures are in `target/run_results.json` under each node's `tests`.
For label-association failures see the `label_leakage_scan` entry above; an explicitly disabled check reads `check <name>: PASS - check disabled`.

## `champion gate: no champion in 'production' yet - passing with a warning`

**Symptom:**

```text
WARN champion gate on 'pr_auc': no champion in 'production' yet - passing with a warning (FR-TEST-06)
gate pr_auc (champion): PASS - no champion registered yet; gate passes (bootstrap)
```

**Why:** a missing champion is the expected bootstrap state of a new model, not an error; blocking would make the first deployment impossible (ADR-10).
It passes *loudly* so a champion that vanished by accident is never mistaken for a bootstrap.

**Fix:** nothing - unless you expected a champion to exist, in which case check the registry stage (`mbt promote` history) and the tracking server you are pointing at.

## `artifact not found: file://.../model.ubj` during a champion build

**Symptom (hard error, exit 1):**

```text
ERROR model model.churn_demo.churn_classifier - artifact not found:
file://.../target/artifacts/churn_classifier/20260706T.../model.ubj
  hint: the artifact store may have been cleaned; re-run the build
```

**Why:** the champion is *registered* but its artifact cannot be loaded, and a champion gate needs its predictions (ADR-9 re-evaluates the champion on the identical split).
Unlike a missing champion (bootstrap pass, above), an unloadable one is a hard error: silently skipping the comparison would promote challengers unchecked (ADR-10).

**Fix:** for a champion artifact, "re-run the build" is not enough - rebuilding trains a new challenger; it does not restore the champion's file.
Restore the artifact store (backup, or the object-store copy if you point `artifact_store` at s3), or deliberately re-baseline: promote a freshly built version to the champion stage.
Note `mbt clean --artifacts-older-than` always preserves stage champions, so this state usually means manual deletion or an artifact_store misconfiguration (wrong root, different machine).

## `gate on slice 'plan_type=platinum' has no slice metrics`

**Symptom (hard error, exit 1):**

```text
ERROR model model.churn_demo.churn_classifier - gate on slice
'plan_type=platinum' has no slice metrics for the challenger
  hint: the slice value must occur in the test split with both classes present;
degenerate slices are dropped from metrics
```

**Why:** a slice gate whose slice has no rows (or only one class) in the test split cannot be evaluated, and passing it silently would fake coverage of a segment that was never measured.

**Fix:** check the slice value's spelling against the data, and its frequency in the *test window* specifically - a value can exist in training data yet be absent from the test period.
If the segment is genuinely rare, widen the test window or drop the slice gate.

## `dataset materialization snapshot mismatch`

**Symptom:**

```text
dataset materialization snapshot mismatch: <current> != <pinned>
  hint: the data moved under a pinned manifest; recompile or restore the data
```

(Snowflake wording: `the manifest pin and the materialized data disagree; recompile`.)

**Why:** a pinned manifest records the exact data snapshot it was compiled against; if the warehouse table or local files changed since, executing the pin would claim reproducibility it no longer has.
Warehouse adapters fail hard on this by design (TSD §8.3).

**Fix:** if the data change is expected, recompile (`mbt compile` / a fresh `mbt build`) so the pin moves with it.
If it is not expected, treat it as an incident: something wrote to a table your manifest pinned.

**This applies to datasets only.** A scoring input (`mbt score`) and the arriving labels (`mbt monitor`) are expected to change every run, so they are exempt: a pinned manifest scores/monitors the live data instead of raising (R2-10).
That is what makes `mbt score --manifest` / `mbt monitor --manifest` usable as the reviewed, pinned artifact a scheduled job runs from - only the training data is held immutable.

## `no champion of 'churn_model' in stage 'production' to score with`

**Symptom (hard error, exit 1):**

```text
no champion of 'churn_model' in stage 'production' to score with
  hint: train and promote the model first (mbt build, then mbt promote)
```

**Why:** unlike a missing gate comparator (a bootstrap state that passes with a warning, ADR-10), a scoring pipeline with nothing to score WITH is an operational failure: `mbt score` would otherwise silently produce nothing every night.
The scoring spec's `stage` (default `production`) names the registry alias the champion is resolved from at run time (ADR-20).

**Fix:** build and promote the referenced model (`mbt build`, then `mbt promote --model <name> --to production`), or point the pipeline's `stage` at the stage you actually register to.

## Rolling back a bad champion (incident procedure)

**Situation:** a version you just promoted to `production` is misbehaving - an `mbt monitor` realized-metric breach, or a downstream incident - and you need to revert to the previous champion now.

**Steps:**

1. Roll back. `mbt rollback` reverts the production champion to the most recent version below it that recorded passing gates - the last known good - so you do not have to look up a version number mid-incident:

   ```text
   mbt rollback --model churn_classifier
   -> ROLLBACK: churn_classifier production reverted from v3 to v2
   -> rolled back churn_classifier to v2 in production
   ```

   To revert to a specific earlier version instead, pass `--to-version N`. In CI, drive this from a `workflow_dispatch` job; `promotions.yml`'s git history remains the audit trail of every forward promotion.
2. Verify: the registry now resolves the reverted version as the production champion and the bad version becomes the newly archived one; `mbt score` and `mbt monitor` pick up the reverted champion on their next run (the champion is resolved from the registry at run time, ADR-20).

**Note:** rollback re-promotes through the same recorded-gate check as `mbt promote` (FR-REG-03), and the last-known-good target already passed its gates, so it reverts cleanly. A version promoted before gate recording, or whose run metadata is gone, needs `--force`.

**If it refuses with `cannot roll back ... no longer exists (aged out by 'mbt clean' or a bucket lifecycle rule)`:** the target's artifact reference survives in the registry but the file behind it is gone, so moving the alias would only relocate the failure to the next `mbt score` (F12). The refusal happens BEFORE the alias moves; pick an earlier version whose artifact still exists (`--to-version N`) or re-train. On a store the probe cannot reach (an unrecognized scheme, or s3 without the s3 extra installed in the operator environment), rollback proceeds but logs `could not verify the artifact` - treat that warning as a prompt to check the artifact by hand.

## `the champion was trained with a different hooks.py than the current project's`

**Symptom (hard error, exit 1):**

```text
the champion was trained with a different hooks.py than the current project's (mbt.hooks_hash mismatch)
  hint: retrain and promote, or check out the commit the champion was built from
```

**Why:** scoring applies the model's `transform_features` hooks from the CURRENT checkout, but the champion was trained with the hooks recorded at its registration (`mbt.hooks_hash`).
Scoring through different feature transforms than the champion learned on is silent skew - worse than a hard stop (ADR-20).

**Fix:** retrain and promote so the champion matches the current hooks, or run the scoring pipeline from the commit the champion was built from (`--manifest` keeps that reproducible).
A champion registered before this release has no `mbt.hooks_hash` tag; scoring then proceeds with a warning that parity cannot be verified.
ADDING a hooks file to a model that had none triggers this too (the hash goes from empty to set): for example, a long-lived showcase stack whose wide champion predates `models/wide_hooks.py` fails wide scoring until `make wide` retrains and promotes.

## `WARN champion has no monitoring baseline (registered by an older mbt)`

**Symptom (exit 0, monitors pass):**

```text
WARN champion has no monitoring baseline (registered by an older mbt);
shift monitors pass without comparison - retrain to capture a baseline (ADR-21)
```

**Why:** shift monitors compare the batch against the baseline captured when the champion trained; a champion registered before baselines existed has nothing to compare against.
In the ADR-10 spirit this passes loudly instead of blocking scoring.

**Fix:** retrain and promote once; every training job now exports a baseline and registration pins it to the version.

## `data adapter '<name>' does not support batch scoring`

**Symptom (hard error, exit 1, before any job runs):**

```text
data adapter 'acme_lakehouse' does not support batch scoring (contract 1.1 adds build_scoring_input and open_predictions)
  hint: upgrade the adapter package, or score against a target whose data adapter supports scoring (the built-in local, snowflake, and spark adapters all do)
```

**Why:** `mbt score` needs the contract 1.1 data-adapter methods (materialize an unlabeled batch; open a prediction store).
All three built-in data adapters (local, snowflake, spark) ship them, so this fires only for a third-party adapter built against contract 1.0, which still loads and trains fine - the capability is probed up front so the failure is immediate and clear (ADR-21).

**Fix:** upgrade the adapter package to a release that implements scoring, or run the pipeline against a target whose data adapter does (any built-in adapter ships it).

## `mbt monitor` says `evaluated 0 of 1 matured prediction run(s)`

**Symptom (exit 0, with a warning):**

```text
WARN run 06e35b21ab994b83: no matured labels joined (join_key: user_id); will retry next monitor run
```

**Why:** the prediction run's maturity lag has passed, but the ground-truth table contains no rows joining to its predictions (labels have not landed yet, or the join key is wrong).
The run is deliberately NOT marked evaluated, so it retries on the next monitor run once labels arrive (ADR-21); the same applies when matured labels are single-class (metrics would be undefined).

**Fix:** nothing, if labels are simply late - the next scheduled `mbt monitor` picks the run up.
If it persists, check `ground_truth.join_key` against the label table's columns and verify the label pipeline delivers to the configured source.

## `mbt monitor` skips a run with `unparseable scored_at`

**Symptom (exit 0, with a warning):**

```text
WARN skipping prediction run 'a1b2c3d4e5f60718': unparseable scored_at 'not-a-timestamp'
```

**Why:** a prediction run's `predictions.json` sidecar carries a `scored_at` that is not an ISO-8601 timestamp.
mbt's own scoring path always writes the manifest anchor, so this points at a run written (or edited) by an external store.
One malformed sidecar is skipped rather than allowed to fail the whole monitor node with a bare `ValueError` (R2-19), so every other run in the store still evaluates.

**Fix:** repair or remove that run directory (`scored_at` must be an ISO-8601 timestamp); the next `mbt monitor` then picks the run up.
If an external store produced it, fix the store's `scored_at` serialization.

## `BREACH: drifted share 0.88 > max 0.30` from the showcase Evidently gate

**Symptom (quality verdict, exit 2):**

```text
  dem_f02                  drift score 1.5929
  log_f09                  drift score 1.5905
  days_since_login         drift score 1.4475
  ...
BREACH: drifted share 0.88 > max 0.30 (phase serving)
```

**Why:** the showcase's wide batch-monthly cadence runs `scripts/evidently_gate.py` on exactly the features `churn_wide_automl` trains on (the committed include list).
The train phase compares the train window against the test window and blocks `mbt promote` on a breach; the serving phase compares each scored batch against the baseline the train phase exported.
Exit 2 is the same quality-verdict semantics as mbt's own gates, so the Airflow DAG fails the task without retries and notifies the owner.

**Fix:** open `drift_report.html` to see the per-feature comparison.
A train-phase breach means the features were already unstable inside the training window: revisit the split boundaries or rerun `scripts/select_features.py` so selection sees the shifted period.
A serving-phase breach means the incoming month shifted (as `make inject-drift` demonstrates for the daily cadence): fix the upstream data or retrain on the new distribution before promoting again.
Raising `--max-drift-share` is a deliberate policy change, not a fix.

## `error: no exported reference at /workspace/monitoring/wide_reference.parquet`

**Symptom (hard error, exit 1):**

```text
error: no exported reference at /workspace/monitoring/wide_reference.parquet; run the train-phase gate first
```

**Why:** the serving-phase gate needs the baseline the train-phase gate exports on a pass.
DAG task containers are ephemeral (`target/` dies with each one), so the baseline must live on the shared `/workspace` mount; a fresh stack that never ran the train phase has nothing there.

**Fix:** run the train phase once after a wide build - `python scripts/evidently_gate.py --phase train --export-reference /workspace/monitoring/wide_reference.parquet` (or simply `make wide`, which does this between the AutoML build and promotion).

## A mistyped flag prints `No such option`

```text
Error: No such option: --deep-snapsht
Try 'mbt state diff --help' for help.
```

Exit code is 1 (hard error), so CI scripts fail fast on typos instead of proceeding without the flag they thought they passed.

## `training job timed out after 3600s and was killed`

**Symptom (node errors, exit 1):**

```text
training job timed out after 2s and was killed (job payload kept at /tmp/mbt-job-…/job.json)
```

**Why:** the target's compute config sets `job_timeout_seconds`, and this job outlived it.
Without the limit, a wedged training job (an infinite loop, a hung network call inside a framework) blocks the whole run forever; with it, the watchdog SIGTERMs the subprocess (SIGKILL after a grace period) and the node reports the reason.
The kept job payload is the exact serialized job for reproduction: `python -m mbt.execute.job <path>` reruns it under a debugger.
These payloads are kept indefinitely for debugging; `mbt clean` ages out the ones older than 7 days so they do not accumulate in the temp dir.

**Fix:** if the job was genuinely making progress, raise `job_timeout_seconds` in `profiles.yml` (or remove it for no limit).
If it was hung, the payload plus the tail of the event log tells you where.
The same key works for the Spark compute adapter (`spark-submit` wording in the message).

## `Internal error: <ExceptionType>: <message>`

**Symptom (any command, exit 1; captured from a real occurrence - a non-UTF-8 spec file, a vector that has since been fixed to parse-error cleanly):**

```text
Internal error: UnicodeDecodeError: 'utf-8' codec can't decode byte 0xff in
position 22: invalid start byte
  hint: this is a bug in mbt; please report it with the command you ran. Set
MBT_DEBUG=1 to see the full traceback.
```

**Why:** this is the CLI's coordinator-side safety net.
Every expected failure (bad spec, missing file, gate breach) has its own friendly message and exit code; anything that reaches this catch-all is by definition a bug in mbt, not in your project.
The message is redacted (the error path is a serialization path too), so secrets never leak into it.

**Fix:** re-run the same command with `MBT_DEBUG=1` in the environment - the catch-all then re-raises, printing the full traceback - and file a bug report with the command and that traceback.
There is nothing to fix on the project side; if the message suggests a project problem that surfaced this way (as a raw exception rather than a friendly error), that mis-routing is part of the bug, so report it too.

## `Permission denied` reading `target/manifest.json` or `target/run_results.json`

**Symptom (exit 1, from whatever reads the control file - a `--state` compile, `mbt docs`, a CI script, or your own tooling; captured from a real reproduction where a root container wrote the file and a uid-1001 host process read it):**

```text
PermissionError: [Errno 13] Permission denied: '.../project/target/run_results.json'
```

**Why:** mbt versions up to v0.1.0 wrote the two control files through `tempfile.mkstemp`, which hardcodes mode `0600`, and the atomic `os.replace` carried that mode onto the destination.
The files therefore ended up readable only by the uid that wrote them, which breaks any handoff between users - a container writing into a bind-mounted workspace its host user then inspects, or one CI step producing artifacts for the next to read.
Later versions create the temp file with the process umask instead, so control files land `0644` like any ordinary write.

**Fix:** the next write of that file heals it, because the rename replaces the destination inode - re-run the command that produces it (`mbt compile`, `mbt build`) as the writing user.
To repair files you cannot regenerate, `chmod 644 target/manifest.json target/run_results.json` as their owner.
Nothing in a control file is secret, so widening the mode loses no protection.

## Informational event lines you may now see

These lines are normal observability output on stderr (or the `--log-format json` stream), not failures.
They were added so paths that used to run silently now report progress; an operator grepping the event log should not mistake them for problems.

| Line (verbatim rendering) | Emitted by |
|---|---|
| `Parsing project 'churn_demo'` / `Parsed 6 resources in 0.14s [OK]` | `mbt parse` start and finish (a parse error renders `[N error(s)]` in place of `[OK]`) |
| `state diff: 1 added, 0 removed, 2 modified` | `mbt state diff` (appends `; env digest CHANGED` when the environment digest moved) |
| `evaluate: 2 node(s) selected on target 'dev'` / `evaluate finished [success]: 2 ok, 0 failed, 0 skipped in 1.2s` | `mbt evaluate` run brackets, matching the other commands |
| `check schema: PASS` / `test test_row_count: PASS` | each built-in check and Python data test (the `FAIL` variant is a symptom - see its entry above) |
| `materialized 1000 rows: test=200, train=800` | a dataset build's per-split row counts (local, snowflake, and spark data adapters; the warehouse adapters prefix the node id) |
| `label join matched 480 of 520 spine rows (92.3%)` | a population-spine build's outcome coverage (F21): how many spine rows survived the inner label join, counted before filters/sampling/windows. Expected to be below 100% when the newest cohort's outcomes have not matured yet; enforce a floor with the `label_join_coverage` check |
| `scoring input materialized 340 rows to score` | a scoring-input build (an empty batch warns `scoring input materialized 0 rows; nothing to score` instead) |
| `tuning complete: 10 trial(s), 2 pruned, best pr_auc=0.8300` | a tuning search summary; per-trial `tuning trial 0: pr_auc=0.8300` lines are debug-level, shown only under `--verbose` or `--log-format json` |
| `feature_shift warn: tenure: psi=0.1800 in the shift warn band [0.15, 0.25]` | a shift in a monitor's optional `warn_threshold` band - elevated but below the fail bar, so the run stays green (exit 0); tune the thresholds or investigate the feature |
| `feature_shift most shifted: tenure=0.1800, age=0.0900, region=0.0400 (top 3 of 12)` | a per-run summary of the most-shifted features (ranked, whether or not any breached), so drift is visible before it crosses a threshold |
| `run 06e35b21ab994b83: already evaluated by a concurrent monitor run; skipping to avoid a double gate/alert` | benign: two overlapping `mbt monitor` runs raced, and the atomic ledger let only one record the evaluation (R2-11); the loser skips - "evaluated exactly once" held |

`--quiet` suppresses all of these; `--log-format json` emits them as structured event objects instead of the rendered text shown here.
