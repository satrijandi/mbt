# FEEDBACK v6: a black-box review of mbt

Review date: 2026-10-04, against `main` (`a5c2324`).
Scope: the product as a user meets it, and the test suite as a mutant meets it.
Findings closed in the five earlier cycles (`design-history/reviews/feedback-v1.md` through `-v5`) are not re-litigated.

Swept and closed 2026-10-05; the progress log at the bottom records every item.

## What this review is

The five earlier cycles read the code: practice (v1-v4), then shape (v5).
This one runs it.
Every finding below was reproduced through the real `mbt` CLI on a project created by `mbt init`, or through the real loader it names, or measured by mutating the source and running the fast suite.
The two exceptions are A-8 and D-1, which were found by reading the scaffold's workflows and the status page, and are marked as such by citing files rather than output.
Everywhere else, where a finding cites code, the code was found after the behaviour was seen.

## What this review is not

It is not a defect sweep of the whole tree, and it is not a criticism of any ADR.
Two findings sit next to deliberate decisions (`features.exclude` leniency in `8ab2543`, and ADR-21's idempotency claim); both say what the decision got right before saying where it stops.

## Method, and what was run

All on this machine (macOS, CPython 3.11 venv from `uv.lock`), in throwaway directories outside the repo.

| What | Result |
|---|---|
| Fast suite with coverage (`pytest -m "not e2e" --cov`) | 1853 passed, 51 skipped, 100.00% coverage, 4m48s. The 51 skips are all opt-in live tiers (Snowflake, showcase, k3d, make runbook). |
| ruff, ruff format, strict mypy over all 12 packages, `mkdocs build --strict`, yamllint | All clean, run on a clean clone of `a5c2324`. |
| GitHub Actions at `a5c2324` | CI, CodeQL, Live integration and Upstream resolution all green on their latest runs. |
| The README quickstart, verbatim | `init` -> `generate_sample_data` -> `build` -> `promote` -> `score` -> `monitor` -> `docs generate` all exit 0; the first build takes 6.7s wall-clock. |
| The released `v0.1.0` tag, installed into a fresh venv | See A-1. |
| About 30 adversarial probes | Spec typos, duplicate keys, degenerate data, overlapping windows, concurrent and interrupted runs, unusual paths, selectors, re-runs. |
| 20 hand-picked mutants over gates, verdicts, monitors and the bootstrap | See C-1. |

**Not run:** the JVM e2e tier and the opt-in live tiers.
CI runs both nightly and both were green on their latest run, so this review spent its time where CI does not look.

## What held up

The probes found more right than wrong, and the strengths are worth naming so a fix does not regress them.

- Spec validation is excellent: a typo in a field name, a hyperparameter name or a gate metric is caught at parse with the JSON path and a did-you-mean hint.
- A model that fails a gate is never registered, so `mbt promote --version N` on it fails with "has no version", and `promote` without `--version` takes the last passing one.
- A filter that leaves zero rows, and a label that is single-class after filtering, both fail the build with a hint that names the cause.
- Ctrl-C during training exits 130 within 0.7s, leaves no orphaned job process, and leaves no dangling RUNNING run in MLflow.
- `mbt score --anchor <fixed>` run twice writes one prediction run, exactly as ADR-21 says.
- A project path with spaces works end to end.

## Findings

| # | Finding | Kind | Size |
|---|---|---|---|
| A-1 | The only release cannot run, and every scaffolded project's CI installs it | Live defect | S |
| A-2 | Overlapping train and test windows put the same rows in both splits, silently | Live defect | S |
| A-3 | A typo in `features.exclude` trains on the column it was meant to guard | Live defect | XS |
| A-4 | Duplicate YAML keys are last-wins in every reviewed file, `promotions.yml` included | Live defect | S |
| A-5 | A selector that matches nothing exits 0, and the scheduled-scoring heartbeat says all is well | Live defect | S |
| A-6 | An apostrophe in the project path fails every dataset build | Live defect | XS |
| A-7 | The scaffold's scheduled scoring is not idempotent on re-run | Live defect | XS |
| A-8 | The scaffold's production workflows train and score on generated sample data | Live defect | S |
| B-1 | Nothing serializes two runs in one project, and the collision is misreported | Robustness | S |
| B-2 | `--log-format json` is not a JSON stream on a fresh project | Robustness | XS |
| B-3 | One invalid model cascades into errors about the resources that ref it | UX | XS |
| B-4 | Results tables truncate node ids wherever stdout is not a terminal | UX | XS |
| B-5 | A user's Ctrl-C is reported as a job error | UX | XS |
| C-1 | 100% line coverage, yet a mutant that lets a stability breach register survives the suite | Test strength | M |
| D-1 | The status page's package and test counts are stale | Docs | XS |

---

## A-1. The only release cannot run, and every scaffolded project's CI installs it

**Reproduce.**

```bash
git clone --branch v0.1.0 https://github.com/satrijandi/mbt v010
uv venv /tmp/v010 && VIRTUAL_ENV=/tmp/v010 uv pip install \
  ./v010/packages/{mbt-adapter-base,mbt-core,mbt-xgboost,mbt-mlflow}
/tmp/v010/bin/mbt --version
```

```
mbt 0.1.0
Traceback (most recent call last):
  ...
  File ".../mbt/cli/main.py", line 1054, in main
    except (click.exceptions.Exit, typer_click_exc.Exit) as exc:
AttributeError: module 'typer._click.exceptions' has no attribute 'Exit'
```

Every command does this, including ones that succeed: `mbt parse` prints "Parsed 3 nodes" and then exits 1.
The fresh resolve took typer 0.27.2, which moved `Exit`; `main` fixed this on 2026-08-29 (`a73dac1`), but no release followed.

With typer held below 0.27, `v0.1.0` gets further and then fails on the profile `mbt init` writes today:

```
Error: invalid Jinja in profiles.yml: 'env' is undefined
```

**Why it matters.** `mbt init` stamps `requirements.txt` with `mbt-* @ git+...@v{mbt.__version__}`, and `mbt/__init__.py:3` still says `0.1.0`.
So every project scaffolded from `main` pins its CI to a tag that is 95 commits and about 9,000 source lines behind the scaffold it ships with.
Every workflow's install step succeeds and every `mbt` step after it fails.
The two failures stack: unpinned, the tag crashes on typer; pinned, it cannot read the scaffold.

**Why nothing caught it.** `tests/test_cli_basics.py:212` asserts that the pin says `@v{mbt.__version__}`, which is true.
No check asks whether that tag contains the code that wrote the scaffold, and no check installs the pinned ref.
`tests/test_wheel_install.py` builds wheels from the working tree, which is the one thing a user's CI never installs.
The scaffold README (`_scaffold/README.md:50`) still says the install fails "until the `v0.1.0` tag exists"; it has existed since 2026-07-22, so the stated failure mode is not the one users hit.
`docs/troubleshooting.md:43` already documents the swap-in-a-SHA workaround, so the symptom has been seen before; the cause is that `__version__` does not move.

**Suggested fix.** Cut `v0.2.0` now; CHANGELOG's Unreleased section already holds 48 entries, including the typer fix.
Then make the mismatch impossible rather than rare: bump `__version__` to the next `.dev0` immediately after each release, and have `mbt init` refuse to stamp a tag for a `.dev` version, pinning the commit it was installed from instead (`direct_url.json`'s `vcs_info.commit_id` for a VCS install, `git rev-parse HEAD` for an editable checkout).
A test that the scaffolded ref is either a SHA or a tag equal to `__version__` with no `.dev` suffix would hold the line.

## A-2. Overlapping train and test windows put the same rows in both splits, silently

**Reproduce.** In the scaffold's `datasets/churn_training_set.yml`, move the train window's end past the test window's start:

```yaml
split:
  strategy: temporal
  train: "-180d:-20d"   # was -180d:-28d
  test: "-28d:now"
```

`mbt build` exits 0, the gate passes, and the model registers.
Joining the two materialized splits on `(user_id, snapshot_date)` finds **101 identical rows in both**: train runs to 2026-09-13, test starts at 2026-09-06.
`label_leakage_scan` passes, and no line on stderr mentions the overlap.

**Why it matters.** The test split is what every threshold gate and the champion gate's paired bootstrap judge.
Rows the model trained on inflate that number in the model's favour, which is the one direction a gate exists to prevent.
A two-character edit in a reviewed YAML file is enough, and a reviewer has to do window arithmetic in their head to see it.

**The asymmetry.** The same protection already exists one window later.
`compile/compiler.py:341` (`_check_out_of_time_follows_test`) rejects an `out_of_time` window that reaches back into test, with the reason written down: "An overlap would score rows the model was evaluated on as if they were new, and flatter every after-test number."
`docs/spec-reference.md:397` documents it, including the parse-time check for same-kind bounds and the compile-time check for mixed ones.
Train against test has neither.

**Suggested fix.** Apply the existing two-stage check to train and test: reject at parse when both bounds are relative or both absolute, and at compile once the anchor orders a mixed pair.
Account for `embargo`, which already widens the gap and should keep being allowed.
Add it as a rule in `parsing/rules.py`, so it runs at parse and compile like the others.

## A-3. A typo in `features.exclude` trains on the column it was meant to guard

**Reproduce.** In the scaffold's model, change `exclude: [user_id]` to `exclude: [user_idd]`.
`mbt parse` and `mbt build` both exit 0.
The run log records the consequence at DEBUG only:

```
DEBUG 6 feature(s): user_id, is_active, tenure_days, monthly_usage, support_tickets, plan_type
```

The README calls this list "explicit leakage guards".

**The decision it refines.** `8ab2543` (2026-09-30) made an unmatched `features.include` entry a hard error after a renamed feature silently fell out of a showcase model, and kept `exclude` lenient because "excluding a column that is not there is harmless".
That is right when the column is genuinely gone.
It is wrong when the entry is a misspelling of a column that is there, which is the failure that commit was written to stop, just on the side where it costs more.
`execute/handles.py:51-71` shows the asymmetry: `include` gets the unmatched check and a difflib hint, `exclude` gets a bare `fnmatchcase`.

**Suggested fix.** Keep the leniency for an entry with no near match, and raise for an entry that matches no column but has a close match among the columns (the same `difflib.get_close_matches` the include hint already uses).
At minimum, emit a WARN naming the entry; a WARN is visible in the PR check, a DEBUG line is not.

## A-4. Duplicate YAML keys are last-wins in every reviewed file, `promotions.yml` included

**Reproduce.** Two keys in the scaffold's model spec:

```yaml
hyperparameters:
  max_depth: 4
  max_depth: 9
```

`mbt parse` exits 0 and `target/manifest.json` records `max_depth: 9`.
The same holds in `promotions.yml`, the file whose merge triggers `promote.yml`:

```yaml
promotions:
  - model: churn_classifier
    version: "3"            # the version the reviewer approved
    to: production
    require_oot_check: true
    version: "5"
```

`mbt.promote.load_promotions_file` returns `version='5'`.

**Why it matters.** mbt's premise is that the reviewed YAML is the model, and the promotion file is the reviewed control over production.
A duplicate key makes the line a reviewer reads differ from the value that runs, with no diagnostic.
Long specs and merge resolutions produce duplicates by accident; a hostile PR can produce one on purpose.

**Where.** Every loader calls `yaml.safe_load` directly: `parsing/loader.py:28`, `config/project.py:75`, `config/profiles.py:160`, `promote.py:48`, `deps.py:38`, `cli/common.py:197`.

**Suggested fix.** One `SafeLoader` subclass whose mapping constructor raises on a repeated key, used by all six call sites, and a parse error that names the file, line and key.
A test that each loader rejects a duplicate keeps a seventh call site from appearing.

## A-5. A selector that matches nothing exits 0, and the scheduled-scoring heartbeat says all is well

**Reproduce.**

```
$ mbt build --select churn_clasifier
build: 0 node(s) selected on target 'dev'
build finished [success]: 0 ok, 0 failed, 0 skipped in 2.0s
$ echo $?
0
```

The same for `--select tag:nope`.
Parse offers did-you-mean everywhere; selection offers nothing.

**Why it matters.** The scaffold's `scheduled_score.yml` runs `mbt score --target prod --select tag:daily` and then pings a dead-man's-switch on success.
Renaming the scoring pipeline's tag from `daily` to `nightly` makes that step select zero nodes, exit 0, and ping the heartbeat.
Scoring stops, and the one mechanism the workflow has for noticing that scoring stopped reports that it is alive.

**Suggested fix.** Distinguish a selector that resolves to no resource from a selection that is legitimately empty.
A name, path or tag selector that matches nothing in the project is a hard error (exit 1) with a did-you-mean.
`state:modified` resolving to nothing is a legitimate no-op and stays exit 0.

## A-6. An apostrophe in the project path fails every dataset build

**Reproduce.** Run the quickstart under `~/O'Brien team/my models`.
`mbt build` fails on the first dataset:

```
ERROR dataset dataset.my_models.churn_training_set - dataset build failed in DuckDB: Parser Error: syntax error at or near ...
  hint: check the dataset's filters and split configuration against the relation's columns
```

The same project under `~/space team/my models` builds fine, and the hint points at the wrong thing.

**Where.** `adapters/local/data.py` already has `_sql_str` (line 60), which escapes quotes, and uses it for every read path (line 208).
The four writes interpolate the output path raw: `COPY (...) TO '{out}'` at lines 344, 386, 400 and 426.

**Suggested fix.** Route the four `TO` targets through `_sql_str`.
Add an apostrophe to the tmp path of one of the local-adapter tests, so the compliance suite runs every engine under a hostile path.

## A-7. The scaffold's scheduled scoring is not idempotent on re-run

**Reproduce.** `mbt score` twice on an unchanged batch with the same champion: two prediction runs land in `predictions/`, `mbt predictions ls` lists both (224 rows each), and `mbt monitor` later evaluates both.
With `--anchor 2026-10-03T06:00:00Z` on both calls, one run lands, as ADR-21 promises.

**Why.** ADR-21's run key hashes the resolved windows.
Without `--anchor`, every invocation compiles against "now", so a `-7d:now` window shifts by seconds and the key changes.
The ADR's claim ("re-running the same manifest ... overwrites its own run idempotently") is true; plain re-running is just not re-running the same manifest.

**Where it bites.** The showcase's Airflow DAG passes `--anchor` from the DAG run's params (`examples/showcase/deploy/dags/mbt_score_dag.py`), so it is idempotent.
The scaffold's `scheduled_score.yml` does not, so the GitHub "Re-run jobs" button duplicates a day's scores into whatever consumes `predictions/`.

**Suggested fix.** Have `scheduled_score.yml` and `scheduled_monitor.yml` derive `--anchor` from the scheduled time rather than the wall clock, the way the showcase does, and say why in a comment.

## A-8. The scaffold's production workflows train and score on generated sample data

All six scaffold workflows run `python scripts/generate_sample_data.py` right after `pip install`, including `prod_build.yml` and `scheduled_score.yml`.
The prod profile reads data from `{{ env('MBT_DATA_ROOT', '.') }}`, so a project that has not set `MBT_DATA_ROOT` trains its production model on freshly generated synthetic rows, registers it, and scores the synthetic batch, all green.
No workflow comment, scaffold README line or doc page says to remove that step once `sources.yml` points at real data.

**Suggested fix.** Keep the step in `pr_check.yml`, where sample data is the point, and drop it from the prod and scheduled workflows.
Make the prod target fail loudly when `MBT_DATA_ROOT` is unset, with `env('MBT_DATA_ROOT')` and no default.
The scaffold README's "Repo settings this project assumes" section is the natural place to say so.

---

## B-1. Nothing serializes two runs in one project, and the collision is misreported

Two `mbt build` processes started together in one project: one succeeds, the other fails with

```
dataset build failed in DuckDB: IO Error: Could not set lock on file ".../test.parquet": Conflicting lock is held ...
  hint: check the dataset's filters and split configuration against the relation's columns
```

They collided only because both compiled in the same second and chose the same dataset directory; a second later, both would have run and both written `target/manifest.json` and `target/run_results.json`, last writer wins.
GitHub-hosted workflows are not exposed, since each job gets a fresh checkout, and the scaffold's per-workflow concurrency groups show the risk is known there.
The exposure is everywhere a project directory is shared: cron or an Airflow worker on one host, and a developer building while a scheduled run is in flight.
The CLI has no lock of its own.

**Suggested fix.** An advisory lock on `target/` for the lifetime of any command that writes there, with an error that names the PID holding it.
Separately, the catch-all hint in `adapters/local/data.py:182-189` fires for every `duckdb.Error`; narrow it to the errors it describes.

## B-2. `--log-format json` is not a JSON stream on a fresh project

On a project with no `mlflow.db` yet, `mbt build --log-format json` writes 32 JSON lines and two that are not:

```
2026/10/04 00:02:32 INFO mlflow.store.db.utils: Creating initial MLflow database tables...
2026/10/04 00:02:32 INFO mlflow.store.db.utils: Updating database tables
```

They come from the alembic migration that `MlflowTracking.prepare()` triggers (`mbt_mlflow/adapter.py:208`).
A strict JSON-lines consumer fails on the first run of every new environment, which is when someone is most likely to be setting it up.

**Suggested fix.** Raise the `mlflow` and `alembic` loggers to WARNING around `prepare()`, or route them through the event bus as `LogMessage`s.

## B-3. One invalid model cascades into errors about the resources that ref it

A model that fails validation is dropped from `model_by_name` in `parsing/project_parser.py:_link`, so every scoring pipeline that refs it also reports

```
scoring/churn_scoring.yml  at /model: scoring references unknown model ref('churn_classifier')
```

The model exists; it is invalid.
Two errors for one mistake teach users to distrust the error list.

**Suggested fix.** Track the names of resources that failed validation, and either suppress dependent "unknown ref" errors or reword them to "references model 'x', which failed validation above".

## B-4. Results tables truncate node ids wherever stdout is not a terminal

With stdout redirected, Rich renders at 80 columns, so the results table shows `dataset.my_models.churn_tra…` and `scoring.my_models.churn_sco…`.
CI logs are where most of these tables are read, and they are never terminals.
`run_results.json` has the full data, so this is presentation only.

**Suggested fix.** When stdout is not a TTY, render with `width` from `COLUMNS` or a generous default, or print one plain line per node instead of a table.

## B-5. A user's Ctrl-C is reported as a job error

Interrupting a build during training prints

```
[2/2] ERROR model model.my_models.churn_classifier in 1.87s - training job exited with code -2 without writing a result (job payload kept at /var/folders/.../job.json)
```

The coordinator itself exits 130 correctly.
The job line reads like a crash, and keeps a reproduction payload in the system temp dir for a failure that needs no reproduction.

**Suggested fix.** Map a negative return code equal to `-SIGINT` to an INTERRUPTED status with no payload kept.

---

## C-1. 100% line coverage, yet a mutant that lets a stability breach register survives the suite

CI enforces 100% line coverage, which proves every line ran, not that any assertion would notice it being wrong.
To measure the second, 20 single-token mutants were applied one at a time to the code that decides whether a model ships or a batch alarms, and the fast suite was run against each.
Killed means at least one test failed.

| # | Mutant | Result | First test to fail |
|---|---|---|---|
| G1 | threshold >= becomes > | SURVIVED | - |
| G2 | tolerance sign (greater) | KILLED | `test_gates.py::test_tolerance_widens_thresholds_in_models_favor_only` |
| G3 | tolerance sign (lower-better) | SURVIVED | - |
| G4 | disparity >= becomes > | SURVIVED | - |
| G5 | disparity zero-best ratio | KILLED | `test_gates.py::test_disparity_gate_perfect_parity_and_all_zero` |
| G6 | bootstrap bound >= becomes > | SURVIVED | - |
| G7 | point delta >= becomes > | SURVIVED | - |
| G8 | delta ignores metric direction | KILLED | `test_gates.py::test_champion_gate_delta_and_direction` |
| G9 | bootstrap used despite confidence: null | KILLED | `test_gates.py::test_point_estimate_fallbacks` |
| G10 | OOT min_rows >= becomes > | SURVIVED | - |
| G11 | disparity needs 1 slice not 2 | KILLED | `test_gates.py::test_disparity_gate_needs_two_slices` |
| J1 | verdict ignores stability monitors | SURVIVED | - |
| J2 | after-test verdict ignores stability | KILLED | `test_report_gates_unit.py::test_the_after_test_verdict` |
| M1 | shift breach <= becomes < | SURVIVED | - |
| M2 | BH <= becomes < | SURVIVED | - |
| M3 | BH no-rejection bar alpha not alpha/m | KILLED | `test_monitor_evaluation.py::test_feature_shift_significance_corrects_for_multiple_comparisons` |
| M4 | warn band uses >= | SURVIVED | - |
| B1 | bootstrap two-sided-ish quantile | KILLED | `test_paired_bootstrap.py::test_clear_improvement_has_positive_lower_bound` |
| B2 | bootstrap keeps degenerate resamples | KILLED | `test_paired_bootstrap.py::test_single_class_resamples_are_skipped` |
| B3 | robust tuning bound wrong tail for lower-better | KILLED | `test_paired_bootstrap.py::test_bootstrap_metric_lower_bound_is_pessimistic_and_deterministic` |

Files: G = `quality/gates.py`, J = `quality/judgement.py`, M = `quality/monitors.py`, B = `mbt_adapter_base/metrics.py`.
Each mutant was one exact string replacement, run with `pytest -m "not e2e" -x` and then reverted; the table is the whole list.

**Reading it.** 10 of 20 mutants survived.
The kills show the suite is strong on what the code does in the common case.
Eight of the ten survivors are **boundaries**: `>=` against `>` at exactly the threshold, which no test sits on.
The other two are not, and they are the ones to fix first.

**J1 is the one that is not a boundary.**
`judge()` (`quality/judgement.py:131`) computes `passed = all_gates_passed(gates) and all_monitors_passed(stability.results)`, and `execute/runners.py:818` uses that `passed` to decide whether a model registers.
Deleting the stability half survives all 1853 tests: a model whose declared `evaluation.stability` monitors breach would register, and nothing would fail.
The after-test verdict's own copy of the same conjunction (J2) is pinned, which is likely why the gap was not visible.

**G3 is the other: direction.**
The tolerance on a threshold gate widens in the model's favour for higher-is-better metrics (pinned by G2's test), and flipping it for lower-is-better metrics survives.
Lower-is-better metrics are not an edge case: `logloss` is in the scaffold's own metric list, and every regression metric since ADR-24 is one.

**Suggested fix.**
First, a test that a breaching `evaluation.stability` monitor fails `judge()` and blocks registration (J1), and a lower-is-better tolerance test (G3).
Then the boundary cases: one test per comparison at exactly the threshold.
Then make this measurement repeatable: run `mutmut` (or `cosmic-ray`) nightly over `quality/`, `execute/seeds.py` and `mbt_adapter_base/metrics.py`, and report the surviving-mutant count the way coverage is reported, with a ratchet rather than a gate at first.
`docs/v0.1-status.md` markets the suite's rigor with the 100% figure; a mutation score next to it would be a stronger and truer claim.

---

## D-1. The status page's package and test counts are stale

`packages/` holds 12 packages and `release.yml` builds all of them (`uv build --all-packages`).
`docs/v0.1-status.md:26` says "11 publishable packages", and a comment in `release.yml` says "10 wheels do not upload atomically".
The same page's NFR-08 row (line 24) says "1699 tests (1611 fast, 88 e2e)"; the fast suite alone now runs 1853, and 103 are deselected as e2e.
CLAUDE.md asks that `v0.1-status.md` stay exactly true.

**Suggested fix.** Correct all three, and either derive the counts in a docs guard or stop stating exact counts that every commit moves.

---

## Recommendation

In order:

1. **A-1**, because it breaks every new user's CI today and its fix is a release.
2. **A-2**, **A-3** and the J1/G3 tests from **C-1**, because each lets a model pass a check it should fail, and all are small.
3. **A-4** and **A-5**, because each makes a reviewed or monitored control say something other than what happens.
4. **A-6** to **A-8**, then the B items as they are touched.
5. The rest of **C-1** as its own small project: the boundary tests, then a nightly mutation run.

A theme runs through A-2, A-3, A-5 and A-7: in each case mbt already had the right protection somewhere else (the out-of-time overlap check, the include check, did-you-mean at parse, the showcase's pinned anchor), and the gap is that it was applied to one side of a symmetric pair.
When fixing one, it is worth asking where its mirror image is.

## Progress log

One appended entry per completed item, carrying symptom, fix, verification and docs, per the shape the earlier cycles use.
This file moves to `design-history/reviews/feedback-v6.md` when the log closes.

### A-1 - v0.2.0 is cut, and main can no longer pin a release that does not contain it

**Release.** `8001129` ("Bump version to 0.2.0") is tagged `v0.2.0` and pushed; `release.yml` re-ran the full CI as its gate, went green, and published the GitHub release with all 24 wheels and sdists attached (the PyPI step stays gated off, as before).
The changelog section records the retraining impact the review asked every release to state: a full retrain, plus the spec edits ADR-29 requires of datasets that used `inputs:`.
The release commit also needed `UPDATE_GOLDEN=1`, because the golden manifest records `mbt_version`; CONTRIBUTING's procedure now says so.

**Making the mismatch impossible.** `main` is now `0.3.0.dev0`.
`mbt init` (`cli/scaffold.mbt_ref`) pins a release build to its own tag and a `.dev` build to the commit it was installed from (`direct_url.json`'s `vcs_info.commit_id`, or `git rev-parse HEAD` for an editable checkout); with neither it refuses, and `--mbt-ref` names the ref explicitly.
`scripts/bump_version.py` accepts `X.Y.Z.devN` and moves the packages' pins on each other (`mbt-adapter-base>=0.3.0.dev0,<0.4`), which it previously left at `<0.2` - a bump to 0.2.0 would have produced packages that could not install together.
`tests/test_cli_basics.py` holds the review's line: the scaffolded ref is a 40-hex SHA, or a tag equal to `v{__version__}` with no `.dev`.
The scaffold README's "until the tag exists" paragraph is gone, and the requirements headers describe both cases.

**Verification.** From the pushed tag, in a clean Python 3.11 venv: `mbt --version` prints `mbt 0.2.0` and exits 0 (v0.1.0 raised `AttributeError` there on the same typer 0.27.2); `mbt init` pins `@v0.2.0`; that project's own `requirements.txt` installs, and `mbt build` trains, gates and registers - the exact sequence every scaffolded project's CI runs.
On `0.3.0.dev0`, an editable checkout's `mbt init` pins its HEAD (`8001129...`), and a development wheel built from the tree refuses with the `--mbt-ref` hint, captured into the runbook.
`test_cli_scaffold_unit.py` covers release, VCS install, editable checkout, and the three no-commit shapes; `test_wheel_install.py` now asserts the refusal and the override against a real wheel.

**Docs.** `docs/installation.md`, `docs/quickstart.md` and `docs/gitops.md` point at v0.2.0; `docs/cli-reference.md` documents `--mbt-ref`; CONTRIBUTING's "Releasing" section requires the `.dev0` bump straight after a tag; the runbook's install entry says why the tag was missing, and a new entry covers the refusal.

### A-2 - a train or validation window that overlaps test is rejected

**Symptom.** Reproduced on a fresh `mbt init` project: `train: "-180d:-20d"` against `test: "-28d:now"` parsed, built and registered, with 101 rows in both splits.

**Fix.** The mirror of the after-test rule, with the same two stages.
`compile/windows.py` gained `anchor_independent` (the same-kind test the after-test rule already spelled inline) and `overlapping_splits` over `DISJOINT_SPLITS = (("train", "test"), ("validation", "test"))`.
The parse rule `dataset.windows` (`parsing/rules.py`) checks each pair whose bounds are all relative or all absolute, on the train window AFTER its embargo; `compiler._check_splits_disjoint` checks the anchored, embargoed bounds of every pair.
Validation against train is deliberately not a pair: a validation window inside the train range is a documented layout (`docs/spec-reference.md`'s own example).

**Verification.** Through the CLI: the relative overlap fails `mbt parse` (exit 1) at `/split/train`; `train: "2026-03-01:-20d"` passes parse and fails `mbt compile` with both resolved bounds; adding `embargo: "10d"` makes the same windows legal.
`test_out_of_time_split.py` pins both phases, the validation pair, and three legal layouts (touching bounds, embargo, validation inside train).
The first mutation-harness run caught a real instance in the suite itself: `report_unit_helpers.with_out_of_time` built `train: "-180d:-60d"` against `test: "-75d:-45d"`; train now ends where test starts.

**Docs.** `docs/spec-reference.md` (split section) and a runbook entry with both captured messages.

### A-3 - an exclude entry that misses a trained column by a typo fails the build

**Symptom.** `exclude: [user_idd]` trained on `user_id`, logged only at DEBUG.

**Fix.** `execute/handles._check_exclude_typos`: an exclude entry that matches no column AND closely matches (difflib, as the include hint) a column that would otherwise be a feature is an error.
`8ab2543`'s leniency is kept where it is right: an entry with no near match (a column that is genuinely gone), a near match that another entry already excludes, and globs.

**Verification.** Reproduced and fixed through `mbt build` (exit 1, the message names `'user_idd' (did you mean 'user_id'?)`); `test_handles_unit.py` pins the error and the three tolerated shapes.

**Docs.** New runbook entry; the include entry's "exclude stays lenient" line now says where that holds.

### A-4 - duplicate YAML keys are refused by every loader

**Symptom.** `max_depth` twice in a model and `version` twice in a `promotions.yml` entry both loaded last-wins, silently.

**Fix.** `mbt/yamlio.py`: one `SafeLoader` subclass whose mapping constructor raises `DuplicateKeyError` (a `ConstructorError`, so every existing `except yaml.YAMLError` reports it unchanged) naming the file, both lines and the key.
Merge keys keep their meaning: only keys written in the same mapping are compared.
All six call sites (`parsing/loader.py`, `config/project.py`, `config/profiles.py`, `promote.py`, `deps.py`, `cli/common.py` for `--vars`) go through it.

**Verification.** `mbt parse` on the duplicated `max_depth` exits 1 with both lines; `test_yaml_duplicate_keys.py` has one test per loader (the `promotions.yml` case is the review's exact file); `tests/test_yaml_loader_guard.py` fails if any `packages/*/src` file calls PyYAML's loaders directly, so a seventh call site cannot appear.

**Docs.** Runbook entry with the captured error.

### A-5 - a selector that names nothing is an error

**Symptom.** `mbt build --select churn_clasifier` selected 0 nodes and exited 0; `tag:nightly` likewise, which is how a renamed tag would keep the scheduled-scoring heartbeat alive.

**Fix.** `dag/selector._check_atoms_resolve`, run by `select_nodes` on `--select` and `--exclude`: a name, `tag:` or `resource_type:` atom matching no resource in the project raises `SelectorError` (exit 1) with a did-you-mean and the known names.
`state:` atoms and an intersection of atoms that each match something stay legitimate empties.
The scaffold's `scheduled_retrain_monthly.yml` shipped relying on "a tag no model carries is a clean no-op"; it is now gated on the repo variable `MBT_MONTHLY_RETRAIN == 'enabled'`, so it neither fails monthly nor pings a heartbeat for nothing.

**Verification.** All four review commands reproduced and now exit 1 with suggestions; `tag:churn,tag:daily` still exits 0 with an empty table.
`test_dag_unit.py` covers names, tags, resource types, a bad atom in a union, `--exclude`, and the legitimate empty; the hypothesis property `test_exclude_subtracts_exactly` now expects the error when its random DAG has no models; `test_cli_basics.py` pins the monthly gate.
Every `--select` in the showcase, fixtures and scaffold was checked to resolve.

**Docs.** `docs/cli-reference.md` (Selectors), `docs/gitops.md`, the scaffold README, and a runbook entry.

### A-6 - the local adapter escapes the paths it writes

**Symptom.** The quickstart under `O'Brien team/` failed every dataset build with a DuckDB parser error.

**Fix.** The four `COPY ... TO '{out}'` targets in `adapters/local/data.py` go through `_sql_str`, like every read path already did.

**Verification.** The whole quickstart (`init`, `build`, `promote`, `score`, `monitor`) passes under `O'Brien team/`.
`DataAdapterCompliance._build` now roots every data-adapter case under `HOSTILE_DIR_NAME = "O'Brien team ü"`, so the local, in-memory and Snowflake compliance suites all run there; with the fix reverted, 5 of the local adapter's 7 compliance cases fail.

### A-7 - scheduled scoring and monitoring pin their anchor

**Fix.** `scheduled_score.yml` and `scheduled_monitor.yml` gained a step that derives `--anchor` from the workflow run's `created_at` (which survives "Re-run jobs") truncated to the cron's slot (06:00 and 07:00 UTC), with an optional `anchor` input for a manual dispatch; the job grants `actions: read` to read it.
The value reaches `mbt` through `env:`, per the repo's no-expressions-in-run-scripts guard.

**Verification.** `tests/test_workflow_supply_chain.py` and `test_cli_basics.py` pass over the edited workflows; the idempotency itself is ADR-21's, already proven by the review (one prediction run with a fixed `--anchor`).

### A-8 - prod workflows no longer train on generated sample data

**Fix.** `generate_sample_data.py` now runs only in `pr_check.yml`.
The prod build, both retrains, scoring and monitoring export `MBT_DATA_ROOT` from the repo variable of that name (only when set, because GitHub renders an unset variable as an empty string), and the scaffold's prod target reads `{{ env('MBT_DATA_ROOT') }}` with no default.
That needed one core change: `profiles.yml` renders as a whole, so an unset default-less variable in `prod` used to break `dev` too.
`config/profiles._raise_for_unset_env` now fails only when the selected target (or the project block around it) reads the variable, judged on the unrendered mapping so `env('X') | int` is caught as well.

**Verification.** `test_profiles_unit.py`: dev loads with the variable unset, prod fails naming it, prod loads once it is set, and a variable in `target:` fails every target.

**Docs.** The scaffold README's "Repo settings this project assumes" names `MBT_DATA_ROOT` and `MBT_MONTHLY_RETRAIN`; `docs/spec-reference.md` states the target-scoped rule.

### B-1 - one writer per project `target/`

**Symptom.** Reproduced: two `mbt build`s started together; one died on "Could not set lock on file ... test.parquet" under the hint "check the dataset's filters".

**Fix.** `mbt/state/lock.py`: an `flock` on `target/.mbt.lock` for the lifetime of every command that writes `target/` (the three orchestrator entry points via `@holds_target_lock`, plus `compile`, `clean` and `docs generate`), reentrant within a process, released by the OS when the holder exits however it exits.
The refusal names the holder's pid, command and start time.
Separately, `adapters/local/data._duckdb_hint` keeps the spec hint only for parser, binder, catalog, conversion and type errors, gives an IO error its own hint, and gives anything else none.

**Verification.** Two concurrent builds: one succeeds, the other exits 1 with `pid 65957 (mbt build, started ...)`.
The first version printed "another process" because the loser read the record in the microseconds before the winner wrote it; the reader now retries briefly.
`test_target_lock.py` holds the lock in a real second process, and covers reentrancy and an unreadable record; `test_local_data_unit.py` pins the three hint classes against real DuckDB errors.

**Docs.** Runbook entry with the captured refusal.

### B-2 - `--log-format json` is a JSON stream on a fresh project, and when a command fails

**Symptom.** Reproduced: 32 JSON lines plus mlflow's "Creating initial MLflow database tables..." and "Updating database tables".

**Fix.** `mbt_mlflow.adapter._quiet_store_setup` raises `mlflow.store.db.utils` and `alembic` to WARNING when the client is first built, which covers every command that touches a store, not only `prepare()`.
Fixing it exposed the mirror image: a failing command ended the stream with a plain-text `Error:` block.
In JSON mode `fail()` (and the internal-error path in `main()`) now emits a `CommandFailed` event instead; text mode is unchanged byte for byte, so every captured runbook message still holds.

**Verification.** On a fresh project `build`, `score` and `monitor --log-format json` produce only JSON lines; a failing `promote` ends with one `CommandFailed` line.
`test_mlflow_edges.py` (fails without the fix) and two `test_cli_main_unit.py` cases pin both halves.

### B-3 - one broken resource is one error

**Fix.** `parsing/project_parser._broken_names` collects the names of resources that failed validation, plus the names still recoverable from a file the loader refused (a duplicate-key file is otherwise well-formed, so it is re-read leniently, for names only); the link phase does not report a ref to one of them as unknown.
A file that is not YAML at all names nothing, so a dangling ref keeps its error and gains a hint naming that file.

**Verification.** The duplicated `max_depth` and an invalid `task` now each produce exactly one error; `test_parse_error_cascade.py` covers both, the unreadable-file hint, and a real typo in a ref (still reported, with did-you-mean).

### B-4 - tables keep their full width off a terminal

**Fix.** `cli/common.print_table`, used by every CLI table: off a terminal and without `COLUMNS`, the table renders at its natural width (capped at 400) and prints through `out_console` unwrapped, because `Console.print` clamps any requested width to the console's own.

**Verification.** `mbt build > file` and `mbt ls > file` print whole node ids; `test_cli_main_unit.py::test_a_piped_table_keeps_its_node_ids_whole`.

### B-5 - Ctrl-C reports INTERRUPTED and keeps no payload

**Fix.** `JobResult.interrupted` (an additive field in `mbt_adapter_base.interchange`) and `interrupted_job_result(returncode)`, used by the local and Spark compute adapters: a job that died of SIGINT drops its payload dir and says so.
`ExecutionContext.run_job` raises `JobInterrupted` (a `KeyboardInterrupt`), and `run_with_lifecycle` reports the node as `INTERRUPTED` before re-raising, so the coordinator still exits 130.

**Verification.** SIGINT to the process group mid-training: `[2/2] INTERRUPTED model ... - interrupted (SIGINT)`, exit 130, no traceback, and the count of `mbt-job-*` dirs unchanged.
`test_local_compute_unit.py` and `test_job_interrupted.py` pin both halves.

**Docs.** `docs/cli-reference.md`'s exit-code table gained `130`.

### D-1 - the status page states counts that a guard derives

**Fix.** "11 publishable packages" is 12; release.yml's "10 wheels" and "all 10 packages" are 12; CONTRIBUTING's "22 wheels/sdists" was stale too (24); the changelog script's "Ten packages" is twelve.
The NFR-08 row no longer states test counts, which every commit moves.
`tests/test_docs_adr_count.py` gained two guards: every package or wheel count in those four files must equal `len(packages/*)` (twice that for wheels/sdists), and the status page may state no exact test count.

### C-1 - the twenty mutants are dead, and the measurement runs nightly

**The two that were not boundaries.**
J1: `test_execution.py::test_a_stability_breach_blocks_registration_with_every_gate_green` drives a real build whose gates all pass and whose `evaluation.stability` prediction-shift monitor breaches, and asserts exit 2, `gate_failed`, and nothing in the registry; `test_gate_boundaries.py` pins the same conjunction at `judge()`.
G3: a lower-is-better tolerance test (logloss widens upward, exactly to its width, no further).

**The boundaries.** `packages/mbt-core/tests/test_gate_boundaries.py` puts one test on each comparison, with values exact in binary floating point: threshold gates in both directions, tolerance in both directions, disparity `min_ratio`, the bootstrap bound and the point delta at `min_delta`, the after-test cell and the stability cell at exactly `min_rows`, a shift monitor at its threshold and just above, the warn band at `warn_threshold`, Benjamini-Hochberg at its rank bar, and ground-truth gates.

**Verification.** All 20 mutants from the table were re-applied one at a time (my reconstruction of each, from its description) against the gate, verdict, monitor, execution and bootstrap tests.
19 died on the first run; B1 ("two-sided-ish quantile", reconstructed as `(1 - c) / 2`) survived, because a clear improvement keeps a positive bound either way.
`test_paired_bootstrap.py::test_the_bound_is_the_one_sided_percentile_of_the_resampled_deltas` recomputes the resamples and pins the exact `1 - confidence` percentile in both metric directions; B1 now dies too, so all 20 are killed.

**Repeatable.** `scripts/mutation_score.py` runs mutmut 3.8 over `quality/*`, `execute/seeds.py` and `mbt_adapter_base/metrics.py`, each package mutated in a temp copy (mutmut assumes one package with `src/`, which a workspace member is and the repo root is not), and compares survivors with `scripts/mutation_baseline.json`: more fails, fewer asks for the baseline to come down.
`.github/workflows/mutation.yml` runs it nightly at 05:53 UTC, after the other heavy tiers, with the same open/comment/close tracking issue as `upstream.yml`.
mutmut lives in its own `mutation` dependency group, locked but outside the dev install and the floors job; adding it to the lock moved no existing version.
The first baseline (2026-10-05): mbt-core 2,223 killed and 464 survived of 2,688, mbt-adapter-base 518 killed and 86 survived of 604, about 69 minutes locally under load.
Most survivors mutate message wording, which is why the bar is a ratchet and not a target.
The harness's first attempt refused to run because the unmutated suite failed, which is how it surfaced the B-4 regression and A-2's overlapping test fixture before either reached a commit.
`tests/test_mutation_workflow.py` pins the ratchet, the targets, the baseline's coverage of them, and the workflow's wiring.

**Docs.** `docs/v0.1-status.md`'s NFR-08 row states the mutation score beside the coverage figure; CLAUDE.md describes the tier and the boundary-test pattern.

---

## Closing verification

The full CLAUDE.md battery, run on the sweep commit `3ef0d18` and again where later commits changed what it covers:

- `uv run pytest -q -m "not e2e" --cov` - **1,940 passed, 51 skipped, coverage 100.00%**; again on the release commit after regenerating the golden manifest's `mbt_version`.
- `uv run pytest -q -m e2e --timeout 1800` - **95 passed, 6 skipped** and 2 failures, both scaffold prod builds that relied on the `MBT_DATA_ROOT` default A-8 removed; the tests now set the variable as a project would, and re-ran green.
- ruff, ruff format, strict mypy over all 12 packages (155 files), pre-commit, `mkdocs build --strict`, yamllint and the advisory audit - clean.
- The mutation tier's first baseline: 2,741 of 3,292 mutants killed, 550 surviving.
- GitHub at `8001129`: CI, CodeQL and release all green; v0.2.0 published.

## What was deliberately not done

- **A selector check for Python data tests.** A test file's `selector:` that matches nothing still binds to nothing silently (`execute/runners.DatasetRunner._binds`); A-5 covered the command-line selectors the review named.
- **A warning when an editable checkout is dirty.** `mbt init` pins the checkout's HEAD, which does not include uncommitted changes; that is the honest commit to pin, and saying so on every `init` from a working tree would be noise.
- **An `interrupted` node status in `run_results.json`.** B-5's INTERRUPTED is an event-level report followed by the exit-130 unwind that already happened; no results file is written on that path, so a new persisted status would have no writer.
