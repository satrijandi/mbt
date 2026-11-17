# FEEDBACK v7: what could have been better in mbt

Review date: 2026-11-17, against `main` (`02edc83`) and the `v0.2.0` tag (`8001129`).
Scope: the product as a user meets it, the released artifact as a new user meets it, and the invariants `CLAUDE.md` says the code upholds.
Findings closed in `design-history/reviews/feedback-v1.md` through `-v6.md` are not re-litigated.

Swept and closed 2026-11-17; the progress log at the bottom records every item.
Two findings below (A-2, C-1) are reopenings of closed ones, and say so.

## Summary

The product is in good shape on `main`: the whole loop from `init` to `monitor` runs clean, the error messages are specific and carry hints, and the one-writer lock does what it says.
The problems are at the edges of what the suite and CI look at.

1. **The only release does not run.** `v0.2.0` fails the quickstart's first `mbt build` on any fresh install, because DuckDB 2.0.0 now resolves. The fix has been on `main` since 2026-11-12 and is not released. This is the same shape as v6 A-1, and nothing is watching for it (A-1, F-1).
2. **Two security claims are weaker than `SECURITY.md` says.** Profile rendering is not sandboxed, and the scaffold's `CODEOWNERS` leaves the files that decide what runs in CI and what ships unowned (B-1, B-2).
3. **Two "load-bearing decisions" in `CLAUDE.md` are only partly true.** "Declared, not probed" covers one of five adapter kinds, and `--log-format json` is a JSON stream only until the next upstream log line (C-1, A-2).

## Method, and what was run

All on this machine (macOS, CPython 3.11), in throwaway directories under the session scratchpad.
Nothing in the repo was modified except this file.

| What | Result |
|---|---|
| The quickstart verbatim on `main`, in the locked venv | `init`, `parse`, `build`, `run --manifest`, `promote`, `score`, `monitor --anchor +15d`, `predictions ls` all exit 0. First build 6.3s wall-clock. |
| `v0.2.0` (`git archive`, the four quickstart packages installed into a fresh venv, newest resolution) then `init` + `build` | **Fails.** See A-1. |
| `main` installed the same way (fresh venv, newest resolution: DuckDB 2.0.0, pandas 3.1.1, pyarrow 26.0.1) | `init --mbt-ref` + `build` exit 0, and two warm re-runs train in 2.8s and 3.5s. |
| About 12 adversarial CLI probes | Unknown selector, unknown target, malformed `--vars`, empty selector value, unknown resource, nonexistent version, `--threads 0` and `-3`, unknown flag, two concurrent builds in one `target/`. |
| Fast suite with `HOME` pointed at an empty directory (no `--cov`) | Passed, and wrote nothing into that `HOME`. |
| `grep` audits of the invariants in `CLAUDE.md` | `yaml.safe_load`, `fetchone()`, `project_dir / "target"` and `mbt.*` champion tag literals hold. `hasattr` and plain-Jinja do not. See C-1 and B-1. |

**Not run:** the e2e tier, the JVM and live tiers, the mutation tier, `--cov`, ruff, mypy, `mkdocs build --strict`, pre-commit, and any GitHub Actions state.
I installed the tag from a local `git archive` with `uv pip install ./path`, not from the `git+https://...@v0.2.0` form the docs give.
The code is identical, the transport is not.

## What held up

- Every error I provoked named the cause and carried a hint (`known resources: ...`, `available targets: dev, prod`, `use --target-path`).
- The one-writer lock (`state/lock.py`) refused the second of two concurrent builds with the holder's pid, start time and a way out, and the first finished normally.
- The fast suite does not depend on the developer's `~/.mbt` or git config, which I had expected to find it doing.
- The status page's package count (13), ADR count (35) and the coverage-floor claims match the tree.
- `mbt build` on a fresh scaffold is 4 to 6 seconds warm.
  The first run in a new venv is about 20 seconds (cold imports), which is normal and not a finding.

## Findings

| # | Finding | Kind | Size |
|---|---|---|---|
| A-1 | The only release cannot run the quickstart on a fresh install | Live defect | S |
| A-2 | `--log-format json` is not a JSON stream again (MLflow 3.18) | Live defect, reopens v6 B-2 | XS |
| A-3 | A built-in check that raises is reported as bad data, exit 2 | Live defect | XS |
| B-1 | `profiles.yml` is rendered by an unsandboxed Jinja environment | Security | XS |
| B-2 | The scaffold's `CODEOWNERS` leaves the files that run CI and ship models unowned | Security | XS |
| C-1 | "Capability is declared, not probed" holds for training adapters only | Invariant | M |
| D-1 | `mbt init` edits `~/.mbt/profiles.yml` and says nothing | UX | XS |
| D-2 | `--threads 0` and `--threads -3` are accepted and silently run serially | UX | XS |
| D-3 | The CLI cannot show what is in the registry | Product gap | M |
| E-1 | `SECURITY.md` supports only 0.1.x | Docs | XS |
| F-1 | Nothing tests the released tag after it ships | Process | S |
| G-1 | `CLAUDE.md` carries incident history an agent has to re-read every session | Docs | XS |

---

## A-1. The only release cannot run the quickstart on a fresh install

**Reproduce.**

```bash
git archive v0.2.0 | tar -x -C /tmp/rel/src
uv venv /tmp/rel/venv && VIRTUAL_ENV=/tmp/rel/venv uv pip install \
  /tmp/rel/src/packages/{mbt-adapter-base,mbt-core,mbt-xgboost,mbt-mlflow}
/tmp/rel/venv/bin/mbt --version          # mbt 0.2.0
cd /tmp/rel/p && /tmp/rel/venv/bin/mbt init relproj && cd relproj
python scripts/generate_sample_data.py
/tmp/rel/venv/bin/mbt build
```

**What happens.**
`duckdb` resolves to 2.0.0, because `mbt-core` declares `duckdb>=1.2` with no ceiling.
The dataset's built-in `no_future_columns` check raises `ResourceInUseException: connection has an unfinished result from execute()`.
The dataset node ends `test_failed`, the model is `skipped`, and the command exits **2**.

The scaffold makes it worse: `mbt init` pins the numerics stack to the versions of the environment that scaffolded the project, so the generated `requirements.txt` says `duckdb==2.0.0`.
A user who follows `docs/installation.md` and then `docs/quickstart.md` reaches this at step 5.
The scaffolded CI installs that same pin set and runs the same `mbt build`, so it should fail the same way; I did not run the generated workflows.

**On `main`.**
The code is fixed: `c367088` (2026-11-12) routes results through `mbt.utils.first_row`, and `test_duckdb_unfinished_results.py` enforces DuckDB 2.0's rule whatever DuckDB is installed.
I confirmed it end to end: `main` installed into a fresh venv that resolved DuckDB 2.0.0 builds green.
But `v0.2.0` is 18 commits behind `main` (2026-10-05 against 2026-11-17), the fix is not released, and there is no `0.2.1`.
The `0.3.0.dev0` bump (`2e2cb83`) stops `main` pinning new projects to the release, which helps people on `main` and does nothing for the people on the tag.

**Why it matters.**
This is the second release in a row that could not run for a new user (v6 A-1 was `v0.1.0`).
For a tool whose pitch is reproducibility, the artifact a stranger installs first is the one that should be verified last.

**Fix.**
1. Cut a release from `main` (the 18 commits include the fix and several advisory bumps).
2. Until then, add one line to `docs/installation.md` and the README: "`v0.2.0` needs `duckdb<2`".
3. Do not cap `duckdb` on `main`.
   It works on 2.0, and a cap would hide the next break the upstream tier is built to find.
4. See F-1 for the check that would have caught this.

## A-2. `--log-format json` is not a JSON stream again

This reopens v6 B-2, closed by quieting two store-setup loggers.

**Reproduce.**

```bash
CLAUDECODE=1 mbt build --log-format json > out.json 2> err.json    # any agent's variables will do
python3 -c 'import json,sys; [json.loads(l) for l in open("err.json") if l.strip()]'
```

**What happens.**
`err.json` has 33 JSON lines and one that is not:

```text
2026/11/17 18:34:52 INFO mlflow.agent.hint: Load the `mlflow-trace-instrumentation` skill at /.../site-packages/mlflow/assistant/skills/... before writing any tracing code; ... Set MLFLOW_DISABLE_AGENT_HINT=1 to silence this.
```

MLflow 3.18.0 prints this at `import mlflow`, to stderr, through its own handler, **only when it detects a coding agent driving the process**: any of a list of environment variables is set (`CLAUDECODE`, `CODEX_*`, `GEMINI_CLI`, ... in `mlflow/agent/hint.py`).
A human at a terminal and a plain CI runner never see it, which is why no test or CI run did.
`mbt_mlflow.adapter._quiet_store_setup` runs after the import and only touches `mlflow.store.db.utils` and `alembic`, so it cannot reach it.
With `MLFLOW_DISABLE_AGENT_HINT=1` the line count is 0.

The line is prose addressed to a coding agent, and it lands in whatever consumes mbt's stderr, which for the callers it fires for is the agent that parses `--log-format json`.
I did not act on it.

**Why the suite missed it.**
`test_a_fresh_store_does_not_print_its_schema_setup` asserts that the string `"database tables"` is absent.
It pins the symptom v6 found, not the invariant ("every stderr line under `--log-format json` parses").
`mlflow>=3.17.2` has no ceiling, so each MLflow release is a fresh chance to add a line.

**Fix.**
1. Set `MLFLOW_DISABLE_AGENT_HINT` (via `os.environ.setdefault`) before `mbt-mlflow` imports `mlflow`, in the plugin module and in the job subprocess environment.
2. Add the invariant as an e2e test against the real MLflow: run a build with `--log-format json` and `json.loads` every stderr line.
   That test would also have caught v6 B-2 without knowing its wording.

## A-3. A built-in check that raises is reported as bad data, exit 2

`quality/checks.py:126-129`:

```python
try:
    result = runner(spec, handle, resolved_windows, params, resource, sources)
except Exception as exc:
    result = TestResult(name=name, passed=False, message=f"check raised: {exc!r}")
```

An exception inside mbt's own check becomes a failed check, and a failed check is exit 2.
`CLAUDE.md` is explicit that 1 is a hard error and 2 is "gates/checks/tests said no", and that tests must not conflate them.
In A-1 this is what made a broken install look like a data problem: a scheduler that routes exit 2 to the data team would have paged the wrong people.

User-written Python data tests can keep "raised means failed" (`python_tests.py:113`), because an assertion is how those tests fail.
The built-in checks are mbt's code, so an exception there is mbt's or the environment's.

**Fix.**
Let exceptions from built-in checks propagate as a hard error (exit 1) naming the check, and keep the catch for user tests.
Add a test that a built-in check which raises exits 1.

## B-1. `profiles.yml` is rendered by an unsandboxed Jinja environment

`SECURITY.md` defines as a vulnerability "any way for a spec, profile, or hook to execute code outside the documented seams (Python hooks and adapters are code by design; YAML and Jinja specs are not)".

Spec rendering honours that: `mbt/jinja/environment.py:43` is a `SandboxedEnvironment`.
Two other sites build a plain `jinja2.Environment`:

- `mbt/config/profiles.py:141`, which renders `profiles.yml`.
- `mbt/execute/job.py:84`, which re-renders the adapter config inside the job.

**Reproduce.**
In a scaffolded project, add a harmless probe under the `dev` target's `vars:` in `profiles.yml`:

```yaml
probe: "{{ lipsum.__globals__['os'].getpid() }}"
```

`mbt compile` exits 0 and `target/manifest.json` contains `"probe": "65281"`, the pid of the mbt process.
The same expression through `_NativeSandbox` raises `SecurityError: access to attribute '__globals__' of 'function' object is unsafe`.

**Why it matters.**
`profiles.yml` is committed by design (CI has no `~/.mbt`), and the scaffold's `pr_check.yml` runs `mbt parse`, `compile` and `build` on every pull request.
So a PR can carry a profile that runs code in the CI job, from a file `SECURITY.md` says is not code.
Anyone who can already edit `hooks.py` could do this anyway, which is why this is XS and not a headline, but the stated boundary is wrong, and B-2 shows the review boundary does not cover it either.

**Fix.**
One factory in `mbt/jinja/` that returns the sandboxed environment, used at all three sites.
Add a guard test that fails on any `jinja2.Environment(` in `packages/*/src`, in the style of `test_target_layout_unit.py`.

## B-2. The scaffold's `CODEOWNERS` leaves the files that run CI and ship models unowned

The scaffold's `CODEOWNERS` is three lines:

```text
/models/    @your-org/ds-team
/datasets/  @your-org/ds-team
/promotions.yml @your-org/mlops-team
```

Not owned by anyone: `.github/workflows/` (including `promote.yml`, the thing that moves a model to production), `profiles.yml` (see B-1), `scoring/`, `sources.yml`, `macros/`, `scripts/`, `mbt_project.yml`, `packages.yml` and `requirements.*`.

A PR that touches only those paths needs no review from either team, even with branch protection on.
`docs/tutorial.md:23` says the scaffold "encodes this split", which is true of what is listed and silent about what is not.
The promotion gate is `promotions.yml`, but the workflow that executes it and the profile that says where it runs are editable around it.

**Fix.**
Own `.github/` and `profiles.yml` by MLOps, and `scoring/`, `sources.yml` and `macros/` by whichever team the project says owns them.
Mention the unowned remainder in each forge's `repo_settings.md`, and extend `tests/test_cli_basics.py` to assert that every top-level path the scaffold creates is either matched by `CODEOWNERS` or listed as deliberately unowned.

## C-1. "Capability is declared, not probed" holds for training adapters only

`CLAUDE.md`: "Never add a `hasattr` probe for an optional adapter method; add a `Capability` and let `capabilities_of` answer."
`capabilities.py` says the 16 `hasattr` probes in `execute/` were replaced.

What remains in `mbt-core/src/mbt/execute/`, found by `grep -n "hasattr("`:

| Site | Probes | Silent consequence when the method is missing or renamed |
|---|---|---|
| `runners.py:175` | `tracking.prepare` | no pre-warm of the tracking store before parallel jobs |
| `runners.py:283` | `compute.terminate` | `--fail-fast` returns without killing in-flight jobs |
| `runners.py:943,945` | `tracking.log_directory` / `log_document` | the run log is not uploaded to the tracker, with no warning (the `if`/`elif` falls through) |
| `job.py:497` | `tracking.log_document` | the inference config and hooks source are not logged |
| `job.py:909` | `tracking.log_trial` | tuning trials are not logged |
| `training_report.py:524,526` | `tracking.log_directory` / `log_document` | the training report is not uploaded, with no warning |
| `training_report.py:545` | `tracking.log_directory` | the model and dataset config JSON are not attached |
| `oot_check.py:273` | `registry.set_version_tags` | the after-test verdict tags are not written (a warning is emitted) |
| `orchestrator.py:474` | `data_adapter.build_scoring_input` and `open_predictions` | the scoring gate itself |

`test_inference_config.py:378` states the pattern as intended ("`log_document` is probed with `hasattr`, like `prepare` and `log_trial`").
So the rule is written down, a test enshrines the opposite, and no guard exists.
The rest of `capabilities.py` is right about the failure mode: a renamed method silently turns a feature off, and mypy cannot see it.

Two stale references to it: issue #1's body cites `orchestrator.py:421` as the `hasattr` gate (it is now `:474`), and the `Supports*` protocols in `protocols.py:236-297` that B-1 called documentation typed as code are still there.

**Fix.**
Do for tracking, registry, compute and data adapters what v5 B-1 did for training adapters: a `Capability` per optional method, `capabilities()` on each base class, the compliance suite asserting declaration and implementation agree in both directions, and `capabilities_of` at the eleven sites.
Then add the guard: a test that fails on `hasattr(` anywhere under `mbt/execute/` and `mbt/quality/`, with an explicit allow-list for the one non-adapter use there (`isoformat` in `job.py`).
If the decision is instead that these are fine, change `CLAUDE.md` to say "training adapters" so the next reader is not misled.

## D-1. `mbt init` edits `~/.mbt/profiles.yml` and says nothing

**Reproduce.**

```bash
mkdir -p /tmp/h && HOME=/tmp/h mbt init brandnew
ls /tmp/h/.mbt            # profiles.yml, 2392 bytes
```

The output is `Created ...` and `Next steps:`.
The code (`cli/scaffold.py:218-233`) creates the file when absent, appends the project's block when present and the project is new, and leaves it alone when the project name already exists.
All three paths are silent.
That is a write outside the project directory, to a file that can hold other projects' targets, and the user is not told.

**Fix.**
Print one line for each outcome (`installed profile 'x' into ~/.mbt/profiles.yml`, `left existing profile 'x' in ~/.mbt/profiles.yml`), and add `--no-home-profiles` for people who keep profiles elsewhere.

## D-2. `--threads 0` and `--threads -3` are accepted

`profiles.yml`'s `threads` is `Field(default=1, ge=1)` (`config/profiles.py:41`).
The command-line option (`cli/main.py:146`) is a bare `int | None`, and `execute/scheduler.py:55` quietly applies `max(1, threads)`.
So the same setting is rejected in one place and silently coerced in the other, and `mbt build --threads 0` reports success.

**Fix.**
`typer.Option(min=1)` on `ThreadsOpt`.

## D-3. The CLI cannot show what is in the registry

`mbt promote` and `mbt rollback` act on registry versions, and nothing in the CLI reads them.
`mbt ls` lists project resources.
`mbt predictions ls` lists prediction runs.
`promote --version 999` answers "list versions in your registry UI", and the rollback runbook's "Verify" step (`docs/troubleshooting.md`, "Rolling back a bad champion") has no command to verify with.

For a tool whose second half is gated promotion, "what is in production, which version, and what gates did it record" should be one read-only command (`mbt versions --model X`, or `mbt ls --registry`).
The data is already there: `promote` and `rollback` read recorded gate results to decide whether to refuse.

## E-1. `SECURITY.md` supports only 0.1.x

The supported-versions table is `0.1.x | yes`.
The only release is 0.2.0 and `main` is 0.3.0.dev0, so as written the current release is unsupported and the supported one is the release that could not run (v6 A-1).

**Fix.**
`0.2.x` yes, `0.1.x` no.
`tests/test_docs_adr_count.py` already guards the package counts in four files, so a guard that the table names the latest tag's minor series is the same kind of check.

## F-1. Nothing tests the released tag after it ships

`release.yml` runs `ci.yml` at tag time, so a tag on a red commit does not publish.
`upstream.yml` re-resolves `main` against the newest dependencies every night.
Neither installs the **tag** and runs the documented path, so a release that was green on the day it was cut goes stale as its open-ended dependency ranges move, and nothing notices until a user does.
That is exactly A-1: `v0.2.0` was green on 2026-10-05, DuckDB 2.0 arrived later, and the tracking-issue machinery that `upstream.yml` has was looking at the wrong ref.

**Fix.**
A nightly `released.yml` that installs the newest `v*` tag the way `docs/installation.md` says, scaffolds a project, and runs the quickstart through `monitor`.
Use the same open/comment/close tracking issue as the other tiers.
It is the only check that sees what a new user sees.
The existing `tests/test_wheel_install.py` has most of the steps.

## G-1. `CLAUDE.md` carries incident history an agent has to re-read every session

`CLAUDE.md` is 16.7 KB.
Its bullets run to 630 characters and hold, besides the rule, the incident that produced it ("Seen twice: a fast-suite run reporting 98.8% and an e2e run reporting 20.2%"; "it is what the job did until 2026-08-27, which is why `duckdb>=1.0` ... survived").
The history is valuable, but it belongs in the ADRs and `design-history/reviews/`, which `CLAUDE.md` already points at, and the rule plus a pointer is what an agent needs in context on every run.
This is taste, not a defect, and the lowest priority here.

---

## Smaller notes

- `execute/job.py` (1651 lines) and `execute/runners.py` (1530) are the two largest modules, and `ModelRunner` alone is 583 lines.
  By a rough branch count over every function, the numbers are mild (the worst scores 29, in `scheduler.execute_plan`), so this is a navigation cost and not a correctness one.
  `_run_train` (161 lines), `_run_score` (154) and `_run_tuning` (139) are the next candidates if the phase seam from ADR-35 is carried inward.
- `mbt build` with nothing changed retrains and registers a new version every time unless `--state` is passed.
  That is documented and intended, but the quickstart's "the reproduced model registers as `v2`" is only true if nothing else registered in between, which is easy to break by re-running a step.
- The fast suite and the e2e tier are green by the repo's own account, and I did not re-measure coverage or mutation score.

## What was deliberately not checked

The Snowflake and Spark live paths, Airflow, the showcase, the mutation baseline, and anything that needs credentials or a JVM.
Issue #1 (the native prediction store) remains the largest open gap in the product story and is already tracked.

---

## Progress log

Every item below was worked on 2026-11-17, against the tree this review was written from.
Each entry gives the symptom, the fix, how it was verified, and the docs that moved.
Where a fix was proved to bite, it says so: the new test was run once with the fix reverted and failed.

### A-1 - the only release cannot run the quickstart

**Interim.**
Before the release, `docs/installation.md` carried a warning ("v0.2.0 needs `duckdb<2`") and `docs/troubleshooting.md` got an entry whose symptom text was captured from the real failure (`check no_future_columns: FAIL - check raised: ResourceInUseException(...)`).
The workaround was verified before it was written down: v0.2.0 installed with `duckdb<2` builds the scaffolded project green (`mbt build` exit 0, 24 s cold).
Nothing caps `duckdb` on `main`, as the finding asked.

**The fix.**
v0.3.0 is cut from this tree: the 18 commits since v0.2.0 (the DuckDB fix and several advisory bumps among them) plus this sweep, with `main` carrying the next development version after it.
The warning is gone from `docs/installation.md`, which pins `@v0.3.0`; the troubleshooting entry stays and now says "upgrade to v0.3.0", because anyone on v0.2.0 still meets it.
The changelog records the release's retraining impact as none: no spec model changed since v0.2.0, and the golden manifest differs from v0.2.0's only in its contract and version strings.
It also records the one behaviour change a scheduler could notice, A-3's exit code.
`released.yml` was written to go red on v0.2.0 and to close its own tracking issue the night v0.3.0 installs and runs; if v0.3.0 is itself broken on a fresh install, that is where it will say so.

### A-2 - `--log-format json` is not a JSON stream again

**Fix.** `mbt_mlflow.adapter` does `os.environ.setdefault("MLFLOW_DISABLE_AGENT_HINT", "1")` at import, before any method imports `mlflow`, so the coordinator and every job subprocess that loads the plugin are covered, and a value the user set wins.
**Verification.**
`test_mlflow_edges.py::test_importing_mlflow_through_the_plugin_prints_nothing_to_an_agent` runs a child process with `CLAUDECODE=1` and asserts the hint is absent and a user's `=0` survives.
`test_cli_basics.py::test_the_json_event_stream_is_json_on_every_line` is the invariant itself, end to end: a real scaffolded `mbt build --log-format json` in an agent-looking environment, every stderr line through `json.loads`.
Both fail with the `setdefault` removed (the e2e one with `1 non-JSON line(s) on stderr`).
**Correction to the finding.** MLflow only prints the line when it detects a coding agent, so the finding now says so; no human or plain CI run was ever affected.

### A-3 - a built-in check that raises is reported as bad data

**Fix.** `quality/checks.py` now separates a verdict from a crash.
A check's own failure (a returned `TestResult`, or the new `CheckFailure` raised from a helper) is exit 2.
Anything else raises `CheckError` (new, in `exceptions.py`, an `MbtError`, exit 1) naming the check, with the original exception as `__cause__`; an `MbtError` a check raised on purpose keeps its own message.
Two boundary cases the old catch-all had been absorbing, found by running the suite after the change:
a column the spec names that the data lacks (`not_null`, `unique`, `accepted_values`) is now an explicit failed check with `column(s) not in the dataset: split.column`, and `label_leakage_scan` on a train split with no label column passes with "nothing to scan", as `class_balance_report` already did.
**Verification.** `test_checks.py` gained three tests (a crash is `CheckError` with exit code 1, an on-purpose `MbtError` passes through untouched, an absent column is a verdict); `test_execution.py::test_dataset_check_failure_fails_downstream` still pins exit 2 for the verdict case.
**Docs.** `design-history/lessons.md`; the troubleshooting entry for A-1 says what a crash looks like now.

### B-1 - `profiles.yml` is rendered by an unsandboxed Jinja environment

**Fix.** `mbt/jinja/sandbox.py` is the one place an environment is built (`sandboxed_environment()` for text, `native_sandboxed_environment()` for specs); `config/profiles.py` and `execute/job.py` use it, and `jinja/environment.py` lost its private copy of the native class.
**Verification.**
`test_jinja_sandbox_guard.py` renders the finding's probe, `{{ lipsum.__globals__['os'].getpid() }}`, through `load_profiles` (now `ConfigError: invalid Jinja in profiles.yml ... unsafe`) and through the job's `_render_adapter_ref` (`SecurityError`), and fails on any direct `jinja2.Environment(`, `jinja2.Template(` or `NativeEnvironment(` in `packages/*/src`.
The guard scanned its own factory's docstring on the first run, which is how I knew it reads the right files.
**Docs.** `CLAUDE.md` states the rule.

### B-2 - the scaffold's `CODEOWNERS` leaves CI and shipping unowned

**Fix.** The scaffold's `CODEOWNERS` now owns every top-level path: the DS team has `/models/`, `/datasets/`, `/sources.yml`, `/macros/`, `/tests/`; MLOps has `/promotions.yml`, `/scoring/`, `/profiles.yml`, `/mbt_project.yml`, `/packages.yml`, `/requirements.*`, `/scripts/`, `/.github/`, `/.woodpecker/`, `/renovate.json`, `/.pre-commit-config.yaml` and itself.
Only `README.md` and `.gitignore` are left unowned, each with a reason in the test.
**Verification.** `test_cli_basics.py::test_every_top_level_scaffold_path_is_owned_or_deliberately_not`, parametrized over both forges, scaffolds a project and fails on any top-level entry no pattern covers, so adding one forces the question.
**Docs.** The scaffold README line, `docs/tutorial.md` (the split, and why), `docs/mlops-alignment.md`.
I did not run GitHub's CODEOWNERS enforcement; the file's semantics are GitHub's and Gitea's, and the test checks coverage, not enforcement.

### C-1 - capability is declared for training adapters only

**Fix.**
`mbt_adapter_base.capabilities` now covers five kinds: `AdapterKind`, eleven new `Capability` members (`PREPARE`, `LOG_TRIAL`, `LOG_DOCUMENT`, `LOG_DIRECTORY`, `SET_VERSION_TAGS`, `LIST_VERSIONS`, `TERMINATE`, `BATCH_SCORING`, `SOURCE_CHECKS`; training kept its eight), `kind_of`, a `DeclaresCapabilities` mixin, and `declaration_problems`, which the compliance suites call.
All 12 `hasattr` probes in `execute/` and the `getattr` probe in `quality/checks.py` became `supports(adapter, Capability.X)`.
Every in-repo adapter declares: MLflow tracking and registry, the local, Spark and Snowflake adapters, the in-memory data adapter, the fakes in `mbt-testing`, and both compute adapters.
An adapter with no `capabilities()` (a third party built before v5) is still served by the method-presence rule, which lives in `capabilities_of` and nowhere in core.
**Verification.**
`test_no_hasattr_probe_guard.py` fails on any `hasattr(` in `mbt-core/src` outside two named exceptions.
`test_capability_kinds.py` covers kinds, multi-method capabilities, the fallback, and each way `declaration_problems` can disagree.
`TrackingAdapterCompliance` and `ComputeAdapterCompliance` are new, `RegistryAdapterCompliance` and `DataAdapterCompliance` gained the declaration test, and the training one now shares the checker; every in-repo adapter subclasses the one for its seam (the Spark compute one is in the e2e tier, like the rest of Spark).
The test that named the probe as intended (`test_inference_config.py`) now describes the capability, and `_TaglessRegistry` in `test_oot_check_flow.py` declares what it lacks instead of hiding a method.
**Docs.** `protocols.py` docstrings, `docs/adapter-authoring.md`, `docs/architecture.md`, `CLAUDE.md`.
**Left alone on purpose.** The `Supports*` protocols stay: `docs/adapter-authoring.md` tells adapter authors to pin each method's signature with them under strict mypy, which is a use and not a leftover.
Issue #1's body still cites `orchestrator.py:421` as the `hasattr` gate; that is a GitHub write on a public issue, so I left it for the maintainer, and its premise is now simply out of date.

### D-1 - `mbt init` edits `~/.mbt/profiles.yml` and says nothing

**Fix.** `init` reports which of `created`, `appended` or `left unchanged` it did, with the path, through the event bus (so `--quiet` and `--log-format json` apply), and `--no-home-profiles` skips the file and says it did.
**Verification.** Four tests in `test_cli_main_unit.py`, one per outcome, each in a sandboxed `HOME`.
**Docs.** `docs/cli-reference.md`.

### D-2 - `--threads 0` and `-3` are accepted

**Fix.** `typer.Option("--threads", min=1)`, as `profiles.yml`'s `threads` already was.
**Verification.** `test_cli_main_unit.py` refuses `0` and `-3` on all five execution commands; through the real CLI: `Invalid value for '--threads': 0 is not in the range x>=1`, exit 1.
**Docs.** `docs/cli-reference.md`.

### D-3 - the CLI cannot show what is in the registry

**Fix.**
`mbt registry ls [--model NAME] [--output table|json]`: every version, newest first, with its stage, whether gates recorded a pass, the after-test verdict, and a short artifact digest, decoded through the champion record (`mbt/registry_view.py`).
Without `--model` it lists every model the project declares under the name each registers as.
It rests on a new optional registry capability, `list_versions(name)` (`Capability.LIST_VERSIONS`), implemented by the MLflow and fake registries and pinned by a new `RegistryAdapterCompliance` test; a registry without it is refused with an error that says so.
MLflow's `search_model_versions` carries no aliases, so a first version of this showed every stage as `-`; it now reads the alias map from the registered model, which the compliance test caught.
`promote --version 999` now answers `run 'mbt registry ls --model NAME'` instead of "your registry UI".
**Verification.** Run against a real MLflow registry holding eight versions: v4 showed `production`, v8 `staging`, as I had left them.
`test_registry_view.py` covers ordering, the three gate states, the after-test words, the empty registry, the all-models form and the refusal.
**Docs.** `docs/cli-reference.md`, the quickstart's Inspect step, and the rollback runbook's Verify step, which finally has a command.

### E-1 - `SECURITY.md` supports only 0.1.x

**Fix.** The table says `0.2.x | yes`, `0.1.x | no`.
**Verification.** `test_docs_adr_count.py::test_security_policy_supports_the_newest_released_series_only` reads the newest `## vX.Y.Z` heading in `CHANGELOG.md`, so the next release that lands without this table moving fails the suite.

### F-1 - nothing tests the released tag after it ships

**Fix.** `.github/workflows/released.yml`: nightly at 06:37 UTC (and on demand, with an optional tag), it checks nothing out, installs the newest published release with the four quickstart packages by git reference into a fresh venv with the cache off, and walks `docs/quickstart.md` through `init`, `parse`, `build`, `run --manifest`, `promote`, `score`, `monitor`, `monitor --anchor +15d` and `predictions ls`, each of which must exit 0.
Failure opens or comments on one tracking issue; recovery closes it.
**Verification.**
`test_released_workflow.py` pins the invariants (no checkout, git-reference install with no lock, cache off, the quickstart in documented order, an empty `HOME`) and runs the notice script against a stubbed `gh` for the green, red, repeated-red and recovery cases.
The existing supply-chain and notice guards pass over it, and it passes yamllint and PyYAML.
I also ran its install line against the real tag from here: it resolves `duckdb 2.0.0`, so on its first night this tier will go red on `v0.2.0` and file the issue, which is the right answer and the proof it would have caught A-1.
I could not run the workflow on GitHub itself.
**Docs.** The README badge, `docs/v0.1-status.md` (NFR-08), `CLAUDE.md`.

### G-1 - `CLAUDE.md` carries incident history

**Fix.** Six incident narratives moved to `design-history/lessons.md` (the coverage-file clobber, mutation timeouts, why `upstream.yml` ignores the lock, `uv lock --upgrade-package`, the no-op floors job, and the two new ones from this review); `CLAUDE.md` keeps each rule and one line of reason, with a pointer.
The file did not get shorter overall, because this review added four rules it needed (declared capabilities for all kinds, the Jinja sandbox, a crashing check, the released-tag tier), but the history an agent re-reads every session is out of it.

### Smaller notes

- **`execute/job.py` and `execute/runners.py` are split** (done in a second pass, after the first pass left it open).
  `runners.py` (1,531 lines) is now the package `execute/runners/`: `context.py` (`ExecutionContext`, `BuildContext`, the node lifecycle), `dataset.py`, `model.py` (`ModelRunner`, `ModelTestRunner`), `scoring.py`, and `champion_config.py` (a champion's inference config, which the model and scoring runners share), with `__init__.py` re-exporting the public names, so every `from mbt.execute.runners import ...` in the tree is unchanged.
  `job.py` (1,655 lines) is the entrypoint and dispatch, 203 lines, with the steps as siblings: `job_setup`, `job_carve`, `job_metrics`, `job_tuning`, `job_backtest`, `job_train`, `job_score`.
  The largest new module is `runners/model.py` at 678 lines, the largest function is untouched.
  The code was moved by line range with a script, not retyped, and ruff then removed unused imports and named the missing ones.
  I checked the result by parsing the old files and the new tree and comparing every definition's AST: 52 moved definitions, 0 differ (allowing the two renames below), and none of the old definitions is missing.
  Two names that now cross a module boundary lost their underscore (`source_access`, `ChampionConfig`); the job modules keep theirs, as one subsystem.
  What the split cost: seven test files repoint an import, and the three tests that patched `mbt.execute.job.get_registry` now patch the module that calls it (`job_setup`, `job_tuning`, `job_score`).
  The mypy `no-untyped-call` relaxation for `mbt.execute.job` moved to `mbt.execute.job_setup`, the only new module that fails under pyarrow 24 (checked in a throwaway environment), so the status page's "nine modules" is still true.
- The quickstart no longer promises "registers as `v2`": it says the next version, and points at `mbt registry ls`.

---

## Closing verification

The full `CLAUDE.md` battery, on the working tree after the last edit:

- `uv run pytest -q -m "not e2e" --cov` - **2,248 passed, 52 skipped, coverage 100.00%** (2,248 against 1,940 at the v6 close; the 52 skips are the opt-in live tiers).
- `uv run pytest -q -m e2e --timeout 1800` with `JAVA_HOME` on JDK 17 - **107 passed, 6 skipped**, including the new JSON-stream invariant, the scaffold ownership test on both forges and the Spark compute compliance.
  A py4j `ValueError: I/O operation on closed file` is printed from a `__del__` at interpreter shutdown after the last test; it is py4j logging into a closed stream, exits 0, and is not new.
- `ruff check`, `ruff format --check`, strict `mypy` over all 13 packages (165 files), `pre-commit run --all-files`, `yamllint` as `CLAUDE.md` prescribes, `mkdocs build --strict` - clean.
- `scripts/audit_dependencies.py` - pip-audit clean, the four accepted advisories still firing.
- By hand, through the real CLI in a sandboxed `HOME`: `init` in all four outcomes, `--threads 0`, `registry ls` against a real MLflow registry, and the released-tag install and quickstart that `released.yml` runs (it goes red on `v0.2.0` today, as it should).
- Each new regression test for A-2, B-1 and the C-1 guards was run once with its fix reverted and failed.

## What is left for the maintainer

Two things, and neither is code.

1. **Issue #1's body** still names `orchestrator.py:421` as the `hasattr` gate for batch scoring.
   It is now `Capability.BATCH_SCORING`; the issue is public, so I did not edit it.
2. **PyPI.**
   The release is a GitHub release with wheels attached; publishing to PyPI still waits on the Trusted Publisher setup in `CONTRIBUTING.md`, and until then every install is a git reference, which is why A-1 mattered as much as it did.
