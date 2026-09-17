# CLI reference

Every `mbt` command is non-interactive, so it runs the same in a terminal, a CI job, and a scheduler.
This page covers the conventions every command shares, then each command in turn.
`mbt <command> --help` prints the same options from the installed version.

```text
mbt [--version] <command> [options]
```

## Conventions

### Exit codes

| Code | Meaning | Examples |
|---|---|---|
| `0` | success | a build that trained, gated, and registered |
| `1` | hard error: something is broken | an invalid spec, a missing file, an unloadable champion, a crashed job, a mistyped flag |
| `2` | quality failure: the pipeline ran, and a verdict said no | a failing gate, check, data test, shift monitor, or realized-metric gate |

CI and schedulers should treat `1` as "page someone" and `2` as "review the model".
A quality verdict is deterministic, so retrying an exit-2 run only reproduces it.

### Output streams

Events - progress, check and gate verdicts, warnings - go to **stderr**, as timestamped text by default or as JSON lines with `--log-format json`.
**stdout** carries command data: the results table after an execution command, and the output of `mbt ls`, `mbt show`, `mbt state diff`, and `mbt predictions`.
So `mbt ls --output json > resources.json` captures clean JSON while progress still reaches the terminal.

### Common options

These behave the same on every command that accepts them.

<div class="cli-options" markdown>

| Option | Meaning |
|---|---|
| `--project-dir PATH` | Project root; defaults to the current directory. mbt changes into it before running, so paths inside config resolve against the project |
| `--profiles-dir PATH` | Directory holding `profiles.yml` (see [search order](spec-reference.md#profilesyml)) |
| `-t`, `--target NAME` | Profile target to run against; defaults to the profile's `target:` |
| `--vars YAML` | A YAML or JSON mapping of vars for this invocation. Precedence: CLI, then target `vars:`, then project `vars:` |
| `-s`, `--select SELECTOR` | Nodes to run; see [Selectors](#selectors). Several selectors separated by spaces form a union |
| `--exclude SELECTOR` | Nodes to subtract from the selection |
| `--threads N` | Independent DAG branches to run in parallel; defaults to the target's `threads` |
| `--fail-fast` | Stop at the first failure: cancel pending nodes and terminate running training jobs (SIGTERM, then SIGKILL after a grace period) |
| `--state PATH_OR_URI` | Reference manifest for `state:` selectors and `mbt state diff`; a local path, `file://`, or `s3://` URI |
| `--state-include-env` | Treat an `env_digest` change as modifying every node, so an environment upgrade retrains everything (ADR-7) |
| `--manifest PATH` | Execute a stored manifest verbatim - no recompile, no re-anchoring - after verifying the running environment against it (ADR-19) |
| `--allow-env-mismatch` | With `--manifest`, downgrade an `env_digest` mismatch from an error to a warning. A freeze-only mismatch always just warns |
| `--anchor TIMESTAMP` | Pin the time anchor every window expression resolves against (ISO 8601, e.g. `2026-06-30T00:00:00Z`); defaults to now |
| `--deep-snapshot` | Pin data snapshots by content hash instead of the default file listing. Slower, and immune to mtime churn from fresh checkouts (ADR-11). Use one scheme per pipeline |
| `--log-format FORMAT` | Event format on stderr: `text` (the default) or `json`, which also suppresses the results table |
| `-q`, `--quiet` | Suppress events |
| `-v`, `--verbose` | Show debug-level events, such as per-trial tuning lines (text mode only) |

</div>

### Paths

Paths you type on the command line (`--state`, `--manifest`, `--from-file`, `--profiles-dir`) are relative to where you run mbt, as in any shell command.
Relative paths inside config files (a `file://` artifact store, a sqlite URI, an adapter `root`) resolve against the project directory, wherever mbt runs from.
URIs such as `s3://...` pass through untouched.

### Environment variables

| Variable | Effect |
|---|---|
| `MBT_PROFILES_DIR` | Directory holding `profiles.yml`, checked after `--profiles-dir` |
| `MBT_LOG_FILE=PATH` | Also append every event as redacted JSON lines to `PATH`, whatever `--log-format` and `--quiet` say. Relative to where mbt runs; each line carries a `run_id`, so overlapping runs can share a file |
| `MBT_OTEL=1` | Emit each command as an OpenTelemetry trace - a root span per command and a child span per node - against your process's configured tracer. Needs `mbt-core[otel]`; mbt ships no exporter, so wire one through the standard `OTEL_*` variables. Setting it without the extra is an error, not a silent no-op |
| `MBT_DEBUG=1` | Print the full traceback for an `Internal error` instead of the one-line summary |

Values in `profiles.yml` read the environment through `{{ env_var('NAME') }}` (secrets, redacted everywhere) and `{{ env('NAME') }}` (plain values); see [Reading the environment](spec-reference.md#reading-the-environment-env_var-vs-env).

### Files mbt writes

| Path | Written by | Contents |
|---|---|---|
| `target/manifest.json` | every compiling command | The pinned, hashed plan: resolved windows, snapshots, config and input hashes, environment digests |
| `target/run_results.json` | every execution command | The most recent run's per-node status, timings, metrics, gates, monitors, and registrations |
| `target/run_results.<command>.json` | the same commands | Identical content, kept per command, so `mbt score` does not overwrite what `mbt build` recorded |
| `target/run_logs/<run_id>/<unique_id>.log` | every execution command and `mbt evaluate` | Everything each node logged, at every level whatever the console shows; a model's log (its dataset's section first) is also uploaded to its tracking run as `logs/train.log` (ADR-30) |
| `target/datasets/<name>/<key>/` | dataset builds | One Parquet file per split, reused while the materialization key matches |
| `target/docs/` | `mbt docs generate` | The static model-card and lineage site |
| `target/json-schemas/` | `mbt parse --write-json-schema` | JSON Schemas for editor autocomplete |

### Selectors

`--select`, `--exclude`, and `mbt ls` share dbt's selector grammar.

```text
atom      := [N]+ body +[N]        # upstream / downstream, optional depth N
body      := name_glob | tag:VALUE | resource_type:VALUE | state:new | state:modified
intersect := atom,atom             # comma = AND
union     := "intersect intersect" # space = OR
```

```bash
mbt build --select churn_classifier+             # the model and everything downstream
mbt build --select +churn_classifier             # the model and everything upstream
mbt build --select tag:weekly,state:modified+    # intersection
mbt build --select "tag:churn tag:upsell"        # union
mbt build --select resource_type:model --exclude tag:experimental
mbt build --select state:modified+ --state state/prod/latest.json
```

`state:` methods need `--state`.
Selection decides which models **train**; every dataset a selected model needs is materialized regardless, from cache when possible (ADR-13).

## Project setup

### `mbt init`

Scaffold a working project: example source, dataset, model, and scoring specs, `profiles.yml`, reference GitHub Actions workflows, pinned CI requirements, pre-commit and Renovate config, `CODEOWNERS`, and a sample-data generator.

```bash
mbt init NAME [--project-dir PATH]
```

`NAME` must start with a letter and contain only letters, digits, and underscores.
The project is created in `NAME/` under `--project-dir`.
Its `profiles.yml` is also installed to `~/.mbt/profiles.yml`, for commands run outside the project: appended when that file already exists, and left alone when it already has a profile named `NAME`.
See the [Quickstart](quickstart.md) for what to do next.

### `mbt deps`

Install the adapter packages the project lists in `packages.yml`.

```bash
mbt deps [--dry-run]
```

When the project has a `requirements.txt`, mbt installs from it (pinned and reproducible); otherwise it installs the `packages.yml` specifiers and warns that the result is unpinned.
Either way it then verifies the installed versions against `packages.yml` and fails naming any mismatch.
`--dry-run` prints what would be installed.

### `mbt clean`

Delete `target/`, or garbage-collect the artifact store.

```bash
mbt clean
mbt clean --artifacts-older-than 30d [--dry-run]
```

<div class="cli-options" markdown>

| Option | Meaning |
|---|---|
| `--artifacts-older-than DURATION` | Instead of deleting `target/`, prune artifact-store run prefixes older than `DURATION` (`30d`, `12h`). Every stage's champion and the latest run's artifacts always survive. `file://` stores only; give object stores a lifecycle rule |
| `--dry-run` | List what the artifact GC would delete, and delete nothing |

</div>

Plain `mbt clean` also removes debugging payloads that failed training jobs left in the temp directory (`mbt-job-*`) once they are older than 7 days.

## Validate and build

### `mbt parse`

Validate every config file and build the DAG, without compiling or executing anything.

```bash
mbt parse [--write-json-schema]
```

Parsing checks schemas (with did-you-mean suggestions), task and adapter compatibility, hyperparameters against each adapter's parameter model, adapter support for the features a spec uses, and cross-resource references - and reports every error in one pass.
`--write-json-schema` also writes JSON Schemas to `target/json-schemas/`; the scaffolded specs point their `yaml-language-server` headers at them.

### `mbt compile`

Render Jinja and profiles, resolve every window against one anchor, pin data snapshots, hash every node, and write `target/manifest.json`.

```bash
mbt compile [--anchor TIMESTAMP] [--deep-snapshot]
```

Two compiles at the same anchor over the same data produce byte-identical manifests.

### `mbt run`

Materialize datasets and train models in DAG order.

```bash
mbt run [--select ...] [--manifest PATH]
```

A dataset's declared `checks` run as part of materializing it, and a model's gates always decide whether it registers, so `run` never registers a model that failed a gate.
What `run` skips is the project's Python data tests in `tests/`; use `mbt build` for those.

`mbt run --manifest target/manifest.json` re-executes a stored plan exactly: same anchor, windows, snapshots, hashes, and seeds.
For the XGBoost, LightGBM, and scikit-learn adapters that reproduces metrics bit for bit.

### `mbt build`

Everything `mbt run` does, plus the Python data tests bound to each dataset, in DAG order.
This is the command CI runs.

```bash
mbt build [--select ...] [--state PATH_OR_URI] [--threads N] [--fail-fast]
```

A node that fails - an error, a failed check or test, a failed gate - skips its downstream nodes, and independent branches keep going unless `--fail-fast` is set.
A model registers only when every one of its gates passes, and lands in the spec's `registration.stage_on_pass` stage.

### `mbt test`

Run dataset checks and data tests, and re-evaluate registered models against their gates.
It never trains.

```bash
mbt test [--select ...]
```

For each selected model with gates, `mbt test` re-evaluates the version in the model's `stage_on_pass` stage on freshly built data.
A model with no version in that stage is skipped with a warning.

## Serve and monitor

### `mbt score`

Run batch scoring pipelines: resolve each pipeline's champion from the registry, check the input batch, write predictions, and compare feature and score distributions against the champion's training-time baseline (ADR-20).

```bash
mbt score [--select tag:daily] [--target prod]
```

Champions resolve by stage alias at run time, so a promotion takes effect on the next run with no spec change.
A failed input check or a shift-monitor breach exits `2`; a missing champion exits `1`.
Re-running at the same anchor overwrites the same prediction run, and a new anchor writes a new one.

### `mbt monitor`

Evaluate stored predictions whose ground truth has matured: join arrived labels, compute realized metrics, and apply the pipeline's `ground_truth.gates` (ADR-21).

```bash
mbt monitor [--select ...] [--threads N] [--fail-fast]
```

Each prediction run is evaluated exactly once; a run whose labels have not arrived yet stays eligible for the next monitor run.
A realized-metric gate breach exits `2`.
Scoring pipelines are independent, so `--threads` evaluates them in parallel.

### `mbt predictions`

Inspect the prediction store and its ground-truth ledger, read-only.

```bash
mbt predictions ls [--output table|json]
mbt predictions show RUN_KEY [--output table|json]
```

`mbt predictions ls` lists every prediction run across scoring pipelines: when it was scored, with which champion version, how many rows, and whether it has matured and been evaluated.
`mbt predictions show` details one run, including its label coverage and realized metrics.

## Registry

### `mbt evaluate`

Re-evaluate one registered model version on freshly built data, without training.

```bash
mbt evaluate --model NAME [--version N | --stage STAGE] [--gates] [--out-of-time]
```

<div class="cli-options" markdown>

| Option | Meaning |
|---|---|
| `--model NAME` | The model to evaluate (required) |
| `--version N` | A specific registry version |
| `--stage STAGE` | Evaluate the version currently in this stage. With neither option, the version in the model's `stage_on_pass` stage |
| `--gates` | Also apply the model's gates to the fresh metrics, and exit `2` if one fails |
| `--out-of-time` | Run the pre-deploy check instead (ADR-30): the version's recorded test window against everything since, reported on its training run. With `--gates`, its after-test and stability gates are judged, and the verdict is recorded on the version |

</div>

`mbt evaluate --model churn_classifier --stage production --gates` is the decay check: the production champion's metrics on today's data, held to the spec's thresholds.
Champion gates are not applicable when the version being evaluated is itself the champion, and report so instead of comparing the model with itself.
After-test gates (`source: out_of_time`) are judged by `mbt build` and by the pre-deploy check, never by a plain re-evaluation.

**The pre-deploy check.**
A version's test window is usually months old by the day it ships.
`mbt evaluate --model NAME --version N --out-of-time --gates` rebuilds that version's dataset with its recorded test window as the reference and an after-test window from the end of the test window to this run's anchor, then scores both with the version itself and its own spec.
The [training report](spec-reference.md#the-training-report-adr-30) lands on the version's training run under `evaluations/<run_id>/`, with metrics prefixed `oot_check.`; no parameter is logged, because the run's parameters describe its training.
The verdict goes on the version as `mbt.oot_check.passed` - `true`, `false`, or `not_gated` when nothing was mature enough to judge - which `mbt promote --require-oot-check` reads.
The check refuses a version registered before mbt recorded dataset windows, a random-split model, and an anchor that is not after the recorded test window.

### `mbt promote`

Move a registered version to a stage, refusing any version whose gates were not recorded as passed.

```bash
mbt promote --model NAME --to STAGE [--version N] [--force] [--require-oot-check]
mbt promote --from-file promotions.yml [--require-oot-check]
```

<div class="cli-options" markdown>

| Option | Meaning |
|---|---|
| `--model NAME` | The registered model name |
| `--to STAGE` | Target stage: `staging`, `production`, or `archived` |
| `--version N` | The version to promote; defaults to the version currently in `staging` |
| `--from-file PATH` | Apply every entry of a reviewed `promotions.yml` (the GitOps path; see the [spec reference](spec-reference.md#promotionsyml)) |
| `--force` | Promote even without a recorded gate pass, or past a failed after-test check. The event is marked `FORCED` |
| `--require-oot-check` | Also refuse a version whose latest after-test verdict is not a pass - never judged, or nothing mature to judge. A pass recorded by the build's own after-test gates counts. A `promotions.yml` entry can ask for the same with `require_oot_check: true` |

</div>

Promoting a version to a stage it already holds re-points the alias at the same version, so replaying a merged `promotions.yml` is safe.
A version whose latest after-test check failed (`mbt.oot_check.passed: false`, from `mbt build` or the pre-deploy check) is refused even without `--require-oot-check`.

### `mbt rollback`

Revert a model's production champion during an incident.

```bash
mbt rollback --model NAME [--to-version N] [--force]
```

With no `--to-version`, mbt picks the newest version below the current champion that recorded passing gates.
It checks that the target's artifact still exists before moving the alias, and goes through the same recorded-gate check as `mbt promote`.
A failed after-test check on the target only warns: incident response must be able to reach the last good version.
See the [rollback procedure](troubleshooting.md#rolling-back-a-bad-champion-incident-procedure).

## Inspect

### `mbt ls`

List resources, filtered by the same selectors `--select` accepts.

```bash
mbt ls [--select ...] [--output table|name|path|json]
```

### `mbt show`

Print one resource's compiled configuration, with secrets redacted.

```bash
mbt show NAME_OR_UNIQUE_ID [--output yaml|json]
```

### `mbt state diff`

Compare the current project with a reference manifest, and say which component of each node changed.

```bash
mbt state diff --state PATH_OR_URI [--deep-snapshot] [--output table|json]
```

Components are `config` (the spec or hooks), `snapshot` (the data), and `upstream` (a changed ancestor).
An `env_digest` change is reported separately and, by default, marks nothing modified (ADR-7).
Compare like with like: diff with `--deep-snapshot` against a baseline that was compiled with it.

### `mbt docs generate`

Render model cards and a lineage site into `target/docs/`.

```bash
mbt docs generate [--manifest PATH]
```

Cards show each model's data windows, features, hyperparameters, metrics, slices, gate results, feature importance, partial dependence, and registry and tracking ids, read from the latest training run.

### `mbt docs serve`

Serve `target/docs/` on `http://127.0.0.1`.

```bash
mbt docs serve [--port 8080]
```

### `mbt run-operation`

Render a macro from `macros/` with the full compile context and print the result.

```bash
mbt run-operation MACRO [--args YAML]
```

`--args` is a YAML or JSON mapping of the macro's arguments, for example `mbt run-operation recent_window --args '{days: 28}'`.
Macros render text; they do not call adapters.

## Global options

<div class="cli-options" markdown>

| Option | Meaning |
|---|---|
| `--version` | Print the installed version and exit |
| `--install-completion` | Install shell completion for the current shell |
| `--show-completion` | Print the completion script, to install it manually |

</div>
