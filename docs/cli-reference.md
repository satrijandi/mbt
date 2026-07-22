# CLI reference

All commands are non-interactive-safe. Exit codes: **0** success, **1** hard
error, **2** quality failure (gate/test/monitor). Events stream to stderr
(human text, or JSON lines with `--log-format json`); stdout carries
command data.
`mbt --version` prints the installed version and exits.

Common flags behave identically everywhere they appear:
`--project-dir`, `--profiles-dir`, `--target/-t`, `--vars`, `--select/-s`,
`--exclude`, `--threads`, `--state`, `--manifest`, `--anchor`, `--quiet`,
`--verbose/-v`.

`--verbose/-v` surfaces debug-level events in the default text output (they
are dropped otherwise); it has no effect under `--log-format json` (which
always carries every event) or `--quiet` (which suppresses all events).

Set `MBT_LOG_FILE=<path>` to additionally append every event, as redacted
JSON lines, to that file: a durable machine-readable timeline for scheduled
jobs that keeps the human console intact (unlike `--log-format json`, which
replaces it) and captures even under `--quiet`. The path is relative to where
you run mbt; parent directories are created, and each run appends (every event
carries a `run_id` to demultiplex overlapping runs).

Set `MBT_OTEL=1` (needs the `otel` extra: `pip install 'mbt-core[otel]'`) to
also emit each command as an OpenTelemetry trace - one root span per command
with a child span per node, carrying status and timing.
mbt emits spans against your process's globally-configured tracer and does not
stand up an exporter itself, so wire the destination through the standard
`OTEL_*` environment (or `opentelemetry-instrument`); with no provider
installed the spans are a no-op.
Setting `MBT_OTEL` without the extra fails loudly rather than dropping
telemetry silently.

Path semantics: paths you type on the command line (`--state`,
`--manifest`, `--from-file`, `--profiles-dir`) are relative to where you
run mbt, shell-style; relative paths inside config (profiles' `file://`
stores, sqlite URIs, adapter roots) resolve against the project directory,
wherever mbt is invoked from. URIs (`s3://...`) pass through untouched.

| Command | Purpose |
|---|---|
| `mbt init <name>` | Scaffold a golden-path project (specs, profiles, CI workflows, pre-commit, Renovate, CODEOWNERS) |
| `mbt deps` | Install adapter packages: prefers the project's pinned `requirements.txt` when present (reproducible), else installs `packages.yml` specifiers with an unpinned-install warning; either way the result is verified against `packages.yml` |
| `mbt parse [--write-json-schema]` | Validate configs, build the DAG; no execution |
| `mbt compile [--anchor TS] [--deep-snapshot]` | Produce `target/manifest.json` |
| `mbt run` | Build datasets + train models in DAG order |
| `mbt build` | run + tests/gates interleaved; a failing node skips its downstream |
| `mbt test` | Data checks/tests + gate re-evaluation of registered versions; never trains |
| `mbt score` | Batch-score scoring pipelines with their registered champions: input checks, predictions to the configured sink, shift monitors (ADR-20) |
| `mbt monitor` | Evaluate matured predictions against arrived labels; realized-metric gates, each prediction run evaluated once (ADR-21) |
| `mbt predictions ls [--output table\|json]` / `mbt predictions show <run_key>` | Inspect the prediction store: which runs exist, which matured, which were evaluated, and their realized metrics (read-only over the score/monitor ledger) |
| `mbt evaluate --model X [--version N \| --stage S] [--gates]` | Re-evaluate a registered artifact on fresh data |
| `mbt promote --model X --to production [--version N] [--force]` | Gate-verified stage transition |
| `mbt promote --from-file promotions.yml` | GitOps promotion from a reviewed file |
| `mbt rollback --model X [--to-version N] [--force]` | Revert the production champion to a prior version (incident rollback) |
| `mbt ls [--output table\|name\|path\|json]` | List resources with selector support |
| `mbt show <name> [--output yaml\|json]` | One resource's compiled config |
| `mbt state diff --state <path-or-URI> [--deep-snapshot] [--output table\|json]` | What changed vs a reference manifest (deep-snapshot both sides to ignore mtime churn) |
| `mbt docs generate` / `mbt docs serve` | Model cards (metrics, gates, feature importance, partial dependence) + lineage site |
| `mbt run-operation <macro> --args '<dict>'` | Render a macro with the compile context |
| `mbt clean [--artifacts-older-than 30d] [--dry-run]` | Delete `target/` (and age out leaked `mbt-job-*` error payloads older than 7 days), or prune old artifact-store run prefixes (champions and the latest run always survive) |

## Selector grammar

```
atom      := [N]+ body +[N]           # upstream/downstream, optional depth
body      := name_glob | tag:V | resource_type:V | state:new|modified
intersect := atom,atom                 # comma = AND
union     := "intersect intersect"     # space = OR
```

`state:` methods need `--state <path-or-URI>` (file:// and s3://).

## Execution flags

- `--threads N` - parallel independent DAG branches (default: the target's).
  Also on `mbt monitor`: scoring nodes are independent, so they evaluate in
  parallel.
- `--fail-fast` - cancel pending work on the first failure and terminate
  in-flight training jobs (SIGTERM, then SIGKILL after a grace period).
  Also on `mbt monitor`.
- `--manifest <path>` - execute a stored manifest verbatim (no recompile,
  no re-anchoring): the reproducibility and audit mechanism. Verifies the
  environment first: an `env_digest` mismatch errors, transitive drift
  (`env_freeze_digest`) warns (ADR-19).
- `--allow-env-mismatch` - downgrade the `--manifest` env check from error
  to warning for deliberate cross-environment runs.
- `--state-include-env` - treat environment digest changes as modifying
  every node.
- `--vars '<yaml dict>'` - override vars for one invocation
  (CLI > target vars > project vars > defaults).
