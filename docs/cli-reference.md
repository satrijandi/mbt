# CLI reference

All commands are non-interactive-safe. Exit codes: **0** success, **1** hard
error, **2** quality failure (gate/test/monitor). Events stream to stderr
(human text, or JSON lines with `--log-format json`); stdout carries
command data.
`mbt --version` prints the installed version and exits.

Common flags behave identically everywhere they appear:
`--project-dir`, `--profiles-dir`, `--target/-t`, `--vars`, `--select/-s`,
`--exclude`, `--threads`, `--state`, `--manifest`, `--anchor`, `--quiet`.

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
| `mbt evaluate --model X [--version N \| --stage S] [--gates]` | Re-evaluate a registered artifact on fresh data |
| `mbt promote --model X --to production [--version N] [--force]` | Gate-verified stage transition |
| `mbt promote --from-file promotions.yml` | GitOps promotion from a reviewed file |
| `mbt ls [--output table\|name\|path\|json]` | List resources with selector support |
| `mbt show <name> [--output yaml\|json]` | One resource's compiled config |
| `mbt state diff --state <path-or-URI> [--deep-snapshot] [--output table\|json]` | What changed vs a reference manifest (deep-snapshot both sides to ignore mtime churn) |
| `mbt docs generate` / `mbt docs serve` | Model cards (metrics, gates, feature importance) + lineage site |
| `mbt run-operation <macro> --args '<dict>'` | Render a macro with the compile context |
| `mbt clean [--artifacts-older-than 30d] [--dry-run]` | Delete `target/`, or prune old artifact-store run prefixes (champions and the latest run always survive) |

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
