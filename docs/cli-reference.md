# CLI reference

All commands are non-interactive-safe. Exit codes: **0** success, **1** hard
error, **2** quality failure (gate/test). Events stream to stderr (human
text, or JSON lines with `--log-format json`); stdout carries command data.

Common flags behave identically everywhere they appear:
`--project-dir`, `--profiles-dir`, `--target/-t`, `--vars`, `--select/-s`,
`--exclude`, `--threads`, `--state`, `--manifest`, `--anchor`, `--quiet`.

| Command | Purpose |
|---|---|
| `mbt init <name>` | Scaffold a golden-path project (specs, profiles, CI workflows, pre-commit, CODEOWNERS) |
| `mbt deps` | Install adapter packages pinned in `packages.yml` |
| `mbt parse [--write-json-schema]` | Validate configs, build the DAG; no execution |
| `mbt compile [--anchor TS] [--deep-snapshot]` | Produce `target/manifest.json` |
| `mbt run` | Build datasets + train models in DAG order |
| `mbt build` | run + tests/gates interleaved; a failing node skips its downstream |
| `mbt test` | Data checks/tests + gate re-evaluation of registered versions; never trains |
| `mbt evaluate --model X [--version N \| --stage S] [--gates]` | Re-evaluate a registered artifact on fresh data |
| `mbt promote --model X --to production [--version N] [--force]` | Gate-verified stage transition |
| `mbt promote --from-file promotions.yml` | GitOps promotion from a reviewed file |
| `mbt ls [--output table\|name\|path\|json]` | List resources with selector support |
| `mbt show <name> [--output yaml\|json]` | One resource's compiled config |
| `mbt state diff --state <path-or-URI> [--output table\|json]` | What changed vs a reference manifest |
| `mbt docs generate` / `mbt docs serve` | Model cards + lineage site |
| `mbt run-operation <macro> --args '<dict>'` | Render a macro with the compile context |
| `mbt clean` | Delete `target/` including the dataset cache |

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
- `--fail-fast` - cancel pending work on the first failure.
- `--manifest <path>` - execute a stored manifest verbatim (no recompile,
  no re-anchoring): the reproducibility and audit mechanism.
- `--state-include-env` - treat environment digest changes as modifying
  every node.
- `--vars '<yaml dict>'` - override vars for one invocation
  (CLI > target vars > project vars > defaults).
