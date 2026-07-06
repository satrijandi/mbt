# ADR-4: Two hashes per node; state:modified compares input_hash

**Status:** accepted

## Decision
`config_hash = sha256(canonical_json(rendered_spec) + hooks_bytes)`;
`input_hash = sha256(config_hash + snapshot_id + sorted(upstream input_hashes))`
computed in topological order. `state:modified` compares `input_hash`.

## Rationale
One comparison captures config, hooks, snapshot, and upstream changes -
simple and correct. A separate train-only hash (so gate edits do not
retrain) is deferred until dogfooding shows the cost matters (see ADR-6).
