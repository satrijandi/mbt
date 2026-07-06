# GitOps & CI

Git is the source of truth for specs; the registry is the source of truth
for artifacts. The compiled manifest links the two: config hash ↔ data
snapshot ↔ model version.

## The loop

1. **PR check** (`pr_check.yml`, shipped by `mbt init`):
   `mbt parse` → `mbt compile --target dev` →
   `mbt state diff --state <latest prod manifest> --output json` →
   `mbt build --target dev --select state:modified+ --state ...` →
   post/update the PR comment (metrics vs champion, gate table, retrained
   nodes, cost estimate) from `run_results.json` + `state_diff.json`.
2. **Prod build** (`prod_build.yml`, on merge): build `state:modified+`
   against prod; on success publish the manifest as the new `latest.json`.
3. **Promotion** (`promote.yml`): a reviewed `promotions.yml` change (pure
   GitOps) or a manually approved `workflow_dispatch` runs
   `mbt promote`, which refuses versions without recorded gate passes.
4. **Scheduled retraining** (`scheduled_retrain.yml`): CI cron +
   `mbt build --select tag:weekly` - freshness arrives as new snapshots;
   no orchestrator concept needed.

## Manifest storage convention

```
s3://<bucket>/mbt/<project>/<target>/manifests/<git_sha>.json
s3://<bucket>/mbt/<project>/<target>/manifests/latest.json
```

The prod build uploads its manifest on success; PR checks pass
`--state s3://.../latest.json`. Teams without S3 use CI artifact storage
with the same layout - `--state` accepts any readable path or URI, and an
unreadable reference is a hard error, never a silent full retrain.

## Why retraining stays cheap

- `state:modified` compares transitive input hashes: config, hooks, pinned
  snapshot, and upstream changes all surface through one comparison.
- Anchor drift alone changes nothing - windows are hashed as expressions.
- Dev targets sample data (`sample_fraction`) and cap tuning
  (`max_tuning_trials`), so PR builds are small by construction.
- The PR comment surfaces the retrained node list and a cost estimate, so
  reviewers see the bill before merging.

## Environment changes

Adapter or Python upgrades change the manifest's `env_digest`. By default
this does **not** mark nodes modified (an adapter bump would retrain
everything); `mbt state diff` reports it prominently and
`--state-include-env` opts in when you *want* the full retrain.
