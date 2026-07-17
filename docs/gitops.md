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
   Every compiling step passes `--deep-snapshot`: CI checkouts are fresh,
   so the default mtime snapshots would flag every dataset on every run
   and the economy loop would silently become a full retrain (ADR-11).
   Use one token scheme on both sides of every comparison.
2. **Prod build** (`prod_build.yml`, on merge): build `state:modified+`
   against prod; on success publish the manifest as the new baseline on
   the `mbt-state` branch (`scripts/publish_state.sh`).
3. **Promotion** (`promote.yml`): a reviewed `promotions.yml` change (pure
   GitOps) or a manually approved `workflow_dispatch` runs
   `mbt promote`, which refuses versions without recorded gate passes.
4. **Scheduled retraining** (`scheduled_retrain.yml`): CI cron +
   `mbt build --select tag:weekly` - freshness arrives as new snapshots;
   no orchestrator concept needed.
5. **Scheduled scoring** (`scheduled_score.yml`): CI cron +
   `mbt score --target prod --select tag:daily` - each pipeline loads its
   model's current production champion from the registry, so promotions
   take effect on the next run; a shift-monitor breach or input-check
   failure exits 2 and fires the alert webhook (ADR-20).
6. **Ground-truth monitoring** (`scheduled_monitor.yml`): CI cron +
   `mbt monitor --target prod` - evaluates realized metrics for prediction
   runs whose labels have matured, exactly once per run (ledger markers in
   the prediction store, ADR-21); a realized-metric gate failure exits 2.

## Manifest storage convention

```
s3://<bucket>/mbt/<project>/<target>/manifests/<git_sha>.json
s3://<bucket>/mbt/<project>/<target>/manifests/latest.json
```

The prod build uploads its manifest on success; PR checks pass
`--state s3://.../latest.json`. Teams without S3 use CI artifact storage
with the same layout - `--state` accepts any readable path or URI, and an
unreadable reference is a hard error, never a silent full retrain.

Model artifacts follow the same split: point `artifact_store` at
`s3://<bucket>/mbt/<project>/artifacts` (needs `mbt-core[s3]`) for any
multi-runner or production topology. Retention: prune local stores with
`mbt clean --artifacts-older-than 30d` (stage champions and the latest
run's artifacts always survive, so champion re-evaluation cannot break);
give S3 stores a bucket lifecycle rule instead.

## Operations: alerting and durable state

Out of the box the prod baseline lives on a dedicated **`mbt-state`
branch**: after a successful prod build, `scripts/publish_state.sh`
appends a commit holding `manifest.json` (git plumbing only - it never
touches the working tree or the current branch) and pushes it; PR checks
and later prod builds restore it with `scripts/fetch_state.sh`, which
exits with a distinct code when no baseline exists yet so the first-ever
run falls back to a full build. So the `state:modified` economy loop
works from the first merge with no extra infrastructure, survives branch
protection on `main` (nothing is ever pushed there), and the branch
history is an audit trail of every published baseline. When state moves
to object storage, swap both script calls for the `aws s3 cp` layout
above. The scheduled retrain, scoring, monitor, and prod build workflows
alert on failure through the `MBT_ALERT_WEBHOOK` secret (any
Slack/Teams-style JSON webhook); unset, the step logs and skips - a
scheduled retrain that fails silently violates continuous training, and a
scoring or monitor run that fails silently hides broken serving.
When a run fails or retrains more than you expected, the
[troubleshooting runbook](troubleshooting.md) maps each deliberate
failure mode (snapshot drift, env mismatch, unloadable champion,
gate-edit retrains) to its cause and fix.

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

The manifest also records `env_freeze_digest`, a hash of every installed
package, so transitive drift (numpy, scipy) is visible even when the
fingerprinted packages match (ADR-19). Executing a stored manifest verifies
both: an `env_digest` mismatch is a hard error (`--allow-env-mismatch`
downgrades it), a freeze-only mismatch warns.

## Installing mbt in CI

The scaffolded workflows install the toolchain from `requirements.txt`, which
pins the mbt packages to an **immutable release tag** via git refs, e.g.:

```text
mbt-core @ git+https://github.com/satrijandi/mbt@v0.1.0#subdirectory=packages/mbt-core
```

This is reproducible - a tag is immutable, so the training environment never
floats and the manifest's `env_digest` stays stable - and installs straight
from a fresh checkout, with no private package index required.

!!! warning "Your CI will not install until the matching `vX.Y.Z` tag exists"
    The mbt repo's `release.yml` builds that tag's wheels (and, once Trusted
    Publishing is configured, publishes them to PyPI) when a version tag is
    pushed. Once mbt is on PyPI you can switch the pins to plain versions
    (`mbt-core==0.1.0`) and set `PIP_INDEX_URL` if an internal index serves them.
