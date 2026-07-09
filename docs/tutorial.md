# mbt tutorial: data scientist + MLOps engineer, step by step

This tutorial walks one data scientist (DS) and one MLOps engineer (MLOps) through the full mbt lifecycle on one shared project:
scaffold, data, specs, a first gated build, registration, promotion, a champion/challenger PR loop, batch scoring, ground-truth monitoring, and production wiring with alerting.

Every step says **who** drives it, **what** they type or edit, **what output to expect**, and **how to verify** it worked before moving on.
Budget about an hour; the training itself takes seconds.

The one idea that holds the whole workflow together:

> Git owns the specs, the registry owns the artifacts, and the compiled manifest links the two (config hash ↔ data snapshot ↔ model version).

## Who owns what

| Area | DS | MLOps |
|---|---|---|
| `sources.yml`, `datasets/`, `models/`, `scoring/`, `tests/` | owns | reviews infra-touching bits |
| `profiles.yml`, CI workflows, secrets, schedules | | owns |
| `promotions.yml` (what runs in production) | proposes | approves |
| Quality gate thresholds (in specs) | owns via PR | |
| Alerting rules and dashboards (outside specs) | consumes | owns |

The scaffold encodes this split in `CODEOWNERS`: model and dataset specs route to the DS team, `promotions.yml` routes to the MLOps team.

## Prerequisites (both)

- Python 3.11+ (3.11 through 3.14 are tested in CI).
- `pip install mbt-core mbt-xgboost mbt-mlflow` (or run from this repo with `uv run mbt ...`).
- A git host with PR reviews if you want the full team loop; every step below also works purely locally.

---

## Step 1 (MLOps): scaffold the project

```bash
mbt init acme_models && cd acme_models
git init && git add -A && git commit -m "mbt scaffold"
```

**You get** a complete working project:
example source/dataset/model/scoring specs, `profiles.yml` with `dev` and `prod` targets, an empty `promotions.yml`, six GitHub workflows (`pr_check`, `prod_build`, `promote`, `scheduled_retrain`, `scheduled_score`, `scheduled_monitor`), state-publishing scripts, a hash-pinned `requirements.txt`, pre-commit config, and `CODEOWNERS`.

**Your job in this step** is `profiles.yml`, the only file that describes environments:

- `dev` points at local files and a sqlite MLflow, with cheap per-target vars (`sample_fraction`, `max_tuning_trials: 5`) so PR builds are small by construction.
- `prod` resolves `MBT_DATA_ROOT` and `MLFLOW_TRACKING_URI` through `{{ env_var(...) }}`.
  Secrets never enter any artifact; the target config is stored unrendered.

**Verify:** the tree above exists, and `mbt parse` exits 0 once data exists (step 2).
Note that `profiles.yml` is gitignored by the scaffold; the canonical copy is also installed to `~/.mbt/profiles.yml`.

## Step 2 (DS): point at data

For the tutorial, generate deterministic sample data:

```bash
python scripts/generate_sample_data.py
```

**Expect** three parquet directories: `data/subscribers/` (5000 training rows), `data/scoring_batch/` (fresh unlabeled rows), and `data/churn_outcomes/` (matured labels for monitoring later).

For a real project, replace this by editing `sources.yml` to point at your parquet:

```yaml
sources:
  - name: lakehouse
    tables:
      - name: subscribers
        path: data/subscribers/*.parquet
```

**Verify:** the parquet paths in `sources.yml` glob to at least one file.

## Step 3 (DS): read the dataset spec

Open `datasets/churn_training_set.yml`.
This file is the entire "training set construction" surface:

- `source:` which table feeds it.
- `label:` the target column and its business definition.
- `filters:` population rules (`is_active = true`, `tenure_days >= 30`).
- `split:` a temporal policy (`train: "-180d:-28d"`, `test: "-28d:now"`) resolved against one compile-time anchor, never against wall-clock per node.
- `checks:` data quality (`not_null`, `no_future_columns`, class balance).
  A label-leakage scan also runs by default without being declared.

**Verify:** nothing to run yet; just confirm the label and filters match your intent.
This file is what your teammates review instead of a notebook.

## Step 4 (DS): read the model spec

Open `models/churn_classifier.yml`.
The config IS the model:

- `task: binary_classification`, `adapter: xgboost`, and `dataset: ref('churn_training_set')` (this `ref()` is what builds the DAG).
- `features.exclude: [user_id]` is an explicit, reviewable leakage guard.
- `hyperparameters.scale_pos_weight: "{{ auto }}"` lets the adapter compute it from class balance at train time.
- `evaluation.gates` declares quality thresholds that block registration when they fail.
- `registration.stage_on_pass: staging` means a passing model lands in the registry at `staging`; humans promote to `production`.
- `seed: 42` is mandatory; every source of randomness derives from it.

**Verify:** `owner:` is set to your team address.
It ends up on model cards, in run artifacts, and (later) as the alert-routing label.

## Step 5 (DS): validate without executing

```bash
mbt parse
```

**Expect:** `Parsed 3 nodes, 3 sources in 0.02s`.
Schema validation with did-you-mean suggestions, task/adapter compatibility, hyperparameter validation, and DAG construction all happen here, with all errors reported in one pass.

Optional but recommended once per machine:

```bash
mbt parse --write-json-schema
```

publishes JSON Schemas under `target/json-schemas/` so your editor autocompletes every spec (the scaffolded files already carry `yaml-language-server` headers).

**Verify:** exit code 0 (`echo $?`).

## Step 6 (DS): first build

```bash
mbt build
```

**What happens, in order:** compile (pin everything into `target/manifest.json`), materialize datasets with checks, train each model in an isolated job, evaluate gates, register passing models, write `target/run_results.json`.

**Expect output like** (numbers vary slightly with your run date; the sample data yields pr_auc around 0.32):

```
Compiled 3 nodes (anchor 2026-07-08T19:54:23Z) -> target/manifest.json
[1/2] dataset churn_training_set   label balance (train): 0=78.4%, 1=21.6%   SUCCESS
[2/2] model churn_classifier       auto-resolved scale_pos_weight = 3.62
      gate pr_auc (threshold): PASS - expected 0.25, got 0.3161
      registered churn_classifier v1 -> staging (mlflow)
build finished [success]: 2 ok, 0 failed in 4.0s
```

**Verify three artifacts:**

1. `target/manifest.json` pins the anchor, per-source `snapshot_id`, per-node `config_hash` and transitive `input_hash`, and environment digests.
2. `target/run_results.json` records statuses, timings, metrics, and full gate records, machine-readable.
3. The registry holds `churn_classifier` v1 at `staging`, and the gate pass was recorded with it.

**Exit codes matter from here on:** 0 success, 1 hard error, 2 quality failure (a gate, check, test, or monitor said no).
CI and orchestrators branch on this distinction.

## Step 7 (both): inspect and document

```bash
mbt ls                                  # resources, tags, paths
mbt show churn_classifier               # fully compiled config, secrets redacted
mbt docs generate && mbt docs serve     # model cards + lineage site
```

**Verify:** the model card shows the metrics, gates, features, and lineage you expect to present at review.

## Step 8 (MLOps): prove reproducibility

```bash
cp target/manifest.json baseline_manifest.json
mbt run --manifest baseline_manifest.json
```

**Expect:** identical metrics to step 6, digit for digit (XGBoost documents an exact determinism tier).
The command first verifies the running environment against the manifest's `env_digest` and refuses on mismatch, so "it reproduced" also means "on a compatible environment".

**Verify:** compare the metrics lines; they must match exactly.
Keep `baseline_manifest.json`; it plays the role of the published prod baseline in step 10.

## Step 9 (both): promote to production

The DS proposes; MLOps approves.
Locally:

```bash
mbt promote --model churn_classifier --version 1 --to production
```

**Expect:** `promoted churn_classifier v1 -> production`.

In the team flow this is a PR to `promotions.yml` (owned by MLOps in CODEOWNERS); merging it triggers the `promote.yml` workflow.
Either path enforces the same rule: **promote refuses any version whose gates were not recorded as passed** (override only with an explicit `--force`).

**Verify:** try promoting a version that never passed gates and confirm the refusal.

## Step 10 (DS): the PR iteration loop

This is the everyday loop, and the step where mbt earns its keep.
Make a change a DS would actually make; edit `models/churn_classifier.yml`:

```yaml
    hyperparameters:
      max_depth: 6            # was 4
      ...
    evaluation:
      gates:
        - metric: pr_auc
          threshold: "{{ var('pr_auc_floor') }}"
        - metric: pr_auc      # NEW: champion/challenger gate
          compare_to: production
          min_delta: 0.005
```

Now run exactly what the `pr_check.yml` workflow runs:

```bash
mbt compile
mbt state diff --state baseline_manifest.json
mbt build --select state:modified+ --state baseline_manifest.json
```

**Expect from `state diff`:** only the model (component `config`) and its downstream scoring pipeline (component `upstream`) are flagged; the dataset is untouched and will not rebuild.
That is the "retrain only what changed" economy; the required upstream dataset is still auto-materialized on a cold runner.

**Expect from the build:** quite possibly a refusal, and that is the system working:

```
gate pr_auc (threshold): PASS - got 0.3109
gate pr_auc (champion):  FAIL - paired bootstrap (1000 resamples):
                         delta lower bound -0.0269 < required 0.005
GATE_FAILED model churn_classifier
build finished [quality_failure]        exit code 2
```

Two lessons to internalize together:

1. A challenger must beat the production champion by `min_delta` at 95% confidence, estimated by a seeded paired bootstrap on the identical pinned test split.
   Even a challenger with a *higher* point metric (say 0.331 vs 0.316) is refused when the confidence bound does not clear the delta; being ahead on test-set noise does not ship (ADR-18).
2. Exit code 2 (quality) is not exit code 1 (broken).
   The PR goes red with a gate table, metrics vs champion, the retrained node list, and a cost estimate in the PR comment; nobody gets paged.

Iterate until the gate passes, then merge.
On merge, `prod_build.yml` rebuilds `state:modified+` against prod, registers to staging, and publishes the manifest as the new baseline.

**Verify:** `echo $?` after the failing build prints 2, and `target/run_results.json` contains the full gate record including the bootstrap bound.

## Step 11 (MLOps): wire the production loops

The scaffold ships the whole loop as GitHub Actions:

| Workflow | Trigger | Runs |
|---|---|---|
| `pr_check.yml` | every PR | parse → compile → state diff vs prod baseline → selective dev build → PR comment |
| `prod_build.yml` | merge to main | prod build of `state:modified+`, then publish the manifest baseline |
| `promote.yml` | `promotions.yml` change or manual dispatch | `mbt promote` |
| `scheduled_retrain.yml` | cron weekly | `mbt build --target prod --select tag:weekly` |
| `scheduled_score.yml` | cron daily | `mbt score --target prod --select tag:daily` |
| `scheduled_monitor.yml` | cron weekly | `mbt monitor --target prod` |

Operational facts to know:

- The prod baseline lives on a dedicated `mbt-state` branch (append-only, published by `scripts/publish_state.sh`), so the incremental loop works from the first merge with zero extra infrastructure.
  Move it to `s3://.../manifests/latest.json` when you outgrow that; `--state` accepts any readable URI.
- Every compiling step in CI passes `--deep-snapshot`, because fresh checkouts rewrite file mtimes and the default mtime snapshot scheme would silently turn the economy loop into a full retrain (ADR-11).
  Any orchestrator with fresh workspaces needs the same flag.
- Set the `MBT_ALERT_WEBHOOK` secret so scheduled failures notify instead of failing silently.

**If you run Airflow**, port only the three scheduled loops (they are time/data events); keep PR check, prod build, and promotion in CI (they are code events).
Use one task per mbt command rather than one per model (mbt is itself the DAG scheduler), run the same pinned image as CI (manifests verify `env_digest`, ADR-19), and map exit codes: 1 retries then pages on-call, 2 never retries (the verdict is deterministic) and notifies the model owner.

**Verify:** push a branch with a one-line spec change and watch `pr_check` retrain only that subgraph and post the comment.

## Step 12 (DS): batch scoring, the serving side

Open `scoring/churn_scoring.yml`; one file is one serving pipeline:

- `model: ref('churn_classifier')` with `stage: production` resolves the champion from the registry at run time, so promotions take effect on the next scheduled run with no coordination.
- `input:` applies the same population filters as training, which keeps the shift monitors honest.
- `monitors:` PSI thresholds for per-feature and score-distribution shift against the champion's training-time baseline.
- `ground_truth:` declares how outcomes join back (label source, `join_key`, `maturity: "14d"`, realized-metric gates).
- `output.path:` where predictions land.

```bash
mbt score
```

**Expect:** the scoring node succeeds and predictions appear under `predictions/churn_scores/`.
A PSI breach or input-check failure would exit 2 and (in CI) fire the alert webhook.

**Verify:** `target/run_results.json` now contains the per-feature PSI values and `rows_scored`.

## Step 13 (both): ground-truth monitoring

```bash
mbt monitor
```

**Expect right after scoring:** `0 matured prediction runs to evaluate`.
That is correct behavior; the spec says labels mature in 14 days, and the ledger guarantees each prediction run is evaluated exactly once.

To see the end state today, simulate the scheduled run two weeks out:

```bash
mbt monitor --anchor <ISO timestamp 15 days from now>
```

**Expect:** realized metrics computed by joining arrived outcomes to stored predictions, e.g. `pr_auc=0.3988  roc_auc=0.7145`, gated against the spec's floor.
A realized-metric gate failure is exit 2, the signal that the production model has decayed and the retrain/promote loop should spin.

**Verify:** run `mbt monitor` again with the same anchor; it evaluates nothing new (exactly-once).

## Step 14 (MLOps): metrics, dashboards, and the two alerting layers

mbt's own gates **enforce**: thresholds live in PR-reviewed specs, a breach blocks registration or fails the run with exit 2.
Your observability stack **observes** what in-band gates structurally cannot: staleness, trends, near-breach headroom, and aggregates.
Keep the two layers separate and never duplicate a spec threshold into a dashboard alert; export the signed *margin* instead so thresholds stay single-sourced in git.

The integration contract is `target/run_results.json` + `target/manifest.json`.
After every prod run (final Airflow task with `trigger_rule=all_done`, or a CI step), push them to a Prometheus Pushgateway as gauges; group per `(job, project, target, command, node)` so selective runs replace only what they touched.
A workable metric spec:

| Metric | Meaning |
|---|---|
| `mbt_node_success` / `mbt_node_duration_seconds` | per-node health and cost |
| `mbt_test_metric{metric=}` | train-time metrics on the pinned test split |
| `mbt_realized_metric{metric=}` | matured ground-truth metrics from `mbt monitor` |
| `mbt_gate_passed` / `mbt_gate_margin{kind=threshold\|champion\|ground_truth}` | gate outcomes and signed headroom |
| `mbt_shift_value` / `mbt_shift_threshold` | PSI per feature and for the score distribution |
| `push_time_seconds` | free per group; powers staleness alerts |

Label every series with the spec's `owner:` field and route Alertmanager on it: quality alerts to the owning DS team's channel, hard errors and staleness to MLOps on-call.
The four rules that matter most:

1. `mbt_gate_passed == 0` - a gate breached (notification layer; mbt already enforced).
2. `time() - push_time_seconds{command=~"score|monitor"} > 8*86400` - the schedule itself silently died (no in-band mechanism can catch this).
3. `mbt_gate_margin >= 0 and mbt_gate_margin < 0.02` - early warning before a breach.
4. `mbt_shift_value >= mbt_shift_threshold` - distribution shift at the spec threshold.

**Verify end to end:** push one gate-failed run and watch the alert arrive in the team channel with the model's owner on it.

## Day-2 cheat sheet

```bash
mbt build --select churn_classifier            # one model + its data
mbt build --select tag:weekly                  # the scheduled retraining set
mbt build --select state:modified+ --state <prod manifest>   # only what changed
mbt test                                       # checks + gates, never trains
mbt evaluate --model churn_classifier --stage production --gates   # champion decay check
mbt score --select tag:daily                   # scheduled batch scoring
mbt monitor                                    # evaluate matured predictions
mbt clean --artifacts-older-than 30d           # GC (champions always survive)
```

| Exit code | Meaning | Typical responder |
|---|---|---|
| 0 | success | nobody |
| 1 | hard error (bug, infra, unloadable champion) | MLOps on-call |
| 2 | quality failure (gate, check, test, shift, realized metric) | model owner (DS) |

## Where to read more

- [Quickstart](quickstart.md) and [Concepts](concepts.md) for the resource model.
- [GitOps & CI](gitops.md) for the CI loop, state storage, and alerting hooks.
- [Troubleshooting](troubleshooting.md) when a run fails or retrains more than you expected; every entry was captured from a real reproduction.
- The ADRs for why things are the way they are: [snapshots](adr/0011-snapshot-mtime-listing.md), [champion bootstrap](adr/0018-paired-bootstrap-champion-gates.md), [env digests](adr/0019-env-freeze-digest-and-manifest-verification.md), [scoring](adr/0020-scoring-resource-and-runtime-champion.md), and [the ground-truth ledger](adr/0021-prediction-store-and-ground-truth-ledger.md).
