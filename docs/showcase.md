# Showcase: the full lifecycle on a dockerized platform stack

The [showcase](https://github.com/satrijandi/mbt/tree/main/examples/showcase) (`examples/showcase`) is a laptop-runnable reference environment that demonstrates mbt end to end on real services instead of local stand-ins.
Where the [tutorial](tutorial.md) walks a team through the concepts, the showcase is the lab where the whole loop actually runs: build, promote, score, monitor, alert.

| Service | Role |
|---|---|
| SeaweedFS | S3-compatible object store: the data lake of gold-layer feature tables (read via `s3a://`) and the MLflow artifact store; its filer UI is published so humans can browse the lake (the raw S3 port takes signed requests only) |
| MLflow (HTTP server) | Tracking + model registry; champion source of truth |
| Spark standalone cluster | Dataset pushdown; H2O AutoML training inside the executors via Sparkling Water |
| JupyterLab | The DS workbench; its terminal runs the same `mbt` as everything else |
| Gitea + Woodpecker | The CI loop: state-diff slim PR checks with update-in-place build-report comments, merge-time economy builds publishing the `mbt-state` baseline, exit-code-classified alerts, and protected GitOps promotion (branch protection + CODEOWNERS on `promotions.yml`) |
| Zot | OCI registry: the digest-pinned deployable unit (runner image + project + compiled manifest) and oras-pushed provenance artifacts (`manifest.json` + `run_results.json` per source sha) |
| Airflow + git-sync | Scheduling/CD: git-sync reconciles the Gitea `deploy` repo (digest pin + DAGs); retrain/score/monitor DAGs run the pinned unit with exit-code routing (quality verdicts never retry) |
| Pushgateway + Prometheus + Grafana | The observability spec from tutorial step 14, implemented verbatim: gauges, dashboards, and the four canonical alert rules |
| k3d + ArgoCD (optional, `MBT_LIVE_SHOWCASE_K3D=1`) | CD fidelity: ArgoCD core syncs the same deploy repo into CronJobs on a k3d cluster attached to the compose network, pulling from zot over insecure HTTP |

Everything mbt-related runs inside one runner image (Jupyter kernel, Spark master and worker, every `mbt` invocation), which makes ADR-19 `env_digest` verification hold by construction.

## Run it

Requirements: docker with ~10GB free RAM, `uv`, and a checkout of this repository.

```bash
cd examples/showcase
make up        # build the runner image (first time ~10 min), boot, seed the lake
make demo      # the whole lifecycle, narrated: build dev -> build prod -> promote -> score -> monitor
make ci        # seed Gitea + Woodpecker: org, churn repo, OAuth app, repo activation
make down      # stop and remove containers, volumes, and the network (the workspace survives)
make clean     # down, then also remove the workspace (~/.cache/mbt-showcase/workspace)
```

`make up` prints every UI URL with its login (JupyterLab, MLflow, Spark, the SeaweedFS lake browser, Grafana, Prometheus, Gitea, Woodpecker, Zot, Airflow).
`make ci` seeds the CI loop headlessly; then log into Woodpecker from the browser with the Gitea account (the compose file gives Woodpecker split-horizon URLs so the OAuth dance works from the host), clone the printed repo URL, and open a PR - Woodpecker runs the state-diff check and posts the mbt build report comment, and merges to main bake the deployable unit, pin its digest in the deploy repo, and feed the Airflow DAGs via git-sync.
`make inject-drift` poisons the scoring batch: `mbt score` exits 2, the pushed breach fires the `MbtShiftBreach` alert, and `make score` recovers.
`make score` and `make monitor` also work standalone: they rerun just the daily scoring stage or just the ground-truth monitoring stage, with the same pinned anchors as the demo.
`make monthly` runs the second, cluster-free cadence: the `tag:monthly` churn pipeline trains, promotes, and scores entirely on the DuckDB batch plane over the synced S3 parquet lake.
`make wide` runs the third, wide batch-monthly cadence (ADR-22): a monthly population spine with an entity crosswalk, three feature histories joined by different keys, matured labels keyed by each cohort's own `inference_date`, a ds-helper LightGBM selection funnel committed as a reviewable diff, DS-declared categorical codes cast by a shared hooks file, sparkling H2O AutoML on the selected columns, and Evidently feature-stability gates (exit 2 blocks promotion; every scored batch re-checks against the exported baseline) beside mbt's own enforcing monitors.
The [showcase README](https://github.com/satrijandi/mbt/blob/main/examples/showcase/README.md) is the full runbook, including the RAM knobs and the documented deviations from the scaffold defaults (snapshot scheme, local scoring plane, PR-scoped registry).
The design of record is [DESIGN.md](https://github.com/satrijandi/mbt/blob/main/examples/showcase/DESIGN.md); phases P1-P6 of its plan are implemented (the k3d/ArgoCD fidelity profile local-only behind its own gate), and a P7 Snowflake warehouse variant is scoped but deliberately parked.

## What it proves

The demo narrative exercises mbt's differentiators against real service boundaries, not mocks:

- Spark reads the lake over the real S3 API and registers models to MLflow over HTTP, with MOJO artifacts landing in the S3 artifact store.
- State-diff slim CI on a real forge: a one-gate-edit PR retrains exactly the edited model (fresh clones cause no dataset churn thanks to URI snapshot tokens), a no-change merge trains nothing yet republishes an identical baseline, and the PR gets an update-in-place build-report comment.
- Exit-code fidelity through CI and the scheduler: Woodpecker collapses failures to pass/fail, so a wrapper records mbt's 1-vs-2 verdict and classifies alerts - a gate failure notifies the spec's owner, a hard error pages on-call; in Airflow, quality verdicts fail on try 1 with no retry while hard errors consume a retry first.
- Gate-verified promotion via `promotions.yml`: pinned-version replays are idempotent, unpinned replays are refused, and the file itself is governed (branch protection + CODEOWNERS; unauthorized direct pushes bounce).
- Manifest-verified reproducible execution (ADR-19): the deployable unit baked into Zot reproduces its own manifest (`mbt run --manifest`; xgboost bit-exact, H2O within its documented 0.02 tier), refuses a tampered environment with exit 1, and its oras provenance artifact is byte-identical to the published `mbt-state` baseline - and secret-free.
- CD that promotion never touches: two scheduled score runs straddling a promotion serve different champions while the deploy repo HEAD and the pinned image digest stay byte-identical.
- Run-time champion resolution (ADR-20): a promotion changes the next scoring run with zero redeploy.
- Adapter portability on one lake: the monthly cadence trains, scores, and ground-truth-monitors the same project's `tag:monthly` pipeline on the DuckDB local adapter - no cluster - while the daily/weekly cadences use Spark, from the same `sources.yml`.
- Real training-set topology (ADR-22 + [naming conventions](naming-conventions.md)): the wide cadence joins every table on the one uniform `inference_date` key through a population spine carrying the `customer_id`-to-`safe_id` entity crosswalk (feature balances describe the previous day, recorded in the spine's informational `as_of_date` column), and joins matured labels keyed by each cohort's own `inference_date` (the gold-layer label contract; a raw observation-dated feed would use `time_offset` instead), panel-samples by `customer_id` with pushdown hash sampling, prunes ~66 joined columns to a committed funnel-selected include list, and scores the newest cohort through the same multi-table shape - identical declarations on the Spark and DuckDB planes.
- Prediction-store idempotency (ADR-21): same-anchor re-runs overwrite one `run_key`, new anchors partition.
- Ground-truth monitoring: realized metrics are evaluated exactly once per prediction run, and a realized-gate breach exits 2, never 1.
- Observability: `run_results.json` becomes Pushgateway gauges, and injected shift makes the provisioned Prometheus rule actually fire.

## Who defines what: the DS / MLOps seam

The showcase project follows the [naming conventions](naming-conventions.md) for its temporal and entity columns, and is split along the same line the [tutorial](tutorial.md) teaches: the DS owns everything that defines the model as an experiment, the MLOps engineer owns everything that defines where and how it runs, and every handoff between them is a reviewable YAML diff.
The wide batch-monthly cadence makes the split concrete:

| Decision | Owner | Where |
|---|---|---|
| Training population, the matured-label contract keyed by `inference_date`, per-table join keys | DS | `project/datasets/wide_churn_training.yml` |
| ID columns: `sample_key: customer_id` (panel sampling) and entity ids as non-features | DS | dataset `sample_key` + model `features.exclude` |
| The split date column and exact train/test cohort boundaries (ISO ranges) | DS | the dataset's `split:` block |
| Ignored columns the selection funnel must never offer, including the time-anchored `tenure_months` | DS | the model's `features.exclude` (honored by `select_features.py`) |
| Numeric-coded categorical features | DS | `CATEGORICAL_CODES` in `project/models/wide_hooks.py` |
| Algorithm, AutoML budget, seed, metrics, gate floors, registration target | DS | `project/models/churn_wide_*.yml` |
| Shift-monitor thresholds, ground-truth maturity and realized gates, the Evidently `--max-drift-share` policy | DS | `project/scoring/wide_retention_scoring.yml` + the `make wide` gate flags |
| The selected feature list itself | DS | rerunning `scripts/select_features.py`; the rewritten include list is the PR diff reviewers see |
| Targets: Spark master, s3a endpoint and credentials, MLflow URIs, artifact store, per-environment `sample_fraction` defaults, the sparkling backend var | MLOps | `project/profiles.yml` (specs stay target-portable) |
| The CI loop: slim PR checks, merge-time prod builds, the `mbt-state` baseline, gate-verified promotion governance | MLOps | `project/.woodpecker/`, protected `promotions.yml` |
| The deployment plane: runner-image version matrix, deployable-unit bake, digest pin, DAGs including `mbt_score_wide`'s monthly cron and its exit-code routing | MLOps | `images/runner/`, `deploy/` |
| Operations: lake seeding and sync, metric push, the shift-breach alert, the monitor cadence | MLOps | `bootstrap/`, the observability profile |

Three handoffs keep the seam clean.
Promotion is a registry event: the next scheduled run resolves the new champion, and CD redeploys nothing (ADR-20).
Failures route by exit code: a quality verdict (exit 2) is deterministic, fails without retries, and notifies the model's DS owner, while a hard error (exit 1) is retried and then pages on-call.
And every DS decision - selected features, excluded columns, thresholds, seeds - lives in committed YAML, so the MLOps-owned pipelines can enforce it without ever needing to understand the model.

## The live test tier

Every claim above is pinned by an opt-in E2E tier that boots its own isolated compose project on ephemeral ports and tears everything down:

```bash
MBT_LIVE_SHOWCASE=1 uv run pytest -q -m live_showcase --timeout 3600 -rA
```

It follows the live-tier double gate: skipped everywhere unless `MBT_LIVE_SHOWCASE=1`, and once opted in, a missing docker fails loudly instead of skipping.
The k3d/ArgoCD module carries one more gate (`MBT_LIVE_SHOWCASE_K3D=1`, needs `k3d` and `kubectl`) and stays local-only.
CI runs the rest nightly via `.github/workflows/live.yml`, alongside the live Snowflake tier.
