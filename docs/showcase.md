# Showcase: the full lifecycle on a dockerized platform stack

The [showcase](https://github.com/satrijandi/mbt/tree/main/examples/showcase) (`examples/showcase`) is a laptop-runnable reference environment that demonstrates mbt end to end on real services instead of local stand-ins.
Where the [tutorial](tutorial.md) walks a team through the concepts, the showcase is the lab where the whole loop actually runs: build, promote, score, monitor, alert.

Its data is deliberately the simplest realistic shape: ONE big table that already lives in the data lake, holding the population, every feature, and the label.

| Service | Role |
|---|---|
| SeaweedFS | S3-compatible object store: the data lake holding the one table (read via `s3a://`) and mbt's artifact store (the model binaries MLflow's registry points at); its filer UI is published so humans can browse the lake (the raw S3 port takes signed requests only) |
| MLflow (HTTP server) | Tracking + model registry; champion source of truth |
| Spark standalone cluster | Dataset pushdown; H2O AutoML training inside the executors via Sparkling Water |
| JupyterLab | The DS workbench; its terminal runs the same `mbt` as everything else, and the committed `project/notebooks/ds_inner_loop.ipynb` walks the DS inner loop (explore the table, build, analyze, sampled what-ifs) with the model staying in reviewed YAML |
| Gitea + Woodpecker | The CI loop: state-diff slim PR checks with update-in-place build-report comments, merge-time economy builds publishing the `mbt-state` baseline, exit-code-classified alerts, and protected GitOps promotion (branch protection + CODEOWNERS on `promotions.yml`) |
| Zot | OCI registry: the digest-pinned deployable unit (runner image + project + compiled manifest) and oras-pushed provenance artifacts (`manifest.json` + `run_results.json` per source sha) |
| Airflow + git-sync | Scheduling/CD: git-sync reconciles the Gitea `deploy` repo (digest pin + DAGs); retrain/score/monitor DAGs run the pinned unit with exit-code routing (quality verdicts never retry) |
| Pushgateway + Prometheus + Grafana | The observability spec from [tutorial step 14](tutorial.md#step-14-mlops-metrics-dashboards-and-the-two-alerting-layers), implemented verbatim: gauges, dashboards, and the four canonical alert rules |
| k3d + ArgoCD (optional, `MBT_LIVE_SHOWCASE_K3D=1`) | CD fidelity: ArgoCD core syncs the same deploy repo into CronJobs on a k3d cluster attached to the compose network, pulling from zot over insecure HTTP |

Everything mbt-related runs inside one runner image (Jupyter kernel, Spark master and worker, every `mbt` invocation), which makes ADR-19 `env_digest` verification hold by construction.

## The one table

`s3://mbt-lake/churn_panel` has one row per customer per month-start `inference_date`:

- `customer_id` and `inference_date`, the natural key;
- `is_active`, the population - churned customers keep their rows, marked inactive;
- 16 named features, among them a numeric-coded categorical (`contract_code`) whose effect is deliberately non-monotone;
- hundreds of columns no model uses, at the realistic scale - the width a real gold-layer table carries;
- `is_churn`, churned during the following month, NULL where nobody knows yet: on inactive rows, and on the newest cohort until its outcome window closes.

The project reads it three ways, and the three differ only in which rows they take:

- the **training set** (`datasets/churn_training.yml`) filters to the active population and splits the labelled cohorts by time, with both windows ending before the newest cohort;
- the **scoring input** reads the same table's newest cohort, the one still waiting on its outcomes;
- the **ground truth** reads that cohort's `is_churn` from the same table once it lands, joined on `(customer_id, inference_date)` - the key alone would match every cohort a customer was ever in.

The models name the 16 columns they train on, so the rest of the table's width never reaches them.
[ADR-29](adr/0029-single-relation-datasets.md) recommends a label-free serving twin of the training relation; the showcase reads the labelled table for all three on purpose, and it is safe because a model's target is never among its features, the prediction store never copies the label, and `mbt monitor` treats a NULL label as an outcome that has not landed yet.

Nothing is committed: `make seed` generates the table deterministically straight into the lake, and `make seed SCALE=huge` writes the realistic shape through the same code.
Only the newest cohort changes after that, and always as a partition rewrite under new file names (the Spark adapter pins an object-store table by its file listing, so an in-place rewrite would change data under a pinned snapshot): `make outcomes` lands its labels, `make inject-drift` poisons its features, and `make reset` restores it byte for byte.

## Run it

Requirements: docker with ~10GB of RAM to spare, `make`, `rsync`, `uv`, and a checkout of this repository.

```bash
cd examples/showcase
make up        # build the runner image (first build 10-15 min), boot, generate the lake table
make demo      # the whole lifecycle, narrated: build dev -> build prod -> promote -> score -> outcomes -> monitor
make ci        # seed Gitea + Woodpecker: org, churn repo, OAuth app, repo activation
make down      # stop and remove containers, volumes, and the network (the workspace survives)
make clean     # down, then also remove the workspace (~/.cache/mbt-showcase/workspace)
```

`make up` prints every UI URL with its login (JupyterLab, MLflow, Spark, the SeaweedFS lake browser, Grafana, Prometheus, Gitea, Woodpecker, Zot, Airflow).
Start where a data scientist would: open JupyterLab and run `project/notebooks/ds_inner_loop.ipynb` top to bottom - it looks at the lake table, builds on the dev target, analyzes the run artifacts, and experiments on a hash-sampled slice without touching the committed specs; the notebook ends where the PR begins, and the make targets below are the platform side of the same story.
`make ci` seeds the CI loop headlessly; then log into Woodpecker from the browser with the Gitea account (the compose file gives Woodpecker split-horizon URLs so the OAuth dance works from the host), clone the printed repo URL, and open a PR - Woodpecker runs the state-diff check and posts the mbt build report comment, and merges to main bake the deployable unit, pin its digest in the deploy repo, and feed the Airflow DAGs via git-sync.
`make score`, `make outcomes` and `make monitor` also work standalone, with the same pinned anchors as the demo.
`make inject-drift` poisons the newest cohort: `mbt score` exits 2, the pushed breach fires the `MbtShiftBreach` alert in Prometheus, and `make reset score` recovers.
The [showcase README](https://github.com/satrijandi/mbt/blob/main/examples/showcase/README.md) is the full runbook, including the scale knob, the RAM budget, and the documented deviations from the scaffold defaults (snapshot scheme, the cluster-free scoring target, PR-scoped registry).
The design of record is [DESIGN.md](https://github.com/satrijandi/mbt/blob/main/examples/showcase/DESIGN.md).

## What it proves

The demo narrative exercises mbt's differentiators against real service boundaries, not mocks:

- Spark reads the lake over the real S3 API and registers models to MLflow over HTTP, with MOJO artifacts landing in the S3 artifact store.
- State-diff slim CI on a real forge: a one-gate-edit PR retrains exactly the edited model (fresh clones cause no dataset churn thanks to URI snapshot tokens), a no-change merge trains nothing yet republishes an identical baseline, and the PR gets an update-in-place build-report comment.
- Exit-code fidelity through CI and the scheduler: Woodpecker collapses failures to pass/fail, so a wrapper records mbt's 1-vs-2 verdict and classifies alerts - a gate failure notifies the spec's owner, a hard error pages on-call; in Airflow, quality verdicts fail on try 1 with no retry while hard errors consume a retry first.
- Gate-verified promotion via `promotions.yml`: pinned-version replays are idempotent, unpinned replays are refused, and the file itself is governed (branch protection + CODEOWNERS; unauthorized direct pushes bounce).
- Manifest-verified reproducible execution (ADR-19): the deployable unit baked into Zot reproduces its own manifest (`mbt run --manifest`; xgboost bit-exact, H2O within its documented 0.02 tier), refuses a tampered environment with exit 1, and its oras provenance artifact is byte-identical to the published `mbt-state` baseline - and secret-free.
- CD that promotion never touches: two scheduled score runs straddling a promotion serve different champions while the deploy repo HEAD and the pinned image digest stay byte-identical.
- Run-time champion resolution (ADR-20): a promotion changes the next scoring run with zero redeploy.
- Prediction-store idempotency (ADR-21): same-anchor re-runs overwrite one `run_key`, new anchors partition.
- Ground-truth monitoring from the same table the model trained on: while the scored cohort's labels are NULL the run waits, once they land it is evaluated exactly once, and a realized-gate breach exits 2, never 1.
- Observability: `run_results.json` becomes Pushgateway gauges, and injected shift makes the provisioned Prometheus rule actually fire (the rules stop at Prometheus: no Alertmanager or Grafana contact point is provisioned, so wiring notifications to people is left to your own alerting stack).

## Who defines what: the DS / MLOps seam

The showcase project is split along the same line the [tutorial](tutorial.md) teaches: the DS owns everything that defines the model as an experiment, the MLOps engineer owns everything that defines where and how it runs, and every handoff between them is a reviewable YAML diff.

| Decision | Owner | Where |
|---|---|---|
| The table itself: its grain, the population flag, the label definition, when a cohort's outcomes land | the data team that owns the lake | upstream of mbt (in the showcase, `bootstrap/churn_panel.py` stands in for it) |
| Which rows train: the population filter, sampling key, split column and exact cohort boundaries, checks | DS | `project/datasets/churn_training.yml` |
| Which columns train: the 16-feature include list and the declared categoricals | DS | `project/models/churn_*.yml` `features:` |
| Algorithm, AutoML budget, seed, metrics, gate floors, registration target | DS | `project/models/churn_*.yml` |
| Which rows score, shift-monitor thresholds, ground-truth join, maturity and realized gates | DS | `project/scoring/retention_scoring.yml` |
| Targets: Spark master, s3a endpoint and credentials, MLflow URIs, artifact store, per-environment `sample_fraction` defaults, the sparkling backend var | MLOps | `project/profiles.yml` (specs stay target-portable) |
| The CI loop: slim PR checks, merge-time prod builds, the `mbt-state` baseline, gate-verified promotion governance | MLOps | `project/.woodpecker/`, protected `promotions.yml` |
| The deployment plane: runner-image version matrix, deployable-unit bake, digest pin, DAGs and their exit-code routing | MLOps | `images/runner/`, `deploy/` |
| Operations: metric push, the shift-breach alert, the monitor cadence | MLOps | the observability profile, `deploy/dags/` |

Three handoffs keep the seam clean.
Promotion is a registry event: the next scheduled run resolves the new champion, and CD redeploys nothing (ADR-20).
Failures route by exit code: a quality verdict (exit 2) is deterministic, fails without retries, and notifies the model's DS owner, while a hard error (exit 1) is retried and then pages on-call.
And every DS decision - features, thresholds, seeds, windows - lives in committed YAML, so the MLOps-owned pipelines can enforce it without ever needing to understand the model.

## The live test tier

Every claim above is pinned by an opt-in E2E tier that boots its own isolated compose project on ephemeral ports and tears everything down:

```bash
MBT_LIVE_SHOWCASE=1 uv run pytest -q -m live_showcase --timeout 3600 -rA
```

It follows the live-tier double gate: skipped everywhere unless `MBT_LIVE_SHOWCASE=1`, and once opted in, a missing docker fails loudly instead of skipping.
Two modules carry one more gate and run in their own invocation: the runbook module (`MBT_LIVE_SHOWCASE_MAKE=1`) drives the README's `make` targets on a second stack, and the k3d/ArgoCD module (`MBT_LIVE_SHOWCASE_K3D=1`, needs `k3d` and `kubectl`) stays local-only.
The [showcase README](https://github.com/satrijandi/mbt/blob/main/examples/showcase/README.md#the-e2e-test-tier-the-honest-version-of-the-demo) gives each tier's wall time and lists every module and what it asserts.

The gated tier cannot see a broken showcase spec from the ordinary test battery, so hermetic modules guard it in the fast suite: every gated module keeps its gate, the runner image's hand pins agree with the declared metadata, and the table's generator and the project's use of it hold - every node reads the one table, ground truth joins on key and cohort, training never reaches the open cohort, and every string feature is declared categorical.
