# mbt showcase: the full lifecycle on a dockerized platform stack

A laptop-runnable reference environment that demonstrates mbt end to end on real services instead of local stand-ins.
The data is ONE big table that already lives in the data lake - `s3://mbt-lake/churn_panel`, one row per customer per month, holding the population flag, every feature, and the label - and everything the showcase trains, scores, and monitors reads that table.
Around it: SeaweedFS is the S3 data lake and artifact store, MLflow (over HTTP) is the tracking server and model registry, a standalone Spark cluster does dataset pushdown and in-executor H2O (sparkling) AutoML training, JupyterLab is the DS workbench, Gitea + Woodpecker run the state-diff CI loop with PR comments and gate-classified alerts, Zot holds the digest-pinned deployable unit and its oras provenance artifacts, Airflow (fed by git-sync from the Gitea `deploy` repo) schedules retrain/score/monitor runs of that unit, and Prometheus + Grafana observe production scoring through the Pushgateway spec documented in the tutorial.

The design of record is [DESIGN.md](DESIGN.md).

Everything mbt-related runs inside ONE runner image (Jupyter kernel, Spark master/worker, MLflow server, every `mbt` invocation), which makes ADR-19 `env_digest` verification hold by construction.

## The one table

```text
s3://mbt-lake/churn_panel/<inference_date>-<state>-<chunk>.parquet

customer_id, inference_date   the natural key: one row per customer per month-start cohort
is_active                     the population - churned customers keep their rows, marked inactive
16 named features             demographics, logins, transactions (contract_code is a coded categorical)
f0000 .. fNNNN                columns no model uses - the "huge" knob
is_churn                      churned the following month; NULL where nobody knows yet
```

The project reads it three ways, and they differ only in which rows they take:

| Reader | Spec | Rows |
|---|---|---|
| Training set | `datasets/churn_training.yml` | active customers in the labelled cohorts (2025-07 .. 2026-05), split by time |
| Scoring input | `scoring/retention_scoring.yml` `input:` | active customers in the newest cohort (2026-06-01), whose label is still NULL |
| Ground truth | `scoring/retention_scoring.yml` `ground_truth:` | the same cohort's `is_churn`, once its outcomes land, joined on `(customer_id, inference_date)` |

The models name the 16 columns they train on; the rest of the width never reaches them.
Nothing is committed under the showcase: `make seed` generates the table deterministically (`bootstrap/churn_panel.py`) straight into the lake, and `make seed SCALE=huge` writes the realistic shape through the same code.

Only the newest cohort ever changes after seeding, and always as a partition rewrite under new file names: `make outcomes` lands its labels (a month has passed), `make inject-drift` poisons its features, and `make reset` puts it back byte for byte.

## Run it

Requirements: docker with ~10GB of RAM to spare (see [Knobs](#knobs) for the measured budget), `make`, `rsync`, `uv`, and this checkout.

```bash
cd examples/showcase
make up        # build the runner image (first build 10-15 min), boot, generate the lake table
make demo      # the whole lifecycle, narrated (build dev -> build prod -> promote -> score -> outcomes -> monitor)
make ci        # seed Gitea + Woodpecker + the deploy repo: org, repos, OAuth app, activation
make down      # stop and remove containers, volumes, and the network (the workspace survives)
make clean     # down, then also remove the workspace (~/.cache/mbt-showcase/workspace)
```

After `make ci`, pushing to main runs prod-build end to end: economy build, `mbt-state` baseline publish, deployable-unit bake to Zot (digest-pinned in the deploy repo), and oras provenance push; git-sync feeds the deploy repo's DAGs into Airflow, where `mbt_retrain`/`mbt_score`/`mbt_monitor` run the pinned unit on demand.

`make up` prints every UI URL with its login (`make urls` re-prints them); a bare `make` lists the targets.

After `make up`, start where a data scientist would: open JupyterLab (http://localhost:8899), open `project/notebooks/ds_inner_loop.ipynb`, and run it top to bottom.
It looks at the lake table (reading only the columns it asks about), walks the YAML that IS the model, builds on the dev target, analyzes the run artifacts, and experiments on a hash-sampled slice with a one-off `--vars` override - the committed specs stay clean throughout.
The notebook ends where the PR begins; `make demo` below is the platform side of that same story.

After `make demo`, look at:

- **MLflow**: registered `churn_automl` versions, the `production` alias set by the promotion, and one experiment named after the project, `churn_lake` (ADR-28). It holds training runs only - named `<run_id>-<model>`, carrying metrics, `mbt.*` provenance tags, and an `inference_config.json` document describing exactly what was trained. `mbt score` and `mbt monitor` log nothing here; their records are in the prediction store, read with `mbt predictions ls` / `show`.
- **Lake browser** (SeaweedFS filer UI, no login): the table under `/buckets/mbt-lake/churn_panel/` and the model artifacts mbt stores (the MOJOs and model files MLflow only points at) under `/buckets/mbt-artifacts/`.
  The raw S3 API port accepts signed requests only (`mbtadmin`/`mbtsecret`), so a bare browser GET there returns `AccessDenied` by design - browse through the filer UI instead.
- **Grafana** (`admin`/`admin`): the "mbt Model Health" dashboard - gate margins, realized metrics, shift-vs-threshold, node durations.
- **Predictions** on disk under `~/.cache/mbt-showcase/workspace/predictions/retention_scores/<run_key>/`.
- The demo's two monitor steps: the first runs before the cohort's outcomes exist and evaluates nothing ("no matured labels joined ... will retry"), the second runs after the demo lands the outcomes (the same step as `make outcomes`) and evaluates the run exactly once.
- `make inject-drift` then Grafana/Prometheus: the newest cohort is poisoned, `mbt score` exits 2 (mbt enforces), and the pushed breach fires the `MbtShiftBreach` alert in Prometheus (observability observes; no Alertmanager is wired, so the alert is visible rather than delivered). `make reset score` recovers.

`make score`, `make outcomes` and `make monitor` also work standalone, with the same pinned anchors as the demo: score the newest cohort with the production champion, land its outcomes, evaluate them.

## Scale

The default table is small so every recipe stays quick; `SCALE=huge` reseeds it at a realistic width and length, through exactly the same code.

| `SCALE` | Customers | Noise columns | Rows | Columns | Parquet in the lake |
|---|---|---|---|---|---|
| `default` | 3,000 | 48 | 47,880 | 68 | 14MB |
| `huge` | 100,000 | 300 | 1,596,000 | 320 | 2.9GB |

`make seed SCALE=huge` then `make demo` is the same lifecycle on the big table: the seed takes about a minute and the demo about 29 minutes on the machine in [Knobs](#knobs), against a few minutes at the default.
`huge` is the biggest shape the stack is sized for - SeaweedFS's capacity is pinned at 6.4GB and the artifact bucket shares it - so a production-sized table (millions of rows by thousands of columns) wants a bigger lake, not a bigger knob.
Two levers keep it tractable, and both are the spec's, not the showcase's:
the models read 16 named columns, so the noise width costs I/O but never training time, and `sample_key: customer_id` hash-samples whole customers in the source query - `--vars '{sample_fraction: 0.1}'` trains on a coherent tenth, and smaller fractions are subsets of larger ones.
What does grow with width is every full read of the table: mbt materializes every column of the relation into a dataset's splits and selects the models' columns afterwards, and `mbt monitor` reads the whole table to take its join keys and label - at `huge`, each monitor run spends about four minutes on that read.
`--customers`, `--noise-columns` and `--rows-per-file` on `bootstrap/churn_panel.py seed` set any shape in between.

## The CI loop (make ci)

`make ci` seeds Gitea with the `mbt-showcase/churn` repo (the project source, `.woodpecker/` pipelines included), creates the OAuth app, re-ups Woodpecker with the real credentials, and activates the repo - all headless (the first Woodpecker API token is minted by a scripted OAuth dance against the host-published ports - the exact flow a browser performs, thanks to Woodpecker's split-horizon URL config in the compose file).
Then work like a user would: log into Woodpecker at `http://localhost:8305` with the Gitea account (`mbtops`/`mbtops-showcase-password`), clone `http://localhost:3305/mbt-showcase/churn`, push to main (prod-build trains the state-modified subgraph on the shared registry and republishes the `mbt-state` baseline), or open a PR (pr-check lints promotions.yml, state-diffs against the published baseline, slim-builds only `state:modified+` on the throwaway `ci` target, and posts the update-in-place `mbt build report` comment).
One known cosmetic seam: Woodpecker's "repository" deep-links point at the in-network Gitea URL (`gitea:3000`), because that URL must stay resolvable by the CI step containers - use the printed `localhost` Gitea URL instead.
Exit-code fidelity survives Woodpecker's binary pass/fail: `scripts/run_mbt.sh` records mbt's 1-vs-2 verdict in `target/ci_exit_class` and classifies the alert it curls to webhook-sink - exit 2 (quality) notifies the failing spec's `owner`, anything else pages on-call.

## The E2E test tier (the honest version of the demo)

Opt-in, following the live-tier double gate: skipped everywhere unless `MBT_LIVE_SHOWCASE=1`; once opted in, a missing docker fails loudly instead of skipping.
Three invocations cover it, from the repo root and one after another - each boots a full stack of its own, and two at once exceed the RAM budget:

```bash
# the main tier: one shared session stack, every module below except the extra-gated ones
MBT_LIVE_SHOWCASE=1 uv run pytest -q -m live_showcase --timeout 3600 -rA

# the runbook tier: the README's make targets, on a second isolated stack
MBT_LIVE_SHOWCASE=1 MBT_LIVE_SHOWCASE_MAKE=1 uv run pytest -q tests/test_showcase_make.py --timeout 3600 -rA

# the k3d + ArgoCD tier (needs k3d and kubectl; local-only)
MBT_LIVE_SHOWCASE=1 MBT_LIVE_SHOWCASE_K3D=1 uv run pytest -q tests/test_showcase_k3d.py --timeout 3600 -rA
```

| Tier | Gate | Wall time (10-core laptop, image already built) | Runs in CI |
|---|---|---|---|
| main | `MBT_LIVE_SHOWCASE=1` | ~21 min | nightly, `live.yml` |
| runbook | + `MBT_LIVE_SHOWCASE_MAKE=1` | ~6 min | nightly, after the main tier |
| k3d + ArgoCD | + `MBT_LIVE_SHOWCASE_K3D=1` | 6-8 min | never - this is its only coverage |

The first run builds the runner image (10-15 minutes) and the harness rebuilds it whenever package sources, `uv.lock` or the image inputs have moved since, so budget for that on top.
`MBT_SHOWCASE_KEEP=1` leaves a session's stack running for a post-mortem instead of tearing it down.

The modules (repo-root `tests/`) share one isolated compose project per session, on ephemeral ports with a tmp workspace, torn down with its volumes at the end:

- `test_showcase_infra.py` - services healthy, real S3 round-trip, the lake holds exactly the one table (browsable from the host through the filer UI; the raw S3 port correctly refuses unsigned requests), `mbt` runs in the image, the h2o-client == pysparkling-embedded-H2O version probe (an exact match is required by H2O; the image pins `h2o==3.46.0.6` for this), and a stopped registry turns `mbt build` into exit 1, never 2.
- `test_showcase_ci.py` - the Woodpecker loop driven exactly as a user would (git pushes and PRs against Gitea): the browser OAuth login works from the host (driven headlessly for the non-admin persona, first consent included), the first push honors `fetch_state.sh` exit 3 and full-builds (and bakes the first deployable unit), a no-change merge trains nothing yet republishes an identical baseline (and re-bakes nothing - the digest pin is untouched), a one-gate-edit PR slim-builds exactly the edited model (no dataset churn across fresh clones - URI snapshot stability), merging it retrains only that model (and pins a fresh unit), an impossible gate fails the pipeline with mbt's exit 2 classified as a quality failure (the PR comment shows `gate_failed`, the shared registry is untouched, webhook-sink records exactly one owner-classified alert), and promotions.yml is governed: branch protection + CODEOWNERS reject the unauthorized direct push, the owner-approved merge runs the promote pipeline, and the production alias moves with the deploy repo byte-identical.
- `test_showcase_provenance.py` - the deployable unit reproduces: the oras provenance artifact is byte-identical to the mbt-state baseline of the same run and secret-free, `mbt run --manifest` inside the pulled unit reproduces metrics (xgboost exactly, H2O within its documented 0.02 tier), and a tampered environment is refused with exit 1 (`--allow-env-mismatch` downgrades to a warning).
- `test_showcase_scheduling.py` - Airflow runs the pinned unit: the retrain DAG builds on the prod target (cluster pushdown from a scheduled container), two score DAG runs straddling a promotion serve different champions while the deploy repo HEAD and digest stay byte-identical (the ADR-20 inversion), and monitor exit codes route correctly (a realized-gate breach fails on try 1 with no retry; a hard error consumes a retry).
- `test_showcase_k3d.py` (extra gate: `MBT_LIVE_SHOWCASE_K3D=1`, local-only) - ArgoCD core in a k3d cluster on the compose network syncs the deploy repo's `k8s/`: the CronJob lands pinned to the baked digest, an insecure-HTTP pull from zot runs the unit, a digest bump rolls the spec, and selfHeal recreates a deleted CronJob.
- `test_showcase_lifecycle.py` - the narrative: dev build from the s3a lake registering to HTTP MLflow with S3 artifacts, sparkling training on the actual cluster, gate-verified GitOps promotion with pinned-replay idempotency and the unpinned-replay refusal, run-time champion resolution, prediction-store idempotency (same anchor overwrites, new anchor partitions), and ground-truth monitoring from the same table (the run waits while the cohort's labels are NULL, evaluates exactly once after they land, and a realized-gate breach exits 2, never 1).
- `test_showcase_obs.py` - run_results -> push_metrics.py -> Pushgateway -> Prometheus, the four canonical alert rules loaded, Grafana healthy, and `MbtShiftBreach` entering pending/firing on injected shift.
- `test_showcase_make.py` (extra gate: `MBT_LIVE_SHOWCASE_MAKE=1`, run in its own pytest invocation - it boots a second full stack) - the runbook itself: the README golden path driven through `make` on an isolated `SHOWCASE_PROJECT` (up, demo, ci + the browser login its output instructs, reset, score, outcomes, monitor, inject-drift + recovery, down, a re-stage over the previous run's output, clean), so these documented commands cannot drift from the tested harness silently.

Three hermetic modules keep the showcase honest in the ordinary fast suite, where the gated modules above only skip: `test_showcase_gates.py` (every gated module keeps its opt-in gate, and opting in without docker fails loudly), `test_showcase_image_pins.py` (the runner image's hand pins and extras closure agree with the declared metadata), and `test_showcase_panel.py` (the table generator is deterministic and its label, drift and reset semantics hold; every node reads the one table; ground truth joins on key and cohort; training never reaches the open cohort; every string feature is declared categorical).

## Deviations from the scaffold defaults (documented, deliberate)

- **No `--deep-snapshot` anywhere**: it would be a no-op. The table lives in the object store, so `SparkDataAdapter.snapshot_id` takes its URI branch and hashes the `df.inputFiles()` listing, which is checkout-mtime-independent already - deep and shallow produce the same token. ADR-11's fresh-checkout problem is a local-path problem. The "one token scheme per pipeline" rule is therefore satisfied with the spark scheme on both the baseline-publish and PR-diff sides, so the `.woodpecker/` pipelines pass no `--deep-snapshot` either (unlike the GitHub scaffold). The flip side: the table's files are immutable, and every change to it is a new file name.
- **Scoring and monitoring run on their own cluster-free `batch` target**: Spark `local[2]` straight off the lake, with champion MOJOs in a local H2O JVM by design - the cluster is train-time only.
- **Anchors are pinned constants** (`2026-06-30T00:00:00Z`; monitor at `2026-07-20T00:00:00Z`, past the 14d maturity) matching the seeded cohorts - wall-clock anchors over fixed-date data rot into empty windows. The `.woodpecker/` pipelines pin the same anchor, which also makes same-source rebuilds byte-identical (`generated_at == anchor`, ADR-19).
- **PR builds use the `ci` target**: a per-run sqlite MLflow and a workspace-local artifact store, so green PRs never register versions or re-point the shared `staging` alias; champion gates render "none (bootstrap)" in PR comments. The merge-time prod-build targets `dev` (spark local[2] + the SHARED registry): cluster/sparkling training from CI step containers is P3 deployable-unit territory, and the cluster path is proven live by the lifecycle tier.
- **One table serves training, scoring and ground truth**, where ADR-29 recommends a label-free serving twin. It is safe here because a model's target is never among its features, the prediction store never copies the label, and ground truth joins on `(customer_id, inference_date)` and treats a NULL label as not yet landed.
- The SeaweedFS buckets are created without any TTL/retention: nothing protects champion objects server-side, so retention rules would silently break champion gates and scoring.
- **Demo-tier security postures, on purpose**: the Woodpecker agent mounts the docker socket, committed demo S3 credentials, a wide-open Gitea, and tokens flowing through `make ci` output. This is a laptop lab, not a hardening reference.

## Knobs

See `.env.example` for host ports (defaults dodge common squatters), S3 credentials, workspace location, the runner image tag, and `DOCKER_SOCK_GID` (the docker-socket group airflow-scheduler joins to run DAG tasks; the Makefile and the test harness probe it - 0 on Docker Desktop, the `docker` group on native Linux).
`SCALE` on `make seed` sizes the lake table ([Scale](#scale)).
RAM guardrails are config, not prose: 1 Spark worker (4 cores / 4g) in the compose file, `spark.cores.max` and executor memory (1-2g) per session and `h2o_max_mem: 1G` on the local-H2O targets in `profiles.yml`, `WOODPECKER_MAX_WORKFLOWS=1`, Airflow on LocalExecutor.
Measured on a 10-core OrbStack machine, summing `docker stats` every few seconds across the stack's containers through a full main-tier run at the default `SCALE`: median 1.1GB, 90th percentile 1.9GB, highest sample 3.0GB.
Sampling can miss a short spike and a bigger `SCALE` needs more, and the stack's configured ceilings (a 4g Spark worker, 1-2g executors, 1G H2O JVMs) add up well past those samples, so leave ~10GB to docker for one stack; the k3d tier adds a k3d node running ArgoCD on top of it (1.5GB at its largest sample).
Budget disk too: the runner image is 2.0GB compressed (5.6GB as `docker images` lists it), the pinned service images add about 1.7GB compressed, and the first build also leaves a pip cache mount behind.
SeaweedFS's capacity is pinned in the compose command (64MB volumes under a 100-volume ceiling) so it never depends on how much of that disk is free; left to its defaults it scaled with free disk, and a nearly full docker disk failed every model upload (`docs/troubleshooting.md`).
