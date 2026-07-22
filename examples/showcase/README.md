# mbt showcase: the full lifecycle on a dockerized platform stack

A laptop-runnable reference environment that demonstrates mbt end to end on real services instead of local stand-ins:
SeaweedFS is the S3 data lake (gold-layer feature tables) and artifact store, MLflow (over HTTP) is the tracking server and model registry, a standalone Spark cluster does dataset pushdown and in-executor H2O (sparkling) AutoML training, JupyterLab is the DS workbench, Gitea + Woodpecker run the state-diff CI loop with PR comments and gate-classified alerts, Zot holds the digest-pinned deployable unit and its oras provenance artifacts, Airflow (fed by git-sync from the Gitea `deploy` repo) schedules retrain/score/monitor runs of that unit, and Prometheus + Grafana observe production scoring through the Pushgateway spec documented in the tutorial.
A second, cluster-free cadence rides the same lake: the `tag:monthly` churn pipeline trains, scores, and monitors entirely on the DuckDB batch plane (`prod_score`) over the synced S3 parquet (SHOW-17).
A third, wide multi-table batch-monthly cadence (`tag:wide`, SHOW-19/SHOW-20) exercises ADR-22 end to end: a monthly population spine carrying the customer_id-to-safe_id crosswalk, three feature-history tables joined by different keys (transactions only reach the panel through the crosswalk), the label joined from one calendar month after each snapshot via `time_offset`, the ds-helper feature-selection funnel committed as a reviewable diff, DS-declared numeric-coded categoricals cast by a shared hooks file, Evidently feature-stability gates around promotion and every monthly scoring batch, and sparkling H2O AutoML on the selected columns.

The design of record is [DESIGN.md](DESIGN.md).
Phases P1-P6 are implemented: P1 (runner image + data/ML core), P2 (CI loop), P3 (deployable unit + provenance), P4 (scheduling + CD + the scoring/promotion/monitoring plane), P5 (observability), and P6 (k3d + ArgoCD - local-only, behind its own `MBT_LIVE_SHOWCASE_K3D=1` gate).
P7 (a Snowflake warehouse variant with Snowflake as read-only data storage) is scoped in DESIGN.md and deliberately parked.

Everything mbt-related runs inside ONE runner image (Jupyter kernel, Spark master/worker, MLflow server, every `mbt` invocation), which makes ADR-19 `env_digest` verification hold by construction.

## Run it

Requirements: docker with ~10GB free RAM, `uv`, and this checkout.

```bash
cd examples/showcase
make up        # build the runner image (first time ~10 min), boot, seed the lake
make demo      # the whole lifecycle, narrated (build dev -> build prod -> promote -> score -> monitor)
make ci        # seed Gitea + Woodpecker + the deploy repo: org, repos, OAuth app, activation
make down      # stop and remove containers, volumes, and the network (the workspace survives)
make clean     # down, then also remove the workspace (~/.cache/mbt-showcase/workspace)
```

After `make ci`, pushing to main runs prod-build end to end: economy build, `mbt-state` baseline publish, deployable-unit bake to Zot (digest-pinned in the deploy repo), and oras provenance push; git-sync feeds the deploy repo's DAGs into Airflow, where `mbt_retrain`/`mbt_score`/`mbt_score_monthly`/`mbt_monitor` run the pinned unit on demand.

`make up` prints every UI URL with its login (`make urls` re-prints them); a bare `make` lists the targets.
After `make demo`, look at:

- **MLflow**: registered `churn_automl` versions, the `production` alias set by the promotion, per-run metrics and `mbt.*` provenance tags, plus one tracking run per monitored prediction run (the realized-performance time series).
- **Lake browser** (SeaweedFS filer UI, no login): the seeded gold tables under `/buckets/mbt-lake/` and the MLflow artifacts under `/buckets/mbt-artifacts/`.
  The raw S3 API port accepts signed requests only (`mbtadmin`/`mbtsecret`), so a bare browser GET there returns `AccessDenied` by design - browse through the filer UI instead.
- **Grafana** (`admin`/`admin`): the "mbt Model Health" dashboard - gate margins, realized metrics, shift-vs-threshold.
- **Predictions** on disk, one directory per cadence under `~/.cache/mbt-showcase/workspace/lake_local/predictions/`: `retention_scores/<run_key>/` (daily), `monthly_retention_scores/` and `wide_retention_scores/` likewise.
- `make inject-drift` then Grafana/Prometheus: the scoring batch is poisoned, `mbt score` exits 2 (mbt enforces), and the pushed breach fires the `MbtShiftBreach` alert (observability observes). `make score` recovers.

`make score` and `make monitor` also work standalone: they rerun just the daily scoring stage (lake sync, `mbt score --select tag:daily`, metric push) or just the ground-truth monitoring stage (all matured cadences), with the same pinned anchors as the demo.
`make monthly` runs the monthly cadence end to end on the DuckDB plane: lake sync, `tag:monthly` retrain, gate-verified promote, and month-start batch scoring - no cluster involved.
`make wide` runs the batch-monthly wide cadence (SHOW-19/SHOW-20, ADR-22) the way a DS team would ship it.
The LightGBM probe builds the full-width panel on the dev target; `scripts/select_features.py` then runs the ds-helper funnel over the materialized train split (drop >95%-missing columns, drop single-value columns, drop |corr| > 0.9 pairs, then a seeded LightGBM randomized search keeping importance > 0) and rewrites `churn_wide_automl`'s committed include list - the printed `git diff --stat` is the reviewable selection, and `target/feature_selection_report.json` documents every stage.
`models/wide_hooks.py` casts DS-declared numeric-coded categoricals (`contract_code`) to string for both wide models, so every adapter treats them as categoricals at train and scoring time alike.
The models' `exclude:` list is the DS's ignored-columns contract, and the funnel honors it: it names the entity ids plus `tenure_months`, a time-anchored feature the funnel would otherwise select - predictive inside the training window, but guaranteed to breach the PSI monitor at serving because every newer cohort's tenure sits above the training window's.
After sparkling AutoML trains on the cluster, `scripts/evidently_gate.py --phase train` checks the selected features for stability between the train and test windows and BLOCKS promotion on a breach (exit 2); on a pass it exports the persisted reference baseline.
After the population-form scoring run on the DuckDB plane, `--phase serving` re-checks the scored batch against that baseline, so the features stay verified stable from training through first deployment and every monthly batch after it (mbt's own PSI/KS shift monitors keep enforcing in parallel; Evidently adds the per-column drift tests and the pre-promotion phase, plus the DS-facing HTML report).
In a real deployment the `mbt_score_wide` DAG runs this cadence on `schedule="0 0 1 * *"` - predictions on every 1st of the month at 00:00, matching the month-start population snapshots.

For the real-world scale this cadence models (~7M rows x up to 2000 columns per feature table), the committed tables are the small deterministic default of `scripts/generate_wide_data.py`; its `--customers`/`--filler-columns` knobs synthesize the same shape as large as your disk allows, and the code paths are identical.
Joins and sampling push down into the source query, so only sampled rows ever leave the lake: `sample_key: customer_id` hash-samples whole customers per `sample_fraction` (a target var or `--vars '{sample_fraction: 0.1}'`; same fraction -> same rows, smaller fractions are subsets of larger ones), which is how the probe + funnel run cheaply on a slice while prod trains on everything.
The width problem is handled by selection, not sampling: only the probe ever reads all columns (once per snapshot, cached by materialization key; parquet is columnar, so the funnel touches only surviving columns), and the committed include list cuts the panel down before AutoML ever sees it.
Reproducibility is a chain: the generator seed fixes the data, the spec `seed: 42` drives the probe, the funnel's randomized search, and AutoML (the seed ladder derives every later stage), hash sampling is deterministic by key, the AutoML spec shape is the documented deterministic one (fixed `max_models`, no time budgets), and the selection itself is a committed diff - so a rerun reproduces the include list byte for byte.

## The CI loop (make ci)

`make ci` seeds Gitea with the `mbt-showcase/churn` repo (the project source, `.woodpecker/` pipelines included), creates the OAuth app, re-ups Woodpecker with the real credentials, and activates the repo - all headless (the first Woodpecker API token is minted by a scripted OAuth dance against the host-published ports - the exact flow a browser performs, thanks to Woodpecker's split-horizon URL config in the compose file).
Then work like a user would: log into Woodpecker at `http://localhost:8305` with the Gitea account (`mbtops`/`mbtops-showcase-password`), clone `http://localhost:3305/mbt-showcase/churn`, push to main (prod-build trains the state-modified subgraph on the shared registry and republishes the `mbt-state` baseline), or open a PR (pr-check lints promotions.yml, state-diffs against the published baseline, slim-builds only `state:modified+` on the throwaway `ci` target, and posts the update-in-place `mbt build report` comment).
One known cosmetic seam: Woodpecker's "repository" deep-links point at the in-network Gitea URL (`gitea:3000`), because that URL must stay resolvable by the CI step containers - use the printed `localhost` Gitea URL instead.
Exit-code fidelity survives Woodpecker's binary pass/fail: `scripts/run_mbt.sh` records mbt's 1-vs-2 verdict in `target/ci_exit_class` and classifies the alert it curls to webhook-sink - exit 2 (quality) notifies the failing spec's `owner`, anything else pages on-call.

## The E2E test tier (the honest version of the demo)

Opt-in, following the live-tier double gate: skipped everywhere unless `MBT_LIVE_SHOWCASE=1`; once opted in, a missing docker fails loudly instead of skipping.

```bash
MBT_LIVE_SHOWCASE=1 uv run pytest -q -m live_showcase --timeout 3600 -rA
```

Modules (repo-root `tests/`), which boot their own isolated compose project on ephemeral ports with a tmp workspace and tear everything down:

- `test_showcase_infra.py` - services healthy, real S3 round-trip, seeded lake (browsable from the host through the filer UI; the raw S3 port correctly refuses unsigned requests), `mbt` runs in the image, and the h2o-client == pysparkling-embedded-H2O version probe (an exact match is required by H2O; the image pins `h2o==3.46.0.6` for this).
- `test_showcase_ci.py` - the Woodpecker loop driven exactly as a user would (git pushes and PRs against Gitea): the browser OAuth login works from the host (driven headlessly for the non-admin persona, first consent included), the first push honors `fetch_state.sh` exit 3 and full-builds (and bakes the first deployable unit), a no-change merge trains nothing yet republishes an identical baseline (and re-bakes nothing - the digest pin is untouched), a one-gate-edit PR slim-builds exactly the edited model (no dataset churn across fresh clones - URI snapshot stability), merging it retrains only that model (and pins a fresh unit), an impossible gate fails the pipeline with mbt's exit 2 classified as a quality failure (the PR comment shows `gate_failed`, the shared registry is untouched, webhook-sink records exactly one owner-classified alert), and promotions.yml is governed: branch protection + CODEOWNERS reject the unauthorized direct push, the owner-approved merge runs the promote pipeline, and the production alias moves with the deploy repo byte-identical.
- `test_showcase_provenance.py` - the deployable unit reproduces: the oras provenance artifact is byte-identical to the mbt-state baseline of the same run and secret-free, `mbt run --manifest` inside the pulled unit reproduces metrics (xgboost exactly, H2O within its documented 0.02 tier), and a tampered environment is refused with exit 1 (`--allow-env-mismatch` downgrades to a warning).
- `test_showcase_scheduling.py` - Airflow runs the pinned unit: the retrain DAG builds on the prod target (cluster pushdown from a scheduled container), two score DAG runs straddling a promotion serve different champions while the deploy repo HEAD and digest stay byte-identical (the ADR-20 inversion), the monthly score DAG runs the `tag:monthly` batch on the DuckDB plane from the scheduler, and monitor exit codes route correctly (a realized-gate breach fails on try 1 with no retry; a hard error consumes a retry).
- `test_showcase_k3d.py` (extra gate: `MBT_LIVE_SHOWCASE_K3D=1`, local-only) - ArgoCD core in a k3d cluster on the compose network syncs the deploy repo's `k8s/`: the CronJob lands pinned to the baked digest, an insecure-HTTP pull from zot runs the unit, a digest bump rolls the spec, and selfHeal recreates a deleted CronJob.
- `test_showcase_lifecycle.py` - the narrative: dev build from the s3a lake registering to HTTP MLflow with S3 artifacts (integration items A2 + A3, live), sparkling training on the actual cluster, gate-verified GitOps promotion with pinned-replay idempotency and the unpinned-replay refusal, run-time champion resolution, prediction-store idempotency (same anchor overwrites, new anchor partitions), and ground-truth monitoring (evaluated exactly once; a realized-gate breach exits 2, never 1).
- `test_showcase_monthly.py` - the monthly cadence (SHOW-17): `tag:monthly` trains on the prod_score plane (DuckDB over the synced lake, no cluster), gate-verified promote, the month-start batch scores with the run-time champion under both shift monitors, and its 30-day labels mature at the pinned monitor anchor and evaluate exactly once.
- `test_showcase_wide.py` - the wide batch-monthly cadence (SHOW-19/SHOW-20, ADR-22): the population-spine dataset (per-table join keys + the 1-calendar-month label offset) builds via Spark pushdown against the s3a lake, the ds-helper funnel reproduces the committed feature list byte-for-byte (selection report included, `contract_code` surviving through the shared hooks cast), `--vars sample_fraction` panel-samples whole customers with the subset property, sparkling AutoML trains on the selected columns, the Evidently train gate passes and exports the serving baseline, the population-form scoring input scores the newest cohort under both shift monitors plus the serving gate, its outcomes evaluate exactly once at maturity, and a poisoned batch trips the serving gate with exit 2.
- `test_showcase_obs.py` - run_results -> push_metrics.py -> Pushgateway -> Prometheus, the four canonical alert rules, and `MbtShiftBreach` actually firing on injected shift.
- `test_showcase_make.py` (extra gate: `MBT_LIVE_SHOWCASE_MAKE=1`, run in its own pytest invocation - it boots a second full stack) - the runbook itself: the README golden path driven through `make` on an isolated `SHOWCASE_PROJECT` (up, demo, ci + the browser login its output instructs, wide, monthly, score, monitor, inject-drift + recovery, down, clean), so these documented commands cannot drift from the tested harness silently.

## Deviations from the scaffold defaults (documented, deliberate)

- **No `--deep-snapshot` on the spark targets**: the spark data adapter rejects it; its URI snapshots hash `df.inputFiles()` listings, which are checkout-mtime-independent because sources live in the object store. The "one token scheme per pipeline" rule is satisfied with the spark scheme on both the baseline-publish and PR-diff sides, so the `.woodpecker/` pipelines pass no `--deep-snapshot` either (unlike the GitHub scaffold). The `prod_score` local-adapter plane DOES pass `--deep-snapshot` (plus a fixed-mtime lake sync) so prediction run_keys stay stable across syncs.
- **Scoring runs on the local data adapter** over `/workspace/lake_local` (synced from the lake): mbt-spark implements no contract-1.1 scoring methods, and champion MOJOs evaluate in a local JVM by design.
- **Anchors are pinned constants** (`2026-06-30T00:00:00Z`; monitor at `2026-07-20T00:00:00Z`, past the 14d maturity) matching the seeded data range - wall-clock anchors over fixed-date data rot into empty windows. The `.woodpecker/` pipelines pin the same anchor, which also makes same-source rebuilds byte-identical (`generated_at == anchor`, ADR-19).
- **PR builds use the `ci` target**: a per-run sqlite MLflow and a workspace-local artifact store, so green PRs never register versions or re-point the shared `staging` alias; champion gates render "none (bootstrap)" in PR comments. The merge-time prod-build targets `dev` (spark local[2] + the SHARED registry): cluster/sparkling training from CI step containers is P3 deployable-unit territory, and the cluster path is proven live by the lifecycle tier.
- The SeaweedFS buckets are created without any TTL/retention: nothing protects champion objects server-side, so retention rules would silently break champion gates and scoring.
- **Demo-tier security postures, on purpose**: the Woodpecker agent mounts the docker socket, committed demo S3 credentials, a wide-open Gitea, and tokens flowing through `make ci` output. This is a laptop lab, not a hardening reference.

## Knobs

See `.env.example` for host ports (defaults dodge common squatters), S3 credentials, workspace location, the runner image tag, and `DOCKER_SOCK_GID` (the docker-socket group airflow-scheduler joins to run DAG tasks; the Makefile and the test harness probe it - 0 on Docker Desktop, the `docker` group on native Linux).
RAM guardrails live in the compose file: 1 Spark worker (4 cores / 4g), executor 1-2g per session, `h2o_max_mem: 1G` on the dev target, `WOODPECKER_MAX_WORKFLOWS=1`, Airflow on LocalExecutor.
Budget roughly: ~5GB steady state with every profile up, 8-9GB peak during sparkling training, +~1.5GB while the optional k3d/ArgoCD profile runs.
