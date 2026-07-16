# mbt showcase: the full lifecycle on a dockerized platform stack

A laptop-runnable reference environment that demonstrates mbt end to end on real services instead of local stand-ins:
SeaweedFS is the S3 data lake (gold-layer feature tables) and artifact store, MLflow (over HTTP) is the tracking server and model registry, a standalone Spark cluster does dataset pushdown and in-executor H2O (sparkling) AutoML training, JupyterLab is the DS workbench, Gitea + Woodpecker run the state-diff CI loop with PR comments and gate-classified alerts, Zot holds the digest-pinned deployable unit and its oras provenance artifacts, Airflow (fed by git-sync from the Gitea `deploy` repo) schedules retrain/score/monitor runs of that unit, and Prometheus + Grafana observe production scoring through the Pushgateway spec documented in the tutorial.

The design of record is [DESIGN.md](DESIGN.md).
All phases are implemented: P1 (runner image + data/ML core), P2 (CI loop), P3 (deployable unit + provenance), P4 (scheduling + CD + the scoring/promotion/monitoring plane), P5 (observability), and P6 (k3d + ArgoCD - local-only, behind its own `MBT_LIVE_SHOWCASE_K3D=1` gate).

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

After `make ci`, pushing to main runs prod-build end to end: economy build, `mbt-state` baseline publish, deployable-unit bake to Zot (digest-pinned in the deploy repo), and oras provenance push; git-sync feeds the deploy repo's DAGs into Airflow, where `mbt_retrain`/`mbt_score`/`mbt_monitor` run the pinned unit on demand.

`make up` prints the UI URLs (JupyterLab, MLflow, Spark, Grafana, Prometheus, Gitea, Woodpecker).
After `make demo`, look at:

- **MLflow**: registered `churn_automl` versions, the `production` alias set by the promotion, per-run metrics and `mbt.*` provenance tags, plus one tracking run per monitored prediction run (the realized-performance time series).
- **Grafana** (`admin`/`admin`): the "mbt Model Health" dashboard - gate margins, realized metrics, shift-vs-threshold.
- **Predictions** on disk: `~/.cache/mbt-showcase/workspace/lake_local/predictions/retention_scores/<run_key>/`.
- `make inject-drift` then Grafana/Prometheus: the scoring batch is poisoned, `mbt score` exits 2 (mbt enforces), and the pushed breach fires the `MbtShiftBreach` alert (observability observes). `make score` recovers.

`make score` and `make monitor` also work standalone: they rerun just the scoring stage (lake sync, `mbt score`, metric push) or just the ground-truth monitoring stage, with the same pinned anchors as the demo.

## The CI loop (make ci)

`make ci` seeds Gitea with the `mbt-showcase/churn` repo (the project source, `.woodpecker/` pipelines included), creates the OAuth app, re-ups Woodpecker with the real credentials, and activates the repo - all headless (the first Woodpecker API token is minted by a scripted OAuth dance running inside the gitea container).
Then work like a user would: clone `http://localhost:3305/mbt-showcase/churn`, push to main (prod-build trains the state-modified subgraph on the shared registry and republishes the `mbt-state` baseline), or open a PR (pr-check lints promotions.yml, state-diffs against the published baseline, slim-builds only `state:modified+` on the throwaway `ci` target, and posts the update-in-place `mbt build report` comment).
Exit-code fidelity survives Woodpecker's binary pass/fail: `scripts/run_mbt.sh` records mbt's 1-vs-2 verdict in `target/ci_exit_class` and classifies the alert it curls to webhook-sink - exit 2 (quality) notifies the failing spec's `owner`, anything else pages on-call.

## The E2E test tier (the honest version of the demo)

Opt-in, following the live-tier double gate: skipped everywhere unless `MBT_LIVE_SHOWCASE=1`; once opted in, a missing docker fails loudly instead of skipping.

```bash
MBT_LIVE_SHOWCASE=1 uv run pytest -q -m live_showcase --timeout 3600 -rA
```

Modules (repo-root `tests/`), which boot their own isolated compose project on ephemeral ports with a tmp workspace and tear everything down:

- `test_showcase_infra.py` - services healthy, real S3 round-trip, seeded lake, `mbt` runs in the image, and the h2o-client == pysparkling-embedded-H2O version probe (an exact match is required by H2O; the image pins `h2o==3.46.0.6` for this).
- `test_showcase_ci.py` - the Woodpecker loop driven exactly as a user would (git pushes and PRs against Gitea): the first push honors `fetch_state.sh` exit 3 and full-builds (and bakes the first deployable unit), a no-change merge trains nothing yet republishes an identical baseline (and re-bakes nothing - the digest pin is untouched), a one-gate-edit PR slim-builds exactly the edited model (no dataset churn across fresh clones - URI snapshot stability), merging it retrains only that model (and pins a fresh unit), an impossible gate fails the pipeline with mbt's exit 2 classified as a quality failure (the PR comment shows `gate_failed`, the shared registry is untouched, webhook-sink records exactly one owner-classified alert), and promotions.yml is governed: branch protection + CODEOWNERS reject the unauthorized direct push, the owner-approved merge runs the promote pipeline, and the production alias moves with the deploy repo byte-identical.
- `test_showcase_provenance.py` - the deployable unit reproduces: the oras provenance artifact is byte-identical to the mbt-state baseline of the same run and secret-free, `mbt run --manifest` inside the pulled unit reproduces metrics (xgboost exactly, H2O within its documented 0.02 tier), and a tampered environment is refused with exit 1 (`--allow-env-mismatch` downgrades to a warning).
- `test_showcase_scheduling.py` - Airflow runs the pinned unit: the retrain DAG builds on the prod target (cluster pushdown from a scheduled container), two score DAG runs straddling a promotion serve different champions while the deploy repo HEAD and digest stay byte-identical (the ADR-20 inversion), and monitor exit codes route correctly (a realized-gate breach fails on try 1 with no retry; a hard error consumes a retry).
- `test_showcase_k3d.py` (extra gate: `MBT_LIVE_SHOWCASE_K3D=1`, local-only) - ArgoCD core in a k3d cluster on the compose network syncs the deploy repo's `k8s/`: the CronJob lands pinned to the baked digest, an insecure-HTTP pull from zot runs the unit, a digest bump rolls the spec, and selfHeal recreates a deleted CronJob.
- `test_showcase_lifecycle.py` - the narrative: dev build from the s3a lake registering to HTTP MLflow with S3 artifacts (integration items A2 + A3, live), sparkling training on the actual cluster, gate-verified GitOps promotion with pinned-replay idempotency and the unpinned-replay refusal, run-time champion resolution, prediction-store idempotency (same anchor overwrites, new anchor partitions), and ground-truth monitoring (evaluated exactly once; a realized-gate breach exits 2, never 1).
- `test_showcase_obs.py` - run_results -> push_metrics.py -> Pushgateway -> Prometheus, the four canonical alert rules, and `MbtShiftBreach` actually firing on injected shift.

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
