# mbt showcase: full-stack docker reference environment + E2E test tier (design)

Status: IMPLEMENTED (all phases).
P1 (runner image + data/ML core), P2 (Gitea + Woodpecker CI loop incl. branch protection + CODEOWNERS on promotions.yml), P3 (Zot deployable unit + oras provenance), P4 (Airflow + git-sync CD + the scoring/promotion/monitoring plane), P5 (observability), P6 (k3d + ArgoCD, local-only behind its own `MBT_LIVE_SHOWCASE_K3D` gate), P7 (the Snowflake warehouse plane, behind credentials) and P8 (the SeaweedFS object-store plane) are implemented and covered by the `live_showcase` test tier; see README.md for what runs today.

This document started as the design and is kept as the design of record: where the implementation deliberately scoped something differently, the sections below say what was built, not what was first proposed.

The test catalog (section 8) maps onto these gated modules, all in repo-root `tests/`:

| Module | Covers | Gate beyond `MBT_LIVE_SHOWCASE=1` |
|---|---|---|
| `test_showcase_infra.py` | SHOW-01/02, including the registry-outage exit-1 contract | - |
| `test_showcase_ci.py` | SHOW-05/06/07/10 (CI side) | - |
| `test_showcase_lifecycle.py` | SHOW-03/04/10/11/12/13 (CLI-driven) | - |
| `test_showcase_monthly.py` | SHOW-17 | - |
| `test_showcase_obs.py` | SHOW-14 | - |
| `test_showcase_provenance.py` | SHOW-08/09 | - |
| `test_showcase_scheduling.py` | SHOW-11 strong form, SHOW-13 routing, SHOW-17's and SHOW-20's scheduled paths | - |
| `test_showcase_seaweedfs.py` | P8, the object-store plane | - |
| `test_showcase_wide.py` | SHOW-19/SHOW-20 (ADR-29) | - |
| `test_showcase_k3d.py` | SHOW-16 | `MBT_LIVE_SHOWCASE_K3D=1` (local-only) |
| `test_showcase_make.py` | SHOW-18 | `MBT_LIVE_SHOWCASE_MAKE=1` (own pytest invocation) |
| `test_showcase_snowflake.py` | P7, the warehouse plane | `MBT_LIVE_SNOWFLAKE=1` + complete `SNOWFLAKE_*` |

Hermetic modules guard the showcase from the ordinary fast suite, because `-m e2e` excludes the live tier and `-m "not e2e"` deselects it: `tests/test_showcase_gates.py` (SHOW-15: every gated module keeps its gate), `tests/test_showcase_image_pins.py` (the runner image's hand pins, the extras closure, host-run S3 credentials), `tests/test_showcase_seaweedfs_plane.py` (the object-store target's addressing and namespacing) and `tests/test_showcase_wide_scripts.py` (the selection funnel, the Evidently gate, the categorical declarations, clean notebooks), plus `packages/mbt-snowflake/tests/test_showcase_snowflake_plane.py` for the warehouse plane.

Modules share one session stack and run in collection (alphabetical) order; the only load-bearing constraint is that `test_showcase_ci` is the first forge consumer (virgin-bootstrap assertion) - everything else provisions or promotes what it needs and scopes score/monitor by cadence tag.
Implementation notes (deliberate scoping vs the sections below):
pr-check builds on the `ci` target and the merge-time prod-build on `dev` (spark local[2] + the shared registry); the scheduled retrain DAG is the cluster-from-CI path (prod target from a pinned unit), tested with the deterministic xgboost workhorse - sparkling stays confined to SHOW-04's module and the wide module (SHOW-19) per the flake-isolation rule.
Baking is gated on "this merge retrained something": docker layer digests embed mtimes, so an unconditional bake would mint a new digest per merge and break the ADR-20 "promotion deploys nothing" claim.
SHOW-05's "exactly one modified node" reads as "exactly the edited model (config) plus its downstream scoring node (upstream)" - scoring depends_on its model, so the lineage flag is correct behavior.
Zot is addressed two ways for one digest: the docker daemon pushes/pulls via the published localhost port (insecure-by-default, the open-question-1 answer, verified on Docker Desktop for macOS, OrbStack and the Linux CI runner), while in-network consumers (oras, k3d) use zot:5000.
Alert routing stops at Prometheus: the four rules evaluate there and the tests read `ALERTS` through its API, and no Alertmanager or Grafana contact point is provisioned (section 7). The only webhook traffic is the CI exit-code classifier's.
This document specifies a dockerized reference environment that demonstrates the full DS + MLOps lifecycle on mbt, and the opt-in E2E test tier that keeps every claim it makes honest.
It extends the existing integration-testing plan: Tier A (hermetic) and Tier B (live, credentialed) exist; this is the showcase tier, gated like Tier B but self-hosted, and it closes the open A2 (MLflow over HTTP) and A3 (real S3 API) items as a side effect.

## 1. Purpose and personas

Two personas walk one story end to end:

- **Data scientist**: iterates in JupyterLab against an offline feature store (SeaweedFS S3), builds datasets with Spark pushdown, trains H2O AutoML (pysparkling on the cluster), registers to MLflow, opens a PR.
- **MLOps engineer**: owns CI (Woodpecker on Gitea), the deployable unit (Zot OCI images with baked manifests), CD (git-reconciled deploy repo, optional ArgoCD), scheduling (Airflow), observability (Prometheus + Grafana), and gate-verified GitOps promotion.

The showcase must exercise mbt's actual differentiators, not generic MLOps plumbing:

1. State-diff slim CI (`state:modified+` against a published baseline; a no-change build trains nothing).
2. Manifest-verified reproducible execution (`mbt run --manifest`, ADR-19 `env_digest` hard-fail on drift).
3. Gate-verified GitOps promotion (`mbt promote --from-file promotions.yml`, refuses versions without `mbt.gates_passed`).
4. Run-time champion resolution (ADR-20: promotion changes the next scheduled run with zero redeploy).
5. Prediction-store idempotency (run_key excludes the anchor; same-anchor re-runs overwrite, ADR-21).
6. Ground-truth monitoring with exit-code-2 semantics (evaluated-once markers, realized-metric gates).

## 2. Topology verdict

**Compose-first, one runner image, k3d/ArgoCD as an optional fidelity tier.**
This synthesizes three competing designs that were independently drafted and adversarially judged; the judges' consensus and every confirmed fatal-flaw fix are folded in below.

- One `docker-compose.yml` under `examples/showcase/compose/` with profiles `core`, `spark`, `dev`, `obs`, `ci`, `orch`. The profiles document the service groups; the Makefile and the test harness both boot all six, because every tier drives more than one group.
- One bridge network (`<compose project>_default`); Woodpecker step containers join it (`WOODPECKER_BACKEND_DOCKER_NETWORK`) so CI steps resolve `seaweedfs`, `mlflow`, `gitea`, `zot` and `webhook-sink` by name.
- **One runner image** used everywhere: Jupyter kernel, Spark master/worker runtime, MLflow server, every Woodpecker mbt step, Airflow task container, and the deployable unit base.
  Same image everywhere makes ADR-19 `env_digest` verification true by construction instead of an ops problem.
- **CD default is a git-sync reconciler**, not ArgoCD: a `deploy` repo in Gitea holds `images.env` (the digest pin plus session wiring) and the DAG files, reconciled into Airflow every 5 seconds by a `git-sync` sidecar.
  This is ArgoCD's essential loop (git as source of truth, auditable rollback = `git revert`) at a fraction of the RAM, and it is honest: ADR-20 means the frequent release event (model promotion) is a registry alias flip, never a deploy, so the only thing left to CD is which unit digest the scheduled jobs run.
- **Optional k3d + ArgoCD tier** for literal fidelity, driven by `tests/test_showcase_k3d.py` rather than a compose profile: `k3d cluster create --network <compose-net>` attaches the node to the compose network, `--host-alias` feeds the compose services into CoreDNS's NodeHosts (pods cannot use docker's embedded DNS, whose 127.0.0.11 is loopback-scoped to the node), ArgoCD core installs with `kubectl apply --server-side` and syncs the deploy repo's `k8s/` CronJob, and `registries.yaml` marks `zot:5000` insecure-HTTP.
  Local-only, behind its own gate, never on the nightly path.

RAM budget: see README.md's "Knobs" section, which carries the measured numbers.
Guardrails are config, not prose: one Spark worker capped at 4 cores / 4g, per-session `spark.cores.max` and executor memory in `profiles.yml`, `h2o_max_mem` per target, `WOODPECKER_MAX_WORKFLOWS=1`, and Airflow on LocalExecutor.

### Two load-bearing shared mounts

The Spark adapters have a hard driver-local-filesystem assumption (ADR-17): split staging writes `coalesce(1)` output to a driver-local temp dir that executors must also see, and `_materialize_for_path_adapter` dirs are read by `h2o.import_file` on executors.

1. `/workspace`: a host directory (`SHOWCASE_WORKSPACE`, default `~/.cache/mbt-showcase/workspace`; a pytest tmp dir under the test tier) bind-mounted at the identical absolute path into JupyterLab, the Spark worker, webhook-sink, and every Airflow-launched unit container. `TMPDIR=/workspace/tmp` wherever mbt can drive the cluster (JupyterLab and the unit containers). Woodpecker steps do not mount it: they build on `dev`/`ci`, whose `local[2]` master keeps the driver and executors in one JVM.
2. `/workspace/lake_local`: the DuckDB plane's local data root, mirrored from the lake bucket by `bootstrap/sync_lake.py` with a fixed mtime (see 4.3 and section 6).

The runner entrypoint preflights `TMPDIR` in every container that sets it and hard-fails if it is unwritable, converting the misleading distance-failure `split ... materialized 0 rows` into an immediate diagnosable error.
The host creates `tmp/` and `monitoring/` itself before booting, because every container runs as root and on native Linux whichever one reached a directory first would own it.

### Spark driver reachability

Spark standalone does not support cluster deploy-mode for Python apps, so every driver runs client-mode inside a dynamically named container and executors must connect back.
The runner entrypoint exports `SPARK_DRIVER_HOST=$(hostname -i)`; the `prod` target sets `spark.driver.host: "{{ env('SPARK_DRIVER_HOST', 'jupyterlab') }}"` and fixed driver/block-manager ports, one pair for the data adapter's session (40400/40401) and one for the sparkling training session (40402/40403), since both coexist in one prod build.

## 3. Service inventory

Every image is pinned to an exact version; `compose/docker-compose.yml` is the source of truth for the tags, and the table names them so a reader can see what was verified together.

| Service | Image | Role |
|---|---|---|
| runner (built, not run) | `mbt-showcase-runner:dev`, built on the host by `scripts/build_image.sh` from a digest-pinned `python:3.11-slim-bookworm` + OpenJDK 17 JRE + the workspace wheels (`mbt-core[s3]`, `mbt-h2o[sparkling]`, `mbt-spark`, `mbt-xgboost`, `mbt-lightgbm`, `mbt-mlflow`, `mbt-evidently`) with third-party pins from `uv.lock`, evidently's included + pyspark 3.5.8 (its bundled jars are `SPARK_HOME`) + hadoop-aws 3.3.4 and the matching AWS SDK bundle + jupyterlab from the committed `image-extras.txt` closure | Universal environment; env_digest identity by construction. A content-hash label rebuilds it whenever package sources, `uv.lock`, or the image inputs move |
| gitea | `gitea/gitea:1.27.3-rootless` | Hosts `churn` project repo + `deploy` repo; branch protection + CODEOWNERS gate on `promotions.yml`; `mbt-state` branch storage |
| woodpecker server + agent | `woodpeckerci/woodpecker-*:v3.18.1` | CI: pr-check, prod-build, promote pipelines; agent mounts the docker socket; repo marked trusted so the bake step may mount that socket; split-horizon URLs (public WOODPECKER_HOST + forge OAuth host, in-network webhook host) so the Gitea OAuth login works from a host browser |
| seaweedfs | `chrislusf/seaweedfs:4.47` (`weed server -s3`, 64MB volumes under a fixed 100-volume ceiling, section 9) | S3-compatible object store: `mbt-lake` bucket (feature store parquet, read via s3a://) and `mbt-artifacts` bucket (artifact stores under `churn_lake`, `churn_seaweedfs`, `churn_snowflake`); the filer UI is published for humans to browse the lake |
| spark-master, spark-worker | runner image running `spark-class ...Master` / `...Worker` (1 worker, 4 cores, 4g) | Standalone cluster: pushdown reads/sampling/windows + sparkling H2O training |
| mlflow | runner image (`mlflow server`, sqlite on a volume) | Tracking + model registry (alias mode); champion source of truth |
| jupyterlab | runner image + `jupyter lab` | DS workbench and the container every `make` recipe runs mbt in |
| airflow | `apache/airflow:3.3.1` (init + api-server + scheduler + dag-processor), **LocalExecutor + `postgres:18.6-alpine`** (sqlite forces SequentialExecutor; invalid with LocalExecutor) + `git-sync:v4.7.1` sidecar | Runs the retrain/score/monitor DAGs on demand; tasks drive the docker SDK to run the pinned digest from the deploy repo's `images.env` |
| zot | `ghcr.io/project-zot/zot:v2.1.16` (v2.1.17+ breaks multi-GB pushes on slow disks, zot#4140) | OCI registry: baked deployable units (`mbt/churn`) and oras-pushed `manifest.json`/`run_results.json` provenance artifacts (`mbt/churn/provenance`) |
| prometheus + pushgateway | `prom/prometheus:v3.13.3`, `prom/pushgateway:v1.11.3` | Metrics per docs/tutorial.md step 14 (the documented spec, implemented verbatim) and the four alert rules |
| grafana | `grafana/grafana:13.1.5` | File-provisioned Prometheus datasource + the "mbt Model Health" dashboard (gate margins, realized metrics, shift vs threshold, node durations) |
| webhook-sink | ~60-line python recorder with `GET /requests` | Records the alerts `scripts/run_mbt.sh` posts to `MBT_ALERT_WEBHOOK`; human-readable in demos, assertable in tests |
| bootstrap (host + one-shot scripts) | `scripts/ci_bootstrap.py` on the host; `bootstrap/seed_lake.py` in the runner | `make ci`: Gitea users, org, repos, token, OAuth app, deploy repo, Woodpecker login + repo activation + secrets. `make up`: buckets (no TTL/retention, section 9) and the seed-data upload. The image is built by `make image` |

Registry addressing, as built: the docker daemon (the bake step and Airflow's unit containers) addresses zot through the published `localhost` port, which docker treats as insecure-by-default with no daemon config; in-network consumers (oras, the k3d nodes) use `zot:5000` over plain HTTP.

## 4. The demo project

An `mbt init`-derived `churn_lake` project, source-of-truth at `examples/showcase/project/`, staged into the workspace by `make up` and pushed into Gitea as the `mbt-showcase/churn` repo by `make ci` (the showcase never depends on a long-lived external repo).

### 4.1 Models

- `churn_automl` (h2o_automl, `max_models: 3`, `include_algos: [GLM, GBM]`, `nfolds: 0`, fixed seed, no `max_runtime_secs`): the star of the DS story, `tags: [churn, weekly]`, threshold + champion gates, `registration: {name: churn_automl, stage_on_pass: staging}`.
- `churn_baseline_xgb` (xgboost): deterministic workhorse for the CI-loop differentiator tests (bit-exact `--manifest` reproduction; H2O's documented determinism tier is 0.02 tolerance, so H2O reproduction asserts within tolerance, never byte-equality).
  Sparkling + remote master stays confined to its own test module so a flake there never poisons the slim-CI/promotion/idempotency assertions.
- `churn_monthly_xgb` (xgboost, SHOW-17): the monthly batch cadence, `tags: [monthly]`, trained/scored/monitored entirely on the `prod_score` plane (DuckDB over the synced lake, no cluster) with its own 30-day-churn tables (`monthly_*`, month-start snapshots generated by `scripts/generate_monthly_data.py`) and its own gate floor (`monthly_pr_auc_floor`); the spec stays target-portable, so CI's spark targets build it when touched.
- The wide batch-monthly cadence (SHOW-19/SHOW-20, ADR-29), `tags: [wide]`, on its own tables (`scripts/generate_wide_data.py`: a `monthly_population` spine carrying the customer_id-to-safe_id crosswalk, `monthly_labels` keyed by each cohort's own `inference_date` and present only once matured (the gold-layer label contract; a raw observation-dated feed would instead be realigned in the panel's own join), and three feature histories joined by DIFFERENT entity keys on the one uniform `inference_date` join key - demographics/logins by `(customer_id, inference_date)`, transactions by `(safe_id, inference_date)` - with the spine carrying the entity crosswalk plus the DS-excluded `as_of_date` (balances describe the previous day) and `loaded_at_time` lineage/audit columns, per docs/naming-conventions.md; ~66 feature-history columns including the numeric-coded `contract_code`, `--customers/--filler-columns` stress knobs; the generator then joins all five into `monthly_panel` (70 columns in the committed shape) and its label-free twin `monthly_panel_scoring`, which is what the specs actually read - ADR-29):
  `notebooks/ds_inner_loop.ipynb` is the DS workbench narrative for this cadence (explore, probe build, funnel, sampled what-if on a scratch copy), executed top to bottom by the live tier so it cannot rot;
  `churn_wide_probe` (lightgbm) materializes the full-width panel and exports gain importance as a sanity cross-check; `scripts/select_features.py` runs the ds-helper funnel over the materialized train split (drop high-missing, drop single-value, drop correlated pairs, then a seeded LightGBM randomized search keeping importance > 0) and commits the winners into `churn_wide_automl`'s include list between BEGIN/END markers, with every stage documented in `target/feature_selection_report.json` - feature selection as a reviewable diff that slim CI retrains on;
  `features.categorical` on both wide specs declares the numeric-coded categoricals (ADR-27), so core retypes them before feature filtering, at train and scoring time alike, and every adapter's native categorical handling picks them up;
  the models' `exclude:` list is the DS-declared ignored-columns contract and the funnel honors it: beyond the entity ids it names `tenure_months`, which is anchored to calendar time and therefore breaches the training-time PSI baseline at serving no matter how predictive it looks inside the training window (the funnel found it, the feature_shift monitor rejected it, and the exclusion records that lesson);
  `churn_wide_automl` (h2o_automl, sparkling on prod) trains on the selected columns and serves through `scoring/wide_retention_scoring.yml`, whose input reads `monthly_panel_scoring` - the same upstream join minus the label, so the newest cohort survives - and whose champion's recorded feature columns are enforced at score time;
  `scripts/evidently_gate.py` (a showcase-local gate; mbt itself only displays Evidently's report, through `mbt-evidently`) enforces feature stability with exit-code-2 semantics beside mbt's own monitors: the train phase (train vs test on the selected features) blocks `mbt promote` and exports the persisted serving baseline, and the serving phase re-checks every scored batch against that baseline, rendering the DS-facing drift report in both phases.

### 4.2 sources.yml (the one-file-serves-all resolution)

One `sources.yml` must serve the Spark training targets, the local-adapter scoring target, and the Snowflake plane:

- Spark data adapter `root: "s3://mbt-lake"` (the `s3://` prefix is exempt from `normalized_adapter_config` project-dir path resolution; a raw `s3a://` root would be mangled).
- Scheme mapping via conf: `spark.hadoop.fs.s3.impl: org.apache.hadoop.fs.s3a.S3AFileSystem`, `fs.s3a.endpoint: http://seaweedfs:8333`, `fs.s3a.path.style.access: "true"`.
- Table paths stay **relative** (`subscribers/*.parquet`), so the local adapter serves the same tables from `root: /workspace/lake_local`.
- Every table ALSO declares `identifier: MBT_SHOWCASE_<TABLE>` for the `snowflake` target (section 11, P7). Declaring both is legal - `SourceTable` rejects only a table with neither - which is what lets one project and one set of specs cover every plane. The local and Snowflake adapters each read one field and ignore the other; Spark reads either, so the spark targets say which via `source_address: path` (see P7's consequences below). All 14 relations carry an identifier even though only the wide cadence runs on Snowflake: compile pins every source a dataset or scoring node references (nine of them) regardless of `--select`, and the five gold tables the panels are joined from carry theirs so `sources.yml` names every table the seeder creates.

### 4.3 Targets (profiles.yml, committed and secret-free)

`profiles.yml` is committed with pure `{{ env_var(...) }}` / `{{ env(...) }}` values (the scaffold gitignores profiles, so CI checkouts have none otherwise); container env supplies `MLFLOW_TRACKING_URI=http://mlflow:5000`, `AWS_ENDPOINT_URL_S3=http://seaweedfs:8333`, keys, region.
Note: boto3's env chain is the only S3 endpoint mechanism (nothing in mbt parses endpoints), so one process talks to exactly one S3 endpoint; fine here since SeaweedFS is the only object store.

| Target | Data | Compute/training | Registry/artifacts | Use |
|---|---|---|---|---|
| `dev` | spark `master: local[2]`, s3a to lake | h2o local backend (`h2o_max_mem: 1G`), `sample_fraction: 1.0` (narrow it per run with `--vars`) | shared MLflow, `s3://mbt-artifacts/churn_lake` | DS fast inner loop; also the merge-time prod-build target |
| `ci` | same as dev | same as dev | **per-run sqlite MLflow + local artifact store** | PR checks: green PRs must never register versions or re-point the shared `staging` alias; tradeoff: champion gates render "none (bootstrap)" in PR comments, documented |
| `prod` | spark `master: spark://spark-master:7077` | `h2o_backend: sparkling`, driver-host conf per section 2 | shared MLflow, `s3://mbt-artifacts/churn_lake` | Prod builds, weekly retrain |
| `prod_score` | **local adapter**, `root: /workspace/lake_local` | local (MOJO scoring is local-JVM by design; the remote cluster is train-time only) | shared MLflow, `s3://mbt-artifacts/churn_lake` | `mbt score` / `mbt monitor` on the cluster-free DuckDB plane, and the monthly cadence's training |
| `seaweedfs` | spark `master: local[2]`, s3a to lake, `predictions_root: /workspace/seaweedfs_predictions` | h2o local backend, in the stack | shared MLflow + `s3://mbt-artifacts/churn_seaweedfs` | The object-store plane (section 11, P8): the same wide cadence built AND scored/monitored off s3a, with no synced copy. Registers `*_seaweedfs` names via `plane_suffix` |
| `snowflake` | **snowflake adapter**, tables by `identifier:` | h2o local backend, on the host | shared MLflow + `s3://mbt-artifacts/churn_snowflake`, both over PUBLISHED ports | The warehouse plane (section 11): the same wide cadence, `--target snowflake`. Runs on the host, not in a container. Registers `*_snowflake` names via `plane_suffix` |

### 4.4 Scoring resource

`scoring/retention_scoring.yml`: `model: churn_automl`, `stage: production`, `tags: [daily]`, input checks, PSI/KS shift monitors, `ground_truth` with a 14-day maturity window and realized-metric gates.
`scoring/monthly_retention_scoring.yml` (`tags: [monthly]`) and `scoring/wide_retention_scoring.yml` (`tags: [wide]`) follow the same shape for their cadences, each with a `-31d:-28d` input window that bridges its 14-day maturity to a one-month label horizon.
Scheduling lives entirely outside the YAML (there is no schedule field); the DAGs select by tag (`--select tag:daily`, `tag:monthly`, `tag:wide`).

### 4.5 Anchors and seed data (the determinism spine)

Seed data is generated with fixed RNG seeds and committed: the daily tables (`tests/fixtures/churn_demo/data/`, snapshots 2026-01-01 .. 2026-06-29) and the monthly and wide tables (`examples/showcase/data/`, month starts 2025-07-01 .. 2026-06-01), uploaded to the lake by `make up`.
Every pipeline, DAG, and test pins anchors to constants: `ANCHOR=2026-06-30T00:00:00Z` for build and score, `MONITOR_ANCHOR=2026-07-20T00:00:00Z` (past maturity) for monitor.
Wall-clock anchors over fixed-date data are a time bomb: relative windows like `-150d:-28d` resolve empty within weeks and every unpinned pipeline rots into `split ... materialized 0 rows`.
Airflow DAGs therefore take `--anchor` from the deploy repo (`showcase_dag_utils.py`, overridable per run through the DAG's `anchor` param), never from `{{ ts }}`.
Anchor time travel is also what makes monitoring demoable today: `mbt monitor --anchor <maturity+> ` evaluates immediately; re-running with the same anchor evaluates nothing (exactly-once proof).
The `.woodpecker/` pipelines pin the same `ANCHOR`, which also makes same-source rebuilds byte-identical manifests (`generated_at == anchor`, ADR-19).

## 5. Golden path (the demo narrative)

1. `make up`: build (or reuse) the runner image, stage the workspace, `docker compose up -d --wait`, seed the lake; the terminal prints every UI URL.
2. **DS inner loop**: open `project/notebooks/ds_inner_loop.ipynb` in JupyterLab - explore the seeded lake, review the YAML, `mbt build --target dev --select churn_wide_probe`, run the selection funnel, try a hash-sampled what-if on a scratch copy; H2O leaderboard lines stream as events when AutoML trains, runs appear in the MLflow UI, model artifacts land in SeaweedFS. (`make demo` step 1 is the same `mbt build --target dev` over every model.)
3. **DS scales out**: `mbt build --target prod`; pushdown sampling runs on the cluster and sparkling H2O trains inside the executors (Spark UI shows the apps); models register and the `staging` alias moves.
4. **PR** (after `make ci`): push a branch, open a Gitea PR; Woodpecker pr-check lints `promotions.yml`, runs parse, compile, `fetch_state.sh` (exit 3 = bootstrap), `state diff --output json`, slim build `--select state:modified+ --state ...` under the `ci` target, then posts the update-in-place `<!-- mbt-pr-comment -->` comment via Gitea's API showing exactly the modified nodes and their gates.
5. **Merge**: prod-build runs the economy build on `dev`, publishes the manifest to `refs/heads/mbt-state` (`publish_state.sh` is pure git plumbing and works against Gitea unchanged), and - when the merge retrained something or no unit exists yet - bakes the deployable unit (`FROM` the exact runner tag + project + `target/manifest.json` compiled inside that same image), pushes it to Zot, oras-pushes `manifest.json`+`run_results.json` as provenance artifacts (manifests are secret-free by construction), and commits the new digest to the deploy repo.
6. **CD**: git-sync reconciles the deploy repo into Airflow; rollback is `git revert` (the optional k3d tier has ArgoCD sync the same repo's `k8s/` CronJob).
7. **Schedules**: the DAGs run the pinned unit - `mbt_retrain` (`mbt build --target prod`, `tag:weekly` by default), `mbt_score` (lake sync, then `tag:daily`), `mbt_score_monthly` (lake sync, then `tag:monthly`), `mbt_score_wide` (lake sync, `tag:wide`, then the Evidently serving gate) and `mbt_monitor`. They are manual-trigger (`schedule=None`) so demos and tests stay deterministic; each DAG's docstring says what a real deployment would schedule instead (`mbt_score_wide`: `0 0 1 * *`). Exit 1 retries then pages on-call, exit 2 never retries and notifies the model owner (per the tutorial's routing rule); the task pushes metrics regardless of outcome.
8. **Promotion**: a PR edits `promotions.yml` (version always pinned); CODEOWNERS + branch protection gate the merge; the promote pipeline runs `mbt promote --from-file`; the next score run serves the new champion with zero redeploy, image digest and deploy repo byte-identical before and after (the ADR-20 inversion, asserted).
9. **Monitoring pays off**: `make inject-drift` poisons the scoring batch, `mbt score` exits 2 (`monitor_failed`), and the pushed `mbt_shift_value >= mbt_shift_threshold` puts the `MbtShiftBreach` rule into pending and then firing in Prometheus; `make score` recovers. The `MbtScheduleStale` rule watches `push_time_seconds` for the one failure no in-band mechanism can catch, a schedule that silently stopped.

## 6. CI design (Woodpecker)

Pipelines live in `.woodpecker/` of the project repo (authored fresh; the GitHub scaffold under `_scaffold/.github/` stays untouched, so `tests/test_cli_basics.py` is unaffected).

- **Exit-code fidelity**: Woodpecker collapses any nonzero exit to "failed", erasing mbt's 1-vs-2 contract.
  Every `mbt build` and `mbt promote` invocation in the pipelines runs through `scripts/run_mbt.sh`: capture code, write `target/ci_exit_class` (`0 ok` / `1 hard-error` / `2 quality-failure`), on failure POST a classified alert to `MBT_ALERT_WEBHOOK` (2-second curl timeout; 1 pages on-call, 2 notifies the failing spec's owner), push metrics best-effort (`push_metrics.py` uses a 5-second timeout and warns and exits 0 when the Pushgateway is away, so observability can never fail a pipeline), re-exit with the original code.
  `target/ci_exit_class` is the verdict left for a human or a later step to read; the test tier asserts the same classification on the payload webhook-sink recorded.
- **Snapshot scheme deviation, documented**: no `--deep-snapshot` anywhere in this project's pipelines.
  It would be a no-op there, not an error: for a URI source the spark data adapter hashes the `df.inputFiles()` listing and ignores `deep`, and that listing is checkout-mtime-independent because the sources live in the object store. The "one token scheme per pipeline" rule is therefore satisfied with the spark scheme on both the baseline-publish and diff sides.
  The `prod_score` local-adapter target is the exception: its score/monitor invocations DO pass `--deep-snapshot`, and the lake sync pins a fixed mtime as well, because mtime tokens over a re-downloaded copy would fork a fresh `run_key` per run and silently destroy prediction-store idempotency.
- **State branch**: `fetch_state.sh`/`publish_state.sh` port unchanged; Woodpecker's clone has no push credential, so a Gitea token is provisioned as a secret and wired into the push remote.
- **PR comment**: `gitea_pr_comment.py`, a faithful port of `pr_comment.js` to Gitea's `issues/{index}/comments` API, same marker, rendered purely from `run_results.json` + `state_diff.json`; the GitHub runner cost line is replaced with total execution time.
  PR builds register into the per-run `ci` registry, so champion gates render "none (bootstrap)" there - the comment shows the PR's own metrics and gates, not a comparison against the shared production champion.
- **promotions.yml lint step**: rejects entries without a pinned `version:` pre-merge (unpinned staging-to-production replays exit 1 by design, because promotion vacates the staging alias).
- **Image bake**: a `docker:29.8.0-cli` step drives the host daemon through the mounted socket (which is why the repo is marked trusted): `docker build -f deploy/Dockerfile`, push to zot over the published localhost port, digest read back from `RepoDigests`, then the `deploy-digest` step commits it to the deploy repo's `images.env` and `k8s/score-cronjob.yaml` as the CI bot. Gated on "this merge retrained something" (see the implementation notes at the top).

## 7. Observability design

**Mechanism: Pushgateway, implementing docs/tutorial.md step 14 verbatim** rather than inventing a spec.
Every mbt surface is a batch job that exits, so there is no live scrape target; Pushgateway persists last-known gauges per grouping key and stamps `push_time_seconds`, which the staleness alert needs.

- `scripts/push_metrics.py` (stdlib-only, baked into the runner image, zero new mbt source code so zero coverage-gate exposure) parses `target/run_results.json`, NOT stderr events (monitor values deliberately do not travel as typed events), and pushes gauges grouped by `(job=mbt, project, target, command, node)`, every series labeled with the spec's `owner` when the manifest carries one.
- Metric names: `mbt_node_success`, `mbt_node_duration_seconds`, `mbt_test_metric{metric=}`, `mbt_realized_metric{metric=}`, `mbt_gate_passed`, `mbt_gate_margin{kind=threshold|champion|ground_truth}` (signed headroom, so alert rules never duplicate spec thresholds), `mbt_shift_value{monitor=,subject=,measure=}`, `mbt_shift_threshold`.
- The four canonical alert rules, in `compose/prometheus/rules.yml`: `MbtGateFailed`, `MbtScheduleStale`, `MbtGateNearBreach` (margin below 0.02) and `MbtShiftBreach` (`for: 10s`). The staleness window is a fixed 30 minutes, so a demo can show it without waiting the tutorial's suggested 8 days.
- Grafana: file-provisioned Prometheus datasource + one dashboard (Model Health: gate margins, realized metrics, shift vs threshold, node durations).
- Routing is deliberately out of scope, as generic plumbing rather than an mbt differentiator (section 1): the rules evaluate in Prometheus and no Alertmanager or Grafana contact point is provisioned, so a firing alert is visible in the Prometheus UI and API but notifies no one. The owner-vs-on-call routing lesson is demonstrated where mbt's exit codes are - `run_mbt.sh`'s webhook classification in CI and `AirflowFailException` in the DAGs.
- Tests assert via the Prometheus HTTP API - the four rules are loaded, and `ALERTS{alertname="MbtShiftBreach"}` appears (pending or firing) after an injected shift - never on notification delivery timing.

## 8. The E2E test tier (the answer to "what kinds of tests")

### Conventions

- Marker `live_showcase` under the existing `live` umbrella; double gate exactly like `live_snowflake`: module-level skipif unless `MBT_LIVE_SHOWCASE=1`, then `pytest.fail` loudly if docker is missing or its daemon unreachable.
- Modules split per concern and share ONE session-scoped stack (all six profiles; the `showcase_stack` fixture in `tests/conftest.py`), because booting the whole platform once per module would multiply its startup and seeding cost; unique basenames; helpers in `showcase_utils.py`. The k3d and make modules add their own gates, and the make module boots a second, isolated stack through the Makefile, so it runs in its own pytest invocation.
- Compose project names are per-session (`mbt-show-<uuid8>`, and `mbt-make-<uuid8>` for the runbook tier) on free ephemeral ports; the `/workspace` bind mount is a pytest tmp dir; all other state lives in compose-project-scoped volumes (the repo-root session guard stays green); teardown is `down -v --remove-orphans` in a finally block, skipped only when `MBT_SHOWCASE_KEEP=1` asks to keep the stack for a post-mortem.
- Failure evidence is dumped before teardown, because nothing survives it: `wait_pipeline` prints the logs of every non-successful Woodpecker step, `wait_dag_run` prints every failed Airflow task attempt, and a failed `compose up` or make target prints per-service log tails.
- The nightly CI job lives in `live.yml` (schedule + manual dispatch, never PRs) on ubuntu-latest (macOS runners have no docker): the main tier, then the make runbook tier after the session stack is gone. The k3d module stays local-only.

### Test catalog

Each row states what the tier ASSERTS today; nothing below is aspirational.

| ID | Kind | What it proves |
|---|---|---|
| SHOW-01 | infra smoke | `compose up --wait` brings every healthchecked service up; MLflow answers `/health` over HTTP; the Spark master reports an ALIVE worker; boto3 round-trips an object against SeaweedFS from inside the runner; the lake holds the seeded tables; the filer UI lists both buckets and the seeded tables without credentials while the raw S3 port refuses an unsigned request with 403; and with the registry stopped, `mbt build` exits 1 - the infra half of the exit-code contract, never 2 |
| SHOW-02 | env sanity | `mbt --version` and `git` work in the runner image; **h2o python client version == pysparkling-embedded H2O version** (H2O requires an exact match; `build_image.sh` pins `h2o==3.46.0.6` to match `h2o-pysparkling-3-5`, and `tests/test_showcase_image_pins.py` holds that pin against mbt-h2o's declared range in the fast suite) |
| SHOW-03 | DS loop (closes A2+A3) | `mbt parse` and `mbt build --target dev` exit 0; `churn_automl` registers to `staging` and the alias, read back over HTTP MLflow, carries `mbt.gates_passed=true`; `churn_baseline_xgb` succeeds; model artifacts exist under `s3://mbt-artifacts/churn_lake/` |
| SHOW-04 | cluster training | `mbt build --target prod --select +churn_automl` (pushdown on `spark://spark-master:7077`, sparkling H2O in the executors) succeeds, registers a newer version that takes the `staging` alias, and the Spark master API shows the application |
| SHOW-05 | slim CI | A one-gate-edit PR: pipeline green; the PR comment's modified lines are exactly `churn_automl` (config) plus its scoring node (upstream), with no dataset and nothing added across fresh clones (URI snapshot stability); the shared registry is untouched. Merging it retrains only `churn_automl` (v2 takes `staging`; the xgboost baseline keeps one version) and pins a freshly baked unit |
| SHOW-06 | state economy | The first-ever push honors `fetch_state.sh` exit 3 and full-builds (both models v1 in `staging` with the gate stamp, the baseline on `mbt-state`, the first unit baked and pinned); a no-change PR comments "Nothing modified" and its merge republishes an identical baseline with zero registry churn and the deploy-repo HEAD untouched |
| SHOW-07 | negative: gates | An impossible-gate PR fails the pipeline; the comment shows `gate_failed` and "registration blocked"; webhook-sink holds exactly one alert, classified `quality-failure`, notifying the owner (`growth-ds@example.com`) and naming the node; the shared registry is unchanged |
| SHOW-08 | provenance | For the first bake: the oras artifact holds exactly `manifest.json` + `run_results.json`, its manifest is byte-identical to the `mbt-state` baseline of the same source sha, `generated_at == anchor`, the recorded git commit is that sha, and neither the S3 secret nor the CI token appears; the unit ships the same manifest bytes, and `mbt run --manifest` inside the pulled unit reproduces xgboost metrics exactly and H2O within the documented 0.02 tier |
| SHOW-09 | negative: env drift | A forged `mbt-*` distribution in the unit makes `mbt run --manifest` exit 1 with the `env_digest` mismatch message; `--allow-env-mismatch` runs it with the same message as a warning |
| SHOW-10 | GitOps promotion | CLI side: a pinned `promotions.yml` moves the `production` alias, its replay is idempotent, and an unpinned entry afterwards exits 1 naming `staging`. CI side: with CODEOWNERS + branch protection on `main`, the DS persona's direct push is rejected, the unapproved merge is refused, and the code owner's approval lets the merge run the promote pipeline, which moves the alias while the deploy repo HEAD and the pinned digest stay byte-identical. (That `mbt promote` refuses a version without the gate stamp is pinned by core's unit tests, not re-proven here.) |
| SHOW-11 | champion resolution | Two `mbt_score` DAG runs straddling a promotion serve different champions (the prediction sidecar's `model_version`) while the image digest and the deploy-repo HEAD stay byte-identical (promotion is a registry event, ADR-20/ADR-5) |
| SHOW-12 | idempotency | A score at the anchor writes one `run_key` directory whose sidecar names the promoted version and a `scored_at` on the anchor date; the same anchor again leaves the same single directory (overwrite); a new anchor adds a second; every run directory carries `_SUCCESS` |
| SHOW-13 | ground truth | `mbt monitor` at the maturity anchor succeeds with an "evaluated" message; the same anchor again evaluates "0 matured prediction runs" (exactly-once); a fresh prediction run monitored with an impossible realized floor (`--vars pr_auc_floor: 0.99`) exits 2 with `monitor_failed`. At the scheduler: that breach fails the `mbt_monitor` task on try 1 with no retry, a clean re-run then succeeds, and a hard error (nonexistent target) consumes the retry (`try_number == 2`) |
| SHOW-14 | observability | After a scored run is pushed, Prometheus holds `mbt_node_success{command="score"}` for the scoring node and `mbt_shift_value{monitor="feature_shift"}`; all four rules are loaded; Grafana's health check passes; an injected shift makes `mbt score` exit 2 (`monitor_failed`), `mbt_shift_value >= mbt_shift_threshold` returns series, and `ALERTS{alertname="MbtShiftBreach"}` appears |
| SHOW-15 | meta: collection hygiene | Every gated showcase module, discovered by filename rather than listed, keeps the opt-in skipif and the `live`/`live_showcase` marks; once opted in, a missing docker binary FAILS rather than skips (the double-gate contract). Runs in the fast suite, where a lost gate would otherwise boot a stack |
| SHOW-16 | optional: CD fidelity | k3d tier only: ArgoCD syncs the deploy repo's `k8s/` CronJob pinned to the same digest CI wrote for Airflow; a pod pulls that unit from zot over insecure HTTP and runs `mbt --version`; a digest change pushed to the deploy repo rolls the CronJob; selfHeal recreates a deleted CronJob |
| SHOW-17 | monthly cadence | `tag:monthly` trains on the prod_score plane (DuckDB over the synced lake, no cluster) with its gates passing; promotion stamps a gate-verified champion; the month-start batch scores with that champion (scoring success means both shift monitors passed); its labels mature at the pinned monitor anchor (`pr_auc > 0.15`, `roc_auc > 0.5`) and evaluate exactly once; the `mbt_score_monthly` DAG scores the batch from the scheduler with the champion while the Spark master sees no new application |
| SHOW-18 | runbook fidelity | Extra gate `MBT_LIVE_SHOWCASE_MAKE=1`: the README golden path through `make` on an isolated `SHOWCASE_PROJECT` - up; demo (daily, monthly and wide prediction runs exist); wide (drift report, selection report, exported reference); seaweedfs (a staged prediction run); ci plus the repo page and the OAuth login its output instructs; monthly; score; monitor; inject-drift and recovery; down leaves no containers; `make workspace` re-stages over the previous run's root-owned output; clean removes the workspace |
| SHOW-19 | wide single-relation cadence (ADR-29) | The probe builds the panel (joined upstream) via Spark on the dev target with its gate passing and `login_days_30d` + `txn_cnt_30d` among the top four importances; `--vars sample_fraction` at 1.0/0.5/0.2 partitions the materialization key, keeps whole customers, and yields strict subsets, with two planes at one fraction agreeing on membership; sparkling AutoML on prod trains the selected columns with gates passing and promotes; the panel's label-free twin scores the newest cohort; its outcomes evaluate once at the monitor anchor (`pr_auc > 0.2`, `roc_auc > 0.5`) |
| SHOW-20 | batch monthly hardening | The ds-helper funnel reproduces the committed `churn_wide_automl.yml` byte for byte, keeps `contract_code`, and reports positive importances, a non-empty zero-importance drop, CV ROC-AUC above 0.6 and seed 42; the committed notebook executes via nbconvert and leaves the committed contract untouched; the Evidently train gate passes, renders the report, and exports a reference holding exactly the selected features; the serving gate passes on the scored batch and exits 2 on a poisoned one, naming drift scores; the `mbt_score_wide` DAG runs sync -> score -> gate from the scheduler and leaves the batch and report on the mount. (The train-phase breach's exit 2 is pinned hermetically in `tests/test_showcase_wide_scripts.py`.) |
| P7 | warehouse plane | Triple-gated: the wide cadence on `--target snowflake` builds with gates passing, registers as `churn_wide_automl_snowflake`, promotes, scores and monitors on the warehouse, and materializes a panel with the same row counts as the DuckDB plane |
| P8 | object-store plane | The wide cadence on `--target seaweedfs`: the panel builds off s3a with non-empty splits and gates passing; versions register as `churn_wide_automl_seaweedfs`; promote, score (every monitor passing, a staged run under `predictions_root`) and monitor (`pr_auc > 0.2`, then exactly-once) run with no synced copy; the panel's row counts equal the DuckDB plane's |

## 9. Known constraints this design respects (do not "fix" silently)

- **No longer a constraint (was: "`mbt score`/`mbt monitor` cannot use the spark data adapter").** `mbt-spark` implemented contract 1.1 (`build_scoring_input`, `open_predictions`) in `6c399c9`, and the candidate follow-up this bullet used to name became P8: the `seaweedfs` target scores and monitors straight off the object store with no sync hop. What still holds is that `prod_score` keeps its synced copy on purpose - it is the cluster-free DuckDB plane the monthly cadence runs on, a second engine by choice rather than by constraint, so do not "fix" the sync away either.
- Remote cluster is train-time only: evaluate/score load MOJOs in a local JVM by design; never promise cluster-side scoring.
- `--manifest` reads local files only; the manifest travels baked inside the image (which is the design), while `--state` accepts s3:// URIs.
- `mbt clean --artifacts-older-than` prunes file:// stores only and nothing protects champion objects server-side: **SeaweedFS buckets are created with no retention/TTL by the seed path** (`bootstrap/seed_lake.py`), not just documented.
- SeaweedFS capacity is pinned in the compose command (`-master.volumeSizeLimitMB=64 -volume.max=100`), never left to its defaults. Those size volumes at 1GB and cap their count at free disk / volume size, while each bucket grows 7 volumes on first write, so on a docker disk with ~7GB free the lake bucket took every slot and every model upload failed with S3 `InternalError`. `tests/test_showcase_image_pins.py` holds the flags; `docs/troubleshooting.md` has the symptom.
- Switching an existing target's artifact-store scheme strands registered champions (fetch rejects cross-scheme refs); the showcase never flips schemes mid-life.
- On an s3:// artifact store MLflow holds pointers (`mbt.artifact.*` tags) and mbt's small documents such as `inference_config.json`, never the model binaries (mbt uploads a model file to MLflow only from a file:// store); the MOJOs live in `s3://mbt-artifacts/`, and the README points humans at the filer UI to see them.
- The dev uv lock resolves pyspark 4.x; the runner image builds the sparkling fork (`mbt-h2o[sparkling]` pins pyspark 3.5.x) in its own resolution with `uv export --frozen` constraints, and the cluster runs the matching Spark 3.5.8 binaries.
- The image build resolves NOTHING fresh, and that is load-bearing rather than tidiness: the base is digest-pinned, mbt's dependencies - evidently among them, through `mbt-evidently` - come from `uv.lock`, and the one dep that is neither (jupyterlab) has its closure pinned in the committed `images/runner/image-extras.txt`.
  Before that file existed, the pip layer resolved against live PyPI on every build, so an upstream release minutes earlier could break it - which is exactly what happened on 2026-08-27, when statsmodels 0.15.0 shipped macOS wheels at 10:34 UTC and Linux cp311 wheels at 14:30, and the 14:12 nightly fell back to the sdist against an image that carries no compiler.
  `scripts/lock_image_extras.sh` regenerates the closure against uv.lock's pins; `tests/test_showcase_image_pins.py` fails in the fast tier if the two drift apart.
- Woodpecker trusted-repo volumes, the docker socket on the agent, and insecure-registry config are demo-tier security postures; the README says so.
- Every service image is pinned to an exact version, and bumps are verified by running the tier, never taken on faith: Renovate is not active on this repo, so nothing moves these pins automatically, and a rolling tag would let upstream change the stack overnight with no diff (Gitea >= 1.26's `PUBLIC_URL_DETECTION` default flip was exactly that). `zot` stays on v2.1.16 on purpose until project-zot/zot#4140 is fixed in a release (the compose file carries the reason).

## 10. Repo layout

Everything lives in this repo under `examples/showcase/` (yamllint covers `examples/`; the coverage gate scopes only `packages/`; tests must live in repo-root `tests/` per testpaths).

```
examples/showcase/
  README.md                    # runbook: make targets, the DS/plane/CI walkthroughs, test tiers,
                               # deviations from the scaffold defaults, knobs and RAM budget
  DESIGN.md                    # this file
  Makefile                     # help, image, workspace, up, seed, urls, demo, ci, score, monthly,
                               # wide, monitor, inject-drift, seaweedfs, snowflake-seed, snowflake,
                               # snowflake-drop, down, clean
  .env.example                 # host ports, S3 credentials, workspace, image tag, DOCKER_SOCK_GID
  images/runner/{Dockerfile,entrypoint.sh}
  images/runner/image-extras.{in,txt}  # the non-mbt image deps and their pinned closure
                               # (constraints.txt is derived from uv.lock at build time,
                               #  by scripts/build_image.sh - it is not a committed file)
  compose/docker-compose.yml   # profiles: core, spark, dev, obs, ci, orch
  compose/{seaweedfs,prometheus,grafana}/...
  bootstrap/{seed_lake.py,sync_lake.py,inject_drift.py,webhook_sink.py}  # staged into /workspace
  data/                        # committed monthly + wide parquet (the daily tables come from
                               # tests/fixtures/churn_demo/data)
  scripts/                     # host-side: build_image.sh, lock_image_extras.sh, ci_bootstrap.py,
                               # generate_monthly_data.py, generate_wide_data.py, seed_snowflake.py
  project/                     # the churn_lake mbt project (source of truth; pushed into Gitea)
    .woodpecker/{pr-check.yml,prod-build.yml,promote.yml}
    datasets/ models/ scoring/ sources.yml profiles.yml promotions.yml
    deploy/Dockerfile          # the deployable unit: runner image + project + compiled manifest
    notebooks/ds_inner_loop.ipynb            # the DS workbench narrative, executed by the live tier
    notebooks/ds_inner_loop_snowflake.ipynb  # the same loop on the warehouse plane (host kernel)
    scripts/{run_mbt.sh,push_metrics.py,gitea_pr_comment.py,lint_promotions.py,
             fetch_state.sh,publish_state.sh,select_features.py,evidently_gate.py}
  deploy/                      # the deploy repo source: images.env, dags/, k8s/ (for the k3d tier)
tests/
  test_showcase_infra.py       # SHOW-01/02 + the registry-outage exit-1 contract
  test_showcase_ci.py          # SHOW-05/06/07/10 (CI side)
  test_showcase_lifecycle.py   # SHOW-03/04/10/11/12/13 (CLI-driven)
  test_showcase_provenance.py  # SHOW-08/09
  test_showcase_scheduling.py  # SHOW-11/13 + SHOW-17/SHOW-20 scheduled paths
  test_showcase_obs.py         # SHOW-14
  test_showcase_k3d.py         # SHOW-16 (extra gate, local-only)
  test_showcase_monthly.py     # SHOW-17
  test_showcase_make.py        # SHOW-18 (extra gate, own invocation)
  test_showcase_wide.py        # SHOW-19/SHOW-20 (ADR-29)
  test_showcase_seaweedfs.py   # P8 object-store plane
  test_showcase_snowflake.py   # P7 warehouse plane (extra gates)
  test_showcase_gates.py       # SHOW-15 (hermetic: fast suite)
  test_showcase_image_pins.py  # hermetic: runner image pins, extras closure, host S3 credentials
  test_showcase_seaweedfs_plane.py  # hermetic: the object-store target's addressing
  test_showcase_wide_scripts.py     # hermetic: funnel, Evidently gate, categoricals, notebooks
  showcase_utils.py
```

## 11. Phasing (each phase independently valuable and tested)

1. **P1 Runner image + data/ML core**: seaweedfs, mlflow, spark, jupyterlab, seeded lake, dev/prod targets. Tests SHOW-01..04. Closes integration items A2 + A3.
2. **P2 Git + CI loop**: gitea, woodpecker, bootstrap seeding, pipelines, exit-code wrapper, Gitea PR comments, state branch. Tests SHOW-05..07.
3. **P3 Deployable unit + provenance**: zot, bake + push, commit-time anchors, oras artifacts, the tamper provenance test. Tests SHOW-08..09.
4. **P4 Scheduling + CD + promotion**: airflow + postgres + git-sync, deploy repo, DAGs with exit-code routing, prod_score plane, promotion flow. Tests SHOW-10..13.
5. **P5 Observability + docs**: prometheus/pushgateway/grafana, dashboards, rules, drift injection; docs page in mkdocs nav; one exactly-true sentence in v0.1-status; nightly workflow. Tests SHOW-14..15.
6. **P6 (optional) ArgoCD fidelity tier**: k3d + ArgoCD over the same deploy repo. Test SHOW-16, local-only.
7. **P7 Snowflake warehouse plane**: IMPLEMENTED 2026-08-28 (parked 2026-07-16, unparked on request).
   Decided direction, unchanged: Snowflake is read-only data storage (source tables, scoring batches, ground-truth labels); predictions, registry, and artifacts stay on mbt's side, so there is no warehouse-resident prediction store.
   Predictions stage as parquet under `predictions_root` (ADR-23 v1); the warehouse-native store is v2, issue #1.
   The plane runs the FULL loop - build, gate, register, promote, score, monitor - because P7(a) turned out to be already done: `mbt-snowflake` implements contract 1.1 (`build_scoring_input`, `open_predictions`).

   **P7(b) was superseded, and that is the load-bearing change.**
   The parked scope called for a separate host-run project BESIDE the showcase, on the grounds that "snowflake sources use `identifier:` and cannot build on the spark `ci`/`dev` targets".
   That assumed a table must choose one addressing scheme.
   It does not: `SourceTable` rejects only a table declaring NEITHER `path:` nor `identifier:`.
   So every table in `sources.yml` now carries both, and the Snowflake plane is a TARGET INSIDE this project - same DAG, same dataset/model/scoring specs, not one line of spec duplicated.
   Switching planes is `--target snowflake`, so the wide shape's data-plane independence is enforced by a test rather than asserted in prose.

   Consequences worth knowing:

   - ALL 14 relations carry an identifier, not just the three the wide cadence reads (`monthly_panel`, `monthly_panel_scoring`, `wide_churn_outcomes`). Compile pins a snapshot for every source a dataset or scoring node references (nine, across the three cadences) regardless of `--select`, so a missing referenced table fails the compile before selection narrows anything; the five gold tables no node references are never pinned, and carry identifiers because the seeder creates them.
     The seeder therefore CREATES all 12 base tables and the 2 panels, but loads rows only into the six the panels and the monitor need (`WIDE_TABLES`: the five gold tables the CTAS joins, plus `wide_churn_outcomes`; held against the specs and the panel SQL by a test). Pinning is a metadata call, so the daily and monthly cadences' six tables stay empty rather than putting ~26k rows of unrelated demo data in the operator's sandbox. `--all-cadences` loads everything, and is the fallback if an account will not pin a never-written table.
   - The two panels mbt reads (ADR-29) are materialized by the seeder with a plain CTAS over the five gold tables, not as dynamic tables. ADR-29 requires a PHYSICAL relation rather than a view (a view's change token is DDL-blind, so a re-deploy that adds a column moves neither of mbt's hashes); a materialized table satisfies that exactly as a dynamic table does, and gives up only auto-refresh, which a plane whose data is static at a pinned anchor never uses. What it buys is the privilege floor: seeding needs nothing beyond `CREATE TABLE`, so the plane runs on a sandbox role that cannot create a dynamic table. A real deployment should still own the panel as a dbt model, where incremental refresh shared across every reader is the point.
   - Registered names are namespaced per plane via the `plane_suffix` var (`""` everywhere, `"_snowflake"` on the new target). Both planes train the same spec, and without this their versions would interleave in the shared registry and quietly corrupt champion resolution. It reaches the spec through `var()`, so it enters the config hash by design - the two planes are genuinely different nodes.
   - Host-run, still. `externalbrowser` SSO needs a real browser and a localhost callback, and the runner image does not ship `mbt-snowflake` (its sparkling extra pins pyspark 3.5.x, which does not resolve cleanly against the connector's `cryptography>=46.0.5` floor). The target reaches the stack's MLflow and S3 over published ports, so `make snowflake` runs mbt on the host rather than through `$(EXEC)`.
   - Adding `identifier:` is state-neutral for the lake planes: source config lives on `ManifestSource`, and node `config_hash` covers only node config. Verified by diffing every resolved node config before and after the change.
   - It is NOT read-neutral for Spark, which is the one adapter that reads both object-store paths and catalog tables.
     The local and Snowflake adapters each read one field and ignore the other, so for them a both-addresses table is unambiguous; Spark refuses to guess, so every spark target (`dev`, `ci`, `prod`, and P8's `seaweedfs`) sets `source_address: path`.
     This was learned the hard way: the first cut of P7 shipped without it, Spark's `_read` silently preferred `identifier` (while `snapshot_id` kept hashing the `path`, so the pin described data the run never read), and the whole lake plane went looking for `MBT_SHOWCASE_*` catalog tables that do not exist.
     Push CI stayed green - every mbt-spark test was JVM-gated behind the `e2e` marker - and only the nightly live tier caught it, which is why the precedence rule now has a fast-tier test of its own (`packages/mbt-spark/tests/test_spark_source_address.py`).

   Testing is two-tier, and the hermetic half is the one that runs by default.
   `packages/mbt-snowflake/tests/test_showcase_snowflake_plane.py` builds the committed wide spec through the real Snowflake adapter with its SQL executed in DuckDB (no account), and holds the both-addresses invariant plus seeder/sources agreement.
   `tests/test_showcase_snowflake.py` is triple-gated (MBT_LIVE_SHOWCASE=1 + MBT_LIVE_SNOWFLAKE=1 + complete SNOWFLAKE_*) and proves the loop on a real account, including a cross-plane assertion that both planes materialize the same panel.
   The triple gate keeps the hermetic grand-suite guarantee intact: the showcase tier still needs docker and nothing else.

8. **P8 Object-store plane**: IMPLEMENTED 2026-09-11.
   The same wide cadence as P7, sourced straight out of SeaweedFS instead of a warehouse: `--target seaweedfs`, one word, no spec edited.

   **What it exists to prove, which P7 could not.**
   P7 closed the serving leg against a second data plane, but only behind three gates - `MBT_LIVE_SHOWCASE=1` + `MBT_LIVE_SNOWFLAKE=1` + complete `SNOWFLAKE_*`.
   On a machine with no account the showcase's "one project, many data planes" claim reverted to prose.
   This target needs docker and nothing else, so the claim is enforced by a test on every run of the default tier (`tests/test_showcase_seaweedfs.py`).

   **The gap it actually fills is the serving leg, not the training one.**
   `dev`/`prod` already trained the wide cadence off s3a, so the build half was covered.
   Every batch leg in the showcase, though, ran on `prod_score` - the LOCAL (DuckDB) adapter over `/workspace/lake_local`, which `bootstrap/sync_lake.py` mirrors out of the bucket.
   So nothing here ever read a live object store at score time, and an s3a read that broke at serving would have been indistinguishable from one that worked.
   `mbt score --target seaweedfs` materializes the batch by reading the bucket and stages the run under `predictions_root` (ADR-23 v1); `mbt monitor` reads the matured labels the same way.
   That exercises `mbt_spark`'s contract-1.1 methods (`build_scoring_input`, `open_predictions`), which shipped in `6c399c9` and had unit coverage only.

   Consequences worth knowing:

   - `prod_score` stays, and its justification changed rather than expired. It used to be documented as a workaround ("mbt-spark has no contract-1.1 scoring methods") - false since `6c399c9`, and the stale comment is corrected in `profiles.yml` and `bootstrap/sync_lake.py`. What it demonstrates is the cluster-free batch plane, which is the whole plane the monthly cadence (SHOW-17) runs on.
   - **No `--deep-snapshot` on this plane, deliberately.** For a URI source `SparkDataAdapter.snapshot_id` hashes the table's input-file listing, which is already mtime-independent, so deep and shallow tokens agree; ADR-11's fresh-checkout problem is a local-path problem. The other recipes pass the flag because they read a downloaded copy whose mtimes really do move. Mixing the two schemes on one pipeline is exactly what the "one token scheme per pipeline" rule forbids.
   - Registered names are namespaced `_seaweedfs` through the same `plane_suffix` var P7 introduced, with its own experiment (`churn_lake__seaweedfs`) and artifact prefix (`s3://mbt-artifacts/churn_seaweedfs`). Three planes now share one registry without interleaving.
   - It runs IN the stack, unlike P7. There is no host-side dependency to justify anything else, so `make seaweedfs` goes through `$(EXEC)` like every other recipe.

   Testing is two-tier, as P7's is, and here BOTH tiers run by default.
   `tests/test_showcase_seaweedfs_plane.py` is hermetic - it holds the target's object-store addressing, the per-plane namespacing across all six targets, and the no-credentials property, in the fast suite.
   That module is not optional politeness: `-m e2e` excludes the `live_showcase` tier and `-m "not e2e"` deselects it, so a full five-phase battery can go green while every showcase spec is broken - which is what happened on the ADR-29 sweep (26 live failures from one root cause, invisible to five green phases).
   `tests/test_showcase_seaweedfs.py` proves the loop on the real stack, including the cross-plane assertion that the object-store and DuckDB planes materialize the same panel.

## 12. Open questions

1. RESOLVED - Zot addressing: the docker daemon uses the published localhost port (insecure-by-default, no daemon config) and in-network consumers use `zot:5000`; see the implementation notes at the top.
2. RESOLVED - the nightly CI job affords the whole main tier, sparkling modules included, plus the make runbook tier on ubuntu-latest (the job has a 120-minute timeout and has run in under an hour); only the k3d tier stays local-only.
3. Whether to eventually ship `mbt init --forge gitea` scaffolding (Woodpecker pipelines + Gitea PR-comment script) upstream once the showcase proves the port; the showcase keeps them project-local until then.
