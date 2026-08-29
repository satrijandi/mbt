# mbt showcase: full-stack docker reference environment + E2E test tier (design)

Status: IMPLEMENTED (all phases; P6 separately gated).
P1 (runner image + data/ML core), P2 (Gitea + Woodpecker CI loop incl. branch protection + CODEOWNERS on promotions.yml), P3 (Zot deployable unit + oras provenance), P4 (Airflow + git-sync CD + the scoring/promotion/monitoring plane), P5 (observability), and P6 (k3d + ArgoCD, local-only behind its own MBT_LIVE_SHOWCASE_K3D gate) are implemented and covered by the `live_showcase` test tier; see README.md for what runs today.
The test catalog below maps onto ten modules: `tests/test_showcase_infra.py` (SHOW-01/02/15), `tests/test_showcase_ci.py` (SHOW-05/06/07/10), `tests/test_showcase_lifecycle.py` (SHOW-03/04 + 10/11/12/13 CLI-driven), `tests/test_showcase_monthly.py` (SHOW-17), `tests/test_showcase_obs.py` (SHOW-14), `tests/test_showcase_provenance.py` (SHOW-08/09), `tests/test_showcase_scheduling.py` (SHOW-11 strong form + SHOW-13 routing + SHOW-17's and SHOW-20's scheduled paths), `tests/test_showcase_k3d.py` (SHOW-16), `tests/test_showcase_make.py` (SHOW-18, its own gate), `tests/test_showcase_wide.py` (SHOW-19/SHOW-20, ADR-22).
Modules share one session stack and run in collection (alphabetical) order; the only load-bearing constraint is that `test_showcase_ci` is the first forge consumer (virgin-bootstrap assertion) - everything else provisions or promotes what it needs and scopes score/monitor by cadence tag.
Implementation notes (deliberate scoping vs the sections below):
pr-check builds on the `ci` target and the merge-time prod-build on `dev` (spark local[2] + the shared registry); the scheduled retrain DAG is the cluster-from-CI path (prod target from a pinned unit), tested with the deterministic xgboost workhorse - sparkling stays confined to SHOW-04's module and the wide module (SHOW-19) per the flake-isolation rule.
Baking is gated on "this merge retrained something": docker layer digests embed mtimes, so an unconditional bake would mint a new digest per merge and break the ADR-20 "promotion deploys nothing" claim.
SHOW-05's "exactly one modified node" reads as "exactly the edited model (config) plus its downstream scoring node (upstream)" - scoring depends_on its model, so the lineage flag is correct behavior.
Zot is addressed two ways for one digest: the docker daemon pushes/pulls via the published localhost port (insecure-by-default, the open-question-1 fallback, smoke-tested on Docker Desktop for macOS), while in-network consumers (oras, k3d) use zot:5000.
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

**Compose-first, one runner image, k3d/ArgoCD as an optional fidelity profile.**
This synthesizes three competing designs that were independently drafted and adversarially judged; the judges' consensus and every confirmed fatal-flaw fix are folded in below.

- One `docker-compose.yml` under `examples/showcase/compose/` with profiles `core`, `spark`, `orch`, `obs`, `dev`, `ci`, so tests boot only what they assert.
- One bridge network; Woodpecker step containers join it (`WOODPECKER_BACKEND_DOCKER_NETWORK`) so CI steps resolve `spark-master`, `seaweedfs`, `mlflow` by name.
- **One runner image** used everywhere: Jupyter kernel, Spark master/worker runtime, every Woodpecker step, Airflow task container, and the deployable unit base.
  Same image everywhere makes ADR-19 `env_digest` verification true by construction instead of an ops problem.
- **CD default is a git-sync reconciler**, not ArgoCD: a `deploy` repo in Gitea holds `images.env` (digest pins) + DAG files, reconciled into Airflow every 30s by a `git-sync` sidecar.
  This is ArgoCD's essential loop (git as source of truth, auditable rollback = `git revert`) at a fraction of the RAM, and it is honest: ADR-20 means the frequent release event (model promotion) is a registry alias flip, never a deploy, so the only thing left to CD is which runner-image digest the scheduled jobs run.
- **Optional `argocd` profile** for literal fidelity: `k3d cluster create --network <compose-net>` (k3d attaches to the compose network; pods resolve compose services via CoreDNS forwarding), ArgoCD core install syncing the same deploy repo rendered as CronJobs, `registries.yaml` marking `zot:5000` insecure-HTTP.
  Demo-only, +~3GB RAM, smoke-tested behind its own gate, never on the test-critical path.

RAM budget: ~4.5GB steady state for the default profiles, 8-9GB peak during sparkling training, +~3GB with the argocd profile.
Guardrails are config, not prose: `WOODPECKER_MAX_WORKFLOWS=1`, `SPARK_WORKER_MEMORY`, executor caps, `h2o_max_mem` per target, and a README knob table.

### Two load-bearing shared mounts

The Spark adapters have a hard driver-local-filesystem assumption (ADR-17): split staging writes `coalesce(1)` output to a driver-local temp dir that executors must also see, and `_materialize_for_path_adapter` dirs are read by `h2o.import_file` on executors.

1. `/workspace` (fixed-name **external** docker volume, created and destroyed by the fixture/Makefile): mounted at the identical absolute path into JupyterLab, every Woodpecker training step (trusted repo volumes are literal strings in committed YAML, hence the fixed name), every Spark worker, and Airflow task containers. `TMPDIR=/workspace/tmp/<pipeline-id>` wherever mbt runs a Spark driver.
2. `/workspace/lake_local`: the scoring plane's local data root, populated by an `aws s3 sync` step from the lake bucket (see 4.3).

Every mbt-running container has an entrypoint preflight that hard-fails if the mount is missing or unwritable, converting the misleading distance-failure `split ... materialized 0 rows` into an immediate diagnosable error.

### Spark driver reachability

Spark standalone does not support cluster deploy-mode for Python apps, so every driver runs client-mode inside a dynamically named container and executors must connect back.
The runner entrypoint exports `SPARK_DRIVER_HOST=$(hostname -i)`; profiles fix `spark.driver.host: "{{ env_var('SPARK_DRIVER_HOST') }}"`, `spark.driver.port: 40400`, `spark.blockManager.port: 40401` (one driver per container IP, so fixed ports are safe).

## 3. Service inventory

| Service | Image | Role |
|---|---|---|
| runner (built, not run) | `zot:5000/mbt/runner:<tag>` from `python:3.11-slim@<digest>` + JDK 17 + Spark 3.5.8 + hadoop-aws jars + workspace wheels + `mbt-h2o[sparkling]` + `mbt-core[s3]` + `mbt-mlflow` + jupyterlab | Universal environment; env_digest identity by construction |
| gitea | `gitea/gitea:1.27.0-rootless` | Hosts `churn` project repo + `deploy` repo; branch protection + CODEOWNERS gate on `promotions.yml`; `mbt-state` branch storage |
| woodpecker server + agent | `woodpeckerci/woodpecker-*:v3.16.0` | CI: pr-check, prod-build, promote pipelines; agent mounts docker socket; repo marked trusted for the `/workspace` volume; split-horizon URLs (public WOODPECKER_HOST + forge OAuth host, in-network webhook host) so the Gitea OAuth login works from a host browser |
| seaweedfs | `chrislusf/seaweedfs:4.39` (`weed server -s3`) | S3-compatible object store: `mbt-lake` bucket (feature store parquet, read via s3a://) and `mbt-artifacts` bucket (artifact store `s3://mbt-artifacts/churn_lake`); the filer UI is published for humans to browse the lake |
| spark-master, spark-worker | runner image running `start-master.sh` / `start-worker.sh` | Standalone cluster: pushdown joins/sampling/windows + sparkling H2O training |
| mlflow | runner image (`mlflow server`, sqlite on a volume) | Tracking + model registry (alias mode, needs >= 2.9); champion source of truth |
| jupyterlab | runner image + `jupyter lab` | DS workbench; terminal runs the same `mbt` as CI |
| airflow | `apache/airflow:3.3.0` (api-server + scheduler + dag-processor), **LocalExecutor + postgres** (sqlite forces SequentialExecutor; invalid with LocalExecutor) + git-sync sidecar | Schedules retrain/score/monitor DAGs; tasks drive the docker SDK to run the pinned digest from `deploy/images.env` |
| zot | `ghcr.io/project-zot/zot:v2.1.16` (v2.1.17+ breaks multi-GB pushes on slow disks, zot#4140) | OCI registry: runner images, baked deployable units, and oras-pushed `manifest.json`/`run_results.json` provenance artifacts |
| prometheus + pushgateway | `prom/prometheus:v3.13`, `prom/pushgateway:v1.11` | Metrics per docs/tutorial.md step 14 (the documented spec, implemented verbatim) |
| grafana | `grafana/grafana:13` | File-provisioned dashboards + unified alerting with owner-label routing (no separate Alertmanager container) |
| webhook-sink | ~40-line python recorder with `GET /requests` | Convergence point for `MBT_ALERT_WEBHOOK` curls and Grafana contact points; human-readable in demos, assertable in tests |
| bootstrap (one-shot) | runner image running `scripts/ci_bootstrap.py` | Idempotent seeding: Gitea org/repos/tokens/branch protection, Woodpecker activation + secrets, buckets (retention explicitly disabled), seed data upload, repo pushes, image build+push |

Registry addressing: in-network `zot:5000` with documented insecure-registries daemon config is the default; `127.0.0.1:5000` publishing is the fallback and must be smoke-tested on Docker Desktop for macOS first (the daemon lives in a VM there).

## 4. The demo project

An `mbt init`-derived `churn` project, source-of-truth at `examples/showcase/project/`, pushed into Gitea by bootstrap (the showcase never depends on a long-lived external repo).

### 4.1 Models

- `churn_automl` (h2o_automl, `max_models: 3`, `include_algos: [GLM, GBM]`, `nfolds: 0`, fixed seed, no `max_runtime_secs`): the star of the DS story, `tags: [weekly]`, threshold + champion gates, `registration: {name: churn_automl, stage_on_pass: staging}`.
- `churn_baseline_xgb` (xgboost): deterministic workhorse for the CI-loop differentiator tests (bit-exact `--manifest` reproduction; H2O's documented determinism tier is 0.02 tolerance, so H2O reproduction asserts within tolerance, never byte-equality).
  Sparkling + remote master stays confined to its own test module so a flake there never poisons the slim-CI/promotion/idempotency assertions.
- `churn_monthly_xgb` (xgboost, SHOW-17): the monthly batch cadence, `tags: [monthly]`, trained/scored/monitored entirely on the `prod_score` plane (DuckDB over the synced lake, no cluster) with its own 30-day-churn tables (`monthly_*`, month-start snapshots generated by `scripts/generate_monthly_data.py`) and its own gate floor (`monthly_pr_auc_floor`); the spec stays target-portable, so CI's spark targets build it when touched.
- The wide multi-table batch-monthly cadence (SHOW-19/SHOW-20, ADR-22), `tags: [wide]`, on its own tables (`scripts/generate_wide_data.py`: a `monthly_population` spine carrying the customer_id-to-safe_id crosswalk, `monthly_labels` keyed by each cohort's own `inference_date` and present only once matured (the gold-layer label contract; a raw observation-dated feed would instead declare `time_offset`), and three feature histories joined by DIFFERENT entity keys on the one uniform `inference_date` join key - demographics/logins by `(customer_id, inference_date)`, transactions by `(safe_id, inference_date)` - with the spine carrying the entity crosswalk plus the DS-excluded `as_of_date` (balances describe the previous day) and `loaded_at_time` lineage/audit columns, per docs/naming-conventions.md; ~66 joined columns including the numeric-coded `contract_code`, `--customers/--filler-columns` stress knobs):
  `notebooks/ds_inner_loop.ipynb` is the DS workbench narrative for this cadence (explore, probe build, funnel, sampled what-if on a scratch copy), executed top to bottom by the live tier so it cannot rot;
  `churn_wide_probe` (lightgbm) materializes the full-width panel and exports gain importance as a sanity cross-check; `scripts/select_features.py` runs the ds-helper funnel over the materialized train split (drop high-missing, drop single-value, drop correlated pairs, then a seeded LightGBM randomized search keeping importance > 0) and commits the winners into `churn_wide_automl`'s include list between BEGIN/END markers, with every stage documented in `target/feature_selection_report.json` - feature selection as a reviewable diff that slim CI retrains on;
  `models/wide_hooks.py` (shared by both wide models via explicit `hooks:` paths) casts DS-declared numeric-coded categoricals to string before feature filtering, at train and scoring time alike, so every adapter's native categorical handling picks them up;
  the models' `exclude:` list is the DS-declared ignored-columns contract and the funnel honors it: beyond the entity ids it names `tenure_months`, which is anchored to calendar time and therefore breaches the training-time PSI baseline at serving no matter how predictive it looks inside the training window (the funnel found it, the feature_shift monitor rejected it, and the exclusion records that lesson);
  `churn_wide_automl` (h2o_automl, sparkling on prod) trains on the selected columns and serves through `scoring/wide_retention_scoring.yml`, whose population-form input joins the same tables minus the label at scoring time;
  `scripts/evidently_gate.py` (evidently pinned in the runner image, never an mbt dep) enforces feature stability with exit-code-2 semantics beside mbt's own monitors: the train phase (train vs test on the selected features) blocks `mbt promote` and exports the persisted serving baseline, and the serving phase re-checks every scored batch against that baseline, rendering the DS-facing drift report in both phases.

### 4.2 sources.yml (the one-file-serves-all resolution)

One `sources.yml` must serve the Spark training targets, the local-adapter scoring target, and the Snowflake plane:

- Spark data adapter `root: "s3://mbt-lake"` (the `s3://` prefix is exempt from `normalized_adapter_config` project-dir path resolution; a raw `s3a://` root would be mangled).
- Scheme mapping via conf: `spark.hadoop.fs.s3.impl: org.apache.hadoop.fs.s3a.S3AFileSystem`, `fs.s3a.endpoint: http://seaweedfs:8333`, `fs.s3a.path.style.access: "true"`.
- Table paths stay **relative** (`gold/subscribers/*.parquet`), so the local adapter serves the same tables from `root: /workspace/lake_local`.
- Every table ALSO declares `identifier: MBT_SHOWCASE_<TABLE>` for the `snowflake` target (section 11, P7). Declaring both is legal - `SourceTable` rejects only a table with neither - which is what lets one project and one set of specs cover all three planes. The local and Snowflake adapters each read one field and ignore the other; Spark reads either, so the spark targets say which via `source_address: path` (see P7's consequences below). All 12 tables carry an identifier even though only the wide cadence trains on Snowflake, because compile pins every referenced source regardless of `--select`.

### 4.3 Targets (profiles.yml, committed and secret-free)

`profiles.yml` is committed with pure `{{ env_var(...) }}` values (the scaffold gitignores profiles, so CI checkouts have none otherwise); container env supplies `MLFLOW_TRACKING_URI=http://mlflow:5000`, `AWS_ENDPOINT_URL_S3=http://seaweedfs:8333`, keys, region.
Note: boto3's env chain is the only S3 endpoint mechanism (nothing in mbt parses endpoints), so one process talks to exactly one S3 endpoint; fine here since SeaweedFS is the only object store.

| Target | Data | Compute/training | Registry/artifacts | Use |
|---|---|---|---|---|
| `dev` | spark `master: local[2]`, s3a to lake | h2o local backend, `sample_fraction: 0.1` | shared MLflow, `s3://mbt-artifacts/...` | DS fast inner loop |
| `ci` | same as dev | same as dev | **per-run sqlite MLflow + local artifact store** | PR checks: green PRs must never register versions or re-point the shared `staging` alias; tradeoff: champion gates render "none (bootstrap)" in PR comments, documented |
| `prod` | spark `master: spark://spark-master:7077` | `h2o_backend: sparkling`, driver-host conf per section 2 | shared MLflow, `s3://mbt-artifacts/...` | Prod builds, weekly retrain |
| `prod_score` | **local adapter**, `root: /workspace/lake_local` | n/a (MOJO scoring is local-JVM by design; remote cluster is train-time only) | shared MLflow | `mbt score` / `mbt monitor` |
| `snowflake` | **snowflake adapter**, tables by `identifier:` | h2o local backend, on the host | shared MLflow + `s3://mbt-artifacts/churn_snowflake`, both over PUBLISHED ports | The warehouse plane (section 11): the same wide cadence, `--target snowflake`. Runs on the host, not in a container. Registers `*_snowflake` names via `plane_suffix` |

### 4.4 Scoring resource

`scoring/retention_scoring.yml`: `model: churn_automl`, `stage: production`, `tags: [daily]`, input checks, PSI/KS shift monitors, `ground_truth` with a 14-day maturity window and realized-metric gates.
Scheduling lives entirely outside the YAML (there is no schedule field); Airflow selects via `--select tag:daily`.

### 4.5 Anchors and seed data (the determinism spine)

Seed data is generated with fixed RNG seeds, dated 2026-01..06, uploaded to the lake at bootstrap.
Every pipeline, DAG, and test pins anchors to constants: `SHOWCASE_ANCHOR=2026-06-30T00:00:00Z`, `MONITOR_ANCHOR=2026-07-20T00:00:00Z` (past maturity).
Wall-clock anchors over fixed-date data are a time bomb: relative windows like `-150d:-28d` resolve empty within weeks and every unpinned pipeline rots into `split ... materialized 0 rows`.
Airflow DAGs therefore pass `--anchor` from deploy-repo config, never `{{ ts }}`.
Anchor time travel is also what makes monitoring demoable today: `mbt monitor --anchor <maturity+> ` evaluates immediately; re-running with the same anchor evaluates nothing (exactly-once proof).
CI pipelines anchor to the commit timestamp (UTC-normalized), making same-commit rebuilds byte-identical manifests.

## 5. Golden path (the demo narrative)

1. `make up`: build runner image, `docker compose up -d --wait`, bootstrap seeds everything; terminal prints all UI URLs.
2. **DS inner loop**: JupyterLab terminal, `git clone http://gitea:3000/mbt-showcase/churn`, edit the model spec, `mbt build --target dev --anchor $SHOWCASE_ANCHOR`; H2O leaderboard streams through the event bus; run appears in MLflow UI; MOJO lands in SeaweedFS.
3. **DS scales out**: `mbt build --target prod`; pushdown sampling/joins run on the cluster (Spark UI shows the app); model registers, `staging` alias set.
4. **PR**: push branch, open Gitea PR; Woodpecker pr-check runs parse, compile, `fetch_state.sh` (exit 3 = bootstrap), `state diff --output json`, slim build `--select state:modified+ --state ...` under the `ci` target, then posts the update-in-place `<!-- mbt-pr-comment -->` comment via Gitea's API showing exactly one modified node and its gates.
5. **Merge**: prod-build runs the economy build on `prod`, publishes the manifest to `refs/heads/mbt-state` (`publish_state.sh` is pure git plumbing and works against Gitea unchanged), bakes the deployable unit (`FROM` the exact runner tag + project + `target/manifest.json` compiled inside that same image), pushes to Zot, oras-pushes `manifest.json`+`run_results.json` as provenance artifacts (manifests are secret-free by construction), and commits the new digest to the deploy repo.
6. **CD**: git-sync reconciles the deploy repo into Airflow; rollback is `git revert` (optional profile: ArgoCD syncs the same repo into k3d CronJobs).
7. **Schedules**: Airflow runs `mbt score --target prod_score --select tag:daily --anchor ...` daily and `mbt monitor` weekly; exit 1 retries then pages on-call, exit 2 never retries and notifies the model owner (per the tutorial's routing rule); a wrapper pushes metrics regardless of outcome.
8. **Promotion**: MLOps opens a PR editing `promotions.yml` (version always pinned); CODEOWNERS + branch protection gate the merge; the promote pipeline runs `mbt promote --from-file`; the next scheduled score run serves the new champion with zero redeploy, image digest and deploy repo byte-identical before and after (the ADR-20 inversion, asserted).
9. **Monitoring pays off**: injected drift makes `mbt monitor` exit 2, `mbt_shift_value >= mbt_shift_threshold` fires the provisioned alert, Grafana routes to the owner; a stale schedule fires the `push_time_seconds` staleness rule that no in-band mechanism can catch.

## 6. CI design (Woodpecker)

Pipelines live in `.woodpecker/` of the project repo (authored fresh; the GitHub scaffold under `_scaffold/.github/` stays untouched, so `tests/test_cli_basics.py` is unaffected).

- **Exit-code fidelity**: Woodpecker collapses any nonzero exit to "failed", erasing mbt's 1-vs-2 contract.
  Every executing step runs through `scripts/run_mbt.sh`: capture code, write `target/ci_exit_class` (`0 ok / 1 hard-error / 2 quality-failure`), classify the alert payload (1 pages on-call, 2 notifies the owner), always push metrics (best-effort with a 2s timeout so an absent Pushgateway never hangs CI), re-exit with the original code.
  The PR-comment step, the alert step, and pytest all read the same file.
- **Snapshot scheme deviation, documented**: no `--deep-snapshot` anywhere in this project's pipelines.
  The spark data adapter raises on it, and URI snapshots hash `df.inputFiles()` listings, which are checkout-mtime-independent because sources live in the object store; the "one token scheme per pipeline" rule is satisfied with the spark scheme on both the baseline-publish and diff sides.
  The `prod_score` local-adapter target is the exception: score/monitor invocations DO pass `--deep-snapshot`, because the lake sync rewrites mtimes on every run and mtime tokens would fork a fresh `run_key` per run, silently destroying prediction-store idempotency.
- **State branch**: `fetch_state.sh`/`publish_state.sh` port unchanged; Woodpecker's clone has no push credential, so a Gitea token is provisioned as a secret and wired into the push remote.
- **PR comment**: `gitea_pr_comment.py`, a faithful port of `pr_comment.js` to Gitea's `issues/{index}/comments` API, same marker, rendered purely from `run_results.json` + `state_diff.json`; the GitHub runner cost line is replaced with total execution time.
- **promotions.yml lint step**: rejects entries without a pinned `version:` pre-merge (unpinned staging-to-production replays exit 1 by design, because promotion vacates the staging alias).
- **Image bake**: buildx (or kaniko, avoiding the docker-socket mount) to `zot:5000`, digest resolved via the Zot API, digest committed to the deploy repo by the CI bot.

## 7. Observability design

**Mechanism: Pushgateway, implementing docs/tutorial.md step 14 verbatim** rather than inventing a spec.
Every mbt surface is a batch job that exits, so there is no live scrape target; Pushgateway persists last-known gauges per grouping key and stamps `push_time_seconds`, which the staleness alert needs.

- `scripts/push_metrics.py` (stdlib-only, baked into the runner image, zero new mbt source code so zero coverage-gate exposure) parses `target/run_results.json`, NOT stderr events (monitor values deliberately do not travel as typed events), and pushes gauges grouped by `(job=mbt, project, target, command, node)`, every series labeled with the spec's `owner`.
- Metric names: `mbt_node_success`, `mbt_node_duration_seconds`, `mbt_test_metric{metric=}`, `mbt_realized_metric{metric=}`, `mbt_gate_passed`, `mbt_gate_margin{kind=threshold|champion|ground_truth}` (signed headroom, so alert rules never duplicate spec thresholds), `mbt_shift_value{monitor=,subject=,measure=}`, `mbt_shift_threshold`.
- The four canonical alert rules, env-templated so tests shrink windows (staleness 60s in test mode, 8d in demo mode): gate failed, schedule stale, gate near-breach, shift breach.
- Grafana: file-provisioned datasource + one dashboard (Model Health) + unified alerting with owner-label notification policies (one fewer container than Alertmanager, same routing lesson); contact point forwards to webhook-sink.
- Tests assert alert firing via the Prometheus HTTP API (`ALERTS{alertstate="firing"}`), not webhook delivery timing.

## 8. The E2E test tier (the answer to "what kinds of tests")

### Conventions

- Marker `live_showcase` under the existing `live` umbrella; double gate exactly like `live_snowflake`: module-level skipif unless `MBT_LIVE_SHOWCASE=1`, then `pytest.fail` loudly if docker/compose are missing.
- Modules split per concern with per-module compose profiles (core/spark/obs/orch/dev) so each boots only what it asserts; unique basenames; helpers in `showcase_utils.py`.
- Compose project names are per-session (`-p mbt_showcase_<uuid8>`); the `/workspace` volume is the one fixed-name external volume, fixture-created and destroyed; all other state lives under pytest tmp dirs (repo-root session guard stays green); teardown is `down -v --remove-orphans` in a finally block.
- Crash tests inject faults deterministically (env var in `run_mbt.sh` that exits after writing predictions but before `_SUCCESS`), never by racing a kill against a fast run.
- Nightly CI job mirrors `live.yml` (schedule + manual dispatch, never PRs) on ubuntu-latest (macOS runners have no docker); the k3d/argocd module stays local-only initially.

### Test catalog

| ID | Kind | What it proves |
|---|---|---|
| SHOW-01 | infra smoke | All services healthy inside the budget; boto3 path-style round-trip against SeaweedFS; nothing written to repo root |
| SHOW-02 | env sanity | `mbt` console script works in the runner image; **h2o python client version == pysparkling-embedded H2O jar version** (H2O requires exact match; pin `h2o==3.46.0.6` in constraints to match `h2o-pysparkling-3-5`, and fail here before any 45-minute debug session) |
| SHOW-03 | DS loop (closes A2+A3) | `mbt build --target dev` exit 0; version 1 + `staging` alias + `mbt.gates_passed`/`mbt.hooks_hash`/`mbt.baseline_uri` tags visible over HTTP MLflow; MOJO object HEADable in the S3 artifact store |
| SHOW-04 | cluster training | Sparkling train on `spark://spark-master:7077`: every split `row_count > 0` (regression for the driver-local staging failure mode), leaderboard events observed, Spark master API shows a completed app |
| SHOW-05 | slim CI | PR with one gate edit: pipeline green, PR comment shows exactly one modified node with `components: [config]`, no dataset churn across fresh clones (URI snapshot stability) |
| SHOW-06 | state economy | No-change build trains nothing (empty/skipped results, baseline still republished, build step fast); first-ever run honors `fetch_state.sh` exit 3 bootstrap |
| SHOW-07 | negative: gates | Gate failure yields `ci_exit_class == 2` (not 1), PR comment shows `gate_failed`, registered-version count unchanged, webhook-sink got exactly one owner-classified alert |
| SHOW-08 | provenance | Pull the baked image from Zot, `mbt run --manifest` reproduces: xgboost metrics byte-equal, H2O within the documented 0.02 tier; `generated_at == anchor`; oras manifest artifact matches the mbt-state baseline; manifest contains zero secrets |
| SHOW-09 | negative: env drift | Tampered environment: exit 1 with the `env_digest` mismatch message; `--allow-env-mismatch` downgrades to warning (the deployable unit is self-checking) |
| SHOW-10 | GitOps promotion | Pinned-version promote via `promotions.yml` pipeline: production alias moves, replay is idempotent (exit 0, same alias); unpinned entry post-promotion exits 1 (`no version in stage staging`); doctored `gates_passed` tag is refused; unauthorized promotions.yml push rejected by CODEOWNERS/branch protection |
| SHOW-11 | champion resolution | Two scheduled score runs straddling a promotion: served `model_version` flips 1 to 2 while image digest and deploy-repo HEAD stay byte-identical (promotion is a registry event, ADR-20/ADR-5) |
| SHOW-12 | idempotency | Same-anchor double score: exactly one `run_key` dir, fresh `_SUCCESS`, second run's id in the sidecar (overwrite, fresh ledger); new anchor partitions; injected crash leaves no `list_runs`-visible half-write, retry replaces it |
| SHOW-13 | ground truth | Monitor at maturity anchor: evaluated-N message, markers written, realized metrics tracked; same-anchor re-run evaluates zero (exactly-once); degraded labels: exit 2, `monitor_failed`, Airflow task failed with `try_number == 1` (no retry on quality verdicts) |
| SHOW-14 | observability | All step-14 gauges present with owner/command labels; injected drift makes `mbt_shift_value >= mbt_shift_threshold` and the provisioned rule reach `firing` via the Prometheus API; staleness rule fires from `push_time_seconds` alone and resolves after a fresh run |
| SHOW-15 | meta: collection hygiene | The fast suite imports all showcase modules cleanly and self-skips without docker; `MBT_LIVE_SHOWCASE=1` without docker fails loudly (pins the double-gate contract) |
| SHOW-16 | optional: CD fidelity | argocd profile only: digest bump in the deploy repo rolls the k3d CronJobs, insecure-registry pull from Zot works, selfHeal recreates a deleted CronJob |
| SHOW-17 | monthly cadence | `tag:monthly` trains on the prod_score plane (DuckDB over the synced lake, no cluster) and passes its gate; the month-start batch scores with the run-time champion under both shift monitors; its 30-day labels mature at the pinned monitor anchor and evaluate exactly once; the `mbt_score_monthly` DAG runs the same batch from the scheduler in the pinned unit |
| SHOW-18 | runbook fidelity | Extra gate MBT_LIVE_SHOWCASE_MAKE=1: the README golden path driven through `make` on an isolated SHOWCASE_PROJECT - up, demo (every cadence's predictions exist), wide, monthly, score, monitor, inject-drift + recovery, down leaves no containers, clean removes the workspace - so the runbook cannot drift from the tested harness silently |
| SHOW-19 | wide multi-table cadence (ADR-22) | The population-spine dataset (per-table join keys, `safe_id` reachable only through the crosswalk, matured labels inner-joined on the cohort's own `inference_date`) builds via Spark pushdown against the s3a lake; `--vars sample_fraction` panel-samples whole customers with the subset property; sparkling AutoML trains on the selected columns and gates pass; the population-form scoring input scores the newest cohort under both shift monitors; its one-month-later outcomes evaluate exactly once at the monitor anchor |
| SHOW-20 | batch monthly hardening | The ds-helper funnel over the probe's full-width materialization reproduces the committed include list byte-for-byte and documents every stage in `target/feature_selection_report.json`; `contract_code` (numeric-coded categorical, cast by the shared `wide_hooks.py` before selection and training) survives selection; the Evidently train-phase gate passes on the stable panel, exports the persisted serving baseline, and a breach exits 2 before `mbt promote`; the serving-phase gate passes on the clean monthly batch and exits 2 on a poisoned one, naming the drifted features; the `mbt_score_wide` DAG runs sync -> score -> gate from the scheduler with quality-exit routing; the committed DS notebook executes end to end via nbconvert and leaves the committed contract untouched |

## 9. Known constraints this design respects (do not "fix" silently)

- `mbt score`/`mbt monitor` cannot use the spark data adapter (no contract-1.1 `build_scoring_input`/`open_predictions`); the scoring plane is local-adapter over a synced lake copy. A future `mbt-spark` contract-1.1 implementation would remove the sync hop (candidate follow-up, outside this showcase).
- Remote cluster is train-time only: evaluate/score load MOJOs in a local JVM by design; never promise cluster-side scoring.
- `--manifest` reads local files only; the manifest travels baked inside the image (which is the design), while `--state` accepts s3:// URIs.
- `mbt clean` refuses s3:// stores and nothing protects champion objects server-side: **SeaweedFS buckets are created with retention/TTL disabled by the seed path**, not just documented.
- Switching an existing target's artifact-store scheme strands registered champions (fetch rejects cross-scheme refs); the showcase never flips schemes mid-life.
- MLflow UI shows pointers/tags, not model binaries (mbt only `log_artifact`s file:// stores); the README explains where the bytes live.
- The dev uv lock resolves pyspark 4.x; the runner image builds the sparkling fork (`mbt-h2o[sparkling]` pins pyspark 3.5.x) in its own resolution with `uv export --frozen` constraints, and the cluster runs the matching Spark 3.5.8 binaries.
- Woodpecker trusted-repo volumes, the docker socket on the agent, and insecure-registry config are demo-tier security postures; the README says so.

## 10. Repo layout

Everything lives in this repo under `examples/showcase/` (yamllint and renovate already cover `examples/`; the coverage gate scopes only `packages/`; tests must live in repo-root `tests/` per testpaths).

```
examples/showcase/
  README.md                    # runbook: make up, golden path, URLs, RAM knobs, teardown,
                               # the two documented deviations (snapshot scheme, local scoring plane)
  DESIGN.md                    # this file
  Makefile                     # up, down, image, seed, demo, score, monitor, monthly, wide, inject-drift, clean
  .env.example
  images/runner/{Dockerfile,constraints.txt,entrypoint.sh}
  compose/docker-compose.yml   # profiles: core, spark, orch, obs, dev, ci (+ argocd)
  compose/{gitea,seaweedfs,prometheus,grafana,airflow}/...
  bootstrap/{seed_lake.py,sync_lake.py,inject_drift.py,webhook_sink.py}
  project/                     # the churn mbt project (source of truth; pushed into Gitea)
    .woodpecker/{pr-check.yml,prod-build.yml,promote.yml}
    models/wide_hooks.py       # shared wide-model hooks: DS-declared categorical codes
    notebooks/ds_inner_loop.ipynb  # the DS workbench narrative, executed by the live tier
    scripts/{run_mbt.sh,push_metrics.py,gitea_pr_comment.py,select_features.py,evidently_gate.py}
    ...
  deploy/                      # the deploy repo source (images.env, DAGs; + k8s/ for the argocd profile)
tests/
  test_showcase_infra.py       # SHOW-01/02/15
  test_showcase_ci.py          # SHOW-05/06/07/10
  test_showcase_lifecycle.py   # SHOW-03/04 + 10/11/12/13 (CLI-driven)
  test_showcase_provenance.py  # SHOW-08/09
  test_showcase_scheduling.py  # SHOW-11/13 + SHOW-17/SHOW-20 scheduled paths
  test_showcase_obs.py         # SHOW-14
  test_showcase_k3d.py         # SHOW-16 (optional argocd/k3d)
  test_showcase_monthly.py     # SHOW-17
  test_showcase_make.py        # SHOW-18
  test_showcase_wide.py        # SHOW-19/SHOW-20 (ADR-22)
  showcase_utils.py
```

## 11. Phasing (each phase independently valuable and tested)

1. **P1 Runner image + data/ML core**: seaweedfs, mlflow, spark, jupyterlab, seeded lake, dev/prod targets. Tests SHOW-01..04. Closes integration items A2 + A3.
2. **P2 Git + CI loop**: gitea, woodpecker, bootstrap seeding, pipelines, exit-code wrapper, Gitea PR comments, state branch. Tests SHOW-05..07.
3. **P3 Deployable unit + provenance**: zot, bake + push, commit-time anchors, oras artifacts, the tamper provenance test. Tests SHOW-08..09.
4. **P4 Scheduling + CD + promotion**: airflow + postgres + git-sync, deploy repo, DAGs with exit-code routing, prod_score plane, promotion flow. Tests SHOW-10..13.
5. **P5 Observability + docs**: prometheus/pushgateway/grafana, dashboards, rules, drift injection; docs page in mkdocs nav; one exactly-true sentence in v0.1-status; nightly workflow. Tests SHOW-14..15.
6. **P6 (optional) ArgoCD fidelity profile**: k3d + ArgoCD over the same deploy repo. Test SHOW-16, local-only.
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

   - ALL 12 tables carry an identifier, not just the 6 the wide cadence reads. Compile pins a snapshot for every source referenced by any dataset or scoring node regardless of `--select`, so a missing identifier fails the compile before selection narrows anything.
     The seeder therefore CREATES all 12 but loads rows into only the 6 the wide cadence reads (`WIDE_TABLES`, held against the specs by a test); pinning is a metadata call, so the other six stay empty rather than putting ~26k rows of daily/monthly demo data in the operator's sandbox. `--all-cadences` loads everything, and is the fallback if an account will not pin a never-written table.
   - Registered names are namespaced per plane via the `plane_suffix` var (`""` everywhere, `"_snowflake"` on the new target). Both planes train the same spec, and without this their versions would interleave in the shared registry and quietly corrupt champion resolution. It reaches the spec through `var()`, so it enters the config hash by design - the two planes are genuinely different nodes.
   - Host-run, still. `externalbrowser` SSO needs a real browser and a localhost callback, and the runner image does not ship `mbt-snowflake` (its sparkling extra pins pyspark 3.5.x, which does not resolve cleanly against the connector's `cryptography>=46.0.5` floor). The target reaches the stack's MLflow and S3 over published ports, so `make snowflake` runs mbt on the host rather than through `$(EXEC)`.
   - Adding `identifier:` is state-neutral for the lake planes: source config lives on `ManifestSource`, and node `config_hash` covers only node config. Verified by diffing every resolved node config before and after the change.
   - It is NOT read-neutral for Spark, which is the one adapter that reads both object-store paths and catalog tables.
     The local and Snowflake adapters each read one field and ignore the other, so for them a both-addresses table is unambiguous; Spark refuses to guess, so all three spark targets set `source_address: path`.
     This was learned the hard way: the first cut of P7 shipped without it, Spark's `_read` silently preferred `identifier` (while `snapshot_id` kept hashing the `path`, so the pin described data the run never read), and the whole lake plane went looking for `MBT_SHOWCASE_*` catalog tables that do not exist.
     Push CI stayed green - every mbt-spark test was JVM-gated behind the `e2e` marker - and only the nightly live tier caught it, which is why the precedence rule now has a fast-tier test of its own (`packages/mbt-spark/tests/test_spark_source_address.py`).

   Testing is two-tier, and the hermetic half is the one that runs by default.
   `packages/mbt-snowflake/tests/test_showcase_snowflake_plane.py` builds the committed wide spec through the real Snowflake adapter with its SQL executed in DuckDB (no account), and holds the both-addresses invariant plus seeder/sources agreement.
   `tests/test_showcase_snowflake.py` is triple-gated (MBT_LIVE_SHOWCASE=1 + MBT_LIVE_SNOWFLAKE=1 + complete SNOWFLAKE_*) and proves the loop on a real account, including a cross-plane assertion that both planes materialize the same panel.
   The triple gate keeps the hermetic grand-suite guarantee intact: the showcase tier still needs docker and nothing else.

## 12. Open questions

1. Zot addressing on Docker Desktop for macOS (in-network vs published localhost): smoke-test before P3 commits to a scheme.
2. Whether the nightly CI subset can afford the spark/sparkling module on ubuntu-latest runners (RAM/time), or whether SHOW-04 stays local-only alongside the k3d tier.
3. Whether to eventually ship `mbt init --forge gitea` scaffolding (Woodpecker pipelines + Gitea PR-comment script) upstream once the showcase proves the port; the showcase keeps them project-local until then.
