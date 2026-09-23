# mbt showcase: full-stack docker reference environment + E2E test tier (design)

Status: IMPLEMENTED.
P1 (runner image + data/ML core), P2 (Gitea + Woodpecker CI loop incl. branch protection + CODEOWNERS on promotions.yml), P3 (Zot deployable unit + oras provenance), P4 (Airflow + git-sync CD + the scoring/promotion/monitoring plane), P5 (observability) and P6 (k3d + ArgoCD, local-only behind its own `MBT_LIVE_SHOWCASE_K3D` gate) are implemented and covered by the `live_showcase` test tier; see README.md for what runs today.
On 2026-09-23 the data model was collapsed to ONE lake table (section 4); section 11 records what that retired.

This document started as the design and is kept as the design of record: where the implementation deliberately scoped something differently, the sections below say what was built, not what was first proposed.

The test catalog (section 8) maps onto these gated modules, all in repo-root `tests/`:

| Module | Covers | Gate beyond `MBT_LIVE_SHOWCASE=1` |
|---|---|---|
| `test_showcase_infra.py` | SHOW-01/02, including the registry-outage exit-1 contract | - |
| `test_showcase_ci.py` | SHOW-05/06/07/10 (CI side) | - |
| `test_showcase_lifecycle.py` | SHOW-03/04/10/11/12/13 (CLI-driven) | - |
| `test_showcase_obs.py` | SHOW-14 | - |
| `test_showcase_provenance.py` | SHOW-08/09 | - |
| `test_showcase_scheduling.py` | SHOW-11 strong form, SHOW-13 routing | - |
| `test_showcase_k3d.py` | SHOW-16 | `MBT_LIVE_SHOWCASE_K3D=1` (local-only) |
| `test_showcase_make.py` | SHOW-18 | `MBT_LIVE_SHOWCASE_MAKE=1` (own pytest invocation) |

Hermetic modules guard the showcase from the ordinary fast suite, because `-m e2e` excludes the live tier and `-m "not e2e"` deselects it: `tests/test_showcase_gates.py` (SHOW-15: every gated module keeps its gate), `tests/test_showcase_image_pins.py` (the runner image's hand pins, the extras closure, the S3 credentials, SeaweedFS capacity) and `tests/test_showcase_panel.py` (the lake table's generator and the project's use of it).

Modules share one session stack and run in collection (alphabetical) order; the only load-bearing constraint is that `test_showcase_ci` is the first forge consumer (virgin-bootstrap assertion).
Everything else provisions or promotes what it needs, and a module that changes the lake table (`land-outcomes`, `inject-drift`) puts it back with `reset` before it finishes.
Implementation notes (deliberate scoping vs the sections below):
pr-check builds on the `ci` target and the merge-time prod-build on `dev` (spark local[2] + the shared registry); the scheduled retrain DAG is the cluster-from-CI path (prod target from a pinned unit), tested with the deterministic xgboost workhorse - sparkling stays confined to SHOW-04's module per the flake-isolation rule.
Baking is gated on "this merge retrained something": docker layer digests embed mtimes, so an unconditional bake would mint a new digest per merge and break the ADR-20 "promotion deploys nothing" claim.
SHOW-05's "exactly one modified node" reads as "exactly the edited model (config) plus its downstream scoring node (upstream)" - scoring depends_on its model, so the lineage flag is correct behavior.
Zot is addressed two ways for one digest: the docker daemon pushes/pulls via the published localhost port (insecure-by-default, verified on Docker Desktop for macOS, OrbStack and the Linux CI runner), while in-network consumers (oras, k3d) use zot:5000.
Alert routing stops at Prometheus: the four rules evaluate there and the tests read `ALERTS` through its API, and no Alertmanager or Grafana contact point is provisioned (section 7).
The only webhook traffic is the CI exit-code classifier's.

## 1. Purpose and personas

Two personas walk one story end to end:

- **Data scientist**: iterates in JupyterLab against one wide table in the lake (SeaweedFS S3), builds datasets with Spark pushdown, trains H2O AutoML (pysparkling on the cluster), registers to MLflow, opens a PR.
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

### The load-bearing shared mount

The Spark adapters have a hard driver-local-filesystem assumption (ADR-17): split staging writes `coalesce(1)` output to a driver-local temp dir that executors must also see, and `_materialize_for_path_adapter` dirs are read by `h2o.import_file` on executors.
So `/workspace` - a host directory (`SHOWCASE_WORKSPACE`, default `~/.cache/mbt-showcase/workspace`; a pytest tmp dir under the test tier) - is bind-mounted at the identical absolute path into JupyterLab, the Spark worker, webhook-sink, and every Airflow-launched unit container, with `TMPDIR=/workspace/tmp` wherever mbt can drive the cluster.
Woodpecker steps do not mount it: they build on `dev`/`ci`, whose `local[2]` master keeps the driver and executors in one JVM.
The `batch` target also stages its prediction runs there (`/workspace/predictions`), so they outlive the container that scored them.

The runner entrypoint preflights `TMPDIR` in every container that sets it and hard-fails if it is unwritable, converting the misleading distance-failure `split ... materialized 0 rows` into an immediate diagnosable error.
The host creates `tmp/` and `monitoring/` itself before booting, because every container runs as root and on native Linux whichever one reached a directory first would own it.

### Spark driver reachability

Spark standalone does not support cluster deploy-mode for Python apps, so every driver runs client-mode inside a dynamically named container and executors must connect back.
The runner entrypoint exports `SPARK_DRIVER_HOST=$(hostname -i)`; the `prod` target sets `spark.driver.host: "{{ env('SPARK_DRIVER_HOST', 'jupyterlab') }}"` and fixed driver/block-manager ports, one pair for the data adapter's session (40400/40401) and one for the sparkling training session (40402/40403), since both coexist in one prod build.

## 3. Service inventory

Every image is pinned to an exact version; `compose/docker-compose.yml` is the source of truth for the tags, and the table names them so a reader can see what was verified together.

| Service | Image | Role |
|---|---|---|
| runner (built, not run) | `mbt-showcase-runner:dev`, built on the host by `scripts/build_image.sh` from a digest-pinned `python:3.11-slim-bookworm` + OpenJDK 17 JRE + the workspace wheels (`mbt-core[s3]`, `mbt-h2o[sparkling]`, `mbt-spark`, `mbt-xgboost`, `mbt-mlflow`) with third-party pins from `uv.lock` + pyspark 3.5.8 (its bundled jars are `SPARK_HOME`) + hadoop-aws 3.3.4 and the matching AWS SDK bundle + jupyterlab from the committed `image-extras.txt` closure | Universal environment; env_digest identity by construction. A content-hash label rebuilds it whenever package sources, `uv.lock`, or the image inputs move |
| gitea | `gitea/gitea:1.27.3-rootless` | Hosts `churn` project repo + `deploy` repo; branch protection + CODEOWNERS gate on `promotions.yml`; `mbt-state` branch storage |
| woodpecker server + agent | `woodpeckerci/woodpecker-*:v3.18.1` | CI: pr-check, prod-build, promote pipelines; agent mounts the docker socket; repo marked trusted so the bake step may mount that socket; split-horizon URLs (public WOODPECKER_HOST + forge OAuth host, in-network webhook host) so the Gitea OAuth login works from a host browser |
| seaweedfs | `chrislusf/seaweedfs:4.47` (`weed server -s3`, 64MB volumes under a fixed 100-volume ceiling, section 9) | S3-compatible object store: `mbt-lake` bucket (the one table, read via s3a://) and `mbt-artifacts` bucket (the artifact store under `churn_lake`); the filer UI is published for humans to browse the lake |
| spark-master, spark-worker | runner image running `spark-class ...Master` / `...Worker` (1 worker, 4 cores, 4g) | Standalone cluster: pushdown reads/sampling/windows + sparkling H2O training |
| mlflow | runner image (`mlflow server`, sqlite on a volume) | Tracking + model registry (alias mode); champion source of truth |
| jupyterlab | runner image + `jupyter lab` | DS workbench and the container every `make` recipe runs mbt in |
| airflow | `apache/airflow:3.3.1` (init + api-server + scheduler + dag-processor), **LocalExecutor + `postgres:18.6-alpine`** (sqlite forces SequentialExecutor; invalid with LocalExecutor) + `git-sync:v4.7.1` sidecar | Runs the retrain/score/monitor DAGs on demand; tasks drive the docker SDK to run the pinned digest from the deploy repo's `images.env` |
| zot | `ghcr.io/project-zot/zot:v2.1.16` (v2.1.17+ breaks multi-GB pushes on slow disks, zot#4140) | OCI registry: baked deployable units (`mbt/churn`) and oras-pushed `manifest.json`/`run_results.json` provenance artifacts (`mbt/churn/provenance`) |
| prometheus + pushgateway | `prom/prometheus:v3.13.3`, `prom/pushgateway:v1.11.3` | Metrics per docs/tutorial.md step 14 (the documented spec, implemented verbatim) and the four alert rules |
| grafana | `grafana/grafana:13.1.5` | File-provisioned Prometheus datasource + the "mbt Model Health" dashboard (gate margins, realized metrics, shift vs threshold, node durations) |
| webhook-sink | ~60-line python recorder with `GET /requests` | Records the alerts `scripts/run_mbt.sh` posts to `MBT_ALERT_WEBHOOK`; human-readable in demos, assertable in tests |
| bootstrap (host + one-shot scripts) | `scripts/ci_bootstrap.py` on the host; `bootstrap/churn_panel.py` in the runner | `make ci`: Gitea users, org, repos, token, OAuth app, deploy repo, Woodpecker login + repo activation + secrets. `make seed`: buckets (no TTL/retention, section 9) and the generated lake table. The image is built by `make image` |

## 4. The demo project and its one table

An `mbt init`-derived `churn_lake` project, source-of-truth at `examples/showcase/project/`, staged into the workspace by `make up` and pushed into Gitea as the `mbt-showcase/churn` repo by `make ci` (the showcase never depends on a long-lived external repo).

### 4.1 The lake table

The whole data model is one table that is assumed to already exist in the lake, the way a gold-layer customer snapshot usually does: `s3://mbt-lake/churn_panel/*.parquet`.

- One row per customer per month-start `inference_date`, 2025-07-01 .. 2026-06-01 (12 cohorts).
- `customer_id` + `inference_date` are the natural key (the dataset's `unique` check asserts it).
- `is_active` is the population: a churned customer keeps its rows, marked inactive, so the dataset and the scoring input both select the population with a filter.
- 16 named features (demographic, login, transaction), among them the numeric-coded `contract_code` (int8, 0 = month-to-month ... 3 = two-year) whose churn effect is deliberately non-monotone, and four string categoricals.
- `f0000 .. fNNNN`: pure-noise columns, the "huge" knob. A real lake table carries hundreds of columns no model uses; the models name the 16 they read.
- `is_churn`: churned during the month after `inference_date`. NULL where no outcome is known - on inactive rows, and on the newest cohort while its outcome window is still open.

Nothing about it is committed.
`bootstrap/churn_panel.py` synthesizes it deterministically (fixed seed) straight into the lake at `make seed`, at whatever scale is asked for (`SCALE=huge` is the realistic lake shape through the same code, generated and uploaded one bounded chunk at a time).
One parquet file per cohort (chunked by `--rows-per-file`), named `<inference_date>-<state>-<chunk>.parquet`.

The newest cohort is the only part that changes after seeding, and each change is a partition rewrite under NEW file names followed by deleting the old ones:

- `land-outcomes` (`make outcomes`): a month passed, the cohort's `is_churn` is filled.
- `inject-drift` (`make inject-drift`): numeric features x3, which breaches the scoring node's PSI monitors.
- `reset` (`make reset`): the cohort as seeded - byte-identical files under the seeded names.

Never in place, and that is load-bearing: for an object-store source the spark adapter pins a snapshot by hashing the table's file LISTING, so an in-place rewrite would change the data under a pinned manifest without changing its snapshot.
Because `reset` restores the exact seeded names and bytes, a snapshot pinned before a change is valid again afterwards, which is what lets the test modules share one stack.

### 4.2 Models

- `churn_automl` (h2o_automl, `max_models: 3`, `include_algos: [GLM, GBM]`, `nfolds: 0`, fixed seed, no `max_runtime_secs`): the star of the DS story and the served champion, threshold + champion gates, `registration: {name: churn_automl, stage_on_pass: staging}`.
- `churn_baseline_xgb` (xgboost): deterministic workhorse for the CI-loop differentiator tests (bit-exact `--manifest` reproduction; H2O's documented determinism tier is 0.02 tolerance, so H2O reproduction asserts within tolerance, never byte-equality).
  Sparkling + remote master stays confined to its own test module so a flake there never poisons the slim-CI/promotion/idempotency assertions.

Both read `ref('churn_training')` and name the same 16 features in `features.include`, with `features.categorical` declaring every string feature plus `contract_code` (ADR-27).
Naming them is the reviewed decision: a column arriving upstream can never start training silently, and editing the list marks the model `state:modified`.

### 4.3 The dataset

`datasets/churn_training.yml` is a slice of the one table: `filters: ["is_active = true"]`, `label: {column: is_churn, horizon: 1mo}`, `sample_key: customer_id`, and a temporal split with explicit cohort boundaries (train `2025-07-01:2026-04-01`, test `2026-04-01:2026-06-01`, `embargo: 1mo`).
Both windows end before the newest cohort, so no row with an unknown outcome is ever a training example.
It declares no `columns:` panel contract on purpose: the table's width is a scale knob, so an enumerated list would go stale with every reseed, and the models' include lists carry the review burden instead.

### 4.4 Scoring resource

`scoring/retention_scoring.yml`: `model: churn_automl`, `stage: production`, and every row source is the same table.

- The input reads `churn_panel` with the same population filter and `window: "-31d:-28d"`, which at the pinned anchor is exactly the newest cohort.
  The table's `is_churn` column is harmless there: a model's target is never one of its features, and the prediction store keeps only `output.columns`, the join keys, and the prediction.
- `ground_truth.label.source` is the same table, joined on `[customer_id, inference_date]`.
  The time column is required: `customer_id` alone matches every cohort a customer was ever in, grading this month's prediction against old outcomes.
  A joined row whose label is still NULL counts as an outcome not yet landed, so before `land-outcomes` the run waits instead of being graded.
- PSI/KS shift monitors, a 14-day maturity, and realized-metric gates.

ADR-29 recommends a label-free serving twin of the training relation; the showcase deliberately reads the labelled table for all three, because the three reasons above make it safe and one table is the point.
Scheduling lives entirely outside the YAML (there is no schedule field).

### 4.5 Targets (profiles.yml, committed and secret-free)

`profiles.yml` is committed with pure `{{ env_var(...) }}` / `{{ env(...) }}` values (the scaffold gitignores profiles, so CI checkouts have none otherwise); container env supplies `MLFLOW_TRACKING_URI=http://mlflow:5000`, `AWS_ENDPOINT_URL_S3=http://seaweedfs:8333`, keys, region.
Every target reads the table through Spark over s3a: `root: s3://mbt-lake` (the `s3://` prefix is exempt from project-dir path resolution), with `fs.s3.impl` mapped to `S3AFileSystem`.
Note: boto3's env chain is the only S3 endpoint mechanism (nothing in mbt parses endpoints), so one process talks to exactly one S3 endpoint; fine here since SeaweedFS is the only object store.

| Target | Data | Compute/training | Registry/artifacts | Use |
|---|---|---|---|---|
| `dev` | spark `master: local[2]` | h2o local backend (`h2o_max_mem: 1G`), `sample_fraction: 1.0` (narrow it per run with `--vars`) | shared MLflow, `s3://mbt-artifacts/churn_lake` | DS fast inner loop; also the merge-time prod-build target |
| `ci` | same as dev | same as dev | **per-run sqlite MLflow + local artifact store** | PR checks: green PRs must never register versions or re-point the shared `staging` alias; tradeoff: champion gates render "none (bootstrap)" in PR comments, documented |
| `prod` | spark `master: spark://spark-master:7077` | `h2o_backend: sparkling`, driver-host conf per section 2 | shared MLflow, `s3://mbt-artifacts/churn_lake` | Prod builds, the retrain DAG |
| `batch` | spark `master: local[2]`, `predictions_root: /workspace/predictions` | h2o local (MOJO scoring is local-JVM by design; the remote cluster is train-time only) | shared MLflow, `s3://mbt-artifacts/churn_lake` | `mbt score` / `mbt monitor`, from `make` and the DAGs: no cluster, straight off the lake |

### 4.6 Anchors (the determinism spine)

Every pipeline, DAG, and test pins anchors to constants: `ANCHOR=2026-06-30T00:00:00Z` for build and score, `MONITOR_ANCHOR=2026-07-20T00:00:00Z` (past maturity) for monitor.
Wall-clock anchors over fixed-date data are a time bomb: relative windows resolve empty within weeks and every unpinned pipeline rots into `split ... materialized 0 rows`.
Airflow DAGs therefore take `--anchor` from the deploy repo (`showcase_dag_utils.py`, overridable per run through the DAG's `anchor` param), never from `{{ ts }}`.
Anchor time travel is also what makes monitoring demoable today: `mbt monitor --anchor <maturity+>` evaluates immediately once the outcomes have landed; re-running with the same anchor evaluates nothing (exactly-once proof).
The `.woodpecker/` pipelines pin the same `ANCHOR`, which also makes same-source rebuilds byte-identical manifests (`generated_at == anchor`, ADR-19).

## 5. Golden path (the demo narrative)

1. `make up`: build (or reuse) the runner image, stage the workspace, `docker compose up -d --wait`, generate the lake table; the terminal prints every UI URL.
2. **DS inner loop**: open `project/notebooks/ds_inner_loop.ipynb` in JupyterLab - look at the lake table (reading only the columns asked about), review the YAML, `mbt build --target dev`, analyze the run artifacts, try a hash-sampled what-if with `--vars`. (`make demo` step 1 is the same `mbt build --target dev` over both models.)
3. **DS scales out**: `mbt build --target prod`; pushdown sampling runs on the cluster and sparkling H2O trains inside the executors (Spark UI shows the apps); models register and the `staging` alias moves.
4. **PR** (after `make ci`): push a branch, open a Gitea PR; Woodpecker pr-check lints `promotions.yml`, runs parse, compile, `fetch_state.sh` (exit 3 = bootstrap), `state diff --output json`, slim build `--select state:modified+ --state ...` under the `ci` target, then posts the update-in-place `<!-- mbt-pr-comment -->` comment via Gitea's API showing exactly the modified nodes and their gates.
5. **Merge**: prod-build runs the economy build on `dev`, publishes the manifest to `refs/heads/mbt-state` (`publish_state.sh` is pure git plumbing and works against Gitea unchanged), and - when the merge retrained something or no unit exists yet - bakes the deployable unit (`FROM` the exact runner tag + project + `target/manifest.json` compiled inside that same image), pushes it to Zot, oras-pushes `manifest.json`+`run_results.json` as provenance artifacts (manifests are secret-free by construction), and commits the new digest to the deploy repo.
6. **CD**: git-sync reconciles the deploy repo into Airflow; rollback is `git revert` (the optional k3d tier has ArgoCD sync the same repo's `k8s/` CronJob).
7. **Schedules**: the DAGs run the pinned unit - `mbt_retrain` (`mbt build --target prod`), `mbt_score` (`mbt score --target batch`) and `mbt_monitor` (`mbt monitor --target batch`). They are manual-trigger (`schedule=None`) so demos and tests stay deterministic. Exit 1 retries then pages on-call, exit 2 never retries and notifies the model owner (per the tutorial's routing rule); the task pushes metrics regardless of outcome.
8. **Promotion**: a PR edits `promotions.yml` (version always pinned); CODEOWNERS + branch protection gate the merge; the promote pipeline runs `mbt promote --from-file`; the next score run serves the new champion with zero redeploy, image digest and deploy repo byte-identical before and after (the ADR-20 inversion, asserted).
9. **Outcomes land**: `mbt monitor` first finds the scored cohort's labels still NULL and waits; `make outcomes` rewrites the cohort with its labels, and the next monitor run evaluates it exactly once.
10. **Monitoring pays off**: `make inject-drift` poisons the newest cohort, `mbt score` exits 2 (`monitor_failed`), and the pushed `mbt_shift_value >= mbt_shift_threshold` puts the `MbtShiftBreach` rule into pending and then firing in Prometheus; `make reset score` recovers. The `MbtScheduleStale` rule watches `push_time_seconds` for the one failure no in-band mechanism can catch, a schedule that silently stopped.

## 6. CI design (Woodpecker)

Pipelines live in `.woodpecker/` of the project repo (authored fresh; the GitHub scaffold under `_scaffold/.github/` stays untouched, so `tests/test_cli_basics.py` is unaffected).

- **Exit-code fidelity**: Woodpecker collapses any nonzero exit to "failed", erasing mbt's 1-vs-2 contract.
  Every `mbt build` and `mbt promote` invocation in the pipelines runs through `scripts/run_mbt.sh`: capture code, write `target/ci_exit_class` (`0 ok` / `1 hard-error` / `2 quality-failure`), on failure POST a classified alert to `MBT_ALERT_WEBHOOK` (2-second curl timeout; 1 pages on-call, 2 notifies the failing spec's owner), push metrics best-effort (`push_metrics.py` uses a 5-second timeout and warns and exits 0 when the Pushgateway is away, so observability can never fail a pipeline), re-exit with the original code.
  `target/ci_exit_class` is the verdict left for a human or a later step to read; the test tier asserts the same classification on the payload webhook-sink recorded.
- **Snapshot scheme deviation, documented**: no `--deep-snapshot` anywhere in this project.
  It would be a no-op, not an error: for a URI source the spark data adapter hashes the `df.inputFiles()` listing and ignores `deep`, and that listing is checkout-mtime-independent because the source lives in the object store. The "one token scheme per pipeline" rule is therefore satisfied with the spark scheme on both the baseline-publish and diff sides.
  The flip side is section 4.1's rule: the table's files are immutable, and a change is a new file name.
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

- Marker `live_showcase` under the existing `live` umbrella; double gate: module-level skipif unless `MBT_LIVE_SHOWCASE=1`, then `pytest.fail` loudly if docker is missing or its daemon unreachable.
- Modules split per concern and share ONE session-scoped stack (all six profiles; the `showcase_stack` fixture in `tests/conftest.py`), because booting the whole platform once per module would multiply its startup and seeding cost; unique basenames; helpers in `showcase_utils.py`. The k3d and make modules add their own gates, and the make module boots a second, isolated stack through the Makefile, so it runs in its own pytest invocation.
- Compose project names are per-session (`mbt-show-<uuid8>`, and `mbt-make-<uuid8>` for the runbook tier) on free ephemeral ports; the `/workspace` bind mount is a pytest tmp dir; all other state lives in compose-project-scoped volumes (the repo-root session guard stays green); teardown is `down -v --remove-orphans` in a finally block, skipped only when `MBT_SHOWCASE_KEEP=1` asks to keep the stack for a post-mortem.
- Failure evidence is dumped before teardown, because nothing survives it: `wait_pipeline` prints the logs of every non-successful Woodpecker step, `wait_dag_run` prints every failed Airflow task attempt, and a failed `compose up` or make target prints per-service log tails.
- The nightly CI job lives in `live.yml` (schedule + manual dispatch, never PRs) on ubuntu-latest (macOS runners have no docker): the main tier, then the make runbook tier after the session stack is gone. The k3d module stays local-only.

### Test catalog

Each row states what the tier ASSERTS today; nothing below is aspirational.

| ID | Kind | What it proves |
|---|---|---|
| SHOW-01 | infra smoke | `compose up --wait` brings every healthchecked service up; MLflow answers `/health` over HTTP; the Spark master reports an ALIVE worker; boto3 round-trips an object against SeaweedFS from inside the runner; the lake holds exactly one table, `churn_panel`; the filer UI lists both buckets and that table without credentials while the raw S3 port refuses an unsigned request with 403; and with the registry stopped, `mbt build` exits 1 - the infra half of the exit-code contract, never 2 |
| SHOW-02 | env sanity | `mbt --version` and `git` work in the runner image; **h2o python client version == pysparkling-embedded H2O version** (H2O requires an exact match; `build_image.sh` pins `h2o==3.46.0.6` to match `h2o-pysparkling-3-5`, and `tests/test_showcase_image_pins.py` holds that pin against mbt-h2o's declared range in the fast suite) |
| SHOW-03 | DS loop | `mbt parse` and `mbt build --target dev` exit 0; `churn_automl` registers to `staging` and the alias, read back over HTTP MLflow, carries `mbt.gates_passed=true`; `churn_baseline_xgb` succeeds; model artifacts exist under `s3://mbt-artifacts/churn_lake/` |
| SHOW-04 | cluster training | `mbt build --target prod --select +churn_automl` (pushdown on `spark://spark-master:7077`, sparkling H2O in the executors) succeeds, registers a newer version that takes the `staging` alias, and the Spark master API shows the application |
| SHOW-05 | slim CI | A one-gate-edit PR: pipeline green; the PR comment's modified lines are exactly `churn_automl` (config) plus its scoring node (upstream), with no dataset and nothing added across fresh clones (URI snapshot stability); the shared registry is untouched. Merging it retrains only `churn_automl` (v2 takes `staging`; the xgboost baseline keeps one version) and pins a freshly baked unit |
| SHOW-06 | state economy | The first-ever push honors `fetch_state.sh` exit 3 and full-builds (both models v1 in `staging` with the gate stamp, the baseline on `mbt-state`, the first unit baked and pinned); a no-change PR comments "Nothing modified" and its merge republishes an identical baseline with zero registry churn and the deploy-repo HEAD untouched |
| SHOW-07 | negative: gates | An impossible-gate PR fails the pipeline; the comment shows `gate_failed` and "registration blocked"; webhook-sink holds exactly one alert, classified `quality-failure`, notifying the owner (`growth-ds@example.com`) and naming the node; the shared registry is unchanged |
| SHOW-08 | provenance | For the first bake: the oras artifact holds exactly `manifest.json` + `run_results.json`, its manifest is byte-identical to the `mbt-state` baseline of the same source sha, `generated_at == anchor`, the recorded git commit is that sha, and neither the S3 secret nor the CI token appears; the unit ships the same manifest bytes, and `mbt run --manifest` inside the pulled unit reproduces xgboost metrics exactly and H2O within the documented 0.02 tier |
| SHOW-09 | negative: env drift | A forged `mbt-*` distribution in the unit makes `mbt run --manifest` exit 1 with the `env_digest` mismatch message; `--allow-env-mismatch` runs it with the same message as a warning |
| SHOW-10 | GitOps promotion | CLI side: a pinned `promotions.yml` moves the `production` alias, its replay is idempotent, and an unpinned entry afterwards exits 1 naming `staging`. CI side: with CODEOWNERS + branch protection on `main`, the DS persona's direct push is rejected, the unapproved merge is refused, and the code owner's approval lets the merge run the promote pipeline, which moves the alias while the deploy repo HEAD and the pinned digest stay byte-identical |
| SHOW-11 | champion resolution | Two `mbt_score` DAG runs straddling a promotion serve different champions (the prediction sidecar's `model_version`) while the image digest and the deploy-repo HEAD stay byte-identical (promotion is a registry event, ADR-20/ADR-5) |
| SHOW-12 | idempotency | A score at the anchor writes one `run_key` directory whose sidecar names the promoted version and a `scored_at` on the anchor date; the same anchor again leaves the same single directory (overwrite); a new anchor adds a second; every run directory carries `_SUCCESS` |
| SHOW-13 | ground truth | From the same table: while the scored cohort's labels are NULL, `mbt monitor` at the maturity anchor evaluates nothing and says the labels have not joined; after `land-outcomes` it evaluates with `pr_auc > 0.2` and `roc_auc > 0.5`; the same anchor again evaluates "0 matured prediction runs" (exactly-once); a fresh prediction run monitored with an impossible realized floor (`--vars pr_auc_floor: 0.99`) exits 2 with `monitor_failed`. At the scheduler: that breach fails the `mbt_monitor` task on try 1 with no retry, a clean re-run then succeeds, and a hard error (nonexistent target) consumes the retry (`try_number == 2`) |
| SHOW-14 | observability | After a scored run is pushed, Prometheus holds `mbt_node_success{command="score"}` for the scoring node and `mbt_shift_value{monitor="feature_shift"}`; all four rules are loaded; Grafana's health check passes; an injected shift makes `mbt score` exit 2 (`monitor_failed`), `mbt_shift_value >= mbt_shift_threshold` returns series, and `ALERTS{alertname="MbtShiftBreach"}` appears |
| SHOW-15 | meta: collection hygiene | Every gated showcase module, discovered by filename rather than listed, keeps the opt-in skipif and the `live`/`live_showcase` marks; once opted in, a missing docker binary FAILS rather than skips (the double-gate contract). Runs in the fast suite, where a lost gate would otherwise boot a stack |
| SHOW-16 | optional: CD fidelity | k3d tier only: ArgoCD syncs the deploy repo's `k8s/` CronJob pinned to the same digest CI wrote for Airflow; a pod pulls that unit from zot over insecure HTTP and runs `mbt --version`; a digest change pushed to the deploy repo rolls the CronJob; selfHeal recreates a deleted CronJob |
| SHOW-18 | runbook fidelity | Extra gate `MBT_LIVE_SHOWCASE_MAKE=1`: the README golden path through `make` on an isolated `SHOWCASE_PROJECT` - up; demo (prediction runs exist and the monitor evaluated one after the outcomes landed); ci plus the repo page and the OAuth login its output instructs; reset, score, outcomes, monitor; inject-drift and recovery; down leaves no containers; `make workspace` re-stages over the previous run's root-owned output; clean removes the workspace |

## 9. Known constraints this design respects (do not "fix" silently)

- Remote cluster is train-time only: evaluate/score load MOJOs in a local JVM by design; never promise cluster-side scoring. That is why scoring has its own cluster-free `batch` target.
- The lake table's files are immutable (section 4.1). A change to the newest cohort writes new file names and deletes the old; rewriting a file in place would change data under a pinned snapshot without moving it.
- Materialization is full-width: a dataset build writes every column of the table into its splits, and the models then read their 16. At `SCALE=huge` that makes every build, and every `mbt monitor` ground-truth read, scan the whole 2.9GB table; it is mbt's behavior rather than the showcase's (README "Scale" has the measured numbers).
- `--manifest` reads local files only; the manifest travels baked inside the image (which is the design), while `--state` accepts s3:// URIs.
- `mbt clean --artifacts-older-than` prunes file:// stores only and nothing protects champion objects server-side: **SeaweedFS buckets are created with no retention/TTL by the seed path** (`bootstrap/churn_panel.py`), not just documented.
- SeaweedFS capacity is pinned in the compose command (`-master.volumeSizeLimitMB=64 -volume.max=100`), never left to its defaults. Those size volumes at 1GB and cap their count at free disk / volume size, while each bucket grows 7 volumes on first write, so on a docker disk with ~7GB free the lake bucket took every slot and every model upload failed with S3 `InternalError`. `tests/test_showcase_image_pins.py` holds the flags; `docs/troubleshooting.md` has the symptom.
- Switching an existing target's artifact-store scheme strands registered champions (fetch rejects cross-scheme refs); the showcase never flips schemes mid-life.
- On an s3:// artifact store MLflow holds pointers (`mbt.artifact.*` tags) and mbt's small documents such as `inference_config.json`, never the model binaries (mbt uploads a model file to MLflow only from a file:// store); the MOJOs live in `s3://mbt-artifacts/`, and the README points humans at the filer UI to see them.
- The dev uv lock resolves pyspark 4.x; the runner image builds the sparkling fork (`mbt-h2o[sparkling]` pins pyspark 3.5.x) in its own resolution with `uv export --frozen` constraints, and the cluster runs the matching Spark 3.5.8 binaries.
- The image build resolves NOTHING fresh, and that is load-bearing rather than tidiness: the base is digest-pinned, mbt's dependencies come from `uv.lock`, and the one dep that is neither (jupyterlab) has its closure pinned in the committed `images/runner/image-extras.txt`.
  Before that file existed, the pip layer resolved against live PyPI on every build, so an upstream release minutes earlier could break it - which is exactly what happened on 2026-08-27, when statsmodels 0.15.0 shipped macOS wheels at 10:34 UTC and Linux cp311 wheels at 14:30, and the 14:12 nightly fell back to the sdist against an image that carries no compiler.
  `scripts/lock_image_extras.sh` regenerates the closure against uv.lock's pins; `tests/test_showcase_image_pins.py` fails in the fast tier if the two drift apart.
- Woodpecker trusted-repo volumes, the docker socket on the agent, and insecure-registry config are demo-tier security postures; the README says so.
- Every service image is pinned to an exact version, and bumps are verified by running the tier, never taken on faith: Renovate is not active on this repo, so nothing moves these pins automatically, and a rolling tag would let upstream change the stack overnight with no diff (Gitea >= 1.26's `PUBLIC_URL_DETECTION` default flip was exactly that). `zot` stays on v2.1.16 on purpose until project-zot/zot#4140 is fixed in a release (the compose file carries the reason).

## 10. Repo layout

Everything lives in this repo under `examples/showcase/` (yamllint covers `examples/`; the coverage gate scopes only `packages/`; tests must live in repo-root `tests/` per testpaths).

```
examples/showcase/
  README.md                    # runbook: make targets, the DS/CI walkthroughs, test tiers,
                               # deviations from the scaffold defaults, knobs and RAM budget
  DESIGN.md                    # this file
  Makefile                     # help, image, workspace, up, seed, urls, demo, ci, score, outcomes,
                               # monitor, inject-drift, reset, down, clean
  .env.example                 # host ports, S3 credentials, workspace, image tag, DOCKER_SOCK_GID
  images/runner/{Dockerfile,entrypoint.sh}
  images/runner/image-extras.{in,txt}  # the non-mbt image deps and their pinned closure
                               # (constraints.txt is derived from uv.lock at build time,
                               #  by scripts/build_image.sh - it is not a committed file)
  compose/docker-compose.yml   # profiles: core, spark, dev, obs, ci, orch
  compose/{seaweedfs,prometheus,grafana}/...
  bootstrap/churn_panel.py     # the one lake table: seed, land-outcomes, inject-drift, reset
  bootstrap/webhook_sink.py    # the CI alert recorder
  scripts/                     # host-side: build_image.sh, lock_image_extras.sh, ci_bootstrap.py
  project/                     # the churn_lake mbt project (source of truth; pushed into Gitea)
    .woodpecker/{pr-check.yml,prod-build.yml,promote.yml}
    datasets/ models/ scoring/ sources.yml profiles.yml promotions.yml
    deploy/Dockerfile          # the deployable unit: runner image + project + compiled manifest
    notebooks/ds_inner_loop.ipynb  # the DS workbench narrative
    scripts/{run_mbt.sh,push_metrics.py,gitea_pr_comment.py,lint_promotions.py,
             fetch_state.sh,publish_state.sh}
  deploy/                      # the deploy repo source: images.env, dags/, k8s/ (for the k3d tier)
tests/
  test_showcase_infra.py       # SHOW-01/02 + the registry-outage exit-1 contract
  test_showcase_ci.py          # SHOW-05/06/07/10 (CI side)
  test_showcase_lifecycle.py   # SHOW-03/04/10/11/12/13 (CLI-driven)
  test_showcase_provenance.py  # SHOW-08/09
  test_showcase_scheduling.py  # SHOW-11/13 at the scheduler
  test_showcase_obs.py         # SHOW-14
  test_showcase_k3d.py         # SHOW-16 (extra gate, local-only)
  test_showcase_make.py        # SHOW-18 (extra gate, own invocation)
  test_showcase_gates.py       # SHOW-15 (hermetic: fast suite)
  test_showcase_image_pins.py  # hermetic: runner image pins, extras closure, S3 credentials
  test_showcase_panel.py       # hermetic: the lake table's generator and the project's use of it
  showcase_utils.py
```

## 11. Phasing (each phase independently valuable and tested)

1. **P1 Runner image + data/ML core**: seaweedfs, mlflow, spark, jupyterlab, seeded lake, dev/prod targets. Tests SHOW-01..04. Closed integration items A2 (MLflow over HTTP) + A3 (real S3 API).
2. **P2 Git + CI loop**: gitea, woodpecker, bootstrap seeding, pipelines, exit-code wrapper, Gitea PR comments, state branch. Tests SHOW-05..07.
3. **P3 Deployable unit + provenance**: zot, bake + push, commit-time anchors, oras artifacts, the tamper provenance test. Tests SHOW-08..09.
4. **P4 Scheduling + CD + promotion**: airflow + postgres + git-sync, deploy repo, DAGs with exit-code routing, the scoring plane, promotion flow. Tests SHOW-10..13.
5. **P5 Observability + docs**: prometheus/pushgateway/grafana, dashboards, rules, drift injection; docs page in mkdocs nav; one exactly-true sentence in v0.1-status; nightly workflow. Tests SHOW-14..15.
6. **P6 (optional) ArgoCD fidelity tier**: k3d + ArgoCD over the same deploy repo. Test SHOW-16, local-only.

**The one-table simplification (2026-09-23).**
Before it, the showcase ran three cadences over fourteen lake tables and six targets: a daily cadence on tables borrowed from `tests/fixtures/churn_demo`, a monthly cadence on a cluster-free DuckDB plane over a synced lake copy (SHOW-17), and a wide cadence whose five gold tables were joined upstream into two panels, with a feature-selection funnel and Evidently stability gates around it (SHOW-19/20).
P7 ran the wide cadence on Snowflake and P8 on SeaweedFS directly, each as its own target with a namespaced registry.
Each piece earned its keep when it landed, but together they buried the lifecycle story under data plumbing, and the ask was plain: one big table, already in the lake, holding everything.
So the data model is now one generated table, one dataset, two models, one scoring node and four targets, and the retired pieces keep their proof elsewhere:
single-relation datasets are ADR-29's and the fixtures' concern, the Snowflake adapter's live proof is `packages/mbt-snowflake/tests/test_snowflake_live.py` (which now builds the showcase's committed dataset spec on a real account), and scoring straight off an object store is what the `batch` target does on every run.
The simplification also surfaced one mbt bug, fixed in core: a ground-truth join that met a NULL label either graded the run as "single-class" or crashed on NaN, where a table carrying a cohort's rows before its outcomes needs it to mean "not yet".

## 12. Open questions

1. RESOLVED - Zot addressing: the docker daemon uses the published localhost port (insecure-by-default, no daemon config) and in-network consumers use `zot:5000`; see the implementation notes at the top.
2. RESOLVED - the nightly CI job affords the whole main tier, sparkling modules included, plus the make runbook tier on ubuntu-latest (the job has a 120-minute timeout); only the k3d tier stays local-only.
3. Whether to eventually ship `mbt init --forge gitea` scaffolding (Woodpecker pipelines + Gitea PR-comment script) upstream once the showcase proves the port; the showcase keeps them project-local until then.
