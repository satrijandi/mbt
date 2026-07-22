# MLOps alignment

mbt's design goal is to make the MLOps textbook loop the default path of least resistance, not an aspiration.
This page audits mbt v0.1 against the practice catalogue at [ml-ops.org](https://ml-ops.org): the [MLOps principles](https://ml-ops.org/content/mlops-principles) and their Data/Model/Code matrix, the [automation maturity levels](https://ml-ops.org/content/mlops-principles), [CRISP-ML(Q)](https://ml-ops.org/content/crisp-ml), the [serving patterns](https://ml-ops.org/content/three-levels-of-ml-software), and the [model governance guide](https://ml-ops.org/content/model-governance).
Every row links to the doc or ADR that carries the claim; the [v0.1 status page](v0.1-status.md) carries the test-suite evidence behind those docs.

Three statuses appear below:

- **Built in** - mbt itself enforces or produces the practice.
- **Recipe** - mbt provides the primitives; the wiring is documented but belongs to the project.
- **Non-goal / organizational** - deliberately outside a build tool's scope, with the seam documented.

## The principles matrix

ml-ops.org summarizes MLOps as a set of principles applied to three artifact classes: Data, ML Model, and Code.
mbt's coverage, cell by cell:

| Principle | Data | ML Model | Code |
|---|---|---|---|
| **Versioning** | Every compiled manifest pins a snapshot token per source; new data arrives as a snapshot change that marks nodes modified ([concepts](concepts.md#identity-and-state), ADR-11) | Specs, hyperparameters, and hooks are reviewed YAML/Python in git; every run lands in MLflow with config and input hashes | One repo holds specs, tests, hooks, and CI; profiles never enter identity (ADR-5), so environments version separately from models |
| **Testing** | Dataset `checks` (schema, `not_null`, `no_future_columns`, class balance, and `label_leakage_scan`, which runs by default) plus Python data tests gate the pipeline with exit code 2 ([spec reference](spec-reference.md)) | Threshold gates, paired-bootstrap champion gates (ADR-18), slice gates, and realized-metric gates block registration ([concepts](concepts.md#quality-gates)) | 750+ unit/property/golden/compliance/E2E tests on mbt itself (v0.1 status NFR-08); adapters must pass a shared compliance suite before they ship |
| **Automation** | Dataset materialization is declarative, cached by snapshot, and auto-materializes upstream dependencies (ADR-13) | Training, Optuna tuning, evaluation, and registration all run from one `mbt build` | The scaffold ships seven workflows: PR check, prod build, promotion, scheduled retrain, monthly retrain, scheduled score, scheduled monitor ([GitOps & CI](gitops.md)) |
| **Reproducibility** | Snapshot pinning end to end; a drifted source under a manifest pin is a hard error, never a silent retrain | Mandatory seeds with documented derivations, per-adapter determinism tiers, bit-identical `--manifest` reruns ([concepts](concepts.md#reproducibility-contract)) | `env_digest` and `env_freeze_digest` verification on manifest execution (ADR-19); the showcase bakes a digest-pinned runner image |
| **Deployment** | The same specs serve dev and prod; targets differ only in profiles (sampling, trial caps, endpoints) | Registry stages with gate-verified promotion; run-time champion resolution makes promotion a zero-redeploy operation (ADR-20) | GitOps loop: reviewed promotion files, protected state baselines, and a digest-pinned deployable unit in the [showcase](showcase.md) |
| **Monitoring** | Input `checks` and PSI/KS feature-shift monitors run on every scoring batch ([concepts](concepts.md#batch-scoring-and-monitoring)) | Prediction-shift monitors, delayed ground-truth realized-metric gates (ADR-21), and staleness alerting | Machine-readable `run_results.json`, typed JSON event stream, webhook alerts, and a Prometheus/Grafana spec proven in the showcase |

## Automation maturity: level 2

ml-ops.org defines three automation levels, from manual notebooks (level 0) through automated training pipelines (level 1) to full CI/CD pipeline automation (level 2).
An mbt project operates at level 2 from `mbt init` onward, because the scaffold ships the CI/CD system rather than describing one:

- **Continuous integration** extends beyond code, exactly as the site demands: every PR compiles, diffs state against the production baseline, and retrains plus gates the modified subgraph on sampled dev data, so data and models are validated on every change.
- **Continuous delivery** publishes gate-passing models to the registry; the showcase extends this to a deployable unit reconciled by Airflow or ArgoCD, where promotion never requires a redeploy.
- **Continuous training** is scheduled retraining plus `state:modified` economy: freshness arrives as new snapshots, and only what actually changed retrains ([GitOps & CI](gitops.md)).
- **Continuous monitoring** is shift monitors, realized-metric gates, and alerting wired to exit-code semantics, so a quality breach (exit 2) is never confused with an infrastructure failure (exit 1).

## CRISP-ML(Q) phases

| Phase | Where it lives with mbt |
|---|---|
| 1. Business & data understanding | Organizational by design, but the spec forces the artifacts into review: `description`, `owner`, `label.definition`, and gates that encode the launch criteria as measurable thresholds |
| 2. Data engineering | Declarative datasets with checks, temporal windows, joins, and reproducible sampling; snapshot pinning makes every input reconstructible |
| 3. Model engineering | Adapter-executed training with mandatory seeds, Optuna tuning that never sees the test split (ADR-8), and full training metadata in the manifest and tracking server |
| 4. Model evaluation | Held-out test metrics, threshold/champion/slice gates; the deploy/no-deploy decision is automated (exit 2 blocks registration) with human promotion approval layered on top |
| 5. Model deployment | Registry stages, gate-verified `mbt promote`, batch scoring pipelines as first-class `scoring` resources |
| 6. Monitoring & maintenance | `mbt score` monitors every batch, `mbt monitor` evaluates realized metrics once labels mature, alerts fire on breach, and decay routes back into retraining through schedules and snapshots |

## Serving pattern: a deliberate choice

The site's three-levels guide catalogues four serving patterns (Model-as-Service, Model-as-Dependency, Precompute, Model-on-Demand) and asks teams to choose deliberately.
mbt chose **precompute (batch) serving**: `mbt score` materializes predictions into a store, every prediction carries a sidecar with the champion version and identity hashes, and the registry is the seam for anything downstream.
Online (request/response) serving is an explicit non-goal ([roadmap](roadmap.md)); serving infrastructure reacts to registry stages instead of mbt deploying services.

## Data science hygiene

- Temporal splits are the default; random splits require a seed and emit parse-time warnings for time-column leakage and entity-straddle hazards ([spec reference](spec-reference.md)).
- The test split is carved at compile time and tuning never touches it (ADR-8).
- Leakage guards are layered: target and time columns are always excluded, `label_leakage_scan` runs by default, and `no_future_columns` checks each split against its own window end.
- Champion gates use a paired-bootstrap lower bound so a challenger ahead on test-set noise alone cannot promote (ADR-18).
- A report-only `class_balance_report` check ships built in; stratified and grouped (`sample_key`) sampling are first-class.

## Data engineering hygiene

- Data versioning needs no extra tool: snapshot tokens per source (mtime listing or content hashes, warehouse change-commit times, lakehouse file listings) are pinned in the manifest (ADR-11).
- Scoring runs are idempotent: same-anchor reruns overwrite a single `run_key`, new anchors partition (ADR-21).
- Exit codes carry meaning everywhere: 0 success, 1 hard error, 2 quality failure, and CI/schedulers preserve the distinction.
- Events go to stderr (human or JSON lines); stdout stays machine-readable command data.
- Secrets resolve via `env_var()` only, are tainted and redacted in output, and never enter manifests.

## Governance

The site's governance guide asks for recording, auditing, validation, approval, and monitoring at every stage.

- **Reconstruction metadata**: the manifest records algorithm, features, transformations (hook bytes are hashed into identity), data snapshots, hyperparameters, environment digests, and the git provenance to rebuild any model ([concepts](concepts.md#reproducibility-contract)).
- **Model cards**: `mbt docs generate` renders a card per model (data window, features, hyperparameters, metrics, slices, gate history, registry/tracking IDs) plus a lineage site with `exposures.yml` impact analysis.
- **Approval gates**: CODEOWNERS on specs and `promotions.yml`, branch protection, CI environment approvals, and `mbt promote` refusing versions without recorded gate passes.
- **Audit trails**: the `mbt-state` branch is an append-only history of every published baseline; the showcase adds oras provenance artifacts per source sha and a once-per-run evaluation ledger.
- **Traceability**: every stored prediction carries the producing model version and identity hashes, satisfying the site's "every prediction traceable to the model version" requirement.
- **Catalog**: the MLflow registry plus the generated docs site serve as the model inventory; a dedicated searchable catalog product is out of scope.

## Honest gaps

The practices below are catalogued on ml-ops.org and are not fully covered.
Consistent with this project's documentation standards, they are stated plainly rather than rounded up.

| Practice | Status |
|---|---|
| Feature store parity between dev and prod | Non-goal in v0.1; a Feast DataAdapter is a v1 candidate ([roadmap](roadmap.md)). The showcase's lake is plain parquet on S3, not a feature store |
| Online serving, input pre-assertions at request time, canary/shadow rollout | Non-goal; batch scoring covers input checks and staged (dev/prod target) validation instead. A shadow-style comparison is a recipe: point a second scoring pipeline at the `staging` stage |
| Named fairness metrics (equalized odds, demographic parity) | A relative-disparity gate ships (`across` + `min_ratio`: gate the worst group's metric as a ratio of the best across a column); the classic named fairness metrics and a formal protected-attribute type are not built in |
| SHAP on the JVM adapters | xgboost/lightgbm cards rank features by SHAP and scoring can attach per-prediction SHAP drivers (`explain_top_k`); partial-dependence curves render for any adapter, but H2O and Spark importance stays model-intrinsic (no SHAP) |
| Multi-family model comparison | Not built in; comparing families means parallel specs over a shared dataset, tracked side by side in MLflow. Cross-validation itself now ships (walk-forward / k-fold / nested via `backtest_folds`/`nested_cv`), so a single pinned held-out split is no longer the only evaluation |
| Non-ML baseline benchmark | No first-class baseline gate; a trivial-model spec can be trained alongside and compared in tracking, but nothing enforces it |
| Drift-triggered retraining | Retraining is scheduled or change-driven; a shift breach alerts (exit 2) but does not automatically enqueue a retrain |
| Numerical-stability monitoring | Input checks cover schema and nulls; there is no dedicated NaN/infinity monitor on features or scores |
| Delivery metrics (deployment frequency, lead time, MTTR, change failure rate) | The raw material exists (`mbt-state` history, registry timestamps, `run_results.json`) but no first-class report computes them |
| OpenTelemetry metrics and exporter wiring | Spans ship opt-in (`MBT_OTEL`, NFR-06): one run/node trace per command against your configured tracer. mbt still exports no metrics and ships no exporter - you supply the `OTEL_*` destination; the events + run-results + Pushgateway/Prometheus spec remain the metrics path |
| Per-attribute dataset profiling catalog | Schema checks and class-balance reports ship; full attribute statistics (min/max/missing ratios/distributions) do not |
| Sensitive-data classification (GDPR data sheets) | Organizational; mbt's contribution is keeping secrets out of manifests and redacting tainted values, not classifying data content |

## Stack canvas, answered

The [MLOps Stack Canvas](https://ml-ops.org/content/mlops-stack-canvas) asks teams to answer each infrastructure block explicitly and record decisions as ADRs.
This repository practices what the canvas preaches: 24 [ADRs](adr/0001-arrow-interchange.md) record the load-bearing decisions with context and consequences, and the canvas blocks map to concrete choices - data versioning (snapshot tokens), experiment management (MLflow), pipelines (declarative DAG), registry (MLflow stages/aliases), deployment (GitOps + batch scoring), monitoring (shift + ground truth), and the metadata store (manifest + run results + tracking).
The buy-vs-build stance is explicit throughout: integrate MLflow, Optuna, and Feast (v1) rather than rebuild them, and keep mbt itself a thin, deterministic build tool.
