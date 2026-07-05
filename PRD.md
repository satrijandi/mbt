# mbt - Product Requirements Document

**Product:** mbt (Model Build Tool) - "dbt for machine learning models"
**Version:** 0.1 · **Status:** Draft for review (revision 2) · **Last updated:** 2026-07-06
**Audience:** the building team (2 engineers + 1 DS design partner)
**Related documents:** [PLAN.md](PLAN.md) (vision, rationale, roadmap) · [TSD.md](TSD.md) (technical design)

---

## 1. Purpose

This document turns the vision in PLAN.md into concrete, testable product requirements.
It covers the full product vision (v0 through v1+), with every requirement tagged by delivery phase so the team can plan, build, and verify incrementally.
PLAN.md remains the source of truth for *why*; this PRD is the source of truth for *what*; TSD.md is the source of truth for *how*.

## 2. Product summary

mbt is a declarative build tool for machine learning models.
A model is described in a reviewed YAML spec: data, algorithm, hyperparameters, evaluation gates, and registration target.
Pluggable adapters execute training; a compiled manifest pins data snapshots, config hashes, seeds, time anchors, and environment digests so runs are reproducible.
State-aware selection (`state:modified+`) retrains only what changed, making ML-in-CI economical.
The core thesis: **the model config IS the model.**

## 3. Goals and success metrics

These are the measurable v0.1 success criteria (from PLAN.md §1.5).

| # | Goal | Metric | Target |
|---|---|---|---|
| G1 | Adoption | Real internal models migrated from notebooks, running the full PR → CI → registry loop | ≥ 2 models |
| G2 | Reproducibility | Re-executing a stored manifest (`mbt build --manifest <path>`) reproduces metrics | Exact for the v0 XGBoost adapter; otherwise within the adapter's documented tolerance tier |
| G3 | Economy | A typical PR retrains only the modified subgraph | CI cost surfaced in the PR comment; unmodified models skipped |
| G4 | Extensibility | A second adapter (LightGBM) built against the public contract + compliance suite | Zero changes to mbt-core required |
| G5 | Time-to-first-model | New user goes from `mbt init` to a trained, registered model via the quickstart | Under 1 hour |

## 4. Non-goals

These are explicitly out of scope, per PLAN.md §1.3 and §9.
Requirements must not creep into these areas.

| Non-goal | Boundary |
|---|---|
| Model serving | mbt stops at the registry + exposures; serving CD (KServe, Argo) consumes mbt outputs |
| Feature store implementation | Integrate (Feast DataAdapter in v1), never build |
| Feature engineering DSL | v0: feature engineering lives in the dataset source (dbt/warehouse) or `hooks.py` |
| Real-time / online learning | Batch training only |
| Labeling and data ingestion | Upstream of mbt |
| A UI | CLI + generated docs site only; no web app |
| Orchestration engine | CI cron covers scheduled retraining in v0; Airflow shells out to mbt in v1 |

## 5. Personas

| Persona | Description | What they need from mbt |
|---|---|---|
| **Dana, Data Scientist** | Builds and owns models; lives in notebooks today | Declare a model in YAML without writing pipeline code; fast local iteration; escape hatch for custom logic; trustworthy evaluation |
| **Miko, MLOps Engineer** | Operates ML infrastructure; reviews DS changes | Reviewable diffs; reproducible runs; CI-friendly commands and exit codes; machine-readable outputs; a plugin contract to extend |
| **CI, the robot** | GitHub Actions (and later other runners) | Non-interactive commands, deterministic behavior, meaningful exit codes, JSON artifacts, state-aware selection to control cost |
| **Riley, Reviewer / Tech Lead** | Approves model changes and promotions | PR diffs that show exactly what changed; metrics vs champion in the PR; promotion as a reviewed change |

## 6. User stories

### Data scientist

- As a DS, I can scaffold a working project with `mbt init` and train my first model within an hour, so adoption is cheap.
- As a DS, I can declare dataset construction (source, filters, label, split policy) in YAML, so my training set is reviewable and reproducible.
- As a DS, I can declare a model (task, adapter, features, hyperparameters, gates, registration) in YAML, so the spec is the single source of truth.
- As a DS, I can exclude leakage-prone columns in config and run leakage checks as tests, so guardrails are part of the spec.
- As a DS, I can declare hyperparameter tuning (engine, trials, search space, objective) and have it capped per environment, so tuning is reproducible and affordable.
- As a DS, I can add custom feature transforms or metrics in a `hooks.py` next to my model spec, so the last 10% never forces me out of the tool.
- As a DS, I can run `mbt build --select my_model` locally against a dev target with sampled data, so iteration is fast.
- As a DS, I get an auto-generated model card with lineage, data window, metrics, and slices, so documentation is free.

### MLOps engineer

- As an MLOps engineer, I can define environments (dev/staging/prod) in `profiles.yml` with different tracking URIs, registries, data samples, and tuning caps, so the same specs run everywhere.
- As an MLOps engineer, I can diff the compiled manifest against production state, so CI retrains only modified nodes.
- As an MLOps engineer, I can re-execute a stored manifest byte-for-byte to audit a training run, so incidents are debuggable.
- As an MLOps engineer, I can build a new training adapter against a versioned public contract and validate it with a compliance suite, so extension never requires core changes.
- As an MLOps engineer, I get structured JSON logs and machine-readable run results, so mbt plugs into existing observability.

### Reviewer

- As a reviewer, I see a PR comment with metrics vs the production champion, gate pass/fail, and a cost estimate, so I can approve model changes like code changes.
- As a reviewer, I can gate promotion to production on a reviewed promotion-file change or a manually approved CD stage, so nothing reaches production without a human decision.

## 7. Functional requirements

Priorities: **Must** (v0.1 does not ship without it), **Should** (v0.1 target, negotiable under pressure), **Could** (stretch), **v1** (post-v0.1).
Phases refer to the delivery roadmap in §9.
Note: a spec *schema* field may ship before the behavior behind it (e.g., slice gates); the schema phase is listed with the resource, the behavior phase with its own requirement.

### 7.1 Project and configuration

| ID | Requirement | Priority | Phase |
|---|---|---|---|
| FR-PROJ-01 | `mbt init <project>` scaffolds a golden-path project: `mbt_project.yml`, example source/dataset/model specs, `profiles.yml` template, pre-commit config, reference CI workflows, CODEOWNERS on `models/` | Must | 0 (full template in 5) |
| FR-PROJ-02 | `mbt_project.yml` defines project name, version, adapter defaults, and var defaults | Must | 0 |
| FR-PROJ-03 | `profiles.yml` (project-local or `~/.mbt/profiles.yml`) defines named targets (dev/staging/prod) with data, tracking, registry, compute adapter configs, artifact store, and per-target vars; selected via `--target` | Must | 1 |
| FR-PROJ-04 | `packages.yml` pins adapter packages; `mbt deps` installs them | Should | 4 |
| FR-PROJ-05 | Secrets resolve via `{{ env_var() }}` only; mbt never writes secret values into the manifest, logs, or run results | Must | 1 |
| FR-PROJ-06 | `--vars` CLI flag overrides project and target vars for a single invocation | Must | 1 |

### 7.2 Resources and specs

| ID | Requirement | Priority | Phase |
|---|---|---|---|
| FR-RES-01 | **source** resources declare external inputs (warehouse tables, lakehouse paths, feature views) in `sources.yml` | Must | 0 |
| FR-RES-02 | **dataset** resources declare training-set construction: source ref, label definition, filters, split policy, checks | Must | 2 (schema in 0) |
| FR-RES-03 | **model** resources declare task, adapter, owner, tags, dataset ref, target column, feature include/exclude, hyperparameters, optional tuning, evaluation protocol + metrics + gates + slices, registration target, materialization, and a mandatory seed | Must | 0 |
| FR-RES-04 | **metric** resources define reusable metric names and parameters in `metrics.yml` | Should | 2 |
| FR-RES-05 | **test** resources cover data tests (schema, nulls, leakage) and model tests (metric gates, slice checks) in `tests/` | Must | 2 |
| FR-RES-06 | **exposure** resources declare downstream consumers for lineage and impact analysis | Should | 3 |
| FR-RES-07 | An optional `hooks.py` per model provides custom feature transforms and custom metrics without breaking the declarative contract; hook edits count as model modifications | Should | 2 |
| FR-RES-08 | The `task` field selects a Pydantic task schema that validates the rest of the model config; task ↔ adapter compatibility is checked at parse time | Must | 0 |
| FR-RES-09 | Temporal split is the default split strategy; random split must be explicitly opted into; the model's declared evaluation protocol must match the dataset's split strategy (parse error otherwise) | Must | 2 |
| FR-RES-10 | `{{ auto }}` values (e.g., `scale_pos_weight`) are resolved by the adapter from the dataset profile at run time and the resolved value is logged and reported | Should | 1 |

### 7.3 Parsing and validation

| ID | Requirement | Priority | Phase |
|---|---|---|---|
| FR-PARSE-01 | `mbt parse` validates all configs against schemas and builds the DAG without executing anything; fast enough for pre-commit use (NFR-03) | Must | 0 |
| FR-PARSE-02 | Validation collects all errors in one pass and reports file, resource name, field path, and a human-actionable message | Must | 0 |
| FR-PARSE-03 | JSON Schema for all resource types is published for editor autocomplete | Must | 0 |
| FR-PARSE-04 | Unknown fields in specs are rejected (typo protection), with a did-you-mean suggestion where possible | Should | 0 |
| FR-PARSE-05 | `mbt ls` lists resources with type, tags, and selector support; `mbt show <resource>` prints the compiled config for one resource; both support JSON output | Must | 0 (`show` in 1) |

### 7.4 Compilation and manifest

| ID | Requirement | Priority | Phase |
|---|---|---|---|
| FR-COMP-01 | `mbt compile` resolves Jinja (`ref`, `source`, `var`, `env_var`), profiles, relative time windows, and data snapshots into `target/manifest.json` | Must | 1 |
| FR-COMP-02 | The manifest pins, per node: resolved config, config hash, upstream dependencies, data snapshot IDs, resolved data windows, seed, and adapter name | Must | 1 |
| FR-COMP-03 | The manifest records an environment digest (Python version + relevant package versions) for reproducibility auditing | Must | 1 |
| FR-COMP-04 | Compilation is deterministic: same inputs at the same time anchor produce a byte-identical manifest; volatile fields (generation timestamp, anchor) are isolated in metadata so tooling can normalize them | Must | 1 |
| FR-COMP-05 | Relative time windows (`"-28d:now"`) resolve against a single manifest-wide anchor; anchor drift alone (same specs, same data snapshot) does not change node identity | Must | 1 |
| FR-COMP-06 | Jinja macros in `macros/` are available in all specs | Should | 2 |

### 7.5 DAG and selection

| ID | Requirement | Priority | Phase |
|---|---|---|---|
| FR-DAG-01 | `ref()` builds a DAG of datasets → models (→ ensembles in v1); cycles are a parse error | Must | 2 |
| FR-DAG-02 | Selectors support: node name globs, `+` upstream/downstream operators (with optional depth), `tag:`, `resource_type:`, `state:new`, `state:modified` | Must | 2 (`state:` in 3) |
| FR-DAG-03 | Comma is intersection, space is union (dbt semantics), e.g. `--select tag:weekly,state:modified+` | Must | 2 |
| FR-DAG-04 | `--exclude` removes nodes from the selection | Should | 2 |
| FR-DAG-05 | Ensembles/stacking: models whose inputs are `ref()`s to other models | v1 | v1 |

### 7.6 Execution: run and build

| ID | Requirement | Priority | Phase |
|---|---|---|---|
| FR-RUN-01 | `mbt run` builds datasets and trains models in DAG order; `mbt build` runs and tests in DAG order, failing a node's downstream if its tests fail | Must | 1 (`build` semantics in 2) |
| FR-RUN-02 | `--threads N` executes independent DAG branches in parallel | Should | 2 |
| FR-RUN-03 | A failed node skips its downstream but does not stop independent branches; `--fail-fast` stops everything | Should | 2 |
| FR-RUN-04 | Every run writes machine-readable `target/run_results.json` with per-node status, timings, metrics, gate results, resolved auto values, and artifact references | Must | 2 |
| FR-RUN-05 | Reproducibility: re-executing a stored manifest (`--manifest`) reproduces metrics exactly, or within the adapter's documented determinism tier (G2) | Must | 1 |
| FR-RUN-06 | mbt warns when a known nondeterminism source is detected (e.g., missing seed passthrough, nondeterministic adapter setting) | Should | 4 |
| FR-RUN-07 | `mbt evaluate --model X` re-evaluates an existing registered artifact on freshly built data, without retraining, optionally applying gates | Should | 3 |
| FR-RUN-08 | `mbt run-operation <macro>` invokes a macro as an escape hatch | Could | 4 |
| FR-RUN-09 | `mbt clean` removes `target/` artifacts | Must | 0 |
| FR-RUN-10 | Training executes through a ComputeAdapter as a serialized job; v0 ships local subprocess execution | Must (local) | 1 |
| FR-RUN-11 | `run`/`build`/`test` accept `--manifest <path>` to execute a previously compiled manifest verbatim (no recompile, no re-anchoring); this is the reproducibility and audit mechanism | Must | 1 |
| FR-RUN-12 | Selecting a model without its upstream dataset still works: required datasets are materialized automatically (cache-aware); selection governs which models *train*, not which data exists | Must | 2 |

### 7.7 Testing and quality gates

| ID | Requirement | Priority | Phase |
|---|---|---|---|
| FR-TEST-01 | `mbt test` runs data tests (schema, nulls, leakage checks) and model tests (metric gates, slice checks) with `--select` support, both standalone and interleaved inside `mbt build` | Must | 2 |
| FR-TEST-02 | Metric gates support absolute thresholds (`threshold`) and champion/challenger comparison (`compare_to: production`, `min_delta`); champion and challenger are compared on the identical pinned test split | Must | 2 |
| FR-TEST-03 | A failing gate blocks registration and exits with code 2 | Must | 2 |
| FR-TEST-04 | Per-slice metrics are reported for declared slices; slice-level gates are supported | Should | 2 (report) / Could (gates) |
| FR-TEST-05 | Dataset checks (`no_future_columns`, `label_leakage_scan`, `class_balance_report`, `schema`, `not_null`) run as part of dataset builds | Should | 2 |
| FR-TEST-06 | When no champion exists yet, `compare_to` gates pass with an explicit warning (bootstrap case); a champion that exists but cannot be loaded is an error, not a pass | Must | 2 |

### 7.8 Tracking and registry

| ID | Requirement | Priority | Phase |
|---|---|---|---|
| FR-REG-01 | Every training run is logged to the tracking backend: params, metrics, artifacts, and mbt metadata (config hash, manifest reference, data snapshot, git commit) | Must | 1 |
| FR-REG-02 | On gate pass, the artifact is registered to the configured registry under the declared name and transitioned to `stage_on_pass` (default staging) | Must | 2 |
| FR-REG-03 | `mbt promote --model X --to production` transitions a registered version, verifying gates were recorded as passed; it supports both promotion-file GitOps and manual CD approval flows | Must | 3 |
| FR-REG-04 | MLflow is the v0 tracking and registry backend; both are adapters behind contracts; stage names are canonical mbt tokens mapped by the registry adapter | Must | 1 |
| FR-REG-05 | Registered model metadata links back to the manifest: config hash ↔ data snapshot ↔ model version | Must | 2 |

### 7.9 Hyperparameter tuning

| ID | Requirement | Priority | Phase |
|---|---|---|---|
| FR-TUNE-01 | Optional `tuning` block: engine, `n_trials`, typed search space, objective metric + direction | Must | 4 |
| FR-TUNE-02 | Optuna is the v0 tuning engine, packaged separately, behind a TuningEngine contract | Must | 4 |
| FR-TUNE-03 | Tuning is seeded and reproducible; trials never see the test split; trial history is logged to tracking | Must | 4 |
| FR-TUNE-04 | Per-target trial caps (e.g., dev = 5, prod = 50) via profiles/vars | Must | 4 |

### 7.10 State and GitOps

| ID | Requirement | Priority | Phase |
|---|---|---|---|
| FR-STATE-01 | `state:modified` selects nodes whose spec, hooks, upstream deps, or pinned data snapshot changed vs a reference manifest (`--state <path-or-URI>`); time-anchor drift over an unchanged snapshot does not by itself count as modified | Must | 3 |
| FR-STATE-02 | `mbt state diff` prints what changed vs a previous manifest (including which component changed: config, hooks, snapshot, upstream, environment), human-readable and as JSON | Must | 3 |
| FR-STATE-03 | A documented manifest storage convention (S3 or CI artifact per environment, with a `latest` pointer) links git commits to manifests | Must | 3 |
| FR-STATE-04 | Reference GitHub Actions ship with the template: PR check (parse + compile + build modified subgraph on dev target), prod build on merge, promotion workflow | Must | 3 |
| FR-STATE-05 | The PR check posts a comment: metrics vs champion, gate pass/fail, retrained node list, and a cost estimate | Should | 3 |
| FR-STATE-06 | Scheduled retraining is expressible as CI cron using tag selection (e.g., `mbt build --select tag:weekly`); no orchestrator concept in v0 | Must | 3 |

### 7.11 Documentation generation

| ID | Requirement | Priority | Phase |
|---|---|---|---|
| FR-DOCS-01 | `mbt docs generate` produces a static site: model cards (lineage, data window, metrics, slices, owner) + DAG lineage view | Must | 4 |
| FR-DOCS-02 | `mbt docs serve` serves the site locally | Should | 4 |
| FR-DOCS-03 | Exposures appear in lineage for impact analysis | Should | 4 |

### 7.12 Adapter ecosystem

| ID | Requirement | Priority | Phase |
|---|---|---|---|
| FR-ADPT-01 | Core defines versioned contracts for adapter kinds (Training, Data, Tracking, Registry, Compute, Tuning); core never imports an ML framework | Must | 1 (extracted as `mbt-adapter-base` in 4) |
| FR-ADPT-02 | Adapters are discovered via Python entry points (`mbt.adapters` group); `pip install mbt-xgboost` + `adapter: xgboost` just works; plugin modules must be cheap to import (no framework imports at parse time) | Must | 1 |
| FR-ADPT-03 | The XGBoost training adapter ships in v0.1 supporting binary classification (regression/multiclass are stretch), deterministic under a fixed seed, with native save/load and an ONNX export path | Must | 1 |
| FR-ADPT-04 | A local DataAdapter builds datasets from Parquet (Iceberg optional) via DuckDB/Polars, including split materialization, caching, and snapshot IDs | Must | 1 |
| FR-ADPT-05 | An adapter compliance test suite validates any adapter against the contract (including determinism and import hygiene); passing it is the ship bar | Must | 4 |
| FR-ADPT-06 | A LightGBM adapter is built using only public contracts + the compliance suite, as the extensibility proof (G4) | Must | 4 |
| FR-ADPT-07 | Each adapter documents its determinism tier; gates use tolerance bands where exact reproducibility is impossible | Must | 4 |

### 7.13 CLI cross-cutting

| ID | Requirement | Priority | Phase |
|---|---|---|---|
| FR-CLI-01 | Every command is non-interactive-safe (no prompts when stdin is not a TTY) | Must | 1 |
| FR-CLI-02 | Exit codes: 0 = success, 1 = error, 2 = test/gate failure (hard errors take precedence over quality failures) | Must | 2 |
| FR-CLI-03 | Human output uses Rich tables/progress; `--log-format json` emits structured JSON event lines | Must | 1 (JSON) / Should (Rich polish) |
| FR-CLI-04 | `--target`, `--vars`, `--select`, `--exclude`, `--threads`, `--state`, `--manifest` behave consistently across commands | Must | 2 |

### 7.14 v1+ requirements (post-v0.1, architecture must accommodate)

These ship after v0.1, driven by dogfooding feedback.
The TSD must show that the v0 architecture accommodates them without redesign (TSD §22).

| ID | Requirement | Phase |
|---|---|---|
| FR-V1-01 | Kubernetes Job and Ray ComputeAdapters: `mbt run` submits training jobs to remote compute via the ComputeAdapter contract | v1 |
| FR-V1-02 | sklearn adapter (Pipeline-based) and PyTorch adapter | v1 |
| FR-V1-03 | Survival and ranking task schemas (XGBoost AFT / ranking objectives), registered by adapters without core changes | v1 |
| FR-V1-04 | Feast DataAdapter: dataset specs reference feature-store feature views | v1 |
| FR-V1-05 | Ensembles/stacking as first-class models with `ref()` inputs from other models | v1 |
| FR-V1-06 | `mbt score`: batch inference from a registered artifact over a dataset resource | v1 |
| FR-V1-07 | Airflow provider (operator wrapping mbt commands, manifest-aware task mapping) | v1 |
| FR-V1-08 | Additional materializations: `ensemble`, `calibrated`, `onnx` | v1 (onnx export path proven in v0) |

## 8. Non-functional requirements

| ID | Requirement |
|---|---|
| NFR-01 | **Reproducibility:** identical manifest → identical metrics, or documented adapter tolerance tier (G2) |
| NFR-02 | **Determinism of compilation:** compile output is stable across machines given the same inputs, environment, and anchor |
| NFR-03 | **Performance:** `mbt parse` on a 50-resource project completes in < 2 s; compile in < 10 s; execution overhead per node (excluding training itself) < 2 s |
| NFR-04 | **Code quality:** mbt-core is Python 3.11+, `mypy --strict` clean, ruff clean, Pydantic v2 schemas, zero ML framework dependencies |
| NFR-05 | **Compatibility:** semantic versioning for all packages; the adapter contract is versioned independently with a deprecation policy from day one |
| NFR-06 | **Observability:** structured JSON logging, OpenTelemetry hooks, machine-readable `run_results.json` |
| NFR-07 | **Security:** secrets only via `env_var()`; never persisted to manifest, logs, run results, or tracking; profiles live outside the repo by default |
| NFR-08 | **Testability of mbt itself:** pytest + hypothesis; golden-file tests for compile output; adapter compliance suite |
| NFR-09 | **Docs:** docs-as-code (mkdocs-material), ADRs for design decisions, external-quality README + one-hour quickstart at v0.1 |
| NFR-10 | **Packaging:** uv + hatchling monorepo; PyPI publication at v0.1 (verify the `mbt` name is available before release) |

## 9. Release plan and acceptance criteria

Timeline: ~18 weeks to v0.1 with 2 engineers + 1 DS design partner (PLAN.md §8).
Scope lists name the primary requirements delivered in that phase; a few schema-only fields land earlier than their behavior, as noted in §7.

| Phase | Duration | Scope | Acceptance criteria (Definition of Done) |
|---|---|---|---|
| **0 - Spike & spec** | 2 wks | FR-PROJ-01/02, FR-RES-01/03/08, FR-PARSE-01..05, FR-RUN-09 | A demo project parses; `mbt init/parse/ls` work; JSON Schema published for editor autocomplete; naming + initial ADRs recorded |
| **1 - Vertical slice** | 4 wks | FR-COMP-01..05, FR-RUN-01/05/10/11, FR-RES-10, FR-REG-01/04, FR-ADPT-01..04, FR-CLI-01/03, FR-PROJ-03/05/06, FR-PARSE-05 (`show`) | `mbt build` trains a real churn model end-to-end from YAML on local Parquet, tracked in MLflow; `mbt build --manifest` reruns with identical metrics |
| **2 - DAG & gates** | 4 wks | FR-DAG-01..04, FR-RES-02/04/05/07/09, FR-TEST-01..06, FR-REG-02/05, FR-RUN-02/03/04/12, FR-COMP-06, FR-CLI-02/04 | Multi-model project with dependencies runs; a failing gate blocks registration with exit code 2; `run_results.json` emitted |
| **3 - GitOps & state** | 3 wks | FR-STATE-01..06, FR-REG-03, FR-RUN-07, FR-RES-06 | The full PR → CI → registry → promotion loop runs on a demo repo; PR comment bot shows metric deltas; model-only PRs retrain just the model (FR-RUN-12) |
| **4 - Adapter hardening** | 3 wks | FR-ADPT-05/06/07, FR-TUNE-01..04, FR-DOCS-01..03, FR-PROJ-04, FR-RUN-06/08 | LightGBM adapter built by "someone else" using only public contracts + compliance suite; Optuna tuning works; docs site generates |
| **5 - v0.1 release** | 2 wks | Packaging, quickstart, template repo, dogfooding | PyPI release; external-quality README + tutorial; 2 real models migrated from notebooks (G1); quickstart under 1 hour (G5) |
| **v1+** | - | FR-V1-01..08 | Driven by dogfooding feedback |

## 10. Dependencies and assumptions

- Python 3.11+ is acceptable to all target users.
- MLflow is available (or trivially self-hostable) in target environments for tracking and registry.
- Training data for v0 use cases is reachable as local/lakehouse Parquet (Iceberg optional); warehouse-native data adapters come later.
- v0 models train comfortably on a single machine (CI runner or dev box); remote compute is v1.
- GitHub Actions is the reference CI; templates are portable in principle.
- A DS design partner is available throughout, and 2-3 flagship models are candidates for migration.
- The `mbt` package name is available on PyPI, or a fallback name is chosen before release.

## 11. Risks and open questions

Positions are maintained in PLAN.md §9; the key product-level ones:

| Risk | Mitigation in requirements |
|---|---|
| YAML hell | Task schemas keep configs small (FR-RES-08); `hooks.py` escape hatch (FR-RES-07); macros (FR-COMP-06); resist adding knobs until dogfooding demands them |
| CI training cost | `state:modified` (FR-STATE-01), anchor-drift exclusion (FR-COMP-05), sampled dev targets (FR-PROJ-03), trial caps (FR-TUNE-04), cost in PR comment (FR-STATE-05) |
| DS adoption | Golden-path init (FR-PROJ-01), one-hour quickstart (G5), model cards (FR-DOCS-01), migrate flagship models with the design partner |
| Nondeterministic training | Determinism tiers per adapter (FR-ADPT-07), tolerance-band gates, nondeterminism warnings (FR-RUN-06) |
| Scope creep toward orchestration/serving | Non-goals in §4 are hard boundaries |

## 12. Glossary

| Term | Meaning |
|---|---|
| **Adapter** | A plugin implementing one or more mbt contracts (Training, Data, Tracking, Registry, Compute, Tuning) |
| **Anchor** | The single UTC timestamp a manifest resolves relative time windows against; pinned at compile, reused verbatim by `--manifest` runs |
| **Manifest** | `target/manifest.json`: the compiled, pinned representation of the whole project for one target |
| **Node** | One resource instance in the DAG (a dataset, a model, ...) |
| **Gate** | A metric condition that must pass for an artifact to be registered/promoted |
| **Champion/challenger** | Comparing a newly trained model against the current production model version on the same pinned test split |
| **Target** | A named environment in `profiles.yml` (dev/staging/prod) |
| **Task schema** | A Pydantic schema selected by the `task` field that validates a model spec for that ML task |
| **Determinism tier** | An adapter's documented reproducibility guarantee (exact / tolerance band) |
