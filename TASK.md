# mbt - Task Breakdown (TASK.md)

**Product:** mbt (Model Build Tool)
**Version:** 0.1 · **Status:** Draft for review · **Last updated:** 2026-07-06
**Related documents:** [PRD.md](PRD.md) (requirements) · [TSD.md](TSD.md) (technical design) · [PLAN.md](PLAN.md) (vision)

This document breaks the v0.1 scope into sprints and tasks with testable DONE definitions.
PRD.md defines *what* and TSD.md defines *how*; this document defines *when* and *in what order*.

---

## 1. How to read this document

- The 18-week plan from PRD §9 is split into nine 2-week sprints for a team of 2 engineers + 1 DS design partner.
- Sprint boundaries align with PRD delivery phases; where a phase spans 3 weeks, the tail is absorbed by the next sprint (noted per sprint).
- Task IDs are `S<sprint>-<nn>` (e.g. `S3-04`) and are referenced from the traceability matrix in §12.
- Every task carries a **Refs** line mapping it to PRD requirement IDs (`FR-*`, `NFR-*`, `G*` from PRD §7, §8, §3) and TSD sections (`§n`) or ADRs (`ADR-n`, TSD §23).
- A task is only DONE when its listed criteria hold **and** the global Definition of Done in §2 holds.
- §12 provides the reverse mapping (requirement → tasks) to verify nothing in the PRD is unowned.

### Sprint overview

| Sprint | Weeks | PRD phase (§9) | Theme | Exit milestone |
|---|---|---|---|---|
| 1 | W1-W2 | Phase 0 - Spike & spec | Parse, schemas, CLI skeleton | Demo project parses; `init`/`parse`/`ls` work |
| 2 | W3-W4 | Phase 1 (first half) | Compile, manifest, contracts | Deterministic manifest with pinned snapshots and hashes |
| 3 | W5-W6 | Phase 1 (second half) | Execution vertical slice | Churn model trains from YAML, tracked in MLflow, reproducible |
| 4 | W7-W8 | Phase 2 (first half) | DAG, datasets, tests, hooks | Multi-node DAG with selectors and dataset checks |
| 5 | W9-W10 | Phase 2 (second half) | Gates, registry, run semantics | Failing gate blocks registration with exit 2 |
| 6 | W11-W12 | Phase 3 (weeks 1-2) | State selection and CI workflows | `state:modified+` retrains only the changed subgraph |
| 7 | W13-W14 | Phase 3 (week 3) + Phase 4 (week 1) | Promotion, evaluate, PR bot, adapter-base | Full PR → CI → registry → promotion loop on demo repo |
| 8 | W15-W16 | Phase 4 (weeks 2-3) | Compliance, LightGBM, tuning, docs | LightGBM passes compliance; Optuna tuning; docs site |
| 9 | W17-W18 | Phase 5 - v0.1 release | Packaging, quickstart, dogfooding | PyPI release; 2 real models migrated; 1-hour quickstart |

## 2. Global Definition of Done (applies to every task)

A task is DONE only when all of the following hold, in addition to its own criteria:

- Code is merged to `main` behind a green CI run.
- `mypy --strict` and `ruff` are clean; no ML framework imports in mbt-core (PRD NFR-04, TSD §2).
- Unit tests cover the new behavior; test layers follow TSD §21.
- User-facing behavior is documented (docstring, `--help` text, or docs page as appropriate; PRD NFR-09).
- Structured error messages state what happened, which resource/file, and what to do next (TSD §17).
- New design decisions that deviate from or refine the TSD are recorded as ADRs in `docs/adr/` (PRD NFR-09).

---

## 3. Sprint 1 (W1-W2) - Spike & spec

**Maps to:** PRD §9 Phase 0.
**Goal:** a demo project parses with rich errors; `mbt init`, `mbt parse`, `mbt ls`, `mbt clean` work; JSON Schema is published.

### S1-01 - Monorepo scaffold and quality toolchain

**Refs:** PRD NFR-04, NFR-08, NFR-10; TSD §2, §21.

DONE when:

- The uv workspace with hatchling builds exists with `packages/mbt-core` importable and the package layout of TSD §2.
- Shared dev tooling (ruff, mypy --strict, pytest, yamllint, pre-commit) runs locally and in a GitHub Actions CI workflow on every PR.
- CI fails on lint, type, or test errors; the repo README states the dev setup in under 10 commands.

### S1-02 - Core resource schemas (project, sources, datasets, models)

**Refs:** PRD FR-PROJ-02, FR-RES-01, FR-RES-02 (schema only), FR-RES-03, FR-PARSE-04; TSD §4, §5.1, §5.2, §5.4, §5.5, §5.6.

DONE when:

- `ProjectConfig`, `SourceGroup`/`SourceTable`, `DatasetSpec`, `ModelSpec`, and all common types from TSD §5.1 exist as Pydantic v2 models with `extra="forbid"`.
- `ModelSpec` enforces the mandatory `seed` with no default (PRD FR-RES-03).
- Unknown fields are rejected with a Levenshtein did-you-mean suggestion (PRD FR-PARSE-04).
- Hypothesis round-trip tests (YAML → model → JSON Schema validation) pass for every schema (TSD §21).
- unique_id construction (`<type>.<project>.<name>`) and per-type name uniqueness errors work (TSD §4).

### S1-03 - Task schema registry

**Refs:** PRD FR-RES-08; TSD §5.6.

DONE when:

- A `TaskType → TaskSchema` registry exists with the `binary_classification` task schema implemented.
- `validate_spec` runs at parse time and rejects metric names outside `allowed_metrics` (hook metrics exempt).
- Task ↔ adapter compatibility is checked at parse time and produces an actionable error on mismatch.

### S1-04 - Jinja capture phase and DAG skeleton

**Refs:** PRD FR-PARSE-01, FR-DAG-01 (initial); TSD §6, §9.1.

DONE when:

- A sandboxed Jinja environment with `StrictUndefined` provides `ref`, `source`, `var`, `env_var`, `target`, `auto` (TSD §6).
- Capture-phase rendering records `ref()`/`source()` edges and returns placeholders without needing profiles or environment.
- A `networkx.DiGraph` is built from captured edges; a cycle is a parse error reporting the full cycle path.

### S1-05 - Parsing pipeline with all-errors validation

**Refs:** PRD FR-PARSE-01, FR-PARSE-02; NFR-03; TSD §7.

DONE when:

- Discovery walks configured paths and matches YAML files to resource schemas by location and top-level key.
- Validation collects all errors in one pass; each error reports file, resource name, JSON-pointer field path, and message (PRD FR-PARSE-02).
- A fixture project seeded with 5+ distinct errors reports all of them in a single `mbt parse` invocation.
- A missing adapter is a parse error naming the pip package to install (TSD §7).

### S1-06 - CLI skeleton: `parse`, `ls`, `clean`

**Refs:** PRD FR-PARSE-01, FR-PARSE-05 (`ls`), FR-RUN-09, FR-CLI-01 (basic); TSD §3.

DONE when:

- The Typer app exposes `mbt parse`, `mbt ls`, `mbt clean` with the global flags from TSD §3 (`--project-dir`, `--profiles-dir`, `--target`, `--vars`, `--log-format`, `--quiet`).
- `mbt ls` supports `--output table|name|path|json`.
- `mbt clean` removes `target/` including the dataset cache.
- No command prompts when stdin is not a TTY; errors exit 1, success exits 0.

### S1-07 - `mbt init` golden-path scaffold (v0 subset)

**Refs:** PRD FR-PROJ-01 (full template lands in S9-01); TSD §3.

DONE when:

- `mbt init <name>` scaffolds `mbt_project.yml`, example source/dataset/model specs, and a `profiles.yml` template.
- `mbt init demo && cd demo && mbt parse` succeeds on a clean machine.
- The scaffold gitignores project-local `profiles.yml` (TSD §18).

### S1-08 - JSON Schema publication

**Refs:** PRD FR-PARSE-03; TSD §5.

DONE when:

- `mbt parse --write-json-schema` emits JSON Schema for all resource types.
- The example specs from the init template validate against the emitted schemas via a standard JSON Schema validator in CI.
- Editor autocomplete is verified once manually in VS Code with the yaml-language-server header.

### S1-09 - ADRs, naming, and parse performance budget

**Refs:** PRD NFR-03, NFR-09; PRD §9 Phase 0 DoD; TSD §20, §23.

DONE when:

- Initial ADRs (at minimum ADR-1..ADR-5 from TSD §23) are recorded in `docs/adr/`.
- The `mbt` PyPI name is checked and the naming decision (or fallback) is recorded (PRD §10).
- A CI perf test parses a generated 50-resource project in < 2 s (PRD NFR-03).

**Sprint 1 exit criteria (PRD §9 Phase 0 DoD):** a demo project parses; `mbt init/parse/ls` work; JSON Schema is published; naming and initial ADRs are recorded.

---

## 4. Sprint 2 (W3-W4) - Compile and contracts

**Maps to:** PRD §9 Phase 1, first half.
**Goal:** `mbt compile` produces a deterministic manifest with pinned anchor, snapshots, and hashes; adapter contracts and plugin discovery exist.

### S2-01 - Profiles and targets

**Refs:** PRD FR-PROJ-03; TSD §5.3.

DONE when:

- `ProfilesFile`/`ProfilesConfig`/`TargetConfig`/`AdapterRef` schemas validate the TSD §5.3 example verbatim.
- Search order is `--profiles-dir`, `$MBT_PROFILES_DIR`, `./profiles.yml`, `~/.mbt/profiles.yml`, first hit wins.
- `--target` selects an output; per-target `vars` (e.g. `sample_fraction`, `max_tuning_trials`) are exposed to `var()`.
- Jinja in profiles renders before validation.

### S2-02 - Resolve-phase Jinja and vars precedence

**Refs:** PRD FR-COMP-01, FR-PROJ-06; TSD §6.

DONE when:

- Resolve-phase rendering substitutes real values for `ref`, `source`, `var`, `env_var`, `target` against the selected target.
- `var()` resolution order is `--vars` CLI > target vars > project vars > default; missing with no default is a compile error naming the variable.
- `--vars` accepts a YAML/JSON dict and applies for a single invocation.

### S2-03 - Secret handling

**Refs:** PRD FR-PROJ-05, NFR-07; TSD §18.

DONE when:

- `env_var()` values are wrapped in a `Secret` type whose `repr`/serialization is `"***"`.
- The manifest stores profile configs unrendered (the `env_var()` expression, not the value).
- A regression test injects a sentinel secret and asserts it appears nowhere in manifest, events, logs, or run_results.

### S2-04 - Window expressions and time anchoring

**Refs:** PRD FR-COMP-05; TSD §5.5, §8.2; ADR-12.

DONE when:

- The window grammar (`"<start>:<end>"`, bare-duration sugar, `d/w/h` units, ISO dates, `now`) parses with error-case tests.
- Compile pins one manifest-wide UTC anchor, overridable with `--anchor <iso-ts>`.
- Window expressions resolve to concrete `[start_ts, end_ts)` ranges stored under `resolved` outside the hashed config.
- A mutation test proves anchor drift alone changes no node hash (PRD FR-COMP-05).

### S2-05 - Adapter contracts and interchange types

**Refs:** PRD FR-ADPT-01; TSD §12.1, §12.2.

DONE when:

- All Protocols (Training, Data, Tracking, Registry, Compute, Tuning, DatasetHandle) and interchange types (ValidationIssue, DeterminismTier, ArtifactRef, MetricResults, DatasetProfile, DatasetLocator, ModelVersion, RunHandle, TrainingJob, JobResult, TestResult, TuningResult, HookContext) exist in `mbt.contracts` as specified in TSD §12.
- Everything crossing a contract boundary is a plain Pydantic model or Arrow table; a CI check asserts mbt-core imports no ML framework (PRD NFR-04).
- Contracts carry a `contract_version` starting at `1.0`.

### S2-06 - Plugin discovery and import hygiene

**Refs:** PRD FR-ADPT-02; TSD §12.3; ADR-14.

DONE when:

- Adapters are discovered via the `mbt.adapters` entry-point group using the `AdapterPlugin` descriptor.
- Core checks `contract_version` compatibility (same major, minor <= core's) and refuses incompatible adapters with an upgrade hint.
- A fake in-repo test plugin is discovered and resolved by name; an unknown adapter name produces the naming-the-package parse error.
- A `sys.modules` test asserts importing a plugin module does not import any ML framework.

### S2-07 - Snapshot pinning (local Parquet)

**Refs:** PRD FR-COMP-02 (snapshot part), FR-ADPT-04 (snapshot part); TSD §8.3; ADR-11.

DONE when:

- `DataAdapter.snapshot_id` for local Parquet hashes the sorted `(relative_path, size, mtime_ns)` file listing.
- `--deep-snapshot` switches to content hashing.
- An explicit `snapshot:` pin in a dataset spec is honored, with a warning when it is no longer current.
- Compile parallelizes snapshot calls across sources (TSD §20).

### S2-08 - Hashing: config_hash, input_hash, env_digest

**Refs:** PRD FR-COMP-02, FR-COMP-03; TSD §8.4; ADR-4, ADR-5.

DONE when:

- `config_hash` covers canonical JSON of the rendered spec plus hooks-file bytes; `description`, `owner`, `tags`, resolved windows, anchor, and all profile values are excluded.
- `input_hash` composes `config_hash + snapshot_id + sorted upstream input_hashes` in topological order.
- `env_digest` covers Python version, mbt packages, and adapters' `fingerprint_packages`.
- Mutation tests prove: hyperparameter edit flips the model only; dataset filter edit flips dataset and downstream model; target switch (dev → prod) flips nothing (ADR-5).

### S2-09 - `mbt compile` and the manifest writer

**Refs:** PRD FR-COMP-01, FR-COMP-04; NFR-02, NFR-03, NFR-08; TSD §8.1, §8.5.

DONE when:

- `mbt compile` runs the full pipeline of TSD §8.1 and writes `target/manifest.json` matching the TSD §8.5 format, including `manifest_schema_version`, git metadata, and `adapter_versions`.
- Two compiles of the same project at the same `--anchor` produce byte-identical manifests (PRD FR-COMP-04).
- A golden-file test compiles `examples/churn_demo` against a checked-in manifest with volatile metadata normalized (TSD §21).
- `manifest_hash` is computed over the canonical manifest with volatile metadata blanked.
- A CI perf test compiles a 50-resource project in < 10 s (PRD NFR-03).

### S2-10 - `mbt show`

**Refs:** PRD FR-PARSE-05 (`show`); TSD §3, §7.

DONE when:

- `mbt show <name>` prints one resource's compile-rendered config with `--output yaml|json`.
- Unknown resource names list close matches.

**Sprint 2 exit criteria:** `mbt compile` on the demo project emits a deterministic, golden-tested manifest with pinned anchor, snapshots, hashes, and env digest; contracts and plugin discovery are in place.

---

## 5. Sprint 3 (W5-W6) - Execution vertical slice

**Maps to:** PRD §9 Phase 1, second half.
**Goal:** `mbt run` trains a real churn model end-to-end from YAML on local Parquet, tracked in MLflow; `mbt run --manifest` reproduces metrics exactly.

### S3-01 - Local DataAdapter: dataset materialization and cache

**Refs:** PRD FR-ADPT-04; TSD §10.4, §13.2.

DONE when:

- `build_dataset` materializes one Parquet file per split under `target/datasets/<name>/<materialization_key>/` via a single DuckDB query.
- The materialization key is `sha256(input_hash + canonical_json(resolved.windows))` and cache hits skip rebuilds (TSD §10.4).
- The handle's `snapshot_id` is verified against the manifest pin; mismatch is an error (data moved under a pinned manifest).
- `from_locator` reopens a materialization; `read` returns Arrow; `profile()` computes counts/schema/label balance once and caches JSON alongside the splits.

### S3-02 - Coordinator/job split and local ComputeAdapter

**Refs:** PRD FR-RUN-10; TSD §10.3, §13.4; ADR-3.

DONE when:

- The job entrypoint `python -m mbt.execute.job <job.json>` consumes a fully serializable `TrainingJob` and returns a `JobResult` via a result file, not stdout.
- The local ComputeAdapter implements `submit`/`wait` by spawning the subprocess; a crashing job yields a structured error result, not a coordinator crash.
- The payload carries env-var names only; the subprocess re-resolves values from its own environment (TSD §18).
- The responsibility split of TSD §10.3 is respected: no registry or gate logic in the job, no framework imports in the coordinator.

### S3-03 - mbt-xgboost training adapter

**Refs:** PRD FR-ADPT-03, FR-RES-10; TSD §13.1.

DONE when:

- The `mbt-xgboost` package ships a plugin descriptor plus Pydantic param models with `extra="forbid"`; `import xgboost` happens only inside adapter methods.
- `train` uses `seed=spec.seed`, fixed `nthread`, `tree_method="hist"`; two runs with the same seed and data produce identical metrics (determinism tier: exact).
- Built-in metrics `roc_auc`, `pr_auc`, `logloss`, `ece`, `recall_at_precision_*`, `precision_at_recall_*` are computed, including slice metrics by group-by.
- `scale_pos_weight: "{{ auto }}"` resolves from the dataset profile at run time; the resolved value is logged and lands in run results (PRD FR-RES-10).
- `export` writes native `.ubj` (ONNX behind the `mbt-xgboost[onnx]` extra); `load` reads an `ArtifactRef` back.

### S3-04 - mbt-mlflow tracking adapter

**Refs:** PRD FR-REG-01, FR-REG-04; TSD §13.3.

DONE when:

- `start_run`/`log`/`end_run`/`resume` work against a local MLflow (sqlite backend) in tests.
- Every training run is tagged with `mbt.config_hash`, `mbt.input_hash`, `mbt.manifest_hash`, `mbt.snapshot_id`, `mbt.git_commit`, `mbt.run_id`.
- Params, metrics, and artifacts are logged from the job; the coordinator can `resume` a run to attach tags.
- Canonical stage tokens map to MLflow stages (`staging → Staging`, etc.), with `use_aliases: true` supported on MLflow ≥ 2.9.

### S3-05 - `mbt run`: planner, sequential scheduler, model runner

**Refs:** PRD FR-RUN-01 (run semantics; `build` interleaving lands in S5-04); TSD §10.1, §10.2, §10.5.

DONE when:

- `mbt run` executes datasets then models in DAG order for the demo project (single thread; parallelism lands in S5-05).
- The model runner follows TSD §10.5: dataset handle → job assembly → submit/wait → run result (gates land in Sprint 5).
- Seed derivation is `spec.seed` for the adapter, `seed + 1` for tuning, `seed + 2` for validation carving (TSD §10.5).
- Failures produce structured node errors and exit 1.

### S3-06 - `--manifest` execution (reproducibility mechanism)

**Refs:** PRD FR-RUN-05, FR-RUN-11; G2; TSD §10.6.

DONE when:

- `mbt run --manifest <path>` skips parse/compile and executes the stored manifest verbatim: same anchor, same resolved windows, same snapshots, same hashes.
- mbt warns when current project files disagree with the manifest but does not re-render.
- An E2E test proves: build → rerun via `--manifest` → identical metrics for the XGBoost adapter (exact tier).

### S3-07 - Typed events and JSON logging

**Refs:** PRD FR-CLI-03 (JSON), NFR-06; TSD §16.

DONE when:

- Typed Pydantic events (`ParseStarted`, `NodeStarted`, `NodeFinished`, `GateEvaluated`, `AutoResolved`, `ArtifactRegistered`, `RunFinished`, ...) exist with two sinks: Rich console and JSON-lines via `--log-format json`.
- Each JSON event line carries `run_id`, `unique_id` (where applicable), and timestamps.
- Job subprocesses emit the same event stream on stdout and the coordinator forwards it.

### S3-08 - CLI hardening: non-interactive safety and error taxonomy

**Refs:** PRD FR-CLI-01; TSD §17.

DONE when:

- All shipped commands are verified prompt-free with stdin detached (CI test).
- The exception taxonomy (`MbtError` → `ConfigError`, `CompilationError`, `AdapterError`, `GateFailure`, `StateError`) is in place and adapter exceptions are wrapped with node context.
- Hard errors exit 1 uniformly (exit 2 semantics land in S5-07).

### S3-09 - churn_demo example and E2E reproducibility test

**Refs:** PRD G2, NFR-08; TSD §2, §21.

DONE when:

- `examples/churn_demo` contains a realistic source + dataset + model spec with committed sample Parquet data, used by golden and E2E tests.
- A CI E2E job runs: `mbt init` template → local MLflow (sqlite) → `mbt run` → assert tracked run exists → `mbt run --manifest` → assert identical metrics (G2).

**Sprint 3 exit criteria (PRD §9 Phase 1 DoD):** `mbt run` trains a real churn model end-to-end from YAML on local Parquet, tracked in MLflow; rerunning via `--manifest` yields identical metrics.

---

## 6. Sprint 4 (W7-W8) - DAG, datasets, tests, hooks

**Maps to:** PRD §9 Phase 2, first half.
**Goal:** multi-node projects with full selector support; dataset construction with split policies and checks; hooks and custom metrics.

### S4-01 - Dataset split policies and cross-resource validation

**Refs:** PRD FR-RES-02, FR-RES-09; TSD §5.5, §5.6.

DONE when:

- Temporal is the default split strategy; random requires explicit opt-in and a `seed`; `stratify_by` applies to random only.
- `model.target` must equal the dataset's `label.column`; mismatch is a parse error (TSD §5.6).
- `evaluation.protocol.split` must equal the dataset's `split.strategy`; mismatch is a parse error (PRD FR-RES-09).
- `evaluation.protocol.test_window`, when set, must resolve to a sub-range of the dataset's test window.
- Every `gates[].metric` and `tuning.objective.metric` must appear in `evaluation.metrics`.

### S4-02 - Dataset construction: filters, sampling, split assignment

**Refs:** PRD FR-ADPT-04 (completion), FR-RES-02; TSD §13.2.

DONE when:

- `filters` (SQL WHERE fragments, ANDed) execute in the DuckDB build.
- Target sampling honors the `sample_fraction` var via hash-sampling on a stable key, deterministically across runs.
- Split assignment implements resolved temporal windows on `time_column` and seeded hash splits for random.
- Golden tests cover both strategies on the churn_demo data, including validation splits.

### S4-03 - Selector grammar and evaluation

**Refs:** PRD FR-DAG-02 (`state:` lands in S6-01), FR-DAG-03, FR-DAG-04; TSD §9.2; NFR-08.

DONE when:

- The grammar of TSD §9.2 parses: name globs, `+` upstream/downstream with optional depth, `tag:`, `resource_type:`, comma intersection, space union.
- `--exclude` evaluates the same grammar and subtracts from the selection.
- Property-based tests over random DAGs verify selector algebra invariants (e.g. `a,b ⊆ a`).
- `--select`/`--exclude` behave identically on `run`, `test`, `ls` (PRD FR-CLI-04 groundwork).

### S4-04 - Full DAG semantics

**Refs:** PRD FR-DAG-01; TSD §9.1.

DONE when:

- `ref()` edges build the dataset → model DAG keyed by unique_id (model → model edges are rejected in v0 with a message pointing at v1 ensembles, PRD FR-DAG-05).
- Cycle errors report the full cycle path.
- The DAG in the manifest (`depends_on`) matches the parsed graph exactly.

### S4-05 - Upstream closure planning

**Refs:** PRD FR-RUN-12; TSD §10.1; ADR-13.

DONE when:

- Every dataset required by a selected model joins the execution plan even if unselected; selection governs which models train.
- Auto-materialization is cache-aware: a warm cache skips the dataset build.
- An E2E test selects only the model on a cold cache and the run succeeds.
- Missing prerequisites that cannot be auto-satisfied fail at plan time with guidance, before any training cost.

### S4-06 - Built-in dataset checks

**Refs:** PRD FR-TEST-05; TSD §11.1.

DONE when:

- `no_future_columns`, `label_leakage_scan`, `class_balance_report` (report-only), `schema`, and `not_null` run as part of dataset builds.
- Each check has positive and negative unit tests; failures set node status `test_failed`.
- Check parameters follow the `CheckSpec` shape from TSD §5.5.

### S4-07 - Python data tests

**Refs:** PRD FR-RES-05; TSD §5.7, §11.1.

DONE when:

- Files in `tests/` exposing `def test_*(dataset, spec) -> TestResult` are discovered and run against materialized datasets.
- Tests bind to resources via the `# mbt: select=<selector>` header or an explicit `tests:` key.
- Failures set `test_failed` and will map to exit code 2 (wired in S5-07).

### S4-08 - hooks.py: custom transforms and metrics

**Refs:** PRD FR-RES-07; TSD §5.8.

DONE when:

- A sibling `models/<name>.py` exposing `transform_features` and/or `custom_metrics` is auto-detected and executed inside the training job only.
- `transform_features` applies per split after read; `features.include/exclude` then applies to the post-hook column set.
- Hook file bytes are hashed into `config_hash`; a hash test proves a hook edit changes the model's `input_hash` (state:modified groundwork).
- `HookContext` provides spec, profile, split, and logger as per TSD §12.1.

### S4-09 - metrics.yml and metric resolution

**Refs:** PRD FR-RES-04; TSD §5.7.

DONE when:

- `MetricSpec` resources parse from `metrics.yml` with `kind: builtin|hook`, params, and `greater_is_better`.
- Sugar names like `recall_at_precision_0.9` parse into `(base_metric, params)` when not explicitly declared.
- Resolution order is: explicit `metrics.yml` > sugar-parsed builtin > adapter builtin > hook metric; unknown names are a parse error listing candidates.

### S4-10 - Jinja macros

**Refs:** PRD FR-COMP-06; TSD §6.

DONE when:

- Macros from `macros/*.jinja` load into the shared environment and are usable in any spec.
- A macro used by the demo project renders correctly in both capture and resolve phases.

**Sprint 4 exit criteria:** a multi-model demo project parses and runs with selectors and `--exclude`; dataset checks and hooks execute; metric resolution is in place.

---

## 7. Sprint 5 (W9-W10) - Gates, registry, run semantics

**Maps to:** PRD §9 Phase 2, second half.
**Goal:** the full quality loop: gates decide registration, failures produce exit 2, results are machine-readable, execution is parallel.

### S5-01 - Gate evaluation engine

**Refs:** PRD FR-TEST-02, FR-TEST-06; TSD §11.2; ADR-9, ADR-10.

DONE when:

- Threshold gates are direction-aware via `greater_is_better`; champion gates require `challenger - champion >= min_delta` (direction-adjusted).
- Gate logic lives in `mbt.quality` as pure comparisons over `MetricResults`, unit-testable with zero ML dependencies.
- Missing champion: gate passes with an explicit WARN event and `champion_version: null` (bootstrap case).
- Champion exists but cannot load: gate errors; no silent pass (ADR-10).
- `GateSpec` validation enforces exactly one of `threshold`/`compare_to`, and `min_delta` only with `compare_to`.

### S5-02 - Champion resolution and in-job champion evaluation

**Refs:** PRD FR-TEST-02; TSD §10.5, §13.3; ADR-9.

DONE when:

- The coordinator resolves the champion via `RegistryAdapter.get_champion` before job submission and passes its `ArtifactRef` in the `TrainingJob`.
- The job loads the champion and evaluates it on the identical pinned test split with the same metric code as the challenger.
- `champion_metrics` returns in the `JobResult` and feeds gate evaluation.

### S5-03 - Registration on gate pass

**Refs:** PRD FR-REG-02, FR-REG-05, FR-TEST-03; TSD §10.5, §13.3.

DONE when:

- On gate pass, the artifact is registered under the declared name and transitioned to `stage_on_pass` (default staging).
- Registration metadata links `config_hash`, `input_hash`, `manifest_hash`, `snapshot_id`, `git_commit`, and `tracking_run_id` (PRD FR-REG-05).
- On gate fail: node status `gate_failed`, no registration, gate results still recorded as tracking tags, exit code 2 (PRD FR-TEST-03).
- Gate results and the registered version are attached to the tracking run via `resume`.

### S5-04 - `mbt build` interleaving and standalone `mbt test`

**Refs:** PRD FR-RUN-01, FR-TEST-01; TSD §10.2, §11.3.

DONE when:

- `mbt build` runs each node's tests/gates immediately after the node and before its children are released; a test or gate failure fails the node for scheduling purposes.
- Standalone `mbt test` runs data tests against existing-or-built materializations and model gate tests via the evaluate machinery against the latest registered version; training is never a side effect.
- A model with no registered version is `skipped` with a warning under `mbt test`.
- `--select` works for both commands.

### S5-05 - Parallel scheduler, failure isolation, fail-fast

**Refs:** PRD FR-RUN-02, FR-RUN-03; TSD §10.2.

DONE when:

- `--threads N` executes independent DAG branches in parallel via `ThreadPoolExecutor`; training remains in subprocesses.
- A node failure marks transitive downstream `skipped` while independent branches continue; `--fail-fast` cancels pending work.
- Scheduler semantics (start-when-parents-succeed, skip propagation, fail-fast) are tested with fake contract-conformant adapters and no ML dependencies (TSD §21).

### S5-06 - run_results.json and slice reporting

**Refs:** PRD FR-RUN-04, FR-TEST-04 (reporting; slice gates deferred to §11 backlog); TSD §10.8.

DONE when:

- Every run/build/test writes `target/run_results.json` matching TSD §10.8: schema version, per-node status, timings, metrics, gates, slices, resolved auto values, artifact and registration references, tracking run id.
- Per-slice metrics for declared `slices` columns appear in results and tracking.
- Statuses are exactly `success | error | skipped | gate_failed | test_failed`.

### S5-07 - Exit codes and uniform flags

**Refs:** PRD FR-CLI-02, FR-CLI-04; TSD §17.

DONE when:

- Exit precedence over the whole invocation is: any hard error → 1; else any `test_failed`/`gate_failed` → 2; else 0.
- A parametrized CLI test matrix asserts `--target`, `--vars`, `--select`, `--exclude`, `--threads`, `--state`, `--manifest` behave identically wherever they appear.

### S5-08 - Multi-model demo and Phase 2 E2E

**Refs:** PRD §9 Phase 2 DoD; NFR-03; TSD §21.

DONE when:

- The demo project has ≥ 2 models sharing a dataset plus one independent branch, running in DAG order with `--threads 2`.
- An E2E test proves a failing gate blocks registration and exits 2 while an independent branch still completes.
- Per-node execution overhead (excluding training) measures < 2 s in the E2E run (PRD NFR-03).

**Sprint 5 exit criteria (PRD §9 Phase 2 DoD):** a multi-model project with dependencies runs; a failing gate blocks registration with exit code 2; `run_results.json` is emitted.

---

## 8. Sprint 6 (W11-W12) - State, diff, CI workflows

**Maps to:** PRD §9 Phase 3, weeks 1-2 (Phase 3 completes in Sprint 7).
**Goal:** state-aware selection against a reference manifest and the CI workflows that use it.

### S6-01 - `state:new` / `state:modified` selection

**Refs:** PRD FR-STATE-01, FR-DAG-02 (`state:` methods); TSD §9.3; ADR-4, ADR-7.

DONE when:

- `state:modified` selects nodes whose `input_hash` differs from the reference manifest; `state:new` selects unique_ids absent from it.
- `--state <path-or-URI>` reads `file://` and `s3://` references.
- Anchor drift over an unchanged snapshot selects nothing; spec, hooks, snapshot, and upstream changes each select the expected nodes (mutation test matrix, TSD §21).
- `env_digest` changes do not mark nodes modified unless `--state-include-env` is passed (ADR-7).

### S6-02 - `mbt state diff`

**Refs:** PRD FR-STATE-02; TSD §14.1.

DONE when:

- Output lists nodes added/removed/modified, each annotated with which component changed: config, hooks, snapshot, upstream.
- The env_digest delta is reported prominently even though it does not modify nodes by default.
- Both Rich table and `--output json` render; the JSON shape is stable and consumed by the S7-04 PR bot.

### S6-03 - Manifest storage convention and schema compatibility

**Refs:** PRD FR-STATE-03, NFR-05; TSD §14.2, §19.

DONE when:

- The S3/CI-artifact layout (`.../manifests/<git_sha>.json` + `latest.json`) is documented and used by the reference workflows.
- Core reads manifest schema N and N-1 and refuses newer schemas with a clear message (TSD §19).
- An unreadable or invalid `--state` reference is a hard error (exit 1), never a silent full retrain.

### S6-04 - Reference workflows: PR check and prod build

**Refs:** PRD FR-STATE-04; TSD §14.3.

DONE when:

- `pr_check.yml` runs parse → compile (dev) → `state diff --output json` vs the latest prod manifest → `build --select state:modified+`.
- `prod_build.yml` builds `state:modified+` on merge to main and uploads the manifest as the new `latest.json` on success.
- Both workflows run green on the demo repository.

### S6-05 - Scheduled retraining template

**Refs:** PRD FR-STATE-06; TSD §14.3.

DONE when:

- A cron workflow template runs `mbt build --select tag:weekly` against prod.
- The quickstart documents tag-based scheduled retraining; no orchestrator concept is introduced.

### S6-06 - Exposures

**Refs:** PRD FR-RES-06; TSD §5.7.

DONE when:

- `ExposureSpec` resources parse from `exposures.yml` with type, `depends_on` refs, owner, and URL.
- Exposures appear in the manifest and in the DAG for impact analysis (`mbt ls --select +my_exposure` style downstream queries work).

**Sprint 6 exit criteria:** on the demo repo, a PR that changes one model retrains only that model's subgraph in CI, driven by `state:modified+` against the stored prod manifest.

---

## 9. Sprint 7 (W13-W14) - Promotion, evaluate, PR bot; adapter-base extraction

**Maps to:** PRD §9 Phase 3 week 3 + Phase 4 week 1.
**Goal:** close the GitOps loop (promotion, PR comment) and start adapter hardening (contract extraction, compliance suite).

### S7-01 - `mbt promote`

**Refs:** PRD FR-REG-03; TSD §14.4.

DONE when:

- `mbt promote --model X --to production [--version N]` resolves the version (explicit, else latest in `stage_on_pass`), verifies gate-pass tags recorded at registration, and transitions the stage.
- Promotion refuses when gates were not recorded as passed; `--force` overrides with a loud event.
- `--from-file promotions.yml` processes a reviewed list of `{model, version, to}` entries (GitOps path).

### S7-02 - Promotion workflow

**Refs:** PRD FR-STATE-04 (promotion workflow); TSD §14.3.

DONE when:

- `promote.yml` triggers on a reviewed `promotions.yml` change or `workflow_dispatch` with environment approval and runs `mbt promote --from-file`.
- The workflow runs green on the demo repository, and the registry reflects the transition.

### S7-03 - `mbt evaluate`

**Refs:** PRD FR-RUN-07; TSD §10.6.

DONE when:

- `mbt evaluate --model X [--version N | --stage S] [--gates]` compiles fresh, builds the model's dataset, loads the registered artifact, and evaluates on the test split without retraining.
- Version resolution: explicit `--version`, else latest in `--stage`, else the model's `stage_on_pass`.
- Results write to run_results with `"command": "evaluate"`; `--gates` applies gate logic to the fresh metrics.

### S7-04 - PR comment bot

**Refs:** PRD FR-STATE-05, G3; TSD §14.3.

DONE when:

- The PR check posts a comment with: metrics vs champion, gate pass/fail table, retrained node list, and a cost estimate (Σ node execution_time × runner rate).
- Content is sourced from `run_results.json` and `state diff --output json` only (no re-computation in workflow scripts).
- The comment updates in place on subsequent pushes instead of stacking.

### S7-05 - Phase 3 E2E: the full loop

**Refs:** PRD §9 Phase 3 DoD; FR-RUN-12; G3.

DONE when:

- On the demo repo: a model-change PR retrains only that model (with auto-materialized dataset), the PR comment shows deltas, merge builds prod and publishes the manifest, and a promotion PR transitions the model to production.
- The E2E is scripted and repeatable, not a one-off manual demo.

### S7-06 - Extract `mbt-adapter-base`

**Refs:** PRD FR-ADPT-01 (Phase 4 extraction); TSD §2, §19.

DONE when:

- Contracts and interchange types move to the `mbt-adapter-base` package, versioned independently (SemVer).
- `mbt.contracts` re-exports everything for compatibility; existing adapters build unchanged.
- Adapter packages depend on `mbt-adapter-base` instead of `mbt-core`.

### S7-07 - Compliance suite v1

**Refs:** PRD FR-ADPT-05; TSD §12.4.

DONE when:

- `mbt_adapter_base.compliance` ships pytest base classes with tiny committed datasets, parametrized over `supported_tasks`.
- Asserted: seed determinism within the declared tier, `resolve_auto` idempotence with no leftover sentinels, param model rejection of unknown params, train → export → load → evaluate round-trip, import hygiene via `sys.modules`.
- mbt-xgboost passes the suite in CI.

**Sprint 7 exit criteria (PRD §9 Phase 3 DoD):** the full PR → CI → registry → promotion loop runs on the demo repo; the PR comment shows metric deltas; model-only PRs retrain just the model.

---

## 10. Sprint 8 (W15-W16) - Adapter hardening, tuning, docs

**Maps to:** PRD §9 Phase 4, weeks 2-3.
**Goal:** the extensibility proof (LightGBM), reproducible tuning, and generated documentation.

### S8-01 - Determinism tiers and tolerance-aware gates

**Refs:** PRD FR-ADPT-07, NFR-01; TSD §11.2, §12.4.

DONE when:

- Every adapter declares a `DeterminismTier` (exact or per-metric tolerances) and documents it.
- Threshold-gate comparisons widen by the tolerance in the model's favor only across rerun comparisons, never for champion deltas (TSD §11.2).
- The compliance suite asserts reruns land within the declared tier.

### S8-02 - mbt-lightgbm: the extensibility proof

**Refs:** PRD FR-ADPT-06, G4; TSD §2, §12.

DONE when:

- The LightGBM adapter is built against published `mbt-adapter-base` contracts and the compliance suite only, with zero changes to mbt-core (verified by an unchanged mbt-core SHA during development).
- It passes the full compliance suite for binary classification.
- A demo model switches adapters by changing only `adapter: lightgbm` plus hyperparameters, and the build passes.
- Friction encountered is written up and folded back into contract docs (G4 evidence).

### S8-03 - Hyperparameter tuning (mbt-optuna)

**Refs:** PRD FR-TUNE-01, FR-TUNE-02, FR-TUNE-03, FR-TUNE-04; TSD §10.5, §13.5; ADR-8.

DONE when:

- The `tuning` block (engine, `n_trials`, typed search space, objective) drives an in-job Optuna TPE loop with `seed + 1`; same seed reproduces the same best params.
- Trials train on train and evaluate on validation; a test proves the test split is never read during tuning (ADR-8).
- Implicit validation carves (temporal: last 20% by time; random: seeded 20%) are reabsorbed in the final fit; explicitly declared validation stays held out.
- `n_trials` is capped by the target's `max_tuning_trials` var (dev 5 / prod 50 in the demo); trial history logs as nested tracking runs.

### S8-04 - Nondeterminism warnings

**Refs:** PRD FR-RUN-06; TSD §13.1.

DONE when:

- mbt warns when a known nondeterminism source is detected (e.g. adapter setting that breaks its declared tier, missing seed passthrough).
- At least the XGBoost non-hist tree method and LightGBM threading cases are covered with tests.

### S8-05 - `mbt run-operation`

**Refs:** PRD FR-RUN-08; TSD §10.7.

DONE when:

- `mbt run-operation <macro> --args '<dict>'` renders the macro with the full compile context and prints the result.
- Adapter-invoking operations are explicitly rejected with a message (out of scope until dogfooding demands them).

### S8-06 - `packages.yml` and `mbt deps`

**Refs:** PRD FR-PROJ-04; TSD §3.

DONE when:

- `packages.yml` pins adapter packages; `mbt deps` installs them into the active environment.
- Version conflicts and contract incompatibilities surface as actionable errors (TSD §12.3).

### S8-07 - Docs generation

**Refs:** PRD FR-DOCS-01, FR-DOCS-02, FR-DOCS-03; TSD §15.

DONE when:

- `mbt docs generate` renders a static site into `target/docs/` from manifest + latest run_results: DAG lineage view plus one model card per model (description, owner, tags, data window + snapshot, features + exclusions, hyperparameters including resolved AUTO values, metrics + slice table, gate history, registry links).
- Exposures appear in the lineage view for impact analysis.
- The site is self-contained (inline CSS/JS, no CDN) and `mbt docs serve` serves it locally.

### S8-08 - OpenTelemetry spans

**Refs:** PRD NFR-06; TSD §16.

DONE when:

- Optional spans per command and per node execution activate via standard `OTEL_*` env vars and no-op otherwise.
- A test asserts span emission with an in-memory exporter and zero overhead when disabled.

**Sprint 8 exit criteria (PRD §9 Phase 4 DoD):** the LightGBM adapter is built by "someone else" using only public contracts + compliance suite; Optuna tuning works within caps; the docs site generates.

---

## 11. Sprint 9 (W17-W18) - v0.1 release

**Maps to:** PRD §9 Phase 5.
**Goal:** ship v0.1 to PyPI with an external-quality onboarding path and prove the goals with real models.

### S9-01 - Full `mbt init` template

**Refs:** PRD FR-PROJ-01 (complete); TSD §14.3, §18.

DONE when:

- The scaffold now includes the pre-commit config, the three reference CI workflows (PR check, prod build, promote), and CODEOWNERS on `models/`.
- `profiles.yml` is written to `~/.mbt/` by default with a project-local template gitignored.
- A fresh `mbt init` project passes `mbt parse` and the pre-commit hooks out of the box.

### S9-02 - Packaging and PyPI release

**Refs:** PRD NFR-05, NFR-10; TSD §19.

DONE when:

- All packages (mbt-core, mbt-adapter-base, mbt-xgboost, mbt-mlflow, mbt-optuna, mbt-lightgbm) publish to PyPI via release-please with conventional commits, SemVer, and pinned inter-package constraints.
- The `mbt` name decision from S1-09 is executed (or the fallback applied consistently everywhere).
- A clean-machine `pip install mbt-core mbt-xgboost mbt-mlflow` smoke test passes the quickstart's first steps.

### S9-03 - Quickstart, README, docs site

**Refs:** PRD G5, NFR-09; TSD §21.

DONE when:

- The mkdocs-material site covers quickstart, spec reference (generated from JSON Schema), CLI reference, adapter authoring guide, and ADRs.
- The README is external-quality: what/why, 5-minute example, badges, links.
- The DS design partner completes `mbt init` → trained, registered model in under 1 hour following only the quickstart, with timings recorded (G5 evidence).

### S9-04 - Migrate two real models

**Refs:** PRD G1, G2, G3; PRD §9 Phase 5.

DONE when:

- Two internal models are migrated from notebooks to mbt specs with the DS design partner, running the full PR → CI → registry loop (G1).
- Each migration's PR demonstrates state-aware retraining and a PR comment with cost (G3 evidence).
- A `--manifest` rerun of each migrated model reproduces metrics within its adapter's tier (G2 evidence).

### S9-05 - Release hardening and polish

**Refs:** PRD FR-CLI-03 (Rich polish), NFR-03; TSD §20.

DONE when:

- Rich console output (tables, progress) is reviewed and polished across commands.
- Performance budgets re-verified on the release candidate: parse < 2 s and compile < 10 s at 50 resources; per-node overhead < 2 s.
- All dogfooding-blocking bugs from S9-04 are fixed or explicitly waived in the release notes.

### S9-06 - v0.1 release and goal checklist

**Refs:** PRD §3 (G1-G5), §9 Phase 5 DoD.

DONE when:

- v0.1 is tagged and published; release notes summarize scope and known limitations.
- The G1-G5 goal table from PRD §3 is filled in with measured evidence and linked artifacts.
- v1 candidate feedback from dogfooding is filed as issues against the §11 backlog.

**Sprint 9 exit criteria (PRD §9 Phase 5 DoD):** PyPI release; external-quality README + tutorial; 2 real models migrated (G1); quickstart under 1 hour (G5).

---

## 12. Deferred and v1 backlog (not scheduled in v0.1 sprints)

These items are tracked so their references stay visible, but they are explicitly out of sprint scope (PRD §7.14; TSD §22 shows the architecture accommodates them).

| Item | Refs |
|---|---|
| Slice-level gates (reporting shipped in S5-06) | PRD FR-TEST-04 (Could); TSD §5.6 |
| Regression / multiclass task support in mbt-xgboost (stretch) | PRD FR-ADPT-03; TSD §13.1 |
| Iceberg source support via `mbt-core[iceberg]` (optional) | PRD §10; TSD §8.3, §13.2 |
| K8s / Ray ComputeAdapters | PRD FR-V1-01; TSD §22 |
| sklearn / PyTorch adapters | PRD FR-V1-02; TSD §22 |
| Survival / ranking task schemas via plugins | PRD FR-V1-03; TSD §12.3, §22 |
| Feast DataAdapter | PRD FR-V1-04; TSD §22 |
| Ensembles / stacking (`ref()` model inputs) | PRD FR-DAG-05, FR-V1-05; TSD §22 |
| `mbt score` batch inference | PRD FR-V1-06; TSD §22 |
| Airflow provider | PRD FR-V1-07; TSD §22 |
| `ensemble` / `calibrated` / `onnx` materializations | PRD FR-V1-08; TSD §5.1 |

## 13. Traceability matrix

### 13.1 Functional requirements → tasks

| PRD ID | Tasks |
|---|---|
| FR-PROJ-01 | S1-07, S9-01 |
| FR-PROJ-02 | S1-02 |
| FR-PROJ-03 | S2-01 |
| FR-PROJ-04 | S8-06 |
| FR-PROJ-05 | S2-03 |
| FR-PROJ-06 | S2-02 |
| FR-RES-01 | S1-02 |
| FR-RES-02 | S1-02 (schema), S4-01, S4-02 |
| FR-RES-03 | S1-02 |
| FR-RES-04 | S4-09 |
| FR-RES-05 | S4-07 |
| FR-RES-06 | S6-06 |
| FR-RES-07 | S4-08 |
| FR-RES-08 | S1-03 |
| FR-RES-09 | S4-01 |
| FR-RES-10 | S3-03 |
| FR-PARSE-01 | S1-04, S1-05, S1-06 |
| FR-PARSE-02 | S1-05 |
| FR-PARSE-03 | S1-08 |
| FR-PARSE-04 | S1-02 |
| FR-PARSE-05 | S1-06 (`ls`), S2-10 (`show`) |
| FR-COMP-01 | S2-02, S2-09 |
| FR-COMP-02 | S2-07, S2-08 |
| FR-COMP-03 | S2-08 |
| FR-COMP-04 | S2-09 |
| FR-COMP-05 | S2-04 |
| FR-COMP-06 | S4-10 |
| FR-DAG-01 | S1-04 (initial), S4-04 |
| FR-DAG-02 | S4-03, S6-01 (`state:`) |
| FR-DAG-03 | S4-03 |
| FR-DAG-04 | S4-03 |
| FR-DAG-05 | §12 backlog (v1) |
| FR-RUN-01 | S3-05 (`run`), S5-04 (`build`) |
| FR-RUN-02 | S5-05 |
| FR-RUN-03 | S5-05 |
| FR-RUN-04 | S5-06 |
| FR-RUN-05 | S3-06, S3-09 |
| FR-RUN-06 | S8-04 |
| FR-RUN-07 | S7-03 |
| FR-RUN-08 | S8-05 |
| FR-RUN-09 | S1-06 |
| FR-RUN-10 | S3-02 |
| FR-RUN-11 | S3-06 |
| FR-RUN-12 | S4-05, S7-05 |
| FR-TEST-01 | S4-06, S4-07, S5-04 |
| FR-TEST-02 | S5-01, S5-02 |
| FR-TEST-03 | S5-03 |
| FR-TEST-04 | S5-06 (reporting); slice gates in §12 backlog |
| FR-TEST-05 | S4-06 |
| FR-TEST-06 | S5-01 |
| FR-REG-01 | S3-04 |
| FR-REG-02 | S5-03 |
| FR-REG-03 | S7-01 |
| FR-REG-04 | S3-04 |
| FR-REG-05 | S5-03 |
| FR-TUNE-01..04 | S8-03 |
| FR-STATE-01 | S6-01 |
| FR-STATE-02 | S6-02 |
| FR-STATE-03 | S6-03 |
| FR-STATE-04 | S6-04, S7-02 |
| FR-STATE-05 | S7-04 |
| FR-STATE-06 | S6-05 |
| FR-DOCS-01 | S8-07 |
| FR-DOCS-02 | S8-07 |
| FR-DOCS-03 | S8-07 |
| FR-ADPT-01 | S2-05, S7-06 |
| FR-ADPT-02 | S2-06 |
| FR-ADPT-03 | S3-03 |
| FR-ADPT-04 | S2-07, S3-01, S4-02 |
| FR-ADPT-05 | S7-07, S8-01 |
| FR-ADPT-06 | S8-02 |
| FR-ADPT-07 | S8-01 |
| FR-CLI-01 | S1-06 (basic), S3-08 |
| FR-CLI-02 | S5-07 |
| FR-CLI-03 | S3-07 (JSON), S9-05 (Rich polish) |
| FR-CLI-04 | S4-03, S5-07 |
| FR-V1-01..08 | §12 backlog |

### 13.2 Non-functional requirements → tasks

| PRD ID | Tasks |
|---|---|
| NFR-01 (reproducibility) | S3-06, S3-09, S8-01 |
| NFR-02 (deterministic compile) | S2-09 |
| NFR-03 (performance) | S1-09 (parse), S2-09 (compile), S5-08 (node overhead), S9-05 (re-verify) |
| NFR-04 (code quality) | S1-01, enforced by §2 global DoD |
| NFR-05 (compatibility) | S6-03 (manifest N-1), S7-06 (contract versioning), S9-02 (SemVer release) |
| NFR-06 (observability) | S3-07, S5-06, S8-08 |
| NFR-07 (security) | S2-03, S3-02 (env-var names only) |
| NFR-08 (testability) | S1-01, S2-09 (golden), S4-03 (property), S6-01 (mutation), S7-07 (compliance), S3-09/S5-08/S7-05 (E2E) |
| NFR-09 (docs) | S1-09 (ADRs), S8-07, S9-03 |
| NFR-10 (packaging) | S1-01, S9-02 |

### 13.3 Success goals → evidence

| Goal (PRD §3) | Evidenced by |
|---|---|
| G1 (adoption: ≥ 2 migrated models) | S9-04 |
| G2 (reproducibility via `--manifest`) | S3-06, S3-09, S9-04 |
| G3 (economy: modified-subgraph CI) | S6-01, S7-04, S7-05, S9-04 |
| G4 (extensibility: LightGBM, zero core changes) | S7-06, S7-07, S8-02 |
| G5 (time-to-first-model < 1 hour) | S1-07, S9-01, S9-03 |
