# mbt - Technical Specification Document

**Product:** mbt (Model Build Tool)
**Version:** 0.1 · **Status:** implemented in v0.1 (historical design document; ADRs now live in `docs/adr/`, section numbers remain cited from code) · **Last updated:** 2026-07-06
**Related documents:** [PLAN.md](PLAN.md) (vision, rationale) · [PRD.md](PRD.md) (requirements)

This document is implementation-ready: an engineer should be able to start coding from it.
Requirement IDs (`FR-*`, `NFR-*`) refer to PRD.md.
Where this document makes a decision the plan left open, the decision is recorded in §23 with rationale.

---

## 1. Architecture overview

```
                 ┌────────────────────────────────────────────────────┐
                 │                mbt-core (coordinator)              │
                 │                                                    │
  YAML specs ──▶ │  parsing ──▶ DAG ──▶ compile ──▶ manifest.json     │
  profiles.yml ─▶│                        │                           │
                 │                        ▼                           │
                 │        execution engine (planner + scheduler)      │
                 │                        │                           │
                 │     ┌──────────┬───────┼────────┬───────────┐      │
                 └─────┼──────────┼───────┼────────┼───────────┼──────┘
                       ▼          ▼       ▼        ▼           ▼
                  DataAdapter Compute  Tracking Registry   TuningEngine
                   (local/    Adapter  Adapter  Adapter     (optuna)
                    duckdb)      │     (mlflow) (mlflow)
                                 ▼
                        training job (subprocess │ k8s/ray in v1)
                        runs TrainingAdapter (xgboost) + hooks
```

Core responsibilities:

1. **Parsing** turns YAML files into validated Pydantic resource objects and extracts `ref()`/`source()` edges.
2. **DAG** builds a `networkx.DiGraph` of nodes, detects cycles, and evaluates selectors.
3. **Compile** renders Jinja, applies profiles/vars, resolves time windows against a pinned anchor, pins data snapshots, computes hashes, and writes `manifest.json`.
4. **Execution** plans and schedules nodes, builds datasets in-process, and delegates each model build to a serialized **training job** executed by the ComputeAdapter.
5. **State** diffs manifests for `state:modified` selection and `mbt state diff`.
6. **Contracts** define the adapter Protocols and interchange types; core never imports an ML framework (NFR-04).

Two load-bearing seams:

- **Data interchange is Apache Arrow** (`pyarrow.Table`) across the DataAdapter → TrainingAdapter boundary; DuckDB, Polars, XGBoost, LightGBM, and sklearn all consume Arrow cheaply (ADR-1).
- **Coordinator vs job:** the coordinator process owns planning, registry access, and gate decisions; everything that needs an ML framework or the training data runs inside a serialized, restartable job (§10.3, ADR-3).
  Adapters compute metrics; core compares them.

## 2. Repository and package layout

Monorepo managed as a **uv workspace** with hatchling builds (NFR-10).

```
mbt/
├── pyproject.toml                  # uv workspace root, shared dev deps (ruff, mypy, pytest)
├── packages/
│   ├── mbt-core/
│   │   ├── pyproject.toml          # deps: typer, rich, pydantic>=2, jinja2, networkx,
│   │   │                           #       polars, duckdb, pyarrow  (no ML frameworks)
│   │   └── src/mbt/
│   │       ├── cli/                # Typer app; one module per command (§3)
│   │       ├── config/             # Pydantic schemas (§5)
│   │       │   └── tasks/          # task schemas + registry (§5.6)
│   │       ├── contracts/          # adapter Protocols + interchange types (§12)
│   │       │                       #   extracted to mbt-adapter-base in Phase 4
│   │       ├── parsing/            # discovery, YAML load, validation, ref extraction (§7)
│   │       ├── jinja/              # environment, context functions, macro loading (§6)
│   │       ├── dag/                # graph build, selector parsing/evaluation (§9)
│   │       ├── compile/            # manifest build, anchoring, hashing, snapshots (§8)
│   │       ├── execute/            # planner, scheduler, runners, job entrypoint (§10)
│   │       ├── quality/            # tests + gate evaluation, champion/challenger (§11)
│   │       ├── state/              # manifest diff (§14)
│   │       ├── adapters/
│   │       │   ├── registry.py     # entry-point discovery, resolution (§12.3)
│   │       │   └── local/          # built-in local Data + Compute adapters (§13.2, §13.4)
│   │       ├── docsgen/            # model cards + lineage site (§15)
│   │       ├── events/             # typed events, logging, OTel (§16)
│   │       ├── artifacts/          # manifest/run_results writers + schema versioning
│   │       └── exceptions.py       # error taxonomy (§17)
│   ├── mbt-adapter-base/           # Phase 4: contracts + compliance suite, versioned separately
│   ├── mbt-xgboost/                # TrainingAdapter (§13.1)
│   ├── mbt-mlflow/                 # TrackingAdapter + RegistryAdapter (§13.3)
│   ├── mbt-optuna/                 # TuningEngine (§13.5)
│   └── mbt-lightgbm/               # Phase 4 extensibility proof
├── examples/churn_demo/            # demo project used by E2E tests and docs
├── docs/                           # mkdocs-material site + ADRs (docs/adr/NNNN-*.md)
└── .github/workflows/              # mbt's own CI + the reference user templates
```

Until Phase 4, `contracts/` lives in mbt-core and adapter packages depend on `mbt-core`.
Phase 4 extracts it as `mbt-adapter-base` (re-exported from `mbt.contracts` for compatibility), so adapters pin against a small, stable package (FR-ADPT-01).

The local DataAdapter and local ComputeAdapter ship inside mbt-core: DuckDB/Polars/PyArrow are data dependencies, not ML frameworks, and a batteries-included core keeps time-to-first-model under an hour (G5, ADR-2).

## 3. CLI surface

All commands are Typer subcommands of `mbt`; all are non-interactive-safe (FR-CLI-01).

**Global flags:** `--project-dir` (default `.`), `--profiles-dir`, `--target`, `--vars '<yaml/json dict>'`, `--log-format text|json`, `--quiet`.

| Command | Purpose | Key flags | Spec |
|---|---|---|---|
| `mbt init <name>` | Scaffold a golden-path project | | §14.3 (CI templates), FR-PROJ-01 |
| `mbt deps` | Install adapter packages pinned in `packages.yml` | | FR-PROJ-04 |
| `mbt parse` | Validate configs, build DAG, no execution | `--write-json-schema` | §7 |
| `mbt compile` | Produce `target/manifest.json` | `--anchor <iso-ts>` | §8 |
| `mbt run` | Build datasets + train models in DAG order | `--select --exclude --threads --state --manifest --fail-fast` | §10 |
| `mbt test` | Data tests + model quality gates | `--select --exclude --state --manifest` | §11.3 |
| `mbt build` | run + test interleaved in DAG order | same as `run` | §10.2 |
| `mbt evaluate` | Re-evaluate an existing artifact on fresh data | `--model --version --stage --gates` | §10.6 |
| `mbt promote` | Registry stage transition | `--model --to --version --from-file --force` | §14.4 |
| `mbt docs generate\|serve` | Model cards + lineage site | `--port` | §15 |
| `mbt ls` | List resources | `--select --output table\|name\|path\|json` | §7 |
| `mbt show <name>` | Print one resource's compiled config | `--output yaml\|json` | §7 |
| `mbt state diff` | Diff current manifest vs a reference | `--state --output table\|json` | §14.1 |
| `mbt run-operation <macro>` | Render/execute a macro (escape hatch) | `--args '<dict>'` | §10.7 |
| `mbt clean` | Delete `target/` (including dataset cache) | | FR-RUN-09 |

Output conventions: inspection commands take `--output` for machine-readable results; long-running commands stream typed events (§16); exit codes are uniform (§17).
`--select/--exclude/--state/--manifest/--threads` have identical semantics everywhere they appear (FR-CLI-04).

## 4. Identity model

Every resource gets a stable **unique_id**: `<resource_type>.<project_name>.<name>`, e.g. `model.churn_project.churn_classifier`.
Resource names must be unique per resource type per project (parse error otherwise).
unique_ids are the node keys in the DAG, the manifest, run results, and selector evaluation.

## 5. Core data model (Pydantic v2 schemas)

All schemas use `model_config = ConfigDict(extra="forbid")` (FR-PARSE-04).
JSON Schema for editor autocomplete (FR-PARSE-03) is exported via `mbt parse --write-json-schema` and published with releases.

### 5.1 Common types

```python
class TaskType(StrEnum):
    BINARY_CLASSIFICATION = "binary_classification"
    MULTICLASS_CLASSIFICATION = "multiclass_classification"   # stretch
    REGRESSION = "regression"                                  # stretch
    RANKING = "ranking"                                        # v1
    SURVIVAL = "survival"                                      # v1

class SplitStrategy(StrEnum):
    TEMPORAL = "temporal"      # default (FR-RES-09)
    RANDOM = "random"          # must be opted into explicitly

class Materialization(StrEnum):
    MODEL_ARTIFACT = "model_artifact"
    ENSEMBLE = "ensemble"      # v1
    CALIBRATED = "calibrated"  # v1
    ONNX = "onnx"              # v1 (export path proven in v0)

class Stage(StrEnum):          # canonical stage tokens; registry adapters map them (§13.3)
    STAGING = "staging"
    PRODUCTION = "production"
    ARCHIVED = "archived"

AUTO = "__mbt_auto__"          # what "{{ auto }}" renders to (§6)
```

### 5.2 Project config (`mbt_project.yml`)

```python
class ProjectConfig(BaseModel):
    name: str                                  # ^[a-z][a-z0-9_]*$
    version: str
    require_mbt_version: str | None = None     # PEP 440 specifier, checked at parse
    model_defaults: dict[str, Any] = {}        # e.g. default adapter
    vars: dict[str, Any] = {}
    model_paths: list[str] = ["models"]
    dataset_paths: list[str] = ["datasets"]
    test_paths: list[str] = ["tests"]
    macro_paths: list[str] = ["macros"]
```

### 5.3 Profiles (`profiles.yml`, FR-PROJ-03)

Search order: `--profiles-dir`, `$MBT_PROFILES_DIR`, `./profiles.yml`, `~/.mbt/profiles.yml` (first hit wins).
Jinja (`env_var`, `var`) is rendered in profiles before validation; resolved secret values never leave process memory (§18).
The file maps project name → `ProfilesConfig` (dbt-style): `ProfilesFile = dict[str, ProfilesConfig]`.

```python
class AdapterRef(BaseModel):
    adapter: str                               # entry-point name, e.g. "mlflow", "local"
    config: dict[str, Any] = {}                # adapter-specific, validated by the adapter

class TargetConfig(BaseModel):
    data: AdapterRef
    tracking: AdapterRef
    registry: AdapterRef
    compute: AdapterRef = AdapterRef(adapter="local")
    artifact_store: str                        # URI: file://, s3://
    threads: int = 1
    vars: dict[str, Any] = {}                  # e.g. sample_fraction, max_tuning_trials

class ProfilesConfig(BaseModel):
    target: str                                # default target name
    outputs: dict[str, TargetConfig]
```

Example:

```yaml
churn_project:
  target: dev
  outputs:
    dev:
      data:     {adapter: local,  config: {root: ./data}}
      tracking: {adapter: mlflow, config: {uri: "http://localhost:5000"}}
      registry: {adapter: mlflow, config: {uri: "http://localhost:5000"}}
      compute:  {adapter: local}
      artifact_store: file://./target/artifacts
      vars: {sample_fraction: 0.1, max_tuning_trials: 5}
    prod:
      data:     {adapter: local,  config: {root: "{{ env_var('DATA_ROOT') }}"}}
      tracking: {adapter: mlflow, config: {uri: "{{ env_var('MLFLOW_URI') }}"}}
      registry: {adapter: mlflow, config: {uri: "{{ env_var('MLFLOW_URI') }}"}}
      artifact_store: "s3://ml-artifacts/churn"
      threads: 4
      vars: {sample_fraction: 1.0, max_tuning_trials: 50}
```

### 5.4 Sources (`sources.yml`, FR-RES-01)

```python
class SourceTable(BaseModel):
    name: str
    path: str | None = None          # for path-based sources (parquet/iceberg)
    identifier: str | None = None    # for warehouse/feature-store sources (v1)
    format: str = "parquet"          # parquet | iceberg
    description: str = ""

class SourceGroup(BaseModel):
    name: str                        # e.g. "lakehouse"
    tables: list[SourceTable]
```

### 5.5 Datasets (`datasets/*.yml`, FR-RES-02)

```python
class LabelSpec(BaseModel):
    column: str
    definition: str = ""             # human documentation, shown on model cards

class SplitSpec(BaseModel):
    strategy: SplitStrategy = SplitStrategy.TEMPORAL
    time_column: str | None = None   # required if temporal
    train: str                       # window (temporal) or fraction "0.8" (random)
    test: str
    validation: str | None = None    # else carved from train when tuning needs it (§10.5)
    stratify_by: str | None = None   # random strategy only
    seed: int | None = None          # random strategy only; required if strategy == random

CheckSpec = str | dict[str, dict[str, Any]]   # "label_leakage_scan" | {not_null: {columns: [...]}}

class DatasetSpec(BaseModel):
    name: str
    source: str                      # "source('lakehouse', 'gold_subscribers')"
    label: LabelSpec
    filters: list[str] = []          # SQL WHERE fragments, ANDed (executed by DuckDB)
    split: SplitSpec
    checks: list[CheckSpec] = []     # named checks (§11.1)
    snapshot: str | None = None      # explicit pin; normally pinned at compile (§8.3)
    tags: list[str] = []
```

**Window expressions.**
A window is `"<start>:<end>"` where each bound is a signed duration relative to the manifest anchor (`-180d`, `-28d`, `now`) or an ISO date; a bare duration `"28d"` is sugar for `"-28d:now"`.
Durations support `d`, `w`, `h`.
Windows are stored as expressions in specs and hashed as expressions; they resolve to concrete UTC timestamps against the manifest anchor at compile time (§8.2, ADR-12).

### 5.6 Models (`models/*.yml`, FR-RES-03) and task schemas (FR-RES-08)

```python
class FeatureSelection(BaseModel):
    include: list[str] = ["*"]       # glob patterns against post-hook dataset columns
    exclude: list[str] = []          # leakage guards; target + split time_column always excluded

class GateSpec(BaseModel):
    metric: str
    threshold: float | None = None       # absolute gate
    compare_to: Stage | None = None      # champion gate vs registry stage
    min_delta: float = 0.0               # only meaningful with compare_to
    slice: str | None = None             # optional per-slice gate (behavior: Could, PRD FR-TEST-04)
    # validator: exactly one of threshold / compare_to is set; min_delta requires compare_to

class EvaluationProtocol(BaseModel):
    split: SplitStrategy = SplitStrategy.TEMPORAL
    test_window: str | None = None       # narrows the dataset test window (see below)

class EvaluationSpec(BaseModel):
    protocol: EvaluationProtocol
    metrics: list[str]                   # names from metrics.yml, adapter built-ins, or hooks
    gates: list[GateSpec] = []
    slices: list[str] = []               # columns for per-slice reporting

class SearchDimension(BaseModel):
    type: Literal["int", "uniform", "loguniform", "categorical"]
    low: float | None = None
    high: float | None = None
    choices: list[Any] | None = None

class TuningObjective(BaseModel):
    metric: str                          # must appear in evaluation.metrics
    direction: Literal["maximize", "minimize"]

class TuningSpec(BaseModel):
    engine: str = "optuna"
    n_trials: int                        # capped by target var max_tuning_trials
    search_space: dict[str, SearchDimension]
    objective: TuningObjective

class RegistrationSpec(BaseModel):
    registry: str | None = None          # defaults to the target's registry adapter
    name: str
    stage_on_pass: Stage = Stage.STAGING

class ModelSpec(BaseModel):
    name: str
    description: str = ""
    task: TaskType
    adapter: str
    owner: str                           # email; required, shown on model cards
    tags: list[str] = []
    dataset: str                         # "ref('churn_training_set')"
    target: str                          # must equal the dataset's label.column
    features: FeatureSelection = FeatureSelection()
    hyperparameters: dict[str, Any] = {}     # validated by the adapter's param model
    tuning: TuningSpec | None = None
    evaluation: EvaluationSpec
    registration: RegistrationSpec | None = None
    materialization: Materialization = Materialization.MODEL_ARTIFACT
    seed: int                            # mandatory, no default (FR-RES-03)
    hooks: str | None = None             # path to hooks.py; sibling <name>.py auto-detected
```

**Cross-resource validation** (parse/compile time):

- `model.target` must equal the referenced dataset's `label.column`; mismatch is an error, not a silent override.
- `evaluation.protocol.split` must equal the dataset's `split.strategy` (FR-RES-09).
  The redundancy is deliberate: it makes the model spec self-describing in review and fails loudly if someone later flips the dataset's split.
- `evaluation.protocol.test_window`, when set, must resolve to a sub-range of the dataset's test window; the model is then evaluated on that narrower slice.
- Every `gates[].metric` and `tuning.objective.metric` must appear in `evaluation.metrics`.

**Task schema mechanism.**
Core keeps a task registry: `TaskType -> TaskSchema`.

```python
class TaskSchema(Protocol):
    task: TaskType
    allowed_metrics: set[str]            # constrains builtin metric names for this task
    def validate_spec(self, spec: ModelSpec) -> list[ValidationIssue]: ...
    def validate_dataset(self, spec: ModelSpec, profile: DatasetProfile) -> list[ValidationIssue]: ...
```

`validate_spec` runs at parse time (metric names, protocol sanity); hook-defined metric names are exempt from `allowed_metrics` and validated by existence in `hooks.py` instead.
`validate_dataset` runs at run time once the dataset profile exists (e.g., binary classification requires a binary label).
Adapters may register additional task schemas via their plugin descriptor (§12.3), which is how survival/ranking arrive in v1 without core changes (FR-V1-03).

Hyperparameters are validated at parse time by the **adapter**: each adapter exposes a plain-Pydantic param model per supported task (e.g. `XGBoostBinaryParams`, §13.1) with `extra="forbid"`; this is possible without importing the ML framework because plugin modules must be import-light (§12.3, ADR-14).
Values equal to `AUTO` skip static validation and are resolved by `resolve_auto` at run time (FR-RES-10).

### 5.7 Metrics, tests, exposures

```python
class MetricSpec(BaseModel):            # metrics.yml (FR-RES-04)
    name: str
    kind: Literal["builtin", "hook"]    # builtin: adapter-computed; hook: from hooks.py
    params: dict[str, Any] = {}         # e.g. recall_at_precision: {precision: 0.9}
    greater_is_better: bool = True

class ExposureSpec(BaseModel):          # exposures.yml (FR-RES-06)
    name: str
    type: Literal["endpoint", "batch_job", "dashboard", "other"]
    depends_on: list[str]               # ref() strings
    owner: str
    url: str | None = None
```

Metric names like `recall_at_precision_0.9` in a model spec are sugar: parsed into `(base_metric, params)` if not explicitly declared in `metrics.yml`.
Metric resolution order: explicit `metrics.yml` entry > sugar-parsed builtin > adapter builtin > hook metric; unknown names are a parse error listing the candidates.
Python data tests in `tests/` are files exposing `def test_*(dataset: pa.Table, spec: DatasetSpec) -> TestResult`, bound to resources via a `# mbt: select=<selector>` header comment or an explicit `tests:` key on the resource.

### 5.8 hooks.py contract (FR-RES-07)

A model may have a sibling `models/<name>.py` exposing any of:

```python
def transform_features(table: pa.Table, ctx: HookContext) -> pa.Table: ...
def custom_metrics(predictions: pa.Table, ctx: HookContext) -> dict[str, float]: ...
```

Hooks run inside the training job (§10.5), never in the coordinator.
`transform_features` is applied per split after read; `features.include/exclude` then applies to the post-hook column set.
The hook file's bytes are hashed into the node's `config_hash` (§8.4) so editing a hook marks the model `state:modified`.

## 6. Jinja templating layer

One shared `jinja2.Environment` (sandboxed, `StrictUndefined`) with these context functions:

| Function | Behavior |
|---|---|
| `ref(name)` | Records a DAG edge; renders to the unique_id at compile time |
| `source(group, table)` | Records a source dependency; renders to the source unique_id |
| `var(name, default=None)` | Resolution order: `--vars` CLI > target vars > project vars > default; missing + no default = compile error |
| `env_var(name, default=None)` | Reads the environment; values are tainted as secrets (§18) |
| `target` | The resolved target name and config (non-secret fields) |
| `auto` | Renders the sentinel `"__mbt_auto__"` (§5.1) |

Rendering is **two-phase**, dbt-style:

1. **Capture phase (parse):** specs are rendered with capturing `ref`/`source` implementations that record edges and return placeholders; `var`/`env_var` return inert placeholders so parse works without profiles or environment.
2. **Resolve phase (compile):** full rendering with real values against the selected target.

Macros from `macros/*.jinja` (FR-COMP-06) are loaded into the environment and usable in any spec.

## 7. Parsing pipeline (`mbt parse`)

```
discover files ─▶ load YAML ─▶ base Pydantic validation ─▶ task-schema + adapter
  param validation ─▶ capture-render Jinja (edges) ─▶ build DAG ─▶ cycle check
  ─▶ cross-resource checks (§5.6) ─▶ ParsedProject
```

1. **Discovery** walks the configured paths; YAML files must match resource schemas by location and top-level key (`models:`, `datasets:`, ...).
2. **Validation** collects *all* errors before failing (not fail-fast), each reported as file + resource + JSON-pointer field path + message (FR-PARSE-02), with Levenshtein did-you-mean for unknown fields (FR-PARSE-04).
3. **Adapter checks:** the model's `adapter` must be installed, its plugin must declare support for the model's `task` (FR-RES-08), and static hyperparameters must satisfy the adapter's param model; a missing adapter is a parse error naming the pip package to install.
4. Output is a `ParsedProject` (resources + DiGraph); `mbt ls` and `mbt show` read from it (`show` additionally compile-renders the one resource).

Performance budget (NFR-03): 50 resources parse in < 2 s.
This works because plugin *modules* are cheap to import by contract (no ML framework imports at module level, §12.3); the frameworks themselves load only inside training jobs.

## 8. Compilation and the manifest (`mbt compile`)

### 8.1 Inputs and steps

```
ParsedProject + profiles(target) + --vars [+ --anchor]
  ─▶ resolve-render Jinja
  ─▶ merge defaults (project model_defaults < spec)
  ─▶ pin anchor; resolve window expressions -> concrete UTC ranges
  ─▶ pin data snapshots (DataAdapter.snapshot_id per source)
  ─▶ compute config_hash per node, then input_hash in topo order
  ─▶ compute env_digest
  ─▶ write target/manifest.json
```

Compilation touches data systems only for snapshot IDs (cheap metadata calls); it never reads data.

### 8.2 Time anchoring (FR-COMP-05, ADR-12)

The manifest records one **anchor**: a UTC timestamp taken from the clock at compile time, overridable with `--anchor <iso-ts>` (used by tests and for deliberate re-pinning).
All relative window expressions resolve against this single anchor into concrete `[start_ts, end_ts)` ranges, stored per dataset node under `resolved` (outside the hashed config).
Consequences:

- **Reproducibility:** `--manifest` execution reuses the stored anchor and resolved windows verbatim, so a rerun reads identical data (G2).
- **State economy:** hashes cover window *expressions*, not resolutions, so mere time passage never marks nodes modified; new data arrives as a snapshot change, which does (FR-STATE-01).
- **Freshness:** every fresh compile re-anchors, so scheduled retrains naturally pick up new windows and new snapshots together.

### 8.3 Snapshot pinning

For each source, compile calls `DataAdapter.snapshot_id(source)`:

- **Iceberg:** the current snapshot ID from table metadata.
- **Local Parquet:** `sha256` over the sorted list of `(relative_path, size, mtime_ns)` of matching files - cheap and stable; a `--deep-snapshot` flag hashes file contents instead when mtimes are untrustworthy (CI caches) (ADR-11).

A dataset spec with an explicit `snapshot:` value freezes it: compile uses the pinned value and warns if it is no longer current.

### 8.4 Hashing (FR-COMP-02)

`config_hash = sha256(canonical_json(rendered_spec) + hooks_file_bytes)` where `canonical_json` is UTF-8, sorted keys, no whitespace, floats via `repr`.
Excluded from hashing: `description`, `owner`, `tags`, resolved window timestamps and the anchor (ADR-12), and everything from profiles (ADR-5: environment must not change node identity - moving dev→prod must not mark nodes modified).
`input_hash = sha256(config_hash + snapshot_id? + sorted(upstream input_hashes))`, computed in topological order; it transitively captures everything that affects the trained artifact.

`env_digest = sha256` over `python==X.Y.Z` plus sorted `package==version` lines for mbt-core, every installed mbt adapter, and each adapter's declared `fingerprint_packages` (e.g. mbt-xgboost declares `xgboost`).

### 8.5 manifest.json format (FR-COMP-01..05)

```jsonc
{
  "metadata": {
    "manifest_schema_version": 1,
    "mbt_version": "0.1.0",
    "project_name": "churn_project",
    "target": "prod",
    "generated_at": "2026-07-06T12:00:00Z",     // volatile; normalized in golden tests
    "anchor": "2026-07-06T12:00:00Z",            // volatile; §8.2
    "vars": { "sample_fraction": 1.0 },          // resolved, secrets excluded
    "env_digest": "sha256:6f2a...",
    "git": { "commit": "abc123", "branch": "main", "dirty": false }
  },
  "nodes": {
    "dataset.churn_project.churn_training_set": {
      "unique_id": "dataset.churn_project.churn_training_set",
      "resource_type": "dataset",
      "name": "churn_training_set",
      "path": "datasets/churn_training_set.yml",
      "depends_on": ["source.churn_project.lakehouse.gold_subscribers"],
      "config": { /* rendered DatasetSpec; window EXPRESSIONS, hashed */ },
      "resolved": {                                // NOT hashed
        "windows": {
          "train": ["2026-01-08T12:00:00Z", "2026-06-08T12:00:00Z"],
          "test":  ["2026-06-08T12:00:00Z", "2026-07-06T12:00:00Z"]
        }
      },
      "snapshot_id": "iceberg:8912734...",         // or "sha256:..." for local parquet
      "config_hash": "sha256:...",
      "input_hash": "sha256:..."
    },
    "model.churn_project.churn_classifier": {
      "unique_id": "model.churn_project.churn_classifier",
      "resource_type": "model",
      "name": "churn_classifier",
      "path": "models/churn_classifier.yml",
      "depends_on": ["dataset.churn_project.churn_training_set"],
      "config": { /* rendered ModelSpec; AUTO sentinels intact */ },
      "adapter": "xgboost",
      "task": "binary_classification",
      "seed": 42,
      "hooks_hash": "sha256:...",                  // null if no hooks.py
      "config_hash": "sha256:...",
      "input_hash": "sha256:..."
    }
  },
  "sources": { "source.churn_project.lakehouse.gold_subscribers": { /* ... */ } },
  "exposures": { /* ... */ },
  "metrics": { /* ... */ },
  "adapter_versions": { "xgboost": {"package": "mbt-xgboost", "version": "0.1.0", "contract": "1.0"} }
}
```

The manifest never contains secret values: profile configs are stored with `env_var()` expressions unrendered (§18).
`manifest_hash` (referenced by run results) is the sha256 of the canonical manifest with volatile metadata (`generated_at`, `anchor`) blanked.

## 9. DAG and node selection

### 9.1 Graph

`networkx.DiGraph`; nodes keyed by unique_id; edge `u -> v` means "v depends on u".
Cycle detection at parse (FR-DAG-01) reports the full cycle path.

### 9.2 Selector grammar (FR-DAG-02..04)

```
spec        := union ;
union       := intersection { " " intersection } ;      (* space: OR, across arg values *)
intersection:= atom { "," atom } ;                      (* comma: AND *)
atom        := [ [digits] "+" ] body [ "+" [digits] ] ; (* graph operators, dbt-style *)
body        := method ":" value | name_glob ;
method      := "tag" | "state" | "resource_type" ;
state values: "new" | "modified"
```

Examples: `churn_classifier+`, `+churn_classifier`, `2+churn_classifier`, `churn_classifier+1`, `tag:weekly,state:modified+`, `resource_type:model`.
`--exclude` evaluates the same grammar and subtracts.
`state:` methods require `--state <path-or-URI>` pointing at a reference manifest; URI schemes `file://` and `s3://` in v0 (FR-STATE-01).
Selectors always evaluate against the manifest being executed (freshly compiled, or the one given via `--manifest`).

### 9.3 `state:modified` semantics

A node is **modified** iff its `input_hash` differs from the same unique_id in the reference manifest.
A node is **new** iff its unique_id is absent from the reference manifest.
Anchor drift alone does not change `input_hash` (§8.2), so unchanged projects over unchanged data select nothing.
Because `input_hash` is transitive, plain `state:modified` already includes downstream-of-changed nodes; `state:modified+` remains the documented CI idiom for clarity and dbt muscle memory.
`env_digest` changes do **not** mark nodes modified by default; `mbt state diff` reports them prominently, and `--state-include-env` opts into treating them as modifying everything (ADR-7).

## 10. Execution engine (`mbt run` / `mbt build`)

### 10.1 Planning

1. Evaluate `--select`/`--exclude` against the manifest → the **selected set** (what the user asked to build/train).
2. **Upstream closure (ADR-13):** every dataset required by a selected model joins the execution plan even if unselected; datasets are cheap materializations and CI runners start cold.
   Selection therefore governs which models *train*, not which data exists (FR-RUN-12).
   (v1: an unselected upstream *model* of an ensemble is not retrained; its artifact is pulled from the registry.)
3. Missing prerequisites that cannot be auto-satisfied fail at plan time with guidance (before any training cost is incurred).

### 10.2 Scheduling

- Topological execution over the plan using `ThreadPoolExecutor(max_workers=threads)` (FR-RUN-02); a node starts when all its in-plan parents succeeded.
- Node failure marks all transitive downstream as `skipped`; independent branches continue; `--fail-fast` cancels pending work (FR-RUN-03).
- Threads only coordinate; training runs in subprocesses (ADR-3), so the GIL never serializes real work.
- `mbt build` interleaves dbt-style: after a node runs, its attached tests/gates run before its children are released; a gate or test failure fails the node for scheduling purposes (FR-RUN-01, FR-TEST-03).

### 10.3 Coordinator / job split (ADR-3)

Everything framework- or data-heavy runs inside a **training job**; the coordinator stays pure.

| Coordinator (mbt process) | Training job (subprocess; K8s/Ray in v1) |
|---|---|
| Planning, scheduling, selector logic | Reconstruct `DatasetHandle` from `DatasetLocator` |
| Dataset builds via DataAdapter (DuckDB, in-process) | `hooks.transform_features` + feature selection |
| Champion lookup via RegistryAdapter (pre-submit) | `resolve_auto`, tuning trials, final training |
| Gate **decisions** (pure arithmetic on returned metrics) | Metric computation for challenger *and* champion |
| Registration + stage transition on gate pass | Artifact export to the artifact store |
| Attaching gate/registration tags to the tracking run | Tracking run creation + param/metric/trial logging |
| Writing `run_results.json` | Returning a serializable `JobResult` |

The job entrypoint is `python -m mbt.execute.job <job.json>`; the payload (`TrainingJob`, §12.1) is fully serializable, which is exactly the seam remote ComputeAdapters reuse in v1 (FR-V1-01).
The rule of thumb: **adapters compute metrics; core compares them.**

### 10.4 Dataset runner (coordinator)

1. Compute the **materialization key** = `sha256(input_hash + canonical_json(resolved.windows))`; reuse the cached materialization under `target/datasets/<name>/<key>/` on hit.
2. On miss, `DataAdapter.build_dataset(spec, ctx)` materializes splits (filters, sampling, split assignment) and writes the cache.
3. Verify the handle's `snapshot_id` equals the manifest pin; mismatch = error (data moved under a pinned manifest).
4. Run dataset checks (§11.1); compute and cache `DatasetProfile`.

The key includes resolved windows because two anchors can slice the same snapshot differently; `input_hash` alone would alias them.

### 10.5 Model runner

Coordinator side:

1. Resolve the adapter plugin and the model's dataset handle (from §10.4); obtain its `DatasetLocator`.
2. If any gate has `compare_to`, resolve the champion now: `RegistryAdapter.get_champion(name, stage)` → `ModelVersion` (with `ArtifactRef`), or none.
3. Assemble `TrainingJob` (node, locator, champion ref, tuning cap from `var('max_tuning_trials')`, artifact store URI, required env-var names) and `compute.submit(job)`; `wait` for `JobResult`.
4. Evaluate gates (§11.2) using the returned challenger and champion `MetricResults`.
5. On pass: `registry.register(artifact, name, metadata={config_hash, input_hash, manifest_hash, snapshot_id, git_commit, tracking_run_id})`, then `transition(version, stage_on_pass)`; attach gate results + version as tags via `tracking.resume(run_id)`.
6. On fail: status `gate_failed`, no registration, tags still recorded.
7. Emit the `RunResult` (metrics, gates, artifact, registration, resolved auto values).

Job side (in order):

1. Reconstruct the dataset handle via `DataAdapter.from_locator`; read splits as Arrow.
2. Apply `hooks.transform_features` per split, then `features.include/exclude` on the post-hook columns.
3. `adapter.resolve_auto(spec, profile)`: replace `AUTO` values, log them.
4. `tracking.start_run(...)`: params (final hyperparameters), mbt tags (hashes, snapshot, git).
5. If tuning: run the trial loop (§13.5) on train vs validation; never touch the test split (ADR-8).
6. Final fit: train on the declared train window with the winning params.
   When tuning carved an *implicit* validation slice, the final fit reabsorbs it (fit on the full declared train window); an *explicitly declared* validation split stays held out, because the user said so.
7. `adapter.evaluate` on the test split (+ `hooks.custom_metrics`); if a champion ref was provided, `adapter.load(champion_ref)` and evaluate it on the *same* test split (ADR-9).
8. `adapter.export` → artifact store; log metrics/artifacts; `end_run`.
9. Return `JobResult{status, metrics, slices, champion_metrics, resolved_auto, artifact_ref, tracking_run_id, error}`.

All seeds derive from `spec.seed` deterministically: adapter seed = `seed`, tuning sampler = `seed + 1`, implicit validation carve = `seed + 2`.

### 10.6 `--manifest` execution and `mbt evaluate`

`mbt run|build|test --manifest <path>` skips parse/compile and executes the given manifest verbatim: same anchor, same resolved windows, same snapshots, same hashes (FR-RUN-11).
This is the reproducibility mechanism (G2) and the audit tool ("re-run exactly what CI ran").
mbt warns if the current project files disagree with the manifest but does not re-render them.

`mbt evaluate --model X [--version N | --stage production] [--gates]` re-evaluates an existing artifact on *fresh* data (FR-RUN-07): compile (fresh anchor) → build the model's dataset → submit an evaluation-only job (load artifact via the registry's `ArtifactRef`, `adapter.evaluate` on the test split) → optionally apply gates → write run results with `"command": "evaluate"`.
Version resolution: explicit `--version`, else the latest version in `--stage` (default: the model's `stage_on_pass`).

### 10.7 run-operation

`mbt run-operation <macro> --args '<dict>'` renders the macro with the full compile context and prints the result; it is a maintenance escape hatch (FR-RUN-08).
Adapter-invoking operations are deliberately out of scope until dogfooding demands them.

### 10.8 run_results.json (FR-RUN-04)

```jsonc
{
  "metadata": { "run_results_schema_version": 1, "run_id": "01J...", "mbt_version": "0.1.0",
                "target": "prod", "manifest_hash": "sha256:...", "anchor": "2026-07-06T12:00:00Z",
                "started_at": "...", "elapsed_s": 812.4,
                "command": "build", "selector": "state:modified+" },
  "results": [
    {
      "unique_id": "model.churn_project.churn_classifier",
      "status": "success",              // success | error | skipped | gate_failed | test_failed
      "execution_time_s": 640.2,
      "metrics": { "pr_auc": 0.4471, "roc_auc": 0.8712, "ece": 0.031 },
      "slices": { "plan_type=pro": { "pr_auc": 0.51 } },
      "gates": [
        { "metric": "pr_auc", "kind": "threshold", "expected": 0.42, "actual": 0.4471, "passed": true },
        { "metric": "pr_auc", "kind": "champion", "champion_version": "7",
          "champion_value": 0.4402, "min_delta": 0.005, "actual_delta": 0.0069, "passed": true }
      ],
      "artifact": { "uri": "s3://ml-artifacts/churn/01J.../model.ubj", "format": "xgboost_ubj" },
      "registration": { "registry": "mlflow", "name": "churn_classifier", "version": "8", "stage": "staging" },
      "tracking_run_id": "mlflow:runs/4f9a...",
      "resolved_auto": { "scale_pos_weight": 11.4 },
      "message": null
    }
  ]
}
```

## 11. Testing and quality gates

### 11.1 Data tests and checks (FR-TEST-01/05)

Built-in named checks (dataset `checks:`): `no_future_columns` (no column whose max timestamp exceeds the split boundary), `label_leakage_scan` (flags features with suspicious association to the label, e.g. |corr| or single-feature AUC above a threshold), `class_balance_report` (report-only), `schema` (declared columns exist with expected types), `not_null` (with a `columns` argument).
Python tests from `tests/` run against the materialized dataset via the contract in §5.7.
Failures set node status `test_failed` and exit code 2.

### 11.2 Gate evaluation (FR-TEST-02/03/06)

Gates are pure comparisons over `MetricResults` returned by jobs (§10.3); the logic lives in `mbt.quality` and is fully unit-testable without ML dependencies.

- **Threshold gates:** direction-aware (`greater_is_better`); with a non-exact determinism tier, the adapter's declared tolerance widens the comparison in the model's favor only across rerun comparisons, never for the champion delta (FR-ADPT-07).
- **Champion gates:** require `challenger_value - champion_value >= min_delta` (direction-adjusted), where both values were computed by the same adapter on the identical pinned test split inside the job (ADR-9).
  If no champion exists, the gate passes with an explicit `WARN` event and `"champion_version": null` (FR-TEST-06, ADR-10).
  If the champion exists but cannot be loaded (framework mismatch, missing artifact), the gate errors; silent skips would rubber-stamp promotions (ADR-10).
- Gate results are attached to the tracking run as tags; `mbt promote` later verifies them (§14.4).

### 11.3 Standalone `mbt test`

`mbt build` runs tests interleaved; standalone `mbt test` must also work (FR-TEST-01):

- **Data tests/checks:** ensure the dataset materialization exists (cache hit or build, §10.4), then run checks and Python tests against it.
- **Model gate tests:** re-evaluate via the `mbt evaluate` machinery (§10.6) against the latest registered version in the model's `stage_on_pass` (override with `--version`); if no version exists, the node is `skipped` with a warning.
  Training something new is never a side effect of `mbt test`.

## 12. Adapter architecture

### 12.1 Interchange types

Everything crossing a contract boundary is a plain Pydantic model (or Arrow table); no framework types, everything serializable:

```python
class ValidationIssue(BaseModel):
    severity: Literal["error", "warning"]
    resource: str                            # unique_id
    field_path: str                          # JSON pointer
    message: str
    hint: str | None = None

class DeterminismTier(BaseModel):
    kind: Literal["exact", "tolerance"]
    tolerances: dict[str, float] = {}        # metric name -> absolute tolerance

class ArtifactRef(BaseModel):
    uri: str
    format: str                              # e.g. "xgboost_ubj", "onnx"
    content_hash: str
    size_bytes: int

class MetricResults(BaseModel):
    metrics: dict[str, float]
    slices: dict[str, dict[str, float]] = {} # "plan_type=pro" -> {metric: value}

class DatasetProfile(BaseModel):
    n_rows: dict[str, int]                   # per split
    columns: dict[str, str]                  # name -> arrow dtype string
    label_balance: dict[str, float] | None   # classification only
    time_range: tuple[str, str] | None

class DatasetLocator(BaseModel):             # serializable pointer to a materialized dataset
    adapter: str
    uri: str                                 # e.g. file://target/datasets/churn/<key>/
    snapshot_id: str

class ModelVersion(BaseModel):
    name: str
    version: str
    stage: Stage | None
    artifact: ArtifactRef | None
    tags: dict[str, str] = {}

class RunHandle(BaseModel):
    run_id: str
    url: str | None = None

class TrainingJob(BaseModel):                # the coordinator -> job payload (§10.3)
    node: ManifestNode                       # resolved spec, hashes, seed
    dataset: DatasetLocator
    champion: ArtifactRef | None
    tuning_cap: int | None
    artifact_store: str
    required_env: list[str]                  # names only; values re-resolved in the job (§18)

class JobResult(BaseModel):
    status: Literal["success", "error"]
    metrics: MetricResults | None
    champion_metrics: MetricResults | None
    resolved_auto: dict[str, Any] = {}
    artifact: ArtifactRef | None
    tracking_run_id: str | None
    error: str | None = None

class TestResult(BaseModel):
    name: str
    passed: bool
    message: str = ""

class TuningResult(BaseModel):
    best_params: dict[str, Any]
    best_value: float
    n_trials: int

@dataclass(frozen=True)
class HookContext:
    spec: ModelSpec
    profile: DatasetProfile
    split: str
    logger: EventEmitter
```

`TrainedModel` is deliberately opaque to core: an object only its own adapter's methods ever touch.

### 12.2 Contracts (FR-ADPT-01)

```python
class DatasetHandle(Protocol):
    snapshot_id: str
    def splits(self) -> set[str]: ...                            # {"train","test","validation"?}
    def read(self, split: str, columns: list[str] | None = None) -> pa.Table: ...
    def profile(self) -> DatasetProfile: ...
    def locator(self) -> DatasetLocator: ...                     # for job payloads

class TrainingAdapter(Protocol):
    name: str
    contract_version: str                    # e.g. "1.0"
    supported_tasks: set[TaskType]
    determinism: DeterminismTier
    def param_model(self, task: TaskType) -> type[BaseModel]: ...
    def validate(self, spec: ModelSpec) -> list[ValidationIssue]: ...
    def resolve_auto(self, spec: ModelSpec, profile: DatasetProfile) -> ModelSpec: ...
    def train(self, spec: ModelSpec, data: DatasetHandle, ctx: RunContext) -> TrainedModel: ...
    def evaluate(self, model: TrainedModel, data: DatasetHandle, split: str,
                 metrics: list[MetricSpec]) -> MetricResults: ...
    def load(self, ref: ArtifactRef) -> TrainedModel: ...        # champion eval, mbt evaluate
    def export(self, model: TrainedModel, format: str, store: ArtifactStore) -> ArtifactRef: ...

class DataAdapter(Protocol):
    name: str
    def snapshot_id(self, source: SourceTable) -> str: ...
    def build_dataset(self, spec: DatasetSpec, ctx: RunContext) -> DatasetHandle: ...
    def from_locator(self, locator: DatasetLocator) -> DatasetHandle: ...   # job side

class TrackingAdapter(Protocol):
    def start_run(self, node: ManifestNode, meta: dict[str, str]) -> RunHandle: ...
    def log(self, run: RunHandle, *, params: dict | None = None, metrics: dict | None = None,
            tags: dict | None = None, artifacts: list[ArtifactRef] | None = None) -> None: ...
    def end_run(self, run: RunHandle, status: str) -> None: ...
    def resume(self, run_id: str) -> RunHandle: ...              # coordinator attaches gate tags

class RegistryAdapter(Protocol):
    def register(self, artifact: ArtifactRef, name: str, metadata: dict) -> ModelVersion: ...
    def get_champion(self, name: str, stage: Stage) -> ModelVersion | None: ...
    def get_version(self, name: str, version: str) -> ModelVersion | None: ...
    def transition(self, version: ModelVersion, stage: Stage) -> None: ...

class ComputeAdapter(Protocol):
    def submit(self, job: TrainingJob) -> JobHandle: ...
    def wait(self, handle: JobHandle) -> JobResult: ...

class TuningEngine(Protocol):
    name: str
    def tune(self, spec: TuningSpec, objective: Callable[[dict[str, Any]], float],
             n_trials: int, seed: int) -> TuningResult: ...
```

### 12.3 Plugin discovery and import hygiene (FR-ADPT-02, ADR-14)

Entry point group `mbt.adapters`; the value is an `AdapterPlugin` descriptor:

```python
class AdapterPlugin(BaseModel):
    name: str                                    # "xgboost", "mlflow", "local"
    contract_version: str                        # pinned contract major.minor
    training: type[TrainingAdapter] | None = None
    data: type[DataAdapter] | None = None
    tracking: type[TrackingAdapter] | None = None
    registry: type[RegistryAdapter] | None = None
    compute: type[ComputeAdapter] | None = None
    tuning: type[TuningEngine] | None = None
    task_schemas: dict[TaskType, type[TaskSchema]] = {}   # v1: survival, ranking
    fingerprint_packages: list[str] = []                   # for env_digest (§8.4)
```

```toml
# mbt-xgboost pyproject.toml
[project.entry-points."mbt.adapters"]
xgboost = "mbt_xgboost.plugin:PLUGIN"
```

**Import hygiene rule:** importing a plugin module (and constructing its adapter classes) must not import the ML framework; frameworks load lazily inside `train`/`evaluate`/`load`/`export`.
This is what lets `mbt parse` validate tasks and hyperparameters within its 2 s budget (NFR-03) while `param_model` stays plain Pydantic.
The compliance suite enforces it (§12.4).

Core checks `contract_version` compatibility at load (same major, minor <= core's) and refuses incompatible adapters with an upgrade hint.

### 12.4 Compliance suite (FR-ADPT-05)

`mbt-adapter-base` ships `mbt_adapter_base.compliance` as pytest fixtures/base classes an adapter repo subclasses:

```python
class TrainingAdapterCompliance:
    adapter: TrainingAdapter                      # provided by subclass fixture
    tiny_datasets: dict[TaskType, DatasetHandle]  # provided; ~1k rows, committed parquet
```

Parametrized over `supported_tasks`, it asserts:

- same seed + same data → identical metrics (within the declared determinism tier);
- `resolve_auto` is idempotent and leaves no `AUTO` sentinels;
- `param_model` rejects unknown params with actionable messages;
- `train` → `export` → `load` → `evaluate` round-trips to the same metrics;
- importing the plugin module does not import the framework (`sys.modules` check, ADR-14).

The suite is the ship bar for adapters and the contract's regression net (G4).

## 13. v0 adapter implementations

### 13.1 mbt-xgboost (FR-ADPT-03)

- `supported_tasks`: `{BINARY_CLASSIFICATION}` in Phase 1; regression/multiclass behind the same param-model mechanism as stretch.
- `mbt_xgboost.plugin` contains only the descriptor and Pydantic param models; `import xgboost` happens inside adapter methods (§12.3).
- Param models mirror XGBoost params (`max_depth`, `learning_rate`, `n_estimators`, `scale_pos_weight`, ...) with `extra="forbid"`; `scale_pos_weight: AUTO` resolves to `n_negative / n_positive` from the profile.
- Training: `xgb.train` on `DMatrix(pa.Table)` with `seed=spec.seed`, `nthread` fixed, `tree_method="hist"`; determinism tier **exact** for CPU hist (documented).
- Built-in metrics: `roc_auc`, `pr_auc`, `logloss`, `ece`, `recall_at_precision_*`, `precision_at_recall_*`; slice metrics via group-by on slice columns.
- Export: native `.ubj` (default) + `onnx` via onnxmltools (optional extra `mbt-xgboost[onnx]`).
- `load` reads `.ubj` from an `ArtifactRef` for champion evaluation and `mbt evaluate`.

### 13.2 Local DataAdapter (`local`, FR-ADPT-04)

- Sources resolve to Parquet globs under `config.root` (Iceberg tables via optional `mbt-core[iceberg]` extra using pyiceberg).
- `build_dataset` runs one DuckDB query: read source → apply `filters` → apply target sampling (`sample_fraction` var, hash-sampled on a stable key for determinism) → assign splits (resolved temporal windows on `time_column`, or seeded hash split for random) → write one Parquet file per split under `target/datasets/<name>/<materialization_key>/` (key per §10.4).
- `from_locator` reopens a materialized directory; `read` returns Arrow via DuckDB; `profile()` computes counts/schema/balance once and caches JSON alongside the splits.

### 13.3 mbt-mlflow (FR-REG-01/04)

- One package, one plugin, two contracts: `MlflowTracking` and `MlflowRegistry`, sharing client config (`uri`).
- `start_run` creates an MLflow run named after the node, tagged with `mbt.config_hash`, `mbt.input_hash`, `mbt.manifest_hash`, `mbt.snapshot_id`, `mbt.git_commit`, `mbt.run_id` (FR-REG-05); `resume` reattaches by run id for the coordinator's gate/registration tags.
- Canonical stages map to MLflow: `staging → Staging`, `production → Production`, `archived → Archived`; on MLflow ≥ 2.9 the adapter can use aliases instead (config `use_aliases: true`).
- `register` creates a model version from the exported artifact; `metadata` lands in version tags; `get_champion` returns the version currently in the given stage (or alias).

### 13.4 Local ComputeAdapter (`local`, FR-RUN-10)

- `submit` spawns `python -m mbt.execute.job <job.json>`: crash/memory isolation, real `--threads` parallelism, and the serialization seam K8s/Ray reuse in v1 (ADR-3).
- The payload carries env-var *names*; the subprocess inherits the environment and re-resolves values itself (§18).
- `wait` reaps the process and parses the `JobResult` from a result file (not stdout, which belongs to logs).

### 13.5 mbt-optuna (FR-TUNE-01..04)

- The trial loop runs inside the training job (§10.5); the engine only proposes parameters.
  Per trial: train on the train split with trial params, evaluate `objective.metric` on the validation split.
- Validation split: the dataset's declared `validation` window, else carved deterministically from train (temporal: last 20% by time; random: seeded 20% with `seed + 2`).
  Tuning never sees the test split (ADR-8); the final-fit rule is in §10.5 step 6.
- `TPESampler(seed=spec.seed + 1)`; `n_trials = min(spec.tuning.n_trials, tuning_cap)` from the job payload (FR-TUNE-04); trial history logged as nested tracking runs (FR-TUNE-03).

## 14. State and GitOps mechanics

### 14.1 `mbt state diff` (FR-STATE-02)

Compares current manifest vs `--state` reference: nodes added / removed / modified, annotated with *which* component changed (config, hooks, snapshot, upstream), plus the env_digest delta.
Output: Rich table or `--output json` (consumed by the PR comment bot).

### 14.2 Manifest storage convention (FR-STATE-03)

```
s3://<bucket>/mbt/<project>/<target>/manifests/<git_sha>.json
s3://<bucket>/mbt/<project>/<target>/manifests/latest.json      # copy, updated on successful prod build
```

CI convention: the prod build uploads its manifest on success; PR checks pass `--state s3://.../latest.json`.
Teams without S3 use CI artifact storage with the same layout; the `--state` flag only needs a readable path/URI.

### 14.3 Reference GitHub Actions (FR-STATE-04/05)

Shipped in the `mbt init` template:

- **pr_check.yml:** `mbt parse` → `mbt compile --target dev` → `mbt state diff --state <latest prod manifest> --output json` → `mbt build --target dev --select state:modified+ --state ...` → post PR comment (metrics vs champion from run_results, gate table, retrained node list, cost estimate = Σ node execution_time × runner rate).
- **prod_build.yml** (on merge to main): `mbt build --target prod --select state:modified+ --state .../latest.json` → upload manifest as new `latest.json` on success.
- **promote.yml:** triggered by a reviewed change to `promotions.yml` (pure GitOps) or `workflow_dispatch` with environment approval; runs `mbt promote --from-file promotions.yml`.

### 14.4 `mbt promote` (FR-REG-03)

`mbt promote --model churn_classifier --to production [--version N]`, or `--from-file promotions.yml` where the file is a reviewed list of `{model, version, to}` entries:

1. Resolve the version (explicit, or latest in the model's `stage_on_pass`).
2. Verify gate-pass tags recorded at registration (§11.2); refuse otherwise (`--force` overrides with a loud event).
3. `RegistryAdapter.transition(version, Stage.PRODUCTION)`; emit event; exit 0.

Serving CD reacts to the registry stage change; mbt's job ends here (non-goal boundary).

## 15. Docs generation (`mbt docs`, FR-DOCS-01..03)

- Input: manifest + latest run_results (+ tracking links).
- Output: static site in `target/docs/`: `index.html` (DAG lineage rendered client-side from an embedded `lineage.json`; datasets, models, exposures) and one model card per model (description, owner, tags, data window + snapshot, features + exclusions, hyperparameters incl. resolved AUTO values, metrics + slice table, gate history, registry links).
- Templates: Jinja HTML in `mbt/docsgen/templates/`, self-contained (inline CSS/JS, no CDN) so the site works on corp networks and artifact hosting.
- `mbt docs serve`: `http.server` on the output dir.

## 16. Observability (NFR-06)

- Every significant occurrence is a typed event (Pydantic): `ParseStarted`, `NodeStarted`, `NodeFinished`, `GateEvaluated`, `AutoResolved`, `ArtifactRegistered`, `RunFinished`, ...
- Two sinks: Rich console renderer (human) and JSON-lines (`--log-format json`, one event per line with `run_id`, `unique_id`, timestamps).
  Job subprocesses emit the same event stream on their stdout; the coordinator forwards it.
- OpenTelemetry: optional spans per command and per node execution, enabled via standard `OTEL_*` env vars; no-op otherwise.

## 17. Error handling and exit codes (FR-CLI-02)

Precedence, evaluated over the whole invocation:

1. Any hard error (config/parse, compilation, adapter, infrastructure, unexpected exception) → **exit 1**.
2. Else any node with `test_failed` or `gate_failed` → **exit 2**.
3. Else → **exit 0**.

| Exit | Meaning | Examples |
|---|---|---|
| 0 | Success | All selected nodes succeeded, gates passed |
| 1 | Error | Validation errors, adapter/job crashes, snapshot mismatch, unreadable `--state` |
| 2 | Quality failure | Gate below threshold, champion delta not met, data test failed |

Exception taxonomy: `MbtError` (base) → `ConfigError` (parse/validation), `CompilationError`, `AdapterError` (wraps adapter exceptions with node context), `GateFailure`, `StateError`.
All user-facing errors carry: what happened, which resource/file, and what to do next.

## 18. Security (NFR-07, FR-PROJ-05)

- Secrets enter only via `{{ env_var() }}`; rendered values are wrapped in a `Secret` type whose `repr`/serialization is `"***"`.
- The manifest stores profile configs **unrendered** (the `env_var()` expression, not the value); events, run_results, and tracking params pass through a redaction filter keyed on tainted values.
- `TrainingJob` carries env-var *names* only (collected by scanning the target config's `env_var()` expressions); the job process re-resolves them from its own environment.
- `mbt init` writes `profiles.yml` to `~/.mbt/` by default and gitignores a project-local one.

## 19. Versioning and compatibility (NFR-05)

- All packages: SemVer, released via release-please with conventional commits.
- `manifest_schema_version` and `run_results_schema_version` are integers; core reads schema N and N-1 (so `--state` works against the previous release's manifests) and refuses newer with a clear message.
- Adapter contract (`mbt-adapter-base`): SemVer'd independently; plugins declare their `contract_version`; core accepts same-major, minor <= core's.
  Deprecations: warn one minor, remove next major.
- `require_mbt_version` in `mbt_project.yml` guards project/tool skew.

## 20. Performance considerations (NFR-03)

- Parse: import-light plugins (§12.3), single pass, all-errors-at-once validation; target < 2 s at 50 resources.
- Compile: snapshot calls parallelized across sources (`ThreadPoolExecutor`); target < 10 s.
- Execution overhead per node < 2 s: dataset reuse via the materialization cache (§10.4), lazy framework imports, Arrow zero-copy hand-off.
- Large data is v1 territory (remote compute); v0 assumes single-machine-sized training data (PRD §10).

## 21. Testing strategy for mbt itself (NFR-08)

| Layer | Approach |
|---|---|
| Schemas | Unit tests + hypothesis (round-trip YAML → model → JSON Schema validation) |
| Selector grammar | Property-based: random DAGs + selector algebra invariants (e.g. `a,b ⊆ a`) |
| Compile | Golden-file tests: `examples/churn_demo` compiles to a checked-in manifest (volatile metadata normalized, `--anchor` pinned); any diff is a reviewable change |
| Hashing/state | Mutation tests: change one field → exactly the expected nodes flip to modified; anchor drift flips none |
| Execution | Fake adapters (in-repo, contract-conformant) for planner/scheduler/gate/skip logic without ML deps |
| Adapters | Compliance suite (§12.4) + adapter-local unit tests on tiny committed datasets |
| E2E | CI job: `mbt init` template → local MLflow (sqlite) → `mbt build` → assert registered version → `mbt build --manifest` → assert identical metrics (G2) |
| Repo hygiene | ruff + mypy --strict + yamllint via pre-commit; enforced in CI |

## 22. v1+ design notes (architecture accommodation, FR-V1-*)

- **K8s/Ray ComputeAdapters (FR-V1-01):** the serialized `TrainingJob` (§10.3) becomes a K8s Job (image contract: mbt-core + the adapter installed; payload mounted; datasets and artifacts exchanged via object-store locators) or a Ray task.
  No planner or scheduler changes: `submit`/`wait` already abstract remoteness, and `DatasetLocator` already abstracts data placement.
- **sklearn/PyTorch adapters (FR-V1-02):** new packages against the same contract; sklearn maps `Pipeline` steps into the param model; PyTorch declares a tolerance determinism tier.
- **Survival/ranking tasks (FR-V1-03):** adapters register task schemas via `AdapterPlugin.task_schemas` (§12.3); core's task registry merges them - no core changes.
- **Feast DataAdapter (FR-V1-04):** `source()` gains a feature-view form; the adapter materializes point-in-time-correct training frames behind the same `DatasetHandle`/`DatasetLocator`.
- **Ensembles (FR-V1-05):** a model with `inputs: [ref('model_a'), ref('model_b')]`; the DAG and manifest already support model→model edges; planning pulls unselected member artifacts from the registry (§10.1); an `ensemble` materialization stacks out-of-fold member predictions as features.
- **`mbt score` (FR-V1-06):** load artifact via `TrainingAdapter.load`, read a dataset resource, write predictions through the DataAdapter - all existing contracts.
- **Airflow provider (FR-V1-07):** an operator shelling out to `mbt build --select ...` per manifest-derived task group; mbt stays orchestrator-agnostic.

## 23. Key design decisions (ADR summaries)

Full ADRs live in `docs/adr/`; decisions made here beyond PLAN.md:

| # | Decision | Rationale |
|---|---|---|
| ADR-1 | **Arrow as the data interchange format** across adapter boundaries | Framework-neutral, zero-copy into XGBoost/Polars/DuckDB; keeps ML types out of core |
| ADR-2 | **Local Data/Compute adapters ship inside mbt-core** | Batteries included for G5 (1-hour quickstart); DuckDB/Polars are data deps, not ML deps, so NFR-04 holds |
| ADR-3 | **Coordinator/job split; training always runs in a subprocess** | Real `--threads` parallelism, crash isolation, pure gate logic in core ("adapters compute metrics, core compares them"), and the exact serialization seam K8s/Ray reuse in v1 |
| ADR-4 | **Two hashes per node: `config_hash` and transitive `input_hash`**; `state:modified` compares `input_hash` | One comparison captures config, hooks, snapshot, and upstream changes; simple and correct. A separate train-only hash (so gate edits don't retrain) is deferred until dogfooding shows the cost matters |
| ADR-5 | **Profiles are excluded from config hashes** and stored unrendered in the manifest | Environment must not change node identity (dev vs prod retrain decisions); keeps secrets out of the manifest |
| ADR-6 | **Gate changes retrain the node** (consequence of ADR-4) | Acceptable v0 cost for hash simplicity; revisit with ADR-4 |
| ADR-7 | **env_digest changes do not mark nodes modified by default** | An adapter version bump would retrain everything; teams opt in via `--state-include-env`. The diff surfaces it loudly either way |
| ADR-8 | **Tuning never sees the test split; implicit validation carves are reabsorbed in the final fit, explicit ones stay held out** | Gates stay honest; final-fit behavior is predictable and respects explicit user declarations |
| ADR-9 | **Champion evaluation reruns the champion on the current pinned test split, inside the job** | Stored champion metrics came from a different data window; fair comparison requires identical data and identical metric code. Cost: one extra evaluate per gated model |
| ADR-10 | **Missing champion = gate passes with WARN; unloadable champion = error** | Bootstrap must not block the first model; silent skips on real failures would rubber-stamp promotions |
| ADR-11 | **Local Parquet snapshot = hash of (path, size, mtime) file listing**, `--deep-snapshot` for content hashing | Cheap by default, correct option where mtimes lie (CI caches) |
| ADR-12 | **Window expressions are hashed; one manifest-wide anchor pins their resolution, stored outside the hashed config** | Reruns via `--manifest` reproduce exactly (G2) while mere time passage never marks nodes modified; new data surfaces as a snapshot change, which does |
| ADR-13 | **Required upstream datasets auto-materialize; selection governs training** | A model-only PR on a cold CI runner must not fail or force selecting the world; dataset builds are cheap, training is the 1000× cost |
| ADR-14 | **Plugin import hygiene: plugin modules and param models import no ML framework** | Parse-time task/hyperparameter validation within the 2 s budget; enforced by the compliance suite |
