# Architecture Blueprint: MBT — A Declarative ML Pipeline Framework

> **Version 4 — Revised with architectural feedback integrated**

MBT (Model Build Tool) is a declarative ML framework where data scientists define training and serving pipelines in YAML, and the framework compiles those declarations into executable task graphs. Data scientists declare *what* they want — data sources, transformations, model configuration, evaluation criteria — and MBT figures out *how* to execute it. Data engineers configure *where* it runs via `profiles.yaml`, and the compiled `manifest.json` bridges both worlds into any orchestrator (Airflow, Prefect, Dagster, etc.).

MBT follows a modular, adapter-based architecture inspired by dbt. The core package (`mbt-core`) is a thin CLI + compiler with zero infrastructure dependencies. Every concrete integration — ML frameworks, data warehouses, orchestrators, model registries — is a separate pip-installable adapter package (`mbt-h2o`, `mbt-snowflake`, `mbt-airflow`, `mbt-mlflow`, etc.). Users install only what they need, and anyone can publish new adapters.

This design draws from four key influences: **dbt** for profiles, targets, CLI ergonomics, the adapter ecosystem, and compile-before-run philosophy; **Ludwig** for fully declarative YAML-driven ML with pluggable backends; **MLflow Recipes** for templated step execution with profile-based environment switching; and **Kedro** for layered configuration and data catalog patterns. MBT takes an AutoML-first approach — data scientists get strong models without manual hyperparameter tuning via adapters like `mbt-h2o`, while the plugin system allows power users to bring any framework.

---

## Table of Contents

1. [Design Philosophy](#1-design-philosophy)
2. [Pipeline YAML Schema — The DS Interface](#2-pipeline-yaml-schema--the-ds-interface)
3. [Pipeline Composition and Reuse](#3-pipeline-composition-and-reuse) **[NEW]**
4. [Profiles — The DE Interface](#4-profiles--the-de-interface)
5. [Compilation — From YAML Declaration to Executable DAG](#5-compilation--from-yaml-declaration-to-executable-dag)
6. [Data Abstraction Layer](#6-data-abstraction-layer) **[NEW]**
7. [Artifact Flow and Inter-Step Communication](#7-artifact-flow-and-inter-step-communication) **[NEW]**
8. [Modular Architecture — mbt-core + Adapter Packages](#8-modular-architecture--mbt-core--adapter-packages)
9. [DRY — Three Levels of Reuse](#9-dry--three-levels-of-reuse)
10. [Orchestrator Integration — Manifest-Driven DAG Generation](#10-orchestrator-integration--manifest-driven-dag-generation)
11. [Data Validation and Quality Checks](#11-data-validation-and-quality-checks) **[NEW]**
12. [Secrets Management](#12-secrets-management) **[NEW]**
13. [Observability and Monitoring](#13-observability-and-monitoring) **[NEW]**
14. [Testing Strategy](#14-testing-strategy) **[NEW]**
15. [CLI Commands](#15-cli-commands)
16. [Project Structure](#16-project-structure)
17. [Comparison with Existing Tools](#17-comparison-with-existing-tools)
18. [Artifact Tracking and Run Results](#18-artifact-tracking-and-run-results)
19. [Known Limitations and Non-Goals](#19-known-limitations-and-non-goals) **[NEW]**
20. [Failure Modes and Threat Model](#20-failure-modes-and-threat-model) **[NEW]**
21. [Conclusion](#21-conclusion)

---

## 1. Design Philosophy

### The DS writes YAML, not code

The fundamental insight driving MBT's architecture comes from observing what data scientists actually produce. They don't think in terms of DAG tasks and artifact edges. They think in terms of:

- **What data** — which table, what time window, which columns
- **What preprocessing** — normalization, feature selection, encoding
- **What model** — which framework, which hyperparameters
- **What evaluation** — which metrics, how many CV folds, what plots

MBT embraces this mental model. A pipeline YAML is a declaration of intent. The framework's compiler reads the declaration, infers which steps are needed, assembles a DAG, and generates an execution plan. If the DS sets `feature_selection.enabled: false`, that step simply doesn't exist in the compiled DAG.

### Two-layer architecture: declaration vs execution

```
┌──────────────────────────────────────────────────────────────┐
│  LAYER 1: DECLARATION (Data Scientist)                       │
│                                                              │
│  churn_training_v1.yaml                                      │
│  "I want a random forest on this table with these splits"    │
│  No tasks, no DAG, no inputs/outputs, no executors           │
└──────────────────────┬───────────────────────────────────────┘
                       │  mbt compile
                       ▼
┌──────────────────────────────────────────────────────────────┐
│  LAYER 2: EXECUTION (Framework + Data Engineering)           │
│                                                              │
│  manifest.json                                               │
│  Resolved DAG of concrete tasks with executors, resources,   │
│  inputs, outputs — ready for local, K8s, or any orchestrator │
└──────────────────────────────────────────────────────────────┘
```

### Separation of concerns

| Role | Owns | Doesn't touch |
|------|------|---------------|
| Data Scientist | `pipelines/*.yaml`, `lib/` custom transforms | `profiles.yaml`, orchestrator DAGs, infrastructure |
| Data Engineer | `profiles.yaml`, orchestrator config, CI/CD | Pipeline YAML, model hyperparameters |
| ML Platform Engineer | MBT plugins, executor backends, orchestrator plugins | Business pipeline definitions |

### Compile before execute

Every `mbt run` first compiles the YAML into a `manifest.json` — a complete, serializable execution plan. This catches configuration errors, missing plugins, invalid parameter combinations, and unreachable backends before any computation starts. Framework plugins contribute to this via `validate_config()`, which performs dry-run validation of framework-specific parameters (e.g., verifying that H2O AutoML's `sort_metric` is valid for the given `problem_type`). This mirrors dbt's compile-then-execute model, and Ludwig's config validation that prevents runtime failures.

### AutoML-first: rationale and when it is not appropriate **[CHANGED]**

MBT defaults to AutoML because it eliminates the single largest time sink in tabular ML — hyperparameter tuning — while producing models that match or exceed manually tuned alternatives in most business contexts. However, AutoML-first is *not* appropriate when:

- **Regulatory requirements** demand a fully reproducible, manually specified model (e.g., credit scoring under SR 11-7).
- **Research workflows** need fine-grained control over model architecture.
- **Inference latency constraints** require a specific model type (e.g., a single lightweight decision tree).

In these cases, data scientists should use explicit framework adapters (`mbt-xgboost`, `mbt-sklearn`) with pinned hyperparameters and set `seed` values for reproducibility. The adapter system supports both modes equally.

---

## 2. Pipeline YAML Schema — The DS Interface

Each pipeline is a single YAML file. One YAML = one pipeline = one deployable unit. The DS defines *what* to build; the framework determines *how* to build it.

### 2.1 Schema Versioning **[NEW]**

Every pipeline YAML must declare a `schema_version` at the top level. This enables the compiler to apply the correct parsing rules and migration logic as the schema evolves. When `mbt compile` encounters a YAML with an older schema version, it emits a warning with migration instructions. Breaking changes to the YAML schema require a major version bump and a documented migration path.

```yaml
schema_version: 1   # required — controls YAML parsing rules

project:
  name: churn_training_v1
  experiment_name: churn_prediction
  owner: customer_analytics_team
  problem_type: binary_classification
  tags: [churn, telecom, production, training]
```

### 2.2 Training Pipeline

```yaml
# pipelines/churn_training_v1.yaml
schema_version: 1

# ============================================================================
# Project Metadata
# ============================================================================
project:
  name: churn_training_v1
  experiment_name: churn_prediction
  owner: customer_analytics_team
  problem_type: binary_classification
  tags: [churn, telecom, production, training]

# ============================================================================
# Deployment Settings
# ============================================================================
deployment:
  mode: batch                          # batch | realtime
  cadence: monthly                     # informational, used in docs and scheduling

# ============================================================================
# Training Configuration
# ============================================================================
training:
  # --------------------------------------------------------------------------
  # Data Source
  # --------------------------------------------------------------------------
  data_source:
    label_table: customer_churn_features
    feature_tables:                    # [CHANGED] explicit join specification
      - table: customer_churn_features
      - table: customer_demographics
        join_key: customer_id          # [NEW] join key
        join_type: left                # [NEW] inner | left | right | full
    data_windows:                      # [CHANGED] unified temporal model
      label_window:
        offset: -1
        unit: month                    # day | week | month | quarter
      train_window:
        duration: 12
        unit: month
      feature_window:                  # optional: different granularity
        duration: 90
        unit: day
    min_rows: 1000

    # With execution_date=2026-03-01:
    # Label window: February 2026 (offset -1 month)
    # Test: February 2026 data (where labels exist)
    # Train: February 2025 → January 2026 (12 months before label month)

  # --------------------------------------------------------------------------
  # Schema
  # --------------------------------------------------------------------------
  schema:
    target:
      label_column: churned
      classes: [0, 1]
      positive_class: 1
    identifiers:
      primary_key: customer_id
      partition_key: snapshot_date
    ignored_columns:
      - customer_name
      - email
      - phone_number

  # --------------------------------------------------------------------------
  # Data Validation [NEW]
  # --------------------------------------------------------------------------
  validation:
    on_failure: fail                   # pipeline-level default: fail | warn | skip_row
    checks:
      - type: null_threshold
        columns: [churned, customer_id]
        max_null_pct: 0.0
      - type: null_threshold
        columns: "*"                   # all non-ignored columns
        max_null_pct: 0.05
      - type: value_range
        column: monthly_charges
        min: 0
        max: 500
      - type: expected_columns
        columns: [customer_id, churned, tenure, monthly_charges]
      - type: custom
        function: lib.custom_validators.check_label_distribution

  # --------------------------------------------------------------------------
  # Feature Transformations
  # --------------------------------------------------------------------------
  transformations:
    normalization:
      enabled: true
      method: standard_scaler          # standard_scaler | min_max_scaler | robust_scaler
    encoding:
      enabled: false
    custom_transforms:                 # hook into lib/ functions
      - lib.custom_transforms.compute_rfm
      - lib.custom_transforms.create_interaction_features

  # --------------------------------------------------------------------------
  # Feature Selection
  # --------------------------------------------------------------------------
  feature_selection:
    enabled: true
    methods:
      - name: lgbm_importance
        threshold: 0.95                # keep top 95% cumulative importance

  # --------------------------------------------------------------------------
  # Model Training
  # --------------------------------------------------------------------------
  model_training:
    framework: h2o_automl              # h2o_automl | xgboost | lightgbm | sklearn
    config:                            # pass-through to framework plugin
      max_runtime_secs: 3600
      max_models: 20
      sort_metric: AUC
      seed: 42
    resources:                         # [CHANGED] moved outside config block
      cpu: "8"
      memory: "32Gi"

  # --------------------------------------------------------------------------
  # Evaluation
  # --------------------------------------------------------------------------
  evaluation:
    primary_metric: roc_auc
    additional_metrics: [accuracy, f1, precision, recall]
    generate_plots: true
    custom_metrics:
      - lib.custom_metrics.business_value_score
```

### 2.3 Data Window Specification **[CHANGED]**

The original blueprint used inconsistent temporal abstractions between training and serving pipelines. The revised schema introduces a unified `data_windows` model with explicit `unit` fields, supporting `day`, `week`, `month`, and `quarter` granularities. Both training and serving pipelines now use the same structure:

```yaml
# Unified data_windows schema (training or serving)
data_windows:
  label_window:
    offset: -1                         # relative to execution_date
    unit: month                        # day | week | month | quarter
  train_window:
    duration: 12
    unit: month
  feature_window:                      # optional: different granularity for features
    duration: 90
    unit: day
```

### 2.4 Multi-Table Join Specification **[NEW]**

The original schema accepted a list of `feature_tables` without specifying how they would be joined. The revised schema requires explicit join configuration when multiple tables are referenced. The first table is treated as the base (left side). Subsequent tables must specify `join_key` and `join_type`. The compiler validates that join keys exist in both tables at compile time.

```yaml
data_source:
  label_table: customer_churn_features
  feature_tables:
    - table: customer_churn_features             # base table
    - table: customer_demographics
      join_key: customer_id                      # single key
      join_type: left                            # inner | left | right | full
    - table: customer_transactions_agg
      join_key: [customer_id, snapshot_date]     # composite key
      join_type: inner
      fan_out_check: true                        # fail if join produces > 1x rows
```

### 2.5 Resources Block Clarification **[CHANGED]**

In the original blueprint, the `resources` block was nested inside `model_training.config`, which is documented as a pass-through to the framework plugin. This was contradictory — framework plugins don't manage infrastructure resources. The revised schema moves `resources` to be a sibling of `config` within `model_training`, and also supports per-step resource overrides at the step level:

```yaml
model_training:
  framework: h2o_automl
  config:                              # pass-through to framework plugin
    max_runtime_secs: 3600
    sort_metric: AUC
  resources:                           # infrastructure (NOT passed to plugin)
    cpu: "8"
    memory: "32Gi"
```

### 2.6 Serving Pipeline

```yaml
# pipelines/churn_serving_v1.yaml
schema_version: 1

project:
  name: churn_serving_v1
  experiment_name: churn_prediction
  owner: customer_analytics_team
  problem_type: binary_classification
  tags: [churn, telecom, production, serving]

deployment:
  mode: batch
  cadence: daily

serving:
  # --------------------------------------------------------------------------
  # Model Source — resolved from a training run
  # --------------------------------------------------------------------------
  model_source:
    registry: mlflow
    run_id: "{{ var('run_id') }}"              # passed at runtime, or pinned
    artifact_snapshot: true                     # [NEW] freeze artifacts locally
    fallback_run_id: "{{ var('fallback') }}"   # [NEW] fallback if primary fails

  # --------------------------------------------------------------------------
  # Scoring Data Source
  # --------------------------------------------------------------------------
  data_source:
    scoring_table: active_customers
    data_windows:
      lookback_window:
        offset: -1
        unit: day                              # [CHANGED] uses unified model

  # --------------------------------------------------------------------------
  # Output
  # --------------------------------------------------------------------------
  output:
    destination: snowflake
    table: ml_predictions.churn_scores
    columns:
      prediction: churn_probability
      label: churn_predicted
      threshold: 0.5
```

### 2.7 Serving Pipeline Resilience **[NEW]**

The original design resolved everything from a single `run_id` at runtime, creating a hard dependency on the model registry being available and the specific run's artifacts being intact. The revised design adds two mechanisms:

**Artifact snapshots:** When `artifact_snapshot: true` is set, `mbt compile` fetches all training artifacts (model, scaler, encoder, feature selector, column order) and bundles them into the compiled manifest directory. The serving pipeline can run entirely from the snapshot without contacting the registry at runtime. This eliminates the runtime dependency.

**Fallback run_id:** If the primary `run_id` fails to resolve (deleted run, registry outage), MBT falls back to `fallback_run_id` with a warning logged. This prevents production serving pipelines from failing due to registry issues while ensuring the team is alerted.

### 2.8 Key Schema Design Decisions

Each YAML section is optional and toggleable. If `transformations` is omitted, no preprocessing step is generated. If `feature_selection.enabled: false`, that step is skipped. The compiler builds the DAG dynamically based on which sections are present and enabled.

The `config` block under `model_training` is a pass-through with compile-time validation. Whatever the DS puts in `config` gets forwarded directly to the framework plugin. MBT's core schema doesn't validate framework-specific hyperparameters — instead, the framework plugin's `validate_config()` method is called at compile time to catch typos and invalid parameter combinations before any computation starts. For example, the H2O AutoML plugin validates that `max_runtime_secs` is positive and `sort_metric` is a recognized value.

**Problem type drives behavior globally.** Setting `problem_type: binary_classification` tells the evaluation plugin to compute ROC AUC, precision, recall, confusion matrix, and the data splitter to use stratified sampling. Setting `problem_type: regression` switches to RMSE, MAE, R², and standard sampling. The DS doesn't configure this per-step — it flows from the single declaration.

**Serving pipelines resolve everything from `run_id`.** Instead of duplicating schema, transformation, and feature definitions from the training pipeline, the serving pipeline references a single MLflow `run_id`. At compile/execution time, MBT fetches the run's metadata and reconstructs the full preprocessing chain — scaler, encoder, feature selector, column order — automatically. This eliminates schema drift between training and serving entirely, because there is no separate schema to go stale. The DE controls which model gets deployed by setting the `run_id`.

---

## 3. Pipeline Composition and Reuse **[NEW]**

Real-world ML projects frequently have multiple models trained on the same feature table with shared preprocessing logic. Without a composition mechanism, teams with 15 models on the same data would have massive duplication across pipeline YAMLs. MBT addresses this with three composition mechanisms.

### 3.1 Base Pipelines (Template Inheritance)

A pipeline YAML can declare a `base_pipeline` that it extends. The child pipeline inherits all sections from the parent and can override any section or individual field. The merge follows dictionary-deep-merge semantics: scalar values are replaced, lists are replaced (not appended), and dictionaries are merged recursively.

```yaml
# pipelines/_base_churn.yaml (convention: underscore prefix for base pipelines)
schema_version: 1

project:
  experiment_name: churn_prediction
  owner: customer_analytics_team
  problem_type: binary_classification
  tags: [churn, telecom]

training:
  data_source:
    label_table: customer_churn_features
    feature_tables:
      - table: customer_churn_features
    data_windows:
      label_window: { offset: -1, unit: month }
      train_window: { duration: 12, unit: month }
    min_rows: 1000
  schema:
    target:
      label_column: churned
      classes: [0, 1]
      positive_class: 1
    identifiers:
      primary_key: customer_id
      partition_key: snapshot_date
    ignored_columns: [customer_name, email, phone_number]
  transformations:
    normalization: { enabled: true, method: standard_scaler }
  evaluation:
    primary_metric: roc_auc
    additional_metrics: [accuracy, f1, precision, recall]
    generate_plots: true
```

```yaml
# pipelines/churn_h2o_v1.yaml — inherits from base, uses H2O AutoML
schema_version: 1
base_pipeline: _base_churn                # inherits all sections

project:
  name: churn_h2o_v1                      # override: unique name required
  tags: [churn, telecom, h2o]             # override: replaces parent tags

training:
  model_training:
    framework: h2o_automl
    config: { max_runtime_secs: 3600, max_models: 20, sort_metric: AUC }
```

```yaml
# pipelines/churn_xgb_v1.yaml — same base, different framework
schema_version: 1
base_pipeline: _base_churn

project:
  name: churn_xgb_v1
  tags: [churn, telecom, xgboost]

training:
  model_training:
    framework: xgboost
    config: { n_estimators: 500, max_depth: 6, learning_rate: 0.1 }
```

### 3.2 YAML Fragments (Shared Sections)

For sharing individual sections across unrelated pipelines (e.g., a common data source or validation config), MBT supports `!include` directives that reference YAML fragment files in an `includes/` directory:

```yaml
# includes/telecom_data_source.yaml
label_table: customer_churn_features
feature_tables:
  - table: customer_churn_features
  - table: customer_demographics
    join_key: customer_id
    join_type: left

# pipelines/churn_training_v1.yaml
training:
  data_source: !include includes/telecom_data_source.yaml
  schema: !include includes/telecom_schema.yaml
  model_training:
    framework: h2o_automl
    config: { max_runtime_secs: 3600 }
```

### 3.3 Composition Rules

Base pipeline resolution is single-level only (no chained inheritance) to keep behavior predictable. Every concrete pipeline must have a unique `project.name`. The compiler resolves base pipelines and includes during the first phase of compilation, before any validation or DAG assembly. `mbt validate --show-resolved` shows the fully resolved YAML so the DS can verify what they'll get.

---

## 4. Profiles — The DE Interface

`profiles.yaml` is owned by data engineers and defines where and how pipelines execute. It mirrors dbt's `profiles.yaml`: target environments, connection details, resource defaults, and infrastructure configuration. Pipeline YAMLs never contain infrastructure details.

```yaml
# profiles.yaml (project root or ~/.mbt/profiles.yaml)
my-ml-project:
  target: dev                              # default target

  # --------------------------------------------------------------------------
  # MLflow Configuration (shared across targets or overridden per target)
  # --------------------------------------------------------------------------
  mlflow:
    tracking_uri: "{{ env_var('MLFLOW_TRACKING_URI') }}"
    registry_uri: "{{ env_var('MLFLOW_REGISTRY_URI') }}"

  # --------------------------------------------------------------------------
  # Secrets Provider [NEW]
  # --------------------------------------------------------------------------
  secrets:
    provider: env                          # env | vault | aws_secrets | k8s_secrets
    config: {}                             # provider-specific config

  # --------------------------------------------------------------------------
  # Observability [NEW]
  # --------------------------------------------------------------------------
  observability:
    metrics:
      type: statsd                         # statsd | prometheus | none
      config:
        host: "{{ env_var('STATSD_HOST') }}"
        port: 8125
        prefix: mbt.pipelines
    logging:
      format: json                         # json | text
      level: INFO

  # --------------------------------------------------------------------------
  # Orchestrator Configuration
  # --------------------------------------------------------------------------
  orchestrator:
    type: airflow                          # airflow | prefect | dagster | argo
    config:
      dag_directory: /opt/airflow/dags
      default_args:
        owner: data-engineering
        retries: 2
        retry_delay_minutes: 5

  # --------------------------------------------------------------------------
  # Target Environments
  # --------------------------------------------------------------------------
  outputs:
    dev:
      executor:
        type: local
        config: {}
      storage:
        type: local
        config:
          base_path: ./local_artifacts
      data_connector:
        type: local_file
        config:
          data_path: ./sample_data
      resources:
        cpu: "2"
        memory: "4Gi"
      mlflow:
        tracking_uri: "sqlite:///mlruns.db"
        experiment_suffix: "_dev"

    staging:
      executor:
        type: kubernetes
        config:
          kubeconfig: "{{ env_var('KUBECONFIG') }}"
          namespace: ml-staging
          service_account: pipeline-runner
          image: "myorg/mbt-runner:latest"
      storage:
        type: s3
        config:
          bucket: myorg-ml-staging
          prefix: pipelines
          region: us-east-1
      data_connector:
        type: snowflake
        config:
          account: "{{ secret('SNOWFLAKE_ACCOUNT') }}"    # [NEW] secrets ref
          warehouse: ML_WH_SMALL
          database: ANALYTICS
          schema: ML_FEATURES
          role: ML_READER
      resources:
        cpu: "4"
        memory: "8Gi"

    prod:
      executor:
        type: kubernetes
        config:
          kubeconfig: "{{ env_var('KUBECONFIG') }}"
          namespace: ml-production
          service_account: pipeline-runner-prod
          image: "myorg/mbt-runner:v2.1-stable"
          node_selector:
            gpu: "true"
      storage:
        type: s3
        config:
          bucket: myorg-ml-production
          prefix: pipelines
          region: us-east-1
          encryption: AES256
      data_connector:
        type: snowflake
        config:
          account: "{{ secret('SNOWFLAKE_ACCOUNT') }}"
          warehouse: ML_WH_LARGE
          database: ANALYTICS
          schema: ML_FEATURES
          role: ML_PRODUCTION
      resources:
        cpu: "8"
        memory: "16Gi"
      orchestrator:                        # per-target override
        type: prefect
        config:
          work_pool: ml-production
          infrastructure: kubernetes
```

Resolution order: CLI flag (`--target prod`) → `profiles.yaml` default → `MBT_TARGET` environment variable. Environment variables are injected via `{{ env_var('KEY') }}` with Jinja2. Secrets are injected via `{{ secret('KEY') }}` through the configured secrets provider.

---

## 5. Compilation — From YAML Declaration to Executable DAG

The compiler is the heart of MBT. It reads a pipeline YAML, resolves base pipelines and includes, inspects which sections are present and enabled, generates the appropriate steps, wires them into a DAG, merges profile configuration, and outputs a `manifest.json`.

### 5.1 Compilation Phases **[NEW]**

Compilation proceeds in five ordered phases:

**Phase 1 — Resolution:** Resolve `base_pipeline` inheritance and `!include` directives. Produce a fully merged, standalone YAML document.

**Phase 2 — Schema validation:** Validate the resolved YAML against the Pydantic schema for the declared `schema_version`. Check required fields, type correctness, and structural constraints.

**Phase 3 — Plugin validation:** Call `validate_config()` on each referenced adapter plugin. This performs framework-specific dry-run validation (e.g., H2O AutoML checks `sort_metric` validity).

**Phase 4 — DAG assembly:** Walk the step registry, evaluate conditions, build the step graph, and wire inputs/outputs. Perform topological sort and cycle detection.

**Phase 5 — Manifest generation:** Merge profile/target configuration with the compiled DAG. Serialize to `manifest.json` with full executor, resource, and connection details.

### 5.2 How YAML Sections Map to Generated Steps

The compiler maintains a step registry — an ordered list of potential steps for each pipeline type (training, serving). Each step has a condition that determines whether it's included in the DAG based on the YAML configuration:

| YAML section | Generated step | Condition | Inputs → Outputs |
|---|---|---|---|
| `training.data_source` | `load_data` | Always (required) | → `raw_data` |
| `training.data_source` (multi-table) | `join_tables` **[NEW]** | Multiple `feature_tables` | `raw_tables` → `joined_data` |
| `training.validation` **[NEW]** | `validate_data` | `validation` section present | `data` → `validated_data` |
| `training.data_source.data_windows` | `split_data` | Always (required) | `raw_data` → `train_set`, `test_set` |
| `training.schema` | — (metadata consumed by `load_data` and `split_data`) | — | — |
| `training.transformations.normalization` | `normalize` | `normalization.enabled: true` | `train_set`, `test_set` → `norm_train`, `norm_test`, `scaler` |
| `training.transformations.encoding` | `encode` | `encoding.enabled: true` | `data` → `encoded_data`, `encoder` |
| `training.feature_selection` | `select_features` | `feature_selection.enabled: true` | `data` → `selected_data`, `selector` |
| `training.model_training` | `train_model` | Always (required) | `data` → `model`, `train_metrics` |
| `training.evaluation` | `evaluate` | Always (required) | `model`, `test_data` → `eval_metrics`, `plots` |
| (implicit) | `log_to_registry` | Always | all artifacts → `run_id` |

Note: Cross-validation is not a separate step. When using H2O AutoML (the default framework), internal cross-validation and hyperparameter search are handled automatically by the framework. The `evaluate` step runs on the held-out test set to validate AutoML's best model. Framework plugins that require explicit CV (e.g., a community sklearn plugin) can implement it within their `train()` method.

For a serving pipeline:

| YAML section | Generated step | Condition |
|---|---|---|
| `serving.model_source` | `load_model` | Always (required) |
| `serving.data_source` | `load_scoring_data` | Always (required) |
| (resolved from `run_id`) | `apply_transforms` | If training run had transformations |
| (implicit) | `predict` | Always |
| `serving.output` | `publish_predictions` | Always (required) |

The serving pipeline resolves its transform chain from the training run's `run_id` — the compiler (or runner) fetches the run's metadata from MLflow and loads the exact preprocessing artifacts (scaler, encoder, feature selector) that were produced during training. No schema section is needed in the serving YAML.

### 5.3 DAG Assembly — Linear and Branching **[CHANGED]**

The compiler builds the DAG by walking the step registry in order and connecting each step's outputs to the next step's inputs. Most ML training pipelines produce linear DAGs, but the compiler supports branching and joining for patterns that require it:

```
Linear DAG (most common):
  load_data ──► split_data ──► normalize ──► train_model ──► evaluate ──► log

With all optional steps enabled:
  load_data ──► join_tables ──► validate ──► split_data ──► normalize ──►
  select_features ──► train_model ──► evaluate ──► log_to_mlflow

Branching DAG (multi-source join):
  load_features_a ─┐
  load_features_b ─┼──► join_tables ──► validate ──► split ──► train ──► evaluate
  load_features_c ─┘

Parallel evaluation DAG (champion/challenger) [future]:
  train_model_a ─┐
                 ├──► compare_models ──► select_champion ──► log
  train_model_b ─┘
```

The compiler uses Python's `graphlib.TopologicalSorter` for ordering and cycle detection. The `execution_batches` field in the manifest groups steps that can run in parallel — steps in the same batch have no dependencies on each other. For linear pipelines, each batch contains a single step. For branching pipelines, independent branches appear in the same batch.

### 5.4 Manifest Output

The compiler produces `target/<pipeline_name>/manifest.json`:

```json
{
  "metadata": {
    "mbt_version": "0.2.0",
    "schema_version": 1,
    "generated_at": "2026-02-13T10:30:00Z",
    "pipeline_name": "churn_training_v1",
    "pipeline_type": "training",
    "target": "prod",
    "problem_type": "binary_classification",
    "artifact_store": {
      "type": "s3",
      "config": { "bucket": "myorg-ml-production", "prefix": "artifacts" }
    }
  },
  "steps": {
    "load_data": {
      "plugin": "mbt.steps.load_data:LoadDataStep",
      "config": {
        "label_table": "customer_churn_features",
        "feature_tables": ["customer_churn_features"],
        "connector": "snowflake",
        "connector_config": { "warehouse": "ML_WH_LARGE", "..." : "..." }
      },
      "resources": { "cpu": "2", "memory": "4Gi" },
      "inputs": [],
      "outputs": ["raw_data"],
      "depends_on": [],
      "idempotent": false
    },
    "split_data": {
      "plugin": "mbt.steps.split_data:SplitDataStep",
      "config": {
        "label_window": { "offset": -1, "unit": "month" },
        "train_window": { "duration": 12, "unit": "month" },
        "min_rows": 1000,
        "stratify_column": "churned"
      },
      "inputs": ["raw_data"],
      "outputs": ["train_set", "test_set"],
      "depends_on": ["load_data"],
      "idempotent": true
    },
    "normalize": {
      "plugin": "mbt.steps.normalize:NormalizeStep",
      "config": { "method": "standard_scaler" },
      "inputs": ["train_set", "test_set"],
      "outputs": ["norm_train", "norm_test", "scaler_artifact"],
      "depends_on": ["split_data"],
      "idempotent": true
    },
    "select_features": {
      "plugin": "mbt.steps.feature_selection:FeatureSelectionStep",
      "config": {
        "methods": [{ "name": "lgbm_importance", "threshold": 0.95 }]
      },
      "inputs": ["norm_train", "norm_test"],
      "outputs": ["selected_train", "selected_test", "selector_artifact"],
      "depends_on": ["normalize"],
      "idempotent": true
    },
    "train_model": {
      "plugin": "mbt_h2o.framework:H2OAutoMLFramework",
      "config": {
        "max_runtime_secs": 3600,
        "max_models": 20,
        "sort_metric": "AUC",
        "seed": 42
      },
      "resources": { "cpu": "8", "memory": "32Gi" },
      "inputs": ["selected_train"],
      "outputs": ["model", "train_metrics"],
      "depends_on": ["select_features"],
      "idempotent": false
    },
    "evaluate": {
      "plugin": "mbt.steps.evaluate:EvaluateStep",
      "config": {
        "primary_metric": "roc_auc",
        "additional_metrics": ["accuracy", "f1", "precision", "recall"],
        "generate_plots": true,
        "problem_type": "binary_classification"
      },
      "inputs": ["model", "selected_test"],
      "outputs": ["eval_metrics", "eval_plots"],
      "depends_on": ["train_model", "select_features"],
      "idempotent": true
    },
    "log_to_mlflow": {
      "plugin": "mbt.steps.log_run:LogRunStep",
      "config": {
        "experiment_name": "churn_prediction",
        "tracking_uri": "https://mlflow.myorg.com"
      },
      "inputs": ["model", "eval_metrics", "eval_plots",
                  "scaler_artifact", "selector_artifact"],
      "outputs": ["mlflow_run_id"],
      "depends_on": ["evaluate"]
    }
  },
  "dag": {
    "parent_map": {
      "load_data": [],
      "split_data": ["load_data"],
      "normalize": ["split_data"],
      "select_features": ["normalize"],
      "train_model": ["select_features"],
      "evaluate": ["train_model", "select_features"],
      "log_to_mlflow": ["evaluate"]
    },
    "execution_batches": [
      ["load_data"],
      ["split_data"],
      ["normalize"],
      ["select_features"],
      ["train_model"],
      ["evaluate"],
      ["log_to_mlflow"]
    ]
  }
}
```

---

## 6. Data Abstraction Layer **[NEW]**

The original design assumed pandas DataFrames as the universal data interchange format between steps and in adapter contracts. This creates a scaling ceiling — pandas loads all data into memory, which fails for large datasets — and forces framework adapters to perform wasteful conversions (e.g., pandas → H2OFrame → pandas).

### 6.1 The MBTFrame Protocol

MBT introduces a lightweight data protocol that adapters can implement. The core framework passes data between steps using an `MBTFrame` wrapper that supports lazy evaluation and format negotiation. Adapters declare which formats they can consume and produce, and the runner handles conversions only when necessary.

```python
# mbt-core: mbt/contracts/data.py
from typing import Protocol, runtime_checkable
import pandas as pd

@runtime_checkable
class MBTFrame(Protocol):
    """Protocol for data interchange between steps."""

    def to_pandas(self) -> pd.DataFrame:
        """Convert to pandas (always supported, may be expensive)."""
        ...

    def num_rows(self) -> int:
        """Return row count without materializing."""
        ...

    def columns(self) -> list[str]:
        """Return column names without materializing."""
        ...

    def schema(self) -> dict[str, str]:
        """Return column name → type mapping."""
        ...


class PandasFrame(MBTFrame):
    """Default wrapper for pandas DataFrames."""
    def __init__(self, df: pd.DataFrame):
        self._df = df

    def to_pandas(self) -> pd.DataFrame:
        return self._df

    def num_rows(self) -> int:
        return len(self._df)

    def columns(self) -> list[str]:
        return list(self._df.columns)

    def schema(self) -> dict[str, str]:
        return {col: str(dtype) for col, dtype in self._df.dtypes.items()}
```

This approach is backwards-compatible: adapters that only work with pandas continue to call `frame.to_pandas()` and work exactly as before. Adapters that support Arrow, Spark, or native formats can check the underlying type and avoid conversion overhead.

### 6.2 Format Negotiation

When the runner passes data between steps, it checks whether the producing step's output format matches the consuming step's preferred input format. If they match (e.g., both prefer Arrow), no conversion occurs. If they don't match, the runner uses the `MBTFrame` protocol to convert. This is transparent to the DS — they never configure data formats.

Framework adapters declare their supported formats:

```python
class FrameworkPlugin(ABC):
    def supported_formats(self) -> list[str]:
        """Return supported input data formats: ['pandas', 'arrow', 'h2o', 'spark']"""
        return ["pandas"]  # default: pandas only
```

---

## 7. Artifact Flow and Inter-Step Communication **[NEW]**

The original blueprint defined step `inputs` and `outputs` as string names in the manifest but did not specify how artifacts actually flow between steps. This is especially critical when steps run as separate Kubernetes pods that do not share a filesystem.

### 7.1 Artifact Store Contract

Each target in `profiles.yaml` configures a `storage` backend (local filesystem, S3, GCS, etc.). The runner uses this backend to persist step outputs and load step inputs. The contract is:

```python
# mbt-core: mbt/contracts/storage.py
class StoragePlugin(ABC):

    @abstractmethod
    def put(self, artifact_name: str, data: bytes, run_id: str,
            step_name: str, metadata: dict | None = None) -> str:
        """Store an artifact. Returns a URI (s3://..., /path/..., etc.)."""
        ...

    @abstractmethod
    def get(self, artifact_uri: str) -> bytes:
        """Retrieve an artifact by URI."""
        ...

    @abstractmethod
    def exists(self, artifact_uri: str) -> bool:
        """Check if an artifact exists (used by retry logic)."""
        ...

    @abstractmethod
    def list_artifacts(self, run_id: str, step_name: str) -> list[str]:
        """List all artifacts for a step in a run."""
        ...
```

### 7.2 How the Runner Passes Artifacts

```
Step Execution Lifecycle:
1. Runner reads step definition from manifest
2. For each input artifact name:
   a. Look up the artifact URI from the producing step's output registry
   b. Call storage.get(uri) to retrieve the serialized artifact
   c. Deserialize into the expected Python object
3. Execute the step's run() method with deserialized inputs
4. For each output artifact:
   a. Serialize the object (pickle, joblib, parquet, or custom)
   b. Call storage.put(name, data, run_id, step_name) to persist
   c. Record the returned URI in the run's artifact registry
5. The artifact registry is itself persisted to storage after each step
```

This means that in a Kubernetes execution environment, each pod only needs access to the configured storage backend (e.g., S3). The runner in each pod retrieves its inputs from S3, executes the step, and writes its outputs back to S3. No shared filesystem is required.

### 7.3 Serialization Strategy

| Artifact Type | Serialization Format | Notes |
|---|---|---|
| DataFrames (`MBTFrame`) | Parquet | Preserves types, efficient, cross-language |
| Trained models | Framework-specific | H2O binary, sklearn joblib, XGB native |
| Scalers, encoders | Joblib | Standard sklearn-compatible serialization |
| Metrics (dict) | JSON | Human-readable, small |
| Plots | PNG / HTML | Matplotlib → PNG, Plotly → HTML |
| Feature selectors | JSON + Joblib | Config (JSON) + fitted object (Joblib) |

### 7.4 Idempotency and Retry

Each step in the manifest declares an `idempotent` flag. Steps marked as idempotent (e.g., `normalize`, `evaluate`) can be safely re-run with the same inputs and will produce the same outputs. Non-idempotent steps (e.g., `load_data` against a mutable source) will be re-run from scratch on retry.

When `mbt retry` is invoked, the runner checks the artifact store for each step's outputs. If a step's outputs exist and the step is idempotent, it is skipped. Otherwise, it is re-executed. This provides safe, efficient retry behavior. A `--force` flag overrides this and re-runs all steps.

---

## 8. Modular Architecture — mbt-core + Adapter Packages

MBT follows the same modular pattern as dbt: a thin core package provides the CLI, compiler, and plugin registry, while every concrete integration is a separate pip-installable adapter. This means `mbt-core` has zero infrastructure dependencies — it doesn't know what Snowflake is, what H2O does, or how Airflow works. Users install only the adapters they need.

### 8.1 Package Taxonomy

```
mbt-core                  # CLI, compiler, manifest, DAG engine, plugin registry
                          # Dependencies: pyyaml, jinja2, pydantic, graphlib
                          # Zero ML/infra/cloud dependencies

Adapter Packages:
──────────────────────────────────────────────────────────────
AutoML Frameworks         # How to train and predict
  mbt-h2o                 # H2O AutoML (depends on: h2o)
  mbt-sklearn             # scikit-learn (depends on: scikit-learn)
  mbt-sagemaker           # SageMaker Autopilot (depends on: boto3, sagemaker)
  mbt-xgboost             # XGBoost (depends on: xgboost)
  mbt-lightgbm            # LightGBM (depends on: lightgbm)

Data Warehouses           # Where data lives
  mbt-snowflake           # Snowflake (depends on: snowflake-connector-python)
  mbt-postgres            # PostgreSQL (depends on: psycopg2)
  mbt-bigquery            # BigQuery (depends on: google-cloud-bigquery)
  mbt-databricks          # Databricks (depends on: databricks-sql-connector)

Orchestrators             # What schedules pipelines
  mbt-airflow             # Apache Airflow (depends on: apache-airflow)
  mbt-prefect             # Prefect (depends on: prefect)
  mbt-dagster             # Dagster (depends on: dagster)

Model Registries          # Where models are tracked
  mbt-mlflow              # MLflow (depends on: mlflow)
  mbt-wandb               # Weights & Biases (depends on: wandb)
  mbt-sagemaker-registry  # SageMaker Registry (depends on: boto3, sagemaker)

Executors                 # Where steps run
  mbt-docker              # Docker containers (depends on: docker)
  mbt-kubernetes          # Kubernetes pods (depends on: kubernetes)
  mbt-ray                 # Ray clusters (depends on: ray)

Storage                   # Where artifacts persist
  mbt-s3                  # AWS S3 (depends on: boto3)
  mbt-gcs                 # Google Cloud Storage (depends on: google-cloud-storage)

Secrets [NEW]             # Where credentials live
  mbt-vault               # HashiCorp Vault (depends on: hvac)
  mbt-aws-secrets         # AWS Secrets Manager (depends on: boto3)
  mbt-k8s-secrets         # Kubernetes Secrets (depends on: kubernetes)
```

Minimal local install (DS working locally with sklearn and MLflow):

```bash
pip install mbt-core mbt-sklearn mbt-mlflow
```

Enterprise install (H2O + Snowflake + Airflow + MLflow + Kubernetes + S3):

```bash
pip install mbt-core mbt-h2o mbt-snowflake mbt-airflow mbt-mlflow mbt-kubernetes mbt-s3
```

AWS-native install (SageMaker everything):

```bash
pip install mbt-core mbt-sagemaker mbt-sagemaker-registry mbt-s3
```

### 8.2 What Lives in mbt-core

`mbt-core` provides:

- The CLI (`mbt init`, `mbt compile`, `mbt run`, `mbt validate`, `mbt test`, `mbt generate-dags`, etc.)
- The compiler (YAML → `manifest.json`, including base pipeline and `!include` resolution)
- The DAG engine (topological sort, execution batches)
- The runner (step execution lifecycle, artifact passing)
- The plugin registry (discovers adapters via `entry_points` at runtime)
- Abstract base classes for all adapter categories (with lifecycle hooks)
- The `MBTFrame` data protocol and `PandasFrame` default implementation
- Pydantic schemas for pipeline YAML validation (versioned)
- Config resolution (`profiles.yaml`, Jinja2 `env_var`, `secret`, config merging)
- A local executor (run steps as subprocesses — the only built-in executor)
- A local file connector (read from local parquet/csv — the only built-in connector)
- A local artifact store (save artifacts to disk — the only built-in storage)
- An environment variable secrets provider (the only built-in secrets backend)
- Test utilities for adapter authors (`MockMBTFrame`, `MockStoragePlugin`, etc.)

This means `mbt-core` alone supports a fully functional local development workflow. A DS can `pip install mbt-core mbt-h2o` and immediately train models against local files without any cloud dependencies.

### 8.3 Adapter Contracts — Abstract Base Classes **[CHANGED]**

Every adapter category has an ABC defined in `mbt-core`. Adapter packages implement these contracts and register via `entry_points`. The revised contracts include lifecycle hooks and use the `MBTFrame` data protocol:

```python
# mbt-core: mbt/contracts/framework.py
class FrameworkPlugin(ABC):
    """Interface for ML framework adapters (mbt-sklearn, mbt-h2o, etc.)."""

    # ── Lifecycle hooks [NEW] ─────────────────────────────────
    def setup(self, config: dict) -> None:
        """Initialize resources (e.g., h2o.init()). Called once per run."""
        pass  # default: no-op

    def teardown(self) -> None:
        """Clean up resources (e.g., h2o.cluster().shutdown()).
        Called after all steps complete."""
        pass  # default: no-op

    def health_check(self) -> bool:
        """Verify the framework is ready. Called by `mbt debug`."""
        return True  # default: always healthy

    # ── Capability declaration [NEW] ──────────────────────────
    def supported_formats(self) -> list[str]:
        """Return supported input data formats."""
        return ["pandas"]  # default: pandas only

    # ── Core contract ─────────────────────────────────────────
    @abstractmethod
    def validate_config(self, config: dict, problem_type: str) -> None:
        """Validate config at compile time. Raise ValueError on invalid params.

        Called during `mbt compile` before any execution starts.
        Adapters should perform dry-run instantiation or parameter checking
        to catch typos and invalid combinations early.
        """
        ...

    @abstractmethod
    def train(self, X_train: MBTFrame, y_train: MBTFrame, config: dict) -> Any:
        """Train a model and return the fitted model object."""
        ...

    @abstractmethod
    def predict(self, model: Any, X: MBTFrame) -> np.ndarray:
        """Generate predictions from a trained model."""
        ...

    @abstractmethod
    def predict_proba(self, model: Any, X: MBTFrame) -> np.ndarray:
        """Generate probability predictions (classification only)."""
        ...

    @abstractmethod
    def serialize(self, model: Any, path: str) -> None:
        """Save model to disk."""
        ...

    @abstractmethod
    def deserialize(self, path: str) -> Any:
        """Load model from disk."""
        ...

    @abstractmethod
    def get_feature_importance(self, model: Any) -> dict[str, float]:
        """Return feature importance scores."""
        ...
```

```python
# mbt-core: mbt/contracts/data_connector.py
class DataConnectorPlugin(ABC):
    """Interface for data warehouse adapters (mbt-snowflake, mbt-postgres, etc.)."""

    @abstractmethod
    def connect(self, config: dict) -> None:
        """Establish connection to the data source."""
        ...

    @abstractmethod
    def read_table(self, table: str, columns: list[str] | None,
                   date_range: tuple | None) -> MBTFrame:     # [CHANGED] returns MBTFrame
        """Read data from a table with optional column and date filtering."""
        ...

    @abstractmethod
    def write_table(self, df: MBTFrame, table: str,
                    mode: str = "overwrite") -> None:          # [CHANGED] accepts MBTFrame
        """Write predictions/results to a table."""
        ...

    @abstractmethod
    def validate_connection(self) -> bool:
        """Test the connection. Called by `mbt debug`."""
        ...
```

```python
# mbt-core: mbt/contracts/orchestrator.py
class OrchestratorPlugin(ABC):
    """Interface for orchestrator adapters (mbt-airflow, mbt-prefect, etc.)."""

    @abstractmethod
    def build(self, manifest: dict, schedule: str | None,
              target: str, **kwargs) -> Any:
        """Build an orchestrator-native DAG/flow from a compiled manifest."""
        ...

    @abstractmethod
    def generate_dag_file(self, manifest_path: str, output_path: str,
                          schedule: str | None, target: str, **kwargs) -> None:
        """Generate a static DAG/flow file that can be deployed to the orchestrator."""
        ...
```

```python
# mbt-core: mbt/contracts/model_registry.py
class ModelRegistryPlugin(ABC):
    """Interface for model registry adapters (mbt-mlflow, mbt-wandb, etc.)."""

    @abstractmethod
    def log_run(self, pipeline_name: str, metrics: dict, params: dict,
                artifacts: dict, tags: dict) -> str:
        """Log a training run. Returns a run_id."""
        ...

    @abstractmethod
    def load_run(self, run_id: str) -> dict:
        """Load run metadata (model, artifacts, params) by run_id."""
        ...

    @abstractmethod
    def load_model(self, run_id: str) -> Any:
        """Load a trained model artifact from a run."""
        ...

    @abstractmethod
    def load_artifacts(self, run_id: str) -> dict:
        """Load all pipeline artifacts (scaler, encoder, etc.) from a run."""
        ...
```

### 8.4 Adapter Versioning and Compatibility **[NEW]**

When `mbt-core` adds a new abstract method to an ABC, existing adapters would break. To prevent this, MBT uses the following compatibility strategy:

**New methods have default implementations.** Any method added after v0.1.0 must include a default implementation in the ABC (as shown with `setup()`, `teardown()`, `health_check()`, and `supported_formats()` above). This means existing adapters continue to work without changes.

**Core version pinning.** Each adapter declares a minimum `mbt-core` version in its `pyproject.toml` (`mbt-core>=0.1.0`). When an ABC gains a new *required* method (which should be extremely rare), the core version is bumped to the next major version and adapters must update their pin.

**Runtime capability detection.** The plugin registry checks which optional methods an adapter implements using `hasattr` and adjusts behavior accordingly. For example, if an adapter doesn't implement `supported_formats()`, the runner defaults to pandas interchange.

### 8.5 Adapter Implementation Example — mbt-h2o

Each adapter is a standalone Python package with its own `pyproject.toml`, dependencies, and `entry_points` registration.

```python
# mbt-h2o/src/mbt_h2o/framework.py
import h2o
from h2o.automl import H2OAutoML
from mbt.contracts.framework import FrameworkPlugin

class H2OAutoMLFramework(FrameworkPlugin):
    """H2O AutoML adapter — automated model selection and hyperparameter tuning."""

    def setup(self, config: dict) -> None:          # [NEW] lifecycle hook
        h2o.init()

    def teardown(self) -> None:                     # [NEW] lifecycle hook
        h2o.cluster().shutdown()

    def health_check(self) -> bool:                 # [NEW] lifecycle hook
        try:
            h2o.init()
            return h2o.cluster().is_running()
        except Exception:
            return False

    def supported_formats(self) -> list[str]:       # [NEW] format declaration
        return ["pandas", "h2o"]

    def validate_config(self, config: dict, problem_type: str) -> None:
        valid_metrics = {
            "binary_classification": ["AUC", "logloss", "AUCPR", "mean_per_class_error"],
            "multiclass_classification": ["logloss", "mean_per_class_error"],
            "regression": ["RMSE", "MAE", "RMSLE", "deviance"],
        }
        sort_metric = config.get("sort_metric")
        if sort_metric and sort_metric not in valid_metrics.get(problem_type, []):
            raise ValueError(
                f"Invalid sort_metric '{sort_metric}' for {problem_type}. "
                f"Valid options: {valid_metrics[problem_type]}"
            )
        max_runtime = config.get("max_runtime_secs", 3600)
        if max_runtime <= 0:
            raise ValueError("max_runtime_secs must be positive")

    def train(self, X_train, y_train, config: dict) -> Any:
        # Use MBTFrame protocol — extract pandas for H2O conversion
        train_df = h2o.H2OFrame(
            pd.concat([X_train.to_pandas(), y_train.to_pandas()], axis=1)
        )
        target_col = y_train.columns()[0]
        aml = H2OAutoML(
            max_runtime_secs=config.get("max_runtime_secs", 3600),
            max_models=config.get("max_models", 20),
            sort_metric=config.get("sort_metric", "AUC"),
            seed=config.get("seed", 42),
        )
        aml.train(y=target_col, training_frame=train_df)
        return aml.leader

    def predict(self, model, X):
        h2o_frame = h2o.H2OFrame(X.to_pandas())
        return model.predict(h2o_frame).as_data_frame()["predict"].values

    def predict_proba(self, model, X):
        h2o_frame = h2o.H2OFrame(X.to_pandas())
        preds = model.predict(h2o_frame).as_data_frame()
        return preds.drop(columns=["predict"]).values

    def serialize(self, model, path):
        h2o.save_model(model, path=path, force=True)

    def deserialize(self, path):
        return h2o.load_model(path)

    def get_feature_importance(self, model):
        varimp = model.varimp(use_pandas=True)
        return dict(zip(varimp["variable"], varimp["relative_importance"]))
```

```toml
# mbt-h2o/pyproject.toml
[project]
name = "mbt-h2o"
version = "0.1.0"
description = "H2O AutoML adapter for MBT"
dependencies = [
    "mbt-core>=0.1.0",
    "h2o>=3.40.0",
]

[project.entry-points."mbt.frameworks"]
h2o_automl = "mbt_h2o.framework:H2OAutoMLFramework"
```

### 8.6 Adapter Implementation Example — mbt-snowflake

```toml
# mbt-snowflake/pyproject.toml
[project]
name = "mbt-snowflake"
version = "0.1.0"
description = "Snowflake data warehouse adapter for MBT"
dependencies = [
    "mbt-core>=0.1.0",
    "snowflake-connector-python>=3.0.0",
    "pandas>=2.0.0",
]

[project.entry-points."mbt.data_connectors"]
snowflake = "mbt_snowflake.connector:SnowflakeConnector"

[project.entry-points."mbt.output_writers"]
snowflake = "mbt_snowflake.writer:SnowflakeWriter"
```

### 8.7 Adapter Implementation Example — mbt-mlflow

```toml
# mbt-mlflow/pyproject.toml
[project]
name = "mbt-mlflow"
version = "0.1.0"
description = "MLflow model registry and experiment tracker adapter for MBT"
dependencies = [
    "mbt-core>=0.1.0",
    "mlflow>=2.10.0",
]

[project.entry-points."mbt.model_registries"]
mlflow = "mbt_mlflow.registry:MLflowRegistry"

[project.entry-points."mbt.experiment_trackers"]
mlflow = "mbt_mlflow.tracker:MLflowTracker"
```

### 8.8 Plugin Discovery at Runtime

When `mbt compile` runs, `mbt-core` discovers all installed adapters via Python's `entry_points` mechanism. No configuration needed — if the package is installed, its adapters are available.

```python
# mbt-core: mbt/core/registry.py
from importlib.metadata import entry_points

class PluginRegistry:
    """Discovers and loads adapter plugins from installed packages."""

    ADAPTER_GROUPS = [
        "mbt.frameworks", "mbt.data_connectors", "mbt.output_writers",
        "mbt.executors", "mbt.storage", "mbt.model_registries",
        "mbt.experiment_trackers", "mbt.orchestrators",
        "mbt.feature_selection", "mbt.normalization", "mbt.encoding",
        "mbt.secrets",                    # [NEW] secrets provider adapters
    ]

    def __init__(self):
        self._plugins = {}
        for group in self.ADAPTER_GROUPS:
            self._plugins[group] = {}
            for ep in entry_points(group=group):
                self._plugins[group][ep.name] = ep

    def get(self, group: str, name: str):
        """Load and return an adapter plugin by group and name."""
        ep = self._plugins.get(group, {}).get(name)
        if ep is None:
            installed = list(self._plugins.get(group, {}).keys())
            raise MissingAdapterError(
                f"No adapter '{name}' found for '{group}'. "
                f"Installed adapters: {installed}. "
                f"Install the adapter package, e.g.: pip install mbt-{name}"
            )
        return ep.load()()

    def list_installed(self) -> dict[str, list[str]]:
        """List all installed adapters by category. Used by `mbt deps list`."""
        return {
            group: list(plugins.keys())
            for group, plugins in self._plugins.items()
            if plugins
        }
```

When the compiler encounters `framework: h2o_automl` in a pipeline YAML, it calls `registry.get("mbt.frameworks", "h2o_automl")`. If `mbt-h2o` is installed, it loads the class. If not, it raises a clear error telling the user to `pip install mbt-h2o`.

### 8.9 Community Adapter Development

Anyone can create a new MBT adapter. The process is:

1. Create a Python package with an implementation of the relevant ABC from `mbt.contracts`
2. Register via `entry_points` in the package's `pyproject.toml`
3. Publish to PyPI (or a private index for company-internal adapters)

Example — a community member builds a CatBoost adapter:

```toml
# mbt-catboost/pyproject.toml
[project]
name = "mbt-catboost"
version = "0.1.0"
dependencies = ["mbt-core>=0.1.0", "catboost>=1.2.0"]

[project.entry-points."mbt.frameworks"]
catboost = "mbt_catboost:CatBoostFramework"
```

Once published, any user can:

```bash
pip install mbt-catboost
```

```yaml
# immediately available in pipeline YAML
model_training:
  framework: catboost
  config:
    iterations: 1000
    learning_rate: 0.03
    depth: 8
```

---

## 9. DRY — Three Levels of Reuse

### Level 1: Project Library (`lib/`)

Shared Python functions used across pipeline steps within one project. The `lib/` directory is auto-added to PYTHONPATH by the framework.

```
my-ml-project/
├── lib/
│   ├── __init__.py
│   ├── custom_transforms.py    # custom encoding, feature engineering
│   ├── custom_metrics.py       # business-specific evaluation metrics
│   └── custom_validators.py    # domain-specific data validation rules
```

When a DS needs custom logic that installed adapters don't cover, they write it in `lib/` and reference it from their pipeline YAML via hook functions.

### Level 2: Adapter Packages (`pyproject.toml`)

MBT adapters and community extensions are standard pip-installable Python packages. The user's `pyproject.toml` declares which adapters and extensions are needed. Standard pip/uv/poetry resolution handles version conflicts — each adapter owns its own heavy dependencies (h2o, snowflake-connector-python, etc.) and they never conflict with each other.

```toml
# pyproject.toml (scaffolded by mbt init)
[project]
name = "my-ml-project"
dependencies = [
    # Core framework
    "mbt-core>=0.1.0",
    # AutoML framework adapter
    "mbt-h2o>=0.1.0",
    # Data warehouse adapter
    "mbt-snowflake>=0.1.0",
    # Orchestrator adapter
    "mbt-airflow>=0.1.0",
    # Model registry adapter
    "mbt-mlflow>=0.1.0",
    # Community extensions
    "mbt-boruta>=0.3.0",          # additional feature selection method
    # Company-internal shared transforms
    "myorg-mbt-utils>=2.1.0",
]

[tool.mbt]
profile = "my-ml-project"
```

```bash
# Standard Python install — no custom mbt deps command needed
pip install .   # or: uv sync, poetry install
```

Once installed, any adapter is immediately available in pipeline YAML and `profiles.yaml`. The `mbt deps list` command scans installed packages for `mbt.*` entry_points and shows the user which adapters are available.

### Level 3: Hook Functions — Custom Code Without Leaving YAML

For logic that is project-specific and doesn't warrant a full plugin, MBT supports hook functions — references to Python functions in `lib/` that the framework calls at specific points in the pipeline. This is inspired by MLflow Recipes' `steps/transform.py` and `steps/custom_metrics.py` pattern.

```yaml
# pipelines/churn_training_v1.yaml
training:
  transformations:
    normalization:
      enabled: true
      method: standard_scaler
    custom_transforms:
      - lib.custom_transforms.compute_rfm
      - lib.custom_transforms.create_interaction_features
  evaluation:
    primary_metric: roc_auc
    custom_metrics:
      - lib.custom_metrics.business_value_score
      - lib.custom_metrics.cost_sensitive_accuracy
```

```python
# lib/custom_transforms.py
def compute_rfm(df: pd.DataFrame, context: dict) -> pd.DataFrame:
    """Compute recency, frequency, monetary features.

    Called by MBT during the transform step.
    Receives the dataframe and pipeline context (params, schema, etc.).
    Must return a transformed dataframe.
    """
    df["recency"] = (context["execution_date"] - df["last_purchase"]).dt.days
    df["frequency"] = df.groupby("customer_id")["order_id"].transform("count")
    df["monetary"] = df.groupby("customer_id")["amount"].transform("sum")
    return df
```

```python
# lib/custom_metrics.py
def business_value_score(y_true, y_pred, context: dict) -> float:
    """Custom metric: expected profit from churn intervention.

    Called by MBT during evaluation.
    Must return a single float value.
    """
    intervention_cost = 50
    save_value = 500
    tp = ((y_pred == 1) & (y_true == 1)).sum()
    fp = ((y_pred == 1) & (y_true == 0)).sum()
    return (tp * save_value) - ((tp + fp) * intervention_cost)
```

### DRY Layer Summary

```
┌──────────────────────────────────────────────────────────────┐
│ ADAPTER PACKAGES (mbt-h2o, mbt-snowflake, mbt-airflow...)   │
│ Shared across teams / orgs / open source                     │
│ Each adapter owns its own heavy dependencies                 │
│ pip-installable via pyproject.toml, registered via            │
│ entry_points. Standard pip/uv/poetry resolution handles      │
│ version conflicts.                                           │
│ Provides: frameworks, connectors, orchestrators, etc.        │
│ Analogy: dbt adapters (dbt-snowflake, dbt-bigquery)          │
├──────────────────────────────────────────────────────────────┤
│ PROJECT LIB (lib/)                                           │
│ Shared across all pipelines in one project                   │
│ Plain Python modules, auto-added to PYTHONPATH               │
│ Provides: custom transforms, metrics, validators             │
│ Analogy: dbt macros/ directory                               │
├──────────────────────────────────────────────────────────────┤
│ PIPELINE YAML (pipelines/*.yaml)                             │
│ Per-pipeline declarations                                    │
│ References adapters, extensions, and lib/                    │
│ Provides: the unique configuration per use case              │
│ Analogy: dbt models/ directory                               │
└──────────────────────────────────────────────────────────────┘
```

---

## 10. Orchestrator Integration — Manifest-Driven DAG Generation

MBT integrates with orchestrators through the compiled manifest. The orchestrator owns scheduling and monitoring; MBT owns pipeline definition and compilation. The orchestrator plugin reads `manifest.json` and translates it into the orchestrator's native format. This means teams can use Airflow, Prefect, Dagster, or any other orchestrator without changing their pipeline YAMLs or manifests.

### 10.1 How It Works

The orchestrator is configured in `profiles.yaml` (see Section 4). The `mbt generate-dags` command reads the compiled manifests and uses the configured orchestrator plugin to generate deployable DAG/flow files. Each MBT step becomes one task in the orchestrator, wired according to the manifest's `parent_map`.

```
manifest.json (orchestrator-agnostic)
  │
  ▼
OrchestratorPlugin.generate_dag_file()
  │
  ├──► Airflow: dags/ml_churn_training_v1.py
  ├──► Prefect: flows/ml_churn_training_v1.py
  └──► Argo:    workflows/ml_churn_training_v1.yaml
```

Every orchestrator plugin follows the same pattern: read the manifest, create one task per step, wire dependencies from `parent_map`, and configure each task to call `mbt exec --pipeline <n> --step <step> --target <target>`.

### 10.2 Airflow Adapter (mbt-airflow)

```python
# mbt-airflow/src/mbt_airflow/dag_builder.py
class AirflowDagBuilder(OrchestratorPlugin):

    def __init__(self, manifest_path, schedule=None, target=None,
                 tags=None, default_args=None):
        with open(manifest_path) as f:
            self.manifest = json.load(f)
        self.target = target or self.manifest["metadata"]["target"]
        self.schedule = schedule
        self.tags = tags or []
        self.default_args = default_args or {}

    def build(self) -> DAG:
        pipeline_name = self.manifest["metadata"]["pipeline_name"]
        dag = DAG(
            dag_id=f"ml_{pipeline_name}",
            schedule=self.schedule,
            default_args=self.default_args,
            tags=self.tags,
            catchup=False,
        )

        airflow_tasks = {}
        for step_name, step_def in self.manifest["steps"].items():
            executor_type = self.manifest["metadata"].get("executor_type", "kubernetes")
            airflow_tasks[step_name] = self._create_operator(
                dag, pipeline_name, step_name, step_def, executor_type
            )

        # Wire dependencies from manifest
        for step_name, parents in self.manifest["dag"]["parent_map"].items():
            for parent in parents:
                airflow_tasks[parent] >> airflow_tasks[step_name]

        return dag

    def _create_operator(self, dag, pipeline_name, step_name, step_def, executor_type):
        if executor_type == "kubernetes":
            resources = step_def.get("resources", {})
            return KubernetesPodOperator(
                dag=dag,
                task_id=step_name,
                name=f"mbt-{pipeline_name}-{step_name}",
                image=self.manifest["metadata"].get("image", "myorg/mbt-runner:latest"),
                cmds=["mbt", "exec",
                       "--pipeline", pipeline_name,
                       "--step", step_name,
                       "--target", self.target],
                container_resources={
                    "requests": {
                        "cpu": resources.get("cpu", "1"),
                        "memory": resources.get("memory", "2Gi"),
                    },
                    "limits": {
                        "cpu": resources.get("cpu", "1"),
                        "memory": resources.get("memory", "2Gi"),
                    },
                },
                get_logs=True,
                is_delete_operator_pod=True,
            )
        else:
            return BashOperator(
                dag=dag,
                task_id=step_name,
                bash_command=(
                    f"mbt exec --pipeline {pipeline_name} "
                    f"--step {step_name} --target {self.target}"
                ),
            )

    def generate_dag_file(self, manifest_path, output_path,
                          schedule=None, target=None, **kwargs):
        """Generate a deployable Airflow DAG file."""
        pipeline_name = self.manifest["metadata"]["pipeline_name"]
        target = target or self.target
        code = f'''
from pathlib import Path
from mbt_airflow import AirflowDagBuilder

dag = AirflowDagBuilder(
    manifest_path="{manifest_path}",
    schedule="{schedule or ''}",
    target="{target}",
    tags=["ml", "{pipeline_name}"],
).build()
'''
        with open(output_path, "w") as f:
            f.write(code)
```

### 10.3 Prefect Adapter (mbt-prefect)

```python
# mbt-prefect/src/mbt_prefect/flow_builder.py
import subprocess, json
from prefect import flow, task

class PrefectFlowBuilder(OrchestratorPlugin):

    def __init__(self, manifest_path, target=None, tags=None):
        with open(manifest_path) as f:
            self.manifest = json.load(f)
        self.target = target or self.manifest["metadata"]["target"]
        self.tags = tags or []

    def build(self):
        pipeline_name = self.manifest["metadata"]["pipeline_name"]
        parent_map = self.manifest["dag"]["parent_map"]
        steps = self.manifest["steps"]

        mbt_tasks = {}
        for step_name, step_def in steps.items():
            resources = step_def.get("resources", {})
            mbt_tasks[step_name] = self._create_task(
                pipeline_name, step_name, resources
            )

        @flow(name=f"ml_{pipeline_name}", tags=self.tags)
        def pipeline_flow():
            futures = {}
            for batch in self.manifest["dag"]["execution_batches"]:
                for step_name in batch:
                    wait_for = [futures[p] for p in parent_map[step_name]
                                if p in futures]
                    futures[step_name] = mbt_tasks[step_name].submit(
                        wait_for=wait_for
                    )
            return futures

        return pipeline_flow

    def _create_task(self, pipeline_name, step_name, resources):
        @task(name=step_name, retries=2, retry_delay_seconds=300,
              tags=[pipeline_name, step_name])
        def mbt_step(wait_for=None):
            subprocess.run([
                "mbt", "exec",
                "--pipeline", pipeline_name,
                "--step", step_name,
                "--target", self.target
            ], check=True)
        return mbt_step
```

### 10.4 Third-Party Orchestrator Packages

Additional orchestrators can be added as third-party packages:

```toml
# pyproject.toml for mbt-dagster
[project.entry-points."mbt.orchestrators"]
dagster = "mbt_dagster:DagsterJobBuilder"

# pyproject.toml for mbt-argo
[project.entry-points."mbt.orchestrators"]
argo = "mbt_argo:ArgoWorkflowBuilder"
```

Once installed, the DE switches orchestrators by changing a single line in `profiles.yaml`:

```yaml
orchestrator:
  type: dagster                    # ← from mbt-dagster package
  config:
    repository: ml-pipelines
```

### 10.5 End-to-End Workflow

```
DS updates pipelines/churn_training_v1.yaml
  │
  ▼
CI runs: mbt validate && mbt compile --target prod
  │
  ▼
target/churn_training_v1/manifest.json
  │
  ▼
CI runs: mbt generate-dags --target prod
  │
  ├──► (if orchestrator=airflow) dags/ml_churn_training_v1.py ──► synced to Airflow
  └──► (if orchestrator=prefect) flows/ml_churn_training_v1.py ──► deployed to Prefect
  │
  ▼
Orchestrator DAG auto-generates:
  ml_churn_training_v1
  ├── load_data
  ├── validate_data              [NEW]
  ├── split_data
  ├── normalize
  ├── select_features
  ├── train_model
  ├── evaluate
  └── log_to_mlflow
```

Each step appears as a separate task in the orchestrator with its own logs, duration, retry history, and monitoring. If `train_model` fails, the orchestrator retries just that step. The DS and the manifest never know or care which orchestrator is running things — the manifest is the orchestrator-neutral contract.

### 10.6 CI/CD Automation

```yaml
# .github/workflows/deploy-pipeline.yaml
on:
  push:
    paths: ["pipelines/**", "lib/**"]
    branches: [main]

jobs:
  deploy:
    steps:
      - uses: actions/checkout@v4
      - run: pip install .              # installs mbt-core + all adapters
      - run: mbt validate --all
      - run: mbt test --all            # [NEW] run DAG assertion tests
      - run: |
          for pipeline in pipelines/*.yaml; do
            name=$(basename "$pipeline" .yaml)
            mbt compile "$name" --target prod
          done
      - run: mbt generate-dags --target prod --output ./generated_dags/
      - run: |
          # Deploy manifests
          aws s3 sync target/ s3://ml-manifests/ \
            --exclude "*" --include "*/manifest.json"
          # Deploy generated DAG files to orchestrator
          aws s3 sync generated_dags/ s3://orchestrator-dags/
```

---

## 11. Data Validation and Quality Checks **[NEW]**

Production ML pipelines require data validation beyond `min_rows`. Schema drift, unexpected nulls, out-of-range values, and distributional shifts are common causes of silent model degradation. MBT integrates data validation as a first-class pipeline step.

### 11.1 Built-in Validation Checks

| Check Type | Description | Parameters |
|---|---|---|
| `null_threshold` | Fail if null percentage exceeds threshold | `columns`, `max_null_pct` |
| `value_range` | Fail if values fall outside min/max bounds | `column`, `min`, `max` |
| `expected_columns` | Fail if required columns are missing | `columns` (list) |
| `unique_key` | Fail if primary key has duplicates | `columns` |
| `type_check` | Fail if column types don't match expectations | `column_types` (dict) |
| `distribution_drift` | Warn if distribution shifts from reference | `reference_run_id`, `threshold` |
| `custom` | Run a user-defined validation function | `function` (lib/ reference) |

### 11.2 Failure Handling

Each validation check can be configured with an `on_failure` policy: `fail` (halt the pipeline), `warn` (log warning and continue), or `skip_row` (remove failing rows and continue). The pipeline-level `on_failure` sets the default; individual checks can override it.

```yaml
validation:
  on_failure: fail                                # pipeline-level default
  checks:
    - type: null_threshold
      columns: [churned, customer_id]
      max_null_pct: 0.0                           # hard fail: key columns must be non-null
    - type: value_range
      column: monthly_charges
      min: 0
      max: 500
      on_failure: skip_row                        # override: remove bad rows
    - type: distribution_drift
      reference_run_id: "{{ var('baseline_run') }}"
      threshold: 0.1
      on_failure: warn                            # override: warn only
```

### 11.3 Custom Validators

```python
# lib/custom_validators.py
def check_label_distribution(df: pd.DataFrame, context: dict) -> tuple[bool, str]:
    """Validate that label distribution is reasonable.

    Returns (passed: bool, message: str).
    """
    label_col = context["schema"]["target"]["label_column"]
    positive_rate = df[label_col].mean()
    if positive_rate < 0.01 or positive_rate > 0.99:
        return False, f"Label imbalance too extreme: {positive_rate:.4f}"
    return True, f"Label positive rate: {positive_rate:.4f}"
```

---

## 12. Secrets Management **[NEW]**

The original blueprint used `{{ env_var('KEY') }}` for all sensitive configuration. While sufficient for local development, production deployments need integration with enterprise secrets management systems. MBT introduces a pluggable secrets provider model.

### 12.1 Secrets Provider Contract

```python
# mbt-core: mbt/contracts/secrets.py
class SecretsPlugin(ABC):
    """Interface for secrets provider adapters."""

    @abstractmethod
    def get_secret(self, key: str) -> str:
        """Retrieve a secret value by key."""
        ...

    @abstractmethod
    def validate_access(self) -> bool:
        """Verify secrets backend is reachable. Called by `mbt debug`."""
        ...
```

Built-in: `EnvironmentSecretsProvider` (reads from env vars, zero deps).

Adapter packages:

- `mbt-vault` → HashiCorp Vault
- `mbt-aws-secrets` → AWS Secrets Manager
- `mbt-k8s-secrets` → Kubernetes Secrets

### 12.2 Usage in profiles.yaml

```yaml
my-ml-project:
  secrets:
    provider: vault
    config:
      vault_addr: "https://vault.myorg.com"
      auth_method: kubernetes
      secret_path: ml-platform/credentials

  outputs:
    prod:
      data_connector:
        type: snowflake
        config:
          account: "{{ secret('snowflake/account') }}"
          user: "{{ secret('snowflake/user') }}"
          password: "{{ secret('snowflake/password') }}"
```

The `{{ secret('key') }}` function is resolved at runtime (not at compile time) to avoid writing secrets into `manifest.json`. The manifest stores the secret *reference* (`secret:snowflake/account`), and the runner resolves it just before step execution.

---

## 13. Observability and Monitoring **[NEW]**

The original blueprint produced only post-hoc `run_results.json` files. Production ML platforms require real-time observability into pipeline execution. MBT provides three layers of observability.

### 13.1 Structured Logging

Every step emits structured JSON log events with consistent fields: `pipeline_name`, `step_name`, `run_id`, `target`, `timestamp`, and `level`. These are written to stdout for container log aggregation (Datadog, Splunk, ELK) and to a local log file.

```json
{"ts": "2026-02-13T10:35:12Z", "level": "INFO", "pipeline": "churn_training_v1",
 "step": "train_model", "run_id": "run_20260213_a7f3", "target": "prod",
 "event": "step_started", "resources": {"cpu": "8", "memory": "32Gi"}}

{"ts": "2026-02-13T10:42:45Z", "level": "INFO", "pipeline": "churn_training_v1",
 "step": "train_model", "run_id": "run_20260213_a7f3", "target": "prod",
 "event": "step_completed", "duration_seconds": 453,
 "metrics": {"automl_leader": "StackedEnsemble_BestOfFamily_1"}}
```

### 13.2 Metrics Emission

The runner emits timing and resource metrics for each step as StatsD/Prometheus-compatible gauges. Teams can configure a metrics backend in `profiles.yaml`:

```yaml
observability:
  metrics:
    type: statsd                       # statsd | prometheus | none
    config:
      host: "{{ env_var('STATSD_HOST') }}"
      port: 8125
      prefix: mbt.pipelines
  logging:
    format: json                       # json | text
    level: INFO
```

Emitted metrics include: `mbt.pipelines.<pipeline>.<step>.duration_seconds`, `mbt.pipelines.<pipeline>.<step>.status` (0=success, 1=failure), `mbt.pipelines.<pipeline>.total_duration_seconds`, and `mbt.pipelines.<pipeline>.<step>.artifact_size_bytes`.

### 13.3 Step-Level Callbacks

The runner supports lifecycle callbacks that adapter or custom plugins can register: `on_step_start`, `on_step_complete`, `on_step_failure`, `on_pipeline_complete`. These enable custom integrations like Slack notifications, PagerDuty alerts, or custom dashboards.

---

## 14. Testing Strategy **[NEW]**

MBT supports testing at three levels: YAML validation, compiled DAG assertions, and integration tests.

### 14.1 YAML Validation (`mbt validate`)

`mbt validate` checks all pipeline YAMLs against the Pydantic schema, resolves base pipelines and includes, and calls `validate_config()` on each referenced adapter. This catches typos, missing fields, invalid parameter combinations, and missing adapters without executing anything.

```bash
mbt validate                                     # validate all pipelines
mbt validate churn_training_v1                   # validate specific pipeline
mbt validate --show-resolved                     # show fully resolved YAML
```

### 14.2 DAG Assertions (`mbt test`)

`mbt test` compiles a pipeline and verifies structural properties of the resulting DAG. Data scientists can write assertion files that declare expected properties:

```yaml
# tests/churn_training_v1.test.yaml
pipeline: churn_training_v1
assertions:
  - type: step_exists
    step: train_model
  - type: step_absent
    step: encode                         # encoding is disabled
  - type: step_order
    before: normalize
    after: train_model
  - type: step_count
    min: 5
    max: 8
  - type: resource_limit
    step: train_model
    memory_max: "64Gi"
```

```bash
mbt test                                         # run all DAG assertion tests
mbt test churn_training_v1                       # test specific pipeline
```

### 14.3 Integration Tests

For end-to-end testing, `mbt run --target dev --dry-run` compiles and validates the full pipeline against the dev target without executing any steps. `mbt run --target dev` executes the full pipeline against local sample data. The example project includes a `sample_data/` directory with representative test data for local development.

### 14.4 Adapter Testing

Each adapter package includes its own test suite. `mbt-core` provides test fixtures and mock objects (`MockMBTFrame`, `MockStoragePlugin`, etc.) to help adapter authors write unit tests without real infrastructure dependencies.

```python
# mbt-core: mbt/testing/fixtures.py
class MockMBTFrame:
    """Mock MBTFrame for adapter unit tests."""
    def __init__(self, data: dict):
        self._df = pd.DataFrame(data)

    def to_pandas(self): return self._df
    def num_rows(self): return len(self._df)
    def columns(self): return list(self._df.columns)
    def schema(self): return {c: str(d) for c, d in self._df.dtypes.items()}


class MockStoragePlugin:
    """In-memory storage for testing."""
    def __init__(self):
        self._store = {}

    def put(self, name, data, run_id, step_name, metadata=None):
        uri = f"mock://{run_id}/{step_name}/{name}"
        self._store[uri] = data
        return uri

    def get(self, uri): return self._store[uri]
    def exists(self, uri): return uri in self._store
```

---

## 15. CLI Commands

```bash
# ─── Pipeline Lifecycle ──────────────────────────────────────
mbt init                                    # scaffold new project
mbt validate                                # validate all pipeline YAMLs
mbt validate churn_training_v1              # validate specific pipeline
mbt validate --show-resolved                # [NEW] show fully resolved YAML
mbt compile churn_training_v1               # compile to manifest.json
mbt compile churn_training_v1 --target prod # compile for specific target
mbt compile --all --target prod             # compile all pipelines

# ─── Running Pipelines ───────────────────────────────────────
mbt run --select churn_training_v1          # run full pipeline
mbt run --select churn_training_v1 --target prod   # run against prod
mbt run --select churn_training_v1 --vars execution_date=2026-02-01
mbt run --select churn_training_v1 --dry-run       # show plan only
mbt run --select "tag:churn"                # run by tag
mbt run --select "tag:training"             # all training pipelines

# ─── Testing [NEW] ───────────────────────────────────────────
mbt test                                    # run all DAG assertion tests
mbt test churn_training_v1                  # test specific pipeline

# ─── Single Step Execution (used by orchestrators) ───────────
mbt exec --pipeline churn_training_v1 --step train_model --target prod

# ─── Orchestrator DAG Generation ─────────────────────────────
mbt generate-dags                           # generate DAGs using profiles.yaml orchestrator
mbt generate-dags --orchestrator prefect    # override orchestrator type
mbt generate-dags --target prod             # generate for specific target
mbt generate-dags --output ./dags/          # custom output directory

# ─── Inspection ──────────────────────────────────────────────
mbt ls                                      # list all pipelines
mbt ls --tags                               # list with tags
mbt dag churn_training_v1                   # print DAG
mbt dag churn_training_v1 --mermaid         # output Mermaid diagram
mbt status churn_training_v1               # show last run status
mbt inspect churn_training_v1 --step train  # show step results

# ─── Package Management ─────────────────────────────────────
mbt deps list                               # list installed MBT plugins
mbt deps check                              # verify all required plugins are installed

# ─── Utilities ───────────────────────────────────────────────
mbt retry churn_training_v1                 # re-run from failure point
mbt retry churn_training_v1 --force         # [NEW] re-run all steps
mbt clean                                   # remove target/ artifacts
mbt debug                                   # validate connections + secrets
mbt debug --target prod                     # validate prod connections
```

---

## 16. Project Structure

### 16.1 mbt-core repository **[CHANGED]**

```
mbt-core/
├── src/
│   └── mbt/
│       ├── __init__.py
│       ├── __main__.py
│       ├── cli.py                     # Typer CLI
│       ├── core/
│       │   ├── compiler.py            # YAML → manifest.json (5 phases)
│       │   ├── dag.py                 # DAG assembly + topological sort
│       │   ├── manifest.py            # Manifest model + serialization
│       │   ├── runner.py              # Step execution engine + artifact passing
│       │   ├── registry.py            # Plugin discovery + entry_points
│       │   ├── selector.py            # --select parser (name, tag:)
│       │   ├── context.py             # Runtime context for steps + hooks
│       │   ├── composition.py         # [NEW] base_pipeline + !include resolution
│       │   └── data.py                # [NEW] MBTFrame protocol + PandasFrame
│       ├── contracts/                 # ABCs for all adapter categories
│       │   ├── framework.py           # FrameworkPlugin ABC (with lifecycle hooks)
│       │   ├── data_connector.py      # DataConnectorPlugin ABC
│       │   ├── output_writer.py       # OutputWriterPlugin ABC
│       │   ├── orchestrator.py        # OrchestratorPlugin ABC
│       │   ├── model_registry.py      # ModelRegistryPlugin ABC
│       │   ├── executor.py            # ExecutorPlugin ABC
│       │   ├── storage.py             # StoragePlugin ABC (with artifact flow)
│       │   ├── secrets.py             # [NEW] SecretsPlugin ABC
│       │   ├── feature_selection.py   # FeatureSelectionPlugin ABC
│       │   ├── normalization.py       # NormalizationPlugin ABC
│       │   └── encoding.py            # EncodingPlugin ABC
│       ├── config/
│       │   ├── schema.py              # Pydantic models for pipeline YAML (versioned)
│       │   ├── profiles.py            # Profile/target resolution
│       │   ├── loader.py              # YAML loading + Jinja2 env_var() + secret()
│       │   └── merger.py              # Config precedence merging
│       ├── steps/                     # Built-in step implementations
│       │   ├── base.py                # Step ABC
│       │   ├── load_data.py
│       │   ├── join_tables.py         # [NEW] multi-table join step
│       │   ├── validate_data.py       # [NEW] data validation step
│       │   ├── split_data.py
│       │   ├── normalize.py
│       │   ├── encode.py
│       │   ├── feature_selection.py
│       │   ├── evaluate.py
│       │   ├── log_run.py             # delegates to model_registry adapter
│       │   ├── load_model.py          # serving: load from registry adapter
│       │   ├── apply_transforms.py    # serving: replay training transforms
│       │   ├── predict.py             # serving: batch predict
│       │   └── publish.py             # serving: write via output_writer adapter
│       ├── builtins/                  # Minimal built-in adapters (zero deps)
│       │   ├── local_executor.py      # run steps as subprocesses
│       │   ├── local_connector.py     # read from local parquet/csv
│       │   ├── local_storage.py       # save artifacts to disk
│       │   └── env_secrets.py         # [NEW] env var secrets provider
│       └── testing/                   # [NEW] test utilities for adapter authors
│           ├── fixtures.py            # MockMBTFrame, MockStorage, etc.
│           └── assertions.py          # DAG assertion engine
├── tests/
├── examples/
│   └── telecom-churn/
│       ├── pyproject.toml
│       ├── profiles.yaml
│       ├── pipelines/
│       │   ├── _base_churn.yaml       # [NEW] shared base pipeline
│       │   ├── churn_training_v1.yaml
│       │   └── churn_serving_v1.yaml
│       ├── includes/                  # [NEW] shared YAML fragments
│       │   ├── telecom_data_source.yaml
│       │   └── telecom_schema.yaml
│       ├── tests/                     # [NEW] DAG assertion tests
│       │   └── churn_training_v1.test.yaml
│       ├── lib/
│       │   ├── custom_transforms.py
│       │   └── custom_metrics.py
│       └── sample_data/
├── pyproject.toml
└── README.md
```

### 16.2 Adapter package repository (example: mbt-h2o)

Each adapter is its own repository/package with a minimal structure:

```
mbt-h2o/
├── src/
│   └── mbt_h2o/
│       ├── __init__.py
│       └── framework.py               # H2OAutoMLFramework implementation
├── tests/
│   └── test_framework.py
├── pyproject.toml                     # depends on mbt-core + h2o
└── README.md
```

### 16.3 User project (created by `mbt init`)

```
my-ml-project/
├── pyproject.toml                 # mbt-core + adapter dependencies
├── profiles.yaml                  # DE: environment configs
├── pipelines/                     # DS: one YAML per pipeline
│   ├── _base_churn.yaml           # [NEW] base pipelines (underscore prefix)
│   ├── churn_training_v1.yaml
│   ├── churn_serving_v1.yaml
│   ├── adoption_training_v1.yaml
│   └── adoption_serving_v1.yaml
├── includes/                      # [NEW] shared YAML fragments
│   ├── telecom_data_source.yaml
│   └── telecom_schema.yaml
├── tests/                         # [NEW] DAG assertion tests
│   ├── churn_training_v1.test.yaml
│   └── adoption_training_v1.test.yaml
├── lib/                           # DS: project-level shared code
│   ├── __init__.py
│   ├── custom_transforms.py
│   ├── custom_metrics.py
│   └── custom_validators.py
├── sample_data/                   # local test data for dev target
├── generated_dags/                # generated by `mbt generate-dags` (gitignored)
│   └── ml_churn_training_v1.py    # orchestrator-specific DAG file
└── target/                        # generated (gitignored)
    ├── churn_training_v1/
    │   └── manifest.json
    ├── churn_serving_v1/
    │   └── manifest.json
    └── ...
```

---

## 17. Comparison with Existing Tools **[CHANGED]**

The following comparison aims to be honest about MBT's strengths and weaknesses relative to existing tools, rather than positioning MBT as superior in all dimensions.

| Concern | dbt | Ludwig | MLflow Recipes | Kedro | ZenML | MBT |
|---|---|---|---|---|---|---|
| DS interface | SQL files | YAML | YAML + Python | Python DSL | Python decorators | YAML |
| Abstraction level | Per-model | Per-feature | Per-step | Per-node | Per-step | Per-pipeline |
| DAG defined by | `ref()` | Fixed (internal) | Fixed template | Code | Code | Compiler from YAML |
| Dynamic DAG | No | No | No | No | No | Yes (toggle sections) |
| Profiles / targets | ✓ | Backend config | ✓ (Jinja) | Layered conf | Stacks | ✓ |
| Plugin ecosystem | Strong (adapters + packages) | Encoders/combiners | Templates | Plugins | Integrations | Modular adapters (mbt-core + mbt-*) |
| Compile before run | ✓ (manifest.json) | ✓ (validation) | Step caching | No | No | ✓ (manifest.json + validate_config()) |
| Orchestrator integration | dbt Cloud/CLI | None | None | None | Orchestrator-agnostic | Pluggable orchestrators |
| Problem domain | Analytics (SQL) | Deep learning | ML (sklearn focus) | General data | General ML | Tabular ML (AutoML-first) |
| Code required | SQL | None (YAML only) | Python steps | Python | Python | None (YAML + optional hooks) |
| **Caching / incremental** | **Yes** | **No** | **No** | **Yes** | **Yes** | **No (planned)** |
| **Notebook integration** | **Limited** | **Yes** | **Yes** | **Yes** | **Yes** | **No (planned)** |
| **Deep learning** | **No** | **Yes** | **No** | **Via code** | **Via code** | **No** |
| **Data scale** | **SQL-native** | **GPU datasets** | **pandas** | **Spark/pandas** | **pandas/Spark** | **MBTFrame (pluggable)** |

**MBT's unique strengths:** Fully declarative pipeline definition with no code required for standard workflows. Dynamic DAG compilation from toggleable YAML sections. Compile-time config validation via framework adapters. Modular adapter ecosystem where every integration is a separate package (like dbt adapters). Pluggable orchestrator integration via compiled manifests. Pipeline composition via base pipelines and includes. AutoML-first approach that gives data scientists strong models without hyperparameter tuning.

**Where MBT is weaker:** No caching or incremental execution — ZenML and Kedro support this, which is important for iterative development workflows. No notebook integration for interactive exploration — every other tool except dbt supports this. No deep learning support — Ludwig excels here, and MBT's YAML schema is designed around structured tabular data. Limited to tabular ML use cases in its current form. No built-in experiment comparison UI — relies on MLflow/W&B for this. The YAML-only approach may frustrate data scientists who prefer code-first workflows.

---

## 18. Artifact Tracking and Run Results

Every pipeline run produces `target/<pipeline_name>/run_results.json`:

```json
{
  "run_id": "run_20260213_103045_a7f3",
  "pipeline_name": "churn_training_v1",
  "target": "prod",
  "execution_date": "2026-02-01",
  "started_at": "2026-02-13T10:30:45Z",
  "completed_at": "2026-02-13T10:47:12Z",
  "status": "success",
  "schema_version": 1,
  "mlflow_run_id": "abc123def456",
  "mlflow_experiment": "churn_prediction",
  "steps": {
    "validate_data": {
      "status": "success",
      "checks_passed": 4,
      "checks_warned": 1,
      "rows_before": 50000,
      "rows_after": 49823
    },
    "train_model": {
      "status": "success",
      "duration_seconds": 423,
      "plugin": "mbt_h2o.framework:H2OAutoMLFramework",
      "config_hash": "sha256:f3a1b2...",
      "automl_leader": "StackedEnsemble_BestOfFamily_1"
    },
    "evaluate": {
      "status": "success",
      "duration_seconds": 45,
      "metrics": {
        "roc_auc": 0.943,
        "accuracy": 0.891,
        "f1": 0.867,
        "precision": 0.834,
        "recall": 0.903
      }
    }
  }
}
```

All metrics, model artifacts, plots, and lineage are also logged to the configured model registry adapter (e.g., `mbt-mlflow`, `mbt-wandb`), making the `run_results.json` a lightweight local record while the registry serves as the system of record for model comparison, registry, and deployment.

---

## 19. Known Limitations and Non-Goals **[NEW]**

Being explicit about what MBT does *not* do is as important as what it does. The following are deliberate non-goals or known limitations of the current architecture.

### 19.1 Explicit Non-Goals

**Deep learning.** MBT is designed for tabular ML. While a community adapter could wrap PyTorch or TensorFlow, the YAML schema, evaluation pipeline, and data windowing model are all designed around structured data. Teams with deep learning needs should use Ludwig, Ray Train, or framework-native tooling.

**Real-time serving.** MBT supports batch inference. Real-time model serving (REST endpoints, gRPC, streaming inference) is out of scope. Models trained by MBT can be exported and deployed to serving platforms (SageMaker Endpoints, Seldon, BentoML) via the model registry, but MBT does not manage the serving infrastructure.

**Feature store integration.** MBT reads from data warehouses and local files. It does not integrate with feature stores (Feast, Tecton) in v0.1. This is a potential future adapter.

**Notebook-first workflows.** MBT is CLI-first by design. There is no Jupyter integration, no magic commands, and no interactive mode. Data scientists who prefer notebook-first exploration should use notebooks for EDA and prototyping, then codify the pipeline in YAML for production.

**Experiment comparison UI.** MBT does not include a web UI for comparing experiment results. It delegates this to model registry tools (MLflow UI, W&B dashboard). `mbt status` and `mbt inspect` provide CLI-based inspection.

### 19.2 Known Limitations (to be addressed)

**No caching or incremental execution.** Currently, every `mbt run` executes all steps from scratch. Step-level caching (skip steps whose inputs haven't changed) is planned for v0.3.

**Single-level pipeline inheritance only.** `base_pipeline` does not support chained inheritance (A extends B extends C). This is deliberate to keep composition predictable, but may be revisited if teams find single-level insufficient.

**No champion/challenger or A/B model comparison.** The DAG engine supports branching, but the YAML schema does not yet expose a way to declare parallel model training with comparison. This is planned for v0.4.

**Last-write-wins for concurrent pipeline runs.** If two pipeline runs write to the same output table concurrently, the last write wins. MBT does not implement output locking or versioned writes.

**No streaming data support.** Data windows assume batch data with snapshot semantics. Streaming data sources (Kafka, Kinesis) are not supported.

---

## 20. Failure Modes and Threat Model **[NEW]**

This section catalogs known failure modes, their impact, and how MBT handles them.

| Failure Mode | Impact | Mitigation |
|---|---|---|
| Model registry unavailable during serving | Serving pipeline cannot load model | `artifact_snapshot: true` freezes artifacts locally at compile time. `fallback_run_id` provides secondary model. |
| Training `run_id` deleted or corrupted | Serving pipeline resolution fails | Artifact snapshots decouple serving from registry. Immutability policy recommended for production runs. |
| Schema drift in source table | Model receives unexpected columns/types | `validation` section catches schema drift at pipeline start. `expected_columns` and `type_check` assertions. |
| Framework adapter crashes mid-training | Partial artifacts in storage | Runner writes artifacts atomically (write to temp, then rename). Retry resets the step cleanly. |
| Secrets backend unreachable | Pipeline cannot connect to data sources | `mbt debug --target prod` validates secrets access before deployment. Secrets resolved at runtime, not compile time. |
| Adapter version incompatible with mbt-core | Import errors or missing methods | New ABC methods have default implementations. Runtime capability detection for optional features. |
| Data source returns zero rows | Model training fails or produces degenerate model | `min_rows` validation check halts pipeline before training begins. |
| Concurrent writes to output table | Data corruption or incorrect predictions | Known limitation. Recommended mitigation: use table partitioning or unique run-specific table names. |
| Jinja2 template error in profiles.yaml | Compile fails with cryptic error | Phase 1 of compilation validates all Jinja2 expressions and reports errors with file + line context. |
| Plugin not installed | Compile fails | Clear error message: "No adapter 'X' found. Install: `pip install mbt-X`". `mbt deps check` validates all required plugins before deployment. |
| Storage backend full or unreachable | Step output cannot be persisted, pipeline fails | Runner retries storage operations with exponential backoff. `mbt debug` validates storage access. |
| Data window returns future data (clock skew) | Train/test contamination | Compiler validates that `execution_date` is not in the future. Data window logic uses strict boundary comparisons. |

---

## 21. Conclusion **[CHANGED]**

MBT's architecture rests on two fundamental separations: data scientists declare *what* they want in YAML, and the framework compiles it into *how* to execute it; and the core framework has zero infrastructure dependencies while every integration is a separate adapter package. This is different from tools that ask DS to write Python pipeline code (Kedro, ZenML, Metaflow) or tools that require DS to define individual tasks and their connections (Ploomber). It is closest in spirit to Ludwig's declarative approach, but applied to the broader problem of tabular ML pipelines with enterprise infrastructure concerns (profiles, pluggable orchestrators, multi-cloud executors), and to dbt's modular adapter ecosystem.

The seven things that must be built right from day one are:

1. **The pipeline YAML schema** — it is the primary API surface for data scientists. Stability, ergonomics, schema versioning, and composition (base pipelines, includes) determine adoption.

2. **The compiler with compile-time validation** — translating toggleable YAML sections into a dynamic DAG while catching config errors via `validate_config()` is the core intellectual contribution. The five-phase compilation model (resolution → schema validation → plugin validation → DAG assembly → manifest generation) must be correct and fast.

3. **The adapter contracts (ABCs)** — these are the stable interfaces that every adapter package depends on. They must include lifecycle hooks (`setup`/`teardown`), data abstraction via `MBTFrame`, and a compatibility strategy (default implementations for new methods, runtime capability detection) for evolving the contracts without breaking existing adapters.

4. **The plugin registry** — discovering installed adapters via `entry_points` and providing clear errors when adapters are missing.

5. **The artifact flow system** — how data and model artifacts move between steps, especially in distributed execution environments where steps run as separate pods with no shared filesystem. The `StoragePlugin` contract, serialization strategy, and artifact registry are critical infrastructure.

6. **The `run_id`-based serving pipeline** — resolving all training artifacts from a single `run_id` with artifact snapshots and fallback mechanisms to eliminate both schema drift and runtime fragility.

7. **Pipeline composition** — base pipelines and `!include` directives to prevent YAML duplication across teams with many models on shared data.

The manifest-as-contract pattern — where the compiled `manifest.json` bridges DS declarations and DE infrastructure — is what makes MBT practical in organizations where these are different teams with different tools. The DS never touches the orchestrator. The DE never touches model hyperparameters. The manifest is the handshake. And the modular adapter architecture means every team can compose exactly the stack they need — `mbt-core` + `mbt-h2o` + `mbt-snowflake` + `mbt-airflow` — without carrying the weight of integrations they don't use.
