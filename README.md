# MBT (Model Build Tool)

**dbt for DS and DE** — A declarative ML pipeline framework for tabular machine learning.

Data scientists define ML workflows in YAML. MBT compiles them into executable DAGs that run locally or on any orchestrator.

```
YAML Pipeline  ──→  mbt compile  ──→  manifest.json  ──→  mbt run  ──→  Results
(what you want)                       (execution plan)                  (metrics, model, predictions)
```

## Why MBT?

| Problem | MBT's Answer |
|---------|-------------|
| Data scientists writing boilerplate DAG code | Declare pipelines in YAML, MBT generates the DAG |
| ML code tightly coupled to infrastructure | Separation of concerns — scientists own YAML, engineers own profiles |
| Config errors discovered at runtime | Compile-time validation catches issues before training starts |
| Switching between frameworks requires rewrites | Pluggable adapters — swap sklearn for H2O by changing one line |
| Dev/prod environment drift | Profile-based configuration, same pipeline runs everywhere |

## Quick Start

### Prerequisites

- Python 3.10+
- [uv](https://docs.astral.sh/uv/) package manager

### Installation

```bash
git clone https://github.com/your-org/mbt.git
cd mbt

# Install all packages in development mode
uv sync
```

### Your First Pipeline

**1. Scaffold a project:**

```bash
uv run mbt init my_project
cd my_project
```

**2. Define a pipeline** in `pipelines/my_model_v1.yaml`:

```yaml
schema_version: 1

project:
  name: my_model_v1
  problem_type: binary_classification
  owner: data_science_team

training:
  data_source:
    label_table: customers

  schema:
    target:
      label_column: churned
      classes: [0, 1]
      positive_class: 1
    identifiers:
      primary_key: customer_id

  model_training:
    framework: sklearn
    config:
      model: RandomForestClassifier
      n_estimators: 100

  evaluation:
    primary_metric: roc_auc
    additional_metrics: [accuracy, f1, precision, recall]
```

**3. Compile and run:**

```bash
uv run mbt compile my_model_v1
uv run mbt run --select my_model_v1
```

MBT compiles the YAML into a 5-step DAG and executes it:

```
load_data → split_data → train_model → evaluate → log_run
```

### Run the Example

```bash
cd examples/telecom-churn/
uv run mbt compile churn_simple_v1
uv run mbt run --select churn_simple_v1
```

## Two-Layer Architecture

MBT enforces a strict separation between **declaration** and **execution**:

```
LAYER 1: DECLARATION (Data Scientist)
  pipelines/churn_v1.yaml
  "I want a random forest on this table with these splits"
            ↓  mbt compile
LAYER 2: EXECUTION (Framework + Data Engineering)
  manifest.json
  Resolved DAG with concrete tasks, executors, and dependencies
```

| Role | Owns | Doesn't Touch |
|------|------|---------------|
| Data Scientist | `pipelines/*.yaml`, `lib/` custom transforms | `profiles.yaml`, infrastructure |
| Data Engineer | `profiles.yaml`, orchestrator config, CI/CD | Pipeline YAML, hyperparameters |

## Pipeline Steps

MBT builds the DAG from the steps you enable. A full training pipeline can include up to 10 steps:

```
load_data → join_tables → validate_data → split_data → normalize → encode → feature_selection → train_model → evaluate → log_run
```

A serving pipeline follows a separate path:

```
load_model → load_scoring_data → apply_transforms → predict → publish
```

Steps are only included when their section is present in the YAML. A minimal pipeline (data source + model config + evaluation) compiles to just 5 steps.

### Available Steps

| Step | Purpose |
|------|---------|
| `load_data` | Load data from configured source (CSV, Parquet, database) |
| `join_tables` | Multi-table joins with fan-out protection |
| `validate_data` | Data quality checks (nulls, ranges, types, uniqueness) |
| `split_data` | Train/test/validation splitting |
| `normalize` | Feature scaling (Standard, MinMax, Robust) |
| `encode` | Categorical encoding (one-hot, label) |
| `feature_selection` | Feature filtering (variance, correlation, mutual information) |
| `train_model` | Model training via framework adapter |
| `evaluate` | Compute metrics and generate plots |
| `log_run` | Log artifacts to model registry |

## Adapters

MBT's core is framework-agnostic. All ML framework and infrastructure integrations are separate packages, discovered via Python entry points.

### Framework Adapters

| Package | Framework | Use Case |
|---------|-----------|----------|
| `mbt-sklearn` | scikit-learn | Manual model selection (RandomForest, LogisticRegression, GBM, SVC, etc.) |
| `mbt-h2o` | H2O AutoML | AutoML-first rapid prototyping (requires Java 8+) |
| `mbt-sagemaker` | AWS SageMaker | Cloud training (XGBoost, LinearLearner, KNN) |

### Registry Adapters

| Package | Service | Purpose |
|---------|---------|---------|
| `mbt-mlflow` | MLflow | Experiment tracking, model registry, artifact storage |

### Switching Frameworks

Change one line in your pipeline YAML:

```yaml
# From sklearn...
model_training:
  framework: sklearn
  config:
    model: RandomForestClassifier

# ...to H2O AutoML
model_training:
  framework: h2o_automl
  config:
    max_runtime_secs: 300
    max_models: 20
```

## Profiles

Profiles separate environment configuration from pipeline logic. Define them in `profiles.yaml`:

```yaml
config:
  default_target: dev

targets:
  dev:
    executor:
      type: local
    storage:
      type: local
      base_path: ./local_artifacts
    data_connector:
      type: local_file
      data_path: ./sample_data
    mlflow:
      tracking_uri: sqlite:///mlflow.db

  prod:
    executor:
      type: local
    storage:
      type: local
      base_path: /mnt/shared/artifacts
    data_connector:
      type: snowflake
      account: "{{ env_var('SNOWFLAKE_ACCOUNT') }}"
    mlflow:
      tracking_uri: "{{ env_var('MLFLOW_TRACKING_URI') }}"
```

Run against a specific target:

```bash
uv run mbt run --select my_pipeline --target prod
```

## Pipeline Composition

Avoid duplication with base pipelines and includes.

**Base pipeline** (`pipelines/_base_churn.yaml`):

```yaml
schema_version: 1

project:
  problem_type: binary_classification
  owner: data_science_team

training:
  data_source:
    label_table: customers
  schema:
    target: !include ../includes/telecom_schema.yaml
  evaluation:
    primary_metric: roc_auc
```

**Child pipeline** inherits and overrides:

```yaml
schema_version: 1
base_pipeline: _base_churn

project:
  name: churn_gradient_boosting_v1

training:
  model_training:
    framework: sklearn
    config:
      model: GradientBoostingClassifier
      n_estimators: 200
```

## CLI Reference

| Command | Description |
|---------|-------------|
| `mbt init <name>` | Scaffold a new MBT project |
| `mbt compile <pipeline>` | Compile YAML to `manifest.json` |
| `mbt run --select <pipeline>` | Execute a compiled pipeline |
| `mbt validate [pipeline]` | Validate YAML schema (all pipelines if none specified) |
| `mbt test [pipeline]` | Run DAG assertion tests |
| `mbt dag <pipeline>` | Visualize pipeline DAG (`--mermaid` for Mermaid output) |
| `mbt debug` | Show installed adapters and test connections |
| `mbt version` | Print MBT version |

### Common Options

- `--target / -t` — Select environment target (default: `dev`)
- `--vars key=value` — Pass variables for Jinja2 template resolution

## Compilation

MBT's compiler runs 5 phases before any code executes:

1. **Resolution** — Resolve `base_pipeline` inheritance and `!include` fragments
2. **Schema Validation** — Validate YAML against Pydantic models
3. **Plugin Validation** — Call each adapter's `validate_config()` for compile-time checks
4. **DAG Assembly** — Build step graph based on enabled features, topologically sort
5. **Manifest Generation** — Write `manifest.json` with resolved executors and dependencies

Errors are caught at compile time, not at runtime.

## Project Structure

```
mbt/
├── mbt-core/                 # Core framework
│   └── src/mbt/
│       ├── cli.py            # CLI entry point
│       ├── core/             # Compiler, runner, DAG, manifest, context
│       ├── steps/            # Built-in pipeline steps
│       ├── contracts/        # Adapter interfaces (ABCs)
│       ├── config/           # YAML schema, profiles, Jinja2 loader
│       ├── builtins/         # Default implementations (local storage, CSV connector)
│       ├── testing/          # Test utilities
│       └── observability/    # Logging and metrics
├── mbt-sklearn/              # scikit-learn adapter
├── mbt-h2o/                  # H2O AutoML adapter
├── mbt-mlflow/               # MLflow registry adapter
├── mbt-sagemaker/            # AWS SageMaker adapter
└── examples/
    └── telecom-churn/        # Reference implementation
```

## Infrastructure Setup

For production deployments, set up the following infrastructure:

1. **Zot** — Container registry
2. **Gitea** — Git hosting (1 account for DS, 1 account for DE)
3. **Woodpecker CI** — CI/CD pipeline runner
4. **H2O server** — For H2O AutoML framework adapter
5. **Kubernetes** — Container orchestration for distributed execution
6. **SeaweedFS** — Distributed storage for artifacts
7. **PostgreSQL** — Offline data warehouse
8. **MLflow** — Model registry and experiment tracking (backed by SeaweedFS + PostgreSQL)

For local development, none of these are required — MBT runs fully locally with built-in defaults.

## Development

### Running Tests

```bash
# Core tests
uv run pytest mbt-core/tests/ -v

# Adapter tests
uv run pytest mbt-sklearn/tests/ -v
uv run pytest mbt-h2o/tests/ -v
uv run pytest mbt-mlflow/tests/ -v
uv run pytest mbt-sagemaker/tests/ -v
```

### Writing an Adapter

Implement the `FrameworkPlugin` interface:

```python
from mbt.contracts.framework import FrameworkPlugin

class MyFramework(FrameworkPlugin):
    name = "my_framework"

    def validate_config(self, config: dict, problem_type: str) -> None:
        """Validate config at compile time."""
        ...

    def train(self, X_train, y_train, config: dict, problem_type: str):
        """Train and return a model."""
        ...

    def predict(self, model, X):
        """Generate predictions."""
        ...
```

Register it via entry points in `pyproject.toml`:

```toml
[project.entry-points."mbt.frameworks"]
my_framework = "my_package.framework:MyFramework"
```

## Inspirations

- **[dbt](https://www.getdbt.com/)** — Profiles, CLI ergonomics, compile-before-run philosophy
- **[Ludwig](https://ludwig.ai/)** — Fully declarative YAML-driven ML
- **[MLflow Recipes](https://mlflow.org/docs/latest/recipes.html)** — Templated step execution with profile switching
- **[Kedro](https://kedro.org/)** — Layered configuration and data catalog patterns

## License

See [LICENSE](LICENSE) for details.
