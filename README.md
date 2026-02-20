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

## Getting Started

This guide will take you from zero to your first trained model in 5 minutes.

### Prerequisites Check

Before installing MBT, verify you have the required tools:

**Python 3.10 or higher:**

```bash
python3 --version
```

Expected output: `Python 3.10.x` or higher

**uv package manager:**

```bash
uv --version
```

Expected output: `uv 0.x.x` or higher

If uv is not installed, see the [installation guide](https://docs.astral.sh/uv/getting-started/installation/).

### Installation

**1. Clone the repository:**

```bash
git clone https://github.com/your-org/mbt.git
cd mbt
```

**2. Install MBT and all adapters:**

```bash
uv sync
```

This installs:
- `mbt-core` — Core framework
- `mbt-sklearn` — Scikit-learn adapter
- `mbt-h2o` — H2O AutoML adapter
- `mbt-mlflow` — MLflow tracking adapter
- All dependencies

**3. Verify installation:**

```bash
uv run mbt version
```

Expected output:
```
MBT version 0.1.0
```

```bash
uv run mbt debug
```

Expected output:
```
✓ Python version: 3.10.x
✓ MBT core version: 0.1.0

Installed adapters:
  Frameworks:
    ✓ sklearn (SklearnFrameworkAdapter)
    ✓ h2o_automl (H2OAutoMLFrameworkAdapter)
  Model Registries:
    ✓ mlflow (MLflowRegistryAdapter)
  Data Connectors:
    ✓ local_file (built-in)
  Storage:
    ✓ local (built-in)
```

### Your First Pipeline: The 5-Minute Example

Let's run the telecom churn prediction example end-to-end.

**Step 1: Navigate to the example**

```bash
cd examples/telecom-churn/
```

**Step 2: Compile the pipeline**

```bash
uv run mbt compile churn_simple_v1
```

**What happens:** MBT reads `pipelines/churn_simple_v1.yaml`, validates the configuration, resolves the `dev` environment from `profiles.yaml`, and generates `target/churn_simple_v1/manifest.json` with a 5-step execution plan.

Expected output:
```
Compiling pipeline: churn_simple_v1
Target: dev

Phase 1: Resolution ✓
Phase 2: Schema Validation ✓
Phase 3: Plugin Validation ✓
Phase 4: DAG Assembly ✓
Phase 5: Manifest Generation ✓

✓ Manifest saved to: target/churn_simple_v1/manifest.json
✓ Compilation successful!

Pipeline: churn_simple_v1
Type: training
Steps: 5
```

**Step 3: Run the pipeline**

```bash
uv run mbt run --select churn_simple_v1
```

**What happens:** MBT executes the 5-step DAG:
1. Loads `sample_data/customers.csv` (20 rows)
2. Splits into train (16 rows) / test (4 rows) sets
3. Trains a RandomForestClassifier
4. Evaluates on the test set
5. Logs artifacts to MLflow

Expected output:
```
🚀 Starting pipeline: churn_simple_v1
   Run ID: run_20260220_143022
   Target: dev

▶ Executing step: load_data
  Loading data from: ./sample_data/customers.csv
  Loaded 20 rows, 5 columns
  Features: ['tenure', 'monthly_charges', 'total_charges']
  Target: churned
  ✓ Completed in 0.11s

▶ Executing step: split_data
  Stratified split on target: churned
  Train set: 16 rows (80.0%)
  Test set: 4 rows (20.0%)
  ✓ Completed in 0.17s

▶ Executing step: train_model
  Framework: sklearn
  Model: RandomForestClassifier
  Features: 3, Training samples: 16
  Training accuracy: 1.0000
  ✓ Completed in 0.08s

▶ Executing step: evaluate
  Evaluation metrics:
    accuracy: 1.0000
    precision: 1.0000
    recall: 1.0000
    f1: 1.0000
    roc_auc: 1.0000
  ✓ Completed in 0.01s

▶ Executing step: log_run
  Logging to MLflow: file://./mlruns
  Run ID: 4d5e616be6aa443aa94683ad0848757a
  ✓ Completed in 0.05s

✅ Pipeline completed successfully
   Total duration: 0.42s
   Artifacts saved to: local_artifacts/run_20260220_143022/
```

**Step 4: Check the results**

View the pipeline execution summary:

```bash
uv run mbt dag churn_simple_v1
```

Expected output:
```
DAG for pipeline: churn_simple_v1
Total steps: 5

Execution Order:
  Batch 1: load_data
  Batch 2: split_data
  Batch 3: train_model
  Batch 4: evaluate
  Batch 5: log_run

Dependencies:
  split_data → [load_data]
  train_model → [split_data]
  evaluate → [train_model]
  log_run → [evaluate]
```

Inspect the compiled manifest:

```bash
cat target/churn_simple_v1/manifest.json | head -20
```

This shows the resolved execution plan with step configurations and dependencies.

**Step 5: Explore generated artifacts**

MBT stores intermediate artifacts in timestamped run directories:

```bash
ls local_artifacts/run_*/
```

You'll see directories for each step:
- `load_data/` — Raw loaded data
- `split_data/` — Train and test sets
- `train_model/` — Trained RandomForest model
- `evaluate/` — Evaluation metrics and plots
- `log_run/` — MLflow run metadata

View the evaluation metrics:

```bash
find local_artifacts/run_* -name "*.json" -path "*/evaluate/*" -exec cat {} \;
```

### Understanding What You Just Did

You just:

1. **Compiled** a declarative YAML pipeline into an executable DAG
2. **Executed** 5 ML steps: load → split → train → evaluate → log
3. **Trained** a scikit-learn RandomForest classifier on customer churn data
4. **Evaluated** the model on a held-out test set
5. **Generated** artifacts ready for production deployment

**Key insight:** You never wrote DAG code, infrastructure logic, or framework-specific boilerplate. MBT compiled your YAML declaration into a runnable execution plan and handled all the plumbing.

### Next Steps

**Try modifying the pipeline:**

Edit `pipelines/churn_simple_v1.yaml` to change the model:

```yaml
model_training:
  framework: sklearn
  config:
    model: GradientBoostingClassifier  # Changed from RandomForestClassifier
    n_estimators: 200
    max_depth: 5
```

Then recompile and run:

```bash
uv run mbt compile churn_simple_v1
uv run mbt run --select churn_simple_v1
```

**Switch to AutoML:**

Change the framework to H2O AutoML for automated model selection:

```yaml
model_training:
  framework: h2o_automl
  config:
    max_runtime_secs: 300
    max_models: 20
    sort_metric: AUC
```

**Create your own pipeline:**

```bash
# Return to project root
cd ../..

# Initialize a new project
uv run mbt init my_ml_project
cd my_ml_project

# Edit pipelines/my_model_v1.yaml with your data source and model config
# See "Quick Reference" section below for YAML structure
```

**Explore other examples:**

```bash
cd examples/telecom-churn/

# Try a pipeline with preprocessing
uv run mbt compile churn_logistic_v1
uv run mbt run --select churn_logistic_v1
```

### Troubleshooting

**"Permission denied" or MLflow path errors:**

MLflow may try to write to the wrong directory. Ensure your `profiles.yaml` uses a relative path:

```yaml
mlflow:
  tracking_uri: "file://./mlruns"  # Note: file:// prefix with relative path
```

**"Manifest not found" error:**

You need to compile before running:

```bash
uv run mbt compile <pipeline_name>   # First
uv run mbt run --select <pipeline_name>  # Then
```

**"ModuleNotFoundError: No module named 'mbt'":**

Ensure you're using the `uv run` prefix:

```bash
# Option 1: Use uv run (recommended)
uv run mbt compile ...

# Option 2: Activate the virtual environment
source .venv/bin/activate
mbt compile ...
```

**Adapter not found (e.g., "sklearn adapter not available"):**

Install the specific adapter:

```bash
uv pip install -e mbt-sklearn/
```

Or reinstall all packages:

```bash
uv sync
```

**H2O connection errors:**

H2O AutoML requires Java 8+ and starts an H2O server. If you see connection errors, ensure Java is installed:

```bash
java -version
```

For local development without H2O, stick to `framework: sklearn`.

---

## Quick Reference

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

## Detailed Example: Telecom Churn Prediction

The `examples/telecom-churn/` project demonstrates a complete ML workflow for binary classification.

### Project Structure

```
telecom-churn/
├── pipelines/
│   ├── _base_churn.yaml               # Base template (inherited by variants)
│   ├── churn_simple_v1.yaml          # Basic RandomForest pipeline
│   ├── churn_logistic_v1.yaml        # LogisticRegression variant
│   └── churn_gradient_boosting_v1.yaml
├── sample_data/
│   ├── customers.csv                  # Training data (20 rows)
│   └── customers_to_score.csv         # Scoring data for batch predictions
├── includes/
│   └── telecom_schema.yaml            # Reusable schema definition
├── profiles.yaml                      # Environment-specific configuration
├── target/                            # Generated manifests (created by compile)
└── local_artifacts/                   # Execution outputs (created by run)
```

### Dataset Overview

The `customers.csv` dataset contains customer telemetry data:

```csv
customer_id,tenure,monthly_charges,total_charges,churned
CUST_00001,12,45.50,546.00,0
CUST_00002,36,89.20,3211.20,1
...
```

- **Features:** `tenure` (months as customer), `monthly_charges` (monthly bill), `total_charges` (lifetime charges)
- **Target:** `churned` (0 = stayed, 1 = left)
- **Size:** 20 samples (16 train, 4 test with 80/20 stratified split)
- **Primary Key:** `customer_id` (excluded from features)

### Pipeline Walkthrough: churn_simple_v1.yaml

This pipeline demonstrates the minimal configuration for binary classification:

```yaml
schema_version: 1

project:
  name: churn_simple_v1
  problem_type: binary_classification  # Determines which metrics to compute
  owner: data_science_team
  tags: [churn, telecom, phase1]

training:
  data_source:
    label_table: customers              # Loads sample_data/customers.csv

  schema:
    target:
      label_column: churned             # Column containing 0/1 labels
      classes: [0, 1]
      positive_class: 1                 # 1 = customer churned (positive outcome)
    identifiers:
      primary_key: customer_id          # Excluded from features

  model_training:
    framework: sklearn                  # Use scikit-learn adapter
    config:
      model: RandomForestClassifier     # Model class name
      n_estimators: 100                 # Model hyperparameters

  evaluation:
    primary_metric: roc_auc             # Optimization target
    additional_metrics: [accuracy, f1, precision, recall]
    generate_plots: true                # Create confusion matrix, ROC curve
```

**This compiles to 5 steps:**

1. `load_data` — Loads customers.csv, separates features and target
2. `split_data` — Creates stratified 80/20 train/test split
3. `train_model` — Trains RandomForest on training set
4. `evaluate` — Computes metrics on test set, generates plots
5. `log_run` — Logs model and metrics to MLflow

### Running Step-by-Step

**Navigate to the example:**

```bash
cd examples/telecom-churn/
```

**Compile the pipeline:**

```bash
uv run mbt compile churn_simple_v1
```

This:
- Parses `churn_simple_v1.yaml` and validates schema
- Resolves `dev` target from `profiles.yaml`
- Builds 5-step DAG based on enabled features
- Generates `target/churn_simple_v1/manifest.json`

**Inspect the manifest:**

```bash
cat target/churn_simple_v1/manifest.json
```

Key sections in the manifest:
- `metadata.pipeline_type`: `"training"`
- `steps`: Dict of 5 step configurations with resolved executors
- `dag.execution_batches`: `[["load_data"], ["split_data"], ["train_model"], ["evaluate"], ["log_run"]]`
- `profile_config`: Resolved storage paths, data connectors, MLflow URI

**Run the pipeline:**

```bash
uv run mbt run --select churn_simple_v1
```

**Check results:**

View run summary:
```bash
uv run mbt dag churn_simple_v1
```

Inspect generated artifacts:
```bash
ls local_artifacts/run_*/
```

You'll see:
- `load_data/` — Loaded DataFrame
- `split_data/` — Train and test sets
- `train_model/` — Trained model (`.pkl` file)
- `evaluate/` — Metrics JSON and plots (if `generate_plots: true`)
- `log_run/` — MLflow run metadata

### Trying Different Configurations

**Switch to Logistic Regression:**

Try a different algorithm:

```bash
uv run mbt compile churn_logistic_v1
uv run mbt run --select churn_logistic_v1
```

This pipeline uses `LogisticRegression` instead of `RandomForestClassifier`.

**Switch to H2O AutoML:**

Edit `pipelines/churn_simple_v1.yaml`:

```yaml
model_training:
  framework: h2o_automl
  config:
    max_runtime_secs: 300
    max_models: 20
    sort_metric: AUC
```

Recompile and run:

```bash
uv run mbt compile churn_simple_v1
uv run mbt run --select churn_simple_v1
```

H2O AutoML will train 20 models and automatically select the best one by AUC.

**Add preprocessing:**

Try the gradient boosting pipeline which includes normalization:

```bash
uv run mbt compile churn_gradient_boosting_v1
uv run mbt run --select churn_gradient_boosting_v1
```

This adds preprocessing steps to the DAG.

### Understanding Profiles

The `profiles.yaml` file defines WHERE and HOW the pipeline runs:

```yaml
telecom-churn:
  target: dev                           # Default environment

  outputs:
    dev:
      executor:
        type: local                     # Run steps sequentially on local machine
      storage:
        type: local
        config:
          base_path: ./local_artifacts  # Where to save intermediate artifacts
      data_connector:
        type: local_file
        config:
          data_path: ./sample_data      # Where to find CSV files
      mlflow:
        tracking_uri: "file://./mlruns" # MLflow experiment tracking location

    staging:
      executor:
        type: local
      storage:
        type: local
        config:
          base_path: ./staging_artifacts
      data_connector:
        type: local_file               # Could be: snowflake, postgres, etc.
        config:
          data_path: ./sample_data
      mlflow:
        tracking_uri: "file://./mlruns_staging"

    prod:
      # Production would use Snowflake, S3, remote MLflow, etc.
      executor:
        type: local                    # Future: kubernetes, airflow
      storage:
        type: local                    # Future: s3, seaweedfs
      data_connector:
        type: local_file               # Future: snowflake
      mlflow:
        tracking_uri: "file://./mlruns_prod"
```

**Run against a different target:**

```bash
uv run mbt run --select churn_simple_v1 --target staging
```

This uses the `outputs.staging` configuration, saving artifacts to `staging_artifacts/` and logging to `mlruns_staging/`.

**Key insight:** The same `churn_simple_v1.yaml` pipeline runs across dev/staging/prod without modification. Only the profile changes.

---

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
