# mbt-mlflow

MLflow model registry adapter for MBT (Model Build Tool).

## What is MLflow?

MLflow is an open-source platform for managing the ML lifecycle, including:
- Experiment tracking (metrics, parameters, artifacts)
- Model registry and versioning
- Model deployment
- Reproducibility

## Installation

```bash
pip install mbt-mlflow
```

## Usage

### Configuration

In your `profiles.yaml`:

```yaml
my-project:
  target: dev

  outputs:
    dev:
      # ... other config ...
      mlflow:
        tracking_uri: "sqlite:///mlruns.db"
        experiment_name: "my_experiment"

    prod:
      mlflow:
        tracking_uri: "https://mlflow.mycompany.com"
        experiment_name: "production"
```

### Automatic Logging

MBT automatically logs to MLflow when you use the `log_run` step (added in Phase 2):

```yaml
# In your pipeline YAML
training:
  # ... data, model training, evaluation ...
```

The framework will automatically:
- Log all evaluation metrics
- Log all training parameters
- Save the trained model
- Save all artifacts (scaler, encoder, feature selector, etc.)
- Tag the run with framework, problem_type, etc.

### Loading Models for Serving

In serving pipelines, reference the MLflow run_id:

```yaml
serving:
  model_source:
    registry: mlflow
    run_id: "{{ var('run_id') }}"  # Pass via command line
```

Then run:

```bash
mbt run --select serving_pipeline --vars run_id=abc123def456
```

## MLflow UI

View your experiments in the MLflow UI:

```bash
# For local SQLite backend
mlflow ui --backend-store-uri sqlite:///mlruns.db

# For remote server
mlflow ui --backend-store-uri https://mlflow.mycompany.com
```

Open http://localhost:5000 to see:
- All training runs
- Metrics comparison
- Parameter comparison
- Artifact downloads

## What Gets Logged

For each training run, MBT logs:

### Metrics
- All evaluation metrics (accuracy, ROC AUC, precision, recall, f1, etc.)
- Training metrics (if available from framework)

### Parameters
- Framework name (sklearn, h2o_automl, etc.)
- Model configuration (n_estimators, max_depth, etc.)
- Problem type (binary_classification, regression, etc.)
- All hyperparameters

### Artifacts
- **model**: Trained model (logged as MLflow pyfunc model)
- **scaler**: Normalization scaler (if used)
- **encoder**: Categorical encoder (if used)
- **feature_selector**: Feature selector (if used)
- Any other pipeline artifacts

### Tags
- `pipeline_name`: Name of the training pipeline
- `framework`: ML framework used (sklearn, h2o_automl)
- `problem_type`: Classification or regression
- `target`: Environment (dev, staging, prod)
- Custom tags from pipeline configuration

## Example: Complete Workflow

### 1. Train a Model

```bash
cd examples/telecom-churn
mbt run --select churn_training_v1
```

Output:
```
✅ Pipeline completed successfully
   MLflow run ID: abc123def456
   View in MLflow UI: http://localhost:5000/#/experiments/1/runs/abc123def456
```

### 2. View in MLflow UI

```bash
mlflow ui
```

Navigate to the run, see metrics:
- ROC AUC: 0.87
- Accuracy: 0.82
- F1: 0.75

### 3. Use Model in Serving

```bash
mbt run --select churn_serving_v1 --vars run_id=abc123def456
```

The serving pipeline automatically:
- Loads the model from MLflow
- Loads all preprocessing artifacts (scaler, encoder)
- Applies transforms in correct order
- Generates predictions

## Local vs Remote MLflow

### Local (Development)

```yaml
mlflow:
  tracking_uri: "sqlite:///mlruns.db"
```

- Simple SQLite database
- No server needed
- Great for development

### Remote (Production)

```yaml
mlflow:
  tracking_uri: "https://mlflow.mycompany.com"
```

- Centralized tracking server
- Team collaboration
- Production model registry

## Authentication

For remote MLflow servers with authentication:

```bash
export MLFLOW_TRACKING_USERNAME=myuser
export MLFLOW_TRACKING_PASSWORD=mypassword
# or use token:
export MLFLOW_TRACKING_TOKEN=mytoken

mbt run --select my_pipeline
```

## Resources

- [MLflow Documentation](https://mlflow.org/docs/latest/index.html)
- [MLflow Tracking](https://mlflow.org/docs/latest/tracking.html)
- [MLflow Model Registry](https://mlflow.org/docs/latest/model-registry.html)
