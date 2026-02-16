# mbt-h2o

H2O AutoML adapter for MBT (Model Build Tool).

## What is H2O AutoML?

H2O AutoML automatically trains and tunes multiple machine learning models, then selects the best performer. This is the **AutoML-first** approach: get strong models without manual hyperparameter tuning.

H2O AutoML trains:
- Generalized Linear Models (GLM)
- Random Forest (DRF)
- Gradient Boosting Machine (GBM)
- XGBoost
- Deep Learning
- Stacked Ensemble (combines best models)

## Installation

```bash
pip install mbt-h2o
```

**Note**: H2O requires Java 8+ to be installed.

## Usage

In your pipeline YAML:

```yaml
training:
  model_training:
    framework: h2o_automl
    config:
      max_runtime_secs: 3600    # 1 hour training budget
      max_models: 20            # Train up to 20 models
      sort_metric: AUC          # Optimize for ROC AUC
      seed: 42                  # For reproducibility
```

## Configuration

### Key Parameters

- **max_runtime_secs** (int): Maximum training time in seconds
  - Default: 3600 (1 hour)
  - H2O will stop training after this time limit

- **max_models** (int): Maximum number of models to train
  - Default: 20
  - Includes base models + ensembles

- **sort_metric** (str): Metric to optimize
  - Classification: "AUC", "AUCPR", "logloss", "mean_per_class_error"
  - Regression: "deviance", "RMSE", "MSE", "MAE", "RMSLE"
  - Default: "AUTO" (H2O chooses based on problem type)

- **seed** (int): Random seed for reproducibility
  - Default: 42

- **nfolds** (int): Number of cross-validation folds
  - Default: 5
  - Set to 0 to disable cross-validation

- **balance_classes** (bool): Balance class distribution
  - Default: false
  - Useful for imbalanced classification problems

## Examples

### Quick Training (5 minutes)

```yaml
model_training:
  framework: h2o_automl
  config:
    max_runtime_secs: 300
    max_models: 10
    sort_metric: AUC
```

### Production Training (1 hour)

```yaml
model_training:
  framework: h2o_automl
  config:
    max_runtime_secs: 3600
    max_models: 50
    sort_metric: AUC
    nfolds: 10
    seed: 42
```

### Imbalanced Classification

```yaml
model_training:
  framework: h2o_automl
  config:
    max_runtime_secs: 1800
    balance_classes: true
    sort_metric: AUCPR  # Area under precision-recall curve
```

### Regression

```yaml
model_training:
  framework: h2o_automl
  config:
    max_runtime_secs: 3600
    max_models: 30
    sort_metric: RMSE
```

## AutoML-First Philosophy

With H2O AutoML, you get:

✅ **No hyperparameter tuning needed** - H2O handles it automatically
✅ **Multiple algorithms tried** - GLM, Random Forest, GBM, XGBoost, Deep Learning
✅ **Automatic ensemble** - Stacked ensemble combines best models
✅ **Fast iteration** - Just adjust time budget, not individual parameters
✅ **Production-ready models** - Best model selected by cross-validation

Compare this to manual sklearn tuning:

**Before (manual tuning)**:
```yaml
framework: sklearn
config:
  model: RandomForestClassifier
  n_estimators: ???  # What value?
  max_depth: ???     # What value?
  min_samples_split: ???  # What value?
  # Hours of experimentation...
```

**After (AutoML)**:
```yaml
framework: h2o_automl
config:
  max_runtime_secs: 3600  # Train for 1 hour, get best model
```

## Memory Configuration

H2O uses 4GB of memory by default. To increase:

```bash
export H2O_MAX_MEM_SIZE="16G"
mbt run --select my_pipeline
```

## Resources

- [H2O AutoML Documentation](https://docs.h2o.ai/h2o/latest-stable/h2o-docs/automl.html)
- [H2O Installation Guide](https://docs.h2o.ai/h2o/latest-stable/h2o-docs/downloading.html)
