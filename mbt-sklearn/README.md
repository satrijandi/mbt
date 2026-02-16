# mbt-sklearn

scikit-learn adapter for MBT (Model Build Tool).

## Installation

```bash
pip install mbt-sklearn
```

## Supported Models

### Classification
- RandomForestClassifier
- LogisticRegression
- GradientBoostingClassifier
- DecisionTreeClassifier
- AdaBoostClassifier
- SVC

### Regression
- RandomForestRegressor
- GradientBoostingRegressor
- LinearRegression
- Ridge
- Lasso

## Usage

In your pipeline YAML:

```yaml
training:
  model_training:
    framework: sklearn
    config:
      model: RandomForestClassifier
      n_estimators: 100
      max_depth: 10
      random_state: 42
```

## Configuration

The `config` section accepts any parameters supported by the chosen sklearn model. See [scikit-learn documentation](https://scikit-learn.org/) for model-specific parameters.

### Common Parameters

- `random_state`: Random seed for reproducibility (default: 42)
- `n_estimators`: Number of trees (for ensemble models)
- `max_depth`: Maximum tree depth
- `min_samples_split`: Minimum samples to split a node
- `min_samples_leaf`: Minimum samples in a leaf node

## Examples

### Logistic Regression

```yaml
model_training:
  framework: sklearn
  config:
    model: LogisticRegression
    max_iter: 1000
    C: 1.0
```

### Gradient Boosting

```yaml
model_training:
  framework: sklearn
  config:
    model: GradientBoostingClassifier
    n_estimators: 200
    learning_rate: 0.1
    max_depth: 5
```
