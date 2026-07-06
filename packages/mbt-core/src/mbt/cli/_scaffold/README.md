# __PROJECT_NAME__

A machine-learning model project built with [mbt](https://github.com/mbt-dev/mbt) -
declarative Model-as-Code: the model config IS the model.

## Quickstart

```bash
# 1. generate sample data (replace with your own parquet later)
python scripts/generate_sample_data.py

# 2. validate specs and build the DAG (fast, no execution)
mbt parse

# 3. train + test + register everything
mbt build

# 4. inspect
mbt ls
mbt show churn_classifier
```

## Layout

- `sources.yml` - external inputs (parquet paths in dev)
- `datasets/` - declarative training-set construction (label, filters, split)
- `models/` - the model specs: task, adapter, hyperparameters, gates, registration
- `tests/` - Python data tests (`def test_*(dataset, spec) -> TestResult`)
- `macros/` - Jinja macros usable in any spec
- `profiles.yml` - environments (dev/prod); keep secrets in `{{ env_var(...) }}`
- `.github/workflows/` - PR check, prod build, promotion, weekly retrain

## Day-to-day

```bash
mbt build --select churn_classifier          # one model (+ its data)
mbt build --select state:modified+ --state state/prod/latest.json
mbt test                                     # data tests + gates, no retraining
mbt docs generate && mbt docs serve          # model cards + lineage
mbt promote --model churn_classifier --to production
```
