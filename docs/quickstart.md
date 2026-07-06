# Quickstart

From zero to a trained, registered, documented model.
Budget: well under an hour; the training itself takes seconds.

## 1. Install

```bash
pip install mbt-core mbt-xgboost mbt-mlflow
```

Python 3.11+ required. This brings the CLI (`mbt`), the XGBoost training
adapter, and MLflow tracking/registry adapters.

## 2. Scaffold a project

```bash
mbt init my_models
cd my_models
```

You get a working golden-path project: an example source, dataset, and model
spec, `profiles.yml` (also installed to `~/.mbt/`), pre-commit config,
reference CI workflows, and CODEOWNERS on `models/`.

## 3. Get data

```bash
python scripts/generate_sample_data.py
```

This writes deterministic sample subscriber data to `data/subscribers/`.
For real projects, point `sources.yml` at your own Parquet:

```yaml
sources:
  - name: lakehouse
    tables:
      - name: subscribers
        path: data/subscribers/*.parquet
```

## 4. Validate

```bash
mbt parse
```

Fast, no execution: schema validation with did-you-mean suggestions, task ↔
adapter compatibility, hyperparameter validation, DAG construction, and
cross-resource checks - all errors reported in one pass.

Add `--write-json-schema` to publish JSON Schemas under
`target/json-schemas/` for editor autocomplete (the scaffolded specs carry
`yaml-language-server` headers pointing at them).

## 5. Build

```bash
mbt build
```

What happens, in order:

1. **compile** - Jinja + profiles resolve; time windows pin against one
   anchor; data snapshots pin; every node gets a config hash and a
   transitive input hash; `target/manifest.json` is written.
2. **datasets** - DuckDB materializes train/test splits per the declared
   windows; built-in checks (`not_null`, `no_future_columns`, ...) and your
   Python data tests run.
3. **models** - each model trains in an isolated subprocess job:
   `{{ auto }}` hyperparameters resolve from the dataset profile, metrics
   compute on the pinned test split, everything logs to MLflow.
4. **gates** - thresholds and champion comparisons decide; passing models
   register to the MLflow registry in `staging`.
5. `target/run_results.json` records statuses, timings, metrics, gates,
   artifacts - machine-readable.

## 6. Inspect

```bash
mbt ls                                   # resources, tags, paths
mbt show churn_classifier                # fully compiled config
mbt docs generate && mbt docs serve      # model cards + lineage site
```

## 7. Reproduce

```bash
mbt run --manifest target/manifest.json
```

Re-executes the stored manifest verbatim - same anchor, same windows, same
snapshots, same seeds - and reproduces the metrics exactly.

## 8. Promote

```bash
mbt promote --model churn_classifier --to production
```

Refuses unless gates were recorded as passed at registration. In CI, wire
this to a reviewed `promotions.yml` change (see [GitOps & CI](gitops.md)).

## Day-to-day loops

```bash
mbt build --select churn_classifier            # one model + its data
mbt build --select tag:weekly                  # scheduled retraining set
mbt build --select state:modified+ --state s3://bucket/mbt/prod/manifests/latest.json
mbt test                                        # checks + gates, never trains
mbt evaluate --model churn_classifier --stage production --gates   # drift check
```
