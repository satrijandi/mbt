# Quickstart

From zero to a trained, registered, documented model.
Budget: well under an hour; the training itself takes seconds.

## 1. Install

The packages are not on PyPI yet, so install from a source checkout:

```bash
git clone https://github.com/satrijandi/mbt && cd mbt
uv sync && source .venv/bin/activate     # puts `mbt` on PATH
```

Once the packages are published this becomes
`pip install mbt-core mbt-xgboost mbt-mlflow`.
Python 3.11+ required (3.11 through 3.14 are tested in CI). This brings the
CLI (`mbt`), the XGBoost training adapter, and MLflow tracking/registry
adapters.

## 2. Scaffold a project

```bash
mbt init my_models
cd my_models
```

You get a working golden-path project: an example source, dataset, and model
spec, `profiles.yml` (also installed to `~/.mbt/`), pre-commit config,
a pinned CI install set (`requirements.txt`), reference CI workflows, and
CODEOWNERS on `models/`.

## 3. Get data

```bash
python scripts/generate_sample_data.py
```

This writes deterministic sample data: the training subscribers
(`data/subscribers/`) plus a fresh scoring batch and matured outcomes
(`data/scoring_batch/`, `data/churn_outcomes/`) that the score and monitor
steps below consume.
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
   windows; built-in checks (`not_null`, `no_future_columns`,
   `label_leakage_scan` - the leakage scan runs by default) and your
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

## 9. Score and monitor

```bash
mbt score      # batch-score scoring/churn_scoring.yml with the production champion
mbt monitor    # once outcomes mature: realized metrics for stored predictions
```

The scaffolded `scoring/churn_scoring.yml` is a whole serving pipeline in
one config: input batch, prediction sink, distribution-shift monitors
against the champion's training-time baseline, and delayed ground-truth
evaluation (see [Concepts](concepts.md#batch-scoring-and-monitoring)).
A shift breach or failed realized-metric gate exits 2, like any gate.

## Day-to-day loops

```bash
mbt build --select churn_classifier            # one model + its data
mbt build --select tag:weekly                  # scheduled retraining set
mbt build --select state:modified+ --state s3://bucket/mbt/prod/manifests/latest.json
mbt test                                        # checks + gates, never trains
mbt evaluate --model churn_classifier --stage production --gates   # champion decay check on fresh data
mbt score --select tag:daily                    # scheduled batch scoring set
mbt monitor                                     # evaluate matured predictions
```
