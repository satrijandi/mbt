# mbt - Model Build Tool

**"dbt for machine learning models."**
A model is a reviewed YAML spec - data, algorithm, hyperparameters, quality gates, registration target.
Pluggable adapters execute training; a compiled manifest pins data snapshots, config hashes, seeds, and environment digests so runs are reproducible; state-aware selection retrains only what changed.

> **The model config IS the model.**

```yaml
# models/churn_classifier.yml
models:
  - name: churn_classifier
    task: binary_classification
    adapter: xgboost
    owner: growth-ds@company.com
    dataset: ref('churn_training_set')
    target: churned_90d
    features:
      exclude: [user_id, email]          # explicit leakage guards
    hyperparameters:
      max_depth: 6
      scale_pos_weight: "{{ auto }}"     # adapter computes from class balance
    evaluation:
      protocol: {split: temporal}
      metrics: [pr_auc, roc_auc]
      gates:
        - metric: pr_auc
          compare_to: production          # champion/challenger
          min_delta: 0.005
    registration:
      name: churn_classifier
      stage_on_pass: staging
    seed: 42
```

## Five minutes to a trained, registered model

```bash
pip install mbt-core mbt-xgboost mbt-mlflow

mbt init my_models && cd my_models
python scripts/generate_sample_data.py   # or point sources.yml at your parquet
mbt build                                # datasets -> training -> tests -> gates -> registry
mbt docs generate && mbt docs serve      # model cards + lineage
```

`mbt build` compiles your specs into a pinned manifest, materializes datasets
with DuckDB, trains each model in an isolated job, evaluates gates against
the registry champion, and registers passing artifacts in MLflow - all from
YAML you can review in a PR.

## Why

Data science teams glue together notebooks, ad-hoc scripts, and bespoke
pipeline code; the *logic* of a model gets buried in imperative Python.
dbt fixed this for analytics by making transformations declarative,
versioned, testable, and dependency-aware. mbt applies the same philosophy
to model building:

| dbt | mbt |
|---|---|
| model = SQL + config | model = **declarative YAML** (+ optional `hooks.py`) |
| adapters: Snowflake, BigQuery... | adapters: **XGBoost, LightGBM**, sklearn/PyTorch later |
| `dbt run` materializes tables | `mbt run` trains & registers **model artifacts** |
| `dbt test` | `mbt test`: data checks + **metric gates vs the champion** |
| `ref()` DAG of models | `ref()` DAG of **datasets -> models** |
| `state:modified` rebuilds | `state:modified` **retrains only what changed** |

## The GitOps loop

1. DS edits a model spec on a branch; the PR check compiles and runs
   `mbt build --select state:modified+ --state <prod manifest>` on the dev
   target - only the changed subgraph retrains.
2. The PR comment shows metrics vs the production champion, gate results,
   the retrained node list, and a cost estimate.
3. Merge builds prod, registers passing models to staging, and publishes
   the manifest.
4. A reviewed `promotions.yml` change (or an approved dispatch) runs
   `mbt promote` - which refuses versions whose gates were not recorded
   as passed.

Reference GitHub Actions for all of this ship inside `mbt init`.

## Reproducibility

The compiled manifest pins *everything*: data snapshot ids, resolved time
windows against a single anchor, config + transitive input hashes, seeds,
and an environment digest.

```bash
mbt run --manifest target/manifest.json   # re-execute a stored manifest verbatim
```

reproduces the original metrics exactly for the XGBoost adapter (documented
determinism tiers per adapter otherwise).

## Repository layout

| Package | What it is |
|---|---|
| `packages/mbt-core` | CLI, parsing, DAG, compile/manifest, execution engine, gates, state, docs |
| `packages/mbt-adapter-base` | Versioned adapter contracts + interchange types + **compliance suite** |
| `packages/mbt-xgboost` | XGBoost training adapter (exact determinism tier, ONNX extra) |
| `packages/mbt-lightgbm` | LightGBM adapter - built against public contracts only (the extensibility proof) |
| `packages/mbt-mlflow` | MLflow tracking + registry adapters |
| `packages/mbt-optuna` | Optuna tuning engine (seeded TPE, per-target trial caps) |
| `packages/mbt-testing` | Fake adapters for testing mbt projects without frameworks |
| `examples/churn_demo` | The demo project used by golden and E2E tests |

## Development

```bash
git clone <repo> && cd mbt
uv sync                                   # whole workspace, all extras
uv run pytest -m "not e2e"               # fast suite
uv run pytest -m e2e                      # full CLI E2E (XGBoost + MLflow)
uv run ruff check . && uv run mypy packages/mbt-core/src
uv run pre-commit install                 # hooks: ruff, yamllint, mypy
```

Design decisions live in [`docs/adr/`](docs/adr/); the docs site sources are
under [`docs/`](docs/) (`uv run mkdocs serve`).

## Status

v0.1: the full PR -> CI -> registry -> promotion loop works end-to-end for
binary classification on local Parquet with XGBoost/LightGBM + MLflow.
See `docs/roadmap.md` for what lands in v1 (remote compute, sklearn/PyTorch,
Feast, ensembles, batch scoring).
