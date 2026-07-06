# Spec reference

Machine-readable JSON Schemas for every file kind are published by
`mbt parse --write-json-schema` (into `target/json-schemas/`); the scaffolded
specs reference them via `yaml-language-server` headers for editor
autocomplete. This page summarizes the shapes.

## mbt_project.yml

```yaml
name: churn_project          # ^[a-z][a-z0-9_]*$
version: "0.1.0"
require_mbt_version: ">=0.1,<0.2"   # optional PEP 440 guard
vars: {pr_auc_floor: 0.42}          # project-level var defaults
model_defaults: {adapter: xgboost}  # merged under every model spec
model_paths: [models]               # discovery paths (defaults shown)
dataset_paths: [datasets]
test_paths: [tests]
macro_paths: [macros]
```

## profiles.yml

```yaml
<project_name>:
  target: dev                       # default target
  outputs:
    dev:
      data:     {adapter: local,  config: {root: .}}
      tracking: {adapter: mlflow, config: {uri: "sqlite:///mlflow.db"}}
      registry: {adapter: mlflow, config: {uri: "sqlite:///mlflow.db"}}
      compute:  {adapter: local}                    # optional, default local
      artifact_store: file://./target/artifacts     # file:// (s3:// in v1)
      threads: 4                                    # optional, default 1
      vars: {sample_fraction: 0.1, max_tuning_trials: 5}
```

Search order: `--profiles-dir`, `$MBT_PROFILES_DIR`, `./profiles.yml`,
`~/.mbt/profiles.yml`. Jinja renders before validation; secrets via
`{{ env_var('NAME') }}` only.

## sources.yml

```yaml
sources:
  - name: lakehouse
    tables:
      - name: subscribers
        path: data/subscribers/*.parquet   # glob under the data adapter root
        format: parquet
```

## datasets/*.yml

Data comes from exactly one of ``source`` (a single table) or ``inputs``
(feature tables joined onto a label table):

```yaml
datasets:
  - name: churn_training_set
    source: source('lakehouse', 'subscribers')
    label:
      column: churned_90d
      definition: "cancelled within 90d of snapshot_date"   # for model cards
    filters: ["is_active = true", "tenure_days >= 30"]      # SQL, ANDed
    split:
      strategy: temporal            # default; random needs explicit seed
      time_column: snapshot_date
      train: "-180d:-28d"           # window expressions vs the anchor
      test: "-28d:now"
      validation: "-42d:-28d"       # optional; else carved when tuning needs it
    checks:                         # run at every dataset build
      - no_future_columns
      - label_leakage_scan
      - class_balance_report        # report-only
      - schema: {columns: {churned_90d: int64}}
      - not_null: {columns: [churned_90d]}
    tests: [test_label_is_binary]   # bind Python data tests by name (optional)
    snapshot: "sha256:..."          # explicit pin (optional; normally compile pins)
    tags: [churn]
```

**Window expressions:** `"<start>:<end>"`, each bound a signed duration
(`-180d`, `-12h`, `2w`), `now`, or an ISO date/timestamp; bare `"28d"` is
sugar for `"-28d:now"`.

**Random splits:** `strategy: random` uses fractions (`train: "0.8"`),
requires `seed`, and supports `stratify_by: <column>`.

### Multi-table datasets and sampling keys

```yaml
datasets:
  - name: churn_training_set
    inputs:
      label: source('snowflake', 'churn_labels')      # the spine: defines examples
      features:
        - source('snowflake', 'customer_features')
        - source('snowflake', 'usage_features')
      join_key: [customer_id, snapshot_date]
      join: left                                       # default; or inner
    label:
      column: churned_90d
    sample_key: [customer_id]                          # stable row identity
    split: {strategy: temporal, time_column: snapshot_date,
            train: "-180d:-28d", test: "-28d:now"}
```

- Feature tables LEFT JOIN onto the label table by `join_key`; examples with
  missing features arrive with NULLs (tree adapters handle them natively).
  Column names must be unique across tables apart from the join key(s).
- Every referenced table is a DAG edge; the dataset's pinned snapshot
  combines all of them, so any input changing marks it `state:modified`.
- `sample_key` (defaults to the join key) drives deterministic sampling and
  seeded random splits: rows are kept when
  `hash(key) % 1e6 < sample_fraction * 1e6`, pushed down into the source
  query. Same fraction -> same rows; smaller fractions are subsets of
  larger ones. Strongly recommended on wide tables.

### Warehouse sources (Snowflake)

```yaml
# profiles.yml
data:
  adapter: snowflake
  config:
    account: "{{ env_var('SNOWFLAKE_ACCOUNT') }}"
    user: "{{ env_var('SNOWFLAKE_USER') }}"
    password: "{{ env_var('SNOWFLAKE_PASSWORD') }}"
    warehouse: ML_WH
    database: ANALYTICS
    schema: GOLD

# sources.yml - warehouse tables use identifier:, not path:
sources:
  - name: snowflake
    tables:
      - name: churn_labels
        identifier: GOLD.CHURN_LABELS
```

Snapshots pin `SYSTEM$LAST_CHANGE_COMMIT_TIME` per table at compile
(`--deep-snapshot`: `HASH_AGG(*)` content fingerprints); rows stream back
as Arrow batches into the standard local materialization, so training jobs
never need warehouse credentials. See `packages/mbt-snowflake/README.md`.

## models/*.yml

```yaml
models:
  - name: churn_classifier
    description: "90-day churn prediction"
    task: binary_classification     # selects the task schema
    adapter: xgboost                # which plugin executes this
    owner: growth-ds@company.com    # required
    tags: [churn, weekly]
    dataset: ref('churn_training_set')
    target: churned_90d             # must equal the dataset's label.column
    features:
      include: ["*"]                # globs over post-hook columns
      exclude: [user_id, email]     # target + time column always excluded
    hyperparameters:                # validated by the adapter's param model
      max_depth: 6
      scale_pos_weight: "{{ auto }}"
    tuning:                         # optional
      engine: optuna
      n_trials: 50                  # capped by the target's max_tuning_trials
      search_space:
        max_depth: {type: int, low: 3, high: 10}
        learning_rate: {type: loguniform, low: 0.005, high: 0.3}
      objective: {metric: pr_auc, direction: maximize}
    evaluation:
      protocol: {split: temporal, test_window: "14d"}  # must match the dataset
      metrics: [pr_auc, roc_auc, ece, recall_at_precision_0.9]
      gates:
        - {metric: pr_auc, threshold: 0.42}
        - {metric: pr_auc, compare_to: production, min_delta: 0.005}
      slices: [plan_type, region]   # per-slice reporting
    registration:
      name: churn_classifier
      stage_on_pass: staging        # canonical stages: staging|production|archived
    materialization: model_artifact
    seed: 42                        # mandatory, no default
    hooks: models/churn_classifier.py   # optional; sibling <name>.py auto-detected
```

Builtin binary-classification metrics: `roc_auc`, `pr_auc`, `logloss`,
`brier`, `accuracy`, `ece`, and parameterized `recall_at_precision_*` /
`precision_at_recall_*`. Lower-is-better defaults: `logloss`, `ece`, `brier`.

## metrics.yml, exposures.yml

```yaml
metrics:
  - name: lift_at_decile
    kind: hook                      # computed by hooks.custom_metrics
    greater_is_better: true

exposures:
  - name: retention_campaign_job
    type: batch_job                 # endpoint | batch_job | dashboard | other
    depends_on: [ref('churn_classifier')]
    owner: lifecycle-eng@company.com
    url: https://internal/jobs/retention
```

## hooks.py

```python
import pyarrow as pa

def transform_features(table: pa.Table, ctx) -> pa.Table:
    """Applied per split after read; feature globs apply to the result."""
    return table.append_column("usage_per_ticket", ...)

def custom_metrics(predictions: pa.Table, ctx) -> dict[str, float]:
    """predictions = the split's table + a 'prediction' column."""
    return {"lift_at_decile": ...}
```

## Python data tests (tests/*.py)

```python
# mbt: select=churn_training_set          <- binding selector (optional)
from mbt.contracts import TestResult

def test_label_is_binary(dataset, spec) -> TestResult:
    values = set(dataset.column(spec.label.column).to_pylist())
    return TestResult(name="test_label_is_binary", passed=values <= {0, 1})
```
