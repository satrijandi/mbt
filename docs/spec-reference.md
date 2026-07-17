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
      # the registry maps mbt stages to registered-model aliases by default;
      # set use_aliases: false for MLflow servers without alias support (<2.9)
      registry: {adapter: mlflow, config: {uri: "sqlite:///mlflow.db"}}
      # job_timeout_seconds kills any training job that outlives it
      # (local and spark compute; omit for no limit)
      compute:  {adapter: local, config: {job_timeout_seconds: 3600}}
      artifact_store: file://./target/artifacts     # or s3://bucket/prefix (s3 extra)
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
      # every timestamp column must stay within its split's OWN window:
      # catches train rows reaching into the test period (temporal
      # leakage), not just absolutely-future values
      - no_future_columns
      - class_balance_report        # report-only
      - schema: {columns: {churned_90d: int64}}
      - not_null: {columns: [churned_90d]}
      # label_leakage_scan runs by default, declared or not. Numeric columns
      # are screened with |corr|, string/categorical columns with Cramér's V
      # (same 0-1 scale; single-level and unique-per-row ID columns are
      # skipped): >= 0.95 fails the build, the 0.85-0.95 warn band is logged
      # without failing. Declare it only to tune or opt out:
      - label_leakage_scan:
          max_abs_correlation: 0.95   # fail bar (|corr| and Cramér's V)
          warn_abs_correlation: 0.85  # warn band floor
          exclude: [reviewed_column]  # skip audited columns
          # enabled: false            # opt out (recorded, never silent)
    tests: [test_label_is_binary]   # bind Python data tests by name (optional)
    snapshot: "sha256:..."          # explicit pin (optional; normally compile pins)
    tags: [churn]
```

**Window expressions:** `"<start>:<end>"`, each bound a signed duration
(`-180d`, `-12h`, `2w`), `now`, or an ISO date/timestamp; bare `"28d"` is
sugar for `"-28d:now"`.

**Random splits:** `strategy: random` uses fractions (`train: "0.8"`),
requires `seed`, and supports `stratify_by: <column>`. Two guardrail
warnings fire at parse time: combining a random split with a `time_column`
(temporal leakage: rows from after the test period can train the model),
and a random split without `sample_key` (rows split independently, so
repeated entities can straddle train and test). `sample_key` is the
grouped-split control: set it to the entity id and hash-based ranking
keeps all of an entity's rows on one side of the split.

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

### Population spines, per-table join keys, and label offsets (ADR-22)

When the examples are defined by a population/cohort table rather than the
label table - and feature tables join by different keys - declare a
`population` spine:

```yaml
datasets:
  - name: wide_churn_training
    inputs:
      population: source('lake', 'monthly_population')  # spine: defines examples
      label:
        source: source('lake', 'monthly_labels')
        using: [customer_id, snapshot_date]
        time_offset: "1mo"      # label.snapshot_date = spine's + 1 calendar month
      features:
        - source: source('lake', 'demographic_history')
          using: [customer_id, snapshot_date]
        - source: source('lake', 'transaction_history')
          using: [safe_id, snapshot_date]   # key introduced by the population
    sample_key: [customer_id]   # panel sampling: keeps whole customers
    label:
      column: is_churn
    split:
      strategy: temporal
      time_column: snapshot_date
      train: "2025-07-01:2026-04-01"    # explicit ISO date ranges work too
      test: "2026-04-01:2026-06-02"
```

- Feature entries are bare `source()` strings (joined by `join_key`) or
  `{source, using}` mappings with their own USING-style columns, applied in
  declaration order - so a column introduced by an earlier join (the
  population's `safe_id`) is usable by a later one. The field is named
  `using`, not `on`: bare `on` is a YAML 1.1 boolean.
- The label join is always **inner** when a population is present: an
  example without an observed outcome is not a training example, so
  population rows whose labels have not matured yet drop out.
- `time_offset` (`1mo`, `-28d`, `2w`, `12h`; `mo` is a calendar month)
  shifts the spine's `split.time_column` when matching the label's
  same-named column, declaring the outcome's observation delay instead of
  pre-aligning dates upstream. The label's join columns are projected away;
  the spine's prediction date is the one true `time_column`.
- Scoring inputs mirror this shape with `spine:` (the same population
  table, no label) and the same per-table `using` support.

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
    task: binary_classification     # or 'regression'; selects the task schema
    adapter: xgboost                # which plugin executes this
    owner: growth-ds@company.com    # required
    tags: [churn, weekly]
    dataset: ref('churn_training_set')
    target: churned_90d             # must equal the dataset's label.column
    features:
      include: ["*"]                # globs over post-hook columns
      exclude: [user_id, email]     # target + time column always excluded
      # numeric columns pass through; string columns train as native
      # categoricals in the tree adapters (unseen levels become missing)
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
      # optional: stop unpromising trials early. "median" prunes a trial when
      # its per-round validation value falls below the median of prior trials
      # at the same step; needs an adapter that reports progress (xgboost and
      # lightgbm do), otherwise trials run to completion with a warning.
      # Pruned counts land in the mbt.tuning.n_pruned tracking tag.
      pruner: median
    evaluation:
      protocol: {split: temporal, test_window: "14d"}  # must match the dataset
      metrics: [pr_auc, roc_auc, ece, recall_at_precision_0.9]
      gates:
        - {metric: pr_auc, threshold: 0.42}
        # champion gates pass when the paired-bootstrap lower bound of the
        # delta clears min_delta (ADR-18); confidence: null opts out
        - {metric: pr_auc, compare_to: production, min_delta: 0.005,
           confidence: 0.95, bootstrap_resamples: 1000}
        # slice gates target one declared slice value; champion slice gates
        # compare point deltas (no bootstrap bound)
        - {metric: pr_auc, threshold: 0.35, slice: plan_type=premium}
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
`precision_at_recall_*` / `lift_at_*` / `gain_at_*` (top-scoring fraction:
`lift_at_0.1` is decile lift, `gain_at_0.25` the share of all positives
captured in the top quartile; ties break by row order, deterministically).
Operating points: `threshold_at_precision_*` reports the smallest score
cutoff meeting the precision target (maximal coverage), and
`threshold_at_recall_*` the largest cutoff meeting the recall target (best
precision) - the deployable decision rule interventions consume; an
unattainable precision target reports the 1.0 sentinel ("predict nothing").
Lower-is-better defaults: `logloss`, `ece`, `brier`.

Regression (`task: regression`, XGBoost/LightGBM) uses `rmse`, `mae`, `r2`,
`mape`; the target must be a numeric column (no 0/1 label check, no
`scale_pos_weight`). Lower-is-better: `rmse`, `mae`, `mape`; `r2` is
higher-is-better. Champion gates and slice metrics work identically - the
metric engine dispatches on the metric name (ADR-24).

## scoring/*.yml

One config is one batch scoring (serving) pipeline, executed by `mbt score`
(ADR-20/21).
The referenced model's registered champion for `stage` is resolved from the
registry at run time; promotions take effect on the next run without a spec
edit.

```yaml
scoring:
  - name: retention_scoring
    description: Nightly churn scores for the retention campaign tool.
    owner: lifecycle-eng@company.com    # required
    tags: [daily]
    model: ref('churn_classifier')      # the DAG edge; exactly one model
    stage: production                   # which champion alias to load (default)

    input:                              # unlabeled, unsplit by design
      source: source('lakehouse', 'scoring_batch')
      # or multi-table, like dataset inputs but with a spine instead of a label:
      # inputs: {spine: source(...), features: [source(...)], join_key: user_id}
      # feature entries may carry their own columns: {source: ..., using: [...]}
      filters: ["is_active = true"]     # SQL WHERE fragments, ANDed
      time_column: snapshot_date        # optional
      window: "-7d:now"                 # optional; resolved against the anchor
      sample_key: user_id               # optional, as on datasets

    checks:                             # label-free subset only
      - schema: {columns: [user_id]}
      - not_null: {columns: [user_id]}  # explicit columns required (no label)
      - no_future_columns

    monitors:                           # distribution shift vs the champion's
      feature_shift:                    # training-time baseline (ADR-21)
        method: psi                     # psi (default) | ks
        threshold: 0.25                 # per feature; breach = exit 2
        include: ["*"]                  # globs over the model's features
        exclude: []
      prediction_shift:
        method: psi
        threshold: 0.25                 # score distribution vs test-split baseline

    ground_truth:                       # delayed evaluation via `mbt monitor`
      label:
        source: source('lakehouse', 'churn_outcomes')
        column: churned_90d
      join_key: user_id                 # joins outcomes to stored predictions
      maturity: "14d"                   # bare duration; evaluate once this old
      metrics: [pr_auc, roc_auc]        # builtin only (no training job runs)
      gates:
        - {metric: pr_auc, threshold: 0.3}   # realized-metric floor

    output:
      format: parquet
      path: predictions/retention_scores    # adapter-interpreted
      columns: [user_id, snapshot_date]     # passthrough identity columns
```

Predictions carry the passthrough columns (the union of `output.columns`,
the ground-truth join key, and `time_column`; at least one is required) plus
a `prediction` column, one directory per run keyed for idempotent re-runs,
with a JSON sidecar recording the champion version and identity hashes.
Input sources and the window expression enter the node's identity exactly
like datasets (`state:modified` means "inputs or model chain changed"); the
ground-truth label table is lineage only, so arriving labels never re-score.

## metrics.yml, exposures.yml

```yaml
metrics:
  - name: campaign_capture_100      # a metric the builtins cannot express
    kind: hook                      # computed by hooks.custom_metrics
    greater_is_better: true

exposures:
  - name: retention_campaign_job
    type: batch_job                 # endpoint | batch_job | dashboard | other
    depends_on: [ref('churn_classifier')]
    owner: lifecycle-eng@company.com
    url: https://internal/jobs/retention
```

## promotions.yml

The GitOps promotion ledger: a reviewed change to this file is what moves a
model between stages (the scaffold's `promote.yml` workflow runs
`mbt promote --from-file promotions.yml` on merge). `mbt promote` refuses
versions whose gates were not recorded as passed, so review + gates stay the
only path to production.

```yaml
promotions:
  - model: churn_classifier      # registration name in the registry
    version: "3"                 # registry version to promote
    to: production               # target stage alias
```

An empty list (`promotions: []`) is valid and no-ops. Re-running an
already-merged file is safe: promoting a version to a stage it already holds
re-points the alias at the same version. Direct
`mbt promote --model X --to production` works too - `--from-file` exists so
the change itself is reviewable (and `--force` is the only way past a version
whose gates were not recorded as passed).

## packages.yml

The project's adapter-package requirements, consumed by `mbt deps`
(installation prefers the project's pinned `requirements.txt` when present;
either way the installed environment is verified against these pins
afterward, and a mismatch fails with the offending package named).

```yaml
packages:
  - package: mbt-xgboost
    version: "~=0.1"
  - package: mbt-mlflow
    version: "~=0.1"
```

## hooks.py

```python
import pyarrow as pa

def transform_features(table: pa.Table, ctx) -> pa.Table:
    """Applied per split after read; feature globs apply to the result."""
    return table.append_column("usage_per_ticket", ...)

def custom_metrics(predictions: pa.Table, ctx) -> dict[str, float]:
    """predictions = the split's table + a 'prediction' column."""
    return {"campaign_capture_100": ...}
```

## Python data tests (tests/*.py)

```python
# mbt: select=churn_training_set          <- binding selector (optional)
from mbt.contracts import TestResult

def test_label_is_binary(dataset, spec) -> TestResult:
    values = set(dataset.column(spec.label.column).to_pylist())
    return TestResult(name="test_label_is_binary", passed=values <= {0, 1})
```
