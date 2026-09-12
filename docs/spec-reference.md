# Spec reference

Machine-readable JSON Schemas for every file kind are published by
`mbt parse --write-json-schema` (into `target/json-schemas/`); the scaffolded
specs reference them via `yaml-language-server` headers for editor
autocomplete. This page summarizes the shapes.

## mbt_project.yml

```yaml
name: churn_project          # ^[A-Za-z][A-Za-z0-9_]*$
version: "0.1.0"
require_mbt_version: ">=0.1,<0.2"   # optional PEP 440 guard
vars: {pr_auc_floor: 0.42}          # project-level var defaults
model_defaults: {adapter: xgboost}  # merged under every model spec
model_paths: [models]               # discovery paths (defaults shown)
dataset_paths: [datasets]
scoring_paths: [scoring]
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
      # only training logs here; the experiment is <project>__<experiment>,
      # or the project name alone when `experiment:` is omitted
      # (see "Where tracking runs land" below)
      tracking: {adapter: mlflow, config: {uri: "sqlite:///mlflow.db", experiment: wide_v2}}
      # the registry maps mbt stages to registered-model aliases by default;
      # set use_aliases: false for MLflow servers without alias support (<2.9)
      registry: {adapter: mlflow, config: {uri: "sqlite:///mlflow.db"}}
      # job_timeout_seconds kills any training job that outlives it
      # (local and spark compute; omit for no limit)
      compute:  {adapter: local, config: {job_timeout_seconds: 3600}}
      # optional: tuning-engine ops knobs (the engine itself is named by the
      # model's tuning spec). optuna: sampler (tpe|random), multivariate (joint
      # TPE), and the median pruner's n_startup_trials / n_warmup_steps
      tuning:   {adapter: optuna, config: {sampler: tpe, multivariate: false}}
      artifact_store: file://./target/artifacts     # or s3://bucket/prefix (s3 extra)
      threads: 4                                    # optional, default 1
      vars: {sample_fraction: 0.1, max_tuning_trials: 5}
```

Search order: `--profiles-dir`, `$MBT_PROFILES_DIR`, `./profiles.yml`,
`~/.mbt/profiles.yml`. Jinja renders before validation.

### Reading the environment: `env_var()` vs `env()`

Two accessors, same lookup, one difference that matters:

| | Marks the value secret | Use for |
|---|---|---|
| `{{ env_var('NAME') }}` | **yes** - tainted, then redacted from every serialized surface | passwords, tokens, private keys, connection strings that embed one |
| `{{ env('NAME') }}` | no | schema and database names, hosts, ports, roots, region, environment name |

Both accept a default (`env('NAME', 'fallback')`), both are re-resolved inside
the training-job subprocess, and both are recorded in the manifest's
`required_env`. Neither ever stores its *value* in the manifest: the target
config is kept unrendered.

**Pick deliberately.** Redaction is exact-substring, so it is unforgiving in
both directions. Use `env_var()` for something that is not a secret and every
occurrence of that string disappears from your logs, your `run_results.json`,
and your model cards - a value of `1` rewrites `0.1234` to `0.***234`. Use
`env()` for something that *is* a secret and it will be printed. When in
doubt, `env_var()`: a redacted log is recoverable, a leaked credential is not.

### Where tracking runs land

**Only training opens a tracking run** (ADR-28). `mbt score` writes to the
prediction store and `mbt monitor` to the ground-truth ledger; neither touches
the tracker, so a scoring cadence cannot fail on a tracker outage, and a
tracking server holds modelling work rather than serving volume. Realized
production metrics are read back with `mbt predictions show <run_key>`.

The experiment name is composed from two names you give:

| where | key | example |
|---|---|---|
| `mbt_project.yml` | `name:` | `churn_lake` |
| `profiles.yml`, tracking config | `experiment:` | `wide_v2` |

```yaml
      tracking:
        adapter: mlflow
        config:
          uri: "..."
          experiment: "{{ env('EXPERIMENT_NAME', 'wide_v2') }}"
```

That gives an MLflow experiment named `churn_lake__wide_v2`. The separator is
two underscores because both halves are themselves snake_case, and a single one
leaves the boundary unreadable - `LOAN_APPLY_PROPENSITY_V1_0_0` does not say
where the project name ends. Omit `experiment:` and the project name stands
alone (`churn_lake`), so a fresh project's runs land under their own name rather
than under a literal `mbt`. One tracking
server can then hold many projects, and one project many modelling efforts,
without collision.

Each run is named `<run_id>-<model>`, where `run_id` is the invocation id mbt
already stamps on the artifact store and on `run_results.json`:

```
experiment "churn_lake__wide_v2"
├─ 20260909T101500Z-a1b2c3d4-churn_wide_automl
│   ├─ trial-000                       # tuning trials nest under their parent
│   └─ trial-001
├─ 20260909T101500Z-a1b2c3d4-churn_wide_probe
└─ 20260901T090000Z-9f3e21bc-churn_wide_automl   # last week's training
```

So re-training a model does not produce a second run with the same name, and a
run says which model it is and when it ran without being opened.

Every run carries `mbt.project`, `mbt.run_id`, and the identity tags
(`mbt.config_hash`, `mbt.input_hash`, `mbt.manifest_hash`, `mbt.snapshot_id`,
`mbt.git_commit`), plus two documents: `inference_config.json` and, when the
model has one, its `hooks.py` source. Model binaries stay in the artifact
store with a pointer tag, not in the tracker.

Experiment names come from `profiles.yml`, which never enters node identity
(ADR-5), so renaming one cannot mark a node `state:modified`.
Note that `var()` in `profiles.yml` reads CLI and project vars, not the
target's own `vars:` block, so build a per-target name from `env()` or write
it literally.

`experiment:` no longer accepts the per-node-kind mapping ADR-26 introduced
(`{model: ..., scoring: ...}`); it is rejected at construction rather than
half-applied.

### What the champion carries

A training run exports `inference_config.json` next to the model artifact and
pins it on the registered version as `mbt.inference_config_uri` (with
`_format`, `_content_hash`, `_size_bytes`), the same way ADR-21 pins the
monitoring baseline. `mbt score` reads the model's spec back from it, so a
scoring run applies the features, transforms and categoricals the champion was
actually trained with rather than whatever the working tree says today:

```jsonc
{
  "schema_version": 1,
  "project": "churn_lake", "model": "churn_wide_automl",
  "trained_at": "20260909T101500Z-a1b2c3d4",
  "spec": { /* the rendered model spec, verbatim */ },
  "resolved": {
    "feature_columns": ["...the exact order the model was fit on..."],
    "target": "churned_90d", "task": "binary_classification",
    "adapter": "h2o", "seed": 42, "calibration": "isotonic",
    "categorical": {...}, "transforms": {...}, "monotonic": {...},
    "operating_points": {"threshold_at_precision_0.35": 0.41}
  },
  "hooks": {"path": "models/wide_hooks.py", "hash": "sha256:..."},
  "identity": {"config_hash": "sha256:...", "manifest_hash": "sha256:...", ...},
  "artifact": {"uri": "s3://...", "format": "mojo", ...},
  "baseline_uri": "s3://..."
}
```

Two things are deliberately NOT taken from the champion. **`hooks.py`** is
arbitrary Python, so mbt keeps executing the git-tracked file from the project
and keeps hard-failing when its hash differs from the champion's
(`mbt.hooks_hash`); the source above is a record, not a load path. And the
**scoring node's own spec** - input, filters, window, checks, output, monitors,
`ground_truth` - is the local manifest's, since it describes this batch rather
than the model. `mbt score` therefore still needs the project checked out.

A champion registered before mbt exported an inference config has none to
read, so scoring falls back to the project's current model spec and warns on
every run until a retrain and promote. A champion whose spec differs from the
project's also warns, naming both hashes: that is expected between merging a
spec edit and promoting the model it produced, and the champion's spec wins.

## sources.yml

```yaml
sources:
  - name: lakehouse
    tables:
      - name: subscribers
        path: data/subscribers/*.parquet   # glob under the data adapter root
        format: parquet                    # parquet (all adapters) | delta (spark only)
```

`format` is a validated vocabulary: `parquet` or `delta` (iceberg is roadmap).
Compilation rejects a referenced source whose format the resolved data adapter
cannot read - `delta` needs the spark data adapter - so a mis-declared format
fails loudly up front instead of being silently read as parquet.

## datasets/*.yml

Data comes from exactly one of ``source`` (a single table) or ``inputs``
(feature tables joined onto a label table):

```yaml
datasets:
  - name: churn_training_set
    source: source('lakehouse', 'subscribers')
    columns:                        # the panel contract (ADR-29), optional
      [user_id, snapshot_date, churned_90d, plan, tenure_days, is_active]
    sample_key: [user_id]           # REQUIRED: the stable row identity
    label:
      column: churned_90d
      definition: "cancelled within 90d of snapshot_date"   # for model cards
      horizon: "90d"                # declarative: when the outcome is observed
    filters: ["is_active = true", "tenure_days >= 30"]      # SQL, ANDed
    split:
      strategy: temporal            # default; random needs explicit seed
      time_column: snapshot_date
      train: "-180d:-28d"           # window expressions vs the anchor
      test: "-28d:now"
      validation: "-42d:-28d"       # optional; else carved when tuning needs it
      embargo: "7d"                 # optional (temporal): drop the train window's
                                    # tail so a row whose label horizon reaches the
                                    # eval window cannot leak (R2-7)
    checks:                         # run at every dataset build
      # every timestamp column must stay within its split's OWN window:
      # catches train rows reaching into the test period (temporal
      # leakage), not just absolutely-future values
      - no_future_columns
      - class_balance_report        # report-only
      - schema: {columns: {churned_90d: int64}}
      - not_null: {columns: [churned_90d]}
      # EACH listed column, individually, must hold no duplicated non-null value
      # in any split. Catches an upstream join that fanned the panel out on a
      # non-unique key - but only where a single column is the key
      - unique: {columns: [user_id]}
      # with source:, unique reads a relation directly instead of the split
      # panel, and treats the columns as ONE COMPOSITE key. That is the form a
      # panel needs: keyed by (entity, date), neither column is unique alone.
      # Point it at the panel itself for its natural key, or at an upstream
      # table to assert the uniqueness your panel depends on
      - unique: {source: lakehouse.ml_churn_panel, columns: [user_id, inference_date]}
      # a column's non-null values must all lie in the allowed set (nulls
      # ignored): catches a categorical that drifted to an unexpected level
      # (a new code, a typo, an upstream enum change)
      - accepted_values: {column: plan, values: [basic, pro, enterprise]}
      # dbt-parity foreign key: every non-null value of the column must exist
      # in the referenced RAW source's field (parent pulled as DISTINCT via the
      # data adapter - size the referenced table like a dimension)
      - relationships: {column: plan_id, to: lakehouse.plans, field: id}
      # the materialized dataset's total row count (all splits) must stay within
      # bounds: a volume floor/ceiling that turns a silently collapsed upstream
      # join into a loud build failure instead of a quietly smaller model
      - row_count: {min: 1000}
      # the newest row must be within max_lag of the anchor ("now"): an
      # upstream-is-stale guard, so a scheduled retrain fails loudly instead of
      # silently training on old data when a source stops updating
      - freshness: {max_lag: 2d}
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

**The panel contract (`columns:`)** declares the exact column set the dataset
expects from its relation.
An undeclared column fails the build, and so does a declared column that is
absent.
It asserts; it never selects.

It matters most when the relation is built upstream (a dbt model, a Snowflake
dynamic table): without it, a feature arriving in that relation is
indistinguishable from a routine data refresh, so nothing in the mbt repo moves
and no reviewer sees the training set change.
With it, adding `feature_N+1` is a one-line diff sitting next to the gates and
the seed, which is where the decision belongs.

It is distinct from the model's `features.include`: `columns:` says what the
panel IS, `features.include` says what one model uses out of it.
Declaring it is optional, but the columns mbt itself reads (the label, the split
time column, the sample key) must be in the list when you do.
The check runs automatically whenever `columns:` is set; you do not also list
`panel_columns` under `checks:`.

**`sample_key` is required.**
It names the stable row identity used for deterministic hash sampling and
seeded random splits.
Without one the digest hashes *every* column, so the column list becomes the
hash preimage: one column arriving upstream re-buckets every row and moves some
across the train/test boundary (measured at three of ten).
The Snowflake and Spark adapters always refused that path; requiring it here is
what stops a spec passing on the dev plane and failing on the prod one.

**`label.horizon`** states how long after the prediction date the outcome is
observed.
It is declarative: it joins nothing and shifts nothing.
Declaring it is what lets mbt check that the three places a project spells this
one number agree - it warns when `split.embargo` is absent or shorter than the
horizon, and when a scoring pipeline's `ground_truth.maturity` is shorter (which
would grade predictions against outcomes that have not happened).

**Leakage scan ceiling:** `label_leakage_scan` is a *univariate* screen on the
*train* split only.
It flags a single column whose association with the label crosses the bar, so it
cannot see multivariate leakage (a combination of columns that reveals the label
while no single one trips the bar) or leakage that surfaces only in the test
split.
Treat a clean scan as necessary, not sufficient: it is a fast tripwire for the
obvious cases (an accidental copy of the label, a feature computed downstream of
the outcome), not a proof of no leakage.

**Window expressions:** `"<start>:<end>"`, each bound a signed duration
(`-180d`, `-12h`, `2w`), `now`, or an ISO date/timestamp; bare `"28d"` is
sugar for `"-28d:now"`. Durations take the fixed units `d`, `w`, `h` and the
calendar units `mo` (months) and `y` (years); calendar units must be whole
numbers and shift by real calendar months (`-3mo` is a true quarter, not
`90d`), clamping the day (`Jan 31 - 1mo -> Feb 28`).

**Embargo (temporal only):** `embargo: <duration>` drops that much off the END
of the resolved train window, so training rows whose label horizon (declared
as `label.horizon`) reaches into the evaluation window cannot leak - set it to
at least that horizon. It is applied in the compiler, so every data adapter
gets the embargoed window; an embargo that consumes the whole train window is a
compile error.

**Random splits:** `strategy: random` uses fractions (`train: "0.8"`),
requires `seed`, and supports `stratify_by: <column>`. Two guardrail
warnings fire at parse time: combining a random split with a `time_column`
(temporal leakage: rows from after the test period can train the model),
and a random split without `sample_key` (rows split independently, so
repeated entities can straddle train and test). `sample_key` is the
grouped-split control: set it to the entity id and hash-based ranking
keeps all of an entity's rows on one side of the split.

Random-split membership is stable and portable (F19): every adapter buckets a
row by the same canonical digest - the unsigned lower 64 bits of the md5 of the
'|'-joined key (Snowflake's `MD5_NUMBER_LOWER64`; local DuckDB and Spark
compute the identical value) modulo 1,000,000 - so membership is a pure
function of the key: it neither shifts as the dataset grows nor differs when
the same spec runs on DuckDB, Snowflake, or Spark, and a model validated
locally trains on the same partition in the warehouse.
Two caveats: `stratify_by` uses exact-fraction ranking instead (per-stratum
proportions cannot come from pure hash buckets), so stratified membership is
size-dependent and local-only; and key columns are hashed via each engine's
CAST-to-string, which renders integers, strings, and DATEs identically
everywhere but can differ for raw TIMESTAMP columns (session formats), so
prefer id/date sampling keys.
Hash-bucket fractions are approximate (each row lands independently); a
temporal split, the default, is window-based and unaffected by all of this.

### Warehouse sources (Snowflake)

```yaml
# profiles.yml
data:
  adapter: snowflake
  config:
    # env() for the identifiers (they belong in logs), env_var() for the secret
    account: "{{ env('SNOWFLAKE_ACCOUNT') }}"
    user: "{{ env('SNOWFLAKE_USER') }}"
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

#### Declaring both addresses

A table may declare `path:` **and** `identifier:` - that is how one `sources.yml`
serves a file plane and a warehouse plane, with `--target` choosing between them
(`examples/showcase` does exactly this across three planes).
Each adapter reads the field it understands and ignores the other, with one
exception: **Spark reads both**, so a target using the Spark data adapter must
say which via `source_address: path` (or `identifier`) in its adapter config.
Leaving it unset fails the compile, naming the table - mbt will not guess which
address a target reads. Tables declaring only one address are unaffected.

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
      # categoricals in the tree adapters (unseen levels become missing).
      # Spark indexes strings to ordinal codes instead (not native), so a
      # categorical-heavy model is NOT apples-to-apples across Spark and a
      # tree adapter - see the mbt-spark README's parity caveat

      # --- per-column treatment (ADR-27), all optional ---
      categorical: [plan_type, contract_code]   # bare list...
      # categorical:                            # ...or a mapping, per column:
      #   plan_type:     {levels: [basic, pro, enterprise]}
      #   region:        {min_frequency: 0.01, null_as_level: true}
      #   contract_code: {max_levels: 20}
      transforms:
        days_since_onboard: {cap: 365, log: true, monotonic: increasing}
        recency_days:       {percentile: batch}
        order_value:        {cap: {min: 0, max: 5000}}
      monotonic:                    # same constraint, standalone spelling
        support_tickets: decreasing
    hyperparameters:                # validated by the adapter's param model
      max_depth: 6
      scale_pos_weight: "{{ auto }}"
    tuning:                         # optional
      engine: optuna
      n_trials: 50                  # capped by the target's max_tuning_trials
      search_space:
        max_depth: {type: int, low: 3, high: 10}
        learning_rate: {type: loguniform, low: 0.005, high: 0.3}
      # robust: true selects on the bootstrap lower bound of the validation
      # metric, not the point estimate (R2-7), so the tuning selection is
      # defended against validation-window luck (the champion gate's idea, ADR-18,
      # applied to tuning). Builtin objective metric only.
      objective: {metric: pr_auc, direction: maximize, robust: true}
      # optional: stop unpromising trials early. "median" prunes a trial when
      # its per-round validation value falls below the median of prior trials
      # at the same step; needs an adapter that reports progress (xgboost and
      # lightgbm do), otherwise trials run to completion with a warning.
      # Pruned counts land in the mbt.tuning.n_pruned tracking tag.
      pruner: median
    evaluation:
      protocol: {split: temporal, test_window: "14d", backtest_folds: 5}  # match the dataset
      metrics: [pr_auc, roc_auc, ece, recall_at_precision_0.9]
      gates:
        - {metric: pr_auc, threshold: 0.42}
        # gate the walk-forward backtest MEAN instead of the single test split
        # (R2-7): needs backtest_folds; whole-split threshold gates only
        - {metric: pr_auc, threshold: 0.40, source: backtest}
        # champion gates pass when the paired-bootstrap lower bound of the
        # delta clears min_delta (ADR-18); confidence: null opts out
        - {metric: pr_auc, compare_to: production, min_delta: 0.005,
           confidence: 0.95, bootstrap_resamples: 1000}
        # slice gates target one declared slice value; champion slice gates
        # compare point deltas (no bootstrap bound)
        - {metric: pr_auc, threshold: 0.35, slice: plan_type=premium}
        # disparity/fairness gate: the metric's worst slice must stay within
        # min_ratio of its best across ALL values of `across` (a declared slice
        # column). The ratio is min/max in (0, 1] (1.0 = parity) and is
        # direction-agnostic - it flags the gap whether the metric is higher-
        # or lower-is-better; `across` needs >= 2 non-degenerate slices.
        - {metric: pr_auc, across: plan_type, min_ratio: 0.8}
      slices: [plan_type, region]   # per-slice reporting; a high-cardinality
                                    # numeric slice column (age, tenure) is
                                    # auto-binned into quartile ranges (e.g.
                                    # age=[25, 40)) instead of one slice per value
    registration:
      name: churn_classifier
      stage_on_pass: staging        # canonical stages: staging|production|archived
    materialization: model_artifact
    seed: 42                        # mandatory, no default
    hooks: models/churn_classifier.py   # optional; sibling <name>.py auto-detected
    calibration: isotonic           # optional: post-hoc probability calibration
                                    # (isotonic|sigmoid), binary only, fit on a
                                    # dedicated slice carved from train
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
`ece` uses equal-frequency (adaptive) bins - `n_bins` (default 10) equal-mass
score buckets - which stays stable on the skewed score distributions common in
churn, where fixed-width bins pile most samples into one bucket.
`calibration` (`isotonic` or `sigmoid`, binary classification only) fits a
post-hoc probability calibrator and applies it to every score, so `ece`/`brier`
and the emitted probabilities reflect calibrated estimates - the lever for the
miscalibration that `scale_pos_weight` rebalancing introduces. The calibrator
is fit on a dedicated slice carved from the train split (20%, the temporal tail
of the train window for a temporal split, else a seeded random slice at
`seed+5`), never on the `validation` split that `early_stopping_rounds` and
tuning select on - reusing the selection split would make the reported
`ece`/`brier` optimistic and overfit the deployed calibrator. Because the slice
comes from train, no `validation` split needs to be declared, and calibration
composes with tuning, early stopping, and `backtest_folds` (each fold carves
its own slice, so a `source: backtest` gate compares calibrated fold models
against a calibrated production model). Tuning trials themselves never
calibrate: a trial calibrator would fit on the very split the objective is
scored on, making a `brier`/`ece` objective circularly optimal. Calibration is
a monotonic transform (ranking metrics like `roc_auc` are preserved) and
travels with the model, so champion and challenger calibrate identically and
the paired gate stays apples-to-apples. All five training adapters (xgboost,
lightgbm, sklearn, spark, h2o) support it; support is probed at parse.

`protocol.backtest_folds: N` adds a cross-validated backtest (R2-7): the model
is refit and evaluated across `N` folds of the training window - time-ordered
walk-forward for a temporal split (refit on each expanding prefix, evaluate on
the next fold) or random k-fold for a random split (each fold is held out once,
train on the rest) - and the card reports each metric's cross-validated mean
and its population std across the folds (rendered `mean ± std`) beside the
single-split value, so a single lucky split can no longer flatter the estimate
and an unstable model (one whose folds disagree, i.e. a large std) is visible at
a glance. A threshold gate can gate the mean instead of the single
split with `source: backtest` (whole-split threshold gates only). It works on all
four adapters; note it refits `N` (or, walk-forward, `N-1`) extra models, so the
training-time cost is real (and larger on the distributed adapters, which
retrain per fold).
`protocol.nested_cv: true` makes it NESTED cross-validation: each outer fold
re-runs the `tuning` search on that fold's train alone and evaluates the
fold-tuned model on the held-out fold, so the reported mean is an unbiased
estimate of the TUNED model (the tuning never sees the fold it is scored on -
for a temporal split the inner tuning uses only each fold's PAST). It needs
`backtest_folds` and a `tuning` block, works on either split, and re-tunes per
fold, so it is the most expensive option.

Regression (`task: regression`, all five adapters) uses `rmse`, `mae`, `r2`,
`mape`; the target must be a numeric column (no 0/1 label check, no
`scale_pos_weight`). Spark trains a `GBTRegressor` and H2O AutoML detects
regression from the numeric target - the same spec runs on every adapter. Lower-is-better: `rmse`, `mae`, `mape`; `r2` is
higher-is-better. Champion gates and slice metrics work identically - the
metric engine dispatches on the metric name (ADR-24).
`tests/fixtures/revenue_demo` is a complete worked regression project (spend
forecasting) with an `rmse` ceiling gate and delayed ground-truth monitoring -
the `task: regression` twin of `tests/fixtures/churn_demo`, and both are built
end to end by the E2E suite on every run.

### Feature treatment (ADR-27)

`include`/`exclude` decide *membership*; `categorical`, `transforms` and `monotonic` decide *treatment* of the columns that survive.
They are separate keys because they act at three different seams: `categorical` retypes a column, `transforms` rewrites its values, and `monotonic` constrains the model rather than the data.

**`transforms`** exists for time-anchored features - `days_since_onboard`, `tenure_days`, `recency_days` - whose distribution translates every month as the population ages, so the model extrapolates into a range it never trained on and the shift monitors fire forever.
Steps apply in the fixed order `cap` -> `log` -> `percentile`, on the same table at train and at score time:

| key | meaning |
| --- | --- |
| `cap: 365` | plateau above 365. `cap: {min: 0, max: 365}` clamps both ends. Declared constants only - a fitted `p99` would need persisted state, so it is deliberately not supported (use `hooks.py`). |
| `log: true` | `log1p`, compressing the tail so a doubling of the raw value moves the feature by a constant. |
| `percentile: batch` | rank within the split or batch being read, into (0, 1], ties taking the group's average rank. Self-normalizing: a uniform shift cancels, because every batch is ranked against itself. The reciprocal is that the feature's value depends on batch composition - see the caveat below. |
| `monotonic: increasing` | pin the model's response direction. Equivalent to naming the column in `features.monotonic`. |

Nulls survive every step as nulls, so tree adapters keep using their own missing branch.
Every step is monotone increasing, so a `monotonic` direction composes with them.
`log` alongside `percentile` is a parse error: a rank is invariant under any monotone transform, so the `log` would do nothing.

Three things worth knowing before reaching for `percentile: batch`.

It makes PSI on that column near-zero *by construction* - both sides become uniform ranks - so it trades monitorability for stability.

It assumes each scoring batch is a representative sample of the scored population, because the rank is computed within whatever batch is being scored (ADR-27).
The same entity with the same raw value therefore gets a different feature value depending on who else is in the batch: a re-score of only high-tenure customers after a failed batch re-spreads that slice's `tenure_days` across the full (0, 1] range, so a uniformly high-tenure population presents to the model as if it spanned the whole tenure distribution.
Predictions are also no longer independent across rows, which is a surprise for anyone reasoning about a batch scorer as a row-wise function.
Filtered, unusually small, or otherwise non-representative batches change the feature's meaning; a `train`-relative percentile would not, and ADR-27 records why it was not taken.

And capping is not free: it discards whatever signal lived in the tail, which is the trade the lever exists to make.

**`monotonic`** reaches the booster, not the data. xgboost and lightgbm support it natively; sklearn only through `estimator: hist_gradient_boosting`; Spark and H2O not at all.
An adapter that cannot honour a declared constraint fails at `mbt parse` naming the adapter, rather than training silently unconstrained.

**`categorical`** is how a DS takes over from dtype inference.
Omit it and nothing changes: string columns are categorical, everything else is a magnitude.
Write it - **including as an empty list** - and it becomes authoritative: a listed column is categorical even when int-coded, and an undeclared string feature is an error rather than a guess.
`categorical: []` therefore asserts "this model has no categoricals" and fails if one appears.

Per-column options, all optional:

| key | meaning |
| --- | --- |
| `levels: [...]` | pin the level set in the spec; anything else becomes `__other__`. A new production value is then a declared, hash-visible change instead of a silent unseen-level-to-missing. |
| `min_frequency: 0.01` | pool train levels below this share into `__other__`. Needs the train-fitted level map, so it is supported on xgboost, lightgbm and sklearn only (Spark and H2O fail at parse). |
| `max_levels: 100` | fail when the train split has more distinct levels than this - the guard that catches a `user_id` declared categorical by accident. Measured on the train split only. |
| `null_as_level: true` | map NULL to an explicit `__missing__` level, so "we don't know" is a category the model can learn from. |

Declared slice columns are not features, so they are never treated and never subject to the authoritative rule.

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
      # ONE relation, the serving twin of the training panel: identical
      # features, no label. Build both from one upstream definition - they are
      # two objects that have to stay in lockstep (ADR-29).
      source: source('lakehouse', 'ml_churn_panel_scoring')
      filters: ["is_active = true"]     # SQL WHERE fragments, ANDed
      time_column: snapshot_date        # optional
      window: "-7d:now"                 # optional; resolved against the anchor
      sample_key: user_id               # optional, as on datasets

    checks:                             # label-free subset only
      - schema: {columns: [user_id]}
      - not_null: {columns: [user_id]}  # explicit columns required (no label)
      - no_future_columns
      - unique: {columns: [user_id]}    # no double-scored rows in the batch
      - accepted_values: {column: plan, values: [basic, pro]}  # no drifted category
      # dbt-parity foreign key against a raw source (label-free, so it guards
      # serving batches too - an unknown plan_id scores garbage silently)
      - relationships: {column: plan_id, to: lakehouse.plans, field: id}
      # newest row within max_lag of the anchor: catches a STALE nightly batch
      # (scoring on old data) the same way it guards a stale retrain
      - freshness: {max_lag: 2d}

    monitors:                           # distribution shift vs the champion's
      feature_shift:                    # training-time baseline (ADR-21)
        method: psi                     # psi (default) | ks
        threshold: 0.25                 # per feature; breach (> threshold) = exit 2
        warn_threshold: 0.15            # optional two-tier bar: (warn, threshold]
                                        # logs a warning without failing the run
        # significance: 0.05            # (method: ks) n-aware fail bar at this
                                        # p-value instead of a fixed threshold,
                                        # so it does not over-fire on large
                                        # nightly batches or under-fire on small
                                        # ones. Kind-matched (F15): numeric
                                        # features get the two-sample KS
                                        # critical value (the statistic's sup is
                                        # evaluated over the merged baseline-
                                        # quantile + current-sample points);
                                        # categorical features get a two-sample
                                        # (contingency) chi-square statistic
                                        # judged at the chi-square critical
                                        # value. Excludes warn_threshold.
                                        # The family is the model's monitored
                                        # FEATURE SET, corrected by Benjamini-
                                        # Hochberg: 40 features at 0.05 would
                                        # otherwise breach ~2 per clean run.
                                        # prediction_shift is a family of one,
                                        # so it is uncorrected.
        include: ["*"]                  # globs over the model's features
        exclude: []
      prediction_shift:
        method: psi
        threshold: 0.25                 # score distribution vs test-split baseline
        warn_threshold: 0.15            # optional warn band, same semantics

    ground_truth:                       # delayed evaluation via `mbt monitor`
      label:
        source: source('lakehouse', 'churn_outcomes')
        column: churned_90d
      join_key: user_id                 # joins outcomes to stored predictions
      maturity: "14d"                   # bare duration (also 3mo, 1y); evaluate once this old
      metrics: [pr_auc, roc_auc]        # builtin only (no training job runs)
      gates:
        - {metric: pr_auc, threshold: 0.3}   # realized-metric floor

    output:
      format: parquet
      path: predictions/retention_scores    # adapter-interpreted
      columns: [user_id, snapshot_date]     # passthrough identity columns
      decision_threshold: 0.5               # optional operating point: emit a
                                            # 0/1 `decision` column (prediction
                                            # >= threshold) and record the cutoff
      explain_top_k: 3                      # optional: emit an `explanation`
                                            # column, each row's top-N features
                                            # by |SHAP| (tree adapters only)
```

`decision_threshold` is the deployable operating point: with it set, `mbt score`
writes a `decision` column beside the `prediction` probability and stamps the
cutoff into the run info, so downstream consumers get a decision rule instead of
re-deriving one out of band.
`explain_top_k` adds local per-prediction attribution: each output row gets an
`explanation` column - a JSON `[[feature, contribution], ...]` of the top-k
features by |SHAP|, ordered by descending magnitude - so a consumer can see
*why* an individual row scored the way it did (tree adapters only; others fail
with an actionable error).

`decision_threshold` can be a fixed float, or - better - the **name** of one of the champion's
operating-point metrics (`threshold_at_precision_<p>` / `threshold_at_recall_<r>`,
e.g. `decision_threshold: threshold_at_precision_0.9`). A named value is resolved
from the registered champion at score time (mbt records each model's operating
points as registry tags at registration), so the cutoff tracks the promoted
model automatically instead of being a constant copied off the model card
(R2-5). The named metric must be in the model's `evaluation.metrics`.

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
