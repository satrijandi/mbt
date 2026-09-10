# Concepts

## Resources

| Resource | File | Purpose |
|---|---|---|
| **source** | `sources.yml` | External inputs: parquet paths (warehouse tables, feature views in v1) |
| **dataset** | `datasets/*.yml` | Declarative training-set definition over ONE relation: source + panel contract + label + filters + split policy + checks. Whatever joins that relation is upstream (ADR-29) |
| **model** | `models/*.yml` | Task, adapter, features, hyperparameters, tuning, gates, registration |
| **scoring** | `scoring/*.yml` | Batch scoring (serving) pipeline: champion + input + prediction sink + shift monitors + delayed ground-truth evaluation |
| **metric** | `metrics.yml` | Reusable metric definitions (`kind: builtin` or `kind: hook`) |
| **test** | `tests/*.py` | Python data tests: `def test_*(dataset, spec) -> TestResult` |
| **exposure** | `exposures.yml` | Downstream consumers, for lineage and impact analysis |

Every resource gets a stable unique id: `<type>.<project>.<name>`
(sources are the one exception: `source.<project>.<group>.<table>`).

## What a dataset reads

A dataset names exactly one relation: a table, a view, or a warehouse dynamic
table.
Whatever assembles it - joining a population to its labels and feature
histories - is upstream, and mbt does not model it (ADR-29).
mbt owns which rows (`filters`, `split` windows, `sample_fraction`), the split
policy, the checks, and the declaration of what the training set is.

```yaml
datasets:
  - name: churn_training_set
    source: source('warehouse', 'ml_churn_panel')
    columns: [customer_id, inference_date, is_churn, age_band, tenure_months]
    sample_key: [customer_id]
    label: {column: is_churn, horizon: "1mo"}
    split: {strategy: temporal, time_column: inference_date,
            train: "2025-07-01:2026-04-01", test: "2026-04-01:2026-06-01",
            embargo: "1mo"}
```

Three of those are load-bearing in a way that is not obvious:

- **`columns:`** is the panel contract. An undeclared column fails the build,
  so a feature arriving in the upstream relation cannot enter the training set
  until the spec asks for it. Without it, "we added a feature" and "the nightly
  refresh ran" look identical from inside mbt, and neither shows up in a PR.
- **`sample_key`** is required. It is the stable row identity for sampling and
  seeded random splits; with none, the digest hashes every column, so one added
  column re-buckets every row and moves some across the train/test boundary.
- **`label.horizon`** declares when the outcome is observed. It executes
  nothing, but it is what lets mbt check `split.embargo` and a scoring
  pipeline's `ground_truth.maturity` against one number instead of three
  hand-kept ones.

The scoring side mirrors this: one relation, the serving twin of the panel with
the same features and no label. They are two upstream objects that must stay in
lockstep, so build both from one shared definition - mbt catches divergence
(the panel contract at build time, the champion's recorded feature columns at
score time) but cannot prevent it.

## The DAG and selection

`ref('churn_training_set')` records an edge; sources connect through
`source('group', 'table')`. Selectors follow dbt semantics:

```
mbt build --select churn_classifier+          # model and downstream
mbt build --select +churn_classifier          # upstream first
mbt build --select tag:weekly,state:modified+ # intersection (comma)
mbt build --select "tag:churn tag:upsell"     # union (space)
mbt build --select resource_type:model --exclude tag:experimental
```

Selection governs which models *train*; datasets a selected model needs are
auto-materialized (cache-aware), so a model-only PR works on a cold CI runner.

## Identity and state

Each node carries two hashes:

- `config_hash` - the rendered spec plus the hooks file bytes. Cosmetic
  fields (`description`, `owner`, `tags`) are excluded; so is everything
  from profiles (environments never change identity).
- `input_hash` - transitive: `config_hash` + pinned data snapshot + all
  upstream input hashes.

`state:modified` selects nodes whose `input_hash` differs from a reference
manifest. Time passing does not change identity: windows are hashed as
*expressions* (`"-28d:now"`), resolved against a single anchor stored
outside the hashed config. New data arrives as a snapshot change - which
does mark the node modified.

## Environments

`profiles.yml` defines targets (dev/prod) with data/tracking/registry/compute
adapters, an artifact store, threads, and per-target vars like
`sample_fraction` and `max_tuning_trials`. Secrets resolve via
`{{ env_var('NAME') }}`, which also marks the value for redaction everywhere
it could be printed; non-secret environment values use `{{ env('NAME') }}`,
which resolves identically but stays readable in logs. Neither enters the
manifest - the target config is stored unrendered.

## Quality gates

```yaml
gates:
  - metric: pr_auc
    threshold: 0.42            # absolute floor
  - metric: pr_auc
    compare_to: production     # champion/challenger
    min_delta: 0.005
```

Champion comparisons re-evaluate the current production version *inside the
training job, on the identical pinned test split*, with the same metric
code as the challenger. The delta must clear `min_delta` at the gate's
one-sided confidence (default 95%), estimated by a seeded paired bootstrap
over per-example predictions - a challenger that is ahead on test-set noise
alone does not promote (ADR-18). No champion yet? The gate passes with a
loud warning (bootstrap). Champion exists but cannot load? Hard error -
never a silent pass.

Gates can also target one declared slice (`slice: plan_type=premium` with
`plan_type` under `evaluation.slices`); a failing slice blocks registration
just like a whole-split gate.

A failing gate blocks registration and exits with code **2**.

## Batch scoring and monitoring

A `scoring` resource is one batch serving pipeline (ADR-20): `mbt score`
loads the referenced model's registered champion (resolved by stage alias at
run time, so promotions take effect on the next scheduled run), materializes
an unlabeled input batch with the model's own hooks and feature selection,
writes predictions through the data adapter, and monitors distributions.

Monitoring lives in the same config. Every scoring run can check input
quality (`checks`), compare per-feature and score distributions against the
champion's training-time baseline (`monitors`, PSI or KS "shift" - "drift"
stays reserved for data-snapshot drift), and a `ground_truth` block lets
`mbt monitor` join outcomes to stored predictions once they mature, compute
realized metrics, and gate on them (ADR-21). Any breach is a quality
failure: node status `monitor_failed`, exit code **2**.

## Escape hatch: hooks.py

A sibling `models/<name>.py` may expose:

```python
def transform_features(table: pa.Table, ctx) -> pa.Table: ...
def custom_metrics(predictions: pa.Table, ctx) -> dict[str, float]: ...
```

Hooks run inside the training job, never in the coordinator. The hook file's
bytes are hashed into the model's identity, so editing a hook marks it
`state:modified`.

## Reproducibility contract

`mbt run --manifest <path>` executes a stored manifest verbatim: same
anchor, same resolved windows, same snapshots, same hashes, same seeds.
It first verifies the running environment against the manifest's digests
and refuses on `env_digest` mismatch (ADR-19; `--allow-env-mismatch`
downgrades this to a warning, transitive-drift mismatches always warn).
All seeds derive from the model's mandatory `seed`: the adapter uses `seed`,
tuning samples with `seed + 1`, implicit validation carves with `seed + 2`,
and champion-gate bootstrap resampling uses `seed + 3` (ADR-18).
Each adapter documents a determinism tier - exact (XGBoost, LightGBM, and
scikit-learn on CPU, single-threaded) or tolerance bands.
