# Concepts

## Resources

| Resource | File | Purpose |
|---|---|---|
| **source** | `sources.yml` | External inputs: parquet paths (warehouse tables, feature views in v1) |
| **dataset** | `datasets/*.yml` | Declarative training-set construction: source + label + filters + split policy + checks |
| **model** | `models/*.yml` | Task, adapter, features, hyperparameters, tuning, gates, registration |
| **metric** | `metrics.yml` | Reusable metric definitions (`kind: builtin` or `kind: hook`) |
| **test** | `tests/*.py` | Python data tests: `def test_*(dataset, spec) -> TestResult` |
| **exposure** | `exposures.yml` | Downstream consumers, for lineage and impact analysis |

Every resource gets a stable unique id: `<type>.<project>.<name>`.

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
`{{ env_var('NAME') }}` and never enter the manifest - the target config is
stored unrendered.

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
code as the challenger. No champion yet? The gate passes with a loud
warning (bootstrap). Champion exists but cannot load? Hard error - never a
silent pass.

A failing gate blocks registration and exits with code **2**.

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
All seeds derive from the model's mandatory `seed`: the adapter uses `seed`,
tuning samples with `seed + 1`, implicit validation carves with `seed + 2`.
Each adapter documents a determinism tier - exact (XGBoost/LightGBM on CPU,
single-threaded) or tolerance bands.
