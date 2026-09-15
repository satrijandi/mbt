# mbt-optuna

The Optuna tuning engine for [mbt](https://github.com/satrijandi/mbt), the declarative build tool for machine learning models.
Declare a search space in the model spec, and mbt runs a seeded, reproducible hyperparameter search inside the training job.

```bash
pip install mbt-optuna         # plus mbt-core and a training adapter
```

mbt is not on PyPI yet; see [Installation](https://satrijandi.github.io/mbt/installation/) for installing from a release tag.

## Declare a search

```yaml
# models/churn_classifier.yml
models:
  - name: churn_classifier
    # ...task, adapter, dataset, target, evaluation, seed
    tuning:
      engine: optuna
      n_trials: 50                       # capped by the target's max_tuning_trials var
      search_space:
        max_depth: {type: int, low: 3, high: 10}
        learning_rate: {type: loguniform, low: 0.005, high: 0.3}
        subsample: {type: uniform, low: 0.6, high: 1.0}
      objective: {metric: pr_auc, direction: maximize, robust: true}
      pruner: median
```

| Field | Meaning |
|---|---|
| `n_trials` | Trials to run; a target's `max_tuning_trials` var caps it, so pull-request builds stay cheap |
| `search_space` | One dimension per hyperparameter: `int`, `uniform`, or `loguniform` with `low`/`high`, or `categorical` with `choices` |
| `objective` | A metric the model computes, its `direction`, and optionally `robust: true` to select on the bootstrap lower bound of the validation metric instead of its point estimate |
| `pruner` | `median` stops a trial whose per-round validation value falls below the median of earlier trials at the same step. It needs an adapter that reports progress (XGBoost and LightGBM do); with any other adapter, trials run to completion and mbt warns |

Each trial scores on a validation split: the dataset's `split.validation` when it declares one, or otherwise a slice mbt carves from the training window (its most recent 20% for a temporal split, a seeded random 20% for a random one), with trials training on the rest.
**Tuning never sees the test split** ([ADR-8](https://satrijandi.github.io/mbt/adr/0008-tuning-never-sees-test/)), so the gated test metrics stay an honest estimate.
Trials never calibrate, even when the spec sets `calibration:`; only the final model, fit with the best parameters, does.

## Operational settings

Sampler settings belong to the environment, not the model, so they live in `profiles.yml` and never change a model's config hash.

```yaml
# profiles.yml, under a target
tuning:
  adapter: optuna
  config: {sampler: tpe, multivariate: true}
```

| Key | Default | Meaning |
|---|---|---|
| `sampler` | `tpe` | `tpe` or `random` |
| `multivariate` | `false` | TPE only: model correlated hyperparameters jointly |
| `group` | `false` | With `multivariate`, model the whole search space as one group |
| `n_startup_trials` | 5 | Median pruner: trials that always run to completion first |
| `n_warmup_steps` | 5 | Median pruner: rounds before a trial can be pruned |

## Reproducibility

The sampler is seeded with the model's `seed + 1`, so the same spec and data propose the same trials and pick the same best parameters.
A run records the best objective value, the trial and pruned counts, and the best parameters in its tracking run, and each trial as a nested run.
A search in which every trial was pruned fails with a hint to raise `n_startup_trials` or drop the pruner.

## License

Apache License 2.0.
