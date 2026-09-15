# mbt-mlflow

MLflow tracking and model-registry adapters for [mbt](https://github.com/satrijandi/mbt), the declarative build tool for machine learning models.
One plugin, `mlflow`, fills two roles: it records training runs, and it is the registry mbt promotes champions in and resolves them from.

```bash
pip install mbt-mlflow         # plus mbt-core
```

mbt is not on PyPI yet; see [Installation](https://satrijandi.github.io/mbt/installation/) for installing from a release tag.

## Configure a target

```yaml
# profiles.yml
my_project:
  target: dev
  outputs:
    dev:
      tracking: {adapter: mlflow, config: {uri: "sqlite:///mlflow.db"}}
      registry: {adapter: mlflow, config: {uri: "sqlite:///mlflow.db"}}
      # ...data, compute, artifact_store
    prod:
      tracking:
        adapter: mlflow
        config:
          uri: "{{ env('MLFLOW_TRACKING_URI') }}"
          experiment: "{{ env('EXPERIMENT_NAME', 'v2') }}"
      registry: {adapter: mlflow, config: {uri: "{{ env('MLFLOW_TRACKING_URI') }}"}}
```

| Key | Role | Default | Meaning |
|---|---|---|---|
| `uri` | both | `sqlite:///mlflow.db` | The MLflow tracking or registry URI. A relative sqlite path resolves against the project directory |
| `experiment` | tracking | the project name | The second half of the experiment name, which mbt composes as `<project>__<experiment>` |
| `use_aliases` | registry | `true` | Map mbt stages to registered-model aliases. Set `false` for MLflow servers older than 2.9, which only have the deprecated stage API |

## Tracking

**Only training opens a run** ([ADR-28](https://satrijandi.github.io/mbt/adr/0028-mlflow-training-only-and-champion-carried-config/)).
`mbt score` and `mbt monitor` write to the prediction store instead, so a scoring schedule cannot fail on a tracking-server outage, and the tracking server holds modelling work rather than serving volume.

Runs land in one experiment per project and `experiment:` value.
With `name: churn_lake` in `mbt_project.yml` and `experiment: wide_v2`, that is `churn_lake__wide_v2`; without `experiment:`, it is just `churn_lake`.
Each run is named `<run_id>-<model>`, so retraining a model never produces two runs with the same name.

Every run carries:

- the hyperparameters and seed as parameters, and the test-split metrics;
- identity tags: `mbt.project`, `mbt.run_id`, `mbt.config_hash`, `mbt.input_hash`, `mbt.manifest_hash`, `mbt.snapshot_id`, `mbt.git_commit`;
- two documents: `inference_config.json` (exactly what was trained and how to score it) and, when the model has one, its `hooks.py` source;
- for a tuned model, the best objective value and `mbt.tuning.*` tags (trial count, pruned count, best parameters), with each trial as a nested run holding its parameters and objective value.

Model binaries stay in mbt's artifact store; the run holds a pointer tag to them, not a copy.

## Registry

mbt's three stages, `staging`, `production`, and `archived`, map to registered-model aliases of the same names, and a version holds at most one of them.
A version registers only after every gate passes, with `mbt.gates_passed=true` and its artifact reference and content hash as version tags, which is what `mbt promote` checks before moving a stage.
Promoting a version to `production` moves the displaced champion to `archived`.
Scoring resolves the champion by alias at run time, so a promotion takes effect on the next scheduled run without redeploying anything.

A locked SQLite backend - the usual local and CI failure when several jobs write at once - is retried with jittered backoff.
A genuinely absent model, version, or alias reads as "no champion", while any other registry error propagates, so a transient outage can never be mistaken for a missing champion and silently skip a champion gate.

## Learn more

- [Where tracking runs land](https://satrijandi.github.io/mbt/spec-reference/#where-tracking-runs-land)
- [GitOps & CI](https://satrijandi.github.io/mbt/gitops/) for promotion from a reviewed `promotions.yml`

## License

Apache License 2.0.
