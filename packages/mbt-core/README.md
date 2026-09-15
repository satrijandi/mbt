# mbt-core

**mbt is dbt for machine learning models.**
A model is a reviewed YAML spec - its data, algorithm, hyperparameters, quality gates, and registration target - and mbt compiles it into a pinned, reproducible plan, trains it through pluggable adapters, gates it against the production champion, and registers it.
State-aware selection retrains only what changed, which makes training economical enough to run on every pull request.

`mbt-core` is the engine and the `mbt` command-line tool.
It includes the local adapters (DuckDB over Parquet for data, a subprocess for compute); training, tracking, and warehouse integrations are separate packages.

```bash
pip install mbt-core mbt-xgboost mbt-mlflow
```

mbt is not on PyPI yet; see [Installation](https://satrijandi.github.io/mbt/installation/) for installing from a release tag.
Python 3.11 through 3.14 are supported.

## Five minutes to a trained, served, monitored model

```bash
mbt init my_models && cd my_models
python scripts/generate_sample_data.py
mbt build                                     # compile -> datasets -> train -> gates -> register
mbt promote --model churn_classifier --to production
mbt score                                     # batch-score with the champion, check for drift
mbt monitor                                   # evaluate predictions once outcomes mature
mbt docs generate && mbt docs serve           # model cards and lineage
```

`mbt init` scaffolds a working project: example specs, `profiles.yml` for development and production targets, reference GitHub Actions workflows for the pull-request check, production build, promotion, and the scheduled retrain, scoring, and monitoring loops, and pinned CI requirements.

## What a model looks like

```yaml
models:
  - name: churn_classifier
    task: binary_classification
    adapter: xgboost
    owner: growth-ds@company.com
    dataset: ref('churn_training_set')
    target: churned_90d
    features:
      exclude: [user_id]
    hyperparameters:
      max_depth: 6
      scale_pos_weight: "{{ auto }}"
    evaluation:
      protocol: {split: temporal}
      metrics: [pr_auc, roc_auc]
      gates:
        - {metric: pr_auc, compare_to: production, min_delta: 0.005}
    registration:
      name: churn_classifier
      stage_on_pass: staging
    seed: 42
```

## Commands

| Command | Does |
|---|---|
| `mbt parse` / `mbt compile` | Validate every spec; pin the plan into `target/manifest.json` |
| `mbt build` / `mbt run` / `mbt test` | Build datasets, train, check, and gate in DAG order, with dbt's `--select` grammar and `state:modified+` |
| `mbt score` / `mbt monitor` / `mbt predictions` | Batch scoring with shift monitors, and delayed ground-truth evaluation |
| `mbt promote` / `mbt rollback` / `mbt evaluate` | Gate-verified stage transitions and re-evaluation of registered versions |
| `mbt ls` / `mbt show` / `mbt state diff` / `mbt docs` | Inspect resources, compare against a production manifest, and generate model cards |

Exit codes carry meaning everywhere: `0` success, `1` hard error, `2` quality failure (a gate, check, or monitor said no).

## Extras

| Extra | Adds |
|---|---|
| `mbt-core[s3]` | `s3://` artifact stores and state manifests |
| `mbt-core[otel]` | OpenTelemetry spans for every command and node (`MBT_OTEL=1`) |

## Documentation

- [Quickstart](https://satrijandi.github.io/mbt/quickstart/) and the team [Tutorial](https://satrijandi.github.io/mbt/tutorial/)
- [Concepts](https://satrijandi.github.io/mbt/concepts/), the [Spec reference](https://satrijandi.github.io/mbt/spec-reference/), and the [CLI reference](https://satrijandi.github.io/mbt/cli-reference/)
- [Adapters](https://satrijandi.github.io/mbt/adapters/) and [GitOps & CI](https://satrijandi.github.io/mbt/gitops/)
- [Source, issues, and contributing](https://github.com/satrijandi/mbt)

## License

Apache License 2.0.
