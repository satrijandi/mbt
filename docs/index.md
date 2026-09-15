# mbt - Model Build Tool

**dbt for machine learning models: declarative, adapter-based, and built for GitOps.**

In mbt a model is a reviewed YAML spec - its data, algorithm, hyperparameters, quality gates, and registration target.
Adapters do the training on XGBoost, LightGBM, scikit-learn, H2O AutoML, or SparkML, reading data from Parquet, Snowflake, or a Spark lakehouse.
A compiled manifest pins data snapshots, config hashes, seeds, time windows, and the environment, so a run can be reproduced, and state-aware selection retrains only what changed.

> The model config IS the model.

```yaml
models:
  - name: churn_classifier
    task: binary_classification
    adapter: xgboost
    owner: growth-ds@company.com
    dataset: ref('churn_training_set')
    target: churned_90d
    features:
      exclude: [user_id]                 # a reviewable leakage guard
    hyperparameters:
      max_depth: 6
      scale_pos_weight: "{{ auto }}"     # computed from the class balance
    evaluation:
      protocol: {split: temporal}
      metrics: [pr_auc, roc_auc]
      gates:
        - metric: pr_auc
          compare_to: production         # beat the champion, with confidence
          min_delta: 0.005
    registration:
      name: churn_classifier
      stage_on_pass: staging
    seed: 42
```

## Start here

<div class="grid cards" markdown>

- **New to mbt?**
  [Install it](installation.md), then follow the [Quickstart](quickstart.md): from `mbt init` to a trained, registered, served, and monitored model in a few minutes.
- **Working in a team?**
  The [Tutorial](tutorial.md) walks a data scientist and an MLOps engineer through the whole lifecycle, including the pull-request loop and production wiring.
- **Coming from dbt?**
  Read [Concepts](concepts.md); refs, sources, selectors, `state:modified`, and profiles all work the way you expect.
- **A data scientist who wants the why?**
  [Training, explained for data scientists](ds-primer.md) covers splits, feature selection, gates, and why exit code 2 is on your side.
- **Setting up CI?**
  [GitOps & CI](gitops.md) covers the PR check, the production build, promotion, and the scheduled loops that `mbt init` ships.
- **Want to see it on real infrastructure?**
  The [Showcase](showcase.md) boots an object-store lake, a Spark cluster, MLflow, a CI forge, Airflow, and Grafana with docker compose, and runs the full lifecycle on them.
- **Something failed?**
  The [Troubleshooting runbook](troubleshooting.md) is indexed by the exact error text.
- **Extending mbt?**
  [Adapter authoring](adapter-authoring.md) and the compliance suite are all you need to ship a new adapter; [Architecture](architecture.md) maps the engine.

</div>

## What mbt does

| Capability | What you get |
|---|---|
| **Declarative models** | Datasets, models, and scoring pipelines are YAML, validated at `mbt parse` with every error reported at once. An optional `hooks.py` covers the rest |
| **Quality gates** | Threshold, champion/challenger (a paired-bootstrap lower bound, not a lucky decimal), slice, and fairness gates block registration. Leakage and data checks run on every build |
| **Reproducibility** | Mandatory seeds, pinned snapshots, and environment digests; `mbt run --manifest` re-executes a stored plan, bit for bit on the exact-tier adapters |
| **Economical CI** | `state:modified+` retrains only the changed subgraph; the scaffolded PR check posts metrics against the production champion |
| **Batch serving** | A `scoring` spec is a whole serving pipeline: the champion resolves from the registry at run time, and shift monitors compare every batch with the training baseline |
| **Ground-truth monitoring** | `mbt monitor` joins outcomes to stored predictions once they mature and gates on realized metrics, evaluating each run exactly once |
| **Pluggable** | Training, data, tracking, registry, compute, and tuning are [adapters](adapters.md) behind a versioned contract |

## Design principles

1. **Declarative first, escape hatches second.** Most models need no custom code; a `hooks.py` covers the rest without breaking the contract.
2. **Adapters own execution.** The engine never imports an ML framework; it defines contracts, and adapters implement them.
3. **Deterministic and reproducible.** The same manifest gives the same results - exactly, for adapters with an exact determinism tier.
4. **CI is the primary user.** Commands are non-interactive, exit codes carry meaning (0 ok, 1 error, 2 quality failure), and every artifact is machine-readable.
5. **State-aware.** Training costs far more than a dbt view, so never retrain what did not change.

## Project status

mbt is at **v0.1**: binary classification and regression, the full PR-to-monitoring loop, and every adapter listed above, proven by an enforced 100% coverage gate on the fast suite, a JVM end-to-end tier, and a nightly run of the dockerized showcase.
The packages are released on GitHub and not yet on PyPI.
See the [v0.1 status](v0.1-status.md) for the evidence and the [Roadmap](roadmap.md) for what comes next.
mbt is open source under the [Apache License 2.0](https://github.com/satrijandi/mbt/blob/main/LICENSE); contributions are welcome - start with [CONTRIBUTING](https://github.com/satrijandi/mbt/blob/main/CONTRIBUTING.md).
