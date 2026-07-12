# mbt - Model Build Tool

**"dbt for machine learning models": declarative Model-as-Code, adapter-based, GitOps-native.**

A model is a reviewed YAML spec: data, algorithm, hyperparameters, evaluation gates, registration target.
Pluggable adapters execute training.
A compiled manifest pins data snapshots, config hashes, seeds, time anchors, and environment digests so runs are reproducible.
State-aware selection (`state:modified+`) retrains only what changed, making ML-in-CI economical.

> The model config IS the model.

## Where to start

- New to mbt? Follow the [Quickstart](quickstart.md) - from `mbt init` to a trained, registered model in well under an hour.
- Coming from dbt? Read [Concepts](concepts.md); most of your muscle memory transfers.
- Setting up CI? See [GitOps & CI](gitops.md) for the PR check / prod build / promotion loop.
- Want to see the whole platform running? The [Showcase](showcase.md) boots an S3 lake, Spark cluster, MLflow, a CI forge, an OCI registry, Airflow, and Grafana with docker compose - and runs the full lifecycle on them.
- Building an adapter? The [Adapter authoring guide](adapter-authoring.md) plus the compliance suite is everything you need - no mbt-core knowledge required.

## Design principles

1. **Declarative first, escape hatches second.** 90% of models need zero custom code; a `hooks.py` covers the rest without breaking the contract.
2. **Adapters own execution.** The core never imports an ML framework; it defines contracts, adapters implement them.
3. **Deterministic and reproducible.** Same manifest, same results - exactly, for adapters with an exact determinism tier.
4. **CI is the primary user.** Non-interactive commands, meaningful exit codes (0 ok / 1 error / 2 quality failure), machine-readable artifacts.
5. **State-aware.** Training is 1000x more expensive than a dbt view; never retrain what did not change.
