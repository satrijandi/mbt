# mbt - Model Build Tool

[![CI](https://github.com/satrijandi/mbt/actions/workflows/ci.yml/badge.svg?branch=main)](https://github.com/satrijandi/mbt/actions/workflows/ci.yml?query=branch%3Amain)
[![Live integration](https://github.com/satrijandi/mbt/actions/workflows/live.yml/badge.svg?branch=main)](https://github.com/satrijandi/mbt/actions/workflows/live.yml?query=branch%3Amain)
[![Upstream resolution](https://github.com/satrijandi/mbt/actions/workflows/upstream.yml/badge.svg?branch=main)](https://github.com/satrijandi/mbt/actions/workflows/upstream.yml?query=branch%3Amain)
[![Docs](https://img.shields.io/badge/docs-satrijandi.github.io%2Fmbt-blue)](https://satrijandi.github.io/mbt/)
[![Python 3.11+](https://img.shields.io/badge/python-3.11%20--%203.14-blue)](https://github.com/satrijandi/mbt/blob/main/pyproject.toml)
[![License Apache-2.0](https://img.shields.io/badge/license-Apache--2.0-blue)](https://github.com/satrijandi/mbt/blob/main/LICENSE)

**dbt for machine learning models.**
A model is a reviewed YAML spec - its data, algorithm, hyperparameters, quality gates, and registration target.
Pluggable adapters do the training; a compiled manifest pins data snapshots, config hashes, seeds, and environment digests so runs are reproducible; and state-aware selection retrains only what changed.

> **The model config IS the model.**

```yaml
# models/churn_classifier.yml
models:
  - name: churn_classifier
    task: binary_classification
    adapter: xgboost
    owner: growth-ds@company.com
    dataset: ref('churn_training_set')
    target: churned_90d
    features:
      exclude: [user_id, email]          # explicit leakage guards
    hyperparameters:
      max_depth: 6
      scale_pos_weight: "{{ auto }}"     # adapter computes from class balance
    evaluation:
      protocol: {split: temporal}
      metrics: [pr_auc, roc_auc]
      gates:
        - metric: pr_auc
          compare_to: production          # champion/challenger
          min_delta: 0.005
    registration:
      name: churn_classifier
      stage_on_pass: staging
    seed: 42
```

**[Documentation](https://satrijandi.github.io/mbt/)** · [Installation](https://satrijandi.github.io/mbt/installation/) · [Quickstart](https://satrijandi.github.io/mbt/quickstart/) · [Tutorial](https://satrijandi.github.io/mbt/tutorial/) · [CLI reference](https://satrijandi.github.io/mbt/cli-reference/) · [Spec reference](https://satrijandi.github.io/mbt/spec-reference/)

## Five minutes to a trained, served, monitored model

```bash
# Not on PyPI yet: install from a source checkout
# (once published: pip install mbt-core mbt-xgboost mbt-mlflow)
git clone https://github.com/satrijandi/mbt && cd mbt
uv sync && source .venv/bin/activate     # puts `mbt` on PATH

mbt init my_models && cd my_models
python scripts/generate_sample_data.py   # or point sources.yml at your own Parquet
mbt build                                # datasets -> training -> checks -> gates -> registry
mbt promote --model churn_classifier --to production
mbt score                                # batch-score with the champion + shift monitors
mbt monitor                              # realized metrics once outcomes mature
mbt docs generate && mbt docs serve      # model cards + lineage
```

`mbt build` compiles your specs into a pinned manifest, materializes datasets with DuckDB, trains each model in an isolated process, evaluates its gates against the registry's champion, and registers passing models in MLflow - all from YAML you review in a pull request.
The [tutorial](https://satrijandi.github.io/mbt/tutorial/) walks a data scientist and an MLOps engineer through the same loop as a team, including the CI wiring and alerting.

## Why

Data science teams glue together notebooks, ad-hoc scripts, and bespoke pipeline code, and the *logic* of a model ends up buried in imperative Python.
dbt fixed this for analytics by making transformations declarative, versioned, testable, and dependency-aware.
mbt applies the same idea to model building:

| dbt | mbt |
|---|---|
| model = SQL + config | model = **declarative YAML** (+ an optional `hooks.py`) |
| adapters: Snowflake, BigQuery, ... | adapters: **XGBoost, LightGBM, scikit-learn, H2O AutoML, SparkML** for training; Parquet, Snowflake, and Spark for data |
| `dbt run` materializes tables | `mbt run` trains and registers **model artifacts** |
| `dbt test` | `mbt test`: data checks + **metric gates against the champion** |
| `ref()` DAG of models | `ref()` DAG of **datasets -> models -> scoring pipelines** |
| `state:modified` rebuilds | `state:modified` **retrains only what changed** |

## The GitOps loop

1. A data scientist edits a model spec on a branch.
   The pull-request check compiles and runs `mbt build --select state:modified+ --state <prod manifest>` on the development target, so only the changed subgraph retrains.
2. The PR comment shows metrics against the production champion, the gate results, the retrained nodes, and a cost estimate.
3. Merging builds production, registers passing models to `staging`, and publishes the manifest as the new baseline.
4. A reviewed change to `promotions.yml` runs `mbt promote`, which refuses any version whose gates were not recorded as passed.
5. Scheduled jobs score new batches with whatever holds `production` and evaluate matured predictions against real outcomes.

Reference GitHub Actions workflows for all of this ship with `mbt init`; see [GitOps & CI](https://satrijandi.github.io/mbt/gitops/).

## Reproducibility

The compiled manifest pins everything: data snapshot ids, time windows resolved against a single anchor, config and transitive input hashes, seeds, and an environment digest.

```bash
mbt run --manifest target/manifest.json   # re-execute a stored manifest verbatim
```

That reproduces the original metrics bit for bit on the adapters with an exact determinism tier (XGBoost, LightGBM, scikit-learn), and within a documented tolerance on H2O and SparkML.

## Repository layout

| Package | What it is |
|---|---|
| [`packages/mbt-core`](packages/mbt-core) | CLI, parsing, DAG, compile and manifest, execution engine, gates, state, docs generation; the local DuckDB and subprocess adapters |
| [`packages/mbt-adapter-base`](packages/mbt-adapter-base) | The versioned adapter contract, interchange types, shared metric engine, and **compliance suite** |
| [`packages/mbt-xgboost`](packages/mbt-xgboost) | XGBoost training adapter (exact determinism tier) |
| [`packages/mbt-lightgbm`](packages/mbt-lightgbm) | LightGBM training adapter, built against the public contract only - the extensibility proof |
| [`packages/mbt-sklearn`](packages/mbt-sklearn) | scikit-learn training adapter: LogisticRegression, Ridge, RandomForest, HistGradientBoosting |
| [`packages/mbt-h2o`](packages/mbt-h2o) | H2O AutoML training adapter (MOJO artifacts; optional Sparkling Water backend) |
| [`packages/mbt-spark`](packages/mbt-spark) | Spark adapters: lakehouse data, `spark-submit` compute, distributed SparkML training |
| [`packages/mbt-snowflake`](packages/mbt-snowflake) | Snowflake data adapter: warehouse-native datasets with push-down sampling |
| [`packages/mbt-mlflow`](packages/mbt-mlflow) | MLflow tracking and registry adapters |
| [`packages/mbt-optuna`](packages/mbt-optuna) | Optuna tuning engine (seeded TPE, median pruning) |
| [`packages/mbt-evidently`](packages/mbt-evidently) | Evidently drift reports beside the training report's own stability tables |
| [`packages/mbt-testing`](packages/mbt-testing) | Fake adapters for testing mbt projects without frameworks |
| [`examples/showcase`](examples/showcase) | A docker-compose reference stack - S3 lake, MLflow, Spark cluster, Gitea and Woodpecker CI, Zot, Airflow, Grafana, optional k3d and ArgoCD - with an opt-in live test tier |
| [`tests/fixtures`](tests/fixtures) | Whole mbt projects the suite drives through the real CLI: `churn_demo` (classification) and `revenue_demo` (regression) |
| [`docs`](docs) | The documentation site sources, including the [architecture decision records](docs/adr) |

## Development

See [CONTRIBUTING.md](CONTRIBUTING.md) for the full verification battery and conventions; the short version:

```bash
git clone https://github.com/satrijandi/mbt && cd mbt
uv sync                                   # whole workspace, all extras
uv run pytest -q -m "not e2e" --cov       # fast suite + the 100% coverage gate
uv run pytest -q -m e2e                   # full CLI end-to-end, including the JVM adapters (Java 17)
uv run ruff check . && uv run ruff format --check .
uv run pre-commit install                 # hooks: ruff, yamllint, mypy
uv run mkdocs serve                       # the docs site, locally
```

CI tests what the metadata claims.
The fast suite runs on every CPython the packages advertise (3.11 through 3.14), under an enforced 100% line-coverage gate.
A `floors` job installs every direct dependency at its declared lower bound, re-asserts that it really did, and runs the fast suite and the dependency-advisory audit against that environment, so a floor nobody supports fails CI instead of a user's install.
An `upstream` job re-resolves to the newest versions the constraints allow and runs the fast and end-to-end tiers nightly, so upstream breakage the lock file hides still surfaces.
Security scanning runs as pip-audit, CodeQL, and gitleaks; every GitHub Action is pinned to a commit digest.

Dependency updates are configured for [Renovate](https://docs.renovatebot.com/) in `renovate.json` (pre-commit hooks, GitHub Actions digests, and Python pins, with the ruff hook grouped with the locked ruff), and take effect once the Renovate GitHub App is installed on the repository, which has not happened yet.

Design decisions live in the [ADRs](https://satrijandi.github.io/mbt/adr/); CI builds the documentation site strictly on every pull request and publishes it from `main`.

## Status

**v0.1 is released** as a GitHub release; PyPI publication is pending.

- **Loop**: PR check -> CI build -> registry -> gate-verified promotion -> batch scoring -> ground-truth monitoring, for binary classification and regression.
- **Backends**: Parquet/DuckDB, Snowflake, or Spark lakehouse data; XGBoost, LightGBM, scikit-learn, SparkML, and H2O AutoML training (Sparkling Water for distributed); MLflow tracking and registry; Optuna tuning.
- **Proof**: an enforced 100% coverage gate on the fast suite, a JVM end-to-end tier with its own coverage floor, and a dockerized showcase that runs the loop nightly against real services.

Goal-by-goal evidence: [v0.1 status](https://satrijandi.github.io/mbt/v0.1-status/).
What comes next - PyTorch, Feast, ensembles, warehouse-native prediction stores, Iceberg: [roadmap](https://satrijandi.github.io/mbt/roadmap/).

## Community

- Questions and bugs: [GitHub issues](https://github.com/satrijandi/mbt/issues) - the [troubleshooting runbook](https://satrijandi.github.io/mbt/troubleshooting/) is indexed by exact error text and answers many of them.
- Security reports: privately, as described in [SECURITY.md](SECURITY.md).
- Contributing: [CONTRIBUTING.md](CONTRIBUTING.md); everyone taking part is expected to follow the [Code of Conduct](CODE_OF_CONDUCT.md).

## License

[Apache License 2.0](LICENSE).
