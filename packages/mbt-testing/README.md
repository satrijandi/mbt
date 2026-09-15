# mbt-testing

Fake, contract-conformant adapters for testing [mbt](https://github.com/satrijandi/mbt) projects, and mbt itself, without ML frameworks, a JVM, or external services.
Use them to test a project's specs, gates, promotion flow, and CI wiring in seconds.

```bash
pip install mbt-testing        # depends on mbt-core
```

mbt is not on PyPI yet; see [Installation](https://satrijandi.github.io/mbt/installation/) for installing from a release tag.

## The `fake` plugin

| Role | Adapter | Behaviour |
|---|---|---|
| training | `fake` | A deterministic "model" whose every builtin metric is its `fake_metric_value` hyperparameter (default 0.5) plus `max_depth` x 0.0001, so a gate's outcome is whatever the test sets and a tuning search still has something to optimize |
| tracking | `fake` | One JSON file per run under `root` (default `target/fake_tracking`) |
| registry | `fake` | One JSON file per model under `root` (default `target/fake_registry`), readable from both the coordinator and job processes |
| compute | `fake` | Runs jobs in-process, which is fast and easy to debug; use the built-in `local` adapter when a test needs real subprocess isolation |
| tuning | `fake` | A seeded random search that honours `n_trials`, with no Optuna dependency |

## Example

```yaml
# profiles.yml
my_project:
  target: test
  outputs:
    test:
      data: {adapter: local, config: {root: .}}
      tracking: {adapter: fake}
      registry: {adapter: fake}
      compute: {adapter: fake}
      artifact_store: file://./target/artifacts
```

```yaml
# a model spec under test: a champion gate that must fail
models:
  - name: churn_model
    task: binary_classification
    adapter: fake
    owner: tests@example.com
    dataset: ref('churn_training_set')
    target: churned
    hyperparameters: {fake_metric_value: 0.61}
    evaluation:
      protocol: {split: temporal}
      metrics: [pr_auc]
      gates:
        - {metric: pr_auc, threshold: 0.7}
    seed: 7
```

`mbt build` on that project exits `2` with a `gate_failed` model, which is the behaviour a test can assert on.
The fake training adapter supports binary classification only.
It passes the same compliance suite as the real adapters, so a project that works against it exercises the same contract.

## License

Apache License 2.0.
