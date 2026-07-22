# mbt-h2o

H2O AutoML as an mbt training adapter: the spec declares the *search*, the
leader model is the artifact (a single self-contained **MOJO** zip).

```yaml
models:
  - name: churn_automl
    task: binary_classification
    adapter: h2o_automl
    dataset: ref('churn_training_set')
    target: churned_90d
    hyperparameters:
      max_models: 20
      include_algos: [GLM, GBM, XGBoost]
      sort_metric: aucpr
      nfolds: 0
    evaluation:
      protocol: {split: temporal}
      metrics: [pr_auc, roc_auc]
      gates:
        - {metric: pr_auc, compare_to: production, min_delta: 0.005}
    seed: 42
```

Rules of the road:

- **No `tuning:` block** - AutoML is the tuner (rejected at parse).
- **Both tasks**: `binary_classification` (score = P(class 1)) and `regression`
  (target-scale predictions). AutoML detects the task from the target - a
  numeric target trains a regressor - so leave `sort_metric` at `AUTO` for
  regression (`auc`/`aucpr`/`logloss` are classification-only).
- **Models-bounded runs are repeatable** (`max_models` + `seed`, tolerance
  determinism tier); `max_runtime_secs*` budgets are time-dependent and
  trigger nondeterminism warnings (FR-RUN-06).
- Champion evaluation and `mbt evaluate` reload the MOJO - no H2O model
  registry required beyond mbt's own.
- Needs a JVM (Java 8-17) wherever training jobs run.

## Distributed: Sparkling Water

```bash
pip install 'mbt-h2o[sparkling]'   # pins pyspark 3.5 + h2o-pysparkling-3.5
```

```yaml
# profiles.yml target vars
vars:
  h2o_backend: sparkling
  spark_master: spark://cluster:7077
  spark_conf: {spark.executor.memory: 8g}
```

Same adapter, same specs: H2O runs on Spark executors, so AutoML trains on
frames that never fit one machine. The H2O <-> Spark version matrix is
strict; keep the extra's pins.
