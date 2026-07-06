# mbt-spark

One plugin, three Spark seams:

## Data: lakehouse dataset construction

```yaml
# profiles.yml
data:
  adapter: spark
  config:
    master: local[*]              # or spark://..., yarn
    conf: {spark.executor.memory: 8g}

# sources.yml - paths (parquet/delta) or catalog identifiers
sources:
  - name: lake
    tables:
      - name: churn_labels
        path: s3://lake/gold/churn_labels     # format: delta also supported
      - name: features
        identifier: gold.customer_features    # Hive/Unity catalog table
```

Joins (`inputs:` feature+label form), filters, deterministic `sample_key`
sampling, and split windows all push down as Spark SQL; splits land in the
standard mbt materialization, so training jobs reopen them with no Spark
session or credentials.

## Compute: training jobs under spark-submit

```yaml
compute:
  adapter: spark
  config: {master: "spark://cluster:7077", deploy_mode: client}
```

The serialized mbt `TrainingJob` runs on the driver via a shipped wrapper -
any training adapter (XGBoost, LightGBM, H2O...) gains cluster-sized memory
without code changes. The driver's Python env must have mbt-core and the
adapter installed. `master: local[*]` doubles as a memory-isolated local
runner and is how the seam is tested.

## Training: distributed SparkML

```yaml
models:
  - name: churn_gbt
    adapter: spark                 # VectorAssembler + GBTClassifier pipeline
    hyperparameters: {max_iter: 100, max_depth: 5}
    ...
```

Session config comes from target vars (`spark_master`, `spark_conf`);
artifacts are zipped PipelineModel directories; determinism tier is
tolerance (distributed reduction order).
