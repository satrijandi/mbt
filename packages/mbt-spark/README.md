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
    predictions_root: s3://lake/predictions   # optional; batch-scoring sink base
    source_address: path          # optional; only for tables declaring BOTH

# sources.yml - paths (parquet/delta) or catalog identifiers
sources:
  - name: lake
    tables:
      - name: churn_labels
        path: s3://lake/gold/churn_labels     # format: delta also supported
      - name: features
        identifier: gold.customer_features    # Hive/Unity catalog table
```

### Tables that declare both a path and an identifier

Spark is the only adapter that reads both object-store directories and catalog
tables, so it is the only one that can be handed an ambiguous source. Declaring
both addresses is a supported pattern - it is how one `sources.yml` serves a
file plane and a warehouse plane, with the target choosing - but *which* one a
given target reads is a property of the target, not of the table, so mbt will
not guess:

- a table declaring one address is read by that address, `source_address` or not;
- a table declaring **both** fails the compile unless `source_address` says which.

Set it once per target (`source_address: path` on the lake targets,
`identifier` on the catalog ones). The error names the table and the knob, and
it fires during compile-time snapshot pinning rather than partway through a
build.

Joins (`inputs:` feature+label form), filters, deterministic `sample_key`
sampling, and split windows all push down as Spark SQL; splits land in the
standard mbt materialization, so training jobs reopen them with no Spark
session or credentials.

Batch scoring (contract 1.1) ships too: `mbt score`/`mbt monitor` materialize
an unlabeled batch through the same Spark path (spine + feature joins, the
`score` window, `sample_key` sampling) and stage prediction runs as parquet
under `predictions_root`/`output.path` (default the project dir), the same
staged store the Snowflake adapter uses. A lakehouse-table-backed store is the
ADR-23 v2 design.

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
    adapter: spark                 # VectorAssembler + GBT (Classifier/Regressor) pipeline
    hyperparameters: {max_iter: 100, max_depth: 5}
    ...
```

Session config comes from target vars (`spark_master`, `spark_conf`);
artifacts are zipped PipelineModel directories; determinism tier is
tolerance (distributed reduction order).

Both tasks are supported: `task: binary_classification` trains a
`GBTClassifier` (score = P(class 1)) and `task: regression` a `GBTRegressor`
(target-scale predictions) - the same spec runs on Spark as on the tree
adapters.

String feature columns are handled: a `StringIndexer` stage is inserted
per string feature (ordinal codes, `handleInvalid=keep` for unseen levels
at score time), so the same spec that trains on the tree adapters also
trains on Spark. Any other non-numeric feature type (timestamp, array, ...)
raises an actionable error rather than a raw JVM `IllegalArgumentException`.

**Categorical parity caveat.** This is ordinal indexing, not native
unordered-categorical handling: the frequency-ranked integer code is fed to
`VectorAssembler` and the `GBT` does numeric threshold splits on that code,
whereas the tree adapters (xgboost / lightgbm / h2o) do true unordered
partition splits over the categories via the shared `encoding.py`. So for a
categorical-heavy dataset the *feature representation* differs and Spark
produces a materially different model family - the metric engine is shared, so
metrics are computed identically, but "same spec, different adapter,
apples-to-apples champion/challenger" does **not** hold with Spark as champion
or challenger against a tree adapter. Keep such a comparison within one adapter
family, or treat the Spark result as a distinct model. A native-categorical
Spark path is future work.
