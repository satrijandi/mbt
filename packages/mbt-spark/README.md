# mbt-spark

Spark adapters for [mbt](https://github.com/satrijandi/mbt), the declarative build tool for machine learning models.
One package covers three roles:

| Role | Adapter | What it does |
|---|---|---|
| data | `spark` | Builds datasets and scoring batches from Parquet, Delta, or catalog tables, with filters, sampling, and splits pushed down as Spark SQL |
| compute | `spark` | Runs any mbt training job under `spark-submit`, so a model gets a driver with cluster-sized memory |
| training | `spark` | Trains a distributed SparkML gradient-boosted trees pipeline |

```bash
pip install mbt-spark    # plus mbt-core; needs Java 17
```

mbt is not on PyPI yet; see [Installation](https://satrijandi.github.io/mbt/installation/) for installing from a release tag.
The package supports pyspark 3.5.8 up to, but not including, 4.2, and is tested on Spark 3.5 and 4.1.
It depends on `mbt-core`, because the compute adapter reuses the engine's job wire format.

## Data: lakehouse datasets

```yaml
# profiles.yml
data:
  adapter: spark
  config:
    master: local[*]                            # or spark://host:7077, yarn
    conf: {spark.executor.memory: 8g}
    root: s3a://lake                            # optional prefix for path sources
    predictions_root: s3a://lake/predictions    # where mbt score stages predictions
    source_address: path                        # only for tables declaring BOTH addresses

# sources.yml - paths (parquet or delta) or catalog identifiers
sources:
  - name: lake
    tables:
      - name: churn_panel
        path: gold/churn_panel
        format: delta                           # parquet is the default
      - name: customer_panel
        identifier: gold.customer_panel         # a Hive or Unity Catalog table
```

A dataset reads **one relation** ([ADR-29](https://satrijandi.github.io/mbt/adr/0029-single-relation-datasets/)); whatever joins it is upstream.
Filters, deterministic `sample_key` sampling, and split windows push down as Spark SQL, and each split lands as one Parquet file in the standard mbt materialization, so training jobs reopen it without a Spark session or credentials.

| Key | Default | Meaning |
|---|---|---|
| `master` | `local[*]` | Spark master URL |
| `conf` | `{}` | Extra Spark configuration |
| `root` | none | Prefix joined to `path:` sources |
| `predictions_root` | `<tmpdir>/mbt-predictions` | Where prediction runs are staged as Parquet; the default is cleared on reboot, so set a durable location in production |
| `source_address` | unset | `path` or `identifier`: which address to read for a table that declares both |

**Snapshots.**
A local path is pinned by its file listing (path, size, mtime), or by file contents under `--deep-snapshot`.
A URI or catalog table is pinned by the hash of its input-file listing, which is already independent of checkout mtimes, so deep and shallow tokens agree there.

**Sampling.**
Rows hash to buckets with `conv(substring(md5(key), 17, 16), 16, 10)`, the same digest as Snowflake's `MD5_NUMBER_LOWER64` and the local DuckDB adapter, so a fraction selects the same rows on every backend.

### Tables that declare both a path and an identifier

Spark is the only adapter that reads both object-store directories and catalog tables, so it is the only one that can be handed an ambiguous source.
Declaring both addresses is a supported pattern - it is how one `sources.yml` serves a file plane and a warehouse plane, with the target choosing - but which one a target reads is a property of the target, not of the table, so mbt will not guess:

- a table declaring one address is read by that address, whatever `source_address` says;
- a table declaring **both** fails the compile unless `source_address` says which.

The error names the table and the setting, and fires during compile-time snapshot pinning rather than partway through a build.

### Batch scoring

`mbt score` and `mbt monitor` run against a Spark target: the unlabeled batch is built through the same path (one relation, filters, `sample_key` sampling, the `score` window), and prediction runs are staged as Parquet under `predictions_root` joined with the scoring node's `output.path`.
A store backed by lakehouse tables is designed in [ADR-23](https://satrijandi.github.io/mbt/adr/0023-warehouse-batch-scoring/) and not yet shipped.
The [showcase](https://github.com/satrijandi/mbt/tree/main/examples/showcase)'s `batch` target scores and monitors this way on every run, straight off an S3-compatible object store.

## Compute: training jobs under spark-submit

```yaml
compute:
  adapter: spark
  config:
    master: spark://cluster:7077
    deploy_mode: client
    conf: {spark.driver.memory: 16g}
    job_timeout_seconds: 7200
```

| Key | Default | Meaning |
|---|---|---|
| `master` | `local[*]` | Spark master URL |
| `deploy_mode` | unset | Passed to `spark-submit --deploy-mode` |
| `conf` | `{}` | Passed as `--conf key=value` |
| `spark_submit` | `spark-submit` | The executable to call |
| `job_timeout_seconds` | none | Kill a job that outlives it, as on the local compute adapter |

The serialized mbt training job runs on the driver through a small wrapper, and its result and event stream come back exactly as they do from a local subprocess.
Any training adapter - XGBoost, LightGBM, H2O - gains cluster-sized memory without code changes.
The driver's Python environment must have mbt-core and the model's training adapter installed.
`master: local[*]` doubles as a memory-isolated local runner.

## Training: distributed SparkML

```yaml
models:
  - name: churn_gbt
    task: binary_classification          # or regression
    adapter: spark
    owner: growth-ds@company.com
    dataset: ref('churn_training_set')
    target: churned_90d
    hyperparameters: {max_iter: 100, max_depth: 5}
    evaluation:
      protocol: {split: temporal}
      metrics: [pr_auc, roc_auc]
    seed: 42
```

| Hyperparameter | Default |
|---|---|
| `max_iter` | 50 |
| `max_depth` | 5 |
| `step_size` | 0.1 |
| `subsampling_rate` | 1.0 |
| `min_instances_per_node` | 1 |
| `max_bins` | 32 |

The adapter assembles a `VectorAssembler` plus `GBTClassifier` (binary classification, where the score is P(class 1)) or `GBTRegressor` (regression) pipeline.
The Spark session comes from the target vars `spark_master` and `spark_conf`, so development can run `local[*]` while production points at a cluster.
Artifacts are zipped `PipelineModel` directories, and the determinism tier is tolerance (0.01), because distributed reduction order is not fixed.
`calibration:` is supported.
`features.monotonic` and `categorical.min_frequency` are not, and fail at `mbt parse`; a scoring pipeline's `output.explain_top_k` is not either, and fails when `mbt score` runs.
Model cards show the GBT model's own feature importance rather than SHAP values.

String feature columns get a `StringIndexer` stage each (ordinal codes, with `handleInvalid=keep` for levels first seen at scoring time), so the same spec that trains on the tree adapters also trains on Spark.
Any other non-numeric feature type raises an actionable error rather than a raw JVM `IllegalArgumentException`.

**Categorical parity caveat.**
This is ordinal indexing, not native categorical handling: the frequency-ranked integer code goes into `VectorAssembler`, and the trees split on it as a number, whereas XGBoost, LightGBM, and H2O split on categories directly.
On a categorical-heavy dataset Spark therefore produces a materially different model family.
The metric engine is shared, so metrics are computed identically, but a champion/challenger comparison between Spark and a tree adapter is not like for like.
Keep such comparisons within one adapter family, or treat the Spark model as a distinct one.

## License

Apache License 2.0.
