# s3_wide - a multi-table dataset from an S3 lake

The same wide multi-table dataset as [`../snowflake_wide`](../snowflake_wide),
but the data source is an **S3 (or S3-compatible) parquet lake** instead of
Snowflake. Five tables joined on `[customer_id, snapshot_date]`:

| Role | Source table (parquet under `root`) | Join key |
|---|---|---|
| population (spine) | `customer_population/*.parquet` | `[customer_id, snapshot_date]` |
| label | `churn_labels/*.parquet` | `[customer_id, snapshot_date]` |
| feature | `demographic_features/*.parquet` | `[customer_id, snapshot_date]` |
| feature | `engagement_features/*.parquet` | `[customer_id, snapshot_date]` |
| feature | `billing_features/*.parquet` | `[customer_id, snapshot_date]` |

**What changed vs `snowflake_wide` is only the data plane** (`profiles.yml` +
`sources.yml`); the dataset and model specs are identical. In mbt, S3 is read
through the **Spark data adapter** over `s3a`:

```yaml
data:
  adapter: spark
  config:
    master: local[*]
    root: s3://my-lake                 # your bucket
    conf:
      spark.hadoop.fs.s3.impl: org.apache.hadoop.fs.s3a.S3AFileSystem
      spark.hadoop.fs.s3a.endpoint: "{{ env_var('AWS_S3_ENDPOINT', 's3.amazonaws.com') }}"
      spark.hadoop.fs.s3a.access.key: "{{ env_var('AWS_ACCESS_KEY_ID') }}"
      spark.hadoop.fs.s3a.secret.key: "{{ env_var('AWS_SECRET_ACCESS_KEY') }}"
```

Sources use `path:` globs relative to `root`, so `customer_population/*.parquet`
resolves to `s3://my-lake/customer_population/*.parquet`.

## Validate the config (offline, no S3, no JVM)

```bash
uv run mbt parse --project-dir examples/s3_wide          # -> Parsed 7 resources [OK]
```

## Run it against S3

The Spark plane needs a JVM (Java 17) and an S3 endpoint (real AWS S3, or an
S3-compatible store like MinIO/SeaweedFS):

```bash
export AWS_ACCESS_KEY_ID=... AWS_SECRET_ACCESS_KEY=...
export AWS_S3_ENDPOINT=s3.amazonaws.com      # or http://minio:9000 for MinIO
export AWS_S3_SSL=true                        # set false for plain-HTTP MinIO/SeaweedFS
export JAVA_HOME=/opt/homebrew/opt/openjdk@17

uv run mbt build --project-dir examples/s3_wide
```

`build` materializes `wide_churn_training` (the five-table join runs in Spark,
reading parquet over s3a), trains `churn_wide`, evaluates its gate, and registers
on pass. Point `root` at your bucket and adjust the source globs to your layout.

## Seeing the join result

The **join semantics are identical** to `snowflake_wide` (same `inputs:` spec).
For an offline, no-account demonstration of exactly what that join produces, run
`snowflake_wide`'s script - it prints the generated query and the joined panel:

```bash
uv run python examples/snowflake_wide/show_wide_join.py
```

The Spark plane produces the same joined columns; the difference is only where
the bytes live (S3 parquet vs Snowflake tables) and the engine that runs the
join (Spark vs Snowflake SQL).
