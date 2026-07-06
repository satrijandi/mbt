# mbt-snowflake

Snowflake DataAdapter for [mbt](../../README.md): declare feature and label
tables living in Snowflake, and mbt builds reproducible, pinned training
sets from them - filters, sampling, and split assignment all push down as
SQL; rows stream back as Arrow batches straight into the standard local
materialization that training jobs consume.

```yaml
# profiles.yml
my_project:
  target: prod
  outputs:
    prod:
      data:
        adapter: snowflake
        config:
          account: "{{ env_var('SNOWFLAKE_ACCOUNT') }}"
          user: "{{ env_var('SNOWFLAKE_USER') }}"
          password: "{{ env_var('SNOWFLAKE_PASSWORD') }}"
          warehouse: ML_WH
          database: ANALYTICS
          schema: GOLD
          role: ML_ROLE            # optional
      ...
```

```yaml
# sources.yml - warehouse tables use `identifier`
sources:
  - name: snowflake
    tables:
      - name: churn_labels
        identifier: GOLD.CHURN_LABELS          # db defaults from config
      - name: customer_features
        identifier: GOLD.CUSTOMER_FEATURES
```

```yaml
# datasets/churn_training_set.yml - feature table(s) + label table + join key
datasets:
  - name: churn_training_set
    inputs:
      label: source('snowflake', 'churn_labels')
      features:
        - source('snowflake', 'customer_features')
        - source('snowflake', 'usage_features')
      join_key: [customer_id, snapshot_date]
    label:
      column: churned_90d
    sample_key: [customer_id]       # deterministic push-down sampling
    split:
      strategy: temporal
      time_column: snapshot_date
      train: "-180d:-28d"
      test: "-28d:now"
```

## Guarantees

- **Snapshots**: compile pins `SYSTEM$LAST_CHANGE_COMMIT_TIME` per table (a
  cheap metadata call); `--deep-snapshot` switches to `HASH_AGG(*)` content
  fingerprints. Any table changing marks the dataset `state:modified`.
- **Reproducible sampling** (`sample_fraction` var): rows are kept when
  `MOD(MD5_NUMBER_LOWER64(<sample_key>), 1e6) < fraction * 1e6` - pushed
  into the warehouse query, so a 1% dev sample of a 7M-row table never
  leaves Snowflake. Same fraction → same rows; smaller fractions are
  subsets of larger ones. Requires `sample_key` (or `inputs.join_key`).
- **Case handling**: unquoted Snowflake identifiers come back UPPERCASE;
  the adapter lowercases result columns to match mbt spec conventions
  (disable with `normalize_case: false`).
- Rows stream via the connector's Arrow batch API into one parquet file per
  split; nothing is held fully in memory on the mbt side.
