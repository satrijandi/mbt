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

## Authentication

The config keys `account`, `user`, `password`, `warehouse`, `database`, `schema`, `role`, and `authenticator` pass straight to `snowflake.connector.connect()`.
Every other documented connector parameter works under `connect_args`.

SSO from a laptop (the usual data-scientist setup - no password anywhere):

```yaml
config:
  account: "{{ env_var('SNOWFLAKE_ACCOUNT') }}"
  user: "{{ env_var('SNOWFLAKE_USER') }}"
  authenticator: externalbrowser
  warehouse: ML_WH
  database: ANALYTICS
  schema: GOLD
  connect_args:
    # cache the SSO token: compile and each dataset job open their own
    # connection, and without this every one prompts the browser again
    client_store_temporary_credential: true
```

Key-pair for CI and service users (new Snowflake accounts enforce MFA on password logins, so automation should use key-pair):

```yaml
config:
  account: "{{ env_var('SNOWFLAKE_ACCOUNT') }}"
  user: "{{ env_var('SNOWFLAKE_USER') }}"
  warehouse: ML_WH
  database: ANALYTICS
  schema: GOLD
  connect_args:
    private_key_file: "{{ env_var('SNOWFLAKE_PRIVATE_KEY_FILE') }}"
```

## Live integration tests

The unit tests run the adapter's generated SQL in DuckDB and need no account.
`tests/test_snowflake_live.py` additionally proves the dialect surfaces (`MD5_NUMBER_LOWER64` sampling, snapshot tokens, Arrow streaming, case rules) and the full local-training loop against a real Snowflake account.
It is double-gated: every test skips unless `MBT_LIVE_SNOWFLAKE=1`, and once opted in, incomplete configuration fails loudly instead of skipping.

```bash
export SNOWFLAKE_ACCOUNT=myorg-myaccount
export SNOWFLAKE_USER=me@example.com
export SNOWFLAKE_AUTHENTICATOR=externalbrowser   # or SNOWFLAKE_PASSWORD / SNOWFLAKE_PRIVATE_KEY_FILE
export SNOWFLAKE_WAREHOUSE=ML_WH
export SNOWFLAKE_DATABASE=ANALYTICS
export SNOWFLAKE_SCHEMA=SANDBOX                  # CREATE TABLE privilege needed here
export SNOWFLAKE_ROLE=ML_ROLE                    # optional

MBT_LIVE_SNOWFLAKE=1 uv run pytest -q -m live_snowflake
```

The suite creates uniquely named `MBT_LIVE_*` tables in that database.schema (a few hundred small rows), drops them at teardown, and touches nothing else.
With `externalbrowser` the whole run needs one browser prompt (the SSO token is cached).
In this repo the suite also runs nightly via `.github/workflows/live.yml` when `SNOWFLAKE_*` repository secrets are configured.
