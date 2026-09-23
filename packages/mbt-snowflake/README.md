# mbt-snowflake

The Snowflake data adapter for [mbt](https://github.com/satrijandi/mbt), the declarative build tool for machine learning models.
Point a dataset at a Snowflake table, view, or dynamic table, and mbt builds reproducible, pinned training sets from it: filters, sampling, and split assignment push down as SQL, and rows stream back as Arrow batches into the standard local materialization that training jobs read.
Training jobs therefore never need warehouse credentials.

```bash
pip install mbt-snowflake          # plus mbt-core and a training adapter
pip install 'mbt-snowflake[sso]'   # adds keyring-backed SSO token caching
```

mbt is not on PyPI yet; see [Installation](https://satrijandi.github.io/mbt/installation/) for installing from a release tag.

## Configure a target

```yaml
# profiles.yml
my_project:
  target: prod
  outputs:
    prod:
      data:
        adapter: snowflake
        config:
          # env() for identifiers (they belong in logs), env_var() for the secret
          account: "{{ env('SNOWFLAKE_ACCOUNT') }}"
          user: "{{ env('SNOWFLAKE_USER') }}"
          password: "{{ env_var('SNOWFLAKE_PASSWORD') }}"
          warehouse: ML_WH
          database: ANALYTICS
          schema: GOLD
          role: ML_ROLE                          # optional
          predictions_root: /mnt/mbt-stage       # where mbt score stages predictions
      tracking: {adapter: mlflow, config: {uri: "{{ env('MLFLOW_TRACKING_URI') }}"}}
      registry: {adapter: mlflow, config: {uri: "{{ env('MLFLOW_TRACKING_URI') }}"}}
      compute: {adapter: local}
      artifact_store: s3://my-bucket/mbt/artifacts
```

| Key | Meaning |
|---|---|
| `account`, `user`, `password`, `warehouse`, `database`, `schema`, `role`, `authenticator` | Passed straight to `snowflake.connector.connect()` |
| `connect_args` | Any other documented connector parameter, such as `private_key_file` |
| `normalize_case` | Default `true`: unquoted Snowflake identifiers come back UPPERCASE, and the adapter lowercases result columns to match spec conventions |
| `predictions_root` | Where prediction runs are staged as Parquet. Default `<tmpdir>/mbt-predictions`, which is cleared on reboot - set a durable path in production |

## Declare the panel

A dataset reads **one relation** ([ADR-29](https://satrijandi.github.io/mbt/adr/0029-single-relation-datasets/)).
Whatever joins the population to its labels and feature histories - a dbt model, a dynamic table - lives upstream, and mbt reads its result.

```yaml
# sources.yml - warehouse relations use identifier:, not path:
sources:
  - name: snowflake
    tables:
      - name: churn_panel
        identifier: GOLD.ML_CHURN_PANEL        # database defaults from the config

# datasets/churn_training_set.yml
datasets:
  - name: churn_training_set
    source: source('snowflake', 'churn_panel')
    columns: [customer_id, snapshot_date, churned_90d, monthly_usage, tenure_days]
    sample_key: [customer_id]                  # deterministic push-down sampling
    label:
      column: churned_90d
      horizon: "90d"
    split:
      strategy: temporal
      time_column: snapshot_date
      train: "-180d:-28d"
      test: "-28d:now"
      embargo: "90d"
```

`columns:` is the panel contract: when the relation is owned by another repository, it is what makes a new feature column a reviewed change in this one.

## Guarantees

- **Snapshots.**
  `mbt compile` pins each relation with `SYSTEM$LAST_CHANGE_COMMIT_TIME` (a cheap metadata call) folded together with a metadata-only column fingerprint, so re-deploying a view or dynamic table with an extra column marks the dataset `state:modified` even though no DML happened.
  `--deep-snapshot` switches to `HASH_AGG(*)` content fingerprints.
  A pinned manifest whose relation has since changed fails the build rather than training on different data.
- **Reproducible sampling.**
  `sample_fraction` keeps rows where `MOD(MD5_NUMBER_LOWER64(<sample_key>), 1000000) < fraction * 1000000`, pushed into the warehouse query, so a 1% development sample of a 7M-row table never leaves Snowflake.
  The same fraction selects the same rows, smaller fractions are subsets of larger ones, and the digest matches the local and Spark adapters, so a model validated on a laptop trains on the same partition in the warehouse.
- **Streaming.**
  Rows stream through the connector's Arrow batch API into one Parquet file per split; nothing is held fully in memory on the mbt side.

For a change token on a view, change tracking must be enabled on the tables it reads; otherwise `mbt compile` fails with `could not read a snapshot token` - see the [troubleshooting runbook](https://satrijandi.github.io/mbt/troubleshooting/).

## Batch scoring

`mbt score` and `mbt monitor` run against a Snowflake target.
The unlabeled scoring batch is read straight from the warehouse, with filters, the `score` window, and `sample_fraction` pushed down exactly as in training.
Prediction runs are staged as Parquet under `predictions_root` joined with the scoring node's `output.path`, using mbt's shared prediction-store layout: per-run directories, idempotent writes per `run_key`, and the ground-truth ledger.

A store that writes predictions back into Snowflake tables is designed in [ADR-23](https://satrijandi.github.io/mbt/adr/0023-warehouse-batch-scoring/) and waits on live verification of the serving leg (`promote`, `score`, `monitor` against a real account), tracked in [issue #1](https://github.com/satrijandi/mbt/issues/1).
Until then predictions land in the staging path, not in a Snowflake table.

## Authentication

**Browser SSO**, the usual data-scientist setup, with no password anywhere.
The browser lands on whatever identity provider your account federates to (Okta, Entra ID, JumpCloud, ...); nothing changes on the mbt side.

```yaml
config:
  account: "{{ env('SNOWFLAKE_ACCOUNT') }}"
  user: "{{ env('SNOWFLAKE_USER') }}"
  authenticator: externalbrowser
  warehouse: ML_WH
  database: ANALYTICS
  schema: GOLD
```

With `authenticator: externalbrowser` the adapter defaults `client_store_temporary_credential` to `true`, because a compile and each job open their own connection and would otherwise each prompt again.
Set it under `connect_args` to override (for example `false` on a shared machine).
Caching needs the connector's keyring backend - install `mbt-snowflake[sso]` - and the account parameter `ALLOW_ID_TOKEN`; without either, each process prompts.
In containers or WSL, where the `externalbrowser` localhost callback can hang, the connector offers `SNOWFLAKE_AUTH_SOCKET_REUSE_PORT=true` with a fixed `SF_AUTH_SOCKET_PORT`.

**Key-pair** for CI and service users; new Snowflake accounts enforce MFA on password logins, so automation should use it.

```yaml
config:
  account: "{{ env('SNOWFLAKE_ACCOUNT') }}"
  user: "{{ env('SNOWFLAKE_USER') }}"
  warehouse: ML_WH
  database: ANALYTICS
  schema: GOLD
  connect_args:
    private_key_file: "{{ env('SNOWFLAKE_PRIVATE_KEY_FILE') }}"
```

Verified against `snowflake-connector-python` 4.7.1.

## Tests

The unit tests execute the adapter's generated SQL in DuckDB, with shim macros for Snowflake-only functions, and need no account.

`tests/test_snowflake_live.py` additionally proves on a real account what a stand-in cannot: the `MD5_NUMBER_LOWER64` digest, snapshot tokens on tables, views, and dynamic tables, Arrow streaming, identifier case rules, the [showcase](https://github.com/satrijandi/mbt/tree/main/examples/showcase)'s committed dataset spec building from a warehouse copy of its lake table, and a full `mbt build` then `mbt run --manifest` loop from a laptop.
It is double-gated: every test skips unless `MBT_LIVE_SNOWFLAKE=1`, and once opted in, incomplete configuration fails loudly instead of skipping.

Credentials live in environment variables, never in `profiles.yml`.
Copy [`.env.example`](https://github.com/satrijandi/mbt/blob/main/packages/mbt-snowflake/.env.example) to `.env` (the repository ignores `.env` files), fill it in, and load it:

```bash
cp packages/mbt-snowflake/.env.example packages/mbt-snowflake/.env   # then edit
set -a; source packages/mbt-snowflake/.env; set +a
MBT_LIVE_SNOWFLAKE=1 uv run pytest -q -m live_snowflake
```

Required: `SNOWFLAKE_ACCOUNT`, `SNOWFLAKE_USER`, `SNOWFLAKE_WAREHOUSE`, `SNOWFLAKE_DATABASE`, `SNOWFLAKE_SCHEMA`, and exactly one of `SNOWFLAKE_AUTHENTICATOR=externalbrowser`, `SNOWFLAKE_PASSWORD`, or `SNOWFLAKE_PRIVATE_KEY_FILE` (with `SNOWFLAKE_PRIVATE_KEY_FILE_PWD` if the key is encrypted); optionally `SNOWFLAKE_ROLE`.
The `MBT_LIVE_SNOWFLAKE=1` switch stays out of `.env` on purpose, so credentials in a shell are never enough to trigger warehouse traffic or an SSO prompt.

The suite creates uniquely named `MBT_LIVE_*` tables and views in that schema, drops them at teardown, and touches nothing else.
It needs `CREATE TABLE` and `CREATE VIEW`; `CREATE DYNAMIC TABLE` is optional, and its one test skips without it.
In this repository the suite also runs nightly through `.github/workflows/live.yml` when the `SNOWFLAKE_*` secrets are configured.

## License

Apache License 2.0.
