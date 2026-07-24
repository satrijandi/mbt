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

For the full data-scientist walkthrough - browser SSO, a server-side seed
script for demo tables, sampled training on huge tables, batch scoring, and
delayed ground-truth monitoring - see
[`examples/snowflake_wide`](../../examples/snowflake_wide/README.md).

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
- **Per-table column projection** (ADR-25): a feature entry's `columns:`
  (keep-list) or `exclude:` (drop-list) becomes a projecting subquery inside
  the generated join, so pruned columns of a wide gold table are never
  scanned or transferred - declare the handful of columns a model needs
  instead of shipping a 3000-column `SELECT *`.
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

## Batch scoring (ADR-23)

`mbt score` and `mbt monitor` run against a Snowflake target: the unlabeled
scoring input is read straight from the warehouse (filters, the `score` window,
and `sample_fraction` push down, exactly like training), and predictions are
staged as parquet under `predictions_root` (adapter config, default the project
dir) joined with the scoring node's `output.path`.

Staging reuses mbt's shared prediction-store layout (per-run directories,
idempotent-by-`run_key` writes, the ground-truth ledger). A warehouse-native
store that writes predictions back into Snowflake tables is designed in ADR-23
and gated on the first credentialed `live_snowflake` run - until then predictions
land in the staging path, not a Snowflake table.

```yaml
# profiles.yml (data adapter)
config:
  adapter: snowflake
  database: ANALYTICS
  schema: GOLD
  predictions_root: /mnt/mbt-stage   # where scoring predictions are staged
```

## Authentication

The config keys `account`, `user`, `password`, `warehouse`, `database`, `schema`, `role`, and `authenticator` pass straight to `snowflake.connector.connect()`.
Every other documented connector parameter works under `connect_args`.

SSO from a laptop (the usual data-scientist setup - no password anywhere).
The browser lands on whatever IdP your Snowflake account federates to (JumpCloud, Okta, Entra ID, ...); that mapping lives in Snowflake's SAML/OIDC config, and nothing changes on the mbt side:

```yaml
config:
  account: "{{ env_var('SNOWFLAKE_ACCOUNT') }}"
  user: "{{ env_var('SNOWFLAKE_USER') }}"
  authenticator: externalbrowser
  warehouse: ML_WH
  database: ANALYTICS
  schema: GOLD
```

With `authenticator: externalbrowser` the adapter defaults
`client_store_temporary_credential` to `true`: compile and each dataset job
open their own connection, and without the cached SSO token every one of
them would prompt the browser again.
Set it explicitly under `connect_args` to override (e.g. `false` on a shared
machine).

Persisting that token cache requires the connector's keyring backend - install the extra:

```bash
pip install 'mbt-snowflake[sso]'    # = snowflake-connector-python[secure-local-storage]
```

Without it the connector silently skips caching and `externalbrowser` re-prompts per connection.
Verified against `snowflake-connector-python` 4.7.1 (current stable; the `>=3.7` floor still holds - both the caching parameter and the `secure-local-storage` extra predate it).
In containers/WSL, where the localhost callback of `externalbrowser` can hang, the connector offers `SNOWFLAKE_AUTH_SOCKET_REUSE_PORT=true` with a fixed `SF_AUTH_SOCKET_PORT` (see its docs).

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
`tests/test_snowflake_live.py` additionally proves the dialect surfaces (`MD5_NUMBER_LOWER64` sampling, snapshot tokens, Arrow streaming, case rules), the wide multi-table join from `examples/snowflake_wide` (a population spine, a label table, and three feature tables joined on `[customer_id, snapshot_date]`), and the full local-training loop against a real Snowflake account.
It is double-gated: every test skips unless `MBT_LIVE_SNOWFLAKE=1`, and once opted in, incomplete configuration fails loudly instead of skipping.

Credentials live in environment variables, never in `profiles.yml` (it is committed and secret-free; values flow in through `env_var()`).
Copy [`.env.example`](.env.example) to `.env` next to it (gitignored - the repo ignores `.env` and `.env.*` everywhere) and load it, or export the variables directly:

```bash
cp packages/mbt-snowflake/.env.example packages/mbt-snowflake/.env   # then edit
set -a; source packages/mbt-snowflake/.env; set +a

MBT_LIVE_SNOWFLAKE=1 uv run pytest -q -m live_snowflake
```

The variables: `SNOWFLAKE_ACCOUNT`, `SNOWFLAKE_USER`, `SNOWFLAKE_WAREHOUSE`, `SNOWFLAKE_DATABASE`, `SNOWFLAKE_SCHEMA` (required), exactly one of `SNOWFLAKE_AUTHENTICATOR=externalbrowser` / `SNOWFLAKE_PASSWORD` / `SNOWFLAKE_PRIVATE_KEY_FILE` (+`_PWD`), and optionally `SNOWFLAKE_ROLE`.
The `MBT_LIVE_SNOWFLAKE=1` gate stays out of `.env` on purpose: credentials sitting in your shell must never be enough to trigger warehouse traffic or an SSO popup by themselves.

The suite creates uniquely named `MBT_LIVE_*` tables in that database.schema (a few hundred small rows), drops them at teardown, and touches nothing else.
With `externalbrowser` the whole run needs one browser prompt (the SSO token is cached).
In this repo the suite also runs nightly via `.github/workflows/live.yml` when `SNOWFLAKE_*` repository secrets are configured.
