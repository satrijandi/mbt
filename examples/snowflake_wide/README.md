# snowflake_wide - a multi-table Snowflake dataset

A minimal mbt project whose data source is **Snowflake**, building one training
dataset from **five tables** joined on `[customer_id, snapshot_date]`:

| Role | Source table | Join key |
|---|---|---|
| population (spine) | `customer_population` | `[customer_id, snapshot_date]` |
| label | `churn_labels` | `[customer_id, snapshot_date]` |
| feature | `demographic_features` | `[customer_id, snapshot_date]` |
| feature | `engagement_features` | `[customer_id, snapshot_date]` |
| feature | `billing_features` | `[customer_id, snapshot_date]` |

The population table is the spine (it defines which `(customer, snapshot)` rows
exist, ADR-22); the three feature tables `LEFT JOIN` onto it and the label
`JOIN`s inner. mbt compiles this into **one Snowflake query per split** - see
`datasets/wide_churn_training.yml`.

## Layout

```
mbt_project.yml                     project name + vars
profiles.yml                        snowflake data adapter (creds via env)
sources.yml                         the 5 tables, each with `identifier:`
datasets/wide_churn_training.yml    the multi-table join (inputs: population/label/features)
models/churn_wide.yml               an XGBoost model over the joined panel
show_wide_join.py                   see the join work WITHOUT a Snowflake account
```

## See the join work - no Snowflake account needed

The docker showcase tier can't host Snowflake (it's a cloud warehouse), so the
honest offline demonstration runs mbt's **real** Snowflake adapter over synthetic
tables, executing the generated Snowflake SQL in DuckDB (the same technique the
adapter's unit tests use):

```bash
uv run python examples/snowflake_wide/show_wide_join.py
```

It prints the generated `SELECT ... LEFT JOIN ... USING (...)` per split and the
joined training/test panels (join keys merged, label join columns projected
away).

Validate the project config (also no creds):

```bash
uv run mbt parse --project-dir examples/snowflake_wide
```

## Run it against real Snowflake

Set the connection env vars (never commit them), then build:

```bash
export SNOWFLAKE_ACCOUNT=... SNOWFLAKE_USER=... SNOWFLAKE_PASSWORD=...
export SNOWFLAKE_WAREHOUSE=... SNOWFLAKE_DATABASE=ANALYTICS SNOWFLAKE_SCHEMA=GOLD

uv run mbt build --project-dir examples/snowflake_wide
```

`build` materializes `wide_churn_training` (the five-table join runs in Snowflake),
trains `churn_wide`, evaluates its gate, and registers on pass. Adapt table
`identifier:`s in `sources.yml` and the join keys in the dataset spec to your
warehouse. See `packages/mbt-snowflake/README.md` for auth options (password,
SSO `externalbrowser`, key-pair) and the live test tier.
