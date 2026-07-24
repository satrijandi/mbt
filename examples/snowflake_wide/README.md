# snowflake_wide - train and score from huge Snowflake tables, as a data scientist

An mbt project whose data source is **Snowflake**, built around the shape most feature stores actually have: a population table, a label table, and several feature tables, all keyed on `[customer_id, snapshot_date]`.

**Column convention**: every table carries the same two join columns under the same names, `customer_id` + `snapshot_date`; the population table holds exactly those join keys and nothing else (it decides which rows exist, ADR-22).
A table's remaining columns are its payload; whether payload is used as a feature, used as the target, or ignored is declared in the specs, per the table below.

| Role | Source table | Join columns (same names everywhere) | Payload columns | Payload treatment |
|---|---|---|---|---|
| population (spine) | `customer_population` | `customer_id`, `snapshot_date` | none | n/a - join keys only |
| label | `churn_labels` | `customer_id`, `snapshot_date` | `is_churn` | training target, never a feature; joined inner |
| feature | `demographic_features` | `customer_id`, `snapshot_date` | `age`, `tenure_months` | features; LEFT JOINed |
| feature | `engagement_features` | `customer_id`, `snapshot_date` | `logins_30d`, `avg_session_min` | features; LEFT JOINed |
| feature | `billing_features` | `customer_id`, `snapshot_date` | `monthly_spend`, `plan_tier` | features; LEFT JOINed |

The join keys themselves are **ignored as features** - declared once in `models/churn_wide.yml` under `features.exclude` (that is also where bookkeeping columns of real tables go: load timestamps, batch ids, and the like).
Join columns are declared per table via `using:` in the dataset and scoring specs; the label's join columns are projected away after the join, so nothing is duplicated.

mbt compiles the five-table join into **one Snowflake query per split**, so the join, the temporal windows, the filters, and the sampling all run in the warehouse.
Only the resulting training panel streams back (Arrow batches into parquet); the huge tables never leave Snowflake.

The same shape serves both halves of the lifecycle:

- `datasets/wide_churn_training.yml` joins all five tables into a labeled training panel.
- `scoring/wide_churn_scoring.yml` joins the spine and the three feature tables (no label) into the newest month's scoring batch.

## Layout

```
mbt_project.yml                     project name + vars
profiles.yml                        dev (SSO browser auth, 5% sample) and prod (key-pair, full data) targets
sources.yml                         the 5 tables, each with `identifier:`
datasets/wide_churn_training.yml    the multi-table training join (population/label/features)
models/churn_wide.yml               an XGBoost model over the joined panel
scoring/wide_churn_scoring.yml      batch scoring + drift monitors + delayed ground truth
seed_demo_tables.py                 create the 5 demo tables in YOUR warehouse, server-side
show_wide_join.py                   see the join work WITHOUT a Snowflake account
```

## 1. See the join work - no Snowflake account needed

The honest offline demonstration runs mbt's **real** Snowflake adapter over synthetic tables, executing the generated Snowflake SQL in DuckDB (the same technique the adapter's unit tests use):

```bash
uv run python examples/snowflake_wide/show_wide_join.py
uv run mbt parse --project-dir examples/snowflake_wide
```

The first prints the generated `SELECT ... LEFT JOIN ... USING (...)` per split and the joined panels; the second validates the whole project config.
Neither needs credentials.

## 2. Connect with browser SSO (JumpCloud, Okta, Entra, ...)

The `dev` target authenticates with `authenticator: externalbrowser`: mbt opens your default browser at whatever IdP fronts your Snowflake account (JumpCloud here), you log in as yourself, and the connector receives the token on a localhost callback.
No password ever touches a file or an environment variable.

Install the SSO extra once so the token is cached in your OS keyring - without it, every mbt job process would pop its own browser window:

```bash
uv pip install 'mbt-snowflake[sso]'
```

Then describe your account in environment variables (never in YAML - `profiles.yml` is committed and secret-free, it reads these via `env_var()`):

```bash
cp packages/mbt-snowflake/.env.example .env    # edit: account, user, warehouse, database, schema
set -a; source .env; set +a
```

Use a **scratch schema** you can create tables in (e.g. `SNOWFLAKE_SCHEMA=SANDBOX`), not a shared gold one.
The whole session needs **one** browser prompt; compile and every dataset/scoring job reuse the cached token.
In containers or WSL, where the localhost callback can hang, see the connector's `SNOWFLAKE_AUTH_SOCKET_REUSE_PORT` docs (pointer in `packages/mbt-snowflake/README.md`).

## 3. Seed the five demo tables (or point at your real ones)

`seed_demo_tables.py` creates the five tables **entirely server-side** with `GENERATOR()` SQL: nothing is uploaded, so the size knob costs warehouse-seconds, not bandwidth.
The data is deterministic and learnable, and the shapes are honest about how real gold tables look:

- The spine is genuinely selective: customers activate in different months, so the population grows month over month.
- The feature tables cover a **strict superset** of the population/label universe: extra customers, an extra month, and (for `engagement_features`) a mid-month snapshot cadence of its own.
  None of those rows ever enter a training set or scoring batch; the spine join drops them in-warehouse.

```bash
uv run python examples/snowflake_wide/seed_demo_tables.py --dry-run   # inspect the SQL first
uv run python examples/snowflake_wide/seed_demo_tables.py             # 50k customers x 5 months
uv run python examples/snowflake_wide/seed_demo_tables.py --customers 2000000 --force
```

It refuses to replace existing tables unless `--force` is given, and `--drop` cleans up.

**Using your own tables instead**: edit the `identifier:`s in `sources.yml` (they may be fully qualified, `DB.SCHEMA.TABLE`), set the join keys in `datasets/wide_churn_training.yml` to your entity + snapshot columns, and swap the absolute demo split windows for relative ones (e.g. `train: "-395d:-62d"`, `test: "-62d:now"`) so every compile re-anchors them.
Everything below works unchanged.

**Read-only sources, writable sandbox** - the usual enterprise grant layout works as-is.
mbt never writes to the schema your source tables live in: training materializes to local parquet and predictions stage to local disk; only this seed script and the live test suite create Snowflake tables, and both target `SNOWFLAKE_DATABASE.SNOWFLAKE_SCHEMA`.
So point those env vars at your personal sandbox (e.g. `ANALYTICS_SANDBOX.SANDBOX_ME`) and fully qualify each source, e.g. `identifier: GOLD.DS.LABEL_CHURN_MONTHLY` - a qualified identifier overrides the profile default, and plain `SELECT` grants on the source schema are all mbt needs (snapshot pinning via `SYSTEM$LAST_CHANGE_COMMIT_TIME` included).

## 4. Train on a 5% sample - the inner loop

```bash
uv run mbt build --project-dir examples/snowflake_wide
```

The default `dev` target has `sample_fraction: 0.05` with `sample_key: [customer_id]`: mbt pushes `MOD(MD5_NUMBER_LOWER64(customer_id), 1e6) < 50000` into the warehouse query, so 5% of **customers** (whole histories, never split mid-customer) come back and train an XGBoost locally.
The build materializes the join, trains, evaluates the PR-AUC gate, and registers the model to `staging` in the local MLflow registry (`mlflow.db`).

Sampling is deterministic and monotone: the same fraction always selects the same customers, and a 5% sample is a subset of the 20% one, so scaling up refines rather than reshuffles your view of the data.

```bash
uv run mbt build --project-dir examples/snowflake_wide --vars 'sample_fraction: 0.2'   # widen the ladder
uv run mbt build --project-dir examples/snowflake_wide --vars 'sample_fraction: 1.0'   # full data, still via SSO
```

The `prod` target is the automation face of the same project: key-pair auth (`SNOWFLAKE_PRIVATE_KEY_FILE`) and `sample_fraction: 1.0`, selected with `--target prod` from CI or a scheduler.

## 5. Score the newest cohort, monitor when labels arrive

The scoring node reads the same spine + feature tables (no label) for exactly the newest month-start cohort - the window `"-31d:now"` resolves against the run's `--anchor`, so re-scoring is snapshot-driven, never clock-driven:

```bash
uv run mbt score   --project-dir examples/snowflake_wide --anchor 2026-06-01T00:00:00Z
uv run mbt predictions ls --project-dir examples/snowflake_wide
```

Scoring resolves the `staging` champion from the registry, checks the input, writes predictions, and evaluates the feature/prediction drift monitors against the champion's training-time baselines.
Predictions are staged as parquet under the profile's `predictions_root` (`./predictions`, ADR-23 v1); the warehouse-native store that writes them back into Snowflake tables is designed in ADR-23 and lands as v2.

Churn labels mature ~30 days after a snapshot, so the ground-truth evaluation runs later, once `scored_at + maturity` passes the monitor's anchor:

```bash
uv run mbt monitor --project-dir examples/snowflake_wide --anchor 2026-07-15T00:00:00Z
```

This joins the arrived labels onto the staged predictions, computes realized PR-AUC/ROC-AUC, and enforces the realized-metric gate (exit code 2 if the deployed model has degraded below the floor - the same code a failing training gate uses).

## 6. Reproduce it exactly

Compile pins a snapshot token per source table (`SYSTEM$LAST_CHANGE_COMMIT_TIME`, a cheap metadata call; `--deep-snapshot` upgrades to `HASH_AGG` content fingerprints):

```bash
uv run mbt compile --project-dir examples/snowflake_wide
uv run mbt run     --project-dir examples/snowflake_wide --manifest target/manifest.json
```

The manifest run verifies every pin against the live tables (a changed table fails loudly instead of training on silently different data) and reproduces the metrics bit-for-bit.
`mbt state diff --state <old-manifest>` tells you which datasets went stale after upstream loads.

## Huge and wide: what pushes down, what to align upstream

The tables this example models are the awkward real ones: millions of rows, one table far bigger than the rest, feature tables that serve the whole company (rows for customers and snapshot dates your model never asked for), and potentially thousands of columns.
What mbt guarantees, and where the DE seam is:

- **Row volume never transits**: the five-table join, the temporal windows, the filters, and the entity sampling are one SQL statement per split executed in Snowflake; only the resulting panel streams back as Arrow batches. It does not matter that one feature table is 100x the size of the others - Snowflake's optimizer sees the whole join.
- **Feature tables may be supersets**: rows outside the population spine (other customers, other dates, other populations entirely) are dropped by the join, in-warehouse, by construction. You never pre-filter a shared gold table just to train on it.
- **Snapshot cadence is an upstream contract**: mbt joins on exact equality of the declared per-table `using:` keys. If a feature table snapshots daily or mid-month and your spine is month-start, the month-start rows join and the rest stay behind (as `engagement_features` demonstrates). If you need as-of semantics ("latest snapshot at or before the spine date"), align it in a DE-owned view (`QUALIFY ROW_NUMBER() ... = 1`) and point `sources.yml` at the view - the training-set contract stays in reviewed SQL instead of ad-hoc notebook joins.
- **Thousands of columns**: the dataset materializes whatever the join produces, as columnar parquet. The model spec then prunes with `features.include`/`exclude` globs (plus an optional `hooks.py` transform for programmatic selection) at training time. For extreme width, put a curated projection view in front of the widest table so the materialized panel carries only plausible features - width, unlike rows, does flow into the materialization, and a reviewed projection is also better MLOps than a 3000-column `include: ["*"]`.
- **The inner loop is the sample ladder, not a smaller cluster**: `sample_fraction` at 0.05 with `sample_key: [customer_id]` gives you a coherent 5% panel of whole customers in seconds, deterministic and monotone, straight from the full-size tables.

## Why this shape - DE and MLOps notes

- **The population spine is a contract** (ADR-22): data engineering owns "who is in the book this month"; the training set and the scoring batch both derive from it, so trained-on and scored populations cannot drift apart by construction.
- **Labels live in their own table**, keyed like everything else. The label for `(customer, 2026-04-01)` is an outcome observed later; keeping it separate from features makes point-in-time correctness auditable and the join leak-free (the label's join columns are projected away).
- **Temporal split, not random**: the model is evaluated on months it never saw, which is how it will be used. The seed ladder (`spec.seed` and friends, ADR-18) keeps every stochastic stage reproducible.
- **Entity-level sampling**: `sample_key: [customer_id]` samples whole customers, so no customer contributes rows to both a dev sample's train and test months by luck of the draw.
- **Push-down everywhere**: joins, windows, filters, and sampling execute in Snowflake; the laptop sees only the panel it trains on. That is the difference between "works on the 10k-row demo" and "works on the 2B-row gold table".
- **Secrets never in the repo**: `profiles.yml` is committed and reads `env_var()`; humans use SSO (no long-lived secret exists at all), automation uses key-pair.
- **Gates before registry, monitors after**: a model must clear its metric floor to be registered; once serving, drift monitors run on every batch and realized metrics run when labels mature - the whole loop exits nonzero for schedulers.

## Testing tiers

| Tier | What it proves | Needs |
|---|---|---|
| `show_wide_join.py` + `mbt parse` | join SQL + project config | nothing |
| fast suite (`uv run pytest -q -m "not e2e"`) | the committed example builds through the real adapter (SQL run in DuckDB) and parses to the advertised DAG | nothing |
| live suite (`MBT_LIVE_SNOWFLAKE=1 uv run pytest -q -m live_snowflake`) | dialect surfaces, sampling reproducibility, snapshot pins, this example's five-table join, and the full laptop training loop on a **real** account | `SNOWFLAKE_*` env vars; one SSO prompt |

The live suite creates its own uniquely named `MBT_LIVE_*` tables and drops them afterwards; it never touches the tables you seeded.
See `packages/mbt-snowflake/README.md` for the full live-tier setup.
