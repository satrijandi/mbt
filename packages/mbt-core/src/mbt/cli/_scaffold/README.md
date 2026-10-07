# __PROJECT_NAME__

A machine-learning model project built with [mbt](https://github.com/satrijandi/mbt) -
declarative Model-as-Code: the model config IS the model.

## Quickstart

```bash
# 1. generate sample data (replace with your own parquet later)
python scripts/generate_sample_data.py

# 2. validate specs and build the DAG (fast, no execution)
mbt parse

# 3. train + test + register everything
mbt build

# 4. inspect
mbt ls
mbt show churn_classifier
```

## Layout

- `sources.yml` - external inputs (parquet paths in dev)
- `datasets/` - declarative training-set construction (label, filters, split)
- `models/` - the model specs: task, adapter, hyperparameters, gates, registration
- `scoring/` - batch scoring (serving) pipelines: champion + input + prediction
  sink + shift monitors + delayed ground-truth evaluation (1 config = 1 pipeline)
- `tests/` - Python data tests (`def test_*(dataset, spec) -> TestResult`)
- `macros/` - Jinja macros usable in any spec
- `profiles.yml` - environments (dev/prod), and it IS committed: CI has no
  `~/.mbt` to read, so a gitignored copy leaves every workflow after `mbt parse`
  failing with "no profiles.yml found". Secrets go through
  `{{ env_var(...) }}`, which keeps the value out of the file and redacts it
  everywhere it could be printed; non-secret environment values (roots, hosts,
  schema names) go through `{{ env(...) }}` so they stay readable in logs.
  Never write a secret value into this file
- `requirements.in` / `requirements.txt` - the CI install set: the four mbt
  packages (mbt-adapter-base, mbt-core, mbt-xgboost, mbt-mlflow) at
  `__MBT_REF__` (the release tag of the mbt that scaffolded this project, or
  the exact commit when that mbt was a development build), plus the numerics
  stack (numpy, scipy, pandas, pyarrow, scikit-learn, duckdb, xgboost, mlflow)
  at the exact versions the mbt install that scaffolded this project was
  running - pinning mbt pins none of them, and they are what decides model
  numerics. Their transitive dependencies still float and nothing is
  hash-verified; a real lock
  (`uv pip compile --generate-hashes requirements.in -o requirements.txt`)
  needs mbt on PyPI, because a git ref carries no wheel hash to record
__FORGE_CI_LAYOUT__
- `scripts/publish_state.sh` / `fetch_state.sh` - the durable prod-state
  baseline: prod builds append the manifest to the `mbt-state` branch,
  PR checks fetch it for `--state` (first run bootstraps with a full build)
- `CODEOWNERS` - who reviews model and dataset specs, and `promotions.yml`

__FORGE_REPO_SETTINGS__

## Day-to-day

```bash
mbt build --select churn_classifier          # one model (+ its data)
bash scripts/fetch_state.sh                  # baseline -> state/prod/latest.json
# --deep-snapshot: the published baseline uses content-hash tokens (see the
# workflows); mixing schemes would flag every dataset as modified
mbt build --deep-snapshot --select state:modified+ --state state/prod/latest.json
mbt test                                     # data tests + gates, no retraining
mbt docs generate && mbt docs serve          # model cards + lineage
mbt promote --model churn_classifier --to production
mbt score                                    # batch-score with the production champion
mbt monitor                                  # realized metrics for matured predictions
```

## Serving and monitoring

`scoring/churn_scoring.yml` is the whole serving pipeline: which champion to
load (resolved from the registry at run time, so promotions take effect on
the next run), what to score, where predictions land, and what to watch.
Every scoring run checks the input, compares feature and score distributions
against the champion's training-time baseline ("shift"), and fails with exit
code 2 on a breach - same semantics as gates. `mbt monitor` runs on its own
schedule and evaluates realized metrics once outcomes mature, gating on the
declared thresholds.
