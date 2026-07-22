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
- `profiles.yml` - environments (dev/prod); keep secrets in `{{ env_var(...) }}`
- `requirements.in` / `requirements.txt` - pinned toolchain for CI; regenerate
  with hashes via `uv pip compile --generate-hashes requirements.in -o requirements.txt`.
  The pins reference the mbt release tag `v__MBT_VERSION__`; until that tag
  exists on the mbt repo, `pip install -r requirements.txt` fails with
  `git checkout -q v__MBT_VERSION__ did not run successfully` - the
  requirements.txt header documents the pin-a-commit workaround
- `.github/workflows/` - PR check, prod build, promotion, weekly + monthly
  retrain, daily scoring, weekly ground-truth monitor
- `scripts/publish_state.sh` / `fetch_state.sh` - the durable prod-state
  baseline: prod builds append the manifest to the `mbt-state` branch,
  PR checks fetch it for `--state` (first run bootstraps with a full build)

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
