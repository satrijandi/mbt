# revenue_demo - the regression test fixture

An XGBoost **regressor** that forecasts each subscriber's next-30-day spend over one committed dataset.
It is the `task: regression` twin of `churn_demo` (which is binary classification), so a data scientist starting a regression model has a working template to read - metrics, gates, slices, and delayed ground-truth monitoring all wired for a continuous target.
It lives under `tests/fixtures/` because a parse guard in the fast suite and an end-to-end build in the E2E suite run it on every CI run, so everything in here is guaranteed to work.

## Run it

From the repo root (the fixture's `profiles.yml` lives in the project dir, all local, sqlite MLflow):

```bash
uv run mbt build --project-dir tests/fixtures/revenue_demo
uv run mbt promote --model spend_regressor --to production --project-dir tests/fixtures/revenue_demo
uv run mbt score --project-dir tests/fixtures/revenue_demo
uv run mbt monitor --project-dir tests/fixtures/revenue_demo
```

The committed parquet under `data/` is deterministic output of `scripts/generate_data.py`; regenerate only deliberately - data bytes enter snapshot hashes.

## What regression changes vs. the churn demo

- `datasets/spend_training_set.yml` - the `label` is the continuous `spend_next_30d` column; the temporal split, `not_null`, and `no_future_columns` checks are identical to the classification demo. The always-on leakage scan runs here too (it bins the continuous target), so a leaked feature is still caught.
- `models/spend_regressor.yml` - `task: regression`. The evaluation metrics are the regression set (`rmse`, `mae`, `r2`) instead of `roc_auc`/`pr_auc`. The gate is on **`rmse`, which is lower-is-better**, so its `threshold` is a **ceiling** (`actual <= threshold`) - the mirror image of the churn demo's `pr_auc` floor. `plan_type` is still a slice, so per-segment error is reported.
- `scoring/spend_scoring.yml` - the same batch-scoring shape (ADR-20/21) with champion-by-stage resolution and PSI/KS shift monitors, but the delayed `ground_truth` block joins realized spend and computes realized `rmse`/`mae`/`r2`, gating on the same rmse ceiling.
- `models/spend_regressor.yml` also carries the **feature-treatment block** (ADR-27), because this is the fixture where every keyword is proven end to end: `tenure_days` is capped, log-compressed and monotone-constrained; `weekly_logins` is percentile-ranked within each batch; `support_tickets` carries the standalone monotone spelling; and `signup_channel` is a declared categorical (so the list is authoritative - any other string feature would be a hard error). The E2E test reads the exported monitoring baseline back and asserts the treatment is visible in it.
- `scripts/generate_data.py` - the generative target is dominated by **numeric** drivers (usage, tenure, support load) plus a smaller categorical plan premium and Gaussian noise, so a real model reaches rmse ~8 while a mean predictor scores ~35 - the `rmse_ceiling: 12` gate is comfortably meaningful. `weekly_logins`/`signup_channel` are genuine noise features the model must ignore.

## Why the cap costs accuracy here, on purpose

`tenure_days` runs to 1000 days and the generator's spend signal stays **linear** the whole way, so `cap: 730` genuinely throws information away: test rmse moves from ~7.6 to ~8.1 against the 12.0 ceiling.
That is deliberate. Plateau capping is a trade - stability in exchange for tail signal - and a fixture where the cap were free would teach the wrong thing.
The number to watch is that the gate still passes with room, which is what makes the trade acceptable rather than merely survivable.

## The gate direction, concretely

`rmse` is registered as lower-is-better, so mbt reads a `threshold` gate on it as a ceiling.
Set `rmse_ceiling` in `mbt_project.yml` below the model's achievable error (e.g. `6.0`) and `mbt build` exits 2 with a gate breach; set it above (the shipped `12.0`) and the build passes.
This is the same gate engine as the classification demo - only the metric's direction flips what the threshold means.
