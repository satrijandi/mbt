# churn_demo - the classification test fixture

A complete mbt project: four XGBoost models and three datasets over one committed source table, exercising most of the spec surface.
It lives under `tests/fixtures/` rather than `examples/` because that is what it is for - the golden-manifest test compiles it in the fast suite, and the E2E suite copies it to a tmp dir and drives it through the real CLI.
Everything in here is therefore guaranteed to work, which also makes it a useful thing to read.

If you want a starting point to copy, run `mbt init` instead (the scaffold is a simplified cousin of this project), or read `examples/showcase` for the platform-scale shape: one big lake table
driven through CI, scheduling, and monitoring on a docker stack.

## Run it

From the repo root (the fixture's `profiles.yml` lives in the project dir, all local, sqlite MLflow):

```bash
uv run mbt build --project-dir tests/fixtures/churn_demo
uv run mbt docs generate --project-dir tests/fixtures/churn_demo
uv run mbt promote --model churn_classifier --to production --project-dir tests/fixtures/churn_demo
uv run mbt score --project-dir tests/fixtures/churn_demo
uv run mbt monitor --project-dir tests/fixtures/churn_demo
```

The pre-deploy flow (ADR-30) trains `churn_classifier_oot` with April held back, then re-checks it once May and June have labels, and promotes only on a passing verdict:

```bash
uv run mbt build --select churn_classifier_oot --anchor 2026-04-30T00:00:00Z --project-dir tests/fixtures/churn_demo
uv run mbt evaluate --model churn_classifier_oot --out-of-time --gates --anchor 2026-06-30T00:00:00Z --project-dir tests/fixtures/churn_demo
uv run mbt promote --model churn_classifier_oot --to production --require-oot-check --project-dir tests/fixtures/churn_demo
```

With no `--version`, the check and the promotion both take the version the build just staged.
Both leave their report on that version's MLflow run: `report/` from the build, `evaluations/<run_id>/` from the check.

The committed parquet under `data/` is deterministic output of `scripts/generate_data.py`; regenerate only deliberately - data bytes enter snapshot hashes, so the golden manifests churn.

## What each piece demonstrates

- `datasets/churn_training_set.yml` - temporal split, label windows, built-in checks, and the **quarantined planted leak**: the generator writes a post-outcome `account_status` column that encodes the label exactly; the dataset declares a reviewed `label_leakage_scan` exclusion and every model excludes it from features. Strip the exclusion and the build blocks at exit 2 with two independent guards firing (the always-on scan and `tests/test_no_leakage.py`).
- `datasets/churn_oot_set.yml` - the same panel on absolute calendar windows (train January-February, test March) with an **after-test window**, `split.out_of_time: "2026-04-01:now"`: rows no training step sees, scored month by month against the test month (ADR-30). Only that window moves with `--anchor`.
- `datasets/upsell_training_set.yml` - a second dataset sharing the same source, so state-aware selection has a real subgraph to prune.
- `models/churn_classifier.yml` - the main spec: `{{ auto }}` scale_pos_weight with `calibration: isotonic` (the lever that fixes the miscalibration the rebalancing introduces: `ece` drops from ~0.21 raw to ~0.04 calibrated on this data, the before/after the e2e asserts), threshold + champion gates, a **fairness/disparity gate** (`across: plan_type`, `min_ratio: 0.6` - the weakest plan tier's `pr_auc` must stay within 60% of the strongest), calibration metrics (`brier`, `ece`), and the operating-point metric `threshold_at_precision_0.35` (the cutoff a 35%-precision retention campaign would deploy).
- `models/churn_classifier.py` - the hooks file: a custom metric computed from the prediction table.
- `models/churn_classifier_deep.yml` - a challenger variant of the same model, for champion/challenger flows.
- `models/churn_classifier_oot.yml` - the training report and its gates (ADR-30): an **after-test gate** (`source: out_of_time`, judged on the weakest month with at least 200 labelled rows, floor `oot_roc_auc_floor`), `evaluation.stability` (score and feature PSI of every after-test month against the test month), and a `report:` block asking for every binning strategy, row-level predictions, month, week-of-month, and day-of-month cells, and Evidently's drift pages (`engine: evidently`, from `mbt-evidently`). The report block is presentation only and does not enter the config hash.
- `models/upsell_classifier.yml` - a second target on its own dataset. The generated data also carries genuine noise features (`weekly_logins` numeric, `signup_channel` categorical) that every model must cope with; the categorical one exercises native categorical handling.
- `scoring/retention_scoring.yml` - a batch scoring pipeline (ADR-20/21): champion resolution by stage at run time, PSI/KS shift monitors against the training-time baseline, and delayed ground-truth evaluation via `mbt monitor`.
- `metrics.yml` / `exposures.yml` - shared metric declarations and downstream exposure lineage (they show up in `mbt docs generate`).
- `tests/test_no_leakage.py` - Python data tests, including the pin that keeps the planted leak leaky (the teaching asset cannot silently rot).
