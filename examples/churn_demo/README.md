# churn_demo

The reference mbt project: three XGBoost models over one committed dataset, exercising most of the spec surface.
The golden-manifest and end-to-end tests build it on every CI run, so everything in here is guaranteed to work.
If you want a gentler, minimal starting point, run `mbt init` instead - the scaffold is a simplified cousin of this project.

## Run it

From the repo root (the demo's `profiles.yml` lives in the project dir, all local, sqlite MLflow):

```bash
uv run mbt build --project-dir examples/churn_demo
uv run mbt docs generate --project-dir examples/churn_demo
uv run mbt promote --model churn_classifier --to production --project-dir examples/churn_demo
uv run mbt score --project-dir examples/churn_demo
uv run mbt monitor --project-dir examples/churn_demo
```

The committed parquet under `data/` is deterministic output of `scripts/generate_data.py`; regenerate only deliberately - data bytes enter snapshot hashes, so the golden manifests churn.

## What each piece demonstrates

- `datasets/churn_training_set.yml` - temporal split, label windows, built-in checks, and the **quarantined planted leak**: the generator writes a post-outcome `account_status` column that encodes the label exactly; the dataset declares a reviewed `label_leakage_scan` exclusion and every model excludes it from features. Strip the exclusion and the build blocks at exit 2 with two independent guards firing (the always-on scan and `tests/test_no_leakage.py`).
- `datasets/upsell_training_set.yml` - a second dataset sharing the same source, so state-aware selection has a real subgraph to prune.
- `models/churn_classifier.yml` - the main spec: `{{ auto }}` scale_pos_weight, threshold + champion gates, calibration metrics (`brier`, `ece`), and the operating-point metric `threshold_at_precision_0.35` (the cutoff a 35%-precision retention campaign would deploy).
- `models/churn_classifier.py` - the hooks file: a custom metric computed from the prediction table.
- `models/churn_classifier_deep.yml` - a challenger variant of the same model, for champion/challenger flows.
- `models/upsell_classifier.yml` - a second target on its own dataset. The generated data also carries genuine noise features (`weekly_logins` numeric, `signup_channel` categorical) that every model must cope with; the categorical one exercises native categorical handling.
- `scoring/retention_scoring.yml` - a batch scoring pipeline (ADR-20/21): champion resolution by stage at run time, PSI/KS shift monitors against the training-time baseline, and delayed ground-truth evaluation via `mbt monitor`.
- `metrics.yml` / `exposures.yml` - shared metric declarations and downstream exposure lineage (they show up in `mbt docs generate`).
- `tests/test_no_leakage.py` - Python data tests, including the pin that keeps the planted leak leaky (the teaching asset cannot silently rot).
