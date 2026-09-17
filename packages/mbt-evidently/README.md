# mbt-evidently

Evidently drift reports for [mbt](https://github.com/satrijandi/mbt), the declarative build tool for machine learning models.
Every mbt training run already writes a report with the model's score and feature stability after its test window; this package adds [Evidently](https://github.com/evidentlyai/evidently)'s data-drift report beside it.

```bash
pip install mbt-evidently      # plus mbt-core and a training adapter
```

mbt is not on PyPI yet; see [Installation](https://satrijandi.github.io/mbt/installation/) for installing from a release tag.

## Turn it on

```yaml
# datasets/churn_training_set.yml - the rows after the test window
datasets:
  - name: churn_training_set
    # ...source, label, sample_key
    split:
      strategy: temporal
      time_column: snapshot_date
      train: "-12mo:-4mo"
      test: "-4mo:-3mo"
      out_of_time: "-3mo:now"
```

```yaml
# models/churn_classifier.yml
models:
  - name: churn_classifier
    # ...task, adapter, dataset, target, seed
    evaluation:
      metrics: [pr_auc, roc_auc]
      report:
        stability:
          engine: evidently
          feature_top_n: 20        # the most important features, plus the score
          max_html_reports: 12     # the whole window, then the newest months
```

`mbt parse` fails with `adapter 'evidently' is not installed` when the package is missing, so a spec never asks for a report the environment cannot render.

## What lands on the run

For the whole after-test window and for each month after it, the training job compares the test split with that period and writes, under the run's `report/` directory:

| File | Contents |
|---|---|
| `stability/evidently/window.html`, `stability/evidently/<YYYY-MM>.html` | Evidently's interactive drift report for that period |
| `stability/evidently/drift_by_period.csv` | Rows compared and the share of columns Evidently calls drifted |
| `stability/evidently/drift_by_column.csv` | Each column's test, score, threshold, and verdict |

`report.html` shows the same tables in its stability section, and `mbt evaluate --out-of-time` writes the same files for its pre-deploy check.
Evidently picks each column's test by type and sample size: a Kolmogorov-Smirnov or chi-square p-value on small samples, a Wasserstein or Jensen-Shannon distance on large ones.

## What it does not do

**It never gates.**
`evaluation.stability` and after-test gates use mbt's own PSI and KS, the statistics `mbt score` monitors in production, so "stable before deploy" and "stable in production" mean the same thing ([ADR-30](https://satrijandi.github.io/mbt/adr/0030-training-report-and-after-test-window/)).
Evidently's verdicts are there to read.

**It never fails a build.**
If Evidently cannot render a period, the report says so in its warnings and training carries on.

Each HTML report embeds its charting library and runs to a few megabytes, which is what `max_html_reports` bounds.
Evidently's version is not part of mbt's environment digest, because a report never changes a model.
The job sets `DO_NOT_TRACK=1` before Evidently loads, so its usage telemetry stays off.

## License

Apache License 2.0.
