# Quickstart

From zero to a trained, registered, served, and monitored model on your laptop.
The whole loop takes a few minutes; training itself takes seconds.

## 1. Install

The packages are not on PyPI yet, so the quickest start is a source checkout, which installs every adapter:

```bash
git clone https://github.com/satrijandi/mbt && cd mbt
uv sync && source .venv/bin/activate     # puts `mbt` on PATH
mbt --version                            # mbt 0.1.0
```

Python 3.11 through 3.14 are supported.
[Installation](installation.md) covers installing from a release tag, the adapter packages, and the optional extras.

## 2. Scaffold a project

```bash
mbt init my_models
cd my_models
```

You get a working project: an example source, dataset, model, and scoring pipeline; `profiles.yml` with `dev` and `prod` targets; seven reference GitHub Actions workflows; a pinned CI install set (`requirements.txt`); pre-commit and Renovate config; and `CODEOWNERS`.
`profiles.yml` is committed, because CI has no `~/.mbt/` to read, and a copy is installed to `~/.mbt/profiles.yml` too.

## 3. Get data

```bash
python scripts/generate_sample_data.py
```

This writes deterministic sample data: 5,000 training rows in `data/subscribers/`, plus a fresh scoring batch (`data/scoring_batch/`) and matured outcomes (`data/churn_outcomes/`) for the scoring and monitoring steps.
For a real project, point `sources.yml` at your own Parquet:

```yaml
sources:
  - name: lakehouse
    tables:
      - name: subscribers
        path: data/subscribers/*.parquet
```

## 4. Validate

```bash
mbt parse
```

```text
Parsed 3 nodes, 3 sources, 0 exposures in 0.02s
```

Parsing executes nothing.
It validates every spec against its schema (with did-you-mean suggestions), checks task and adapter compatibility and hyperparameters, builds the DAG, and reports every error in one pass.
Add `--write-json-schema` to write JSON Schemas to `target/json-schemas/`; the scaffolded specs already point their `yaml-language-server` headers there, so your editor autocompletes them.

## 5. Build

```bash
mbt build
```

```text
Compiling against target 'dev'
Compiled 3 nodes in 0.19s (anchor 2026-09-14T11:21:38Z) -> target/manifest.json
build: 2 node(s) selected on target 'dev'
[1/2] START dataset dataset.my_models.churn_training_set
materialized 2087 rows: test=322, train=1765
check class_balance_report: PASS - label balance (train): 0=78.867%, 1=21.133%
check no_future_columns: PASS
check not_null: PASS
check label_leakage_scan: PASS
test test_label_is_binary: PASS - label classes: [0, 1]
test test_minimum_rows: PASS - 2087 rows
[1/2] SUCCESS dataset dataset.my_models.churn_training_set in 0.14s
[2/2] START model model.my_models.churn_classifier
auto-resolved scale_pos_weight = 3.731903
gate pr_auc (threshold): PASS - expected 0.3, got 0.4578
registered churn_classifier v1 -> staging (mlflow)
[2/2] SUCCESS model model.my_models.churn_classifier in 2.95s
build finished [success]: 2 ok, 0 failed, 0 skipped in 5.3s
```

Events like these go to stderr (timestamps omitted here), and a results table goes to stdout.
In order, `mbt build`:

1. **compiles** - renders Jinja and profiles, resolves the time windows against one anchor, pins a data snapshot, gives every node a config hash and a transitive input hash, and writes `target/manifest.json`;
2. **builds datasets** - DuckDB materializes the train and test splits the windows declare (the `dev` target samples half the rows), then runs the built-in checks, including the label-leakage scan that runs by default, and your Python data tests;
3. **trains models** - each in its own subprocess: `{{ auto }}` hyperparameters resolve from the data, metrics compute on the pinned test split, and the run is logged to MLflow;
4. **gates** - thresholds and champion comparisons decide, and a model that passes every gate registers in the MLflow registry at `staging`;
5. **records results** - `target/run_results.json` (and `target/run_results.build.json`) hold every node's status, timing, metrics, and gate verdicts.

## 6. Inspect

```bash
mbt ls                                   # resources, tags, paths
mbt show churn_classifier                # the fully compiled config, secrets redacted
mbt docs generate && mbt docs serve      # model cards + lineage at http://127.0.0.1:8080
```

## 7. Reproduce

```bash
mbt run --manifest target/manifest.json
```

This re-executes the stored manifest verbatim - same anchor, windows, snapshots, and seeds - and reproduces the metrics exactly, after checking the environment still matches the one the manifest was compiled in.
The reproduced model registers as `v2`.

## 8. Promote

```bash
mbt promote --model churn_classifier --to production
```

```text
promoted churn_classifier v2 -> production
```

`mbt promote` moves the version currently in `staging`, and refuses any version whose gates were not recorded as passed.
In CI, a reviewed change to `promotions.yml` drives it instead (see [GitOps & CI](gitops.md)).

## 9. Score and monitor

```bash
mbt score
```

```text
scoring input materialized 224 rows to score
check not_null: PASS
feature_shift most shifted: monthly_usage=0.0794, tenure_days=0.0258, support_tickets=0.0222 (top 3 of 5)
score finished [success]: 1 ok, 0 failed, 0 skipped in 3.5s
```

`scoring/churn_scoring.yml` is a whole serving pipeline in one file: which champion to load (resolved from the registry when the run starts), which batch to score, where predictions land (`predictions/churn_scores/`), and which distributions to watch against the champion's training-time baseline.
A failed input check or a shift beyond its threshold exits `2`, like a failing gate.

```bash
mbt monitor
```

Right after scoring this reports `0 matured prediction runs to evaluate`: the spec says outcomes mature after 14 days, and a prediction is only graded once its outcome is known.
To see the end state now, run the monitor as if it were two weeks later:

```bash
mbt monitor --anchor "$(python -c 'import datetime as d; print((d.datetime.now(d.timezone.utc) + d.timedelta(days=15)).strftime("%Y-%m-%dT%H:%M:%SZ"))')"
mbt predictions ls
```

The monitor joins the outcomes to the stored predictions, computes realized metrics, applies the pipeline's ground-truth gates, and records the run as evaluated, so running it again evaluates nothing new.
`mbt predictions ls` shows every prediction run and whether it has matured and been evaluated.

## Day-to-day

```bash
mbt build --select churn_classifier            # one model and the data it needs
mbt build --select tag:weekly                  # a scheduled retraining set
mbt build --select state:modified+ --state state/prod/latest.json   # only what changed
mbt test                                       # checks, data tests, and gates - never trains
mbt evaluate --model churn_classifier --stage production --gates    # the champion on fresh data
mbt score --select tag:daily                   # a scheduled scoring set
mbt monitor                                    # evaluate matured predictions
```

Next: the [Tutorial](tutorial.md) walks the same project through a team's pull-request loop and production wiring, and [Concepts](concepts.md) explains the model behind it.
