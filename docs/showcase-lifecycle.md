# Showcase lifecycle on Airflow

`make lifecycle` runs the showcase's whole model lifecycle the way production runs it: every retrain, score and monitor is an Airflow DAG run, executing mbt inside a deployable unit pinned by digest.
`make demo` tells the same story by hand, running mbt inside the JupyterLab container; this page is for watching the scheduled side actually run.

Every command and quoted output below was captured from a real run on a fresh stack, in the order given here.
Timings are from a 10-core laptop with the runner image already built.
The [walkthrough](showcase-walkthrough.md) covers the same platform step by step through the UIs, including governance and PRs; this page is the one-command version of its scheduling half.

## Before you start

You need docker with about 10GB of RAM to spare, `make`, `rsync`, `uv`, and a checkout of the mbt repository.
Run every `make` command from `examples/showcase`:

```bash
cd examples/showcase
```

`make up` rebuilds the runner image only when its inputs changed; `make image` forces a rebuild (10-15 minutes on a cold cache).

## 1. Start from nothing

```bash
make clean
```

Stops and removes every container, volume and network of the `mbt-showcase` compose project, then deletes the workspace (`~/.cache/mbt-showcase/workspace`).
Gitea, Woodpecker, MLflow, the lake and Airflow all start empty, so nothing from an earlier run can leak in.

## 2. Boot the stack

```bash
make up
```

Builds the runner image if needed (the one image that holds mbt, Spark, H2O and JupyterLab), stages the workspace, starts every service, and generates the lake table `s3://mbt-lake/churn_panel`: nine weekly cohorts, the newest of which (2026-09-28) has no labels yet.
It takes about 3 minutes and ends with every UI URL, the last of them:

```text
  Airflow       http://localhost:8280 (admin / admin; DAGs appear after make ci)
```

## 3. Optional: the hand-driven demo

```bash
make demo
```

The same lifecycle by hand inside JupyterLab, with no Airflow: dev build, prod build, promote, score, monitor, land outcomes, monitor.
Skip it if you only want the scheduled side.
Running it first is a harder test for the next step: it leaves a production champion behind, so the lifecycle's retrained model has to beat it at the champion gate before it can be promoted.

## 4. Run the lifecycle

```bash
make lifecycle; echo "exit: $?"
```

It takes about 7 minutes on a fresh stack, most of it the first unit bake, and about 3.5 minutes on a rerun.
Each DAG step prints the run's Airflow URL, its state changes, and the tail of the task log, which is mbt's own output from inside the pinned unit.
If any DAG run ends in anything but `success`, `make` stops there with exit 1.

| Step | What runs | What it proves |
|---|---|---|
| (setup) | `make ci`, only if the Gitea repo does not exist yet | the forge, the repos and CI exist |
| 1. Deployable unit | the first prod-build bakes and pins the unit; Airflow registers the DAGs | the DAGs have a pinned unit to run |
| 2. Reset | the newest cohort goes back to as-seeded, labels empty | every run starts from the same data |
| 3. `mbt_retrain` | `mbt build --target prod` in the unit: cluster pushdown, AutoML, gates, registration | retraining runs on a schedule |
| 4. Promote | `mbt promote` moves the `production` alias | promotion is a registry event, not a deploy |
| 5. `mbt_score` | `mbt score --target batch` with the run-time champion | scoring serves the new champion without a redeploy |
| 6. `mbt_monitor` | ground-truth monitoring before the labels exist | monitoring waits instead of failing |
| 7. Outcomes land | the newest cohort's labels are written | a week has passed |
| 8. `mbt_monitor` | ground-truth monitoring with the labels in | realized metrics are graded, exactly once |

### Setup: the CI loop

If `http://localhost:3305/mbt-showcase/churn` does not exist yet, the target runs `make ci` first.
That creates the `mbtops` and `mbtds` accounts, the `mbt-showcase/churn` repo (the project) and `mbt-showcase/deploy` (what Airflow runs), connects Woodpecker to Gitea, and turns CI on:

```text
  CI ready:
  Gitea        http://localhost:3305  (mbtops / mbtops-showcase-password)
```

### 1. Deployable unit

```text
==> [1/8] Deployable unit: prod-build baked and pinned by digest, DAGs live in Airflow
    no deployable unit pinned yet: pushing to main so prod-build bakes the first one
    prod-build pipeline #1: http://localhost:8305/repos/1/pipeline/1
    deployable unit baked and pinned: localhost:15000/mbt/churn@sha256:3bdd6a27...
    Airflow has mbt_retrain, mbt_score, mbt_monitor registered and unpaused
```

The DAGs run whatever image `images.env` in the deploy repo pins, and a fresh stack pins none.
So the target commits `LIFECYCLE.md` to the churn repo's `main` as `mbtops` - exactly what a merge produces - and Woodpecker's prod-build does the rest: trains on the `dev` target, publishes the state baseline, bakes the unit, pushes it to Zot, and writes its digest into the deploy repo.
git-sync then hands the deploy repo to Airflow, and the target waits until all three DAGs are registered and unpaused.
On a rerun the unit is already pinned, and this step says so and moves on.

### 2. Reset

```text
==> [2/8] The newest cohort as seeded: its outcome week is still open, labels NULL
wrote 2026-09-28 (open): 4440 rows, 1 file(s), 1309797B
```

Puts the newest cohort back exactly as seeded, so a rerun scores and monitors the same rows as the first run.

### 3. Retrain

```text
==> [3/8] Airflow mbt_retrain: mbt build --target prod in the unit (cluster + sparkling H2O)
    mbt_retrain: success (189s)
          delta lower bound 0.000000 at 95% confidence
09:21:51  registered churn_automl v4 -> staging (mlflow)
```

Airflow starts the pinned unit as a container on the compose network, which runs `mbt build --target prod`: the training set is pushed down to the Spark cluster, H2O AutoML trains inside the executors, and the gates run - the `pr_auc` floor, and the paired-bootstrap comparison against the current production champion.
A model that passes is registered and moves the `staging` alias.

### 4. Promote

```text
==> [4/8] Gate-verified promotion of the retrained staging champion (a registry event)
promoted churn_automl v4 -> production
```

`mbt promote` checks that the staging version passed its gates and moves the `production` alias to it.
Nothing is redeployed: the unit and the deploy repo stay exactly as they were, and the next scoring run picks the new champion up on its own.

### 5. Score

```text
==> [5/8] Airflow mbt_score: the newest cohort, scored by the run-time production champion
│ scoring.churn_lake.retention_scoring │ success │ 17.47s │ rows_scored=2035 │
```

The unit runs `mbt score --target batch`: it resolves the production champion when the run starts, scores the newest cohort's active customers, checks feature and prediction shift against the champion's training baseline, writes the predictions, and pushes metrics to Prometheus.

### 6. Monitor before the outcomes

```text
==> [6/8] Airflow mbt_monitor before the outcomes land: nothing matured, so it waits
09:23:18  WARN run 86f2f0b17dab1b30: no matured labels joined (join_key:
          evaluated 0 of 1 matured prediction run(s)
```

The cohort's labels are still empty, so there is nothing to grade yet.
The run succeeds and leaves the prediction run for the next monitor; this `WARN` is the expected outcome, not a failure.

### 7. Outcomes land

```text
==> [7/8] A week passes: the newest cohort's outcomes land in the same table
wrote 2026-09-28 (matured): 4440 rows, 1 file(s), 1310773B
```

Rewrites the newest cohort with its `is_churn` labels filled in - the outcome week closing.

### 8. Monitor against the outcomes

```text
==> [8/8] Airflow mbt_monitor: realized metrics and gates against the landed labels
          evaluated 1 of 1 matured prediction run(s)
│ scoring.churn_lake.retenti… │ success │ 8.23s │ pr_auc=0.3248                │
│                             │         │       │ roc_auc=0.8036               │
```

The same DAG again: it joins the predictions to the real labels on `(customer_id, inference_date)`, computes the realized metrics, and checks the realized `pr_auc` gate.
Each prediction run is graded exactly once; a later monitor finds nothing left to do.
The target ends with where to look:

```text
  Airflow DAG runs  http://localhost:8280/dags  (admin / admin)
  Predictions       /Users/you/.cache/mbt-showcase/workspace/predictions/retention_scores/
  Grafana           http://localhost:3390  (mbt Model Health)
```

## 5. Check it yourself

- **Airflow** (http://localhost:8280, `admin` / `admin`), **Dags**: one green run each for `mbt_retrain` and `mbt_score`, two for `mbt_monitor`. Open a run, then its task, then **Logs**, to see the same mbt output.
- **Woodpecker** (http://localhost:8305, log in with the Gitea account `mbtops` / `mbtops-showcase-password`): pipeline #1 is green.
- **MLflow** (http://localhost:5501): `churn_automl` has a new version, and the `production` alias points at it.
- **Predictions** on disk: the newest run directory holds `ground_truth.marker.json`, the record that it was graded.

  ```bash
  ls ~/.cache/mbt-showcase/workspace/predictions/retention_scores/*/
  ```

- **Grafana** (http://localhost:3390, `admin` / `admin`): the "mbt Model Health" dashboard shows the fresh score and monitor metrics.

## 6. Run it again

```bash
make lifecycle; echo "exit: $?"
```

Step 1 now reports `deployable unit already pinned` and skips the bake.
The retrain registers the next version, which must pass the paired champion gate against the one you just promoted, and the rest repeats on the same data.

## Output that is not an error

Every DAG step prints a few lines that look alarming and are not:

| Line | What it means |
|---|---|
| `Setting default log level to "WARN".` | Spark's startup banner |
| `WARN NativeCodeLoader: Unable to load native-hadoop library for your platform... using builtin-java classes where applicable` | Hadoop uses its Java implementation, as it does in every container |
| `WARN MetricsConfig: Cannot locate configuration: tried hadoop-metrics2-s3a-file-system.properties,hadoop-metrics2.properties` | no Hadoop metrics config exists, and none is needed |
| `WARN SparkStringUtils: Truncated the string representation of a plan since it was too large.` | a debug print is shortened because the table is wide |
| `WARN run ...: no matured labels joined ... will retry next monitor run` | step 6's expected "waiting for labels" outcome |

A real failure ends the target: the failing step prints `<dag> run <run id> ended failed: <airflow url>` and `make` reports `Error 1`.
The task log printed just above it is the evidence; the full log is at that URL.
