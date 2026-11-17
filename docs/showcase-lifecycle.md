# Showcase lifecycle on Airflow

`make lifecycle` runs the showcase's whole model lifecycle the way production runs it: every retrain, score and monitor is an Airflow DAG run, drawn by `mbt-airflow` from the compiled project and executing mbt inside a deployable unit pinned by digest.
A dataset or a model is a group of tasks, one per step ([ADR-35](adr/0035-phases-of-a-node.md)); a scoring node is one task.
`make demo` tells the same story by hand, running mbt inside the JupyterLab container; this page is for watching the scheduled side actually run.

Every command and quoted output below was captured from a real run on a fresh stack, in the order given here.
Timings are from one run on a 10-core laptop with every image already built and other work running beside it, so expect them to vary.
The [walkthrough](showcase-walkthrough.md) covers the same platform step by step through the UIs, including governance and PRs; this page is the one-command version of its scheduling half.

## Before you start

You need docker with about 10GB of RAM to spare, `make`, `rsync`, `uv`, and a checkout of the mbt repository.
Run every `make` command from `examples/showcase`:

```bash
cd examples/showcase
```

`make up` builds the stack's images from this checkout (about 20 minutes the first time, nothing pulled from a registry) and afterwards rebuilds only what changed; `scripts/build_images.sh --force runner` rebuilds the runner even when nothing did.

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

Builds the stack's images if needed (among them the runner, the one image that holds mbt, Spark, H2O and JupyterLab), stages the workspace, starts every service, and generates the lake table `s3://mbt-lake/churn_panel`: nine weekly cohorts, the newest of which (2026-09-28) has no labels yet.
It takes about 3 minutes and ends with every UI URL, the last of them:

```text
  Airflow       http://localhost:8280 (started by make ci: admin / admin)
```

## 3. Optional: the hand-driven demo

```bash
make demo
```

The same lifecycle by hand inside JupyterLab, with no Airflow: dev build, prod build, promote, a weaker challenger the champion gate refuses, score, monitor, land outcomes, monitor.
Skip it if you only want the scheduled side.
Running it first is a harder test for the next step: it leaves a production champion behind, so the lifecycle's retrained model has to beat it at the champion gate before it can be promoted.

## 4. Run the lifecycle

```bash
make lifecycle; echo "exit: $?"
```

It took about 35 minutes on a fresh stack in the run quoted below, 15 of them the retrain, and takes less on a rerun, which skips the CI setup and the first bake.
Each DAG step prints the run's Airflow URL, its state changes, and the tail of the run's task logs in the order the tasks ran, which is mbt's own output from inside the pinned unit.
Each task ends with a `result` line: the node's status, its metrics, and any message, as `mbt-airflow` read them back from the finished container.
If any DAG run ends in anything but `success`, `make` stops there with exit 1.

| Step | What runs | What it proves |
|---|---|---|
| (setup) | `make ci`, only if the Gitea repo does not exist yet | the forge, the repos and CI exist |
| 1. Deployable unit | the first prod-build bakes and pins the unit; Airflow registers the DAGs | the DAGs have a pinned unit to run |
| 2. Reset | the newest cohort goes back to as-seeded, labels empty | every run starts from the same data |
| 3. `mbt_retrain` | `mbt build --target prod` in the unit, each node a group of tasks: the dataset is materialized and checked, then the model is trained, judged against its gates and registered - cluster pushdown, AutoML, gates, registration | retraining runs on a schedule, and each step can be retried alone |
| 4. Promote | `make protect`, then a `promotions.yml` PR from `mbtds`, approved and merged by `mbtops`, whose promote pipeline runs `mbt promote --from-file` | production changes only through review, and promotion is a registry event, not a deploy |
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
    deployable unit baked and pinned: localhost:15000/mbt/churn@sha256:ca078b5e...
    git-sync has handed the pin to Airflow: every DAG task runs that unit
    Airflow has mbt_retrain, mbt_retrain_baseline, mbt_score, mbt_monitor registered and unpaused
```

The DAGs run whatever image `images.env` in the deploy repo pins, and are drawn from the `manifest.json` that prod-build commits in the same commit as the pin.
A fresh stack has neither, so it has no DAGs yet.
So the target commits `LIFECYCLE.md` to the churn repo's `main` as `mbtops` - exactly what a merge produces - and Woodpecker's prod-build does the rest: trains on the `dev` target, publishes the state baseline, bakes the unit, pushes it to Zot, and writes its digest into the deploy repo.
git-sync then copies the deploy repo to Airflow every 5 seconds, so for a moment Gitea has the pin and the copy a DAG task reads does not; the target waits until that copy pins the same unit, then until all four DAGs (`mbt_retrain`, `mbt_retrain_baseline`, `mbt_score`, `mbt_monitor`) are registered and unpaused.
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
    mbt_retrain: success (905s)
...
  04:25:24  check unique: PASS - source lake.churn_panel: (customer_id,
            inference_date) unique
  04:25:26  [1/1] SUCCESS dataset dataset.churn_lake.churn_training in 21.99s
result {"node": "dataset.churn_lake.churn_training", "status": "success"}
--- mbt_retrain.churn_automl.train try 1 (success) ---
...
  04:33:41  [1/1] SUCCESS model model.churn_lake.churn_automl in 439.12s
result {"metrics": {"logloss": 0.2697652113107183, "pr_auc": 0.33488637292468415, "roc_auc": 0.8066368397803306}, "node": "model.churn_lake.churn_automl", "status": "success"}
--- mbt_retrain.churn_automl.gate try 1 (success) ---
...
  04:34:33  gate pr_auc (threshold): PASS - expected 0.2, got 0.3349
  04:34:33  WARN champion gate on 'pr_auc': no champion in 'production' yet -
            passing with a warning (FR-TEST-06)
  04:34:33  gate pr_auc (champion): PASS - no champion registered yet; gate passes
            (bootstrap)
  04:34:33  [1/1] SUCCESS model model.churn_lake.churn_automl in 0.02s
result {"metrics": {"logloss": 0.2697652113107183, "pr_auc": 0.33488637292468415, "roc_auc": 0.8066368397803306}, "node": "model.churn_lake.churn_automl", "status": "success"}
--- mbt_retrain.churn_automl.register try 1 (success) ---
...
  04:36:01  registered churn_automl v2 -> staging (mlflow)
  04:36:02  [1/1] SUCCESS model model.churn_lake.churn_automl in 1.60s
result {"metrics": {"logloss": 0.2697652113107183, "pr_auc": 0.33488637292468415, "roc_auc": 0.8066368397803306}, "node": "model.churn_lake.churn_automl", "status": "success"}
```

The DAG has six tasks, in this order:

| Task | What it runs | Seconds |
|---|---|---|
| `mbt_compile` | asks `scripts/lake_anchor.py` for the lake's anchor and compiles once at it | 85 |
| `churn_training.materialize` | builds the dataset, pushed down to the Spark cluster | 98 |
| `churn_training.checks` | runs the dataset's checks | 75 |
| `churn_automl.train` | trains with H2O AutoML inside the Spark executors | 494 |
| `churn_automl.gate` | runs the gates: the `pr_auc` floor, and the paired-bootstrap comparison against the current production champion | 51 |
| `churn_automl.register` | registers the model that passed and moves the `staging` alias | 90 |

The seconds are what Airflow reported for each task in the run above.
Airflow starts the pinned unit as a container on the compose network for each task, and each runs `mbt build --target prod` on exactly its node and its step (`--phase`) against the one manifest the first task compiled.
The steps of a node share one target directory under the shared workspace, so the model that `train` fitted is the model `gate` judges and `register` registers.

A step can be retried on its own: a registry blip while registering retries `register` alone instead of training the model again, and a model that fails its gates is the `gate` step's exit 2, which is not retried.
That has a price, which the table shows: every step starts a container, a Spark session and a compile against the manifest, so even a step that does almost nothing took between 51 and 90 seconds here (`gate` spent 0.02 seconds in mbt's own node time inside its 51-second task).
`phases=False` on the DAG draws one task per node instead, as [Airflow](airflow.md#a-node-is-a-group-of-steps) describes.

### 4. Promote

```text
==> [4/8] Governed promotion: mbtds opens a promotions.yml PR, mbtops approves, CI promotes
committed CODEOWNERS: promotions.yml @mbtops
protected main
    mbtds pins churn_automl v2 for production in promotions.yml on promote-churn_automl-v2-1794890177
    mbtds opened PR #1: http://localhost:3305/mbt-showcase/churn/pulls/1
    pr-check pipeline #3: http://localhost:8305/repos/1/pipeline/3
    mbtops (the code owner of promotions.yml) approves and merges PR #1
    promote pipeline #4: http://localhost:8305/repos/1/pipeline/4
    production now serves churn_automl v2: a registry event, the unit unchanged
```

Nothing here is done by hand, and nothing skips the team's rules.
`make protect` puts `promotions.yml` under review (CODEOWNERS, and a rule on `main` that only `mbtops` may approve and that binds administrators too).
The data scientist `mbtds` pins the version the retrain staged in a `promotions.yml` PR; pr-check lints it (an unpinned entry fails) and state-diffs it (nothing to retrain); `mbtops`, the code owner, approves and merges it; and the merge's promote pipeline runs `mbt promote --from-file promotions.yml`, which checks again that the version passed its gates before it moves the `production` alias.
Nothing is redeployed: the unit and the deploy repo stay exactly as they were, and the next scoring run picks the new champion up on its own.
If the staged version already serves production, the step says so and opens no PR.

### 5. Score

```text
==> [5/8] Airflow mbt_score: the newest cohort, scored by the run-time production champion
  04:41:54  scoring input materialized 2035 rows to score
result {"metrics": {"rows_scored": 2035.0}, "node": "scoring.churn_lake.retention_scoring", "status": "success"}
```

The unit runs `mbt score --target batch`: it resolves the production champion when the run starts, scores the newest cohort's active customers, checks feature and prediction shift against the champion's training baseline, writes the predictions, and pushes metrics to Prometheus.

### 6. Monitor before the outcomes

```text
==> [6/8] Airflow mbt_monitor before the outcomes land: nothing matured, so it waits
  04:43:00  WARN run 4ff62b2d450dcc88: no matured labels joined (join_key:
            customer_id, inference_date); will retry next monitor run
  04:43:00  [1/1] SUCCESS scoring scoring.churn_lake.retention_scoring in 3.67s -
            evaluated 0 of 1 matured prediction run(s)
result {"message": "evaluated 0 of 1 matured prediction run(s)", "node": "scoring.churn_lake.retention_scoring", "status": "success"}
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
  04:43:41  [1/1] SUCCESS scoring scoring.churn_lake.retention_scoring in 6.14s -
            evaluated 1 of 1 matured prediction run(s)
result {"message": "evaluated 1 of 1 matured prediction run(s)", "metrics": {"pr_auc": 0.3248397552053703, "roc_auc": 0.8035563810211698}, "node": "scoring.churn_lake.retention_scoring", "status": "success"}
```

The same DAG again, with the same two tasks: it joins the predictions to the real labels on `(customer_id, inference_date)`, computes the realized metrics, and checks the realized `pr_auc` gate.
Each prediction run is graded exactly once; a later monitor finds nothing left to do.
The target ends with where to look:

```text
  Airflow DAG runs  http://localhost:8280/dags  (admin / admin)
  Predictions       /Users/you/.cache/mbt-showcase/workspace/predictions/retention_scores/
  Grafana           http://localhost:3390  (mbt Model Health)
```

## 5. Check it yourself

- **Airflow** (http://localhost:8280, `admin` / `admin`), **Dags**: one green run each for `mbt_retrain` and `mbt_score`, two for `mbt_monitor`. Open a run to see its graph: `mbt_compile`, then for the retrain the task groups `churn_training` (`materialize`, `checks`) and `churn_automl` (`train`, `gate`, `register`), run in that order; open a task, then **Logs**, to see the same mbt output.
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
The retrain registers the next version, which must pass the paired champion gate against the one you just promoted, the promotion opens a second reviewed PR, and the rest repeats on the same data.

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
A task that fails with a hard error is retried once, and the log shows every attempt: a `try 1 (failed)` followed by a `try 2 (success)` is a step that recovered, and the run is green.
Each failed attempt prints its exception under `Task failed with exception`; the [troubleshooting runbook](troubleshooting.md#readtimeout-read-timed-out-read-timeout60-fails-a-showcase-dag-task-before-mbt-runs) covers the one seen so far.
