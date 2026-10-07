# Showcase walkthrough

This is the hands-on companion to the [showcase](showcase.md): every command to type, what you should see after it, and where to look.
Part 1 is the correct path, from an empty machine to a model scored in production and monitored against real outcomes.
Part 2 goes past it: the champion gate refusing a weaker model, and the weekly model living through more weeks.
Part 3 is the wrong turns: each one's symptom, why it happens, and how to get back on the path.

Every command and every quoted output on this page was captured from a real run on a fresh stack; timings are from a 10-core laptop with the runner image already built.

## Before you start

You need docker with about 10GB of RAM to spare, `make`, `rsync`, `uv`, `git`, and a checkout of the mbt repository.
Run the `make` commands from `examples/showcase`.

The showcase has two people in it, and the walkthrough switches between them on purpose:

| Account | Password | Plays | Can |
|---|---|---|---|
| `mbtops` | `mbtops-showcase-password` | the platform owner | push to `main`, approve and merge PRs, own `promotions.yml` |
| `mbtds` | `mbtds-showcase-password` | a data scientist | push branches and open PRs; never push `main` once it is protected |

Airflow and Grafana have their own login, `admin` / `admin`.

The data is synthetic and fixed-dated: nine weekly cohorts, every Monday from 2026-08-03 to 2026-09-28, where 2026-09-28 is the newest and its outcome week is still open.
Every command takes its anchor from that newest cohort - `2026-09-29T00:00:00Z` to build and score, `2026-10-09T00:00:00Z` to monitor - so what you see matches this page whatever today's date is, until you move the lake on with `make advance-week` (part 2).

## Part 1: the correct path

| Step | You do | What it shows |
|---|---|---|
| [1](#1-boot-the-stack) | `make up` | the platform boots and the lake table is generated |
| [2](#2-set-up-the-ci-loop) | `make ci` | Gitea and Woodpecker are wired together |
| [3](#3-clone-with-credentials) | `make clone` | one working copy per person, with the credentials a push needs |
| [4](#4-first-push-builds-everything) | first push to `main` | with no baseline yet, CI builds everything and bakes the deployable unit |
| [5](#5-log-into-woodpecker) | log into Woodpecker | watching pipelines |
| [6](#6-a-pull-request-retrains-only-what-changed) | a PR that edits one model | CI retrains exactly that model and reports it on the PR |
| [7](#7-merge-it) | merge the PR | `main` retrains the same model and pins a new unit |
| [8](#8-put-promotion-under-review) | `make protect` | production becomes a reviewed decision |
| [9](#9-promote-through-a-reviewed-pr) | a promotion PR, approved and merged | the `production` alias moves, nothing is redeployed |
| [10](#10-score-in-production) | trigger `mbt_score` in Airflow | the newest cohort is scored with the production champion |
| [11](#11-monitor-against-real-outcomes) | `mbt_monitor`, `make outcomes`, `mbt_monitor` | realized metrics, evaluated exactly once |
| [12](#12-break-it-on-purpose-and-recover) | `make inject-drift`, then `make reset score` | an alert fires and clears |
| [13](#13-tear-down) | `make down` or `make clean` | back to an empty machine |
| [14](#14-a-weaker-challenger-meets-the-champion-gate) | `make challenger` | the champion gate refuses a worse model; production is untouched |
| [15](#15-a-week-passes) | `make week` | the lake moves on a week, and a week of operations runs |

### 1. Boot the stack

```bash
cd examples/showcase
make up
```

It first runs `make doctor`, which checks docker's memory and disk, the socket group Airflow needs, the ports, and leftover networks, and stops with the fix when something will not work.
The runner image comes prebuilt from `ghcr.io/satrijandi/mbt-showcase-runner` when one is published for your checkout; otherwise the first run builds it (10-15 minutes), and after that `make up` takes under a minute and a half.
It starts the lake, MLflow, JupyterLab, Spark and observability; Gitea, Woodpecker and Airflow start with `make ci` in step 2.
It ends by generating the lake table and printing every URL:

```text
wrote 2026-09-21 (matured): 4260 rows, 1 file(s), 1261169B
wrote 2026-09-28 (open): 4440 rows, 1 file(s), 1309797B
seeded s3://mbt-lake/churn_panel: 33480 rows x 68 columns; churn rate among labelled active rows 12.6%

  JupyterLab    http://localhost:8899
  MLflow        http://localhost:5501
  ...
  Gitea         http://localhost:3305 (after make ci: mbtops / mbtops-showcase-password)
  Woodpecker    http://localhost:8305 (after make ci: click login, use the Gitea account)
  Airflow       http://localhost:8280 (admin / admin; DAGs appear after make ci)
```

`matured` means the cohort's labels are known; `open` means its outcome week has not closed yet.
`make urls` prints the list again at any time.

### 2. Set up the CI loop

```bash
make ci
```

It takes under a minute: it creates both accounts, the `mbt-showcase/churn` repository (the project) and `mbt-showcase/deploy` (what Airflow runs), connects Woodpecker to Gitea, and turns CI on.
It ends with:

```text
  CI ready:
  Gitea        http://localhost:3305  (mbtops / mbtops-showcase-password)
  Woodpecker   http://localhost:8305  (click login, use the Gitea account)
  Repo         http://localhost:3305/mbt-showcase/churn
```

`make ci` does not protect any branch; step 8 does that, once the first builds have run.

### 3. Clone with credentials

```bash
make clone
```

It clones the repository twice into `~/showcase-work`, outside the mbt checkout - `ops` as the platform owner and `ds` as the data scientist - and each clone commits as its own account (the output spells out your home directory where this page writes `~`):

```text
  ~/showcase-work/ops  (as mbtops)
  ~/showcase-work/ds  (as mbtds)
```

`SHOWCASE_WORK=<dir> make clone` puts them elsewhere; running it again leaves existing clones alone.
The login rides in each clone's remote URL, because Gitea lets anyone read the repository but accepts a push only with credentials.
By hand, the first clone is:

```bash
git clone http://mbtops:mbtops-showcase-password@localhost:3305/mbt-showcase/churn.git ~/showcase-work/ops
```

These are throwaway demo logins, which is the only reason a password in a URL is acceptable here; for anything real, use a Gitea access token (Settings, then Applications) instead.
From here on, work in a second terminal in `~/showcase-work`, and keep the first one in `examples/showcase` for the `make` commands.

### 4. First push builds everything

The repository was seeded before CI was switched on, so nothing has been built yet.
Any push to `main` starts the first build:

```bash
cd ~/showcase-work/ops
git commit --allow-empty -m "kick off the first build"
git push
```

The prod-build pipeline takes about five minutes.
There is no published baseline to compare against yet, so it builds every node:

```text
fetch_state: no mbt-state branch on origin yet (bootstrap - no baseline)
No prod baseline yet; building everything (bootstrap).
[1/3] SUCCESS dataset dataset.churn_lake.churn_training in 9.82s
[2/3] SUCCESS model model.churn_lake.churn_automl in 83.70s
[3/3] SUCCESS model model.churn_lake.churn_baseline_xgb in 24.77s
publish_state: pushed target/manifest.json to branch mbt-state (...)
bake planned (trained=3 have_unit=0)
deploy repo pinned to new unit digest
```

In MLflow (http://localhost:5501), both models now have version 1 under the `staging` alias.
A `promote` workflow also runs and prints `applied 0 promotion(s)`: on a branch's first push Woodpecker cannot tell which files changed, so it runs every workflow, and `promotions.yml` is still empty.

### 5. Log into Woodpecker

Open http://localhost:8305, click **gitea**, and sign in to Gitea as `mbtops`.
You land on the repository list with `mbt-showcase / churn` and its pipelines; click into a pipeline to read each step's log.
The first time an account logs in, Gitea asks whether to let Woodpecker use it: click **Authorize Application**.
(`make ci` already did this for `mbtops`, so you only see the consent page as `mbtds`.)

### 6. A pull request retrains only what changed

Work as the data scientist from here, in the `ds` clone:

```bash
cd ~/showcase-work/ds
git checkout -b deeper-xgb
# in models/churn_baseline_xgb.yml, change   max_depth: 4   to   max_depth: 5
git commit -am "churn_baseline_xgb: max_depth 4 -> 5"
git push -u origin deeper-xgb
```

Gitea's reply includes a link to open the PR, on `http://localhost:3305/...`.
On the comparison page click **New Pull Request**, then **Create Pull Request**.

The pr-check pipeline takes about a minute.
It compares the branch with the published baseline and builds only what the change touches, on a throwaway `ci` target, so a PR never touches the shared registry:

```text
state diff: 0 added, 0 removed, 1 modified
build: 1 node(s) selected on target 'ci'
[2/2] SUCCESS model model.churn_lake.churn_baseline_xgb in 10.07s
gitea_pr_comment: created comment on PR #1
```

The PR now carries an **mbt build report** comment: what changed (`modified: model.churn_lake.churn_baseline_xgb (config)`), the nodes it ran, and each gate with its verdict (`pr_auc`, `threshold 0.2`, actual `0.3190`, **PASS**).
Pushing another commit to the branch updates the same comment instead of adding a new one.

### 7. Merge it

Sign in to Gitea as `mbtops`, open the PR, and click **Create merge commit** (twice: once to open the merge form, once to confirm).
The merge runs prod-build on `main`, which again retrains only the changed model, republishes the baseline, and bakes a new deployable unit because something was retrained:

```text
build: 1 node(s) selected on target 'dev'
[2/2] SUCCESS model model.churn_lake.churn_baseline_xgb in 8.87s
bake planned (trained=2 have_unit=1)
deploy repo pinned to new unit digest
```

MLflow now shows `churn_baseline_xgb` version 2 under `staging`; `churn_automl` is untouched at version 1.

### 8. Put promotion under review

So far anyone with write access can push `main` and merge.
Production should be a reviewed decision; in the first terminal:

```bash
make protect
```

It does two things as `mbtops`.
It commits a `CODEOWNERS` file (`promotions.yml @mbtops`), so every change to `promotions.yml` asks the platform owner for review.
And it protects `main`: only `mbtops` may push it directly, every merge needs one approval from `mbtops`, and administrators get no override.

```text
committed CODEOWNERS: promotions.yml @mbtops
protected main
```

The rule appears in Gitea under the repository's **Settings**, then **Branches**.
Running `make protect` again keeps the existing `CODEOWNERS` and resets the rule to this one (`main was already protected; rule updated to match`).
If you set the rule up by hand instead, tick **Administrators must follow branch protection rules**: `mbtops` is a Gitea administrator, and without it an administrator can merge a PR nobody approved ([wrong turn](#an-administrator-merged-a-pr-nobody-approved)).
Committing `CODEOWNERS` is a push to `main`, so it starts a prod-build pipeline, which finds nothing to retrain.

### 9. Promote through a reviewed PR

Look up the version to promote: in MLflow, `churn_automl` version 1 is under `staging`.
As `mbtds`, propose it:

```bash
cd ~/showcase-work/ds
git checkout main && git pull
git checkout -b promote-automl-v1
cat > promotions.yml <<'EOF'
promotions:
  - model: churn_automl
    to: production
    version: '1'
EOF
git commit -am "promote churn_automl v1 to production"
git push -u origin promote-automl-v1
```

Open the PR as in step 6.
Its checks pass, but the merge box says:

```text
This pull request doesn't have enough required approvals yet. 0 of 1 approvals granted from users or teams on the allowlist.
```

As `mbtops`, open the PR's **Files Changed** tab, click **Review**, then **Approve**, and merge with **Create merge commit**.
The merge runs the promote workflow:

```text
+ bash scripts/run_mbt.sh mbt promote --from-file promotions.yml
applied 1 promotion(s) from
/woodpecker/src/gitea/mbt-showcase/churn/promotions.yml
```

`churn_automl` version 1 now carries the `production` alias.
Nothing was retrained and nothing was baked (`nothing retrained and a deployable unit exists; skipping bake`): promotion is a registry event, and the next scoring run picks the new champion up with no redeploy.

### 10. Score in production

Open Airflow (http://localhost:8280, `admin` / `admin`), click **Dags**, then `mbt_score`, then **Trigger** at the top right.
The dialog shows an empty `anchor`: left empty, the run takes the lake's own anchor (the day after its newest cohort) inside the unit; click **Trigger** again.
The run takes about 40 seconds; the task log ends with:

```text
[1/1] SUCCESS scoring scoring.churn_lake.retention_scoring in 15.76s
score finished [success]: 1 ok, 0 failed, 0 skipped in 25.2s
│ scoring.churn_lake.retention_scoring │ success │ 15.76s │ rows_scored=2035 │
```

It scored the 2,035 active customers of the newest cohort (2026-09-28) with whatever model holds `production`.
The predictions are under `~/.cache/mbt-showcase/workspace/predictions/retention_scores/`.
`make score` runs the same scoring from the terminal.

### 11. Monitor against real outcomes

Trigger `mbt_monitor` the same way.
The scored cohort's labels are still NULL, so there is nothing to grade yet, and the run says so without failing:

```text
WARN run 802f6e7b8ad76f9b: no matured labels joined (join_key:
     customer_id, inference_date); will retry next monitor run
     evaluated 0 of 1 matured prediction run(s)
```

Now let the week pass - in the first terminal, in `examples/showcase`, rewrite the cohort with its outcomes:

```bash
make outcomes
```

Trigger `mbt_monitor` again, and it grades the run:

```text
evaluated 1 of 1 matured prediction run(s)
│ scoring.churn_lake.retenti… │ success │ 7.53s │ pr_auc=0.3248 │
```

Trigger it a third time and it evaluates nothing (`0 matured prediction runs to evaluate`): each prediction run is graded exactly once.

The monitor anchor, 2026-10-09, is later than the data's newest date on purpose: it only means something after `make outcomes`, which stands in for the outcome week passing.

### 12. Break it on purpose and recover

In the first terminal, in `examples/showcase`:

```bash
make inject-drift
```

This poisons the newest cohort's features and scores it.
Scoring refuses the shifted batch with exit 2, the quality-failure code:

```text
[1/1] MONITOR_FAILED scoring scoring.churn_lake.retention_scoring in
      14.79s - monitor breach: age_years: psi=6.1707 exceeds 0.2500 (n=2035
      vs baseline n=10500); ...
```

In Prometheus (http://localhost:9490, **Alerts**) `MbtShiftBreach` goes pending within about 20 seconds and firing within about 30; in Grafana (http://localhost:3390, `admin` / `admin`) the **mbt Model Health** dashboard plots the shift against its threshold.
Alertmanager then delivers it to the alert inbox, the same one CI's alerts land in; http://localhost:9309/requests shows it on the owner route, because a shift breach is a quality verdict for the model's owner rather than a page for on-call:

```text
/alert/owner firing [('MbtShiftBreach', 'notify'), ('MbtShiftBreach', 'notify'), ...]
```
Recover with:

```bash
make reset score
```

The cohort is restored and rescored, and the alert clears within a minute.
`make reset` restores the cohort exactly as seeded, which also takes back the outcomes from step 11; run `make outcomes` again to grade a new prediction run.

### 13. Tear down

In `examples/showcase`:

```bash
make down
```

This stops and removes the containers, their volumes and the network; the workspace under `~/.cache/mbt-showcase/workspace` survives.
`make clean` does the same and then removes the workspace too.
After either, start again from step 1; the next `make up` reuses the runner image.

## Part 2: watch the gate say no, and let time pass

Two things the correct path above never shows: the champion gate refusing a model, and the weekly model living through more than one week.
Both run in `examples/showcase` on a stack where step 9 put a champion in production; the version numbers below are from one run and will differ on yours.

### 14. A weaker challenger meets the champion gate

```bash
make challenger
```

A working copy drops the eight activity features from `churn_automl` and retrains on the shared registry.
The champion gate scores the production model on the challenger's own test rows - through the production model's own features - and the challenger loses by far more than the gate allows:

```text
gate pr_auc (threshold): PASS - expected 0.2, got 0.2049
gate pr_auc (champion): FAIL - paired bootstrap (1000 resamples):
          delta lower bound -0.157604 at 95% confidence
mbt build exited 2; production serves churn_automl v2
the champion gate refused the weaker challenger; production is unchanged
```

The challenger still clears the absolute floor (0.2); it is the comparison with production that keeps it out.
Nothing is staged, and the committed project is untouched (the copy lives under the workspace's `tmp/` and is removed).

### 15. A week passes

```bash
make week
```

The lake moves on a week: the 2026-09-28 cohort's outcome week closes, so its labels land, and the 2026-10-05 cohort arrives, still open.
Every anchor follows, because they all come from the newest cohort (`make anchors` prints them).
Then the week's operations run in order - grade last week's predictions, retrain on the newer month, promote the retrain if its gates staged it, score the new cohort:

```text
advanced s3://mbt-lake/churn_panel to 2026-10-05
==> [3/5] Weekly retrain on the newer month (champion gate vs production)
          delta lower bound -0.008912 at 95% confidence
==> [4/5] The retrain passed its gates and was staged: promote it
promoted churn_automl v3 -> production
```

The champion gate is a non-inferiority gate (`min_delta: -0.02`): a retrain may replace production if it is, at 95% confidence, no more than 0.02 PR-AUC worse on the same rows.
Some weeks it is not, and the step says so instead of failing:

```text
breach: pr_auc: challenger delta lower bound -0.0258 < required -0.02
==> [4/5] The champion gate refused this week's retrain (exit 2): production keeps its champion
```

The lake will not move past today's date - a cohort from next week describes customers who do not exist yet:

```text
advancing 1 week(s) moves the lake to 2026-10-12, after today (2026-10-07): that cohort has not happened yet. Pass --simulate-future (make advance-week FUTURE=1) to simulate it anyway.
```

`make week FUTURE=1` simulates such weeks, and each says `(simulated: after today)`.
Run it a few times and Grafana's panels become time series.
`make seed` takes the lake back to 2026-09-28.

## Part 3: wrong turns

Each entry starts with the symptom as you will see it.

| Where | Symptom |
|---|---|
| step 3 | [`repository ... not found` cloning](#repository-not-found-cloning-the-repo) |
| step 4 | [`could not read Username` or `Authentication failed` pushing](#authentication-failed-pushing) |
| step 6 | [the PR fails with `gate_failed`](#the-pr-fails-with-gate_failed) |
| step 6 | [the PR fails: `features.include ... names column(s) the dataset does not have`](#the-pr-fails-with-a-feature-the-dataset-does-not-have) |
| step 9 | [the PR fails at `lint-promotions`](#the-pr-fails-at-lint-promotions) |
| step 9 | [`Not allowed to push to protected branch main`](#not-allowed-to-push-to-protected-branch-main) |
| step 9 | [`Does not have enough approvals`](#does-not-have-enough-approvals) |
| step 9 | [an administrator merged a PR nobody approved](#an-administrator-merged-a-pr-nobody-approved) |
| step 9 | [the promote step fails: `has no version`](#the-promote-step-fails-has-no-version) |
| step 10 | [`no champion of 'churn_automl' in stage 'production' to score with`](#no-champion-to-score-with) |
| step 11 | [`mbt_monitor` failed on the first try and did not retry](#mbt_monitor-failed-and-did-not-retry) |
| step 11 | [`mbt_monitor` failed twice](#mbt_monitor-failed-twice) |
| any | [running `make ci` again logs everyone out of Woodpecker](#running-make-ci-again-logs-everyone-out-of-woodpecker) |

### `repository ... not found` cloning the repo

```text
remote: Not found.
fatal: repository 'http://localhost:3305/mbt-showcase/churn.git/' not found
```

**Why:** the repository does not exist until `make ci` creates it; `make up` does not start Gitea at all.

**Fix:** run `make ci`, then clone again.

### `Authentication failed` pushing

With no login in the clone URL, git asks for one, and with nothing to answer it:

```text
fatal: could not read Username for 'http://localhost:3305': terminal prompts disabled
```

With a wrong password:

```text
remote: Failed to authenticate user
fatal: Authentication failed for 'http://localhost:3305/mbt-showcase/churn.git/'
```

**Why:** Gitea serves the repository to anyone but accepts a push only from an account with write access.

**Fix:** `make clone` makes clones that carry the login, or put it in an existing clone's remote URL:

```bash
git remote set-url origin http://mbtops:mbtops-showcase-password@localhost:3305/mbt-showcase/churn.git
```

On macOS, a wrong password you typed once may be saved in the keychain and offered again; remove it with `printf "protocol=http\nhost=localhost:3305\n\n" | git credential-osxkeychain erase`.

### The PR fails with `gate_failed`

The build step exits 2 and the PR comment shows the gate:

```text
| model.churn_lake.churn_baseline_xgb | pr_auc | threshold 0.99 | 0.3190 | - | **FAIL** |

> ❌ 1 node(s) failed - registration blocked.
```

**Why:** the model trained but did not clear one of its gates, here a PR-AUC floor of 0.99 no model reaches.
Exit 2 is a quality verdict, not a crash, so the alert goes to the model's owner rather than on-call; the webhook sink (http://localhost:9309/requests) records it as:

```text
{"source": "mbt-ci", "class": "quality-failure", "notify": "owner", "owner": "growth-ds@example.com", ...}
```

Nothing reaches the shared registry: a PR builds on the throwaway `ci` target.

**Fix:** improve the model or, if the gate is wrong, change the gate - either way in a new commit on the branch, which reruns the check and updates the same comment.

### The PR fails with a feature the dataset does not have

```text
[2/2] ERROR model model.churn_lake.churn_baseline_xgb in 3.09s -
      features.include of model 'churn_baseline_xgb' names column(s) the
      dataset does not have: txn_count_30d
      hint: every include entry must match at least one column after
      transform_features; did you mean: 'txn_count_30d' -> txn_cnt_30d,
      txn_amt_sum_30d (the dataset has 68 columns)
```

**Why:** a name in the model's `features.include` list matches no column in the table, almost always a typo (the table has `txn_cnt_30d`).
This is a hard error (exit 1), so the alert pages on-call.
mbt releases before this check dropped the name silently and trained on one feature fewer; if your runner image predates it, the PR goes green instead, so read the diff.

**Fix:** correct the name; the hint suggests the closest columns.
See also the [troubleshooting entry](troubleshooting.md#featuresinclude-of-model-name-names-columns-the-dataset-does-not-have).

### The PR fails at `lint-promotions`

```text
lint_promotions: entries without a pinned `version:`: churn_automl
Unpinned promotions are not replayable (promotion vacates the staging alias); pin the exact version you reviewed.
```

The build step is skipped, and the PR comment reads:

```text
No `run_results.json` produced: mbt never executed. Either an earlier step failed (lint-promotions rejects an unpinned `promotions.yml` entry) or the build failed before running a node. The failing step's log in Woodpecker says which.
```

**Why:** an entry without `version:` would promote "whatever is in staging" at merge time, which is not what the reviewer saw, and promoting it moves the `staging` alias away, so replaying the same file later fails.

**Fix:** add `version: '<n>'` with the exact version you reviewed in MLflow.

### `Not allowed to push to protected branch main`

```text
remote: error: Not allowed to push to protected branch main
 ! [remote rejected] main -> main (pre-receive hook declined)
```

**Why:** after step 8, only `mbtops` may push `main`; this was a push by `mbtds`.
That is the protection working.

**Fix:** push a branch and open a PR (step 9).
Your commit is still in your local `main`; move it to a branch with `git checkout -b <branch>`, and put `main` back with `git branch -f main origin/main`.

### `Does not have enough approvals`

The merge box says `0 of 1 approvals granted from users or teams on the allowlist`, and a merge through the API answers `405 {"message":"Does not have enough approvals"}`.

**Why:** step 8 requires one approval from `mbtops`.
An approval from anyone else, or a comment, does not count.

**Fix:** as `mbtops`: **Files Changed**, **Review**, **Approve**, then merge.

### An administrator merged a PR nobody approved

There is no error: the merge simply succeeds, and the PR's reviews show only the review CODEOWNERS requested, never an approval.

**Why:** `mbtops` is a Gitea administrator, and by default Gitea lets administrators merge past required approvals.

**Fix:** run `make protect`: it resets the rule to one that sets `block_admin_merge_override`.
If you created the rule by hand in Gitea, tick **Administrators must follow branch protection rules** on it.

### The promote step fails: `has no version`

After the merge, the promote workflow fails with exit 1:

```text
+ bash scripts/run_mbt.sh mbt promote --from-file promotions.yml
Error: model 'churn_baseline_xgb' has no version '9'
```

It is a hard error, so the alert pages on-call (`"class": "hard-error", "notify": "on-call"`).

**Why:** `promotions.yml` pinned a version the registry does not have.
Nothing checks that before the merge, so the reviewer is the check.

**Fix:** fix forward with another reviewed PR that pins a version MLflow shows; merging it reruns the promotion.
Never repair `main` by hand: the file on `main` is the record of what production should be.

### No champion to score with

```text
│ scoring.churn_lake.retentio… │ error │ 0.03s │ no champion of 'churn_automl' in stage 'production' to score with │
```

**Why:** scoring serves whatever model holds the `production` alias, and none does yet.

**Fix:** promote one (step 9), then score again.
See also the [troubleshooting entry](troubleshooting.md#no-champion-of-churn_model-in-stage-production-to-score-with).

### `mbt_monitor` failed and did not retry

The task fails on its first try with:

```text
AirflowFailException: mbt exited 2: quality verdict (gate/check/monitor). Deterministic - not retried; notify the model owner, not on-call.
```

The task log above it names the breach, for example `realized pr_auc=0.3248 failed >= 0.99`.

**Why:** the model's realized performance missed a gate on the scoring node.
A quality verdict does not change on a retry, so the DAG does not retry it, and it goes to the model's owner.

**Fix:** it is working as intended; the model needs attention.
(The DAG's `vars` parameter is how this page produced the failure: `pr_auc_floor: 0.99`.)

### `mbt_monitor` failed twice

Both tries fail with:

```text
Error: target 'prod_typo' not defined for project 'churn_lake'
AirflowException: mbt exited 1: hard error - Airflow retries, then on-call.
```

**Why:** exit 1 is a hard error - a crash, a missing service, a mistyped parameter - which may be transient, so Airflow retries once before paging on-call.
Here the DAG's `target` parameter names a target `profiles.yml` does not define.

**Fix:** trigger again with a defined target; the scoring DAGs use `batch`.

### Running `make ci` again logs everyone out of Woodpecker

`make ci` prints `repo mbt-showcase/churn already seeded; skipping push` and finishes; the pipeline history is all still there, but Woodpecker asks every account to log in again, and each sees Gitea's **Authorize Application** page once more.

**Why:** each `make ci` creates a fresh OAuth application in Gitea and restarts Woodpecker on it, so earlier logins belong to the old application.
Woodpecker's database lives in a volume, so pipelines, the repo's activation and its secrets survive (the `gitea_token` secret is updated to the token this run minted); Gitea is untouched.

**Fix:** none needed; log in again.
