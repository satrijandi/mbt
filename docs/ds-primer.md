# The training pipeline, explained for data scientists

This page explains how model training works here, aimed at a data scientist joining the team.
It uses the [showcase](showcase.md)'s batch-monthly churn model as the running example; the [tutorial](tutorial.md) is the full two-persona walkthrough, and the [naming conventions](naming-conventions.md) page defines the date columns used below.

## The big idea: your model is a config file, not a notebook

In this stack, a model is not "whatever my notebook produced last Tuesday."
It is a small YAML file that says: here is my data, here is my target, here are my features, here is my algorithm, here is the quality bar.
mbt reads that file and does the training for you, the same way every time.
Your notebook is still where you explore and analyze - it is just not where the model lives.
That one idea is what makes everything else possible: review, reproducibility, and automation.

## Step 1: Say what the training data is

You declare the training set, you do not code it.
For churn it looks like this: one **population table** lists who we are predicting for each month (every customer active on the 1st - that date is `inference_date`, the day the prediction is FOR).
Three **feature tables** hold what we know about those customers - demographics, login activity, transactions - and they join to the population on `inference_date` plus an id (`customer_id`, or `safe_id` for transactions).
One **label table** says who actually churned.
Two things about it matter a lot:

- The label for the June 1st cohort only exists once June is over, so the label table only contains rows for cohorts whose outcome window has CLOSED.
  That means you physically cannot train on an outcome nobody knows yet - the join just finds nothing.
  This is the main defense against label leakage, the classic mistake where information from the future sneaks into training.
- You also declare exact date ranges: train on cohorts from July 2025 through March 2026, test on April and May 2026.
  Splitting by TIME (not randomly) matters because your model will be used on future customers - testing it on a later period is the honest rehearsal of that.

## Step 2: Pick features with a repeatable recipe, not vibes

Our feature tables are wide - picture up to 2000 columns, most of them useless.
Instead of hand-picking, you run one script that does a four-stage funnel: drop columns that are nearly all missing, drop columns with only one value, drop one of each highly correlated pair, then train a small LightGBM with cross-validation and keep only features that actually got used.
The script writes the winning list INTO the model's YAML file.
So your feature selection shows up as a normal code diff that a senior DS reviews in your pull request - "why did `contract_code` come in, why did `avg_session_min` fall out" is a review conversation, not a mystery.

Some columns are banned from ever entering: the ids (they are join keys, not signals), audit columns like `loaded_at_time`, and - a real lesson from the showcase - `tenure_months`, which looks predictive in training but is anchored to calendar time, so it ALWAYS looks "drifted" in production a month later.
Those live in an `exclude:` list in the YAML, and the funnel respects it.

## Step 3: Train, with every random choice pinned

You run one command: `mbt build`.
It joins the tables, applies the split, runs your `hooks.py` (small versioned Python transforms, e.g. casting a numeric code like `contract_code` to a category), and trains - for churn, an H2O AutoML that tries a few models and keeps the best.
The spec has `seed: 42`, and every random decision in the whole pipeline (sampling, splits, the search, validation carves) is derived from that one seed.
Combined with pinned data snapshots, this means anyone can rerun your exact experiment and get the same numbers.
"It worked on my machine" stops being a sentence anyone says.

While iterating, you can train on a slice: `sample_fraction: 0.1` keeps a random-but-stable 10% of CUSTOMERS (all their months together, so time patterns survive), and you train on everything only when it is promising.

## Step 4: The model has to pass gates before it counts

After training, the model is scored on the held-out test months and checked against **gates**: the PR-AUC has to clear a floor that is meaningfully above the base churn rate (a gate below the base rate would pass a coin flip), and if a model is already in production, the challenger must beat it by a statistically defensible margin - not just a lucky decimal.
The showcase adds a stability check: the selected features' distributions are compared between the train and test windows, and if they shifted too much, promotion is blocked.
Exit codes tell you what happened: 0 means trained and passed, 2 means a quality bar said no (that is feedback for you, not a system error), 1 means something actually broke.

## Step 5: Ship by merging, not by exporting

If the gates pass, the model artifact is registered in MLflow automatically.
You open a PR with your YAML changes; CI retrains only what you touched and posts the metrics vs the production champion as a comment.
Promotion to production is itself a reviewed change.
After that, a scheduler scores the newest cohort every month with whatever model currently holds the production title, monitors watch for drift, and once labels mature the system measures how the model ACTUALLY did.
You never hand anyone a pickle file.

## What this means for your day

Explore freely in Jupyter - the showcase ships a worked notebook, `project/notebooks/ds_inner_loop.ipynb`, that walks this whole loop cell by cell - but make every real decision land in YAML.
When something fails with exit 2, read it as the system protecting you: a leaky feature, an unstable distribution, a model that is not actually better.
The pipeline is opinionated precisely so that your good ideas survive contact with production, and so the you-of-six-months-from-now can reproduce what the you-of-today did.
