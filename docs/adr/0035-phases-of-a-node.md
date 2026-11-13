# ADR-35: A node has phases: `mbt run --phase` runs one step of it, so an orchestrator can give each step its own task

**Status:** accepted

**Builds on:** ADR-32 (a private target path, a shareable dataset cache, exact-node execution), ADR-33 (an mbt project is an Airflow DAG), ADR-34 (an Airflow task can run mbt in a container), ADR-18 (champion gates), ADR-30 (the training report and tracking tags).

## Context

ADR-33 drew one task per node.
That is the right grain for the graph between nodes, and the wrong one inside a model: a `model` node is a training job that can run for hours, followed by gates and a registry write that take seconds.
As one task, a failed MLflow call after a two-hour training made Airflow retry the node, which meant training again.
A failed gate and a crashed trainer were the same red box.
And a GPU pool or queue could only be given to the whole node, including the part that needs no GPU.

The pipeline a practitioner draws is data collection, preprocessing, training, writing to the registry.
Those names do not all correspond to a seam in mbt, so this ADR starts from the seams that are there.

A **dataset** is built by one call to the data adapter (`build_dataset`): it reads the source, windows and splits it, projects the columns, and caches the result.
Collecting and preprocessing are one recipe inside that call, with no intermediate form to cache, so they are not separable without a new data-adapter contract.
After the build come the declared checks.

A **model** is a training job in a subprocess (it trains, and evaluates on the test split, and logs the artifact to tracking), then, in the coordinator, the gates (threshold, champion, stability), then the registry write and the tracking tags.
Feature preprocessing happens inside the training job, through the model's hooks.

## Decisions

1. **The phases are the seams that exist.**
   A dataset has `materialize` and `checks`; a model has `train`, `gate` and `register`.
   `materialize` is the first half of the dataset runner as it already was; `train` ends where the job returns its result; `gate` is `judge`; `register` is the registry write, the tags, and the log upload.
   Splitting collection from preprocessing is deliberately not offered: it needs an intermediate dataset form in every data adapter, and would pay off only where raw extraction is expensive enough to cache by itself.

2. **`--phase` on `run` and `build`, exact-node, and strict.**
   `mbt run --phase train --select churn_classifier` runs that step of that node and nothing else, so it implies `--no-auto-materialize` (ADR-32): a step is by definition not the whole node, and must not pull in what the node would otherwise have built.
   Every selected node must have the phase.
   `--phase train` over a selection that includes a dataset is refused before any work starts, because skipping the dataset would let an orchestrator believe it had been built.
   `test`, `score` and `monitor` have one step per node and refuse `--phase`.
   With no `--phase`, a node runs all of its steps in one process, in order, as it always did.

3. **The steps hand their work over through two things that already exist.**
   A dataset's steps share the dataset cache: `materialize` writes it, `checks` reads it and builds nothing, and a missing materialization is the same exit-1 error as for a model (ADR-32).
   A model's steps share a small state file under the node's `--target-path`, `phases/<model>.json`: `train` writes the job result and the version of the champion it compared against, `gate` adds its judgement, and `register` adds the version and the registration.
   The state carries the manifest hash and the node, and a step refuses a state from a different plan.
   So the steps of one model belong to one target path.
   They never run at once (each waits for the one before it), which is what the target lock guarantees, so ADR-32's one writer per directory still holds.

4. **The whole node is the three steps composed in memory.**
   The model runner is `train`, `judge`, `register` and `finish` over one state object; run whole, the state is never written, and the order of every side effect (gate history, registration, tags, log upload) is the order it had.
   The same tests that covered the whole node cover the steps, and `test_phase_execution.py` asserts that the steps in sequence leave the metrics, gates and registration the whole run leaves.

5. **The champion a gate judges against is the one the job saw.**
   The training job computes the paired-bootstrap comparison against the champion resolved when it started (ADR-18), and the gate labels its result with that champion's version.
   `train` records the version, and `gate` uses it, so the verdict describes the comparison actually made even if a promotion lands between the two steps.
   This is the same thing that was true inside one process, just with a gap in it.

6. **The exit codes stay put, and now point at a step.**
   `gate` exits 2 when a gate says no, and nothing else does; every other failure of every step is 1.
   `register` after a failed gate exits 1 (there is nothing to register), so a retry loop cannot register a model that was judged unfit.
   An orchestrator can therefore route on the task: a red `gate` goes to the model's owner, a red `train` or `register` is infrastructure.

7. **`register` is idempotent once the registry has the version.**
   The version is written to the state the moment the registry returns it, before the stage is set; a retry that finds it sets the stage on that version and does not register the artifact again.
   A second `register` after a completed one returns the stored registration.
   One window is left, and is named here rather than hidden: a process killed after the registry accepted the artifact and before the state was written registers it twice on retry.
   The registry protocol has no lookup by tracking run, so closing the window means widening that protocol, which this change does not.

8. **`mbt-airflow` draws the steps.**
   `plan_tasks(..., phases=True)` (the default) makes a node of `build` or `run` a group of tasks, `<node>.<phase>`, in the order above.
   A node waits for the last step of each node it reads, so a model trains only once its dataset's checks have passed, and the first step of a root waits for the compile task.
   All of a node's steps run against one target directory, and each reads its own result from `run_results.json`, which the next step then overwrites.
   `phases=False` draws one task per node, as ADR-33 did.

## Consequences

- **A failed registration retries in seconds.**
  This was the motivating cost.
  The same holds for a failed gate step (re-run it after fixing the gate threshold or the champion) and for `checks` (re-run it without rebuilding the dataset).
- **The graph has more tasks and the same shape.**
  A model is three and a dataset is two, grouped under the node in Airflow's graph view.
  `churn_demo`'s weekly DAG goes from five tasks to eleven.
- **`run_results.json` describes one step.**
  After `train` a model has metrics and an artifact and no gates; `metadata.phase` says which step the file is from, so a half-finished node is not mistaken for a finished one.
- **A model's time is a running total.**
  Each step's result reports the node's time so far (kept in the state file), because a dashboard reads the last result a node pushed and the last step is the registry write.
  The events still carry each step's own time, and a retried step adds its time again, which is what the node cost.
- **The gate-history counter counts evaluations, so a retried `gate` counts twice.**
  It only drives the reuse warning (D-3), and over-counting errs toward warning.
- **Preprocessing stays inside the training job.**
  A DAG cannot show it as a task, and a failure in a hook is a failure of `train`.
- **An mbt older than this one cannot run these DAGs.**
  `--phase` is part of the CLI the tasks call, so the image a DAG runs and the `mbt-airflow` that drew it are released together, as they already had to be for ADR-33's manifest.
