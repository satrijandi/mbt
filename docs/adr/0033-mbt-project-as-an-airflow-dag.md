# ADR-33: An mbt project is an Airflow DAG: one task per node, drawn from the manifest, run through the CLI

**Status:** accepted

**Builds on:** ADR-32 (a private target path, a shareable dataset cache, exact-node execution), ADR-19 (`--manifest` execution), ADR-13 (upstream datasets auto-materialize), ADR-20 (the champion resolves at run time).

## Context

`dbt-cosmos` turns a dbt project into an Airflow DAG: it reads the manifest, makes one task per node, and wires the edges from `depends_on`.
mbt had the pieces for that (a manifest with edges, the selector grammar, `--manifest` execution) and, since ADR-32, a CLI that can run exactly one node against private state.
What it did not have was the package that does the drawing.
The showcase ran each command as a single Airflow task, so a retrain was one opaque box: no per-model state, no per-model retry, no graph.

A prototype (`examples/prototype-mbt-airflow`, throwaway) drew `churn_demo` four ways and ran it under Airflow's `dag.test()` through the real CLI.
This ADR records what that settled, and `mbt-airflow` is the production form of it.

## Decisions

1. **The tasks run the CLI; there is no in-process API.**
   Each task starts `mbt` as a subprocess and maps its exit code to an Airflow failure mode.
   That is the one code path every other caller uses, it keeps the heavy execution environment (XGBoost, MLflow) off the scheduler, and it is the coordinator/job split of ADR-3 applied one level up.
   The package supports local execution (`mbt` is an executable on the worker).
   The argument vector is built apart from how it is run, so container execution is a later addition and not a rewrite.

2. **The DAG is drawn by mbt's own planner, from a manifest compiled ahead of time.**
   `plan_tasks` calls `plan_execution`, so `select` and `exclude` are the CLI's selector grammar, and a model selection is drawn with the datasets it needs (ADR-13).
   That closure is required, not a convenience: every node task runs `--no-auto-materialize` (ADR-32), so a model whose dataset is not a task of its own would read an empty cache.
   `tag:weekly` selects two models and the DAG has five tasks, counting the compile root.
   A reimplemented selector would drift from the CLI, which is why `mbt-core` is a dependency of the package.
   It is importable where DAGs are parsed: it costs about 0.15 seconds warm and brings pyarrow, duckdb and networkx.
   It installed beside Airflow 3.3.1 under Airflow's published constraints, and resolves beside 3.4.0 in the workspace lock, with no conflict in either.

3. **A DAG is one command's.**
   `build`, `run` and `test` draw datasets and models; `score` and `monitor` draw scoring nodes.
   This is mbt's partition (the planner's `executable=`), not a new rule.
   It matters because the manifest has an edge from a scoring node to its model, while scoring reads the registry's champion at run time and runs on its own cadence (ADR-20).
   Drawing that edge would put every daily score behind a weekly retrain.
   Keeping only the edges inside the plan drops it with no special case.
   A selection that matches nodes the command never runs (`tag:daily` under `build`) is an error, because a DAG of one compile task would pass green having run nothing.

4. **Compile once per run, then execute that manifest verbatim.**
   The graph is drawn from a manifest compiled ahead of time and used only for its shape.
   The first task compiles again at the run's own anchor, and every node task runs `--manifest` on that file, so snapshots, windows, seeds and cache keys agree across the run, and ADR-19 verifies the environment on each task.
   `--vars` goes on the compile task only, because the manifest carries them: the dataset task and the model task then compute the same cache key by construction.
   The anchor is the `anchor` param if set, else the end of the data interval, else the logical date, else the creation time of the run.

5. **A drawn DAG refuses to run against a project it no longer describes.**
   If a model gained the selected tag after the DAG was drawn, the new model would never run behind a green DAG.
   The compile task re-plans the fresh manifest and fails, without a retry, when the task ids differ, naming the nodes on each side.

6. **Exit codes keep their meaning.**
   Exit 2 (a gate, check, test or monitor said no) is `AirflowFailException`: it is deterministic, so it fails at once and goes to the model owner.
   Any other nonzero exit is a retryable `AirflowException` that quotes the tail of mbt's stderr, because a hard error has no per-node result to read.
   A missing executable also fails at once.
   A task's outcome is read from its own `run_results.json`, which is unambiguous only because each task owns its target directory (ADR-32), so the failure message carries the gate detail.

7. **One workspace per run, and cleanup only on success.**
   A run's tasks keep their private target directories and the one shared dataset cache under a directory named for the DAG and the run.
   It must be visible to every worker, so with more than one machine it is a shared volume (ADR-32 refuses an object-store cache).
   An optional final task removes it, but only when every node succeeded, so a failed run keeps its logs and results.
   It deletes a path built from a template, so it refuses anything that is not strictly inside the workspace root.

8. **Promotion is not a task.**
   It stays a reviewed GitOps step between the build DAG and the score DAG.

## Consequences

- **`apache-airflow` is a dependency of the package** (`>=3.3.1,<4`, the `airflow.sdk` authoring surface), and the dev lock grew by 57 packages.
  The one existing package that moved was `fastapi`, because Airflow's API server needs a newer one.
- **The floor is measured, and Airflow's tree is kept out of the floors job.**
  Airflow supports an install only under its own published constraints, and 3.1.x and 3.2.x constrain `click==8.3.1`, which cannot install beside mbt-core (`click>=8.3.3`, a security floor).
  An unconstrained 3.1.0 does not even import against current `structlog`, and `AirflowFailException` is only in `airflow.sdk.exceptions` from 3.2.
  So the floor is 3.3.1, the oldest release tested, by a CI job that installs it under its own constraints.
  The repo's floors job does not install the package at all, because Airflow's tree lifts `pydantic` and `packaging` above the floors mbt-core declares, which would stop that job testing them.
- **A real executor with parallel branches is not exercised.**
  The end-to-end test uses `dag.test()`, which runs tasks one at a time.
  Two model tasks writing one sqlite MLflow database at once is the first thing a deployment will meet, and it needs a tracking server, not a sqlite file.
- **`state:` selectors cannot draw a DAG.**
  A DAG is drawn before any run, so there is no reference state to compare against, and the planner says so.
- **The DAG's shape can be stale between publishes of the manifest.**
  Decision 5 turns that from a silent miss into a failure, but the manifest still has to be republished when the project's nodes change.
- **Every task re-renders the project for ADR-19's drift warning.**
  That is cheap next to training and unchanged here.
- **It found DuckDB 2.0 breaking mbt-core.**
  Installing under Airflow's constraints selects `duckdb 2.0.0`, where `execute(...).fetchone()` leaves a result open that the next query on the connection refuses to run beside.
  Several checks reuse a connection per split, so they failed on real projects, not only in the two tests the nightly run named (issue #32).
  Every such read now goes through `first_row`, and a test enforces the 2.0 rule on whatever DuckDB is installed.
- **The showcase's own DAGs still run one task per command.**
  Moving them onto `mbt-airflow` is a separate change with its own live verification.
