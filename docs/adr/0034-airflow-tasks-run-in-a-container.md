# ADR-34: An Airflow task can run mbt in a container: a runner seam, files read back through it, and an anchor the project names

**Status:** accepted

**Builds on:** ADR-33 (an mbt project is an Airflow DAG), ADR-32 (a private target path per task), ADR-19 (`--manifest` execution verifies the environment that compiled it).

## Context

ADR-33 ran every task as a subprocess on the worker and said container execution was "a later addition and not a rewrite".
The showcase is the case that needed it, and it is a fair test of the claim.
Its three DAGs start the deployable unit through the Docker API, because the unit is the code that runs: an image pinned by digest in the deploy repo, so that bumping the pin (or reverting it) is the deploy.
Moving those DAGs onto `mbt-airflow` asked the package for four things it did not have:

- a task that runs in a container, not on the worker;
- an anchor read from the data inside that container (the lake's newest cohort), not from the clock of the Airflow run;
- a way for a wrapper around `mbt` to find its task's results, so that the showcase can push each node's metrics to its gateway;
- a manifest to draw the graph from, which is the showcase's to publish and not the package's.

This ADR decides the first three.
The fourth is the deploy repo's, and it is described with the showcase.

## Decisions

1. **A runner is the seam.**
   An operator builds the argument vector and decides what its exit code means (ADR-33, decision 6).
   A runner only starts that vector somewhere, and hands back the exit code, the tail of stderr, and the files the operator asked for.
   `LocalRunner` is the subprocess the package always had; `DockerRunner` is new.
   Anything that decides a failure mode stays in the operator, so a container task fails exactly as a local one does: exit 2 is `AirflowFailException` with no retry, any other nonzero exit is retried and quotes mbt's own last lines.
   A third place to run (a Kubernetes pod) is one more class, not a change to an operator.

2. **A container is one process, and it inherits nothing.**
   `MbtDocker(unit)` takes a `DockerUnit` (image, network, volumes, working directory, environment) or a function that returns one.
   A function is called when the task runs and not when the DAG file is parsed: the deploy repo's pin can change between the two, and a missing pin should fail the task with the owner's own message ("no unit pinned yet") rather than at import.
   The worker's environment is not passed on, because a container starts clean and a secret that reached it by accident is worse than one that is missing; everything it needs is named in `DockerUnit.env`.
   The Docker client is built with `timeout=600` by default, because docker-py's 60 seconds is shorter than a container create can take on a busy daemon, and the symptom was a task that failed before mbt ran.

3. **Files come back through the runner.**
   The workspace of a containerised run is a path in the container, on a volume the unit mounts.
   The scheduler that launched the container usually has no mount of its own, and giving it one means its compose file must know a host path that is only known after the stack is configured.
   So the runner reads what the operator needs (a node's `run_results.json`, the compile task's `manifest.json`) out of the stopped container with the archive API, before it removes the container.
   The same applies to cleanup: the workspace is removed by running `rm -rf` in the unit, after the same strictly-inside-the-root check, which compares the container's paths as text because the worker cannot resolve them.
   `workspace_root` has no default with `docker`, because the worker's temp directory is a place the container cannot see.

4. **Output arrives when the container exits.**
   A stream of the container's logs holds one HTTP read open, and a quiet stretch of a long training (an AutoML search that logs nothing for ten minutes) would reach the client's read timeout and lose the stream.
   Polling with `since` can repeat or drop lines at the second boundary.
   So the task log of a container task is written when the container ends, which is what the showcase's tasks always did, and `on_kill` stops the container (SIGTERM, then SIGKILL after 30 seconds) instead of leaving it training after Airflow gave up on the task.
   Live streaming is a refinement for when a deployment needs it.

5. **A project can name its own anchor.**
   The anchor of a run is the `anchor` param when set (a pin always wins), else what an `anchor_command` prints, else the run's own time (ADR-33, decision 4).
   The command runs through the same runner as mbt, so it sees what mbt sees (a command that lists a lake's cohort files does so with the unit's credentials), and it must print exactly one anchor.
   Only the compile task asks, once per run, and every node task then runs that manifest, so the anchor cannot differ between tasks.
   A failing command is a hard error and is retried, because the usual cause is infrastructure; a command that prints something other than one anchor fails at once, because retrying cannot change what it prints.

6. **A task tells the process around `mbt` where it is.**
   Every mbt process gets `MBT_AIRFLOW_TARGET_PATH` (the task's private target directory) and, for a node task, `MBT_AIRFLOW_MANIFEST` (the run's manifest).
   They are set after the configured environment, so it cannot shadow them.
   A wrapper script given as `mbt_executable` can then act on the task's results without parsing the argument vector: one that pushes each node's metrics to a gateway after `mbt` exits, then re-exits with mbt's code, is the case that asked for this.

7. **The Docker SDK is an extra.**
   `pip install 'mbt-airflow[docker]'` brings it, and it is imported only when a container is started, so a worker that runs mbt locally never needs it and an install without it says how to get it.

## Consequences

- **The unit tests fake the SDK; a real daemon checks the rest.**
  The tests record what the runner asks of a fake client, which covers every branch but cannot say what Docker does with the calls.
  The calls were run once against a real daemon (exit code, stderr kept apart from stdout, the environment merged, a file read back from a stopped container's bind mount, a missing file, `rm -rf` through the unit), and the showcase's live tier, which drives its Airflow through a real stack, is where the container path is exercised end to end.
- **The worker needs the Docker socket.**
  That is root on the host, as it already was for the showcase's scheduler.
  A deployment that cannot grant it wants a pod runner, which this seam makes a small addition.
- **Parallel branches need the shared volume.**
  The unit's volume is what carries the dataset cache from the dataset task to the model task, so on more than one Docker host it must be a volume all of them mount.
  ADR-32 still refuses an object-store cache.
- **The pinned image is the code that runs, and the DAG is drawn from a manifest.**
  They must move together, or a DAG is drawn for one project and runs another.
  The stale-DAG check of ADR-33 (decision 5) fails the run when they have drifted; publishing the manifest in the same commit as the image pin is what keeps them from drifting.
