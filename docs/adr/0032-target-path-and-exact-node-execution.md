# ADR-32: A private target path, a shareable dataset cache, and exact-node execution

**Status:** accepted

**Amends:** ADR-13 (the auto-materialize default stays; there is now an opt-out).

## Context

mbt has one way to run a project: one process, one `target/`, and a plan that holds every dataset a selected model needs.
That is right for a developer and for CI, and it is the wrong shape for an orchestrator that renders a project as a graph and runs one node per task.
`dbt-cosmos` is the model: it reads the manifest, makes one Airflow task per node, and wires the edges from `depends_on`.
The manifest, the selector grammar, and `--manifest` execution (ADR-19) already give mbt the first half of that.
Three properties of the core stood in the way of the second half.

1. **`target/` is one writer.**
   Every writing command holds `target/.mbt.lock` (FEEDBACK v6 B-1), and `run_results.json` and `manifest.json` are single files.
   Two tasks on one worker, or on workers sharing a volume, either refused each other or overwrote each other.
   `project_dir / "target" / ...` was spelled out at about twenty sites, so there was no one place to change.
2. **A model run builds its datasets.**
   ADR-13 adds every dataset a selected model needs to the plan.
   A model task that runs after its dataset task therefore re-enters the dataset's node: it probes the cache, and on a miss it builds.
   On a worker that does not share the dataset task's disk the probe misses and the dataset is silently rebuilt, once per model.
   An orchestrator that drew a dataset node and a model node got neither a graph that meant what it showed nor a way to say "run exactly this".
3. **A shared cache has no writer protection.**
   The dataset cache is content-addressed, so sharing it between invocations is correct by construction.
   But a build empties its directory before writing, and the only thing that had kept two builders of one key apart was the `target/` lock, which a shared cache sits outside of.

## Decisions

1. **`TargetLayout` answers "where does X live".**
   `mbt/state/layout.py` holds two roots and every well-known name under them.
   `root` holds per-invocation state: `manifest.json`, `run_results*.json`, `run_logs/`, `gate_history.json`, `docs/`, `json-schemas/`, `scoring_inputs/`, and the lock.
   `datasets` holds the dataset cache and defaults to `<root>/datasets`.
   No other module spells `"target"` any more; a test that searches the package source for it keeps that true.

2. **`--target-path` (`MBT_TARGET_PATH`) sets `root`.**
   It is accepted by every command that reads or writes it: `parse`, `compile`, `run`, `build`, `test`, `score`, `evaluate`, `monitor`, `predictions`, `docs`, `clean`.
   The lock moves with it, so the rule stays "one writer per target directory" and tasks that each own a directory do not contend.
   It follows the CLI path convention: what you type is relative to where you ran mbt, and a URI is refused (the directory must be local, though it may be a mounted shared volume).
   A path that contains the project directory is refused, because mbt rewrites what is under it.

3. **`--datasets-path` (`MBT_DATASETS_PATH`) sets `datasets` independently.**
   It is accepted by `run`, `build`, `test`, `score`, and `evaluate`.
   Two roots because the two halves have opposite sharing rules: state is private per task, and the cache is the hand-off between a dataset node and the models that read it.
   Neither flag changes any hash: a manifest is byte-identical wherever it was written, and the golden manifests do not move.

4. **`--no-auto-materialize` runs exactly the selection.**
   The planner skips the upstream-dataset pull (ADR-13) and the plan is the selected nodes and nothing else.
   A model then gets its dataset from the cache: `ExecutionContext.dataset_handle` falls back from "the handle this run's own dataset node stored" to "the cached materialization at `<datasets>/<name>/<key>/_SUCCESS`", the same probe a dataset node makes.
   Finding nothing is an error on that node (`dataset '<name>' is not materialized in <datasets-path>`), which exits 1: it is a missing prerequisite, not a quality verdict, so an orchestrator retries it and does not page a model owner.
   The check is lazy on purpose.
   `mbt test` skips a model with no gates before it ever touches a dataset, and a preflight over the plan would fail a run that needed nothing.
   `score` accepts the flag for uniformity and ignores it, because a scoring plan never pulls datasets in.
   Without the flag nothing changes.

5. **A build of a shared key is one critical section.**
   `materialization_lock` takes a per-key lock beside the key directory (not in it, because a build empties it).
   It is probe, then lock, then probe again, then build: a warm cache never takes a lock and stays readable from a read-only mount, and the loser of a race waits (saying so, once), then finds the winner's `_SUCCESS` and reads it.
   Unlike the target lock, which refuses, this one waits, because the loser needs exactly what the winner is building.

6. **`mbt clean --target-path` deletes only what mbt owns.**
   The default `target/` is deleted as before.
   A directory the user named is deleted only if it holds a `.mbt.lock`, which every mbt command leaves there; anything else is refused.
   `mbt clean --target-path ~` must not remove a home directory.

## Consequences

- **The orchestrator shape works with the CLI that exists.**
  A compile task runs `mbt compile --target-path A`, and each node task runs `mbt build --manifest A/manifest.json --select <node> --no-auto-materialize --target-path B_n --datasets-path D`.
  A provider package built on this is a thin renderer over `Manifest.graph()`.
  This ADR is the core half of FR-V1-07; the provider is not part of it.
- **Tasks share state only through the dataset cache and the registry.**
  Every task reads the same manifest, so snapshots, windows, and seeds are identical across the graph.
  That consistency is something a per-task compile could not give.
- **The cache must be a directory the tasks can all see.**
  A mounted volume works; an object-store location is refused (`--datasets-path s3://...`).
  Two workers with no shared disk still rebuild, which is correct and slow, not wrong.
  An object-store cache needs a cache-hit probe that is not a local `_SUCCESS` file, and is a follow-up.
- **`gate_history.json` counts less.**
  It is per target directory by design (D-3) and understates on a fresh runner on purpose.
  Per-task directories understate it further.
  It is a warning signal with no policy attached, so the safe direction holds.
- **`mbt clean --artifacts-older-than` keeps the latest run of one target path.**
  Its keep-set also protects every stage champion (ADR-10), so a promoted model is never at risk.
  An artifact trained under a task's own target path and not yet promoted is protected only if the same path is passed, or if it is younger than the cutoff.
- **Dataset-cache growth is nobody's job here.**
  `mbt clean` removes a target path, and a shared `--datasets-path` outside it is left alone.
  The orchestrator that created the directory (a per-run workspace, say) is the one that removes it.
- **`--manifest` still re-renders the project on every task.**
  The drift warning of ADR-19 compiles the project files against the stored anchor and compares hashes, per task.
  That is cheap next to training and unchanged here; if it matters at hundreds of nodes it deserves its own switch.
