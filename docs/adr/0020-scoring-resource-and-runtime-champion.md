# ADR-20: Scoring pipelines are a resource kind; champions resolve at run time

**Status:** accepted

Batch scoring was a sanctioned v1 direction (the roadmap's `mbt score`), while online serving stays a non-goal: mbt's serving surface terminates like a job, never as a long-running prediction service.
One `scoring` YAML config declares one batch serving pipeline end to end: which model's champion to load, what data to score, where predictions land, and what to monitor.
Only `mbt score` executes scoring nodes; `run`/`build`/`test` keep their training semantics, so training cadence and serving cadence never couple.

The champion is resolved from the registry by stage alias at RUN time, not pinned in the manifest.
Registry state is deliberately excluded from node identity (ADR-5), so a promotion is not an `input_hash` event and never marks a scoring node modified; scheduled runs simply pick up the new champion on their next invocation.
The resolved version is recorded in run_results, the tracking tags, and the prediction sidecar, so every scored row remains attributable.
The alternative, pinning the champion version into the manifest, would serve stale models after every promotion until the next compile and was rejected as violating registry-as-source-of-truth.

Scoring inputs are unlabeled and unsplit by design: they materialize as a single `score` split, and label-dependent checks are rejected at parse time.
Input identity follows dataset rules exactly: source snapshots and window expressions enter `config_hash`/`input_hash` (ADR-11/ADR-12 parity), so re-scoring is snapshot-driven, never clock-driven.
The ground-truth label table is lineage but NOT identity: its snapshot is pinned on the manifest's sources for observability, yet excluded from the scoring node's `snapshot_id`, because labels maturing later must never flag the scorer as modified.

Feature-transform parity is enforced, not assumed: registration stamps `mbt.hooks_hash`, and a scoring run hard-fails when the current project's hooks differ from what the champion was trained with (a missing tag on pre-feature champions warns, in the ADR-10 spirit).
Hooks must be row-stable for scoring (no filtering or reordering); the job enforces the row count and errors otherwise.

Distribution monitoring extends "jobs compute, core compares" (ADR-3): the scoring job emits shift statistics, and the coordinator applies the declared thresholds.
Vocabulary: this codebase reserves "drift" for data-snapshot drift (ADR-11); distribution monitoring is called "shift" (`feature_shift`, `prediction_shift`).
A breach sets the new `monitor_failed` node status, which maps to exit code 2 exactly like gate and test failures, so existing CI alerting fires unchanged.

Scoring as a dataset flavor was rejected because `label` and `split` are load-bearing there; a separate kind keeps both schemas honest.
