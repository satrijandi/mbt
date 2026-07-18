# Architecture

This page is the contributor's map of the `mbt-core` engine: how one CLI command travels from YAML on disk to a registered model, and where the seams are.
It is deliberately internals-facing.
If you are *using* mbt, read [Concepts](concepts.md) and the [Spec reference](spec-reference.md) instead; if you are *writing an adapter*, read [Adapter authoring](adapter-authoring.md) - you never need anything on this page to do that.

Every load-bearing decision below has an Architecture Decision Record under [`docs/adr/`](adr/0001-arrow-interchange.md); the [ADR index by subsystem](#where-the-decisions-live) at the end maps each part of the engine to the record that justifies it.
When the code looks surprising, the ADR is the authority - read it before "fixing" the surprise.

## The shape of the system

mbt is a `uv` workspace monorepo.
One package holds the engine; every framework integration is a separate, independently versioned package that depends only on the contract layer.

```
                 mbt-core            the engine: CLI, parse, compile, DAG,
                    │                execution, quality, state, docs
                    │ re-exports
                    ▼
             mbt-adapter-base        the contract: protocols + interchange
              ▲   ▲   ▲   ▲          types + shared metric engine + the
              │   │   │   │          adapter compliance suite
   ┌──────────┘   │   │   └──────────┐
mbt-xgboost   mbt-mlflow   mbt-spark   mbt-snowflake   mbt-lightgbm
mbt-h2o       mbt-optuna   mbt-testing   ...            (adapters)
```

The one rule that keeps this honest: **`mbt-core` never imports an ML framework, and no adapter imports `mbt-core`.**
Adapters build against `mbt-adapter-base` only, and the compliance suite's `test_no_core_imports` fails any adapter that reaches into core internals.
`mbt.contracts` is a thin re-export of `mbt-adapter-base` (`packages/mbt-core/src/mbt/contracts.py`) so core code and the contract share one source of truth while the contract stays versioned on its own cadence.

| Package | Layer | Responsibility |
|---|---|---|
| `mbt-core` | engine | CLI, parsing, DAG, compile/manifest, execution engine, gates, state, docs generation |
| `mbt-adapter-base` | contract | Adapter protocols, interchange types, the shared metric engine, and the compliance test suite |
| `mbt-xgboost`, `mbt-lightgbm`, `mbt-h2o` | training adapters | Fit a model, predict, export an artifact, declare a determinism tier |
| `mbt-spark` | data + compute + training | Lakehouse datasets, `spark-submit` compute, distributed SparkML training |
| `mbt-snowflake` | data adapter | Warehouse-native datasets with push-down sampling and batch scoring |
| `mbt-mlflow` | tracking + registry | Experiment tracking and the model registry |
| `mbt-optuna` | tuning engine | Seeded TPE hyperparameter search |
| `mbt-testing` | fake adapters | Framework-free adapters so a project's specs can be tested without JVMs or GPUs |

`mbt-lightgbm` is the extensibility proof: it is built against the public contract only, with no privileged access, so anyone can ship an adapter the same way.

## One command, end to end

Every execution command (`build`, `run`, `test`, `score`) enters the same pipeline in `execute/orchestrator.py`.
The CLI layer (`cli/main.py`) only parses flags and constructs an `InvocationOptions`; the orchestrator does the work.

```
 flags ─► parse ─► compile ─────► plan ─────► schedule ─────► run_results.json
          │        │              │           │               (+ exit code)
          │        │              │           └─ ThreadPoolExecutor: threads
          │        │              │              coordinate, subprocesses compute
          │        │              └─ select the models to train; auto-join their
          │        │                 upstream datasets (ADR-13)
          │        └─ resolve windows against ONE anchor, compute both hashes,
          │           freeze the environment digest → manifest.json
          └─ YAML → rendered specs + the DAG (refs and sources become edges)
```

The functions, in order (`execute/orchestrator.py`):

1. `prepare()` - parse the project (`parsing/`), load `profiles.yml` (`config/profiles.py`), then either compile a fresh manifest (`compile/compiler.py`) or, with `--manifest`, read a stored one **verbatim** and verify the environment it runs in (ADR-19).
2. `plan_execution()` (`execute/planner.py`) - turn selectors into the set of nodes to run, plus every upstream dataset a selected model needs.
3. `execute_plan()` (`execute/scheduler.py`) - walk the DAG in parallel, calling a runner per node.
4. `run_command()` assembles a `RunResults` (`artifacts/run_results.py`) and writes `target/run_results.json`; its exit code is the command's exit code.

`mbt evaluate` and `mbt monitor` are siblings of this flow; neither trains.
`evaluate` (`run_evaluate` in the orchestrator) reuses `prepare()` and the model runner to re-score a registered artifact on fresh data.
`monitor` (`execute/monitor.py`) reuses `prepare()` and the scheduler but runs its node logic entirely in the coordinator - joining matured labels to stored predictions with DuckDB - and spawns no job subprocess at all.

## The coordinator / job split

This is the single most important design in the engine (ADR-3).
Training and scoring never run in the `mbt` process.
Each model node is executed as its own subprocess - literally `python -m mbt.execute.job <job.json>` - spawned by the compute adapter (`adapters/local/compute.py`).

The mantra is **"core compares, jobs compute."**
Only model `train`/`evaluate`/`score` nodes cross into a subprocess; dataset materialization, every quality *decision*, and all of `mbt monitor` stay in the coordinator.

| The coordinator process (`mbt ...`) | The job subprocess (`mbt.execute.job`) |
|---|---|
| Parse, compile, plan, schedule | Load hooks, resolve `AUTO` from the dataset profile |
| Resolve the champion from the registry (`ModelRunner._champion`) | Tune (never sees the test split, ADR-8) |
| Assemble the `TrainingJob` payload | Fit the final model |
| Spawn + supervise the subprocess | Predict, and compute metrics for the challenger **and** the re-loaded champion on the identical pinned test split |
| **Decide** gates (`quality/gates.py`) | Compute paired-bootstrap delta bounds (ADR-18) - but not the pass/fail |
| Register passing artifacts, transition stages | Export the artifact + the monitoring baseline (ADR-21), log to the tracker |
| Write `run_results.json`, map the exit code | Write `<job.json>.result.json`, stream events on stdout |

Why a subprocess and not a thread:

- **Isolation.** A segfault in a native training library, or an OOM, kills one job, not the whole run.
- **Real parallelism.** `--threads` maps to concurrent subprocesses, so the GIL never serializes the actual training (the coordinator's threads only orchestrate those jobs; the in-process dataset builds and checks lean on DuckDB, which releases the GIL).
- **A serialization seam.** The boundary is a JSON `TrainingJob` in and a JSON `JobResult` out. That exact seam is what the Spark adapter (and Kubernetes/Ray in v1) reuses to run the same job remotely: `parse_job_line`, `result_path_for`, and `parse_job_timeout` in `adapters/local/compute.py` are public API for that reason.

The wire contract, precisely:

- **Input:** the coordinator writes a `TrainingJob` (a Pydantic model from `mbt-adapter-base`) to `job.json`. It carries the node, the dataset locator, resolved windows, the raw (unrendered) adapter refs, the resolved metric specs, the champion artifact reference, and environment-variable **names** only. Its `mode` field selects `train`, `evaluate`, or `score` - the one payload drives all three, so "TrainingJob" is really the generic compute-job envelope.
- **Secrets never cross the wire as values.** The payload lists required env-var names; the subprocess inherits the environment and re-resolves `env_var()` itself (`execute/job.py::_render_adapter_ref`). Anything the job taints is redacted before its result or events touch disk.
- **Output:** the job writes a `JobResult` to `job.json.result.json`. The subprocess exit code is internal (`0` ok, `3` failed); the **result file is authoritative**, so a job that dies without writing one becomes a coordinator-side `error`.
- **Events:** the job emits newline-delimited JSON events on **stdout**; the compute adapter's `wait()` reads them line by line and re-emits them on the coordinator's own event bus, so a subprocess's logs interleave into the main run in real time (`compute.py::parse_job_line`).

Operational guard rails live in the same adapter: `job_timeout_seconds` arms a watchdog that SIGTERM-then-SIGKILLs an overrunning job, and `terminate()` lets the scheduler reclaim in-flight jobs when `--fail-fast` trips.

## Compilation and identity

Compilation turns rendered specs into a `Manifest` (`artifacts/manifest.py`) - the pinned, executable plan.
The reproducibility guarantees all live here.

### Two hashes

Every node carries two hashes (`compile/hashing.py`, ADR-4):

- **`config_hash`** - `sha256` of the canonical JSON of the rendered spec plus the `hooks.py` bytes.
  Excluded on purpose: the cosmetic fields `description`, `owner`, `tags`; the resolved time windows and the anchor (ADR-12); and everything from `profiles.yml` (ADR-5, so a dev/prod switch never changes identity).
- **`input_hash`** - `sha256(config_hash | snapshot_id | sorted(upstream input_hashes))`.
  It is transitive, so one comparison captures a config edit, a hook edit, a new data snapshot, or any upstream change.

`state:modified` is exactly an `input_hash` inequality against a reference manifest.

### Time is not identity

Windows are hashed as **expressions** (`"-28d:now"`), then resolved to concrete bounds against a **single anchor** captured once per compile and stored outside the hashed config (`compile/windows.py`, ADR-12).
So a `mbt build` at noon and one at midnight over the same data produce byte-identical node identities; only a new snapshot marks a node modified.

### The environment digest

Two digests pin the runtime (`compile/hashing.py`, ADR-19):

- **`env_digest`** - a targeted signal: Python version, the `mbt-*` packages, and each loaded adapter's declared fingerprint packages.
- **`env_freeze_digest`** - a pip-freeze-like hash of every installed distribution, so transitive drift (a numpy bump that shifts numerics without touching a fingerprinted package) is still visible.

`mbt run --manifest` verifies both: an `env_digest` mismatch is a hard error (`--allow-env-mismatch` downgrades it to a warning), a freeze-only mismatch always warns.

### Manifest stability

`manifest_hash()` blanks `generated_at` and `anchor` before hashing, so two compiles at the same anchor are byte-identical - the property the golden-manifest tests depend on.

## The DAG, selection, and state

The graph is a `networkx.DiGraph` built from resource dependencies (`dag/graph.py`): `ref('training_set')` records a model→dataset edge, `source('group', 'table')` records a dataset→source edge.

**Selection** (`dag/selector.py`) follows dbt semantics: `+node` (upstream), `node+` (downstream), `tag:`, `resource_type:`, `state:modified`, `--exclude`, comma for intersection and space for union.
Selection governs which **models train**; the planner then auto-joins every upstream **dataset** a selected model needs, even if unselected, because datasets are cheap materializations and CI runners start cold (`execute/planner.py`, ADR-13).
That is why a model-only PR builds green on a fresh runner.

**State** (`state/diff.py`) loads a reference manifest and compares `input_hash`es to mark nodes modified.
Environment drift alone does **not** mark nodes modified - an adapter bump would otherwise retrain everything - so `env_digest` differences are reported prominently but only act as a selector under `--state-include-env` (ADR-7).

## The execution engine

Four modules split the work of running the plan:

| Module | Role |
|---|---|
| `execute/planner.py` | Selectors + state → the `ExecutionPlan` (`selected`, `execution_set`, topological `order`). |
| `execute/scheduler.py` | Walk the DAG with a `ThreadPoolExecutor`: start a node when its in-plan parents succeed, mark transitive descendants `skipped` on failure, keep independent branches going, honor `--fail-fast`. |
| `execute/runners.py` | The coordinator side of each node type. Builds the `TrainingJob`, submits it, applies gates/checks/monitors to the result, registers. |
| `execute/job.py` | The subprocess entrypoint - all framework-heavy work (see the split above). |

The scheduler keeps at most `threads` jobs outstanding so `--fail-fast` can actually cancel not-yet-started work, and it calls back into the orchestrator's `cancel_active_jobs` to terminate running subprocesses when fail-fast trips.

There are four runners, all sharing one node-lifecycle wrapper (`run_with_lifecycle`) that emits `NodeStarted`/`NodeFinished` and times the body:

- **`DatasetRunner`** - materialize (or reuse) the dataset, then run its `checks` and Python data tests.
  Materialization is cache-aware: the key is `sha256(input_hash + resolved windows [+ sample_fraction])`, so a warm `target/datasets/<name>/<key>` with a `_SUCCESS` marker is reused, and a sampled dev build never satisfies a full build's cache probe.
- **`ModelRunner`** - resolve the champion, assemble the job, run it, call `evaluate_gates`, and register the artifact **only if every gate passes** (transitioning it to `stage_on_pass` and stamping `mbt.gates_passed=true`, plus the config/input/hooks hashes and the baseline reference, into the registry metadata).
- **`ScoringRunner`** - resolve the champion from the registry **at run time by stage alias** (so a promotion takes effect on the next scheduled run without a spec edit, ADR-5/ADR-20), verify hooks parity against the champion's `mbt.hooks_hash`, materialize the input, run input `checks` (a failure skips scoring entirely), score, then evaluate shift `monitors`.
- **`ModelTestRunner`** - `mbt test` on a model re-evaluates the latest registered version against the current champion; it **never trains** (TSD §11.3). No registered version means `skipped`, not a train.

`mbt evaluate` and `mbt test` share `ModelRunner.evaluate_artifact` and the `evaluation_node_result` assembler, so the two commands cannot drift on error handling or the metrics/gates shape.

## Quality: gates, checks, monitors

All three verdicts are pure comparisons in the coordinator, with zero ML dependencies (`quality/`); the adapter or job supplies the numbers.

- **Gates** (`quality/gates.py`) - a *threshold* gate compares the challenger metric to an absolute floor, widened by the adapter's determinism tolerance in the model's favor only.
  A *champion* gate compares against the production version re-evaluated inside the same job on the same pinned test split; with a confidence set (the default), it passes only when the **paired-bootstrap delta lower bound** clears `min_delta` (ADR-18), so a challenger that is ahead on test-set noise alone does not promote.
  No champion yet → pass with a loud warning (ADR-10). A champion that exists but cannot load → hard error, never a silent pass.
- **Checks** (`quality/checks.py`) - declarative data assertions on datasets and on scoring inputs: `schema`, `not_null`, `no_future_columns`, `class_balance_report`, and a `label_leakage_scan` that is **auto-appended to every dataset build** unless you opt out (numeric correlation or categorical association above threshold against the label fails the build). Scoring inputs skip the leakage scan - there is no label to leak.
- **Monitors** (`quality/monitors.py`) - scoring-time distribution shift (PSI or KS) of features and scores against the champion's training-time baseline, and, via `mbt monitor`, realized-metric gates once ground-truth labels mature (ADR-21).

Any quality verdict of "no" is exit code **2** and a distinct node status (`gate_failed`, `test_failed`, `monitor_failed`), kept separate from a hard `error` (exit **1**).

!!! note "Exit codes are load-bearing"
    `0` success, `1` hard error, `2` quality failure (a gate/check/test/monitor said no).
    CI depends on the distinction, and tests assert on it; the coordinator maps node statuses to these in `RunResults.exit_code()`.

## The adapter boundary

Adapters are discovered through the `mbt.adapters` entry-point group and loaded lazily (`adapters/registry.py`).
Loading a plugin imports only its cheap descriptor module - **no ML framework may import at module level** (ADR-14) - which is what keeps `mbt parse` inside its budget.
The registry checks contract compatibility on load: an adapter's contract major must equal core's, and its minor must not be newer.

A plugin (`AdapterPlugin`) bundles typed component slots, instantiated on demand via `registry.component(kind, name, config)`:

| Component | What it does | Example |
|---|---|---|
| `training` | Fit / predict / evaluate / export a model; declare a determinism tier and supported tasks | xgboost, lightgbm, h2o, spark |
| `data` | Build datasets and scoring inputs from sources; open the prediction store | local (DuckDB), snowflake, spark |
| `tracking` | Log params, metrics, artifacts, and tuning trials | mlflow |
| `registry` | Register versions, resolve champions by stage, transition stages | mlflow |
| `compute` | Run a `TrainingJob` (subprocess, `spark-submit`, cluster) | local, spark |
| `tuning` | Search hyperparameters against an objective | optuna |

Two `Supports*` capabilities are optional and probed with `hasattr` rather than declared: batch-scoring data adapters add `build_scoring_input`/`open_predictions` (contract 1.1, ADR-23), and a training adapter that sets `data_access == "path"` receives its splits as Parquet files rather than in-memory Arrow, so JVM/cluster frameworks ingest natively while still seeing exactly what Arrow adapters see (ADR-17).

The compliance suite in `mbt-adapter-base` (`TrainingAdapterCompliance`, `PredictionStoreCompliance`) is the ship bar: subclass it, keep `test_no_core_imports` green, and the adapter is correct by construction.

## Events and artifacts

**Events** (`events/`) are structured objects on a process-wide bus.
In text mode a `ConsoleSink` renders them one-per-line to **stderr** (`err_console`), keeping **stdout** clean for command *data* (`mbt ls`, `mbt show`).
In `--log-format json` a `JsonLinesSink` writes redacted JSON to stdout instead.
Job subprocesses always use the JSON-lines sink, and the coordinator forwards those lines onto its own bus - one event stream regardless of how many subprocesses ran.
Every sink redacts tainted values, so a credential in a traceback never reaches a log or a result file.

The engine writes two artifacts to `target/`, and the difference matters:

- **`manifest.json`** is the *plan*: the pinned, hashed, reproducible description of what *would* run. Deterministic at a given anchor.
- **`run_results.json`** is the *outcome*: per-node status, metrics, gate results, registrations, and timings from what *did* run.

`mbt docs generate` (`docsgen/`) reads both to produce model cards and lineage.

## Promotion

Promotion is deliberately outside node identity (ADR-5): which version is the champion is registry state, not spec.
`mbt promote` (`promote.py`) refuses to promote a version whose gates were not recorded as passed - it reads the `mbt.gates_passed` tag that `ModelRunner._register` stamps at registration time - so an artifact can only reach production through a gate it actually cleared.
Because scoring resolves the champion by stage alias at run time, a promotion takes effect on the next scheduled `mbt score` with no rebuild.

## Module map

Where to start reading, by directory under `packages/mbt-core/src/mbt/` (~11k LOC total):

| Directory | What is in it |
|---|---|
| `cli/` | The Typer app, flag parsing, the `CLIContext`, and the `_scaffold/` templates `mbt init` stamps out |
| `parsing/` | YAML + Jinja → a validated `ParsedProject` (specs, tests, sources) |
| `compile/` | `compiler.py` (build the manifest), `hashing.py` (the two hashes + env digests), `windows.py` (anchor-relative windows) |
| `dag/` | `graph.py` (the `networkx` DAG) and `selector.py` (the dbt-style selection grammar) |
| `execute/` | The engine: `orchestrator`, `planner`, `scheduler`, `runners` (coordinator side), `job` (subprocess), `monitor` |
| `quality/` | Pure comparison modules: `gates`, `checks`, `monitors`, `metrics`, `python_tests`, `hooks` |
| `adapters/` | `registry.py` (plugin discovery/versioning) and the built-in `local` adapter (DuckDB data, subprocess compute) |
| `config/` | `profiles.py`, `project.py`, and the `tasks/` schemas (binary classification, regression) |
| `artifacts/` | The `Manifest` (plan) and `RunResults` (outcome) models |
| `events/` | The event bus, the typed event models, and the console / JSON-lines sinks |
| `state/` | `diff.py` - `input_hash` comparison that powers `state:modified` |
| `docsgen/` | Model cards and lineage for `mbt docs generate` |

The interchange types these modules pass around (`ManifestNode`, `TrainingJob`, `JobResult`, the specs, the adapter protocols) are defined in `mbt-adapter-base`, not here; `mbt.contracts` re-exports them.

## Where the decisions live

The engine is best read alongside the ADRs that shaped it.
Start with the decision, not the code:

| Subsystem | Read |
|---|---|
| Adapter boundary & interchange | ADR-1 (Arrow), ADR-2 (local adapters in core), ADR-14 (import hygiene), ADR-15 (contract refinements), ADR-17 (JVM adapters, path data access) |
| Coordinator / job split | ADR-3 |
| Identity & reproducibility | ADR-4 (two hashes), ADR-5 (profiles excluded), ADR-12 (windows & anchor), ADR-19 (env digest & `--manifest` verification) |
| Selection, state & datasets | ADR-7 (env not modifying), ADR-11 (snapshot listing), ADR-13 (datasets auto-materialize), ADR-16 (multi-table inputs & key sampling), ADR-22 (population spine & per-table joins) |
| Gates & tuning | ADR-6 (gate edits retrain), ADR-8 (tuning never sees test), ADR-9 (champion re-evaluated in job), ADR-10 (missing vs unloadable champion), ADR-18 (paired-bootstrap gates) |
| Scoring & monitoring | ADR-20 (scoring resource & runtime champion), ADR-21 (prediction store & ground-truth ledger), ADR-23 (warehouse batch scoring) |
| Task verticals | ADR-24 (regression as a second vertical) |

The pre-implementation sketches under `design-history/` (`PRD.md` and `TSD.md`) are kept as glossaries for the `FR-*`/`NFR-*` requirement IDs and the `TSD §N` anchors still cited throughout the code; the ADRs supersede their design decisions (ADR-15 explicitly supersedes the `TSD.md` sketch).
