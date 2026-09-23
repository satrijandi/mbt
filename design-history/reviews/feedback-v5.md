# FEEDBACK v5: an architecture review of mbt

Review date: 2026-09-20, against `main` (`f1433bc`).
Scope: whole tree, read for module depth rather than for practice compliance.
The four earlier cycles (`design-history/reviews/feedback-v1.md` through `-v4`) reviewed MLOps, DevSecOps, GitOps, data engineering and data science practice; none of them reviewed the shape of the code itself.
Findings closed in those cycles are not re-litigated.

This file sits at the repo root while the sweep is in flight, per `design-history/README.md`.
It moves to `design-history/reviews/feedback-v5.md` when the progress log at the bottom closes.

## What this review is

It asks one question of each part of the tree: **is this a deep module, or a shallow one?**
A module is deep when a lot of behaviour sits behind a small interface, and shallow when the interface is nearly as large as the implementation.
"Interface" here means everything a caller must know to use the module correctly, which includes invariants, ordering obligations and error modes, not just the type signature.
A seam is the place where that interface lives.
One adapter behind a seam makes it hypothetical; two make it real.

The repo has no `CONTEXT.md`, so the domain nouns below come from `docs/concepts.md`: source, dataset, model, scoring pipeline, champion, gate, manifest, coordinator, job.

## What this review is not

It is not a defect sweep, and it is not a request to restructure the tree.
Most of what follows is friction that has not yet cost anything.
Three findings are live defects and are marked as such.

It is also not a criticism of the decisions in `docs/adr/`.
Two findings touch an accepted ADR, and both say so in place rather than proposing around it.

## Method, and what is measured

Every count in this document was measured on this machine at review time, not estimated.
Where a claim is a count, the command that produced it is reproducible from the file references given.
The architecture walk was done by three agents reading `execute/`, the adapter seam, and the CLI and parsing front half; every headline claim was then re-checked directly before being written here, and two reported figures were corrected downward in the process (adapter duplication from 59% to a measured 50%, and `training_report`'s reaches into `_JobRuntime` from 38 occurrences to 10 distinct fields).

Scoping followed churn.
`parsing/project_parser.py` and `cli/main.py` are the two most-edited source files of the last 40 commits (7 and 6).
Seven of the last 100 commits changed two or more data adapters in lockstep.

**No verification battery was run for this document**, because it changes no code.
The tree was clean at `f1433bc` when the review started and is clean now apart from this file.

## The three live defects

These are not architectural preferences.
They are wrong today, and each one is a consequence of a shallow seam rather than a local mistake.

A fourth live defect, **D-1**, is recorded in the data-science addendum below.
It is kept separate because it is not a consequence of a seam: the seam around it is fine, and the statistics passing through it are not.

### 1. An empty `out_of_time` split has two different severities

The local adapter emits the warning as a typed event; Spark and Snowflake emit a bare string, which the bus wraps at the default level.

```
packages/mbt-core/src/mbt/adapters/local/data.py:193   LogMessage(level="warn", unique_id=...)
packages/mbt-spark/src/mbt_spark/data.py:300           emit(<str>)  -> level "info", unique_id None
packages/mbt-snowflake/src/mbt_snowflake/adapter.py:321  emit(<str>)  -> level "info", unique_id None
```

`mbt/events/models.py:25` defaults `level` to `"info"`, and `mbt/events/bus.py:30-35` wraps any non-`Event` object in a bare `LogMessage`.
So the same condition, on the same project, is a warning or an informational line depending only on which data adapter ran.

The three tests that cover the condition cannot catch it.
`packages/mbt-core/tests/test_out_of_time_split.py:270` asserts `level == "warn"`; `packages/mbt-spark/tests/test_spark_adapters.py:172` asserts a substring of `str(m)`, which cannot see the level at all.

### 2. `spec.seed + 3` is taken twice

ADR-18 names `+3` as the champion-gate bootstrap rung.
`packages/mbt-core/src/mbt/execute/job.py:536` uses it there, with the full ladder written out as a comment.
`packages/mbt-core/src/mbt/execute/job.py:721` uses the same rung for the tuning-time robust objective, with no comment and no reference to the other site.

Whether the reuse is deliberate cannot be determined from either call site.
Today's test proves that two rungs differ; nothing proves the set is collision-free.

### 3. `mbt ls` cannot use a selector it documents

Reproduced through the real CLI against `tests/fixtures/churn_demo`:

```
$ mbt ls --select state:modified
Error: selector 'state:modified' requires --state <path-or-URI> pointing at a reference manifest
$ mbt ls --state foo.json
Error: No such option: --state
```

`mbt ls` hand-rolls its node view instead of reusing `Manifest.selectable_nodes()`, and never passes `state` to the selector.
Its own docstring promises "the same selectors --select accepts", and `docs/cli-reference.md:89` repeats that promise with `state:new | state:modified` in the grammar.

## Findings

| # | Finding | Strength | Size |
|---|---|---|---|
| A-1 | The dataset build recipe sits above the data seam | Strong | M |
| A-2 | The CLI has no composition root, and the copies have drifted | Strong | M |
| A-3 | The parser's 19 rules are wired into a transcript nothing can address | Strong | M |
| A-4 | The training adapters re-type each other | Strong | M |
| A-5 | The champion record is 53 string keys with no owner | Strong | M |
| B-1 | Capability is dispatched by `hasattr`, not by the declared protocols | Worth exploring | S |
| B-2 | `training_report.py` is a file split of `job.py`, not a seam | Worth exploring | M |
| B-3 | The seed ladder has no home | Worth exploring | S |
| B-4 | The event seam accepts `object` | Worth exploring | M |
| C-1 | Nothing owns "judge this job result" | Smaller | S |
| C-2 | `feature_columns` is a return value smuggled through state | Smaller | S |
| C-3 | `oot_check` drives `ModelRunner` through four private methods | Smaller | S |
| C-4 | `ReportData.tables` is a string-keyed namespace with a silent fallback | Smaller | S |
| C-5 | `AdapterPlugin.task_schemas` is dead machinery | Smaller | XS |
| C-6 | `mbt.contracts` is a pass-through | Smaller | S |

---

## A-1. The dataset build recipe sits above the data seam

**Files.** `packages/mbt-adapter-base/src/mbt_adapter_base/materialization.py`, `.../protocols.py:286-309` and `:421-461`, `packages/mbt-core/src/mbt/adapters/local/data.py:163-462`, `packages/mbt-spark/src/mbt_spark/data.py:277-420`, `packages/mbt-snowflake/src/mbt_snowflake/adapter.py:291-429`.

**Symptom.** `build_dataset` is one fixed eleven-step recipe, and all three data adapters write it out.
Only four steps are genuinely per-engine: read the relation, apply filters, express the sampling predicate, write the splits.
The other seven are policy, and they are identical.

The prologue is the same five lines in all three:

```python
self._verify_snapshot(ctx)
output_dir = ctx.output_dir
output_dir.mkdir(parents=True, exist_ok=True)
for stale in output_dir.glob("*"):
    stale.unlink()
```

So is the epilogue: the zero-row loop with its `OUT_OF_TIME_SPLIT` exemption, the row-count event, and `write_materialization_metadata`.
The Spark and Snowflake sources say so in comments: "mirrors the local and snowflake adapters", "mirrors the local adapter".
That is a contract held in prose.

The bucket-edge arithmetic is the part that matters most, because it decides which rows train.
It is written three times: `local/data.py:441-449`, `mbt_snowflake/sql.py:138-146`, `mbt_spark/data.py:379-390`.
Its reference implementation is then rebuilt in three separate test files, and the local one's docstring admits the arrangement: "each adapter's tests pin their SQL to this one reference".

**Why it costs.** ADR-30 added exactly one new split concept, `split.out_of_time`.
Landing it required editing all three adapters: `+24`, `+46`, `+17` lines (commit `f354b88`).
Live defect 1 above is the part of that edit that came out inconsistent.

**Two more symptoms of the same seam.**
`count_source_duplicates` and `read_source_distinct` are implemented by all three data adapters and declared on no protocol.
Core reaches them through `getattr` at `quality/checks.py:306` and `:712`, raising a hand-written "does not support source-level checks" when absent.
An adapter author reading `protocols.py` cannot learn that a `DataAdapter` needs them.

There is also no `DataAdapterCompliance`.
`compliance/suite.py` ships exactly two base classes, for training adapters and prediction stores.
`mbt-testing` ships fakes for the training, tracking, registry, tuning, reporting and compute seams, and none for the data seam, so every dataset test runs real DuckDB.

**Deletion test.** Delete `materialization.py` today and complexity does not concentrate, it re-scatters into three adapters.
It holds the data side of the contract (the handle, the metadata writer, two helpers) but not the control flow, so it is a bag of helpers with ordering obligations rather than a module.

**Shape of the fix.** Move the recipe into `materialization.py` behind one call, and leave each adapter a four-method engine interface.
Add a `DataAdapterCompliance` that asserts bucket membership against one reference digest, and declare the two source-check methods on the protocol.
An in-memory engine then becomes cheap, which gives the seam the second adapter its tests have never had.

**Precedent.** This is the same move as commit `8ea7b6d`, "Delete the join: 2,300 lines that three adapters each had to reimplement", one layer down.

---

## A-2. The CLI has no composition root, and the copies have drifted

**Files.** `packages/mbt-core/src/mbt/cli/main.py` (1,191 lines, 6 of the last 40 commits), `packages/mbt-core/src/mbt/cli/common.py:180`, `packages/mbt-core/src/mbt/execute/orchestrator.py:107-148`.

**Symptom.** The work lives in the command body, so there is no callable "do the thing".
`make_ctx(project_dir, profiles_dir, target, vars_, log_format, quiet, verbose)` appears at 17 call sites with seven positional arguments.
`parse_project(cli.project_dir, cli_vars=cli.cli_vars)` appears at 10.
`cli.profiles(parsed)` appears at 8.

The same three-step composition is retyped four times and has drifted:

| command | prints parse warnings | anchor | pins snapshots |
|---|---|---|---|
| `compile` | yes | from `CompileOptions` | yes |
| `show` | no | `now()` | yes, live |
| `state diff` | no | from `CompileOptions` | yes |
| `docs generate` | no | `now()` | yes, live |

`print_warnings` exists at exactly two call sites, `main.py:424` and `:456`.
So `mbt show` and `mbt docs generate` silently drop the split-protocol leakage warnings the parser produces.

Both also compile with no `CompileOptions`, which means they re-anchor to `now()` and pin snapshots against the live data system in order to print one spec.
Neither has an `--anchor` flag, so `mbt show`'s resolved windows drift from `target/manifest.json` with wall-clock time.
ADR-12 exists to pin exactly that, and `orchestrator.py:141-148` already documents a read-only path that avoids it, serving `predictions ls/show`.
`show` and `docs generate` route around it.

**Three commands skip `make_ctx` entirely.**
The one that matters is `clean`, whose default branch calls `shutil.rmtree` on a raw, unresolved typer path with no context and no event bus (`main.py:313-317`).
`make_ctx` raises `ConfigError` on a non-directory; this branch prints `nothing to clean` instead.
All four `clean` tests pass an absolute `--project-dir`, so the relative-path divergence is untested.

Live defect 3 above is the third of these.

**Deletion test.** `main.py` fails as a CLI adapter and passes as a module, which is the problem.
Deleting it loses real behaviour that exists nowhere else: `ls`'s node view and four output formats, `show`'s resource lookup and redaction, `promote --from-file`, `clean`'s GC orchestration.
None of that is typer plumbing, and none of it is reachable except through the CLI runner.

**Test surface today.** 32 of the 42 tests in `test_cli_main_unit.py` drive `runner.invoke`.
The 10 that do not test `main()`'s exception remapping and `setup_bus`, never a command body.
The extracted helpers in `cli/common.py` each have direct unit tests; `make_ctx`, the composition root, is never called directly by any test in the repo.

**Shape of the fix.** Fold `make_ctx` into `CLIContext` so establishing project-dir semantics is a type obligation rather than a convention, give each command a `*_cmd(cli, ...)` seam, and route the read-only commands through the existing `prepare_readonly`.

**Constraint.** ADR-14 and CLAUDE.md make lazy imports load-bearing.
Any extraction must keep `from mbt.x import y` inside the extracted function rather than hoisting it to module top, and new path options must still go through `ctx.resolve_cli_path`.

---

## A-3. The parser's 19 rules are wired into a transcript nothing can address

**Files.** `packages/mbt-core/src/mbt/parsing/project_parser.py` (1,502 lines, 7 of the last 40 commits, the repo's hottest file), `packages/mbt-core/src/mbt/compile/compiler.py:258-366`.

**Symptom.** The import interface is small and correct: `parse_project`, `ParsedProject`, `ParsedResource`, `SourceEntry`.
Behind it are 19 `_check_*` / `_resolve_*` / `_validate_*` functions, each already shaped correctly as a pure function of its inputs.
They are wired into `_link_and_check` (from `project_parser.py:961`), which is not a dispatcher but a transcript: a fixed sequence of calls that nothing else can reach.

This is what drives the churn.
Every ADR that adds a cross-resource invariant edits the same places: a new `_check_*`, a call line inside `_link_and_check`, and often `_validate_dataset_windows`.
ADR-30 `+87`, ADR-22 `+123/-21`, ADR-27 `+52`, ADR-29 `-71`.

**The compile-phase hole.** `compile/compiler.py` discards `res.spec`, the object the parser validated, and re-renders `res.raw` with target vars before re-validating from scratch (`compiler.py:264`, `:332`, `:358`).
That re-render is deliberate and necessary: the capture phase cannot see target scope, so a `var()` used inside a spec can genuinely differ per target.
But of the parser's 19 rules, compile re-runs exactly one, `validate_hyperparameters` (`compiler.py:366`), plus one compile-local check of its own, `_check_out_of_time_follows_test` (`compiler.py:280`).

So a target var that moves `spec.target`, `evaluation.protocol.test_window`, or a gate's `source` passes `mbt parse` and reaches execution unchecked.
ADR-5's Nuance names the mechanism, and says nothing about the checks being skipped, so ADR-5 is not cover for this.

**Supporting friction.**
`ParsedResource.raw` carries the comment "original YAML mapping (pre-Jinja), for resolve phase" (`project_parser.py:94`): a staging field whose only consumer is another module, which uses it to redo the parser's work.
`_link_and_check` completes `ParsedResource` by mutation, so a `ParsedResource` is invalid until a private function has run over it, which `test_project_parser_unit.py:912` has to know when it builds one by hand.
Two dead parameters mark the seam: `_resolve_dataset(report=...)` never touches `report`, and `validate_hyperparameters(phase=...)` never reads `phase` though both call sites pass it.

**Deletion test.** `parse_project` passes: delete it and callers must re-author discovery, capture rendering and 19 invariants.
`_link_and_check` fails: delete it, keep the rules, and the caller loses only a fixed traversal order.
It is a shape, not a module.

**Shape of the fix.** Make the rules a list, and fold `raw` and `renderer` behind `ParsedProject.render_for_target(uid, ctx)`.
Adding a rule becomes one entry instead of three edits, and the compile-phase hole closes by construction because the same list runs on the resolved specs.

**Test surface today.** Verifying one rule means editing fixture YAML by string replacement and running a full parse, as `tests/test_out_of_time_split.py:84-91` does.
A registry makes each rule a two-spec table test, and "every rule runs" one test over the list.

---

## A-4. The training adapters re-type each other

**Files.** `packages/mbt-{xgboost,lightgbm,sklearn,h2o}/src/.../adapter.py`, `packages/mbt-spark/src/mbt_spark/training.py`, `packages/mbt-adapter-base/src/mbt_adapter_base/{protocols.py,training_helpers.py,compliance/suite.py}`.

**Symptom.** Normalising only the framework name, **87 of the LightGBM adapter's 173 substantive lines appear verbatim in the XGBoost adapter**, which is 50%.

`evaluate` is identical in three packages but for the model type annotation:

```python
# xgboost:325, lightgbm:283, sklearn:363
def evaluate(self, model, data, split, metrics, slices=None) -> MetricResults:
    from mbt_adapter_base.training_helpers import evaluate_split

    table = data.read(split)
    return evaluate_split(table, model.target, self._scores(model, table), metrics, slices)
```

`predict` likewise.
`_fit_calibrator` recurs in all five packages, each with a near-verbatim docstring saying it is "the same mechanism as the xgboost adapter".

The surface an adapter author must learn is 22 named things: 5 properties, 9 required methods, 4 optional protocols, 3 undeclared `supports_*` class variables, and the undeclared `__init__(config: dict)` convention.
Behind it, only four things actually vary: `train`, `_scores`, `export`, `load`.

The tests duplicate in the same shape.
After normalising the framework name, `test_xgboost_categorical.py` and `test_lightgbm_categorical.py` differ by **6 lines**, and their four test function names are identical.

**Deletion test.** Delete `evaluate`, `predict`, `shap_importance` and `_fit_calibrator` from all five adapters and put them on a shared base keyed on the one abstract hook every adapter already has, `_scores`.
Nothing per-framework is lost, because every adapter already routes through `_scores`.
The interface drops from 9 required methods to about 5.

**Sequencing.** The three Arrow adapters share `_scores(model, table)`; h2o and Spark take `(model, data, split)` because of the `data_access` path/arrow divide (ADR-17).
Settle B-1 first, or scope this to the three Arrow adapters.

**Related.** `SparkAdapterError` and `SnowflakeAdapterError` are identical classes in two packages, neither deriving from a shared base, so `hint` is flattened into the message string and core cannot render it as a hint field the way it does for `MbtError.hint`.
`compliance/suite.py:567` documents a protocol named `SupportsReportingTrainer` that exists nowhere in the repo; the real name is `SupportsTrainWithReport`.

---

## A-5. The champion record is 53 string keys with no owner

**Files.** `packages/mbt-core/src/mbt/execute/runners.py:648-686` and `:905-1000`, `packages/mbt-core/src/mbt/{promote.py,execute/oot_check.py,execute/inference_config.py}`, `packages/mbt-mlflow/src/mbt_mlflow/adapter.py:348-368`.

**Symptom.** `RegistryAdapter.register` takes `metadata: dict[str, str]`.
There are **53 distinct `mbt.*` keys** across 11 files in 2 packages.
Exactly one has a name, and it is written back as a raw literal anyway:

```
promote.py:76    OOT_CHECK_TAG = "mbt.oot_check.passed"
runners.py:93        "mbt.oot_check.passed": verdict,

promote.py:133   tags.get("mbt.gates_passed") == "true"
runners.py:651       "mbt.gates_passed": "true",
runners.py:716       "mbt.gates_passed": str(all_gates_passed(gates)).lower()
```

The `ArtifactRef` to tags codec, the same four keys `_uri` / `_format` / `_content_hash` / `_size_bytes`, is hand-written three times.
`job_result.artifact` is an `ArtifactRef`, gets shredded into four strings, and is reassembled four frames later at `runners.py:933-936`.

`read_inference_config`'s docstring says "callers check the tag exists first", an obligation held in prose and honoured by one caller (`oot_check.py:44`).

**Why it costs.** The load-bearing champion contract, which artifact, which hooks hash, whether gates passed, where the baseline lives, is conveyed by key spelling.
A typo on the write side is caught only by whichever e2e run happens to read that key back.
There is no `RegistryAdapterCompliance`, so a second registry adapter has nothing to build against; `mbt-mlflow/tests/` is the de facto contract.

**Deletion test.** Not deletable as-is, because the keys are real state.
The finding is that no module owns them.

**Shape of the fix.** A `ChampionRecord` model in the contract surface with `pack()` and `unpack()`.
The adapter dumps it to backend tags; core stops parsing strings.
A compliance case asserting `get_version(register(record)).record == record` would pin in one place what `promote.py`, `oot_check.py` and `runners.py` each separately assume.

---

## B-1. Capability is dispatched by `hasattr`, not by the declared protocols

**Files.** `packages/mbt-adapter-base/src/mbt_adapter_base/protocols.py:143-145` and `:199-256`, `packages/mbt-core/src/mbt/execute/*`, `packages/mbt-core/src/mbt/parsing/project_parser.py:659` and `:695-716`.

**Symptom.** Four capability protocols are declared: `SupportsTrainWithReport`, `SupportsFeatureImportance`, `SupportsShapImportance`, `SupportsExplain`.
Each has two or more real adapters, so the seams themselves are real.
But no dispatch goes through them.
What decides is 16 `hasattr` probes across `execute/` (`job.py` 7, `runners.py` 4, `training_report.py` 3, `orchestrator.py` 1, `oot_check.py` 1) plus two `getattr` probes in `quality/checks.py`.

Delete all four protocol declarations and nothing changes at runtime.
They are documentation typed as code, and mypy never checks the dispatch, so a renamed method silently disables a capability.

Three more capability flags are on no protocol at all and are probed by the parser by name: `supports_calibration`, `supports_monotonic_constraints`, `supports_categorical_pooling`.
sklearn contradicts its own flag, because the flag has no granularity: it advertises a blunt yes at `adapter.py:123-126`, then `validate` raises for the case the flag advertised.

The compliance suite gates a test on `getattr(adapter, "supports_calibration", False)`, so a typo in the class-variable name silently skips the test rather than failing it.

**Shape of the fix.** One `capabilities() -> frozenset[Capability]` on the protocol, replacing the probes and the class variables.
sklearn can then return an estimator-accurate set, and a skipped compliance test becomes unmissable.

**Touches ADR-17.** `data_access: "path" | "arrow"` was a deliberate choice and its staging cost is accepted there, so the flag itself is not the finding.
The narrow part worth reopening is that it is half-honoured: core branches on it at 6 sites, but h2o's `evaluate` and Spark's `predict` still call `data.read(split)`, so path adapters round-trip through Arrow on every non-train path anyway.
The flag saves a read only inside `train`.

---

## B-2. `training_report.py` is a file split of `job.py`, not a seam

**Files.** `packages/mbt-core/src/mbt/execute/job.py:108-125`, `:1140-1168`, `:1455-1475`; `packages/mbt-core/src/mbt/execute/training_report.py:34-48`.

**Symptom.** `training_report.py` imports `job.py`'s private context type across a module boundary:

```python
if TYPE_CHECKING:
    from mbt.execute.job import _JobRuntime
```

`_JobRuntime` is a 13-field dataclass, five of them typed `Any`.
`training_report` reaches 10 of those 13 fields, and takes `runtime` as the first parameter of ten of its thirteen public functions.

The dependency is bidirectional.
`job.py` cannot import `training_report` at module scope, so four function-local imports exist purely to dodge the cycle (`job.py:384`, `:1136`, `:1234`, `:1446`), and `training_report` takes a callback back into `job.py` to do its staging.
CLAUDE.md and ADR-14 make lazy imports idiomatic here, which is precisely why this cycle is invisible: the convention hides it rather than justifying it.

The interface is a six-step ordered protocol the caller must restate, and it is restated twice, at `job.py:1140-1168` and `:1455-1475`, differing only in `include_train`, `kind` and `artifact_path`.

**Deletion test.** Paste `training_report.py` into `job.py`: nothing else in the repo changes, the cycle disappears, and the number of concepts a reader holds is identical.
Complexity moves sideways.
The file split hides nothing, which is the signal to give it a real interface rather than to merge it.

**Test surface today.** `packages/mbt-core/tests/exec_unit_helpers.py:130-190` hand-builds a `_JobRuntime` with `ctx=None, store=None` and a 15-field `SimpleNamespace`, with a comment apologising for needing a real `ManifestNode`.
Production code then carries defensiveness shaped by that double: `getattr(runtime.job, "anchor", "")` appears three times against a field that is declared with a default.

**Shape of the fix.** One public entry point returning a summary, with `scored_splits`, `report_meta` and `publish_report` made private.
The six-step sequence then exists once.

---

## B-3. The seed ladder has no home

**Files.** `packages/mbt-core/src/mbt/execute/job.py:264, 536, 615, 668, 721, 840, 921`; `packages/mbt-core/src/mbt/execute/training_report.py:41`.

**Symptom.** Seven rungs, every one an inline arithmetic expression, documented in five prose locations: CLAUDE.md, ADR-18, `docs/concepts.md`, and twice as a comment inside `job.py`.
None of those is executable.
The only rung with a symbolic name is `PERMUTATION_SEED_OFFSET`, and it lives in a different file from the other six.

Live defect 2 above is the consequence: `+3` is taken twice.

**Deletion test.** Nothing to delete; the module is missing.
A `seeds.py` with one named function per rung would be deep by the strictest reading: a trivial implementation behind a name that is the entire point, and the only place the ladder can be read as a whole.

**Test surface today.** `test_job_unit.py:529` proves two rungs differ.
Nothing proves the set is collision-free, which a registry makes a three-line property test.

---

## B-4. The event seam accepts `object`

**Files.** `packages/mbt-adapter-base/src/mbt_adapter_base/protocols.py:52-55`, `packages/mbt-core/src/mbt/events/{bus.py:30-40,models.py:18-27}`, `packages/mbt-core/src/mbt/execute/monitor.py:249-253`.

**Symptom.** `EventSink.emit(event: object)` teaches a caller nothing, so the bus guesses: any non-`Event` is stringified into a bare `LogMessage` at the default level.
Live defect 1 is that guess reaching production through the data seam.

The second consequence is a sink filtering another module's events by matching English:

```python
message = getattr(event, "message", None)
if isinstance(message, str) and "to score" in message:
    return
```

Reword a data adapter's log line and the suppression that ADR-20's monitor path depends on silently stops working.

`docs/troubleshooting.md` already treats error wording as a contract, and is asserted against real reproductions.
That discipline does not extend to the strings a sink matches on.

**Shape of the fix.** Type the seam, and keep the string fallback at the hook boundary only, where foreign objects genuinely arrive.
Severity then becomes a property of the event rather than of the adapter, and sinks filter on identity.

---

## C. Smaller findings

### C-1. Nothing owns "judge this job result"

`runners.py:795-820` and `oot_check.py:210-223` run the same five-call sequence: gates, stability, verdict, passed, failure summary.
Each leaf has a tight unit test in `test_report_gates_unit.py`; the composition is tested only through `run_command`.

The composition is where the coupling is.
`after_test_verdict` identifies after-test gates by `gate.period is not None` (`runners.py:83`), which is an internal detail of `gates.py:174`.
`GateSpec.source == "out_of_time"` is the real discriminator and is available on the spec.
Any future gate kind that populates `period` silently changes every recorded verdict, and both the leaf test and `_out_of_time_result` still pass.

`evaluate_stability` also returns `[]` for two different facts, "not declared" and "declared but nothing mature", and `runners.py:84` consumes `not stability` as if it meant one.
That distinction is what separates `not_gated` from `true`, and it is carried by a warning log line rather than by the return type.

### C-2. `feature_columns` is a return value smuggled through state

`handles.py:207` initialises it to `None`; it is populated as a side effect of the first `read()` (`handles.py:232-233`).
`job.py:1051` makes the obligation explicit: `runtime.transformed.read("train")  # resolves the feature columns`.
Seven readers rely on that having happened and all fall back to `or []`.

The write side and the read side disagree about what empty means.
`_export_inference_config` writes `feature_columns` into the champion's inference config, `ScoringRunner._champion_config` reads it back as `pinned_features`, and `_apply_pin` treats a non-`None` pin as authoritative, so an empty list is a valid pin of zero features rather than a missing pin.

`TransformedDatasetHandle` itself is one of the genuinely deep modules in `execute/` and earns its keep; it is the attribute that does not.
Separately, `job.py` reaches into its private `_base` eight times, once just to format a log line (`job.py:1108-1109`).

### C-3. `oot_check` drives `ModelRunner` through four private methods

`oot_check.py:188, 189, 198, 210` call `_metric_specs`, `_assemble_job`, `_upload_log` and `_gate_results`.
`_assemble_job` already has an `artifact` parameter that this path bypasses in favour of patching the returned model, so there are now two ways to set the same field, and `champion_spec` has no parameter at all.

`_pin_windows` (`oot_check.py:106-113`) mutates the in-memory manifest, and the value is read back four frames later by `DatasetRunner._materialize`.
`run_oot_check` must call it before `DatasetRunner.run`; nothing enforces that, and nothing names it.

All eight tests in `test_oot_check_flow.py` go through `run_command` then `run_evaluate`, so there is no seam at which "given a version and a manifest, what windows get pinned" can be asserted.

### C-4. `ReportData.tables` is a string-keyed namespace with a silent fallback

The builder-to-writer contract is eight magic strings in `_TABLE_FILES`, plus `f"binning_{bins.name}"`, plus two keys a third module injects after `build_report` returned (`training_report.py:447-448`).
Any key that is none of those lands in `evaluation/binning/<key>.csv`, a misroute that never errors.

The payload is also mutated back across the seam: `write_report` writes into the data it was handed, so `build_report`'s output is not final until someone writes it.
Guarding is inconsistent within one function: `data.tables["stability_scores"]` is unguarded at `writer.py:532`, and `data.tables.get("stability_features", [])` is guarded twelve lines later.

`render.py` exports 13 symbols, every one consumed by exactly one caller.
It is a private helper file with a public-looking surface, which is not a defect but means the trio is two deep modules and a section header rather than three.

### C-5. `AdapterPlugin.task_schemas` is dead machinery

Zero plugins in the repo populate it.
Task schemas already register from core at `config/tasks/__init__.py:12`.
Delete the field and `registry.py:104-111`'s `_register_task_schemas` and nothing breaks.

Related: `TrackingAdapter`, `RegistryAdapter`, `TuningEngine` and `ReportingEngine` each have exactly one real implementation, with the second "adapter" being the test fake.
Those are hypothetical seams by the two-adapter rule.
None is worth collapsing today, because mlflow, optuna and evidently are all genuinely optional, but it is worth knowing that the fakes exist to make core's tests run rather than to prove the abstraction.

### C-6. `mbt.contracts` is a pass-through

`packages/mbt-core/src/mbt/contracts.py` is `from mbt_adapter_base import *` plus three names.
36 source files import from it, 54 import from `mbt_adapter_base` directly, and adapters use both paths in one file (`mbt_spark/data.py:30` and `:36`).
The star re-export makes jump-to-definition unreliable, which costs an agent reading the tree more than it costs a human with an IDE index.

Deletion test: delete it and complexity vanishes, because the rewrite is mechanical.

**This contradicts ADR-15 §1**, which recorded the re-export deliberately.
Its stated reason was to "skip a painful mid-project extraction (S7-06 done early)", and that extraction has since happened: the contracts do live in `mbt-adapter-base`.
The shim was migration scaffolding that outlived its migration.
Worth reopening as cleanup only, and only after the findings above; it changes no behaviour.

---

## Recommendation

Take **A-1** first.

It is the only finding already causing a live defect, and this repo has run the play before one layer down: `8ea7b6d` deleted a join that three adapters each had to reimplement, and the same three adapters are still reimplementing the build that wraps it, including the arithmetic that decides which rows train.
Three real adapters make the seam genuine, and everything proposed to move behind it is pure policy, so the risk is low relative to the leverage.
It also unblocks the two things the data seam has never had: an in-memory engine to test against, and a compliance suite to hold a fourth adapter to.

**A-2** is the reasonable alternative if the preference is to follow churn rather than defect weight.
It has a proven user-facing bug, it moves roughly 25 tests off the CLI runner, and it stops two inspection commands touching the live data system.

The three live defects are each small and independent of their parent finding.
They can land first, on their own, if the larger work is not being taken now.

---

# Addendum: a data-science pass

Added 2026-09-21, against `main` (`e25bdcc`), after the architecture pass above closed.

## Why this is in the same file

This addendum asks a different question from the rest of the document: **not is the code the right shape, but are the statistics correct?**
It is filed here rather than as its own cycle because nothing in it is worth a sixth review round on its own, and because two of its findings share a fix with findings already listed above (D-4 lands as one entry in A-3's rule registry; D-1's carve reuses machinery A-1 does not touch).

The v1 and v2 cycles both reviewed data-science practice and closed a long list of findings there.
Nothing below re-litigates them.
Where a finding is adjacent to a closed one, it says so and says what is new.

## Method, and one correction

Findings D-2 through D-5 were read out of the source.
D-1 was **measured**, because the first version of this finding was wrong.

It was originally written as "the deployed threshold is badly biased and will systematically miss its precision target."
A simulation against mbt's own `compute_metric` falsified the strong form of that claim: at realistic test-window sizes the point bias is under 0.01, which is not worth anyone's time.
The finding survives in a narrower and better-evidenced form, and the boundary between where it matters and where it does not is now a measured number rather than an assertion.
D-3 was cut in half the same way, after its proposed fix turned out to be already shipped.

The D-1 simulation is described precisely enough in place to rebuild in about twenty lines: draw a label at a base rate, draw a score from a normal shifted by the label, call `compute_metric` with a `threshold_at_precision_<p>` spec on one draw, then measure the realized precision at that cutoff on both that draw and an independent one.
It was run against this tree's `mbt_adapter_base.metrics`, not a reimplementation.
No script is checked in: the repo-root session guard in `conftest.py` fails any run that leaves new entries there, and this is not a test.

## Findings

| # | Finding | Strength | Size |
|---|---|---|---|
| D-1 | The deployed operating point is selected on the split that reports it, and its support is never counted | Strong (live defect) | S |
| D-2 | Early stopping leaves the final fit when the validation carve is reabsorbed | Strong | M |
| D-3 | Nothing counts how many times one test window has arbitrated a gate | Smaller | S |
| D-4 | No rule catches auto-rebalancing plus a calibration metric with no calibrator | Worth exploring | XS |
| D-5 | Two estimators are noisier than they present | Smaller | S |

Not repeated here: the `spec.seed + 3` collision is **live defect 2** and **B-3** above.
It is a data-science defect as much as an architectural one, and B-3's fix is the right one.

---

## D-1. The deployed operating point is selected on the split that reports it, and its support is never counted

**Files.** `packages/mbt-core/src/mbt/execute/job.py:1122` and `:1179`, `packages/mbt-core/src/mbt/execute/inference_config.py:35-42`, `packages/mbt-core/src/mbt/execute/runners.py:683-686` and `:1104-1123`, `packages/mbt-adapter-base/src/mbt_adapter_base/metrics.py:122-149`.

**Symptom.** `threshold_at_precision_<p>` is computed on the test split at `job.py:1122`, carried into the inference config at `job.py:1179`, filtered out by `operating_points()` as a deployable cutoff, written to the registry as `mbt.operating_point.<metric>`, and resolved into the production decision rule by `_resolve_operating_point`.

So the cutoff is chosen by scanning the test set's own PR curve for the best point, and the precision at that point is then reported from the same rows.
A threshold is a fitted parameter, and this one is fitted on the evaluation split.

**What is actually wrong, measured.** The point bias is small and shrinks fast, so the interesting variable is not the test-window size but **the number of rows above the selected cutoff**.

400 replications per row, mbt's own `compute_metric`, separate draws for "test" and "fresh":

```
      n   base  target  on-test   fresh     bias  P(miss)    n>=t
   5000   0.02    0.50   0.5000  0.4950  +0.0050     53%      24
   5000   0.02    0.70   0.7794  0.6496  +0.1297     62%       7
   5000   0.05    0.50   0.5000  0.5022  -0.0022     51%     146
   5000   0.05    0.70   0.7113  0.7049  +0.0064     53%      44
  20000   0.02    0.50   0.5000  0.5047  -0.0047     53%      83
  20000   0.02    0.70   0.7251  0.6895  +0.0356     58%      22
  20000   0.05    0.50   0.5000  0.5008  -0.0008     50%     565
  20000   0.05    0.70   0.7015  0.7018  -0.0004     54%     172
```

Two things fall out.

The bias is negligible whenever the cutoff is supported by roughly a hundred rows or more, and becomes serious when it is not: at 7 rows above the cutoff, a reported 0.78 is a realized 0.65.
That is the rare-positive, high-precision corner, which is exactly the retention-campaign shape the feature was built for (`churn_demo` runs `threshold_at_precision_0.35` at a ~20% base rate, comfortably outside the danger zone, which is why no test has ever caught this).

`P(miss)`, the share of runs where realized precision lands under the target, sits at 50% to 62% in every row including the well-supported ones.
That is structural rather than a bias: the threshold is a point estimate of "the smallest cutoff meeting the target", so it lands under the target about half the time by construction.
Nothing in mbt says so.
`docs/spec-reference.md:589` describes it as "the smallest score cutoff meeting the precision target", which reads as a guarantee.

**Why it counts as live.** A user who writes `decision_threshold: threshold_at_precision_0.9` on a low-prevalence problem gets a cutoff supported by a handful of rows, a number that reads as a promise, no interval, and no warning.
The v1 cycle built the whole plumbing path for this value (`feedback-v1.md:341-344`) and reviewed it as a plumbing problem, which it was; the selection question was never asked.

**Shape of the fix, cheapest first.**
Count the rows at or above the returned cutoff inside `_threshold_at_precision` / `_threshold_at_recall` and refuse to emit an operating point supported by fewer than some floor, or emit it with a loud warning.
mbt already holds `y_true` and `y_score` at that moment, so this is a few lines and it converts the dangerous corner into an error.

Then, for the structural half: select the operating point on rows that are not the reporting rows.
The carve machinery exists.
`_tail_carve_indices` (`job.py:546`) is already shared by the validation carve (`seed + 2`) and the calibration carve (`seed + 5`), and an operating-point carve is the same move at the next rung, `seed + 7` once B-3's registry lands.
Report the realized precision at that fixed cutoff on test as the honest number.
ADR-30's after-test window is then the natural place to check it held.

**Scope note.** This is a metrics-layer and job-layer change.
It touches no adapter and no ADR, and the registry tag format does not change.

---

## D-2. Early stopping leaves the final fit when the validation carve is reabsorbed

**Files.** `packages/mbt-core/src/mbt/execute/job.py:844` and `:1103`, `packages/mbt-adapter-base/src/mbt_adapter_base/training_helpers.py` (`note_early_stopping_without_validation`), `packages/mbt-xgboost/src/mbt_xgboost/adapter.py:229-272`.

**Symptom.** When a dataset declares no `validation` split, `_carve_validation` carves one from train at `seed + 2` and tuning trials use it, both to early-stop and to score the objective.
The final fit then reabsorbs the carve: `job.py:844` keeps the reabsorption deliberate and `job.py:1103` sets `fit_handle = runtime.handle`, which has no validation split.
`xgboost.train` therefore receives `evals=[]` and `early_stopping_rounds=None`, and runs all `n_estimators` rounds.

The reabsorption itself is correct and is recorded as a strength in `feedback-v1.md:1185` for ADR-8 compliance.
What is new here is the consequence, which that entry did not follow through: every trial that voted on the hyperparameters stopped early, and the model those hyperparameters were selected for is not the model that ships.

The condition is not silent.
`note_early_stopping_without_validation` emits a message, added precisely because it used to be.
But it is worded as a configuration hint, "declare split.validation on the dataset to stop early", which reads as "you are missing a feature" rather than "the model about to be registered has a different effective depth from the one your search scored".

**Why it costs.** The gap between the tuned model and the shipped model widens with `n_estimators` and with how early the trials were stopping.
A search that converged on aggressive learning rates because early stopping was protecting it produces a final model with no such protection.
Nothing downstream can see this: the gates evaluate the shipped model honestly, so it fails only when it is genuinely worse, and silently ships a differently-regularized model when it is not.

**Shape of the fix.** Two options, both contained.
Keep the carve held out for the final fit when `early_stopping_rounds` is set, accepting the lost training rows, which is what the reabsorption exists to avoid and therefore a real trade.
Or carry the trials' `best_iteration` through `TuningResult` and use its median as the final `n_estimators`, which keeps every training row and ships the complexity the search actually chose.
The second is the standard move and does not touch ADR-8.

Whichever lands, the warning text should say what the consequence is rather than what the user did not declare.

---

## D-3. Nothing counts how many times one test window has arbitrated a gate

**Files.** `packages/mbt-core/src/mbt/execute/runners.py:604-630`, `packages/mbt-core/src/mbt/quality/gates.py`, `packages/mbt-core/src/mbt/execute/oot_check.py`.

**Symptom.** Every training run evaluates its gates against the test window, and mbt is built to be run repeatedly.
Nothing records how many times a given `(dataset, window)` pair has judged a candidate, and nothing surfaces it.

Over enough iterations a held-out test window stops being held out.
The decisions are not being made by the model any more, they are being made by the analyst reading pass/fail and adjusting, which is selection on the test set conducted one commit at a time.

This is adjacent to `feedback-v1.md:1186` (P2, best-trial selection as max-over-trials on one carve), which was closed by the robust bootstrap objective and nested CV.
That closed the bias inside a single run.
This is the bias across runs, which those fixes do not reach.

**What already exists, and a correction.** This finding was first drafted with a second half recommending that a passing after-test verdict be made a precondition for `promote`.
That was wrong, and it is withdrawn: the enforcement is already built and shipped.
`promote.py:79-115` refuses a version whose recorded verdict is `false` **unconditionally**, and `--require-oot-check` (or `require_oot_check: true` on a `promotions.yml` entry) additionally refuses one that was never judged or whose check found nothing mature.
It is documented in `docs/cli-reference.md:307` and `:312`, `docs/gitops.md:28` and `docs/troubleshooting.md:686`, and driven end to end at `tests/test_e2e_churn_demo.py:416`.
That the opt-in half defaults off is a deliberate policy position, not an oversight, and this review has no standing to reopen it.

**What is left, and it is narrower.** Only the counting half survives.
ADR-30 gives a model a fresh window to be judged against, which is the strong defence and is present.
What no part of the system does is notice that the *same* test window has now arbitrated twenty candidates.

**Shape of the fix.** Record on each training run how many times its `(dataset uid, resolved test window)` pair has been gate-evaluated, and warn past a threshold.
The state module already tracks enough to answer it, and a warning carries no policy.
This is worth doing for the same reason the backtest std is reported next to the backtest mean: it tells a reader how much to trust a number that otherwise looks unconditional.

---

## D-4. No rule catches auto-rebalancing plus a calibration metric with no calibrator

**Files.** `packages/mbt-core/src/mbt/config/tasks/binary.py:27-42`, `packages/mbt-adapter-base/src/mbt_adapter_base/training_helpers.py` (`resolve_scale_pos_weight`).

**Symptom.** `scale_pos_weight: '{{ auto }}'` resolves to `(1 - p) / p` and deliberately destroys probability calibration.
Measuring `brier` or `ece` on the result, with no `calibration:` set, reports a calibration number for scores that were miscalibrated on purpose.

`BinaryClassificationSchema.validate_spec` is the natural home for the rule and currently holds exactly one, that a slice column may not be the target.
`validate_dataset` warns on extreme imbalance, which is the adjacent concern, not this one.

**Status of the adjacent closed finding.** R2-8 (`feedback-v1.md:84`) was this exact combination, and it was closed by building calibration and then, in v2, by fixing the demo fixture (`feedback-v2.md:365` added `calibration: isotonic` to `churn_classifier.yml`).
Both halves fixed the instance.
Neither added a guard, so a user writing the same three lines today gets the same silently meaningless number the demo used to report.

**Shape of the fix.** One `ValidationIssue` at warning severity when the spec sets an auto or explicitly large `scale_pos_weight`, declares `brier` or `ece`, and sets no `calibration`.
The hint writes itself, because the fixture is the worked example.

This is one entry in A-3's rule registry once that lands, and one function plus one call line before it does.

---

## D-5. Two estimators are noisier than they present

**Permutation importance.** `packages/mbt-core/src/mbt/execute/training_report.py:223-300` shuffles each feature once, on up to 5,000 rows, then normalizes the drops to sum to 1.
sklearn's equivalent defaults to 5 repeats because a single shuffle is high variance, and the normalization to a tidy fraction makes the output read as more settled than it is.
Two runs at different seeds can reorder mid-table features.
The fix is `n_repeats` with a mean, or reporting the spread alongside the value; the cost is linear in repeats and this path only runs for adapters reporting no native importance.

**The leakage scan's numeric screen.** `packages/mbt-core/src/mbt/quality/checks.py:577-644` screens numeric columns with duckdb's `corr`, which is Pearson.
That catches linear leakage.
A monotone but nonlinear leak, which is the common shape when a leaked column is a transformed or bucketed version of the label horizon, can sit under the 0.95 bar while being perfectly predictive.
The categorical path does not share the weakness, because Cramér's V is association rather than correlation.
Spearman is `corr` over `rank()` in duckdb, so screening on `max(|pearson|, |spearman|)` is a small change to one query and keeps both thresholds meaningful.

Neither is a defect.
Both are numbers that look more precise than they are, in a document whose whole purpose is that people trust its numbers.

---

## Recommendation for this addendum

Take the first half of **D-1**, the support count, on its own and immediately.
It is a few lines in `metrics.py`, it needs no carve, no seed rung and no ADR, and it converts the one corner where this genuinely misleads into a loud failure.

**D-4** is the next cheapest and is one rule.

**D-2** is the one with the largest gap between how quietly it fails and how much it changes what ships, and it is the right thing to take if only one substantial item is being worked.

D-1's second half wants the seed registry from B-3, so it sequences behind it.
**D-3** and **D-5** are cleanup, and D-3 is now small enough that it only earns its place if the state module is being opened anyway.

None of this displaces A-1 as the repo-level first move.
The D-series is narrow and mostly independent of the architecture work, with the single overlap that D-4 becomes trivial once A-3 lands.

---

## Progress log

One appended entry per completed item, carrying symptom, fix, verification and docs, per the shape the four earlier cycles use.
This file moves to `design-history/reviews/feedback-v5.md` when the log closes.

### B-3 - the seed ladder has a home, and `+3` is no longer taken twice (live defect 2)

**Symptom.** Seven seeded stages, every one an inline `spec.seed + N`, documented in five prose locations and executable in none.
`+3` was taken by both the champion gate's paired bootstrap (`job.py:536`, ADR-18) and the tuning-time robust objective (`job.py:721`), so the resamples that SELECT a model and the resamples that JUDGE it were the same draw.

**Fix.** `packages/mbt-core/src/mbt/execute/seeds.py`: a `SeedRung` IntEnum that is the whole ladder, plus one named accessor per rung.
Every call site in `job.py` and `training_report.py` now names its rung instead of doing arithmetic.
The tuning objective took a new rung, `TUNING_OBJECTIVE_BOOTSTRAP = 7`; `OPERATING_POINT_CARVE = 8` is reserved for D-1.
`PERMUTATION_SEED_OFFSET` moved out of `training_report.py` into the ladder.

**Verification.** `packages/mbt-core/tests/test_seeds_unit.py` (14 cases): offsets are distinct, offsets are contiguous from zero, every rung has an accessor whose name is derived from it, and each accessor agrees with its rung.
The naming test caught a real drift while being written (rung `TUNING_OBJECTIVE_BOOTSTRAP` against accessor `tuning_objective_seed`).

**Docs.** CLAUDE.md's seed-ladder line and `docs/concepts.md` now point at `execute/seeds.py` as the executable home.

### B-4 / live defect 1 - the event seam is typed, so severity belongs to the event

**Symptom.** `EventSink.emit(event: object)`, and the bus wrapped any non-`Event` in a bare `LogMessage` at the DEFAULT level.
So an empty `out_of_time` split was a WARN from the local adapter and an informational line from Spark and Snowflake, on the same project.
A sink also filtered another module's events by matching English (`"to score" in message`).

**Fix.** `mbt_adapter_base/events.py` now defines `Event`/`LogMessage` (an adapter package cannot import core, which is why the seam could not be typed before) plus the events adapters emit: `DatasetMaterialized`, `EmptyAfterTestSplit`, `ScoringInputMaterialized`, `EarlyStoppingWithoutValidation`, `AdapterMessage`.
`mbt/events/models.py` re-exports them, so `isinstance(x, Event)` means the same thing on both sides.
`EventSink.emit` and `EventBus.emit` take `Event`; the string fallback moved to `HookEventSink`, used only for `HookContext.logger`, where user code genuinely arrives.
`monitor._LabelReadEvents` now suppresses by type.

**Verification.** mypy (strict, 12 packages) found the one remaining bare-string emit, in `mbt-h2o`'s leaderboard - which is the point of typing the seam.
`test_events_unit.py` proves coercion happens at the hook boundary and nowhere else; `test_monitor_unit.py` proves a `LogMessage` whose text says "to score" is NOT suppressed, which a substring match could never express.

**Docs.** `docs/troubleshooting.md`'s event-line table no longer says the warehouse adapters prefix the node id, because they no longer word the line at all.

### A-1 - the dataset build recipe moved below the data seam

**Symptom.** `build_dataset` was one fixed eleven-step recipe and all three data adapters wrote it out; only four steps are per-engine.
Two of the three said "mirrors the local adapter" in a comment - a contract held in prose, and the prose did not hold (live defect 1 above).
The bucket-edge arithmetic that decides which rows train was written three times, and re-derived a fourth, fifth and sixth time in three separate test files.

**Fix.** `materialization.py` now holds `build_dataset_materialization` / `build_scoring_materialization` (the policy) and `DatasetBuildEngine` (the four per-engine methods: `verify_snapshot`, `write_dataset_splits`, `write_scoring_batch`, `build_failure`).
`split_fractions` + `bucket_ranges` are the one copy of the split arithmetic, and `reference_bucket` / `reference_split` are the one canonical Python reference.
`AdapterFailure` in `mbt_adapter_base/errors.py` gives Spark and Snowflake a shared base with `hint` as a FIELD (A-4's "Related" note).
`count_source_duplicates` and `read_source_distinct` are now declared on the `DataAdapter` protocol.

**Fix, second half.** `DataAdapterCompliance` (7 cases) pins every engine to the reference bucket, to one `DatasetMaterialized` per build, and to WARN for an empty after-test split.
`mbt_testing.InMemoryDataAdapter` is the data seam's second adapter - it shares no code with DuckDB, Spark or Snowflake and produces identical splits, events and severities.

**Verification.** All four engines pass `DataAdapterCompliance`: local, in-memory, Snowflake (its stub runs the adapter's real generated SQL in DuckDB) and Spark (e2e, real session).
Full fast suite green; mypy clean.

### A-2 / live defect 3 - the CLI has a composition root, and `mbt ls` can use the selectors it documents

**Symptom.** `make_ctx` was a free function called at 17 sites with seven positional arguments, and three commands skipped it - one of which, `clean`, called `shutil.rmtree` on a raw unresolved typer path with no context and no event bus.
The same three-step compose was retyped four times and had drifted at every one: `mbt show` and `mbt docs generate` dropped the parser's warnings and re-anchored to `now()`, so `mbt show`'s resolved windows drifted from `target/manifest.json` with wall-clock time - the drift ADR-12 exists to pin.
`mbt ls` hand-rolled its node view and never passed `state`, so `--select state:modified` could not work and `--state` was not an option.

**Fix.** `make_ctx` folded into `CLIContext.enter`, so establishing project-dir semantics is a construction obligation.
`CLIContext.parse()` warns by default and `CLIContext.compile()` is the one three-step composition, always threading `--anchor` through `CompileOptions`.
`show` and `docs generate` gained `--anchor`; `clean` routes both branches through the composition root.
`mbt/cli/inspect.py` holds `ls_cmd` / `show_cmd` / `clean_target_cmd` as `*_cmd(cli, ...)` seams that return their output instead of printing it.
`ls` selects over `Manifest.selectable_nodes()` with the same state index `mbt build` uses, and takes `--state` / `--state-include-env` / `--anchor`.

**Verification.** Reproduced live defect 3 through the real CLI against `tests/fixtures/churn_demo` before and after: `mbt ls --state ref.json --select state:modified` now returns `churn_classifier` and `retention_scoring` after a seed edit, and `--state` is accepted.
`packages/mbt-core/tests/test_cli_inspect_unit.py` (11 cases) drives the composition root and all three seams directly, with no `CliRunner` - including the relative-`--project-dir` `clean` divergence that all four existing runner-based clean tests missed by passing an absolute path.

**Docs.** `docs/cli-reference.md`'s `mbt ls`, `mbt show` and `mbt docs generate` sections document the new flags and say what `--anchor` pins.

### A-3 / D-4 - the parser's rules are an addressable list, and compile runs them too

**Symptom.** Nineteen `_check_*`/`_validate_*` functions wired into `_link_and_check`, which was not a dispatcher but a transcript: a fixed sequence nothing else could reach.
That shape drove the churn (the parser is the repo's most-edited source file): every ADR adding an invariant edited a new function, a call line in the transcript, and often `_validate_dataset_windows`.
It also left a hole - `compile/compiler.py` re-renders `res.raw` with target vars (deliberate and necessary, ADR-5) but re-ran exactly ONE of the nineteen rules, so a target var moving `spec.target`, `evaluation.protocol.test_window` or a gate's `source` passed `mbt parse` and reached execution unchecked.

**Fix.** `mbt/parsing/rules.py`: `RULES` is 14 entries, each a `Rule(name, applies_to, check, phases)`.
`RuleTarget` is structural, so a `ParsedResource` satisfies it at parse time and a `ResolvedTarget` (carrying the target-rendered spec) satisfies it at compile time - which is how the same list runs in both phases.
`run_rules` is called from `parse_project` after linking and from `compile_project` after resolve-rendering.
`_link_and_check` became `_link`: it links, and nothing else.
The two unresolved-reference SYNTAX rules declare `phases=("parse",)`, because the resolve phase rewrites `source('a','b')` into a unique_id - an opt-out a reader sees in the registry, rather than the silent default it used to be.
`_check_maturity_vs_horizon` now walks the LINKED graph instead of re-parsing `ref('name')` off the model spec, which silently found nothing at compile time.
A-3's two dead parameters are gone: `_resolve_dataset(report=...)` and `validate_hyperparameters(phase=...)`.

**D-4 lands as one entry**, exactly as the addendum predicted: `model.rebalancing_vs_calibration_metrics` warns when a spec sets an auto or large `scale_pos_weight`, declares `brier` or `ece`, and sets no `calibration`.
R2-8 was this combination and was closed twice - by building calibration, then by fixing the demo fixture - and neither half added a guard.

**Verification.** `packages/mbt-core/tests/test_parse_rules_unit.py` (9 cases).
The compile-hole regression test was confirmed to BITE: with the compile-phase `run_rules` commented out it reports `DID NOT RAISE ConfigError`, which is the defect.
D-4 has the full table - warns without a calibrator, quiet with one, quiet when no calibration metric is declared.
`project_parser.py` went from 1,502 to 957 lines; full fast suite green (1,772 passed), mypy clean across 12 packages.

### B-1 / A-4 - capability is declared, and the training adapters stopped re-typing each other

**B-1 symptom.** Four capability protocols were declared and NO dispatch went through them: 16 `hasattr` probes across `execute/` decided, plus two `getattr` probes in `quality/checks.py`.
Deleting all four declarations changed nothing at runtime.
Three more flags were on no protocol at all and were probed by name, and sklearn contradicted its own: `supports_monotonic_constraints = True` for the library, then `validate` raised for the estimators that cannot honour one - the flag had no granularity.
The compliance suite gated on `getattr(adapter, "supports_calibration", False)`, so a typo in the class-variable name silently SKIPPED the test.

**B-1 fix.** `mbt_adapter_base/capabilities.py`: a `Capability` enum and `capabilities(spec) -> frozenset[Capability]` on the `TrainingAdapter` protocol.
It takes the spec, which is what lets sklearn answer per estimator instead of advertising a blunt yes.
Core calls `capabilities_of` / `supports`; `capabilities_of` keeps the old method-presence + `supports_*` rule as a compatibility path, so a third-party adapter built against the pre-v5 contract behaves exactly as it did.

**A-4 symptom.** 87 of the LightGBM adapter's 173 substantive lines appeared verbatim in the XGBoost adapter.
`evaluate` was identical in three packages but for a type annotation; so was `predict`; `_fit_calibrator` recurred in all five.
The author-facing surface was 22 named things while only four varied.

**A-4 fix.** `mbt_adapter_base/base.py`: `ArrowTrainingAdapter` keys `evaluate`, `predict` and calibrator fitting on the one abstract hook every Arrow adapter already had, `_scores`.
SHAP lives on a SEPARATE `ShapArrowTrainingAdapter`, which the work itself proved necessary: putting `shap_importance` on the shared base made scikit-learn - which cannot produce contributions - declare the capability and then raise `NotImplementedError`, and six compliance cases failed immediately.
That is B-1's failure mode in miniature, so the inheritance now says what is true.
h2o and Spark stay off the base and declare `capabilities()` directly, which is the sequencing A-4 asked for (the ADR-17 path/arrow divide).
`AdapterFailure` gives `SparkAdapterError` and `SnowflakeAdapterError` a shared base with `hint` as a field.
`compliance/suite.py:567` named a protocol that exists nowhere (`SupportsReportingTrainer`); it now names the capability.

**Verification.** A new compliance case, `test_declared_capabilities_match_what_the_adapter_can_do`, asserts BOTH directions: a declared capability must have a callable method, and a present optional method must be declared - so adding `explain` without declaring it can no longer leave the capability dark.
The duplicated categorical suites (122 lines each, differing by six) became `CategoricalAdapterCompliance` plus two 18-line subclasses.
The three Arrow adapters total 1,222 lines, down from 1,342; full fast suite green (1,778 passed), mypy clean across 12 packages.

### A-5 - the champion record has an owner

**Symptom.** `RegistryAdapter.register` took `metadata: dict[str, str]`, and 53 distinct `mbt.*` keys lived across 11 files in 2 packages.
Exactly one had a name, `promote.OOT_CHECK_TAG`, and it was written back as a raw literal anyway.
The `ArtifactRef` codec - the same four `_uri`/`_format`/`_content_hash`/`_size_bytes` keys - was hand-written three times, and `job_result.artifact` was shredded into four strings and reassembled four frames later.
`read_inference_config`'s docstring said "callers check the tag exists first", an obligation held in prose and honoured by one of its callers.

**Fix.** `mbt_adapter_base/champion.py`: `ChampionRecord` with `pack()`/`unpack()`, `AfterTestVerdict`, and `pack_artifact`/`unpack_artifact` as the one codec.
The key names are unchanged by design - champions registered by older mbt must keep resolving, so this moves who owns the spelling, not the spelling.
`runners.py` builds a record instead of spelling 20 keys; `promote.py`, `oot_check.py`, `inference_config.py` and the mlflow adapter read named constants.
`has_inference_config` is the prose obligation as a function.

**Verification.** `RegistryAdapterCompliance` (5 cases), with the load-bearing one being exactly what A-5 asked for: `ChampionRecord.unpack(get_version(register(record)).tags) == record`.
BOTH registry implementations run it - MLflow and the `mbt-testing` fake - so the seam has the second adapter it needed, and the fake now proves the abstraction rather than only making core's tests run (C-5's related note).
Full fast suite green (1,788 passed), mypy clean.

### B-2 - `training_report.py` has an interface instead of a file boundary

**Symptom.** It imported `job.py`'s PRIVATE context type across a module boundary for the type of its own first parameter, and the dependency was bidirectional: `job.py` could not import it at module scope, so four function-local imports existed purely to dodge the cycle.
Lazy imports are idiomatic here (ADR-14), which is precisely why the cycle was invisible - the convention hid it rather than justifying it.
The interface was a six-step ordered protocol the caller had to restate, restated twice, differing only in `include_train`, `kind` and `artifact_path`.

**Fix.** `mbt/execute/job_runtime.py` holds `JobRuntime` (public), so `job.py` and `training_report.py` both depend on it and neither depends on the other for its types.
`job.py` imports `training_report` at module scope now; the four cycle-dodging imports are gone.
`produce_report(runtime, spec, model, *, kind, stage, include_train, ...)` is the one entry point, returning a `ReportOutcome`; `scored_splits`, `report_meta` and `publish_report` are private.
`_feature_importance` moved into the report step as `feature_importance`, folding in the permutation fallback that both call sites wrote out.
`getattr(runtime.job, "anchor", "")` - production defensiveness shaped by the test double - is now `runtime.job.anchor`, against a field declared with a default.

**Verification.** Both former call sites are one call each.
The two tests that monkeypatched `_feature_importance` now patch `capabilities_of` instead, which states the actual condition ("the adapter declares no importance capability") rather than removing the function that asks.
Full fast suite green (1,788 passed), mypy clean.

### C-1 / C-2 / C-3 / C-4 / C-5 / C-6 - the smaller findings

**C-1 (nothing owns "judge this job result").** `mbt/quality/judgement.py` now owns the five-call sequence that `runners.py` and `oot_check.py` each wrote out; `judge()` returns a `Judgement`.
Two real defects came out of it.
`after_test_verdict` identified after-test gates by `gate.period is not None`, an internal detail of how `_out_of_time_result` is built - `GateResult.source` now carries `GateSpec.source`, stamped in ONE place in `evaluate_gates` so a new gate kind cannot forget it.
And `evaluate_stability` returned `[]` for both "not declared" and "declared but nothing mature", which is exactly what separates `not_gated` from `true`; it returns a `StabilityOutcome` with `declared`/`judged`/`results`, and its `__bool__` raises so the ambiguity cannot be reintroduced by a truthiness test.
Three direct tests of the composition, which was previously reachable only through `run_command`.

**C-2 (`feature_columns` smuggled through state).** It was a public attribute initialised to `None` and populated as a side effect of the first `read()`, with `job.py` carrying `read("train")  # resolves the feature columns` as a comment-held obligation and seven readers falling back to `or []`.
It is `feature_columns(split=None)`, which resolves on demand; the comment and all seven `or []` are gone.
An EMPTY recorded pin is now a MISSING pin on both sides, closing the write/read disagreement that would have projected a batch onto zero features.
`TransformedDatasetHandle.base` is public, so `job.py` stopped reaching into `_base` eight times.

**C-3 (`oot_check` drives `ModelRunner` through four private methods).** `check_job`, `after_test_gates` and `upload_run_log` are the seams that path needs; `_assemble_job` gained the `champion_spec` / `champion_feature_columns` / `tracking_run_id` parameters the caller used to patch on afterwards, so there is one way to set each field instead of two-or-none.
`_pin_windows` is `pin_check_windows`: public, named, and with its ordering obligation stated - and `test_oot_check_unit.py` (4 cases) asserts what it pins, which is the seam the eight flow tests could not reach.
The typed seam immediately found a latent hole: `version.artifact` is optional and the old untyped `model_copy` let `artifact=None` through to fail later somewhere else. It now raises, matching the scoring path's stance (ADR-10).

**C-4 (`ReportData.tables` is a string-keyed namespace).** `TableKey` is an enum in `builder.py` (the producer owns the vocabulary), `binning_key()` names the open-ended ones, and `table_path()` is TOTAL - an unknown key raises instead of landing in `evaluation/binning/<key>.csv`, a misroute that never errored.
`write_report` no longer mutates the payload it was handed except to record the document list, which is the one thing only it can know.
The two guards twelve lines apart now agree.
`render.py`'s docstring states that it is `writer`'s private vocabulary (the review says this is "not a defect", so it is documented rather than restructured).

**C-5 (`AdapterPlugin.task_schemas` is dead machinery).** Deleted, with `_register_task_schemas` and its test.
`docs/roadmap.md` and `docs/adapter-authoring.md` promised it, so both now say the hook will be designed against the first adapter that needs one - a speculative extension point with no implementation had nothing to validate its shape against.

**C-6 (`mbt.contracts` is a pass-through).** Both stated costs are fixed: core's 39 files import from `mbt_adapter_base` directly, and the star re-export is explicit, so jump-to-definition works.
**The module is NOT deleted, and this deviates from the finding.** `from mbt.contracts import TestResult` is the line `mbt init` stamps into every scaffolded project's `tests/test_data_quality.py` and the one `docs/spec-reference.md` tells users to write - so the rewrite is mechanical inside this repo but breaking outside it.
ADR-15 §1 is therefore half-superseded rather than wrong: its stated reason has expired, and the module earned a better one. The docstring records that, so the next reviewer does not have to rediscover it.

**Verification.** Full fast suite green (1,801 passed), mypy clean across 12 packages.

### D-1 - an operating point is a fitted parameter, and its support is counted

**Symptom.** `threshold_at_precision_<p>` is selected by scanning the test split's own PR curve and the precision at that cutoff is then reported from the same rows.
The measured point bias is small at realistic support and serious without it: at 7 rows above the cutoff a reported 0.78 is a realized 0.65.
`docs/spec-reference.md` described the value as "the smallest score cutoff meeting the precision target", which reads as a guarantee.

**Fix.** `metrics.operating_point_support(y_score, threshold)` counts the rows a cutoff rests on, and `MIN_OPERATING_POINT_SUPPORT = 100` carries the measured table that justifies the number.
`job.py` checks every `threshold_at_*` metric after the report splits are scored, reporting the support on the positive path and WARNING loudly below the floor.
`docs/spec-reference.md` now says a threshold is an estimate, not a guarantee, and that it lands under the target about half the time by construction - which is a property of taking a point estimate of a minimum, not a defect.

**Verification.** Reproduced the corner against the real `compute_metric` before writing the runbook entry: at a 2% base rate with a 0.7 target the cutoff rests on 37 rows, reports 0.703 on test and realizes 0.606 on an independent draw.
That captured output is `docs/troubleshooting.md`'s new entry, as the runbook requires.
Tests cover both halves of the table - the thin corner warns, the `churn_demo` shape does not.

**Not done: the carve.** D-1's second half (select the operating point on rows that are not the reporting rows, at `seed + 7`) is NOT implemented.
`SeedRung.OPERATING_POINT_CARVE = 8` is reserved for it.
The support count converts the dangerous corner into a loud failure, which is what the addendum recommends taking "on its own and immediately"; the carve changes which rows train and what every existing project's threshold resolves to, and it wants its own cycle with its own before/after measurement rather than riding along with an architecture sweep.

### D-2 - the shipped model has the complexity the search chose

**Symptom.** With no declared `validation` split, tuning trials early-stop on a carve that the final fit then reabsorbs (ADR-8, correctly).
So every trial that voted on the hyperparameters stopped early, and the model that ships trains all `n_estimators` rounds.
Nothing downstream can see it: the gates evaluate the shipped model honestly, so it fails only when it is genuinely worse and silently ships a differently-regularized model when it is not.

**Fix.** The second of the addendum's two options, which is the standard move and does not touch ADR-8.
`Capability.BEST_ITERATION` and `SupportsBestIteration` are declared (B-1's machinery); XGBoost and LightGBM implement `best_iteration`.
Tuning collects each trial's rounds - only when the final fit will reabsorb the carve AND `early_stopping_rounds` is set AND the adapter can report them - and `_carry_trial_rounds` sets the final `n_estimators` to their median, dropping the now-inert `early_stopping_rounds`.
The warning's wording says the consequence rather than the missing declaration, as D-2 asked.

**Verification.** End to end with a real XGBoost tuning run: the trials early-stopped between 6 and 10 rounds and the final fit shipped 8, not the declared 400.
A unit test pins the median/drop behaviour and the no-op case.

### D-3 - a reused test window is now visible

**Symptom.** Every training run evaluates its gates against the test window, mbt is built to be run repeatedly, and nothing recorded how many times a `(dataset, window)` pair had judged a candidate.

**Fix.** `mbt/state/gate_log.py` counts evaluations per `(dataset uid, resolved test window)` in `target/gate_history.json` and warns past 20.
It carries no policy, deliberately: the addendum withdrew the enforcement half itself, because `promote` already refuses a `false` verdict unconditionally and `--require-oot-check` is a shipped opt-in whose default is a deliberate policy position.
A malformed or unwritable log reads as empty and never fails a run - the count is a signal, not a contract.

### D-5 - two estimators stopped presenting as more settled than they are

**Permutation importance** now averages `PERMUTATION_REPEATS = 5` shuffles per feature and reports the widest spread across shuffles alongside the result, so a reader can see how much the ranking moved.
sklearn's equivalent defaults to 5 for the same reason.

**The leakage scan** screens numeric columns on `max(|pearson|, |spearman|)` and names which one decided.
Pearson catches linear leakage only; a monotone-but-nonlinear leak - the common shape when a leaked column is a transformed or bucketed version of the label horizon - can sit under the 0.95 bar while being perfectly predictive.
Spearman is `corr` over `rank()`, so it is one query and both thresholds keep meaning what they say.
The rank path made a constant column return NaN where Pearson returned NULL, so both are screened out now: "no variance" is not "perfectly associated".

**Verification.** A strictly increasing, sharply convex relationship is now caught (association > 0.99, reported as `|spearman|`) where Pearson alone would not have flagged it; a linear one still reports as `|pearson|`.

---

## Closing verification

The full CLAUDE.md battery, run at the end of the sweep:

- `uv run pytest -q -m "not e2e" --cov` - **1,867 passed, 71 skipped, coverage 100.00%** (CI's gate).
- `uv run pytest -q -m e2e --timeout 1800` - **97 passed, 6 skipped** (includes the JVM tier: Spark and H2O).
- `uv run ruff check . && uv run ruff format --check .` - clean.
- `uv run mypy` over all 12 packages, strict - **153 source files, no issues**.
- `uv run pre-commit run --all-files` - all hooks pass.
- `uv run mkdocs build --strict` - clean.
- `uv run yamllint` over packages, examples, fixtures and workflows - clean.
- `uv run python scripts/audit_dependencies.py` - clean; the 7 accepted advisories all still fire.

Two live defects were reproduced through the real CLI or a real run before and after the fix (defect 3's `mbt ls --select state:modified`, D-1's thin-support corner), and the A-3 compile-hole regression test was confirmed to fail with the fix removed.

## What was deliberately not done

- **D-1's operating-point carve.** Reserved as `SeedRung.OPERATING_POINT_CARVE`; see the D-1 entry.
- **Deleting `mbt.contracts` (C-6).** Both costs the finding names are fixed, but the module is the public import path stamped into every scaffolded project; see the C-6 entry and the module docstring.
