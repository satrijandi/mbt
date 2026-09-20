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

## Progress log

Nothing has been worked yet.
One appended entry per completed item, carrying symptom, fix, verification and docs, per the shape the four earlier cycles use.
This file moves to `design-history/reviews/feedback-v5.md` when the log closes.
