# ADR-28: Tracking is training-only, runs are timestamped, and the champion carries its own inference config

**Status:** accepted

**Supersedes:** ADR-26. **Amends:** ADR-20 (§ champion resolution), ADR-21 (§ realized metrics).

## Context

Three things were true of mbt's tracking output at once, and each made the
other two harder to see.

**Three code paths opened runs, and two of them were not experiments.**
Training a model, scoring a batch, and evaluating a matured prediction run all
called `start_run`.
ADR-26 separated the second and third into their own experiment on the
argument that a production record and an experiment record are different
kinds, but kept both logging on the argument that realized metrics beside
offline metrics is the comparison governance asks for.
That reasoning held for the monitor run and never held for the scoring run: a
batch summary is a row count and two shift statistics that the prediction
sidecar already records, written once per batch, forever.
The separate namespace made the volume tolerable rather than making the
records useful.

**Runs were named after the node, so they collided by construction.**
`mlflow.runName` was `node.name`.
Train `churn_wide_automl` twice and MLflow holds two runs called
`churn_wide_automl`, distinguishable only by opening them and reading
`mbt.run_id`.
The showcase's `mbt_serving` experiment showed the sharper version of the same
defect: the score run and the monitor run reuse one scoring node, so both were
named `wide_retention_scoring`.

**Nothing named the project, and the experiment was named after the tool.**
The default experiment was the literal string `mbt`.
A tracking server holding two mbt projects had no way to tell them apart, and
the showcase's two data planes - the lake plane and the Snowflake plane -
wrote into one experiment, separated only by a `_snowflake` suffix on the
registered model name.

Underneath all three sat a fourth problem that only appears in a real
deployment.
`log_artifact` fires only for a `file://` artifact ref, because a model binary
belongs in mbt's artifact store with a pointer in the tracker (ADR-21).
The showcase's artifact store is `s3://`, so every run it has ever produced
carries tags and metrics and *no documents at all*.
The tracker recorded that a model was trained without recording what was
trained.

## Decisions

1. **Only training opens a tracking run.**
   `mbt score` writes to the prediction store and `mbt monitor` to the
   ground-truth ledger; neither touches the tracker.
   The coordinator does not even build a tracking adapter for an invocation
   with no model nodes, so a scoring cadence cannot fail on a tracker outage
   and does not create an experiment on a store that has never trained.

2. **The experiment is `<project>__<experiment>`, composed in core.**
   The project is `name:` in `mbt_project.yml`; the experiment is `experiment:`
   in the target's tracking config, which the profiles pre-render already lets
   you write as `env('EXPERIMENT_NAME')`.
   With no `experiment:` the project name stands alone, so a fresh project's
   runs land under their own name rather than under `mbt`.
   Composition lives in `mbt.runtime.tracking_adapter_config`, not in each
   adapter, so the rule is stated once and applies to every tracker.

   **Amended: the separator was a single underscore until `EXPERIMENT_SEPARATOR`
   landed.**
   Both halves are themselves snake_case, so one underscore left the boundary
   unreadable: `LOAN_APPLY_PROPENSITY_V1_0_0` does not say where the project
   name ends, and `a_b` + `c` collides with `a` + `b_c` on a single name.
   Two underscores make the boundary legible.
   They do not make it recoverable - a project name may itself contain `__` -
   so nothing should parse a composed name back apart.
   The same change relaxed `ProjectConfig.name` to `^[A-Za-z][A-Za-z0-9_]*$`,
   because the project half is an org-facing label once it reaches the tracker
   and forcing it lowercase forces a casing the org does not use.
   Resource names stay lowercase: the project name is only a `unique_id`
   segment, while a resource name is what a selector matches.

3. **The run name is `<run_id>-<model>`, also composed in core.**
   `run_id` is the coordinator's existing per-invocation id
   (`20260909T101500Z-a1b2c3d4`), which already prefixes the artifact store and
   already tags every run.
   Core passes the composed name as `mbt.run_name` and each adapter maps it to
   its own convention - MLflow onto `mlflow.runName`.
   A run is therefore unique by construction, sorts chronologically, and says
   which model it is without being opened.

4. **Every training run exports an inference config, and scoring reads it.**
   `inference_config.json` goes to the artifact store beside the model and is
   pinned on the registered version as `mbt.inference_config_uri` and friends -
   the same shape ADR-21 gave the monitoring baseline, reused rather than
   reinvented.
   It carries the node's rendered spec verbatim, so `ModelSpec.model_validate`
   reconstructs exactly what was trained and `config_hash` re-verifies it, plus
   the resolved feature column list, which the manifest cannot answer at all:
   `features.include: ["*"]` does not say which columns the model was fit on.
   `mbt score` builds its `ModelSpec` from that document rather than from the
   local manifest.

5. **A spec divergence warns; a missing config falls back.**
   A promotion is deliberately outside node identity (ADR-5), so the working
   tree and the champion legitimately differ between a retrain and a promote.
   Scoring uses the champion's spec and says which two hashes disagreed.
   A champion registered before this ADR has no config to read, so scoring
   falls back to the local manifest with a loud warning - the same shape
   ADR-21 gave a champion registered before baselines existed.

6. **`hooks.py` is not shipped with the champion.**
   Hooks are arbitrary Python that runs inside the feature path at train and
   score time.
   Everything declarative now comes from the champion, but the hook file keeps
   being read from the git checkout and `_check_hooks_parity` keeps hard-failing
   on a hash mismatch.
   The source is logged to the training run as a readable record, so the run
   describes the model completely, but mbt does not execute code it fetched
   from a registry.

7. **`log_document` is an optional tracking capability.**
   Probed with `hasattr`, like `prepare` and `log_trial`.
   It uploads a local file mbt wrote itself, which is what puts the inference
   config and the hooks source on a run whose model artifact lives in `s3://`.
   The existing `file://`-only branch in `log(artifacts=...)` is unchanged: it
   is about model binaries, which deliberately stay out of the tracker.

## Rejected

**Keeping monitor runs while dropping score runs.**
The realized-metrics-beside-offline-metrics argument is real, and this is the
decision with the least comfortable trade in the ADR.
It loses the production-performance time series from the tracker.
What it does not lose is the data: the ground-truth marker in each prediction
run's directory already records realized metrics, coverage, matched rows, the
champion version and the gate outcome, and `mbt predictions show <run_key>`
reads them back.
The line drawn here is that the tracker holds what was *tried*, and the
prediction store holds what *happened*, with no path writing to both.
A split rule ("this production record yes, that one no") is the thing that
made ADR-26 hard to explain.

**Shipping hooks with the champion.**
It is what would make checkout-free scoring true for every model rather than
only for hook-free ones.
Rejected because it moves the trust boundary from "a merged pull request" to
"anyone who can register a model version", and hash verification does not help:
it proves the file is the one that was registered, not that the registered one
is safe.

**Moving model bytes into MLflow.**
The most literal reading of "everything in MLflow", and it would make the
artifact store unnecessary at score time.
Rejected as a much larger change - it re-points `export`/`load` across six
training adapters and orphans mbt's artifact GC - for a benefit the pointer
already delivers, since resolving a champion means reading the registry anyway.

**Fetching the inference config through the tracking adapter.**
Would make the tracker load-bearing for scoring, so a tracker outage would
break a scoring cadence that today does not depend on it, and every
third-party tracking adapter would have to grow a read method.
The registry tag plus artifact store reuses plumbing that already exists and
that scoring already touches to load the model.

**Composing the run name inside each adapter.**
Deriving `<run_id>-<node>` from `mbt.run_id` in the MLflow adapter worked, but
it would have to be restated in every other tracker, and the experiment name is
already composed in core. One naming policy, one place.

## Consequences

- **Upgrading splits history at the upgrade point.** Existing runs stay in
  `mbt` and `mbt_serving`; new training runs land in `<project>` or
  `<project>__<experiment>`. Nothing moves and nothing is lost. There is no
  config that reproduces an arbitrary old name: `experiment:` supplies the
  second half only, and core always composes, so the remedy is to RENAME the
  old experiment to the composed name before the first build after upgrading -
  mbt resolves by name and adopts it, history intact
  (`docs/troubleshooting.md`). The separator change from `_` to `__` splits
  history a second time, for targets that set `experiment:` only; a project
  with no `experiment:` key is unaffected.
- **A per-node-kind `experiment:` mapping is now an error**, not a value mbt
  silently picks one name out of. The message names ADR-26 so the fix is
  obvious.
- **Champions registered before this ADR score with the project's current
  spec**, warning on every run until a retrain and promote. That is the same
  migration ADR-21's baselines had.
- **Artifact GC keeps the config with its model for free.** `mbt clean
  --artifacts-older-than` prunes whole run prefixes and keeps any prefix
  holding a champion-referenced file; the config, the baseline and the model
  share one prefix, so a champion can never age into a state where its weights
  survive but the spec that feeds them does not. Reusing ADR-21's placement
  rather than inventing one is what buys this.
- **`mbt score` still needs the project checkout**, for the scoring node's own
  spec (input, filters, window, checks, output, monitors, ground truth) and for
  `hooks.py`. What moved to the champion is the model side.
- **The anchor became a first-class `TrainingJob` field.** It had been riding
  in `tracking_meta`, which is why removing tracking from the score job would
  otherwise have silently emptied the prediction sidecar's `scored_at` and
  broken maturity. Metadata a job needs is job payload.
- Experiment and run names live in `profiles.yml` and in the invocation, both
  outside node identity (ADR-5), so none of this can mark a node
  `state:modified` and no golden manifest moves.
