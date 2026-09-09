# ADR-26: One tracking experiment per node kind

**Status:** superseded by [ADR-28](0028-mlflow-training-only-and-champion-carried-config.md)

ADR-28 removed the serving-side tracking runs this ADR existed to separate:
`mbt score` and `mbt monitor` no longer open runs at all, so there is one kind
of record left and one namespace to hold it. The `experiment` key survives as
a single name, composed with the project name; the mapping shape below is now
rejected at construction. Everything here is kept for the reasoning, which is
what ADR-28 argues against.

## Context

Three code paths open a tracking run: training a model, scoring a batch, and
evaluating a matured prediction run against ground truth (ADR-21).
All three called one `_experiment_id()` resolved from a single `experiment`
key in the tracking adapter's config, defaulting to `mbt` - so every run mbt
has ever produced landed in one experiment.

That default conflates two different kinds of record.
A training run is an experiment record: a spec, its hyperparameters, its
offline metrics, compared against sibling attempts and kept because the
comparison is the point.
A scoring run is a production record: one batch, its row count, its shift
statistics, and the champion version that produced it.
The monitor run is a production record too - realized metrics for a
prediction run that has matured.

The practical cost is that the second kind grows with serving cadence while
the first grows with modelling work.
A model retrained monthly and scored daily produces 12 runs of the first kind
and ~730 of the second per year, in one namespace, sorted together, with the
tracker's comparison UI - built for the first kind - defaulting to a view
dominated by the second.
The showcase makes this concrete: `make snowflake` runs build, promote,
score, and monitor through a single target, so its four runs stacked into one
`mbt` experiment with two of them sharing a name (the monitor run reuses the
scoring node, hence its name).

The industry split is sharper than mbt's was.
MLflow's own documentation defines a run as "an execution of some piece of
data science code" and never addresses production inference; Databricks'
reference MLOps architecture routes batch predictions to tables and monitors
those, keeping MLflow out of the inference path entirely.
mbt already agrees on the load-bearing half - predictions go to a prediction
store, never into the tracker (ADR-21) - and what remains in the tracker is
job-level metadata: a pointer plus a summary.
Keeping that is deliberate.
Realized production metrics for a model version, sitting in the same store as
that version's offline metrics, is the comparison model governance actually
asks for, and losing it would mean answering "did this champion hold up" with
a join across two systems.
The defect was the shared namespace, not the logging.

## Decisions

1. **The experiment resolves per node kind.** `start_run` picks the
   experiment from `node.resource_type`, so a model's runs and a scoring
   node's runs are separately addressable.
   Monitor runs reuse the scoring node and therefore follow it: one
   experiment holds everything that happened in production for that scoring
   node, which is the question an operator asks.

2. **Defaults separate; an explicit name collapses.** Unset means `mbt` for
   models and `mbt_serving` for scoring.
   `experiment: <name>` sends every kind to that one name - both the
   pre-ADR-26 behavior and the supported way back to a single experiment.
   `experiment: {model: ..., scoring: ...}` names them individually; a kind
   left out of the mapping keeps its default.
   The training default stays `mbt` so an existing store's training history
   does not move.

3. **A wrong kind is an error, not a fallback.** An unrecognized key in the
   mapping (`models:` for `model:`) raises at construction, and a scalar that
   is neither a name nor a mapping is rejected rather than stringified.
   Both would otherwise present as "the override silently did nothing", found
   weeks later by someone wondering why an experiment is empty.
   A node kind that produces no runs (`dataset`) is likewise an error rather
   than being filed under training.

4. **`prepare()` creates every configured experiment.** It already existed to
   keep parallel jobs from racing on sqlite migrations; with more than one
   experiment in play it must create all of them up front, or two jobs of
   different kinds race to create theirs.

5. **Trials follow their parent.** Nested tuning runs are training-time by
   construction (ADR-8: the trial loop runs inside the training job), so they
   resolve to the model experiment rather than re-deriving one.

## Consequences

- Upgrading an existing store splits history at the upgrade point: scoring
  and monitor runs written before it stay in `mbt`, later ones land in
  `mbt_serving`.
  Nothing moves and nothing is lost; setting `experiment: mbt` restores the
  single namespace.
- A store that has only ever trained now shows an empty `mbt_serving`
  experiment, created by `prepare()`.
  That is the price of not racing.
- Registered model versions link to runs by id, and MLflow does not require
  the version and its run to share an experiment, so registration and the
  champion/challenger lookups are unaffected.
- Experiment names live in `profiles.yml`, which is excluded from node
  identity (ADR-5), so none of this can mark a node `state:modified`.
- Plane separation stays a different axis: the showcase keeps its warehouse
  runs distinct from its lake runs by registered-model suffix and artifact
  prefix, not by experiment.
  Per-plane experiments remain available by setting `experiment` on that
  target.
