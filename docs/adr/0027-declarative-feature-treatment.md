# ADR-27: Declarative feature treatment - transforms, monotone constraints, declared categoricals

**Status:** accepted

## Context

Two gaps kept landing on the same YAML block, and both were being answered with prose.

**Time-anchored features drift by construction.**
`days_since_onboard`, `tenure_days`, `recency_days` are genuinely predictive and genuinely non-stationary: every surviving row's value grows a day per day, so the serving distribution translates away from the training one every month and PSI/KS fire forever.
mbt's only answer was `docs/ds-primer.md`, which told the DS to put such columns in `features.exclude` - throwing away real signal to buy a quiet monitor.
The levers that actually fix this are small and well known: cap the plateau, compress the tail, rank within the batch, and constrain the model's response direction so it cannot invert once the population ages past the training range.
None of them were expressible.

**Categorical handling was inferred, never declared.**
`mbt_adapter_base.encoding` keys off arrow dtype: string is categorical, numeric is a magnitude, anything else is an error.
An int-coded category (`contract_code`: 0 = month-to-month ... 3 = two-year) therefore trains as an ordinal magnitude, which silently costs whatever part of its effect is non-monotone in the code.
The showcase worked around this by hand, in `examples/showcase/project/models/wide_hooks.py`, whose docstring said so: *"mbt has no `categorical:` spec field on purpose - adapters infer categoricals from dtype - and this hook is the sanctioned seam for numeric-coded ones."*

## Why this belongs in the spec, given that we twice said it did not

`design-history/PRD.md` lists "feature engineering DSL" as a v0 non-goal, bounded as *"feature engineering lives in the dataset source (dbt/warehouse) or `hooks.py`"*, and ADR-15 §4 says non-numeric features are an actionable error - *"exclude them or encode via hooks (no hidden encoding in v0)"*.

Both still hold, and this ADR does not touch them.
Deriving columns, joining tables, and business logic stay in the warehouse or in `hooks.py`; nothing here computes a new feature.
What this adds is a **closed five-keyword vocabulary over columns that already exist** - `cap`, `log`, `percentile`, `monotonic`, `categorical` - and it belongs in the spec for reasons a hook cannot satisfy:

- **It must be identical at train and at score time.** A hook already is, but only because the hook file's bytes are hashed and the champion's `mbt.hooks_hash` is checked; the spec gets the same guarantee from `config_hash` for free, with no parity machinery.
- **It must be checkable against the adapter before anything runs.** "This adapter cannot enforce a monotone constraint" is a `mbt parse` error here. A hook is opaque Python: mbt cannot know a constraint was intended, so it cannot warn that it was not applied.
- **It must be legible.** The model card can say "`tenure_days`: cap at 730, log1p, monotone increasing". It cannot say what an arbitrary hook did.
- **It must reconcile with the shift monitors.** The monitoring baseline is built from the post-treatment train split, so treatment and monitoring see the same numbers by construction, and the DS can reason about why a PSI dropped.

The precedent is `calibration:` (R2-8): also modelling logic, also expressible in a hook, also promoted to a closed spec enum because it had to compose with tuning, the walk-forward folds, and the seed ladder.
The line we are holding is *closed vocabulary, not open DSL*.
`cap: p99` is refused for exactly this reason (see "Rejected").

## Decisions

1. **Three sibling keys under `features:`, not one blended block**, because they act at three different seams: `categorical` retypes a column, `transforms` rewrites its values, `monotonic` constrains the model rather than the data.
   `monotonic` may also be written inline inside a `transforms` entry, since a DS configuring one column should not have to write it in two places; both spellings resolve to one canonical map, and disagreeing spellings are a parse error.

2. **Numeric treatment is stateless, and applies in a fixed order: `cap` -> `log` -> `percentile`.**
   `cap` takes declared constants (`365`, or `{min: 0, max: 365}`), `log` is `log1p`, and `percentile: batch` ranks the value within whatever split or batch is being read, into (0, 1], with ties taking the group's average rank.
   No reference distribution is fitted, so train and score agree without a side-car artifact, and no new plumbing joins the training job to the scoring job.
   Ranking against the batch itself is the point: every subscriber ageing by the same amount moves every value together, so the ranks are unchanged and the feature is stationary by construction.
   `percentile` is a `Literal["batch"]` rather than a boolean so `train` stays reserved for the fitted variant; `log` alongside `percentile` is a parse error because a rank is invariant under any monotone transform.

3. **All four transforms are monotone increasing**, so a declared `monotonic` direction survives them and the two blocks compose without a caveat.

4. **Treatment applies in `TransformedDatasetHandle`, after `features.include/exclude`.**
   That is the single choke point every path already runs through - training, the validation and calibration carves, each walk-forward fold, and scoring - and `_materialize_for_path_adapter` stages its output to parquet, so Spark and H2O receive treated data without knowing the feature exists.
   Declared slice columns ride along for evaluation but are not features, so they are never treated.

5. **`features.categorical` absent infers; present is authoritative.**
   Omitting the key keeps dtype inference exactly as before, so no existing project changes behaviour.
   Once the key is present the list is the whole truth: a listed column is categorical even when int-coded, and an undeclared string feature is a hard error naming the column.
   That is what makes `categorical: []` a meaningful assertion ("this model has none") rather than a synonym for omitting the key.

6. **Core retypes declared categoricals to string**, which is exactly what every adapter's dtype inference already keys off - so LightGBM's native categoricals, sklearn's per-family ordinal-vs-one-hot split, H2O's `asfactor` and Spark's `StringIndexer` all pick them up with no adapter change.
   This is the `wide_hooks.py` trick, promoted from a showcase workaround to a spec field.
   The stateless level policies ride along in core: `levels` pins the level set in the spec, `null_as_level` maps NULL to an explicit `__missing__`, and `max_levels` fails a column with too many distinct train levels (the guard that catches an identifier declared categorical by accident).

7. **`min_frequency` is the one policy that lives adapter-side**, in `mbt_adapter_base.encoding`.
   Pooling a rare tail needs the train-fitted level map - and that map is *already* computed by `train_categories`, already persisted with the artifact, and already reused by `categorical_codes` at predict time.
   Putting pooling there adds no new state; putting it in core would have required inventing some.
   Once `__other__` is a level, a value unseen at train time codes to it instead of to NaN, which is what asking for a pooled bucket means.

8. **A declaration the adapter cannot honour fails at `mbt parse`.**
   `supports_monotonic_constraints` and `supports_categorical_pooling` are probed on the adapter class beside the existing `supports_calibration` probe; sklearn answers the blunt yes and then refines it in `validate(spec)`, because only `hist_gradient_boosting` takes `monotonic_cst`.
   Spark and H2O declare both False explicitly rather than relying on the `getattr` default, with the reason recorded next to the flag.
   A constraint the DS believes is protecting them but that was silently dropped is worse than no constraint.

## Rejected

**`cap: p99` and `percentile: train`** - the fitted variants.
Both would need a reference distribution computed on train, persisted next to `baseline.json`, carried on the registry version as tags, and re-read by the scoring job: the full ADR-21 plumbing, for a semantics that is strictly *worse* at the problem this ADR exists to solve, since a fixed mapping does not cancel a population shift.
The names are reserved and the escape hatch is `hooks.py`, which can compute anything and is hashed.

**Blending `monotonic` into `transforms` alone.**
It is not a transform; it never touches the data. Keeping it addressable as its own map is what makes the parse-time capability check obvious and the model card honest.

**Warning instead of failing on an unsupporting adapter.**
Considered for adapter-swap ergonomics and rejected: a warning in a CI log is not a control, and the whole value of a monotone constraint is that you can rely on it.

**`min_frequency` computed per batch,** matching `percentile: batch`.
Rejected as a correctness hazard rather than a stability feature: the level-to-code map is fixed by the artifact, so a level common in train but rare in one batch would silently move rows to a different code. `levels` is the stateless way to pin a level set, and it is a *declared*, hash-visible decision rather than a data-dependent one.

## Consequences

- **Every model's `config_hash` flips once on upgrade**, because specs hash their full field set and `FeatureSelection` gains three keys (ADR-7 names this case explicitly; ADR-16 and ADR-25 each recorded the same one-time flip).
  `state:modified` flags every model for one cycle; `tests/golden/churn_demo_manifest.json` was regenerated, and the diff is exactly the three new keys plus the hash cascade.
- **`percentile: batch` blinds the shift monitor on that column** - both sides become uniform ranks, so PSI goes to roughly zero by construction.
  That is the trade, stated plainly: stability bought with monitorability. A DS who needs to see the raw column move should keep it under the scoring node's `checks:`, or not rank it.
- **Capping is not free**, and the `revenue_demo` fixture is calibrated to show that rather than to hide it: `cap: 730` on `tenure_days` moves test rmse from 7.6 to 8.1 against a 12.0 ceiling, because that fixture's generative signal really is linear to 1000 days.
- **`min_frequency` is unavailable on Spark and H2O**, which is a real gap and a parse error rather than a silent difference. It sits beside the existing F24 categorical-parity caveat in the mbt-spark README: ordinal indexing is a different model family, and a level map pooled elsewhere is not the one `StringIndexer` indexes against.
- The showcase's `wide_hooks.py` becomes redundant for its stated purpose and is retired in favour of `categorical: [contract_code]`.
