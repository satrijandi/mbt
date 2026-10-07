# ADR-31: Reads take a column projection hint; dataset builds stay full-width

**Status:** accepted

**Amends:** ADR-15 (the data adapter contract: 1.3 adds `DataBuildContext.columns`).

## Context

Every read a data adapter performs used to write every column of its relation.
For a dataset that is the training set; for `mbt monitor` it is the matured labels, read through `build_scoring_input` and narrowed to the join keys and the label afterwards.
The showcase's lake table made the cost visible: at `SCALE=huge` it is 320 columns wide, and each `mbt monitor` run that evaluated spent two to three minutes materializing all of them to keep three.

The same width reaches dataset builds, so the obvious fix is to project those too: the union of the columns the consuming models name.
That fix is wrong, for a reason ADR-9 already fixed in place.
A champion gate re-scores the production model on the challenger's test split, inside the job, from the dataset the challenger was built from.
That model was trained from an earlier spec, and nothing stops the current spec from having dropped a feature it reads.
A dataset projected to the current specs would then lack a column the champion needs, and the gate - the decision that guards production - would fail on a missing column instead of comparing models.
Hooks (`features.include` matches the post-hook column set), Python data tests, and the after-test check against an older registered version all have the same shape: a reader whose columns the dataset node cannot know at build time.

## Decision

1. `DataBuildContext` gains `columns: tuple[str, ...] | None` (contract 1.3), a projection hint.
   When it is set, an engine writes only those columns, in the relation's own order, applying them last so filters, sampling and split windows still read the whole relation.
   A named column the relation lacks is a build failure, worded once for every engine by `materialization.projected_columns`.
   `DataAdapterCompliance` pins the behavior for every engine: order, the window still filtering on a projected-away column, and the missing-column failure.
2. Core sets the hint only where it selects those columns afterwards anyway, so an engine that ignores it (a 1.2 plugin) stays correct and only reads more than it needs.
   Today that is one caller: `mbt monitor`'s label read, which asks for the ground truth's join keys and label column.
3. Dataset builds stay full-width.
   The dataset node cannot know every reader of its materialization (decision context above), so it does not guess; a project whose relation is far wider than its models should narrow it upstream, where ADR-29 puts the panel's shape.

## Consequences

- `mbt monitor` scans three columns of a wide label table instead of all of them; on the showcase's huge table the evaluating monitor run no longer pays minutes for the width.
- A 1.3 data plugin needs a 1.3 core (the registry refuses a newer minor); a 1.2 plugin keeps loading.
- Scoring inputs are not projected either, for the same reason as datasets: the scored champion's columns are resolved at run time and hooks may read raw columns.
- The Snowflake adapter's `base_relation` lost its never-populated project-away list; the projection is the one way to narrow a select.
