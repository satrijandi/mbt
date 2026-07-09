# ADR-18: Champion gates decide on a paired-bootstrap lower bound

**Status:** accepted

A point-estimate delta promotes challengers on test-set noise: with the small test windows temporal splits produce, `delta >= min_delta` is not evidence of improvement.
Champion gates therefore run a seeded paired bootstrap on the pinned test split: each resample draws rows with replacement once and scores both models on those same rows, so sampling noise cancels and only the model difference remains (ADR-9 already guarantees both models were evaluated on the identical split).
The gate passes only when the one-sided lower confidence bound of the delta clears `min_delta`.

Defaults are 95% confidence and 1000 resamples, configurable per gate via `confidence` and `bootstrap_resamples`; `confidence: null` opts back into point estimates.
The bootstrap seed derives from the model seed (train: `seed`, tuning: `seed + 1`, implicit validation carve: `seed + 2`, bootstrap: `seed + 3`), so bounds are byte-reproducible.

Slice gates and hook-metric gates keep point comparisons; core has no per-example predictions for them.
Resamples that collapse to a single class are skipped; if every resample degenerates, the bound falls back to the point delta and the gate result message says so.
