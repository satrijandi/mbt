# ADR-30: The training report - an after-test window, same-slot periods, and stability judged against the test set

**Status:** accepted

**Amends:** ADR-4 (§ what `config_hash` covers), ADR-6 (§ which evaluation edits retrain), ADR-21 (§ raw rows beside a model), ADR-28 (§ which paths write to a tracking run).

## Context

A data scientist deciding whether to deploy a candidate asks five questions that a training run could not answer.

**How did the pipeline run?**
The run held tags, test metrics, the inference config and the hooks source.
The events that explain a result - split sizes, AUTO resolutions, tuning trials, carve sizes, warnings - went to a terminal and were gone.

**How do the scores rank and separate?**
Every number on the run was a scalar.
There were no row-level predictions, no bin table, no positive rate per score band, and no score distribution.
`lift_at_0.1` answers one point on a curve the reviewer needs whole.

**Did the scores and features stay stable after the test window?**
This is the question the deployment actually turns on.
A temporal split puts the test window where labels have matured, which is usually months before the day the model ships.
Nothing looked at the rows between the end of the test window and now.
mbt had the machinery - PSI and KS against a baseline, with Benjamini-Hochberg control (ADR-20, ADR-21) - but only a scoring run used it, after deployment.

**Does performance hold on labels that arrived since?**
The same months carry labels as they mature.
A model whose AUC held on the test month and then sagged month over month is a model that should not ship, and no report showed it.

**What must serving reproduce?**
The model's own config reached the run as a JSON document; the dataset's never reached the job at all.
Parameters held only hyperparameters, so the tracker's comparison view could not show a filter, a window or a feature treatment.

A sixth gap surfaced while building the answer.
`evaluation.protocol.test_window` was documented as narrowing the dataset's test window and was resolved at compile time, but nothing in execution applied it.
A model declaring `test_window: "-7d:now"` was evaluated on the full 28-day test split, with identical metrics - reproduced on `churn_demo` before the fix.

## Decisions

1. **A dataset may declare an after-test window, `split.out_of_time`.**
   It is temporal only, resolves against the anchor like every other window (ADR-12), and must start at or after the end of the test window.
   Parse rejects an overlap when both bounds are the same kind; compile rejects it once a mixed relative/absolute pair has an anchor to order against.
   Every data adapter already iterated the resolved windows generically, so the split materializes with no adapter-specific SQL.

2. **The after-test split is the one split allowed to be empty or unlabelled.**
   A window ending at `now` is routinely empty until upstream catches up, so an empty `out_of_time` warns instead of failing the build, in all three data adapters.
   Its rows may carry a null label for an outcome not yet observed, so `not_null` skips the label column there, and `row_count` and Python data tests read only the labelled splits.
   Feature-level checks and `no_future_columns` still cover it.

3. **Training never sees it.**
   A training view hides the split from everything the training path enumerates - the fit, the validation and calibration carves, path-adapter staging and hooks.
   This is ADR-8's guarantee applied to the rows reserved for judging the model after its test window.
   The same view is where `test_window` now narrows the test split, which closes the gap described above.

4. **"Same slot versus test" is the period semantics.**
   `month` compares the whole test window with each calendar month after it.
   `week_of_month` compares W1 (days 1-7) of the test month(s) with W1 of every later month, then W2 (8-14) through W5 (29-31).
   `day_of_month` compares day d with day d.
   Comparing like slots keeps within-month patterns - paydays, billing cycles - from reading as drift.
   A grain finer than the time column's resolution is skipped with a log line: a monthly panel has no day-of-month view.
   A cell counts toward performance only when it is mature (`cell_end + label.horizon <= anchor` when a horizon is declared, else its label is present), has both classes, and has enough rows; stability uses every row.

5. **Stability is judged against the test set, with mbt's own statistics.**
   The reference is a baseline built from the test split, not the ADR-21 training baseline, because the question is whether the population moved after the model was evaluated.
   `evaluation.stability` reuses the scoring monitors' `feature_shift` and `prediction_shift` shapes and their evaluator, Benjamini-Hochberg included, so "stable before deploy" and "stable in production" are one definition.
   Evidently, when chosen, renders its report beside mbt's tables and never decides pass/fail: a second, differently-parameterized verdict would make the gate depend on which engine was installed.
   Report engines are a seventh plugin role (`AdapterPlugin.reporting`, contract 1.2), and Evidently ships as its own package, `mbt-evidently`, so its dependency tree - a web server stack, statsmodels, nltk, `plotly<6` - never reaches a project that does not ask for it.
   An engine's version stays out of the environment digest, and an engine that fails is a warning on the report, never a failed build.

6. **Reports are always produced; gates are opt-in.**
   `evaluation.gates` gains `source: out_of_time` with a `period` grain and a `min_rows` floor, and the worst mature cell decides.
   A gate with no mature cell passes with a warning and is recorded as not applicable - the ADR-10 stance - so a model built before its after-test labels mature is not blocked by the calendar.
   Promotion can demand a real pass separately.

7. **`evaluation.report` is presentation, and is not identity.**
   Binning strategies, the period grains shown, importance depth and the report engine decide what the report shows and nothing else.
   `config_hash` leaves the block out, as it leaves out `description`, `owner` and `tags`, so re-binning a report never retrains a model.
   Everything that decides pass/fail - gates, `evaluation.stability`, `split.out_of_time` - stays hashed, which is ADR-6 unchanged.

8. **Four binning strategies, edges frozen on the test split.**
   `quantile` (equal-frequency, default 10), `fixed_width` (default 0.05 on a probability), `custom` (declared edges, with default risk bands on a probability), and `top_percent` (cumulative cutoffs with capture, lift and precision).
   Edges and cutoffs fitted from data are fitted once, on test scores, and applied unchanged to train, test and every later cell, so a row in the table means the same score range everywhere.
   `binning: all` renders the four.
   For regression, "positive rate" reads as mean label per bin, and custom edges must be declared because a prediction has no scale to default to.

9. **Row-level predictions are opt-in.**
   `report.predictions.enabled` writes keys, time, split, label and class probabilities per row to the run, optionally capped by a deterministic hash of the sample key.
   ADR-21 declined raw samples beside a model for size and data retention, and rows keyed by `sample_key` are exactly that concern, so the choice belongs to the project rather than to a default.

10. **The model and dataset configs become flat tracking parameters.**
    The resolved model spec - after AUTO resolution and tuning - and the dataset spec with its resolved windows are flattened under `model.` and `dataset.`, beside the existing bare hyperparameter keys, which stay so run history remains comparable.
    User-keyed maps are encoded as one JSON value because a column name is not always a valid parameter key, and values past the tracker's length limit are truncated by mbt with a marker, the full value living in the logged config documents.

11. **A pre-deploy check appends to the training run it validates.**
    `mbt evaluate --out-of-time` rebuilds the version's dataset with its recorded test window and an after-test window from that test end to the new anchor, runs the same report code with the version's own spec, and appends the result under `evaluations/<run_id>/` on the version's training run.
    It never logs parameters - they are immutable and its windows differ - and it records its verdict on the model version.
    This amends ADR-28's "only training opens a tracking run": the check opens nothing, and a record of whether a trained candidate still holds is part of what was tried.

## Rejected

**Judging stability against the training baseline.**
It is what the scoring monitors do, and reusing it would have been free.
It answers a different question: whether the serving population resembles the one the model learned from.
Before deployment the question is whether anything moved after the model was evaluated, and the test set is the population that evaluation used.

**Consecutive-period comparisons as the verdict.**
Month against the previous month shows a trend, and every step can be small while the total is not.
Same-slot-versus-test measures the distance from the evaluated state directly.

**Letting Evidently's drift share gate.**
It would make the verdict depend on an optional plugin's version and presets, and disagree with the production monitor on the same data.

**Hashing the report block.**
It would retrain every model whose owner changed a bin width.
The report config is recorded on the run the report belongs to, which is where a reviewer reads it.

**Failing an after-test gate that has nothing mature to judge.**
Every build between a model's training and its labels maturing would fail for a reason no one can act on.

## Consequences

- **Every node's `config_hash` flips once on upgrade.** New optional fields enter the rendered dump even while unset, exactly as ADR-16, ADR-22, ADR-27 and ADR-29 recorded; `state:modified` flags every model for one cycle and the golden manifest regenerates.
- **`test_window` now changes results.** A model that declared it has been evaluated on its whole test split until now; its metrics move on the next build to the ones its spec always described.
- **A remote job image must upgrade with the coordinator.** The job payload gains fields and interchange models reject unknown ones, so a skewed image fails loudly rather than silently dropping the report.
- **Positional alignment is assumed.** Keys and times are aligned with predictions by row position, the rule scoring already enforces; a report that needs them requires row-stable hooks, and a hook that reorders without changing the count cannot be detected.
- **Reports cost time and space.** Predicting train and the after-test window adds a predict call each, and Evidently HTML runs to megabytes per page, which is why its page count is capped.
- **One more package to release.** `mbt-evidently` needs its own PyPI project and trusted publisher before a tag can publish every package, and Evidently's own releases are ceilinged below 0.8 until the snapshot parsing is re-verified.
