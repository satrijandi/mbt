# ADR-24: Regression as a second task vertical (name-dispatched metrics)

**Status:** accepted

v0.1 shipped one task vertical, binary classification. Regression was a declared stretch goal, and the architecture was pre-wired for it: a `TaskType` enum with `REGRESSION`, a task-schema registry (`register_task_schema`), `param_model(task)`, and per-adapter `supported_tasks`.
This ADR adds regression end to end for the Python adapters (XGBoost, LightGBM) without disturbing the binary path.

## Metrics dispatch on the name, not the task

The pivotal decision: the metric engine dispatches on the **metric name**, not on a task passed down through the adapter layer.
The binary and regression builtin name sets are disjoint (`roc_auc`/`pr_auc`/… vs `rmse`/`mae`/`r2`/`mape`), so `compute_metric(spec, y_true, y_score)` selects the engine from `spec.name` alone.
This matters because an adapter's `evaluate()` receives metric specs and a score column but no task - threading a task through every adapter (and the `MetricSpec`, which has no task field) would have been invasive and error-prone.
Instead `evaluate_split` → `compute_results` → `compute_metric` route each metric to `compute_binary_metric` or the new `compute_regression_metric`, and the paired-bootstrap champion gate (ADR-18) calls the same dispatcher.
The single prediction column already carries a probability for binary and a target-scale value for regression, so scoring, monitoring (PSI/KS on continuous scores), and the prediction store need no change at all.

A rejected alternative was a `task` argument on `evaluate()`/`compute_results`.
It is more explicit but forces every adapter and the interchange `MetricSpec` to carry a task, for no behavioral gain over name dispatch given the disjoint name sets.

## Direction and validation

Metric direction lives in one place (`quality/metrics.LOWER_IS_BETTER`): `rmse`/`mae`/`mape` join `logloss`/`ece`/`brier` as lower-is-better; `r2` defaults to higher-is-better.
The champion paired-bootstrap already takes `greater_is_better` explicitly, so regression gates decide correctly with no gate-side change.
`RegressionSchema.validate_dataset` validates the target by **dtype** (a numeric Arrow column) rather than class count, and drops `scale_pos_weight`; the two-class/0-1 checks are binary-only.

## Adapters

XGBoost and LightGBM gain a regression param model (`reg:squarederror`/`regression` objective, `rmse` eval metric, no `scale_pos_weight`) sharing a common base with the binary params so the shared hyperparameters stay defined once; `param_model(task)` and `_params` branch on `spec.task`; `supported_tasks` gains `REGRESSION`.
The raw booster output is already the prediction for both tasks, so `predict`/`_scores`/`feature_importance` are untouched.
The optional tuning-report path (used by Optuna pruning, whose contract is a higher-is-better per-round value) reports validation AUC for binary and **-RMSE** for regression, so pruning maximizes identically.

The compliance suite gains a `tiny_regression_dataset`, a regression metric set, and a `test_regression_train_predict_evaluate` guarded on `REGRESSION in supported_tasks`, so every regression-capable adapter is held to a real learning bar (R² well above 0) automatically.

## Scope and non-goals

- **JVM adapters (H2O, Spark) stay classification-only.** They are hardwired to `GBTClassifier` / `asfactor()` and reading a probability vector; a faithful regression path is a separate change and is not bundled here.
- **Multiclass remains out of scope and is genuinely architectural, not additive.** The single prediction probability column is load-bearing across predict → prediction store → monitor → ground-truth join; N classes need an N-column probability matrix rippling through all of them. It gets its own ADR when it lands.
