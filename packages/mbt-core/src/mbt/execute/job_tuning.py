"""Hyperparameter tuning inside the job (TSD §13.5, R2-7)."""

from typing import Any

from mbt.adapters.registry import get_registry
from mbt.events import get_bus
from mbt.events.models import LogMessage
from mbt.exceptions import ConfigError
from mbt.execute.job_carve import _carve_validation
from mbt.execute.job_metrics import _metrics_for
from mbt.execute.job_runtime import JobRuntime
from mbt.execute.job_setup import (
    _job_vars,
    _materialize_for_path_adapter,
    _render_adapter_ref,
)
from mbt.execute.seeds import (
    tuning_objective_bootstrap_seed,
    tuning_seed,
)
from mbt_adapter_base import (
    MetricSpec,
    ModelSpec,
    TuningResult,
)
from mbt_adapter_base.capabilities import Capability, supports

#: Robust tuning bootstrap (R2-7): fewer resamples than a gate's 1000 because it
#: runs once per trial during the search; the one-sided confidence matches the
#: champion-gate default.
_TUNING_BOOTSTRAP_RESAMPLES = 200


_TUNING_BOOTSTRAP_CONFIDENCE = 0.95


def _robust_objective_value(
    runtime: JobRuntime, model: Any, spec: ModelSpec, objective_spec: MetricSpec
) -> float:
    """Bootstrap lower bound of the validation objective metric (R2-7): the
    tuning selection is then defended against validation-window luck, not made
    on a single-carve point estimate. Predicts once on validation and resamples
    the (label, score) rows with a fixed seed, so trials are compared on the same
    resamples and the search stays reproducible."""
    import numpy as np

    from mbt_adapter_base.metrics import bootstrap_metric_lower_bound

    predictions = runtime.adapter.predict(model, runtime.handle, "validation")
    y_score = np.asarray(
        predictions.column("prediction").to_numpy(zero_copy_only=False), dtype=float
    )
    y_true = np.asarray(predictions.column(spec.target).to_numpy(zero_copy_only=False), dtype=float)
    return bootstrap_metric_lower_bound(
        objective_spec,
        y_true,
        y_score,
        confidence=_TUNING_BOOTSTRAP_CONFIDENCE,
        n_resamples=_TUNING_BOOTSTRAP_RESAMPLES,
        seed=tuning_objective_bootstrap_seed(spec.seed),
    )


def _run_tuning(
    runtime: JobRuntime,
    spec: ModelSpec,
    tracking: Any = None,
    run_handle: Any = None,
    *,
    carve_time_range: tuple[Any, Any] | None = None,
) -> tuple[ModelSpec, TuningResult | None]:
    tuning = spec.tuning
    if tuning is None:
        return spec, None
    job = runtime.job
    if job.tuning_engine is None:
        raise ConfigError(
            f"model declares tuning but no tuning engine was provided for {tuning.engine!r}",
            resource=job.node.unique_id,
        )
    engine_ref = _render_adapter_ref(job.tuning_engine, _job_vars(job))
    engine = get_registry().component("tuning", engine_ref.adapter, engine_ref.config)

    carved = _carve_validation(runtime, time_range=carve_time_range)
    tune_handle: Any = carved or runtime.handle
    if carved is not None and getattr(runtime.adapter, "data_access", "arrow") == "path":
        tune_handle = _materialize_for_path_adapter(carved, spec)
    explicit_validation = "validation" in runtime.transformed.splits()
    # D-2: with no declared validation split the final fit reabsorbs the carve
    # (ADR-8), so it early-stops on nothing while every trial that voted on the
    # hyperparameters DID stop early. Collect what the trials actually used.
    collect_rounds = (
        not explicit_validation
        and spec.hyperparameters.get("early_stopping_rounds") is not None
        and supports(runtime.adapter, Capability.BEST_ITERATION, spec)
    )

    objective_spec = next(
        (
            m
            for m in [*runtime.builtin_specs, *runtime.hook_specs]
            if m.name == tuning.objective.metric
        ),
        None,
    )
    if objective_spec is None:
        raise ConfigError(
            f"tuning objective {tuning.objective.metric!r} is not a resolved metric",
            resource=job.node.unique_id,
        )
    if tuning.objective.robust and objective_spec.kind != "builtin":
        raise ConfigError(
            f"tuning objective {tuning.objective.metric!r} is a hook metric; "
            "robust (bootstrap) selection supports builtin metrics only",
            resource=job.node.unique_id,
        )

    tune_runtime = JobRuntime(**{**runtime.__dict__, "handle": tune_handle})
    trial_counter = {"index": 0}
    #: Rounds each trial's fit actually kept (D-2). Collected only when the
    #: final fit will REABSORB the tuning carve, because that is the case where
    #: the final fit has nothing to stop on.
    trial_rounds: list[int] = []

    adapter_reports = supports(runtime.adapter, Capability.TRAIN_WITH_REPORT, spec)
    if tuning.pruner is not None and not adapter_reports:
        get_bus().emit(
            LogMessage(
                level="warn",
                unique_id=job.node.unique_id,
                message=(
                    f"tuning.pruner {tuning.pruner!r} is set but adapter "
                    f"{spec.adapter!r} does not report training progress "
                    "(no train_with_report); trials run to completion"
                ),
            )
        )

    def objective(trial_params: dict[str, Any], report: Any = None) -> float:
        # Trials never calibrate: the objective is scored on the same
        # validation split a trial calibrator would fit on, which would make a
        # calibration-sensitive objective (brier/ece) circularly optimal (F17).
        # The final fit calibrates on its own dedicated slice instead.
        trial_spec = spec.model_copy(
            update={
                "hyperparameters": {**spec.hyperparameters, **trial_params},
                "calibration": None,
            }
        )
        if report is not None and adapter_reports:
            # A pruning report may raise out of the training loop; the engine
            # owns that exception and marks the trial pruned.
            model = runtime.adapter.train_with_report(trial_spec, tune_handle, runtime.ctx, report)
        else:
            model = runtime.adapter.train(trial_spec, tune_handle, runtime.ctx)
        if collect_rounds:
            rounds = runtime.adapter.best_iteration(model)
            if rounds:
                trial_rounds.append(int(rounds))
        if tuning.objective.robust:
            value = _robust_objective_value(tune_runtime, model, spec, objective_spec)
        else:
            results = _metrics_for(tune_runtime, model, "validation", with_slices=False)
            value = float(results.metrics[tuning.objective.metric])
        index = trial_counter["index"]
        trial_counter["index"] += 1
        # Per-trial progress on the bus at debug (visible under --verbose or
        # --log-format json) so a long search is not opaque while it runs.
        get_bus().emit(
            LogMessage(
                level="debug",
                unique_id=job.node.unique_id,
                message=f"tuning trial {index}: {tuning.objective.metric}={value:.4f}",
            )
        )
        # Trial history as nested tracking runs where the adapter supports it
        # (FR-TUNE-03); optional so simple trackers keep working.
        if (
            tracking is not None
            and run_handle is not None
            and supports(tracking, Capability.LOG_TRIAL)
        ):
            tracking.log_trial(run_handle, index, trial_params, value)
        return value

    n_trials = tuning.n_trials
    if job.tuning_cap is not None:
        n_trials = min(n_trials, job.tuning_cap)
        if n_trials < tuning.n_trials:
            get_bus().emit(
                LogMessage(
                    unique_id=job.node.unique_id,
                    message=(
                        f"tuning capped at {n_trials} trial(s) by the target's "
                        f"max_tuning_trials (requested {tuning.n_trials}) (FR-TUNE-04)"
                    ),
                )
            )
    result = engine.tune(tuning, objective, n_trials=n_trials, seed=tuning_seed(spec.seed))

    # Final fit reabsorbs an *implicit* carve; an explicit validation split
    # stays held out because the user declared it (TSD §10.5 step 6).
    hyperparameters = {**spec.hyperparameters, **result.best_params}
    hyperparameters = _carry_trial_rounds(runtime, hyperparameters, trial_rounds)
    tuned = spec.model_copy(update={"hyperparameters": hyperparameters})
    return tuned, result


def _carry_trial_rounds(
    runtime: JobRuntime, hyperparameters: dict[str, Any], trial_rounds: list[int]
) -> dict[str, Any]:
    """Ship the complexity the search chose, not the one the spec declared (D-2).

    When the dataset declares no ``validation`` split, core carves one for the
    tuning trials and the final fit REABSORBS it (ADR-8, deliberately - those
    rows are training data). So every trial that voted on the hyperparameters
    stopped early, and the model that ships would otherwise train all
    ``n_estimators`` rounds: a search that converged on an aggressive learning
    rate BECAUSE early stopping was protecting it produces a final model with
    no such protection.

    Nothing downstream can see that. The gates evaluate the shipped model
    honestly, so it fails only when it is genuinely worse and silently ships a
    differently-regularized model when it is not.

    The median of the trials' best iterations keeps every training row and
    ships the depth the search actually selected. ``early_stopping_rounds`` is
    dropped alongside it, because with no validation split it is inert and
    leaving it in the recorded spec reads as if the final fit stopped early.
    """
    from statistics import median

    if not trial_rounds:
        return hyperparameters
    rounds = max(1, int(median(trial_rounds)))
    declared = hyperparameters.get("n_estimators")
    carried = {**hyperparameters, "n_estimators": rounds}
    carried.pop("early_stopping_rounds", None)
    get_bus().emit(
        LogMessage(
            unique_id=runtime.job.node.unique_id,
            message=(
                f"final fit uses n_estimators={rounds} (median of {len(trial_rounds)} "
                f"tuning trial(s), which early-stopped between {min(trial_rounds)} and "
                f"{max(trial_rounds)} rounds) instead of the declared {declared}: the "
                "final fit reabsorbs the validation carve, so it has nothing to stop "
                "on and would otherwise ship a less-regularized model than the search "
                "scored"
            ),
        )
    )
    return carried
