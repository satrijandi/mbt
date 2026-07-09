"""Optuna pruner integration: median pruning over reported trial progress
(section 3.5: previously every trial trained to completion)."""

import pytest
from mbt_optuna.engine import OptunaTuningEngine

from mbt_adapter_base import TuningSpec

STEPS_PER_TRIAL = 10
N_TRIALS = 12

#: Aggressive knobs so a 12-trial study actually prunes (profile-level
#: engine config, not spec surface - ops tuning is not model identity).
ENGINE_CONFIG = {"n_startup_trials": 2, "n_warmup_steps": 1}


def _spec(pruner: str | None = None, direction: str = "maximize") -> TuningSpec:
    payload = {
        "engine": "optuna",
        "n_trials": N_TRIALS,
        "search_space": {"quality": {"type": "uniform", "low": 0.0, "high": 1.0}},
        "objective": {"metric": "pr_auc", "direction": direction},
    }
    if pruner is not None:
        payload["pruner"] = pruner
    return TuningSpec.model_validate(payload)


def _ramping_objective(work: dict, *, minimize: bool = False):
    """Each trial ramps toward its quality over 10 steps (the fake-adapter
    curve shape); `work` counts the training steps actually executed."""

    def objective(params, report=None):
        if report is not None:
            for step in range(STEPS_PER_TRIAL):
                work["steps"] += 1
                report(step, params["quality"] * (step + 1) / STEPS_PER_TRIAL)
        return (1.0 - params["quality"]) if minimize else params["quality"]

    return objective


def test_median_pruner_prunes_and_saves_work() -> None:
    work = {"steps": 0}
    result = OptunaTuningEngine(ENGINE_CONFIG).tune(
        _spec(pruner="median"), _ramping_objective(work), n_trials=N_TRIALS, seed=7
    )
    assert result.n_trials == N_TRIALS
    assert result.n_pruned > 0, "no trial was pruned"
    assert work["steps"] < N_TRIALS * STEPS_PER_TRIAL, "pruning saved no training work"
    # the winner is a completed trial's true final value
    assert 0.0 <= result.best_value <= 1.0
    assert result.best_value == pytest.approx(result.best_params["quality"])


def test_without_pruner_the_old_objective_signature_still_works() -> None:
    def legacy_objective(params):  # no report kwarg at all
        return params["quality"]

    result = OptunaTuningEngine(ENGINE_CONFIG).tune(
        _spec(), legacy_objective, n_trials=N_TRIALS, seed=7
    )
    assert result.n_pruned == 0
    assert result.n_trials == N_TRIALS


def test_pruning_is_deterministic() -> None:
    runs = []
    for _ in range(2):
        work = {"steps": 0}
        runs.append(
            OptunaTuningEngine(ENGINE_CONFIG).tune(
                _spec(pruner="median"), _ramping_objective(work), n_trials=N_TRIALS, seed=11
            )
        )
    assert runs[0].best_params == runs[1].best_params
    assert runs[0].n_pruned == runs[1].n_pruned


def test_minimize_direction_flips_the_report_sign() -> None:
    """Reported values are higher-is-better by contract; in a minimize study
    the engine negates them so the pruner still cuts the WEAK trials."""
    work = {"steps": 0}
    result = OptunaTuningEngine(ENGINE_CONFIG).tune(
        _spec(pruner="median", direction="minimize"),
        _ramping_objective(work, minimize=True),
        n_trials=N_TRIALS,
        seed=7,
    )
    assert result.n_pruned > 0
    # minimizing 1-quality: the winner still corresponds to the HIGHEST quality
    assert result.best_value == pytest.approx(1.0 - result.best_params["quality"])
    assert result.best_value < 0.5, "pruner cut strong trials instead of weak ones"


def test_pruner_field_validates() -> None:
    import pydantic

    assert _spec(pruner="median").pruner == "median"
    assert _spec().pruner is None
    with pytest.raises(pydantic.ValidationError):
        _spec(pruner="hyperband")
