"""Optuna tuning engine (TSD §13.5, FR-TUNE-01..04).

The trial loop runs inside the training job; this engine only proposes
parameters. ``import optuna`` happens lazily inside ``tune`` (ADR-14).
"""

from typing import TYPE_CHECKING, Any

from mbt_adapter_base import TuningResult, TuningSpec
from mbt_adapter_base.interchange import TuningObjectiveFn

if TYPE_CHECKING:
    import optuna


class OptunaTuningEngine:
    """Seeded TPE sampling: same seed -> same proposals -> same best params."""

    name = "optuna"

    def __init__(self, config: dict[str, Any] | None = None) -> None:
        self.config = config or {}

    def _suggest(self, trial: "optuna.Trial", spec: TuningSpec) -> dict[str, Any]:
        params: dict[str, Any] = {}
        for name, dim in spec.search_space.items():
            if dim.type == "categorical":
                assert dim.choices is not None
                params[name] = trial.suggest_categorical(name, dim.choices)
            elif dim.type == "int":
                assert dim.low is not None and dim.high is not None
                params[name] = trial.suggest_int(name, int(dim.low), int(dim.high))
            elif dim.type == "loguniform":
                assert dim.low is not None and dim.high is not None
                params[name] = trial.suggest_float(name, dim.low, dim.high, log=True)
            else:  # uniform
                assert dim.low is not None and dim.high is not None
                params[name] = trial.suggest_float(name, dim.low, dim.high)
        return params

    def tune(
        self,
        spec: TuningSpec,
        objective: TuningObjectiveFn,
        n_trials: int,
        seed: int,
    ) -> TuningResult:
        import optuna

        optuna.logging.set_verbosity(optuna.logging.WARNING)
        pruner = None
        if spec.pruner == "median":
            # Deterministic given the trial history; knobs come from the
            # engine's profile config, not the spec (they are ops tuning,
            # not model identity).
            pruner = optuna.pruners.MedianPruner(
                n_startup_trials=int(self.config.get("n_startup_trials", 5)),
                n_warmup_steps=int(self.config.get("n_warmup_steps", 5)),
            )
        study = optuna.create_study(
            direction=spec.objective.direction,
            sampler=optuna.samplers.TPESampler(seed=seed),
            pruner=pruner,
        )
        maximize = spec.objective.direction == "maximize"

        def run_trial(trial: "optuna.Trial") -> float:
            params = self._suggest(trial, spec)
            if pruner is None:
                return objective(params)

            def report(step: int, value: float) -> None:
                # The contract value is higher-is-better; the pruner compares
                # in study direction, so flip for minimize objectives.
                trial.report(value if maximize else -value, step)
                if trial.should_prune():
                    raise optuna.TrialPruned()

            return objective(params, report=report)  # type: ignore[call-arg]

        study.optimize(run_trial, n_trials=n_trials)
        pruned = sum(1 for t in study.trials if t.state == optuna.trial.TrialState.PRUNED)
        complete = [t for t in study.trials if t.state == optuna.trial.TrialState.COMPLETE]
        if not complete:
            raise ValueError(
                "tuning finished with no completed trials (all pruned or failed); "
                "raise n_startup_trials or drop the pruner"
            )
        return TuningResult(
            best_params=dict(study.best_params),
            best_value=float(study.best_value),
            n_trials=len(study.trials),
            n_pruned=pruned,
        )
