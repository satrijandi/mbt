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
        study = optuna.create_study(
            direction=spec.objective.direction,
            sampler=optuna.samplers.TPESampler(seed=seed),
        )

        def run_trial(trial: "optuna.Trial") -> float:
            return objective(self._suggest(trial, spec))

        study.optimize(run_trial, n_trials=n_trials)
        return TuningResult(
            best_params=dict(study.best_params),
            best_value=float(study.best_value),
            n_trials=len(study.trials),
        )
