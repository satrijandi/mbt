"""Optuna engine edges: every search-space dimension type, the all-pruned
failure mode, and the plugin descriptor (complements test_pruning)."""

from typing import Any

import pytest
from mbt_optuna.engine import OptunaTuningEngine

from mbt_adapter_base import CONTRACT_VERSION, TuningSpec


def test_suggest_covers_every_dimension_type() -> None:
    spec = TuningSpec.model_validate(
        {
            "engine": "optuna",
            "n_trials": 5,
            "search_space": {
                "booster": {"type": "categorical", "choices": ["gbtree", "dart"]},
                "max_depth": {"type": "int", "low": 2, "high": 6},
                "learning_rate": {"type": "loguniform", "low": 1e-4, "high": 1e-1},
                "subsample": {"type": "uniform", "low": 0.5, "high": 1.0},
            },
            "objective": {"metric": "pr_auc", "direction": "maximize"},
        }
    )
    proposals: list[dict[str, Any]] = []

    def objective(params: dict[str, Any]) -> float:
        proposals.append(params)
        return params["subsample"]

    result = OptunaTuningEngine({}).tune(spec, objective, n_trials=5, seed=7)
    assert result.n_trials == 5 and result.n_pruned == 0
    assert set(result.best_params) == {"booster", "max_depth", "learning_rate", "subsample"}
    for params in proposals:
        assert params["booster"] in ("gbtree", "dart")
        assert isinstance(params["max_depth"], int) and 2 <= params["max_depth"] <= 6
        assert 1e-4 <= params["learning_rate"] <= 1e-1
        assert 0.5 <= params["subsample"] <= 1.0


def test_no_completed_trials_is_an_actionable_error() -> None:
    import optuna

    spec = TuningSpec.model_validate(
        {
            "engine": "optuna",
            "n_trials": 3,
            "search_space": {"quality": {"type": "uniform", "low": 0.0, "high": 1.0}},
            "objective": {"metric": "pr_auc", "direction": "maximize"},
        }
    )

    def objective(params: dict[str, Any]) -> float:
        raise optuna.TrialPruned()  # every trial aborts

    with pytest.raises(ValueError, match="no completed trials"):
        OptunaTuningEngine({}).tune(spec, objective, n_trials=3, seed=7)


def _quad_spec() -> TuningSpec:
    return TuningSpec.model_validate(
        {
            "engine": "optuna",
            "n_trials": 12,
            "search_space": {
                "x": {"type": "uniform", "low": 0.0, "high": 1.0},
                "y": {"type": "uniform", "low": 0.0, "high": 1.0},
            },
            "objective": {"metric": "pr_auc", "direction": "maximize"},
        }
    )


def _best(config: dict[str, Any], seed: int = 7, n_trials: int = 12) -> dict[str, Any]:
    # correlated objective: rewards x and y moving together
    spec = _quad_spec()
    result = OptunaTuningEngine(config).tune(
        spec, lambda p: -((p["x"] - p["y"]) ** 2), n_trials=n_trials, seed=seed
    )
    return dict(result.best_params)


@pytest.mark.filterwarnings("ignore::optuna.exceptions.ExperimentalWarning")
def test_multivariate_tpe_is_opt_in_and_deterministic() -> None:
    # multivariate changes the proposals versus the default independent TPE ...
    assert _best({"multivariate": True}) != _best({})
    # ... but stays reproducible for a fixed seed + config (the engine's promise)
    assert _best({"multivariate": True}) == _best({"multivariate": True})


def test_random_sampler_is_selectable_and_deterministic() -> None:
    first = _best({"sampler": "random"})
    assert first == _best({"sampler": "random"})  # seeded -> reproducible
    # ... and genuinely different from TPE. This needs MORE trials than TPE's
    # n_startup_trials (10 by default): below that bar TPE is still sampling
    # from its own seeded RandomSampler, so whether the two agree is decided by
    # optuna's internal RNG wiring rather than by the sampler choice. At 12
    # trials they happened to coincide on every optuna we support except 4.9,
    # which would have forced the floor up to the newest release to satisfy an
    # accident. Past the bar TPE is model-based and the comparison is real.
    assert _best({"sampler": "random"}, n_trials=30) != _best({}, n_trials=30)


def test_unknown_sampler_is_an_actionable_error() -> None:
    with pytest.raises(ValueError, match="unknown sampler 'bogus'"):
        OptunaTuningEngine({"sampler": "bogus"}).tune(
            _quad_spec(), lambda p: p["x"], n_trials=3, seed=7
        )


def test_plugin_descriptor_wires_the_tuning_engine() -> None:
    from mbt_optuna.plugin import PLUGIN

    assert PLUGIN.name == "optuna"
    assert PLUGIN.contract_version == CONTRACT_VERSION
    assert PLUGIN.tuning is OptunaTuningEngine
    assert PLUGIN.fingerprint_packages == ["optuna"]
